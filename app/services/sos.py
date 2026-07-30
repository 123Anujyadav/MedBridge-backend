"""
The SOS emergency workflow.

Raise, track, hand over, resolve. Three rules run through the whole module:

* **One open emergency per patient.** Pressing the button twice must not create
  two records — a responder seeing duplicates cannot tell whether one patient
  is in trouble or two are.
* **Reachability is decided here, once.** A patient reaches only their own
  emergencies; a clinician reaches the ones assigned to them and the unclaimed
  queue; an administrator reaches everything. Every read and every write goes
  through `_load_for_actor`, so there is no route that forgot to check.
* **Announce after the commit.** Callers hand the notifier the payload only
  once the transaction has landed. A responder told about an emergency that
  then rolled back has been sent to a patient who never raised one.

This module records and announces state. It dispatches nothing and messages
nobody: SMS, WhatsApp, voice and push are Phase 3, behind
`app.services.sos_notifications`.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    AuthorizationException,
    BusinessRuleValidationException,
    EntityNotFoundException,
)
from app.models.emergency import (
    ACTIVE_SOS_STATUSES,
    EmergencyRequest,
    EmergencyStatusEvent,
    TERMINAL_SOS_STATUSES,
)
from app.models.emergency_profile import EmergencyProfile
from app.models.patient import Patient
from app.models.user import User
from app.schemas.sos import SOSTriggerRequest
from app.services.emergency_profile import build_maps_url, format_address

logger = logging.getLogger(__name__)


ALLOWED_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "pending": ("accepted", "doctor_assigned", "ambulance_dispatched",
                "hospital_reached", "resolved", "cancelled"),
    "accepted": ("doctor_assigned", "ambulance_dispatched", "hospital_reached",
                 "resolved", "cancelled"),
    "doctor_assigned": ("ambulance_dispatched", "hospital_reached", "resolved",
                        "cancelled"),
    "ambulance_dispatched": ("hospital_reached", "resolved", "cancelled"),
    "hospital_reached": ("resolved", "cancelled"),
    "resolved": (),
    "cancelled": (),
}
"""
Which states may follow which.

An explicit machine rather than "set whatever was sent". Without it a stray
request could move a resolved emergency back to pending, or mark an ambulance
as having reached hospital before it was dispatched — and the timeline, which
is what an incident review reads, would record it as fact.

Forward moves may skip stages; backward moves are refused outright. Skipping is
deliberate: an emergency that turns out to need no ambulance still has to be
closable as `resolved`. Allowing only adjacent steps meant the sole way out of
`pending` was `cancelled`, which files a genuine emergency that was handled as
a false alarm — the record would then say nothing happened when something did.

`resolved` and `cancelled` are terminal and have no exits at all.
"""

MISSING_PROFILE_MESSAGE = (
    "Your emergency profile is incomplete. Add an emergency contact and your "
    "registered address before raising an SOS."
)
MISSING_LOCATION_MESSAGE = (
    "No location is available. Allow location access, or save your current "
    "location on your emergency profile, then try again."
)
ALREADY_ACTIVE_MESSAGE = "Emergency already active."


def _age_from(date_of_birth: Optional[str]) -> Optional[int]:
    """
    Age in whole years, or None if the stored date cannot be read.

    Best effort by design: an unparseable date of birth must not stop an
    emergency being raised. Responders would rather have the record without an
    age than not have the record.
    """
    if not date_of_birth:
        return None
    try:
        born = date.fromisoformat(str(date_of_birth)[:10])
    except (ValueError, TypeError):
        return None
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def to_response(
    emergency: EmergencyRequest,
    patient: Optional[Patient] = None,
    include_timeline: bool = True,
) -> dict:
    """Flatten an emergency into the shape every portal renders."""
    location = emergency.location or {}
    return {
        "id": emergency.id,
        "patient_id": emergency.patient_id,
        "status": emergency.status,
        "patient_name": emergency.patient_name,
        "patient_phone": emergency.patient_phone,
        "patient_age": _age_from(getattr(patient, "date_of_birth", None)),
        "blood_type": getattr(patient, "blood_type", None),
        "contact_name": emergency.contact_name,
        "contact_phone": emergency.contact_phone,
        "contact_relationship": emergency.contact_relationship,
        "assigned_doctor_id": emergency.assigned_doctor_id,
        "assigned_doctor_name": emergency.assigned_doctor_name,
        "address": location.get("address"),
        "latitude": location.get("lat"),
        "longitude": location.get("lng"),
        "maps_url": emergency.maps_url,
        "triggered_at": emergency.created_at,
        "resolved_at": emergency.resolved_at,
        "cancelled_at": emergency.cancelled_at,
        "cancel_reason": emergency.cancel_reason,
        "created_by": emergency.created_by,
        "is_active": emergency.status in ACTIVE_SOS_STATUSES,
        "timeline": [
            {
                "status": event.status,
                "note": event.note,
                "actor_role": event.actor_role,
                "actor_name": event.actor_name,
                "created_at": event.created_at,
            }
            for event in (emergency.status_events or [])
        ] if include_timeline else [],
    }


class SOSService:
    # ── loading and scoping ──────────────────────────────────────────────

    async def _get(
        self, db: AsyncSession, emergency_id: uuid.UUID
    ) -> Optional[EmergencyRequest]:
        result = await db.execute(
            select(EmergencyRequest)
            .options(selectinload(EmergencyRequest.status_events))
            .where(EmergencyRequest.id == emergency_id)
        )
        return result.scalars().first()

    async def _load_for_actor(
        self, db: AsyncSession, emergency_id: uuid.UUID, actor: User
    ) -> EmergencyRequest:
        """
        Fetch an emergency the caller is entitled to, or refuse.

        The single choke point for reachability. Note that "not yours" and "does
        not exist" both raise `EntityNotFoundException`: answering 403 for one
        and 404 for the other would let a caller enumerate which emergency ids
        are real.
        """
        emergency = await self._get(db, emergency_id)
        if emergency is None:
            raise EntityNotFoundException("Emergency", str(emergency_id))

        if actor.role == "admin":
            return emergency

        if actor.role == "patient":
            if emergency.patient_id != actor.id:
                raise EntityNotFoundException("Emergency", str(emergency_id))
            return emergency

        if actor.role == "doctor":
            assigned_to_me = emergency.assigned_doctor_id == actor.id
            unclaimed = (
                emergency.assigned_doctor_id is None
                and emergency.status in ACTIVE_SOS_STATUSES
            )
            if assigned_to_me or unclaimed:
                return emergency
            raise EntityNotFoundException("Emergency", str(emergency_id))

        raise AuthorizationException("You may not view this emergency.")

    # There is deliberately no `has_active_emergency` helper. "One emergency at
    # a time" is now answered by the same query that loads the patient in
    # `trigger`, and a second copy of the rule — with its own idea of which
    # statuses count as active — is how the two drift apart.

    async def get_active_for_patient(
        self, db: AsyncSession, patient_id: uuid.UUID
    ) -> Optional[EmergencyRequest]:
        """The patient's open emergency, if they have one."""
        result = await db.execute(
            select(EmergencyRequest)
            .options(selectinload(EmergencyRequest.status_events))
            .where(
                and_(
                    EmergencyRequest.patient_id == patient_id,
                    EmergencyRequest.status.in_(ACTIVE_SOS_STATUSES),
                )
            )
            .order_by(EmergencyRequest.created_at.desc())
        )
        return result.scalars().first()

    # ── raising ──────────────────────────────────────────────────────────

    async def trigger(
        self, db: AsyncSession, patient_id: uuid.UUID, payload: SOSTriggerRequest,
        *, require_profile: bool = True, address_override: Optional[str] = None,
    ) -> tuple[dict, EmergencyRequest]:
        """
        Raise an emergency for this patient.

        Runs the checks in the order the workflow specifies: profile complete,
        then a usable position, then no emergency already open. Returns the
        response *and* the row, so the caller can commit before announcing it.

        `require_profile` exists for the legacy panic route, which predates the
        emergency profile and supplies its own coordinates and address in the
        request body. Demanding a profile there would have added a precondition
        to a published endpoint that never had one — breaking existing callers
        in the name of a rule written after them. The SOS workflow itself always
        requires the profile, as the specification states.
        """
        # The patient, their emergency profile, and whether one of their
        # emergencies is already open — in one round trip. These were three
        # sequential waits on a remote database, on the path a patient is
        # staring at an SOS button waiting for. Each answer is needed before the
        # row can be written, and none of them depends on the others, so there
        # is no reason to pay the network three times for them.
        loaded = (await db.execute(
            select(
                Patient,
                EmergencyProfile,
                select(EmergencyRequest.id)
                .where(
                    EmergencyRequest.patient_id == patient_id,
                    EmergencyRequest.status.in_(ACTIVE_SOS_STATUSES),
                )
                .exists()
                .label("already_active"),
            )
            .outerjoin(EmergencyProfile, EmergencyProfile.id == Patient.id)
            .where(Patient.id == patient_id)
        )).first()
        if loaded is None:
            raise EntityNotFoundException("Patient", str(patient_id))
        patient, profile, already_active = loaded

        # ── step 1: the emergency profile must be complete ───────────────
        if profile is not None and profile.deleted_at is not None:
            profile = None
        has_profile = profile is not None and bool(profile.contact_phone) and bool(profile.city)

        if require_profile and not has_profile:
            raise BusinessRuleValidationException(MISSING_PROFILE_MESSAGE)

        # ── step 2: a position, live if offered, stored otherwise ────────
        if payload.latitude is not None and payload.longitude is not None:
            latitude, longitude = payload.latitude, payload.longitude
            if profile is not None:
                # The live fix is also written back to the profile, so the most
                # recent known position survives the emergency being resolved.
                profile.latitude = latitude
                profile.longitude = longitude
                profile.maps_url = build_maps_url(latitude, longitude)
                profile.location_updated_at = datetime.now(timezone.utc)
        elif profile is not None and profile.latitude is not None and profile.longitude is not None:
            # Permission refused, or unavailable. The stored position is what
            # the patient last agreed to share, and is better than nothing.
            latitude, longitude = profile.latitude, profile.longitude
        else:
            raise BusinessRuleValidationException(MISSING_LOCATION_MESSAGE)

        # ── step 5: never two open at once ───────────────────────────────
        # Answered by the query above. The checks still run in the order the
        # workflow specifies — profile, then position, then this — so the message
        # a patient gets for an incomplete profile is unchanged.
        if already_active:
            raise BusinessRuleValidationException(ALREADY_ACTIVE_MESSAGE)

        emergency = EmergencyRequest(
            patient_id=patient_id,
            patient_name=f"{patient.first_name} {patient.last_name}".strip(),
            patient_phone=patient.phone,
            location={
                "lat": latitude,
                "lng": longitude,
                "address": (
                    format_address(profile) if profile is not None
                    else (address_override or "")
                ),
            },
            maps_url=build_maps_url(latitude, longitude),
            contact_name=profile.contact_name if profile else None,
            contact_phone=profile.contact_phone if profile else None,
            contact_relationship=profile.contact_relationship if profile else None,
            status="pending",
            created_by="patient",
            # Nothing has been dispatched. The previous implementation set this
            # true with a fabricated unit and a twelve-minute ETA the moment the
            # button was pressed; telling a patient help is on the way when
            # nothing has been arranged is the worst thing this module could do.
            ambulance_dispatched=False,
        )
        # The first timeline entry is attached through the relationship rather
        # than by setting `emergency_id`. The primary key is generated by the
        # model's default at *flush* time, not by `db.add`, so reading
        # `emergency.id` here would hand the event a null foreign key — the
        # relationship lets SQLAlchemy order the two inserts and wire them in
        # one flush instead of two.
        emergency.status_events = [EmergencyStatusEvent(
            status="pending",
            note="Emergency raised by patient.",
            actor_user_id=patient_id,
            actor_role="patient",
            actor_name=emergency.patient_name,
        )]
        db.add(emergency)
        await db.flush()

        logger.info("[SOS_RAISED] emergency=%s patient=%s", emergency.id, patient_id)
        return to_response(emergency, patient), emergency

    # ── moving it along ──────────────────────────────────────────────────

    def _record_event(
        self, db: AsyncSession, emergency: EmergencyRequest, status: str,
        note: Optional[str], actor_id: Optional[uuid.UUID],
        actor_role: Optional[str], actor_name: Optional[str],
    ) -> None:
        db.add(EmergencyStatusEvent(
            emergency_id=emergency.id,
            status=status,
            note=note,
            actor_user_id=actor_id,
            actor_role=actor_role,
            actor_name=actor_name,
        ))

    async def _actor_name(self, db: AsyncSession, actor: User) -> str:
        """A human name for the timeline, falling back to the address."""
        if actor.role == "doctor":
            from app.models.doctor import Doctor

            doctor = await db.get(Doctor, actor.id)
            if doctor:
                return f"Dr. {doctor.first_name} {doctor.last_name}".strip()
        elif actor.role == "patient":
            patient = await db.get(Patient, actor.id)
            if patient:
                return f"{patient.first_name} {patient.last_name}".strip()
        return actor.email

    async def update_status(
        self, db: AsyncSession, emergency_id: uuid.UUID, actor: User,
        new_status: str, note: Optional[str] = None,
    ) -> tuple[dict, EmergencyRequest]:
        """
        Move an emergency to its next state.

        Only clinicians and administrators may. A patient's one write is
        cancellation, which has its own entry point — letting them mark
        themselves as having reached hospital would put a clinical claim in the
        record that nobody clinical made.
        """
        if actor.role not in ("doctor", "admin"):
            raise AuthorizationException(
                "Only clinical staff may change an emergency's status."
            )

        emergency = await self._load_for_actor(db, emergency_id, actor)

        current = emergency.status
        if current in TERMINAL_SOS_STATUSES:
            raise BusinessRuleValidationException(
                f"This emergency is already {current} and cannot be changed."
            )
        allowed = ALLOWED_TRANSITIONS.get(current, ())
        if new_status not in allowed:
            raise BusinessRuleValidationException(
                f"An emergency cannot move from {current} to {new_status}."
            )

        # A doctor acting on an unclaimed emergency takes it. Otherwise the
        # queue would keep offering it to everyone while one of them worked it.
        if actor.role == "doctor" and emergency.assigned_doctor_id is None:
            emergency.assigned_doctor_id = actor.id
            emergency.assigned_doctor_name = await self._actor_name(db, actor)

        emergency.status = new_status
        if new_status == "ambulance_dispatched":
            emergency.ambulance_dispatched = True
        if new_status == "resolved":
            emergency.resolved_at = datetime.now(timezone.utc)

        self._record_event(
            db, emergency, new_status, note=note, actor_id=actor.id,
            actor_role=actor.role, actor_name=await self._actor_name(db, actor),
        )
        await db.flush()
        await db.refresh(emergency, ["status_events"])

        patient = await db.get(Patient, emergency.patient_id)
        logger.info(
            "[SOS_STATUS] emergency=%s %s -> %s by %s",
            emergency.id, current, new_status, actor.id,
        )
        return to_response(emergency, patient), emergency

    async def assign_doctor(
        self, db: AsyncSession, emergency_id: uuid.UUID, actor: User,
        doctor_id: uuid.UUID,
    ) -> tuple[dict, EmergencyRequest]:
        """Hand an emergency to a named clinician. Administrators only."""
        if actor.role != "admin":
            raise AuthorizationException(
                "Only an administrator may assign an emergency to a clinician."
            )

        from app.models.doctor import Doctor
        from app.services.doctor_access import assert_doctor_may_practise

        emergency = await self._load_for_actor(db, emergency_id, actor)
        if emergency.status in TERMINAL_SOS_STATUSES:
            raise BusinessRuleValidationException(
                f"This emergency is already {emergency.status}."
            )

        doctor = await db.get(Doctor, doctor_id)
        if doctor is None:
            raise EntityNotFoundException("Doctor", str(doctor_id))

        doctor_user = await db.get(User, doctor_id)
        if doctor_user is None or not doctor_user.is_active:
            raise BusinessRuleValidationException(
                "That clinician's account is not active."
            )
        # The same approval rule the doctor portal enforces. An unverified
        # clinician must not be handed an emergency they could not open.
        await assert_doctor_may_practise(db, doctor_user)

        emergency.assigned_doctor_id = doctor_id
        emergency.assigned_doctor_name = f"Dr. {doctor.first_name} {doctor.last_name}".strip()
        if emergency.status in ("pending", "accepted"):
            emergency.status = "doctor_assigned"

        self._record_event(
            db, emergency, emergency.status,
            note=f"Assigned to {emergency.assigned_doctor_name}.",
            actor_id=actor.id, actor_role=actor.role,
            actor_name=await self._actor_name(db, actor),
        )
        await db.flush()
        await db.refresh(emergency, ["status_events"])

        patient = await db.get(Patient, emergency.patient_id)
        logger.info("[SOS_ASSIGNED] emergency=%s doctor=%s", emergency.id, doctor_id)
        return to_response(emergency, patient), emergency

    async def cancel(
        self, db: AsyncSession, emergency_id: uuid.UUID, actor: User,
        reason: Optional[str] = None,
    ) -> tuple[dict, EmergencyRequest]:
        """
        Stand an emergency down.

        A patient may cancel their own — a false alarm should be withdrawable by
        the person who raised it, without waiting for staff. Clinicians and
        administrators may cancel any they can reach.
        """
        emergency = await self._load_for_actor(db, emergency_id, actor)

        if emergency.status in TERMINAL_SOS_STATUSES:
            raise BusinessRuleValidationException(
                f"This emergency is already {emergency.status}."
            )

        emergency.status = "cancelled"
        emergency.cancelled_at = datetime.now(timezone.utc)
        emergency.cancel_reason = (reason or "").strip() or None

        self._record_event(
            db, emergency, "cancelled", note=emergency.cancel_reason,
            actor_id=actor.id, actor_role=actor.role,
            actor_name=await self._actor_name(db, actor),
        )
        await db.flush()
        await db.refresh(emergency, ["status_events"])

        patient = await db.get(Patient, emergency.patient_id)
        logger.info("[SOS_CANCELLED] emergency=%s by=%s", emergency.id, actor.id)
        return to_response(emergency, patient), emergency

    # ── reading ──────────────────────────────────────────────────────────

    async def get_one(
        self, db: AsyncSession, emergency_id: uuid.UUID, actor: User
    ) -> dict:
        emergency = await self._load_for_actor(db, emergency_id, actor)
        patient = await db.get(Patient, emergency.patient_id)
        return to_response(emergency, patient)

    async def list_for_actor(
        self, db: AsyncSession, actor: User, active_only: bool = True,
        limit: int = 50,
    ) -> list[dict]:
        """
        The emergencies this caller may see, newest first.

        The scope is built into the query rather than filtered afterwards, so a
        row the caller is not entitled to is never loaded in the first place:

        * **patient** — their own, and only their own;
        * **doctor** — assigned to them, plus the unclaimed active queue, which
          is how an emergency reaches a clinician who can accept it. Another
          doctor's assigned emergency is not visible;
        * **admin** — everything.
        """
        stmt = (
            select(EmergencyRequest)
            .options(selectinload(EmergencyRequest.status_events))
            .order_by(EmergencyRequest.created_at.desc())
            .limit(limit)
        )

        if actor.role == "patient":
            stmt = stmt.where(EmergencyRequest.patient_id == actor.id)
        elif actor.role == "doctor":
            stmt = stmt.where(
                or_(
                    EmergencyRequest.assigned_doctor_id == actor.id,
                    and_(
                        EmergencyRequest.assigned_doctor_id.is_(None),
                        EmergencyRequest.status.in_(ACTIVE_SOS_STATUSES),
                    ),
                )
            )
        elif actor.role != "admin":
            raise AuthorizationException("You may not view emergencies.")

        if active_only:
            stmt = stmt.where(EmergencyRequest.status.in_(ACTIVE_SOS_STATUSES))

        rows = list((await db.execute(stmt)).scalars().all())
        if not rows:
            return []

        # One query for every patient in the page rather than one per row —
        # the age and blood group come from `patients`, and a dashboard
        # listing twenty emergencies would otherwise issue twenty lookups.
        patient_ids = {row.patient_id for row in rows}
        patients = {
            p.id: p for p in (await db.execute(
                select(Patient).where(Patient.id.in_(patient_ids))
            )).scalars().all()
        }
        return [to_response(row, patients.get(row.patient_id)) for row in rows]


sos_service = SOSService()
