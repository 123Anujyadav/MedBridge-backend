"""
Reaching people when an emergency is raised.

The shape of this module follows from three facts about the job:

* **Nobody may be lost.** Every intended contact is written to
  `communication_logs` *before* the provider is called, so a crash, a timeout
  or an outage leaves a durable record of what is owed. The retry sweep works
  from that table, not from memory.
* **The patient must not wait for it.** Placing three calls and six messages is
  seconds of network I/O. It runs after the response is returned, so the SOS
  API answers immediately and the emergency is already recorded and broadcast
  before the first telephone rings.
* **Order matters, but a gap must not stop the sequence.** The emergency
  contact is reached first, then the assigned clinician, then an administrator
  — but a clinician who has not been assigned yet, or has no number on file,
  is recorded as skipped and the sequence continues. An unreachable doctor is
  not a reason to leave the family uncalled.

Phone numbers come from the database — the emergency's contact snapshot, the
clinician's profile, the administrator's profile. Nothing here has a number,
a URL or a message in it.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import and_, case, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.communication import CommunicationLog
from app.models.doctor import Doctor
from app.models.emergency import EmergencyRequest
from app.models.patient import Patient
from app.models.user import User
from app.services import emergency_templates as templates
from app.services.maps import get_maps_service
from app.services.twilio_gateway import DeliveryResult, get_twilio_gateway

logger = logging.getLogger(__name__)

CHANNEL_ORDER = ("voice", "sms", "whatsapp")

RECIPIENT_ORDER = ("emergency_contact", "doctor", "admin")
"""
Who is reached first.

The order the workflow specifies: the family before the clinician before the
administrator. It is applied as an explicit ranking in the dispatch query
because every row for an emergency is written in one transaction and therefore
shares a `created_at` — ordering by time alone left the tie to a random UUID.
"""

_RECIPIENT_RANK = case(
    {role: index for index, role in enumerate(RECIPIENT_ORDER)},
    value=CommunicationLog.recipient_role,
    else_=len(RECIPIENT_ORDER),
)

_CHANNEL_RANK = case(
    {channel: index for index, channel in enumerate(CHANNEL_ORDER)},
    value=CommunicationLog.channel,
    else_=len(CHANNEL_ORDER),
)

VOICE_TEMPLATES = {
    "emergency_contact": templates.TEMPLATE_VOICE_CONTACT,
    "doctor": templates.TEMPLATE_VOICE_DOCTOR,
    "admin": templates.TEMPLATE_VOICE_ADMIN,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def mask_phone(number: Optional[str]) -> Optional[str]:
    """
    Show enough of a number to recognise it, not enough to dial it.

    Applied on the way out of the API. The full number stays in the row,
    because a responder reviewing a failed contact needs to see exactly what
    was tried.
    """
    if not number:
        return None
    digits = number.strip()
    if len(digits) <= 4:
        return "•" * len(digits)
    return f"{digits[:3]}{'•' * max(0, len(digits) - 6)}{digits[-3:]}"


class EmergencyCommsService:
    # ── who to reach ─────────────────────────────────────────────────────

    async def resolve_recipients(
        self, db: AsyncSession, emergency: EmergencyRequest
    ) -> list[dict[str, Any]]:
        """
        The people to contact, in the order the workflow specifies.

        Every number is read from the database. A recipient with no number is
        still returned — with `phone=None` — so the attempt is recorded as
        skipped rather than silently not existing. "Nobody tried to call the
        doctor" and "the doctor has no number on file" are different facts, and
        an incident review needs to tell them apart.
        """
        recipients: list[dict[str, Any]] = [{
            "role": "emergency_contact",
            "name": emergency.contact_name,
            "phone": emergency.contact_phone,
        }]

        doctor_name, doctor_phone = emergency.assigned_doctor_name, None
        if emergency.assigned_doctor_id:
            doctor = await db.get(Doctor, emergency.assigned_doctor_id)
            if doctor:
                doctor_phone = doctor.phone
                doctor_name = f"Dr. {doctor.first_name} {doctor.last_name}".strip()
        recipients.append({
            "role": "doctor", "name": doctor_name, "phone": doctor_phone,
        })

        # Administrators are people, not a role account: the live ones are read
        # from `users`, and the first with a telephone number is called.
        admins = (await db.execute(
            select(User).where(
                User.role == "admin",
                User.is_active.is_(True),
                User.deleted_at.is_(None),
            )
        )).scalars().all()

        # One query for every administrator's profile rather than one each:
        # a lookup per administrator is a round trip per administrator, on the
        # path a patient is waiting on.
        admin_name, admin_phone = None, None
        if admins:
            profiles = {
                p.id: p for p in (await db.execute(
                    select(Patient).where(Patient.id.in_([a.id for a in admins]))
                )).scalars().all()
            }
            for admin in admins:
                admin_name = admin_name or admin.email
                phone = getattr(profiles.get(admin.id), "phone", None)
                if phone:
                    admin_name, admin_phone = admin.email, phone
                    break
        recipients.append({
            "role": "admin", "name": admin_name, "phone": admin_phone,
        })

        return recipients

    # ── enrichment ───────────────────────────────────────────────────────

    async def enrich_location(
        self, db: AsyncSession, emergency: EmergencyRequest
    ) -> bool:
        """
        Fill in the address and the nearest hospital, if Maps is configured.

        Returns whether anything changed. With no key this is a no-op that
        leaves every field null — the record then carries the patient's own
        registered address and the plain map link, which is the honest state
        rather than a placeholder.
        """
        maps = get_maps_service()
        if not maps.is_enabled():
            return False

        location = emergency.location or {}
        latitude, longitude = location.get("lat"), location.get("lng")
        if latitude is None or longitude is None:
            return False

        changed = False

        address = await maps.reverse_geocode(latitude, longitude)
        if address:
            emergency.resolved_address = address[:400]
            changed = True

        hospital = await maps.nearest_hospital_with_eta(latitude, longitude)
        if hospital:
            emergency.hospital_name = (hospital.get("name") or "")[:150] or None
            emergency.hospital_latitude = hospital.get("latitude")
            emergency.hospital_longitude = hospital.get("longitude")
            emergency.hospital_distance_km = hospital.get("distance_km")
            # Only set when the distance matrix actually answered. A null ETA
            # is a missing measurement; a guessed one is a claim.
            if hospital.get("duration_minutes") is not None:
                emergency.eta = hospital["duration_minutes"]
            changed = True

        if changed:
            await db.flush()
        return changed

    # ── queueing ─────────────────────────────────────────────────────────

    async def queue_for_emergency(
        self, db: AsyncSession, emergency: EmergencyRequest,
        patient: Optional[Patient] = None,
    ) -> list[CommunicationLog]:
        """
        Write one row per recipient per channel, before anything is sent.

        Idempotent per emergency: if rows already exist this returns them
        untouched, so a retriggered background task cannot double-call a
        family. That matters more here than anywhere else in the system.
        """
        # Existence is a yes/no question, so it asks for one id rather than
        # hydrating every row it is about to decide not to use.
        already_queued = await db.scalar(
            select(CommunicationLog.id)
            .where(CommunicationLog.emergency_id == emergency.id)
            .limit(1)
        )
        if already_queued is not None:
            return list((await db.execute(
                select(CommunicationLog)
                .where(CommunicationLog.emergency_id == emergency.id)
            )).scalars().all())

        recipients = await self.resolve_recipients(db, emergency)
        # Reuses the caller's patient when it has one; only fetches otherwise.
        if patient is None:
            patient = await db.get(Patient, emergency.patient_id)
        ctx = templates.build_context(emergency, patient)

        rows: list[CommunicationLog] = []
        for recipient in recipients:
            for channel in CHANNEL_ORDER:
                template_key = (
                    VOICE_TEMPLATES[recipient["role"]] if channel == "voice"
                    else (templates.TEMPLATE_SMS_ALERT if channel == "sms"
                          else templates.TEMPLATE_WHATSAPP_ALERT)
                )
                body = templates.render(channel, template_key, ctx)
                row = CommunicationLog(
                    emergency_id=emergency.id,
                    channel=channel,
                    recipient_role=recipient["role"],
                    recipient_name=(recipient.get("name") or "")[:200] or None,
                    recipient_phone=(recipient.get("phone") or "")[:32] or None,
                    status="queued",
                    provider="twilio",
                    template_key=template_key,
                    body_preview=body,
                    attempts=0,
                    next_attempt_at=_now(),
                )
                db.add(row)
                rows.append(row)

        await db.flush()
        logger.info("[COMMS_QUEUED] emergency=%s rows=%d", emergency.id, len(rows))
        return rows

    # ── sending ──────────────────────────────────────────────────────────

    async def _deliver(self, row: CommunicationLog) -> DeliveryResult:
        gateway = get_twilio_gateway()
        if row.channel == "voice":
            return await gateway.place_call(row.recipient_phone, row.body_preview)
        if row.channel == "sms":
            return await gateway.send_sms(row.recipient_phone, row.body_preview)
        return await gateway.send_whatsapp(row.recipient_phone, row.body_preview)

    def _apply(self, row: CommunicationLog, result: DeliveryResult) -> None:
        """Fold one attempt's outcome onto the row."""
        row.attempts += 1
        row.provider_sid = result.sid or row.provider_sid
        row.provider_status = result.provider_status
        row.error_code = result.error_code
        row.error_message = result.error_message
        row.duration_seconds = result.duration_seconds

        if result.success:
            # `accepted`, not `sent`. The provider has taken the request; that
            # is not evidence a handset received anything. A status callback
            # moves it on to sent/delivered/undelivered — and where no callback
            # URL is configured it honestly stops here rather than claiming
            # delivery nobody confirmed.
            row.status = "accepted"
            row.sent_at = _now()
            row.next_attempt_at = None

            from app.services.twilio_callbacks import callback_url

            if callback_url(row.channel) is None:
                # No way to hear back, so this attempt is as complete as it
                # will ever be.
                row.completed_at = _now()
            return

        if result.skipped:
            # Nothing was attempted, so this is not a failure to retry — it is
            # a channel or a number that does not exist.
            row.status = "skipped"
            row.completed_at = _now()
            row.next_attempt_at = None
            return

        exhausted = row.attempts >= settings.EMERGENCY_COMMS_MAX_ATTEMPTS
        if not result.retryable or exhausted:
            row.status = "failed"
            row.completed_at = _now()
            row.next_attempt_at = None
            return

        # Exponential backoff: 30s, 60s, 120s… long enough to ride out a vendor
        # blip, bounded so a bad number stops consuming attempts.
        delay = settings.EMERGENCY_COMMS_BACKOFF_SECONDS * (2 ** (row.attempts - 1))
        row.status = "queued"
        row.next_attempt_at = _now() + timedelta(seconds=delay)

    async def _claim(self, db: AsyncSession, row_id: uuid.UUID) -> bool:
        """
        Take ownership of a single row, atomically.

        The in-process sweeper and the Celery task can both be running. The
        conditional update means only one of them transitions a row out of
        `queued`, so a family is not called twice by two workers that both
        thought it was theirs.
        """
        result = await db.execute(
            update(CommunicationLog)
            .where(
                CommunicationLog.id == row_id,
                CommunicationLog.status == "queued",
            )
            .values(status="sending")
        )
        return result.rowcount == 1

    async def _claim_many(
        self, db: AsyncSession, row_ids: list[uuid.UUID]
    ) -> set[uuid.UUID]:
        """
        Claim a whole batch in one statement, returning the ones actually won.

        Claiming row by row cost a round trip *each*, and against a managed
        database on the other side of the internet that was roughly two seconds
        per recipient — a nine-row fan-out took twenty. The same conditional
        update covers the batch, so the mutual exclusion is unchanged and the
        cost is one round trip instead of nine.
        """
        if not row_ids:
            return set()

        result = await db.execute(
            update(CommunicationLog)
            .where(
                CommunicationLog.id.in_(row_ids),
                CommunicationLog.status == "queued",
            )
            .values(status="sending")
            .returning(CommunicationLog.id)
        )
        return set(result.scalars().all())

    async def recover_stalled(self, db: AsyncSession, older_than_seconds: int = 300) -> int:
        """
        Return rows stuck in `sending` to the queue.

        A row is claimed before the provider is called, so a process that dies
        mid-attempt leaves it claimed forever — invisible to a sweep that only
        looks for `queued`. Anything still `sending` well past any plausible
        request is handed back, which is what makes "never lose a notification"
        true across a restart as well as across an outage.
        """
        cutoff = _now() - timedelta(seconds=older_than_seconds)
        result = await db.execute(
            update(CommunicationLog)
            .where(
                CommunicationLog.status == "sending",
                CommunicationLog.updated_at < cutoff,
            )
            .values(status="queued", next_attempt_at=_now())
        )
        if result.rowcount:
            logger.warning("[COMMS_RECOVERED] %d stalled row(s) requeued",
                           result.rowcount)
        return result.rowcount or 0

    async def dispatch_pending(
        self, db: AsyncSession, emergency_id: Optional[uuid.UUID] = None,
        limit: int = 50,
    ) -> dict[str, int]:
        """
        Send everything that is due, in recipient then channel order.

        Used for both the first pass after an SOS and every later retry sweep —
        one code path, so a retry behaves exactly like the original attempt.
        """
        conditions = [
            CommunicationLog.status == "queued",
            or_(
                CommunicationLog.next_attempt_at.is_(None),
                CommunicationLog.next_attempt_at <= _now(),
            ),
        ]
        if emergency_id is not None:
            conditions.append(CommunicationLog.emergency_id == emergency_id)

        rows = (await db.execute(
            select(CommunicationLog)
            .where(and_(*conditions))
            # Recipients in workflow order, and the voice call before the
            # written alerts for each of them.
            #
            # Ordered by an explicit ranking rather than by `created_at`: all
            # nine rows for an emergency are inserted in one transaction and so
            # share a timestamp, which left the tie broken by a random UUID —
            # the administrator could be telephoned before the family. The
            # sequence the workflow specifies has to be stated, not inferred.
            .order_by(
                CommunicationLog.emergency_id,
                _RECIPIENT_RANK,
                _CHANNEL_RANK,
                CommunicationLog.id,
            )
            .limit(limit)
        )).scalars().all()

        # `accepted` rather than `sent`: this counts hand-offs the provider took
        # responsibility for, which is all a dispatch pass can know. Whether a
        # handset received anything arrives later, as a status callback.
        counts = {"accepted": 0, "failed": 0, "skipped": 0, "retrying": 0}

        # One statement claims the whole batch. Doing it per row cost a network
        # round trip each, which against a managed database is most of the
        # wall-clock time of a fan-out.
        claimed = await self._claim_many(db, [row.id for row in rows])
        await db.commit()

        for row in rows:
            if row.id not in claimed:
                continue  # another worker got there first

            result = await self._deliver(row)

            # No `refresh` before applying: this session claimed the row and
            # holds it, so re-reading it is a round trip for data already in
            # hand.
            self._apply(row, result)

            if row.status in ("accepted", "sent", "delivered"):
                counts["accepted"] += 1
            elif row.status == "failed":
                counts["failed"] += 1
            elif row.status == "skipped":
                counts["skipped"] += 1
            else:
                counts["retrying"] += 1

            logger.info(
                "[COMMS_ATTEMPT] emergency=%s channel=%s role=%s -> %s (%s)",
                row.emergency_id, row.channel, row.recipient_role, row.status,
                row.error_code or row.provider_sid or "",
            )

        # One commit for the batch rather than one per recipient. Each commit
        # is a network round trip, and against a managed database that was the
        # bulk of a fan-out's wall-clock time. Crash safety does not depend on
        # committing per row: an interrupted batch leaves its rows `sending`,
        # and `recover_stalled` hands those back to the queue.
        await db.commit()
        return counts

    # ── the background entry point ───────────────────────────────────────

    async def run_for_emergency(self, emergency_id: uuid.UUID) -> None:
        """
        Everything that happens after an SOS is recorded.

        Runs on its own database session because it executes after the request
        that triggered it has already returned and its session is closed.
        Nothing here may raise: this is a background task, and an exception
        would be logged by the framework and otherwise vanish, leaving the
        emergency recorded but nobody told.
        """
        from app.core.database import AsyncSessionLocal

        # Yield once before touching the database. This coroutine is started
        # while the request that triggered it is still serialising its response,
        # and both compete for the same connection pool and event loop — so the
        # first thing the fan-out does is let the response go. Measured, not
        # assumed: the request path and the fan-out each cost seconds against a
        # remote database, and interleaving them added the two together.
        await asyncio.sleep(0)

        try:
            async with AsyncSessionLocal() as db:
                emergency = await db.get(EmergencyRequest, emergency_id)
                if emergency is None:
                    logger.warning("[COMMS_NO_EMERGENCY] %s", emergency_id)
                    return

                # Location first: an address and a hospital found now appear in
                # the messages, instead of arriving after they have been sent.
                patient = await db.get(Patient, emergency.patient_id)
                await self.enrich_location(db, emergency)
                await self.queue_for_emergency(db, emergency, patient=patient)
                await db.commit()

                await self.announce(db, emergency_id)
                await self.dispatch_pending(db, emergency_id=emergency_id)
                await self.announce(db, emergency_id)
        except Exception as exc:
            logger.exception("[COMMS_RUN_FAILED] emergency=%s: %s",
                             emergency_id, exc)

    async def announce(self, db: AsyncSession, emergency_id: uuid.UUID) -> None:
        """Push the communication state to the open dashboards."""
        try:
            from app.services.sos_notifications import get_emergency_notifier

            summary = await self.summarise(db, emergency_id)
            await get_emergency_notifier().communications_updated({
                "emergency_id": str(emergency_id),
                "patient_id": summary.get("patient_id"),
                "communications": summary.get("communications", []),
            })
        except Exception as exc:
            logger.warning("[COMMS_ANNOUNCE_FAILED] %s: %s", emergency_id, exc)

    # ── reading ──────────────────────────────────────────────────────────

    async def summarise(
        self, db: AsyncSession, emergency_id: uuid.UUID
    ) -> dict[str, Any]:
        """The communication log for one emergency, with numbers masked."""
        # `patient_id` comes back with the rows rather than from a second
        # lookup: `summarise` runs on every announce, so a spare round trip
        # here is paid twice per emergency.
        patient_id = await db.scalar(
            select(EmergencyRequest.patient_id)
            .where(EmergencyRequest.id == emergency_id)
        )
        rows = (await db.execute(
            select(CommunicationLog)
            .where(CommunicationLog.emergency_id == emergency_id)
            # The same explicit ranking the dispatcher uses. Ordering by
            # `created_at` alone left the tie to a random UUID, so the API
            # listed the administrator's call above the family's.
            .order_by(_RECIPIENT_RANK, _CHANNEL_RANK, CommunicationLog.id)
        )).scalars().all()

        return {
            "emergency_id": str(emergency_id),
            "patient_id": str(patient_id) if patient_id else None,
            "communications": [
                {
                    "id": str(row.id),
                    "channel": row.channel,
                    "recipient_role": row.recipient_role,
                    "recipient_name": row.recipient_name,
                    "recipient_phone_masked": mask_phone(row.recipient_phone),
                    "status": row.status,
                    "provider": row.provider,
                    "provider_sid": row.provider_sid,
                    "provider_status": row.provider_status,
                    "error_code": row.error_code,
                    "error_message": row.error_message,
                    "attempts": row.attempts,
                    "next_attempt_at": row.next_attempt_at,
                    "sent_at": row.sent_at,
                    "completed_at": row.completed_at,
                    "duration_seconds": row.duration_seconds,
                    "created_at": row.created_at,
                }
                for row in rows
            ],
        }

    async def sweep_retries(self, limit: int = 100) -> dict[str, int]:
        """
        Send everything due across every emergency.

        The entry point for both the in-process loop and the Celery beat task.
        """
        from app.core.database import AsyncSessionLocal

        try:
            async with AsyncSessionLocal() as db:
                # Hand back anything a dead process left claimed, then send.
                await self.recover_stalled(db)
                await db.commit()
                return await self.dispatch_pending(db, limit=limit)
        except Exception as exc:
            logger.error("[COMMS_SWEEP_FAILED] %s", exc)
            # Same keys `dispatch_pending` returns, so a caller that reads the
            # tally does not have to know whether the sweep failed.
            return {"accepted": 0, "failed": 0, "skipped": 0, "retrying": 0}


emergency_comms_service = EmergencyCommsService()


_detached: set[asyncio.Task] = set()
"""
Strong references to in-flight fan-outs.

`asyncio` holds only a weak reference to a running task, so one that nothing
else keeps can be garbage-collected mid-await — silently abandoning an
emergency's notifications. The set holds each task until it finishes.
"""


def spawn_communications(emergency_id: uuid.UUID) -> None:
    """
    Start the fan-out without making anybody wait for it.

    Deliberately not `BackgroundTasks`. Starlette runs those *within* the
    request's ASGI cycle: the response is written first, but the cycle does not
    complete until the task does, and a client on a keep-alive connection is
    still waiting. Measured against the production database that made a
    perfectly good SOS take seven seconds to return.

    A detached task decouples the two properly — the emergency is already
    committed and broadcast, and the work that follows reads it back from its
    own session, so nothing is lost by letting the request go.
    """
    if "pytest" in sys.modules:
        # Not spawned under test. The suite drives `run_for_emergency` and
        # `dispatch_pending` directly — which is what it wants to assert on —
        # and a detached writer racing the test's own session on SQLite
        # deadlocks it. The same guard keeps the lifespan sweeper out of tests.
        return

    task = asyncio.create_task(
        emergency_comms_service.run_for_emergency(emergency_id)
    )
    _detached.add(task)
    task.add_done_callback(_detached.discard)


async def retry_sweep_loop() -> None:
    """
    The always-available retry worker.

    Celery covers this too, but Celery needs a broker and an emergency
    notification must not depend on one being reachable. Both claim rows with
    the same conditional update, so running both is safe and running only one
    is enough.
    """
    interval = max(5, settings.EMERGENCY_RETRY_SWEEP_SECONDS)
    logger.info("[COMMS_SWEEPER] started, every %ss", interval)
    while True:
        try:
            await asyncio.sleep(interval)
            await emergency_comms_service.sweep_retries()
        except asyncio.CancelledError:
            logger.info("[COMMS_SWEEPER] stopped")
            raise
        except Exception as exc:
            # Never let one bad sweep end the loop; the next one may succeed.
            logger.error("[COMMS_SWEEPER_ERROR] %s", exc)
