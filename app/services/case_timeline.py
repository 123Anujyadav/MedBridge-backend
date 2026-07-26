"""
Case timeline and audit trail.

The timeline has two kinds of entry, and the difference is stated in the payload
rather than blurred:

**Recorded** events come from `audit_logs`. They are written at the moment an
action happens, so they carry the actor, the reason, and the before/after values
of whatever changed. These are authoritative.

**Derived** milestones are read from the clinical tables themselves — the case
row, the intake session, symptoms, reports, prescriptions, appointments. They
exist because the platform's history predates event recording, and because some
milestones are simply facts about a row (a prescription was created when its
`created_at` says it was). They carry a real timestamp from a real row and are
labelled `source="derived"`; they never carry an actor the record cannot prove.

What neither kind does is invent. A milestone with no timestamp behind it is
omitted, not dated by guesswork, and every derived event is scoped by the case
foreign key — never by "the patient's most recent case".
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthorizationException, EntityNotFoundException
from app.models.appointment import Appointment
from app.models.audit import AuditLog
from app.models.case import Case, Symptom
from app.models.intake import IntakeExtractedEntity, IntakeSessionRecord
from app.models.patient import Patient
from app.models.prescription import Prescription
from app.models.report import Report
from app.models.user import User

logger = logging.getLogger(__name__)

# Actor vocabulary. `ai` and `system` are not users; they are recorded with a
# NULL user_id and a stable display name.
ACTOR_LABELS = {
    "patient": "Patient",
    "doctor": "Doctor",
    "ai": "AI Assistant",
    "admin": "Administrator",
    "system": "System",
}

# event_type -> filter category, for the timeline's filter chips.
_CATEGORIES: dict[str, str] = {
    "case.created": "clinical",
    "case.status_changed": "clinical",
    "case.urgency_changed": "clinical",
    "case.assigned": "clinical",
    "case.opened": "doctor",
    "case.closed": "clinical",
    "case.notes_added": "clinical",
    "case.diagnosis_updated": "clinical",
    "patient.symptoms_submitted": "patient",
    "patient.report_viewed": "patient",
    "patient.report_downloaded": "patient",
    "attachment.uploaded": "patient",
    "ai.intake_started": "ai",
    "ai.entities_extracted": "ai",
    "ai.summary_generated": "ai",
    "ai.urgency_assessed": "ai",
    "ai.specialist_recommended": "ai",
    "ai.summary_approved": "doctor",
    "report.generated": "reports",
    "report.status_changed": "reports",
    "report.approved": "reports",
    "report.rejected": "reports",
    "report.archived": "reports",
    "report.flagged_follow_up": "reports",
    "prescription.created": "prescriptions",
    "appointment.scheduled": "appointments",
    "appointment.rescheduled": "appointments",
    "system.notification_sent": "system",
}

FILTER_CATEGORIES = (
    "ai", "doctor", "patient", "system", "clinical",
    "reports", "prescriptions", "appointments",
)


def _iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value or "").strip()
    return text or None


def _sort_key(event: dict[str, Any]) -> str:
    """Sortable timestamp. Undated events cannot exist, so this is total."""
    return event["timestamp"] or ""


class CaseTimelineService:
    """Builds the case timeline and appends audit entries."""

    # ── Write ────────────────────────────────────────────────────────────

    async def record(
        self,
        db: AsyncSession,
        *,
        event_type: str,
        description: str,
        actor: Optional[User] = None,
        actor_type: Optional[str] = None,
        actor_name: Optional[str] = None,
        case_id: Optional[uuid.UUID] = None,
        patient_id: Optional[uuid.UUID] = None,
        resource: str = "Case",
        resource_id: Optional[str] = None,
        field: Optional[str] = None,
        previous: Optional[Any] = None,
        new: Optional[Any] = None,
        reason: Optional[str] = None,
        ip_address: str = "server",
        status: str = "success",
    ) -> AuditLog:
        """
        Append one immutable event.

        Actor identity is taken from the authenticated `User` when there is one
        and is never accepted from a request body. AI and system events pass an
        explicit `actor_type` with no user, which is why `user_id` is nullable.

        Failures here are swallowed by callers: an audit write must never be the
        reason a clinical action fails. It is logged loudly instead.
        """
        if actor is not None:
            resolved_type = actor_type or actor.role or "system"
            resolved_name = actor_name or actor.email
        else:
            resolved_type = actor_type or "system"
            resolved_name = actor_name or ACTOR_LABELS.get(resolved_type, "System")

        if resolved_type not in ACTOR_LABELS:
            resolved_type = "system"

        entry = AuditLog(
            user_id=actor.id if actor is not None else None,
            user_name=resolved_name[:200],
            user_role=(actor.role if actor is not None else resolved_type)[:50],
            action=event_type.upper().replace(".", "_")[:100],
            resource=resource[:100],
            resource_id=str(resource_id or case_id or "")[:100],
            ip_address=ip_address[:50],
            status=status,
            details=description[:5000],
            case_id=case_id,
            patient_id=patient_id,
            actor_type=resolved_type,
            event_type=event_type[:60],
            field_changed=field[:80] if field else None,
            # None, not "": a NULL means "this event changed no tracked value",
            # while an empty string would read as "changed to blank".
            previous_value=None if previous is None else str(previous)[:2000],
            new_value=None if new is None else str(new)[:2000],
            reason=(reason or None),
        )
        db.add(entry)
        await db.flush()
        return entry

    async def safe_record(self, db: AsyncSession, **kwargs: Any) -> None:
        """`record` that never propagates — for use inside clinical writes."""
        try:
            await self.record(db, **kwargs)
        except Exception:
            logger.exception(
                "[TIMELINE] failed to record %s", kwargs.get("event_type")
            )

    # ── Read ─────────────────────────────────────────────────────────────

    async def build(
        self,
        db: AsyncSession,
        case_id: uuid.UUID,
        user: User,
        *,
        categories: Iterable[str] | None = None,
        search: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        The full chronological history of one case, newest first.

        Authorisation mirrors the rest of the platform: the case's own patient,
        its assigned doctor, or an admin.
        """
        case = await self._load_authorised_case(db, case_id, user)

        recorded = await self._recorded_events(db, case_id)
        derived = await self._derived_events(db, case)

        # Recorded events win: when both describe the same moment, the one with
        # a proven actor and before/after values is the better record.
        recorded_keys = {(e["event_type"], e["timestamp"]) for e in recorded}
        events = recorded + [
            e for e in derived if (e["event_type"], e["timestamp"]) not in recorded_keys
        ]

        events = self._apply_filters(
            events, categories=categories, search=search,
            date_from=date_from, date_to=date_to,
        )
        events.sort(key=_sort_key, reverse=True)

        total = len(events)
        page = events[skip : skip + limit]
        return {
            "case_id": case_id,
            "total": total,
            "returned": len(page),
            "skip": skip,
            "limit": limit,
            "has_more": skip + len(page) < total,
            "events": page,
        }

    async def _load_authorised_case(
        self, db: AsyncSession, case_id: uuid.UUID, user: User
    ) -> Case:
        case = await db.get(Case, case_id)
        if case is None or case.deleted_at is not None:
            raise EntityNotFoundException("Case", str(case_id))

        if user.role == "patient":
            if case.patient_id != user.id:
                raise AuthorizationException("Access denied to this case history.")
        elif user.role == "doctor":
            if case.doctor_id != user.id:
                raise AuthorizationException("Access denied to this case history.")
        elif user.role != "admin":
            raise AuthorizationException("Access denied to this case history.")
        return case

    async def recent_for_doctor(
        self, db: AsyncSession, doctor_id: uuid.UUID, *, limit: int = 15
    ) -> list[dict[str, Any]]:
        """
        Latest events across every case this doctor owns.

        The dashboard activity feed reads the same event store as the per-case
        timeline rather than assembling a second, parallel history — scoped by
        `cases.doctor_id`, so it can never surface another clinician's activity.
        """
        own_cases = select(Case.id).where(Case.doctor_id == doctor_id).scalar_subquery()
        result = await db.execute(
            select(AuditLog)
            .where(AuditLog.case_id.in_(own_cases))
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return [
            {
                "id": str(row.id),
                "event_type": row.event_type or row.action.lower(),
                "category": _CATEGORIES.get(row.event_type or "", "clinical"),
                "title": self._title_for(row.event_type or row.action),
                "description": row.details or "",
                "timestamp": _iso(row.created_at),
                "actor_type": row.actor_type,
                "actor_label": ACTOR_LABELS.get(row.actor_type, "System"),
                "actor_name": row.user_name,
                "case_id": str(row.case_id) if row.case_id else None,
            }
            for row in result.scalars().all()
        ]

    async def _recorded_events(
        self, db: AsyncSession, case_id: uuid.UUID
    ) -> list[dict[str, Any]]:
        """Audit entries explicitly linked to this case by foreign key."""
        result = await db.execute(
            select(AuditLog)
            .where(AuditLog.case_id == case_id)
            .order_by(AuditLog.created_at.desc())
        )
        rows = list(result.scalars().all())
        return [
            {
                "id": str(row.id),
                "event_type": row.event_type or row.action.lower(),
                "category": _CATEGORIES.get(row.event_type or "", "clinical"),
                "title": self._title_for(row.event_type or row.action),
                "description": row.details or "",
                "timestamp": _iso(row.created_at),
                "actor_type": row.actor_type,
                "actor_label": ACTOR_LABELS.get(row.actor_type, "System"),
                "actor_name": row.user_name,
                "field_changed": row.field_changed,
                "previous_value": row.previous_value,
                "new_value": row.new_value,
                "reason": row.reason,
                "source": "recorded",
            }
            for row in rows
        ]

    async def _derived_events(
        self, db: AsyncSession, case: Case
    ) -> list[dict[str, Any]]:
        """
        Milestones read from the case's own rows.

        Every query below is scoped by the case foreign key. Nothing is included
        because it merely belongs to the same patient.
        """
        events: list[dict[str, Any]] = []

        def add(event_type, title, description, timestamp, actor_type,
                actor_name=None, **extra):
            # No timestamp means the milestone cannot be honestly placed on a
            # chronological axis, so it is left out entirely.
            stamp = _iso(timestamp)
            if not stamp:
                return
            events.append({
                "id": f"derived:{event_type}:{stamp}",
                "event_type": event_type,
                "category": _CATEGORIES.get(event_type, "clinical"),
                "title": title,
                "description": description,
                "timestamp": stamp,
                "actor_type": actor_type,
                "actor_label": ACTOR_LABELS.get(actor_type, "System"),
                "actor_name": actor_name or ACTOR_LABELS.get(actor_type, "System"),
                "field_changed": extra.get("field"),
                "previous_value": extra.get("previous"),
                "new_value": extra.get("new"),
                "reason": None,
                "source": "derived",
            })

        patient = await db.get(Patient, case.patient_id)
        patient_name = (
            f"{patient.first_name} {patient.last_name}".strip()
            if patient else case.patient_name
        )

        add("case.created", "Case Created",
            case.symptom_summary or "Consultation case opened.",
            case.created_at, "patient", patient_name)

        # Symptoms recorded against this case.
        symptoms = list((await db.execute(
            select(Symptom).where(Symptom.case_id == case.id)
            .order_by(Symptom.created_at.asc())
        )).scalars().all())
        if symptoms:
            add("patient.symptoms_submitted", "Patient Submitted Symptoms",
                ", ".join(s.name for s in symptoms[:8]),
                symptoms[0].created_at, "patient", patient_name)

        # AI intake, scoped by routed_case_id.
        intake = (await db.execute(
            select(IntakeSessionRecord)
            .where(IntakeSessionRecord.routed_case_id == case.id)
            .order_by(IntakeSessionRecord.created_at.desc()).limit(1)
        )).scalars().first()

        if intake is not None:
            add("ai.intake_started", "AI Medical Intake Started",
                f"Language {intake.language}, {intake.followup_rounds} follow-up round(s).",
                intake.created_at, "ai")

            entity_count = len(list((await db.execute(
                select(IntakeExtractedEntity.id)
                .where(IntakeExtractedEntity.session_id == intake.id)
                .where(IntakeExtractedEntity.was_accepted.is_(True))
            )).scalars().all()))
            if entity_count:
                add("ai.entities_extracted", "AI Entity Extraction Completed",
                    f"{entity_count} clinical entities extracted with evidence.",
                    intake.created_at, "ai")

            snapshot = intake.medical_case_snapshot
            if isinstance(snapshot, dict) and snapshot:
                add("ai.summary_generated", "AI Clinical Summary Generated",
                    str(snapshot.get("summary_for_doctor") or "Structured case generated."),
                    intake.updated_at, "ai")
                if snapshot.get("urgency"):
                    add("ai.urgency_assessed", "AI Urgency Assessment Completed",
                        f"Assessed urgency: {snapshot['urgency']}.",
                        intake.updated_at, "ai",
                        field="urgency", new=snapshot["urgency"])
                if snapshot.get("recommended_specialty"):
                    add("ai.specialist_recommended", "AI Recommended Specialist",
                        f"Recommended {snapshot['recommended_specialty']}.",
                        intake.updated_at, "ai",
                        field="specialty", new=snapshot["recommended_specialty"])

        if case.assigned_at:
            add("case.assigned", "Doctor Assigned",
                case.doctor_name or "Assigned to a clinician.",
                case.assigned_at, "system")

        # Reports linked to this case.
        for report in (await db.execute(
            select(Report).where(Report.case_id == case.id)
            .order_by(Report.created_at.asc())
        )).scalars().all():
            add("report.generated",
                "AI Clinical Report Generated" if report.ai_generated
                else "Clinical Report Created",
                report.title, report.created_at,
                "ai" if report.ai_generated else "doctor",
                report.doctor_name)

        for rx in (await db.execute(
            select(Prescription).where(Prescription.case_id == case.id)
            .order_by(Prescription.created_at.asc())
        )).scalars().all():
            add("prescription.created", "Prescription Created",
                rx.diagnosis, rx.created_at, "doctor", rx.doctor_name)

        for appt in (await db.execute(
            select(Appointment).where(Appointment.case_id == case.id)
            .order_by(Appointment.created_at.asc())
        )).scalars().all():
            add("appointment.scheduled", "Follow-up Scheduled",
                f"{appt.date} {appt.time} — {appt.reason}",
                appt.created_at, "doctor", appt.doctor_name)

        if case.completed_at:
            add("case.closed", "Case Closed",
                f"Case marked {case.status}.", case.completed_at, "doctor",
                case.doctor_name)

        return events

    # ── Filtering ────────────────────────────────────────────────────────

    @staticmethod
    def _apply_filters(
        events: list[dict[str, Any]],
        *,
        categories: Iterable[str] | None,
        search: str | None,
        date_from: str | None,
        date_to: str | None,
    ) -> list[dict[str, Any]]:
        wanted = {c for c in (categories or []) if c in FILTER_CATEGORIES}
        if wanted:
            events = [e for e in events if e["category"] in wanted]

        if search:
            needle = search.strip().casefold()
            events = [
                e for e in events
                if needle in " ".join(
                    str(e.get(k) or "") for k in
                    ("title", "description", "actor_name", "event_type",
                     "previous_value", "new_value", "reason")
                ).casefold()
            ]

        if date_from:
            events = [e for e in events if (e["timestamp"] or "") >= date_from]
        if date_to:
            # Inclusive of the whole day when a bare date is supplied.
            upper = date_to if len(date_to) > 10 else f"{date_to}T23:59:59.999999+00:00"
            events = [e for e in events if (e["timestamp"] or "") <= upper]

        return events

    @staticmethod
    def _title_for(event_type: str) -> str:
        """Human title from a semantic key, without inventing wording."""
        titles = {
            "case.created": "Case Created",
            "case.opened": "Doctor Opened Case",
            "case.assigned": "Doctor Assigned",
            "case.notes_added": "Doctor Added Clinical Notes",
            "case.diagnosis_updated": "Diagnosis Updated",
            "case.status_changed": "Case Status Changed",
            "case.urgency_changed": "Urgency Changed",
            "case.closed": "Case Closed",
            "patient.symptoms_submitted": "Patient Submitted Symptoms",
            "patient.report_viewed": "Patient Viewed Report",
            "patient.report_downloaded": "Patient Downloaded Report",
            "ai.intake_started": "AI Medical Intake Started",
            "ai.entities_extracted": "AI Entity Extraction Completed",
            "ai.summary_generated": "AI Clinical Summary Generated",
            "ai.urgency_assessed": "AI Urgency Assessment Completed",
            "ai.specialist_recommended": "AI Recommended Specialist",
            "ai.summary_approved": "Doctor Approved AI Summary",
            "report.generated": "Clinical Report Generated",
            "report.status_changed": "Report Status Changed",
            "report.approved": "Doctor Approved Report",
            "report.rejected": "Report Rejected",
            "report.archived": "Report Archived",
            "report.flagged_follow_up": "Report Flagged for Follow-up",
            "prescription.created": "Prescription Created",
            "appointment.scheduled": "Follow-up Scheduled",
        }
        if event_type in titles:
            return titles[event_type]
        return event_type.replace(".", " ").replace("_", " ").title()


case_timeline_service = CaseTimelineService()
