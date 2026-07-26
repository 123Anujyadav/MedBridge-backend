"""
Operational and clinical analytics for one doctor.

Three rules shaped this service:

**Only measurable things are reported.** Where the platform does not record what
a metric would need — AI processing time is never timed, and medications carry a
name but no therapeutic class — the metric comes back `None` with the reason in
`unavailable_metrics`. A zero would read as a measurement, and a plausible
number would be a lie a clinician might act on.

**Diagnoses come from clinicians, never from the model.** `common_diagnoses` is
built from `prescriptions.diagnosis`, which only a doctor writes. AI-suggested
differentials are deliberately excluded — counting them would turn model output
into apparent clinical consensus.

**Aggregate in SQL, not in Python.** Every figure below is a `COUNT`, `AVG` or
`GROUP BY` bounded to this doctor's own rows. The previous implementation pulled
entire case and patient collections into memory and looped, which is fine at
demo scale and falls over at clinic scale.

Everything is scoped through `cases.doctor_id`, so one clinician's dashboard can
never surface another's patients.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import String, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.appointment import Appointment
from app.models.audit import AuditLog
from app.models.case import Case, Symptom
from app.models.intake import IntakeSessionRecord
from app.models.notification import NotificationItem
from app.models.patient import Patient
from app.models.prescription import Medication, Prescription
from app.models.report import Report

logger = logging.getLogger(__name__)

RANGE_PRESETS = {
    "today": 0,
    "yesterday": 1,
    "7d": 6,
    "30d": 29,
    "90d": 89,
}

OPEN_CASE_STATES = ("intake", "ai_processing", "routed", "in_consultation", "prescribed")
PENDING_REPORT_STATES = ("pending", "pending_review", "needs_revision")
APPROVED_REPORT_STATES = ("ready", "approved", "reviewed")

# Metrics the data model cannot support, and why. Surfaced to the client so the
# dashboard can say "not measured" rather than rendering an empty chart.
UNAVAILABLE = {
    "avg_ai_processing_time_seconds":
        "The AI pipeline does not record per-request timings, so processing "
        "time cannot be measured.",
    "medication_categories":
        "Medications are stored by name only; no therapeutic classification is "
        "recorded, so categories cannot be derived. Top medications by name are "
        "reported instead.",
}


def resolve_range(
    preset: Optional[str], date_from: Optional[str], date_to: Optional[str]
) -> tuple[datetime, datetime, str]:
    """
    Turn a preset or explicit dates into an inclusive UTC window.

    A custom range wins over a preset; an unrecognised preset falls back to 30
    days rather than silently returning all history.
    """
    now = datetime.now(timezone.utc)
    today = now.date()

    if date_from or date_to:
        try:
            start_date = date.fromisoformat(date_from) if date_from else today
            end_date = date.fromisoformat(date_to) if date_to else today
        except ValueError:
            start_date, end_date = today - timedelta(days=29), today
        label = "custom"
    elif preset == "yesterday":
        start_date = end_date = today - timedelta(days=1)
        label = preset
    else:
        days = RANGE_PRESETS.get(preset or "30d", 29)
        start_date, end_date = today - timedelta(days=days), today
        label = preset or "30d"

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    start = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
    return start, end, label


def _rounded(value: Any, digits: int = 1) -> Optional[float]:
    """Round, or None. Never coerces a missing average into 0."""
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


class DoctorAnalyticsService:
    """Builds the doctor dashboard analytics payload."""

    async def build(
        self,
        db: AsyncSession,
        doctor_id: uuid.UUID,
        *,
        preset: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> dict[str, Any]:
        start, end, label = resolve_range(preset, date_from, date_to)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        own_cases = select(Case.id).where(Case.doctor_id == doctor_id).scalar_subquery()
        own_patients = (
            select(Case.patient_id).where(Case.doctor_id == doctor_id).scalar_subquery()
        )

        summary = await self._summary(db, doctor_id, own_patients, today_str, start, end)
        workload = await self._workload(db, doctor_id, own_cases, start, end)
        patients = await self._patients(db, doctor_id, own_patients, own_cases, start, end)
        ai = await self._ai(db, doctor_id, own_cases, start, end)
        reports = await self._reports(db, own_patients, own_cases, start, end)
        prescriptions = await self._prescriptions(db, doctor_id, start, end)
        appointments = await self._appointments(db, doctor_id, today_str, start, end)

        return {
            "range": {
                "preset": label,
                "date_from": start.date().isoformat(),
                "date_to": end.date().isoformat(),
            },
            "summary": summary,
            "workload": workload,
            "patients": patients,
            "ai": ai,
            "reports": reports,
            "prescriptions": prescriptions,
            "appointments": appointments,
            "unavailable_metrics": [
                {"metric": k, "reason": v} for k, v in UNAVAILABLE.items()
            ],
        }

    # ── Summary cards ────────────────────────────────────────────────────

    async def _summary(
        self, db: AsyncSession, doctor_id, own_patients, today_str, start, end
    ) -> dict[str, Any]:
        appts_today = await db.scalar(
            select(func.count(Appointment.id))
            .where(Appointment.doctor_id == doctor_id)
            .where(Appointment.date == today_str)
        )
        seen_today = await db.scalar(
            select(func.count(func.distinct(Appointment.patient_id)))
            .where(Appointment.doctor_id == doctor_id)
            .where(Appointment.date == today_str)
            .where(Appointment.status == "completed")
        )
        pending_reports = await db.scalar(
            select(func.count(Report.id))
            .where(Report.patient_id.in_(own_patients))
            .where(Report.status.in_(PENDING_REPORT_STATES))
        )
        follow_up = await db.scalar(
            select(func.count(Report.id))
            .where(Report.patient_id.in_(own_patients))
            .where(Report.flagged_for_follow_up.is_(True))
        )
        critical = await db.scalar(
            select(func.count(Case.id))
            .where(Case.doctor_id == doctor_id)
            .where(Case.urgency_level == "critical")
            .where(Case.status.in_(OPEN_CASE_STATES))
        )
        completed = await db.scalar(
            select(func.count(Appointment.id))
            .where(Appointment.doctor_id == doctor_id)
            .where(Appointment.status == "completed")
            .where(Appointment.created_at.between(start, end))
        )
        unread = await db.scalar(
            select(func.count(NotificationItem.id))
            .where(NotificationItem.user_id == doctor_id)
            .where(NotificationItem.read.is_(False))
        )
        return {
            "todays_appointments": appts_today or 0,
            "pending_ai_reviews": pending_reports or 0,
            "completed_consultations": completed or 0,
            "pending_reports": pending_reports or 0,
            "critical_cases": critical or 0,
            "follow_up_cases": follow_up or 0,
            "unread_notifications": unread or 0,
            "patients_seen_today": seen_today or 0,
        }

    # ── Clinical workload ────────────────────────────────────────────────

    async def _workload(
        self, db: AsyncSession, doctor_id, own_cases, start, end
    ) -> dict[str, Any]:
        opened = await db.scalar(
            select(func.count(Case.id))
            .where(Case.doctor_id == doctor_id)
            .where(Case.created_at.between(start, end))
        )
        completed = await db.scalar(
            select(func.count(Case.id))
            .where(Case.doctor_id == doctor_id)
            .where(Case.completed_at.between(start, end))
        )
        pending = await db.scalar(
            select(func.count(Case.id))
            .where(Case.doctor_id == doctor_id)
            .where(Case.status.in_(OPEN_CASE_STATES))
        )

        # Consultation duration: assignment to completion, over cases that have
        # both timestamps. Cases missing either are excluded rather than treated
        # as zero-length.
        #
        # The pairs are averaged in Python rather than with SQL date arithmetic:
        # `extract(epoch ...)` is PostgreSQL-only and this application also runs
        # on SQLite. It is still a single bounded query, not a loop of them.
        consult_pairs = (await db.execute(
            select(Case.assigned_at, Case.completed_at)
            .where(Case.doctor_id == doctor_id)
            .where(Case.assigned_at.isnot(None))
            .where(Case.completed_at.isnot(None))
            .where(Case.completed_at.between(start, end))
        )).all()
        consult_seconds = self._average_seconds(consult_pairs)

        # Review turnaround: report creation to its recorded approval event.
        # Sourced from the audit trail, so it measures a real decision moment
        # rather than a status column that may have been set at any time.
        review_pairs = (await db.execute(
            select(Report.created_at, func.min(AuditLog.created_at))
            .select_from(Report)
            .join(AuditLog, AuditLog.resource_id == cast(Report.id, String))
            .where(AuditLog.event_type == "report.approved")
            .where(AuditLog.case_id.in_(own_cases))
            .where(AuditLog.created_at.between(start, end))
            .group_by(Report.id, Report.created_at)
        )).all()
        review_seconds = self._average_seconds(review_pairs)

        by_specialty = await self._group_counts(
            db, Case.specialty,
            [Case.doctor_id == doctor_id, Case.created_at.between(start, end)],
        )
        by_urgency = await self._group_counts(
            db, Case.urgency_level,
            [Case.doctor_id == doctor_id, Case.created_at.between(start, end)],
        )

        return {
            "cases_opened": opened or 0,
            "cases_completed": completed or 0,
            "pending_cases": pending or 0,
            "avg_consultation_minutes": _rounded(
                consult_seconds / 60 if consult_seconds else None
            ),
            "avg_review_minutes": _rounded(
                review_seconds / 60 if review_seconds else None
            ),
            "cases_by_specialty": by_specialty,
            "cases_by_urgency": by_urgency,
        }

    @staticmethod
    def _average_seconds(pairs: list) -> Optional[float]:
        """
        Mean gap in seconds between two timestamps, or None.

        None rather than 0 when nothing qualifies: a doctor who has completed no
        consultations has no average duration, and rendering "0 min" would claim
        instantaneous consultations.
        """
        deltas = [
            (later - earlier).total_seconds()
            for earlier, later in pairs
            if earlier and later and later >= earlier
        ]
        return sum(deltas) / len(deltas) if deltas else None

    # ── Patient analytics ────────────────────────────────────────────────

    async def _patients(
        self, db: AsyncSession, doctor_id, own_patients, own_cases, start, end
    ) -> dict[str, Any]:
        # A patient is "new" when their first case with this doctor falls in the
        # window, and "returning" when they had one before it.
        first_seen = (
            select(
                Case.patient_id.label("patient_id"),
                func.min(Case.created_at).label("first_case"),
            )
            .where(Case.doctor_id == doctor_id)
            .group_by(Case.patient_id)
            .subquery()
        )
        new_patients = await db.scalar(
            select(func.count()).select_from(first_seen)
            .where(first_seen.c.first_case.between(start, end))
        )
        returning = await db.scalar(
            select(func.count(func.distinct(Case.patient_id)))
            .where(Case.doctor_id == doctor_id)
            .where(Case.created_at.between(start, end))
            .where(Case.patient_id.in_(
                select(first_seen.c.patient_id).where(first_seen.c.first_case < start)
            ))
        )

        gender = await self._group_counts(
            db, Patient.gender, [Patient.id.in_(own_patients)], model=Patient
        )

        # Ages are bucketed in Python because date_of_birth is stored as text,
        # which no portable SQL date function can bucket. One bounded query,
        # not a per-patient lookup.
        dobs = list((await db.execute(
            select(Patient.date_of_birth).where(Patient.id.in_(own_patients))
        )).scalars().all())
        age_distribution = self._bucket_ages(dobs)

        symptoms = await self._group_counts(
            db, Symptom.name, [Symptom.case_id.in_(own_cases)], model=Symptom, limit=10
        )

        # Doctor-confirmed only: a prescription diagnosis is written by a
        # clinician. AI differentials are deliberately not counted here.
        diagnoses = await self._group_counts(
            db, Prescription.diagnosis,
            [Prescription.doctor_id == doctor_id,
             Prescription.created_at.between(start, end)],
            model=Prescription, limit=10,
        )
        specialties = await self._group_counts(
            db, Case.specialty,
            [Case.doctor_id == doctor_id, Case.created_at.between(start, end)],
            limit=10,
        )

        return {
            "new_patients": new_patients or 0,
            "returning_patients": returning or 0,
            "age_distribution": age_distribution,
            "gender_distribution": gender,
            "common_symptoms": symptoms,
            "common_diagnoses": diagnoses,
            "diagnoses_source": "Doctor-issued prescriptions only",
            "most_requested_specialties": specialties,
        }

    @staticmethod
    def _bucket_ages(dobs: list[str | None]) -> list[dict[str, Any]]:
        """WHO-ish age bands. Unparseable dates are skipped, never defaulted."""
        buckets = {"0-17": 0, "18-35": 0, "36-50": 0, "51-65": 0, "65+": 0}
        year_now = datetime.now(timezone.utc).year
        for dob in dobs:
            if not dob:
                continue
            try:
                age = year_now - int(str(dob).split("-")[0])
            except (ValueError, IndexError):
                continue
            if not 0 <= age <= 130:
                continue
            if age < 18:
                buckets["0-17"] += 1
            elif age <= 35:
                buckets["18-35"] += 1
            elif age <= 50:
                buckets["36-50"] += 1
            elif age <= 65:
                buckets["51-65"] += 1
            else:
                buckets["65+"] += 1
        return [{"name": k, "value": v} for k, v in buckets.items() if v]

    # ── AI analytics ─────────────────────────────────────────────────────

    async def _ai(self, db: AsyncSession, doctor_id, own_cases, start, end) -> dict[str, Any]:
        analyses = await db.scalar(
            select(func.count(IntakeSessionRecord.id))
            .where(IntakeSessionRecord.routed_case_id.in_(own_cases))
            .where(IntakeSessionRecord.created_at.between(start, end))
        )

        # Review outcomes come from the audit trail, so each is a decision a
        # clinician actually made rather than an inference about intent.
        async def audit_count(*event_types: str) -> int:
            return await db.scalar(
                select(func.count(AuditLog.id))
                .where(AuditLog.case_id.in_(own_cases))
                .where(AuditLog.event_type.in_(event_types))
                .where(AuditLog.created_at.between(start, end))
            ) or 0

        accepted = await audit_count("ai.summary_approved", "report.approved")
        rejected = await audit_count("report.rejected")
        modified = await audit_count("case.diagnosis_updated", "report.status_changed")

        confidence = await db.scalar(
            select(func.avg(Case.ai_confidence_score))
            .where(Case.doctor_id == doctor_id)
            .where(Case.ai_confidence_score > 0)
            .where(Case.created_at.between(start, end))
        )

        return {
            "analyses_generated": analyses or 0,
            "suggestions_reviewed": accepted + rejected + modified,
            "suggestions_accepted": accepted,
            "suggestions_modified": modified,
            "suggestions_rejected": rejected,
            # None, not 0: a doctor with no scored cases has no average, and a
            # displayed 0% would read as uniformly unconfident AI.
            "avg_confidence_percent": _rounded(confidence * 100 if confidence else None),
            "avg_processing_time_seconds": None,
        }

    # ── Report analytics ─────────────────────────────────────────────────

    async def _reports(
        self, db: AsyncSession, own_patients, own_cases, start, end
    ) -> dict[str, Any]:
        rows = (await db.execute(
            select(Report.status, func.count(Report.id))
            .where(Report.patient_id.in_(own_patients))
            .where(Report.created_at.between(start, end))
            .group_by(Report.status)
        )).all()
        by_status = {status: count for status, count in rows}

        # Same audit-derived measurement as review turnaround: creation to the
        # moment approval was actually recorded.
        approval_pairs = (await db.execute(
            select(Report.created_at, func.min(AuditLog.created_at))
            .select_from(Report)
            .join(AuditLog, AuditLog.resource_id == cast(Report.id, String))
            .where(AuditLog.event_type == "report.approved")
            .where(AuditLog.case_id.in_(own_cases))
            .where(AuditLog.created_at.between(start, end))
            .group_by(Report.id, Report.created_at)
        )).all()
        seconds = self._average_seconds(approval_pairs)
        avg_approval = _rounded(seconds / 60 if seconds else None)

        return {
            "generated": sum(by_status.values()),
            "approved": sum(by_status.get(s, 0) for s in APPROVED_REPORT_STATES),
            "rejected": by_status.get("rejected", 0),
            "pending": sum(by_status.get(s, 0) for s in PENDING_REPORT_STATES),
            "shared_with_patients": by_status.get("shared", 0),
            "archived": by_status.get("archived", 0),
            "by_status": [{"name": k, "value": v} for k, v in sorted(by_status.items())],
            "avg_approval_minutes": avg_approval,
        }

    # ── Prescription analytics ───────────────────────────────────────────

    async def _prescriptions(
        self, db: AsyncSession, doctor_id, start, end
    ) -> dict[str, Any]:
        issued = await db.scalar(
            select(func.count(Prescription.id))
            .where(Prescription.doctor_id == doctor_id)
            .where(Prescription.created_at.between(start, end))
        )
        with_follow_up = await db.scalar(
            select(func.count(Prescription.id))
            .where(Prescription.doctor_id == doctor_id)
            .where(Prescription.follow_up_date.isnot(None))
            .where(Prescription.created_at.between(start, end))
        )
        top_medications = await self._group_counts(
            db, Medication.name,
            [Medication.prescription_id.in_(
                select(Prescription.id).where(Prescription.doctor_id == doctor_id)
                .scalar_subquery()
            )],
            model=Medication, limit=10,
        )

        # Monthly trend, bucketed in Python. `date_trunc` is PostgreSQL-only and
        # this runs on SQLite too; one query plus a dict beats dialect-specific
        # SQL for a handful of buckets. Months with no activity are omitted
        # rather than zero-filled, so a quiet month is not read as a collapse.
        stamps = (await db.execute(
            select(Prescription.created_at)
            .where(Prescription.doctor_id == doctor_id)
            .where(Prescription.created_at.between(start, end))
        )).scalars().all()

        buckets: dict[str, int] = {}
        for stamp in stamps:
            if stamp:
                buckets[stamp.strftime("%Y-%m")] = buckets.get(stamp.strftime("%Y-%m"), 0) + 1

        return {
            "issued": issued or 0,
            "follow_up_prescriptions": with_follow_up or 0,
            "top_medications": top_medications,
            "medication_categories": [],
            "trend": [
                {
                    "period": period,
                    "month": datetime.strptime(period, "%Y-%m").strftime("%b"),
                    "value": count,
                }
                for period, count in sorted(buckets.items())
            ],
        }

    # ── Appointment analytics ────────────────────────────────────────────

    async def _appointments(
        self, db: AsyncSession, doctor_id, today_str, start, end
    ) -> dict[str, Any]:
        rows = (await db.execute(
            select(Appointment.status, func.count(Appointment.id))
            .where(Appointment.doctor_id == doctor_id)
            .where(Appointment.created_at.between(start, end))
            .group_by(Appointment.status)
        )).all()
        by_status = {status: count for status, count in rows}

        today_count = await db.scalar(
            select(func.count(Appointment.id))
            .where(Appointment.doctor_id == doctor_id)
            .where(Appointment.date == today_str)
        )
        upcoming = await db.scalar(
            select(func.count(Appointment.id))
            .where(Appointment.doctor_id == doctor_id)
            .where(Appointment.date > today_str)
            .where(Appointment.status.in_(("scheduled", "confirmed")))
        )

        return {
            "today": today_count or 0,
            "upcoming": upcoming or 0,
            "completed": by_status.get("completed", 0),
            "cancelled": by_status.get("cancelled", 0),
            "no_show": by_status.get("no_show", 0),
            "by_status": [{"name": k, "value": v} for k, v in sorted(by_status.items())],
        }

    # ── Shared helper ────────────────────────────────────────────────────

    @staticmethod
    async def _group_counts(
        db: AsyncSession,
        column,
        filters: list,
        *,
        model=Case,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """One GROUP BY, descending by count. Empty when nothing matches."""
        stmt = select(column, func.count()).select_from(model)
        for condition in filters:
            stmt = stmt.where(condition)
        stmt = (
            stmt.where(column.isnot(None))
            .group_by(column)
            .order_by(func.count().desc())
            .limit(limit)
        )
        rows = (await db.execute(stmt)).all()
        return [{"name": str(name), "value": count} for name, count in rows if name]


    # ── Export shaping ───────────────────────────────────────────────────

    @staticmethod
    def to_sections(data: dict[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
        """
        Flatten the payload into labelled rows for CSV and PDF.

        Both exports share this, so they cannot drift from each other or from
        the dashboard. `None` renders as "Not measured" rather than as 0 or a
        blank cell, keeping the distinction visible in an exported document.
        """
        def fmt(value: Any) -> str:
            if value is None:
                return "Not measured"
            if isinstance(value, list):
                if not value:
                    return "No data"
                return "; ".join(
                    f"{item.get('name')}: {item.get('value')}"
                    if isinstance(item, dict) else str(item)
                    for item in value[:10]
                )
            return str(value)

        def rows(block: dict[str, Any]) -> list[tuple[str, str]]:
            return [
                (key.replace("_", " ").title(), fmt(value))
                for key, value in block.items()
            ]

        sections: list[tuple[str, list[tuple[str, str]]]] = [
            ("Summary", rows(data.get("summary", {}))),
            ("Clinical Workload", rows(data.get("workload", {}))),
            ("Patient Analytics", rows(data.get("patients", {}))),
            ("AI Analytics", rows(data.get("ai", {}))),
            ("Report Analytics", rows(data.get("reports", {}))),
            ("Prescription Analytics", rows(data.get("prescriptions", {}))),
            ("Appointment Analytics", rows(data.get("appointments", {}))),
        ]
        unavailable = data.get("unavailable_metrics") or []
        if unavailable:
            sections.append((
                "Metrics Not Measured",
                [(u["metric"].replace("_", " ").title(), u["reason"]) for u in unavailable],
            ))
        return sections

    @classmethod
    def to_csv(cls, data: dict[str, Any]) -> str:
        import csv
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["Section", "Metric", "Value"])
        writer.writerow([
            "Range", "Period",
            f"{data['range']['date_from']} to {data['range']['date_to']}",
        ])
        for section, entries in cls.to_sections(data):
            for label, value in entries:
                writer.writerow([section, label, value])
        return buffer.getvalue()


doctor_analytics_service = DoctorAnalyticsService()
