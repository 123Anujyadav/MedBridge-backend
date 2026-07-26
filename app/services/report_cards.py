"""
Clinical summary projection for the doctor's report list.

The AI Reports list used to show a title, a patient name and a one-line summary,
so a doctor had to open every report to learn anything. This service enriches
each row with the patient, case, AI intake and record counts a clinician needs
to triage at a glance.

Two properties matter as much as the content:

**Constant query count.** Everything is bulk-loaded with `IN` and `GROUP BY`
against the whole page of reports — nine queries whether the page holds one
report or five hundred. A per-card lookup would have been far simpler to write
and would have turned one list view into an N+1 storm.

**No model call.** The AI summary shown on a card is the one already stored by
the intake pipeline or the report itself. Generating summaries here would mean
an LLM round-trip per card on every list load.

Nothing is fabricated: absent values come back as `None` or empty so the card
can omit the field rather than print a placeholder.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.appointment import Appointment
from app.models.case import Case
from app.models.intake import IntakeSessionRecord
from app.models.patient import Patient
from app.models.prescription import Prescription
from app.models.report import Report
from app.services.clinical_review import (
    _age_from_dob,
    _clean_list,
    _confidence,
    _iso,
    _text,
)

logger = logging.getLogger(__name__)

# Case states in which further evidence is still expected.
_OPEN_CASE_STATES = ("intake", "ai_processing", "routed", "in_consultation")


class ReportCardService:
    """Builds the enriched report cards for `GET /doctor/reports`."""

    async def list_cards(
        self,
        db: AsyncSession,
        doctor_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
        status: str | None = None,
        urgency: str | None = None,
        flagged: bool | None = None,
    ) -> list[dict[str, Any]]:
        reports = await self._load_reports(
            db, doctor_id, skip=skip, limit=limit,
            status=status, urgency=urgency, flagged=flagged,
        )
        if not reports:
            return []

        patient_ids = {r.patient_id for r in reports}

        cases = await self._load_cases(db, patient_ids)
        cases_by_id = {c.id: c for c in cases}

        patients = await self._load_patients(db, patient_ids)
        intakes = await self._load_intakes(db, set(cases_by_id) | {c.id for c in cases})
        appointment_dates = await self._latest_appointment_dates(db, patient_ids)
        visit_counts = await self._completed_visit_counts(db, patient_ids)
        document_counts = await self._document_counts(db, patient_ids)
        rx_counts, follow_ups = await self._prescription_facts(db, patient_ids)

        cards: list[dict[str, Any]] = []
        for report in reports:
            # Case context is attached ONLY when the report is genuinely linked
            # to a case. Falling back to the patient's most recent case would
            # print that case's chief complaint, urgency and AI summary onto an
            # unrelated document — a lab result would read as though it were
            # about the patient's open consultation.
            case = cases_by_id.get(report.case_id) if report.case_id else None
            patient = patients.get(report.patient_id)
            intake = intakes.get(case.id) if case else None
            snapshot = (
                intake.medical_case_snapshot
                if intake and isinstance(intake.medical_case_snapshot, dict)
                else {}
            )
            cards.append(
                self._build_card(
                    report=report,
                    case=case,
                    patient=patient,
                    intake=intake,
                    snapshot=snapshot,
                    appointment_date=appointment_dates.get(report.patient_id),
                    visits=visit_counts.get(report.patient_id, 0),
                    documents=document_counts.get(report.patient_id, 0),
                    prescriptions=rx_counts.get(report.patient_id, 0),
                    has_follow_up=bool(follow_ups.get(report.patient_id)),
                )
            )
        return cards

    # ── Bulk loaders (one query each) ────────────────────────────────────

    async def _load_reports(
        self,
        db: AsyncSession,
        doctor_id: uuid.UUID,
        *,
        skip: int,
        limit: int,
        status: str | None = None,
        urgency: str | None = None,
        flagged: bool | None = None,
    ) -> list[Report]:
        """
        Reports for patients under this doctor's care, optionally filtered.

        Same ownership rule the endpoint enforced before: scoped through the
        doctor's own cases so one clinician cannot read another's patients.
        Filtering reuses `ReportBulkService._apply_filters` so the list and the
        "select all matching" set can never disagree about what matches.
        """
        from app.services.report_bulk import report_bulk_service

        own_patients = (
            select(Case.patient_id).where(Case.doctor_id == doctor_id).scalar_subquery()
        )
        stmt = select(Report).where(Report.patient_id.in_(own_patients))
        stmt = report_bulk_service._apply_filters(
            stmt, status=status, urgency=urgency, flagged=flagged
        )
        result = await db.execute(
            stmt.order_by(Report.created_at.desc()).offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def _load_cases(
        self, db: AsyncSession, patient_ids: set[uuid.UUID]
    ) -> list[Case]:
        result = await db.execute(select(Case).where(Case.patient_id.in_(patient_ids)))
        return list(result.scalars().all())

    async def _load_patients(
        self, db: AsyncSession, patient_ids: set[uuid.UUID]
    ) -> dict[uuid.UUID, Patient]:
        result = await db.execute(select(Patient).where(Patient.id.in_(patient_ids)))
        return {p.id: p for p in result.scalars().all()}

    async def _load_intakes(
        self, db: AsyncSession, case_ids: set[uuid.UUID]
    ) -> dict[uuid.UUID, IntakeSessionRecord]:
        if not case_ids:
            return {}
        result = await db.execute(
            select(IntakeSessionRecord)
            .where(IntakeSessionRecord.routed_case_id.in_(case_ids))
            .order_by(IntakeSessionRecord.created_at.desc())
        )
        # Newest first, so setdefault keeps the most recent session per case.
        by_case: dict[uuid.UUID, IntakeSessionRecord] = {}
        for record in result.scalars().all():
            if record.routed_case_id:
                by_case.setdefault(record.routed_case_id, record)
        return by_case

    async def _latest_appointment_dates(
        self, db: AsyncSession, patient_ids: set[uuid.UUID]
    ) -> dict[uuid.UUID, str]:
        """Aggregate rather than fetching rows: only the latest date is shown."""
        result = await db.execute(
            select(Appointment.patient_id, func.max(Appointment.date))
            .where(Appointment.patient_id.in_(patient_ids))
            .group_by(Appointment.patient_id)
        )
        return {pid: date for pid, date in result.all() if date}

    async def _completed_visit_counts(
        self, db: AsyncSession, patient_ids: set[uuid.UUID]
    ) -> dict[uuid.UUID, int]:
        result = await db.execute(
            select(Appointment.patient_id, func.count(Appointment.id))
            .where(Appointment.patient_id.in_(patient_ids))
            .where(Appointment.status == "completed")
            .group_by(Appointment.patient_id)
        )
        return {pid: count for pid, count in result.all()}

    async def _document_counts(
        self, db: AsyncSession, patient_ids: set[uuid.UUID]
    ) -> dict[uuid.UUID, int]:
        """Reports with a retrievable file — the honest 'uploaded' definition."""
        result = await db.execute(
            select(Report.patient_id, func.count(Report.id))
            .where(Report.patient_id.in_(patient_ids))
            .where(Report.file_url.isnot(None))
            .group_by(Report.patient_id)
        )
        return {pid: count for pid, count in result.all()}

    async def _prescription_facts(
        self, db: AsyncSession, patient_ids: set[uuid.UUID]
    ) -> tuple[dict[uuid.UUID, int], dict[uuid.UUID, Optional[str]]]:
        """Count and latest follow-up date in one pass, for the count and the badge."""
        result = await db.execute(
            select(
                Prescription.patient_id,
                func.count(Prescription.id),
                func.max(Prescription.follow_up_date),
            )
            .where(Prescription.patient_id.in_(patient_ids))
            .group_by(Prescription.patient_id)
        )
        counts: dict[uuid.UUID, int] = {}
        follow_ups: dict[uuid.UUID, Optional[str]] = {}
        for pid, count, follow_up in result.all():
            counts[pid] = count
            follow_ups[pid] = follow_up
        return counts, follow_ups

    # ── Card assembly ────────────────────────────────────────────────────

    def _build_card(
        self,
        *,
        report: Report,
        case: Optional[Case],
        patient: Optional[Patient],
        intake: Optional[IntakeSessionRecord],
        snapshot: dict[str, Any],
        appointment_date: Optional[str],
        visits: int,
        documents: int,
        prescriptions: int,
        has_follow_up: bool,
    ) -> dict[str, Any]:
        confidence = _confidence(
            snapshot.get("overall_confidence", {}).get("score")
            if isinstance(snapshot.get("overall_confidence"), dict)
            else snapshot.get("overall_confidence")
        )
        if confidence is None and intake is not None:
            confidence = _confidence(intake.overall_confidence)
        if confidence is None and case is not None:
            confidence = _confidence(case.ai_confidence_score)
        if confidence is None:
            confidence = _confidence(report.ai_confidence_score)

        symptoms = _clean_list(snapshot.get("symptoms"), limit=8) or _clean_list(
            case.ai_extracted_symptoms if case else None, limit=8
        )

        # The stored summary, in priority of specificity. Never regenerated.
        ai_summary = _text(
            snapshot.get("summary_for_doctor"),
            _text(report.summary, _text(case.symptom_summary if case else "")),
        )

        red_flags = _clean_list(snapshot.get("red_flags")) or _clean_list(
            intake.red_flags if intake else None
        )

        card: dict[str, Any] = {
            # Everything ReportResponse already carried.
            "id": report.id,
            "patient_id": report.patient_id,
            "case_id": report.case_id,
            "patient_name": report.patient_name,
            "type": report.type,
            "title": report.title,
            "summary": report.summary,
            "content": report.content,
            "doctor_name": report.doctor_name,
            "hospital_name": report.hospital_name,
            "date": report.date,
            "status": report.status,
            "file_url": report.file_url,
            "file_size": report.file_size,
            "ai_generated": report.ai_generated,
            "ai_confidence_score": report.ai_confidence_score,
            "tags": report.tags or [],
            "vitals": report.vitals,
            "flagged_for_follow_up": bool(report.flagged_for_follow_up),
            # Patient. Age and gender are patient-level facts, so they resolve
            # from the profile and remain correct on reports with no case.
            "patient_age": _age_from_dob(patient.date_of_birth if patient else None)
            or (case.patient_age if case and case.patient_age else None),
            "patient_gender": (
                (patient.gender if patient else None)
                or (case.patient_gender if case else None)
            ),
            "patient_short_id": str(report.patient_id).split("-")[0],
            "appointment_date": appointment_date,
            "assigned_doctor": (case.doctor_name if case else None) or report.doctor_name,
            # Case.
            "case_status": case.status if case else None,
            "chief_complaint": _text(
                snapshot.get("chief_complaint"),
                _text(case.symptom_summary if case else ""),
            )
            or None,
            "extracted_symptoms": symptoms,
            "specialty": (case.specialty if case else None),
            "urgency_level": (case.urgency_level if case else None),
            "ai_confidence": confidence,
            "language_detected": (intake.language if intake else None),
            "case_created_at": _iso(case.created_at) if case else None,
            "case_updated_at": _iso(case.updated_at) if case else None,
            # Medical.
            "allergies": _clean_list(patient.allergies if patient else None, limit=8),
            "chronic_conditions": _clean_list(
                patient.chronic_conditions if patient else None, limit=8
            ),
            "current_medications": _clean_list(
                patient.medications if patient else None, limit=8
            ),
            "uploaded_reports_count": documents,
            "previous_visits_count": visits,
            "previous_prescriptions_count": prescriptions,
            # AI.
            "ai_summary": ai_summary,
        }
        card["indicators"] = self._indicators(
            report=report,
            case=case,
            confidence=confidence,
            red_flags=red_flags,
            documents=documents,
            has_follow_up=has_follow_up,
        )
        return card

    @staticmethod
    def _indicators(
        *,
        report: Report,
        case: Optional[Case],
        confidence: Optional[dict[str, Any]],
        red_flags: list[str],
        documents: int,
        has_follow_up: bool,
    ) -> list[dict[str, str]]:
        """
        Badges, every one traceable to a stored value.

        Confidence is omitted rather than defaulted when no score exists, so a
        missing measurement never reads as a low one.
        """
        out: list[dict[str, str]] = []

        if confidence:
            tone = {"High": "success", "Medium": "warning", "Low": "error"}[
                confidence["level"]
            ]
            out.append({"label": f"{confidence['level']} Confidence", "tone": tone})

        if red_flags:
            out.append({"label": "Emergency", "tone": "error"})

        if case and case.urgency_level == "critical":
            out.append({"label": "Critical", "tone": "error"})

        status_badges = {
            "pending": ("Needs Review", "warning"),
            "pending_review": ("Needs Review", "warning"),
            "needs_revision": ("More Information Requested", "warning"),
            "approved": ("Approved", "success"),
            "ready": ("Approved", "success"),
            "rejected": ("Rejected", "error"),
        }
        if report.status in status_badges:
            label, tone = status_badges[report.status]
            out.append({"label": label, "tone": tone})

        if documents == 0 and case and case.status in _OPEN_CASE_STATES:
            out.append({"label": "Awaiting Reports", "tone": "neutral"})

        # Either a prescription carries a follow-up date, or a clinician flagged
        # the report explicitly. Both mean the same thing to whoever reads it.
        if has_follow_up or report.flagged_for_follow_up:
            out.append({"label": "Follow-up Required", "tone": "info"})

        return out


report_card_service = ReportCardService()
