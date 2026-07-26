"""
Clinical Review Workspace: a read-only projection over the existing record.

"View Full AI Analysis" previously rendered a report's title, patient name and
body text. That is enough to read a document but not enough to make a clinical
decision, which needs the patient's profile, the AI intake case, the evidence
trail and the case history side by side.

This service composes those from tables that already exist. It writes nothing
except through `save_consultation` and `approve_ai_summary`, which update
columns already on `cases` — there is no new storage and no duplicated business
logic. Report authoring stays in `AIClinicalReportService`, prescriptions stay
in `ConsultationService`.

Fabrication rules, consistent with the rest of the platform:

* A value the record does not hold is reported as absent, and the reason lands
  in `data_gaps` or `AISuggestions.notes`.
* BMI is computed by the same helper the vitals dashboard uses, and only when
  both height and weight exist.
* Confidence is surfaced only when the pipeline recorded a score.
* AI advisory output is grounded in the assembled record and is always labelled
  as a suggestion. If Groq is unavailable the record-derived items are still
  returned and `source` reports `records`.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ai_provider import get_groq_client
from app.core.exceptions import AuthorizationException, EntityNotFoundException
from app.models.appointment import Appointment
from app.models.case import Case, Symptom
from app.models.doctor import Doctor
from app.models.intake import IntakeSessionRecord
from app.models.patient import Patient
from app.models.prescription import Medication, Prescription
from app.models.report import Report
from app.models.user import User
from app.services.vitals import VitalsService

logger = logging.getLogger(__name__)

UNKNOWN = "Unknown"

# `reports.type` -> workspace evidence bucket.
_EVIDENCE_CATEGORIES: dict[str, str] = {
    "lab_result": "lab",
    "lab_report": "lab",
    "blood_test": "lab",
    "imaging": "imaging",
    "scan": "imaging",
    "x_ray": "imaging",
    "mri": "imaging",
    "ct_scan": "imaging",
    "ultrasound": "imaging",
    "ai_report": "ai_analysis",
    "ai_analysis": "ai_analysis",
    "medical_report": "clinical",
    "discharge_summary": "clinical",
    "consultation": "clinical",
    "vital_signs": "clinical",
}

_SUGGESTION_SYSTEM_PROMPT = """You are a clinical decision-support assistant. \
A licensed physician is reviewing a case and wants advisory prompts to evaluate.

ABSOLUTE RULES:
- Ground every item in the CASE RECORD provided. Never introduce a symptom, \
allergy, medication, lab value, measurement or history item that is not there.
- These are SUGGESTIONS for a physician to consider, never diagnoses. Phrase \
them as considerations, not conclusions.
- Report drug interactions ONLY between medications actually listed in the \
record, or between a listed medication and a listed allergy. If the record \
lists fewer than two medications and no allergies, return an empty list.
- Never invent numbers, dosages, reference ranges or trial data.
- `completed_investigations` lists tests already performed for this patient. Do \
NOT suggest repeating one of them unless the record gives a reason to; prefer \
investigations that have not been done.
- If a category has no grounded content, return an empty list for it. An empty \
list is correct and expected; padding it is a serious error.

Reply with ONLY a JSON object using exactly these keys:
{
  "differential_diagnoses": ["conditions worth evaluating, most plausible first"],
  "drug_interaction_warnings": ["interactions between listed medications/allergies"],
  "red_flag_symptoms": ["recorded findings that warrant urgent attention"],
  "suggested_lab_tests": ["laboratory tests worth ordering"],
  "suggested_imaging": ["imaging studies worth ordering"],
  "clinical_guideline_summary": "2-4 sentences of standard management context",
  "possible_contraindications": ["cautions given this patient's record"],
  "relevant_medical_history": ["items from this patient's history that bear on the case"],
  "medication_alerts": ["alerts about the patient's current medications"]
}"""


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _clean_list(raw: Any, limit: int = 25) -> list[str]:
    """Coerce a stored or model value into non-empty, non-placeholder strings."""
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            item = item.get("name") or item.get("value") or ""
        text = str(item).strip()
        if text and text.casefold() not in {
            "none", "unknown", "n/a", "none reported", "not recorded",
        }:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _text(raw: Any, fallback: str = "") -> str:
    value = str(raw or "").strip()
    return value if value and value.casefold() != UNKNOWN.casefold() else fallback


def _age_from_dob(date_of_birth: str | None) -> Optional[int]:
    """Age in whole years, or None when the stored date cannot be parsed."""
    if not date_of_birth:
        return None
    try:
        year = int(str(date_of_birth).split("-")[0])
    except (ValueError, IndexError):
        return None
    age = datetime.now(timezone.utc).year - year
    return age if 0 <= age <= 130 else None


def _bmi_category(bmi: float | None) -> Optional[str]:
    """Standard WHO bands. Descriptive only — it drives a label, not a decision."""
    if bmi is None:
        return None
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25.0:
        return "Normal"
    if bmi < 30.0:
        return "Overweight"
    return "Obese"


def _confidence(score: Any) -> Optional[dict[str, Any]]:
    """
    Build a confidence reading, or None.

    A zero or absent score means the pipeline never recorded one; showing "0%"
    would misrepresent that as a measured low-confidence result.
    """
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if value <= 0.0:
        return None
    value = min(value, 1.0)
    level = "High" if value >= 0.8 else "Medium" if value >= 0.5 else "Low"
    return {
        "score": round(value, 4),
        "percentage": int(round(value * 100)),
        "level": level,
    }


class ClinicalReviewService:
    """Assembles the four-section review workspace for one report."""

    # ── Read ─────────────────────────────────────────────────────────────

    async def build_review(
        self, db: AsyncSession, doctor_id: uuid.UUID, report_id: uuid.UUID
    ) -> dict[str, Any]:
        report = await self._load_owned_report(db, report_id, doctor_id)

        # Case context is attached ONLY when the report is genuinely linked to a
        # case. This previously fell back to the patient's most recent case,
        # which meant an ingested lab result opened showing an unrelated
        # consultation's chief complaint, AI summary, urgency, red flags and
        # timeline. A workspace that quietly answers about the wrong case is
        # more dangerous than one that admits it has no case context.
        case = await db.get(Case, report.case_id) if report.case_id else None

        patient = await db.get(Patient, report.patient_id)
        if patient is None:
            raise EntityNotFoundException("Patient", str(report.patient_id))

        doctor = await db.get(Doctor, case.doctor_id) if case and case.doctor_id else None
        intake = await self._load_intake(db, case.id) if case else None
        symptoms = await self._load_symptoms(db, case.id) if case else []
        all_reports = await self._load_reports(db, report.patient_id)
        # Patient-scoped: the evidence panel is explicitly prescribing history.
        prescriptions = await self._load_prescriptions(db, report.patient_id)
        # Case-scoped: the timeline describes this case only, so a prescription
        # written for a different consultation must not appear on it.
        case_prescriptions = (
            await self._load_prescriptions(db, report.patient_id, case_id=case.id)
            if case
            else []
        )
        appointment = await self._load_appointment(db, report.patient_id, case)
        visit_count = await self._count_visits(db, report.patient_id)

        snapshot: dict[str, Any] = {}
        if intake is not None and isinstance(intake.medical_case_snapshot, dict):
            snapshot = intake.medical_case_snapshot

        overview = self._build_overview(
            patient=patient,
            case=case,
            doctor=doctor,
            appointment=appointment,
            visit_count=visit_count,
        )
        analysis = self._build_analysis(case, intake, snapshot, symptoms)
        evidence = self._build_evidence(
            report=report,
            case=case,
            all_reports=all_reports,
            prescriptions=prescriptions,
        )
        suggestions = await self._build_suggestions(
            overview=overview, analysis=analysis, evidence=evidence
        )
        timeline = self._build_timeline(
            case=case, intake=intake, report=report,
            prescriptions=case_prescriptions, appointment=appointment,
        )
        comparison = self._build_comparison(
            case=case, intake=intake, snapshot=snapshot, report=report
        )

        return {
            "report_id": report.id,
            "report_title": report.title,
            "report_status": report.status,
            "report_content": report.content or "",
            "report_file_url": report.file_url,
            "case_id": case.id if case else None,
            "case_status": case.status if case else None,
            "patient_overview": overview,
            "ai_analysis": analysis,
            "medical_evidence": evidence,
            "ai_suggestions": suggestions,
            "timeline": timeline,
            "comparison": comparison,
            "data_gaps": self._data_gaps(overview, analysis, evidence, case),
        }

    # ── Write (existing columns only) ────────────────────────────────────

    async def save_consultation(
        self,
        db: AsyncSession,
        doctor_id: uuid.UUID,
        case_id: uuid.UUID,
        *,
        clinical_notes: str,
        diagnosis: Optional[str],
        complete_case: bool,
    ) -> dict[str, Any]:
        """
        Persist the doctor's working notes on the case.

        `diagnosis` is folded into the notes text rather than dropped: the `Case`
        model has no diagnosis column, and assigning one silently discards it
        (PROJECT_ANALYSIS.md §11.1). The authoritative diagnosis is the one
        recorded on the issued report and prescription.
        """
        from app.services.case_timeline import case_timeline_service

        case = await self._load_owned_case(db, case_id, doctor_id)
        actor = await db.get(User, doctor_id)

        # Captured before mutation so the audit entry can state what actually
        # moved rather than only where it ended up.
        previous_notes = case.notes or ""
        previous_status = case.status

        notes = clinical_notes.strip()
        if diagnosis and diagnosis.strip():
            notes = f"Working diagnosis: {diagnosis.strip()}\n\n{notes}".strip()

        case.notes = notes
        if complete_case:
            case.status = "completed"
            case.completed_at = datetime.now(timezone.utc)
        elif case.status in ("intake", "ai_processing", "routed"):
            case.status = "in_consultation"
        await db.flush()

        common = {
            "actor": actor,
            "actor_type": "doctor",
            "case_id": case.id,
            "patient_id": case.patient_id,
        }
        if notes != previous_notes:
            await case_timeline_service.safe_record(
                db, event_type="case.notes_added",
                description="Clinical notes updated on the case.",
                field="notes",
                previous=previous_notes[:500] or None,
                new=notes[:500], **common,
            )
        if diagnosis and diagnosis.strip():
            await case_timeline_service.safe_record(
                db, event_type="case.diagnosis_updated",
                description=f"Working diagnosis recorded: {diagnosis.strip()}",
                field="diagnosis", new=diagnosis.strip(), **common,
            )
        if case.status != previous_status:
            await case_timeline_service.safe_record(
                db,
                event_type="case.closed" if complete_case else "case.status_changed",
                description=f"Case status moved to {case.status}.",
                field="status", previous=previous_status, new=case.status, **common,
            )
        # `updated_at` is server-generated and expires on flush. The timeline
        # reads it, and a lazy refresh outside the async context raises
        # MissingGreenlet — so reload the row here, inside it.
        await db.refresh(case)

        intake = await self._load_intake(db, case.id)
        # Everything feeding this case's timeline is scoped to this case.
        # `reports[0]` was the patient's newest report regardless of case, so a
        # report issued for a different consultation filled in this case's
        # "Clinical Report Issued" milestone.
        reports = await self._load_reports(db, case.patient_id, case_id=case.id)
        prescriptions = await self._load_prescriptions(db, case.patient_id, case_id=case.id)
        appointment = await self._load_appointment(db, case.patient_id, case)
        latest_report = reports[0] if reports else None

        logger.info(
            "[CLINICAL_REVIEW_SAVED] case=%s doctor=%s status=%s completed=%s",
            case.id, doctor_id, case.status, complete_case,
        )
        return {
            "case_id": case.id,
            "status": case.status,
            "notes": case.notes,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "timeline": self._build_timeline(
                case=case, intake=intake, report=latest_report,
                prescriptions=prescriptions, appointment=appointment,
            ),
        }

    async def approve_ai_summary(
        self, db: AsyncSession, doctor_id: uuid.UUID, case_id: uuid.UUID, summary: str
    ) -> dict[str, Any]:
        """
        Record the doctor's sign-off on the AI summary.

        Stored on `cases.notes` with an explicit attribution line so the record
        shows a clinician accepted it, rather than the acceptance being implicit.
        """
        from app.services.case_timeline import case_timeline_service

        case = await self._load_owned_case(db, case_id, doctor_id)
        actor = await db.get(User, doctor_id)
        approved_at = datetime.now(timezone.utc)

        stamp = f"[AI summary reviewed and approved {approved_at.date().isoformat()}]"
        body = summary.strip()
        existing = (case.notes or "").strip()
        case.notes = f"{stamp}\n{body}" if not existing else f"{stamp}\n{body}\n\n{existing}"
        if case.status in ("intake", "ai_processing", "routed"):
            case.status = "in_consultation"
        await db.flush()

        await case_timeline_service.safe_record(
            db, event_type="ai.summary_approved",
            description="Doctor reviewed and approved the AI clinical summary.",
            actor=actor, actor_type="doctor",
            case_id=case.id, patient_id=case.patient_id,
            field="ai_summary_approved", previous="false", new="true",
        )

        logger.info(
            "[CLINICAL_REVIEW_AI_APPROVED] case=%s doctor=%s", case.id, doctor_id
        )
        return {
            "case_id": case.id,
            "status": case.status,
            "approved_summary": body,
            "approved_at": approved_at.isoformat(),
        }

    # ── Loading ──────────────────────────────────────────────────────────

    async def _load_owned_report(
        self, db: AsyncSession, report_id: uuid.UUID, doctor_id: uuid.UUID
    ) -> Report:
        """Mirrors the ownership rule used by the report mutation routes."""
        report = await db.get(Report, report_id)
        if report is None or report.deleted_at is not None:
            raise EntityNotFoundException("Report", str(report_id))

        linked = await db.scalar(
            select(Case.id)
            .where(Case.doctor_id == doctor_id)
            .where(Case.patient_id == report.patient_id)
            .limit(1)
        )
        if linked is None:
            raise AuthorizationException(
                "You are not authorised to review this patient's report."
            )
        return report

    async def _load_owned_case(
        self, db: AsyncSession, case_id: uuid.UUID, doctor_id: uuid.UUID
    ) -> Case:
        case = await db.get(Case, case_id)
        if case is None or case.deleted_at is not None:
            raise EntityNotFoundException("Case", str(case_id))
        if case.doctor_id != doctor_id:
            raise AuthorizationException("You are not authorised to edit this case.")
        return case

    async def _load_intake(
        self, db: AsyncSession, case_id: uuid.UUID
    ) -> Optional[IntakeSessionRecord]:
        result = await db.execute(
            select(IntakeSessionRecord)
            .where(IntakeSessionRecord.routed_case_id == case_id)
            .order_by(IntakeSessionRecord.created_at.desc())
            .limit(1)
        )
        return result.scalars().first()

    async def _load_symptoms(self, db: AsyncSession, case_id: uuid.UUID) -> list[Symptom]:
        result = await db.execute(select(Symptom).where(Symptom.case_id == case_id))
        return list(result.scalars().all())

    async def _load_reports(
        self,
        db: AsyncSession,
        patient_id: uuid.UUID,
        *,
        case_id: uuid.UUID | None = None,
        limit: int = 50,
    ) -> list[Report]:
        """
        Reports for the patient, or for one specific case.

        As with prescriptions, the evidence panel wants the patient's whole
        document history while the case timeline wants only this case's report.
        """
        stmt = select(Report).where(Report.patient_id == patient_id)
        if case_id is not None:
            stmt = stmt.where(Report.case_id == case_id)
        result = await db.execute(
            stmt.order_by(Report.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())

    async def _load_prescriptions(
        self,
        db: AsyncSession,
        patient_id: uuid.UUID,
        *,
        case_id: uuid.UUID | None = None,
        limit: int = 10,
    ) -> list[tuple[Prescription, list[Medication]]]:
        """
        Prescriptions for the patient, or for one specific case.

        Both scopes are legitimate but they are not interchangeable: the
        evidence panel shows the patient's prescribing history, while the case
        timeline must show only what was prescribed *for this case*. Passing
        `case_id` is what keeps the second from silently becoming the first.
        """
        stmt = select(Prescription).where(Prescription.patient_id == patient_id)
        if case_id is not None:
            stmt = stmt.where(Prescription.case_id == case_id)
        result = await db.execute(
            stmt.order_by(Prescription.created_at.desc()).limit(limit)
        )
        prescriptions = list(result.scalars().all())
        if not prescriptions:
            return []

        meds = await db.execute(
            select(Medication).where(
                Medication.prescription_id.in_([p.id for p in prescriptions])
            )
        )
        by_rx: dict[uuid.UUID, list[Medication]] = {}
        for med in meds.scalars().all():
            by_rx.setdefault(med.prescription_id, []).append(med)
        return [(p, by_rx.get(p.id, [])) for p in prescriptions]

    async def _load_appointment(
        self, db: AsyncSession, patient_id: uuid.UUID, case: Optional[Case]
    ) -> Optional[Appointment]:
        stmt = select(Appointment).where(Appointment.patient_id == patient_id)
        if case is not None:
            stmt = stmt.where(Appointment.case_id == case.id)
        result = await db.execute(
            stmt.order_by(Appointment.date.desc(), Appointment.time.desc()).limit(1)
        )
        return result.scalars().first()

    async def _count_visits(self, db: AsyncSession, patient_id: uuid.UUID) -> int:
        """Completed appointments — the honest definition of a prior visit."""
        result = await db.execute(
            select(Appointment.id)
            .where(Appointment.patient_id == patient_id)
            .where(Appointment.status == "completed")
        )
        return len(list(result.scalars().all()))

    # ── Section builders ─────────────────────────────────────────────────

    def _build_overview(
        self,
        *,
        patient: Patient,
        case: Optional[Case],
        doctor: Optional[Doctor],
        appointment: Optional[Appointment],
        visit_count: int,
    ) -> dict[str, Any]:
        # Same helper the vitals dashboard uses, so BMI cannot diverge.
        bmi = VitalsService._bmi(patient.weight, patient.height)

        return {
            "patient_id": patient.id,
            "patient_name": f"{patient.first_name} {patient.last_name}".strip(),
            "age": _age_from_dob(patient.date_of_birth)
            or (case.patient_age if case and case.patient_age else None),
            "gender": patient.gender or (case.patient_gender if case else None),
            "blood_group": patient.blood_type or None,
            "height_cm": patient.height,
            "weight_kg": patient.weight,
            "bmi": bmi,
            "bmi_category": _bmi_category(bmi),
            "allergies": _clean_list(patient.allergies),
            "chronic_conditions": _clean_list(patient.chronic_conditions),
            "current_medications": _clean_list(patient.medications),
            "previous_visits": visit_count,
            "appointment_date": (
                f"{appointment.date} {appointment.time}".strip()
                if appointment
                else None
            ),
            "appointment_status": appointment.status if appointment else None,
            "assigned_doctor": (
                f"Dr. {doctor.first_name} {doctor.last_name}".strip()
                if doctor
                else (case.doctor_name if case else None)
            ),
            "assigned_doctor_specialty": (
                doctor.specialty if doctor else (case.specialty if case else None)
            ),
        }

    def _build_analysis(
        self,
        case: Optional[Case],
        intake: Optional[IntakeSessionRecord],
        snapshot: dict[str, Any],
        symptoms: list[Symptom],
    ) -> dict[str, Any]:
        # Confidence: prefer the intake session's recorded score, then the case's.
        confidence = _confidence(
            snapshot.get("overall_confidence", {}).get("score")
            if isinstance(snapshot.get("overall_confidence"), dict)
            else snapshot.get("overall_confidence")
        )
        if confidence is None and intake is not None:
            confidence = _confidence(intake.overall_confidence)
        if confidence is None and case is not None:
            confidence = _confidence(case.ai_confidence_score)

        extracted = _clean_list(snapshot.get("symptoms")) or _clean_list(
            case.ai_extracted_symptoms if case else None
        )
        if not extracted:
            extracted = _clean_list([s.name for s in symptoms])

        recommended = _text(
            snapshot.get("recommended_specialty"),
            _text(case.ai_specialty_recommendation if case else None),
        )
        reason = ""
        if recommended and case is not None and case.specialty:
            reason = (
                f"Intake matched the presentation to {recommended}; "
                f"the case is assigned to {case.specialty}."
                if recommended != case.specialty
                else f"Intake matched the presentation to {recommended}."
            )

        return {
            "chief_complaint": _text(
                snapshot.get("chief_complaint"),
                _text(case.symptom_summary if case else None),
            ),
            "ai_summary": _text(
                snapshot.get("summary_for_doctor"),
                _text(case.symptom_summary if case else None),
            ),
            "extracted_symptoms": extracted,
            "symptom_timeline": [
                {
                    "name": s.name,
                    "severity": s.severity or None,
                    "duration": _text(s.duration) or None,
                    "body_part": s.body_part or None,
                }
                for s in symptoms
            ],
            "possible_causes": _clean_list(snapshot.get("differential_considerations")),
            "severity": _text(snapshot.get("severity")) or None,
            "onset": _text(snapshot.get("onset")) or None,
            "duration": _text(snapshot.get("duration")) or None,
            "urgency_level": (case.urgency_level if case else None),
            "confidence": confidence,
            "recommended_specialist": recommended or None,
            "recommendation_reason": reason or None,
            "emergency_indicators": _clean_list(snapshot.get("red_flags"))
            or _clean_list(intake.red_flags if intake else None),
            "language_detected": (intake.language if intake else None),
            "conversation_summary": (intake.transcript if intake else "") or "",
            "missing_information": _clean_list(snapshot.get("missing_information")),
            "has_ai_intake": intake is not None,
        }

    def _build_evidence(
        self,
        *,
        report: Report,
        case: Optional[Case],
        all_reports: list[Report],
        prescriptions: list[tuple[Prescription, list[Medication]]],
    ) -> dict[str, Any]:
        def to_doc(r: Report) -> dict[str, Any]:
            category = _EVIDENCE_CATEGORIES.get(r.type, "other")
            return {
                "report_id": r.id,
                "title": r.title,
                "type": r.type,
                "category": category,
                "date": r.date,
                "summary": (r.summary or "")[:500],
                "status": r.status or "",
                "doctor_name": r.doctor_name,
                "file_url": r.file_url,
                # Only a stored uploads path is servable by the download route.
                "downloadable": bool(r.file_url and r.file_url.startswith("/uploads/")),
                "ai_generated": bool(r.ai_generated),
                "ai_confidence_score": r.ai_confidence_score,
            }

        docs = [to_doc(r) for r in all_reports]
        return {
            # Anything with a retrievable file is a document the doctor uploaded
            # or the platform produced.
            "uploaded_reports": [d for d in docs if d["downloadable"]],
            "lab_reports": [d for d in docs if d["category"] == "lab"],
            "imaging_and_scans": [d for d in docs if d["category"] == "imaging"],
            "ai_report_analysis": [d for d in docs if d["category"] == "ai_analysis"],
            "historical_reports": [d for d in docs if d["report_id"] != report.id],
            "case_attachments": [
                {
                    "name": str(a.get("name", "")) if isinstance(a, dict) else str(a),
                    "type": str(a.get("type", "")) if isinstance(a, dict) else "",
                    "url": a.get("url") if isinstance(a, dict) else None,
                }
                for a in (case.attachments or [] if case else [])
            ],
            "doctor_notes": (case.notes or "") if case else "",
            "previous_prescriptions": [
                {
                    "prescription_id": rx.id,
                    "diagnosis": rx.diagnosis,
                    "notes": rx.notes or "",
                    "status": rx.status,
                    "doctor_name": rx.doctor_name,
                    "follow_up_date": rx.follow_up_date,
                    "created_at": _iso(rx.created_at),
                    "medications": [
                        {
                            "name": m.name,
                            "generic_name": m.generic_name,
                            "dosage": m.dosage or "",
                            "frequency": m.frequency or "",
                            "duration": m.duration or "",
                            "special_instructions": m.special_instructions or "",
                            "status": m.status or "",
                            "side_effects": _clean_list(m.side_effects),
                            "interactions": _clean_list(m.interactions),
                        }
                        for m in meds
                    ],
                }
                for rx, meds in prescriptions
            ],
        }

    def _build_timeline(
        self,
        *,
        case: Optional[Case],
        intake: Optional[IntakeSessionRecord],
        report: Optional[Report],
        prescriptions: list[tuple[Prescription, list[Medication]]],
        appointment: Optional[Appointment],
    ) -> list[dict[str, Any]]:
        """
        The case history, built from real timestamps.

        A stage is `completed` only when a row or timestamp backs it. Stages that
        have not happened stay `pending` with no timestamp — a timeline that
        shows every step as done regardless of reality is worse than none.
        """
        first_rx = prescriptions[0][0] if prescriptions else None
        follow_up = (
            first_rx.follow_up_date
            if first_rx and first_rx.follow_up_date
            else (appointment.date if appointment else None)
        )
        notes = (case.notes or "").strip() if case else ""

        def event(key, label, ts, detail="") -> dict[str, Any]:
            return {
                "key": key,
                "label": label,
                "status": "completed" if ts else "pending",
                "timestamp": ts,
                "detail": detail,
            }

        events = [
            event(
                "case_created", "Patient Created Case",
                _iso(case.created_at) if case else None,
                case.symptom_summary[:120] if case and case.symptom_summary else "",
            ),
            event(
                "ai_intake", "AI Medical Intake",
                _iso(intake.created_at) if intake else None,
                f"{intake.followup_rounds} follow-up round(s), language {intake.language}"
                if intake else "No AI intake session on file",
            ),
            event(
                "ai_analysis", "AI Analysis Completed",
                _iso(intake.updated_at) if intake and intake.medical_case_snapshot else None,
                f"Structured case generated, {len(intake.red_flags or [])} red flag(s)"
                if intake and intake.medical_case_snapshot else "",
            ),
            event(
                "doctor_opened", "Doctor Opened Case",
                _iso(case.assigned_at) if case and case.assigned_at else None,
                case.doctor_name if case and case.doctor_name else "",
            ),
            event(
                "doctor_notes", "Doctor Notes",
                _iso(case.updated_at) if case and notes else None,
                f"{len(notes)} characters recorded" if notes else "",
            ),
            event(
                "prescription", "Prescription",
                _iso(first_rx.created_at) if first_rx else None,
                first_rx.diagnosis if first_rx else "",
            ),
            event("follow_up", "Follow-up", follow_up or None, ""),
            event(
                "report_issued", "Clinical Report Issued",
                _iso(report.created_at) if report else None,
                report.title if report else "",
            ),
            event(
                "completed", "Completed",
                _iso(case.completed_at) if case and case.completed_at else None,
                case.status if case else "",
            ),
        ]
        return events

    def _build_comparison(
        self,
        *,
        case: Optional[Case],
        intake: Optional[IntakeSessionRecord],
        snapshot: dict[str, Any],
        report: Report,
    ) -> dict[str, Any]:
        patient_input = (intake.transcript if intake else "") or (
            case.symptom_summary if case else ""
        )
        patient_source = (
            "AI intake transcript (patient's own words)"
            if intake and intake.transcript
            else "Case symptom summary"
            if case
            else ""
        )

        ai_parts = []
        if snapshot.get("summary_for_doctor"):
            ai_parts.append(str(snapshot["summary_for_doctor"]))
        causes = _clean_list(snapshot.get("differential_considerations"))
        if causes:
            ai_parts.append("Considerations for clinician: " + ", ".join(causes))
        ai_interpretation = "\n\n".join(ai_parts) or _text(
            case.symptom_summary if case else ""
        )

        doctor_notes = (case.notes or "").strip() if case else ""
        doctor_decision = doctor_notes or (report.content or "")
        decided = bool(doctor_notes) or report.status in ("ready", "approved", "reviewed")

        return {
            "patient_input": patient_input,
            "patient_input_source": patient_source,
            "ai_interpretation": ai_interpretation,
            "ai_interpretation_source": (
                "AI intake structured case" if snapshot else "Case record"
            ),
            "doctor_decision": doctor_decision,
            "doctor_decision_source": (
                "Doctor case notes" if doctor_notes else f"Issued report: {report.title}"
            ),
            "doctor_has_decided": decided,
        }

    # ── AI advisory ──────────────────────────────────────────────────────

    async def _build_suggestions(
        self,
        *,
        overview: dict[str, Any],
        analysis: dict[str, Any],
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Advisory prompts for the clinician.

        Record-derived items are always present. The model adds to them when it
        is reachable; when it is not, the record-derived set is returned alone
        rather than an empty panel.
        """
        notes: list[str] = []

        # Interactions already recorded against dispensed medications are fact,
        # not suggestion, so they are seeded before any model call.
        recorded_interactions: list[str] = []
        for rx in evidence["previous_prescriptions"]:
            for med in rx["medications"]:
                for interaction in med["interactions"]:
                    recorded_interactions.append(f"{med['name']}: {interaction}")

        base = {
            "differential_diagnoses": list(analysis["possible_causes"]),
            "drug_interaction_warnings": recorded_interactions,
            "red_flag_symptoms": list(analysis["emergency_indicators"]),
            "suggested_lab_tests": [],
            "suggested_imaging": [],
            "clinical_guideline_summary": "",
            "possible_contraindications": [],
            "relevant_medical_history": (
                overview["chronic_conditions"]
                + [f"Allergy: {a}" for a in overview["allergies"]]
            ),
            "medication_alerts": [],
            "source": "records",
            "generated": False,
            "notes": notes,
        }

        med_count = len(overview["current_medications"]) + sum(
            len(rx["medications"]) for rx in evidence["previous_prescriptions"]
        )
        if med_count < 2 and not overview["allergies"]:
            notes.append(
                "No drug interaction review is possible: fewer than two "
                "medications and no allergies are on file."
            )

        client = get_groq_client()
        if not client.is_configured:
            notes.append(
                "AI suggestions unavailable — showing record-derived items only."
            )
            return base

        payload = {
            "age": overview["age"],
            "gender": overview["gender"],
            "blood_group": overview["blood_group"],
            "bmi": overview["bmi"],
            "allergies": overview["allergies"],
            "chronic_conditions": overview["chronic_conditions"],
            "current_medications": overview["current_medications"],
            "chief_complaint": analysis["chief_complaint"],
            "ai_summary": analysis["ai_summary"],
            "symptoms": analysis["extracted_symptoms"],
            "severity": analysis["severity"],
            "duration": analysis["duration"],
            "onset": analysis["onset"],
            "urgency": analysis["urgency_level"],
            "red_flags": analysis["emergency_indicators"],
            "intake_considerations": analysis["possible_causes"],
            "missing_information": analysis["missing_information"],
            "prescribed_medications": [
                {
                    "name": m["name"],
                    "dosage": m["dosage"],
                    "frequency": m["frequency"],
                    "recorded_interactions": m["interactions"],
                }
                for rx in evidence["previous_prescriptions"]
                for m in rx["medications"]
            ],
            "prior_diagnoses": [
                rx["diagnosis"] for rx in evidence["previous_prescriptions"]
            ],
            # Without this the model re-suggests investigations the patient has
            # already had, which wastes a clinician's attention and, acted on,
            # the patient's time and money.
            "completed_investigations": [
                {"title": d["title"], "type": d["type"], "date": d["date"],
                 "summary": d["summary"]}
                for d in evidence["lab_reports"] + evidence["imaging_and_scans"]
            ],
        }

        try:
            generated = await client.complete_json(
                system_prompt=_SUGGESTION_SYSTEM_PROMPT,
                user_content="CASE RECORD:\n"
                + json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                max_tokens=1800,
                temperature=0.2,
            )
        except Exception:
            logger.exception("[CLINICAL_REVIEW] AI suggestion generation failed")
            notes.append("AI suggestions unavailable — showing record-derived items only.")
            return base

        if not generated:
            notes.append("AI suggestions unavailable — showing record-derived items only.")
            return base

        def merged(key: str) -> list[str]:
            """Record-derived items first, then novel model items."""
            seen = {item.casefold() for item in base[key]}
            out = list(base[key])
            for item in _clean_list(generated.get(key)):
                if item.casefold() not in seen:
                    out.append(item)
                    seen.add(item.casefold())
            return out

        return {
            "differential_diagnoses": merged("differential_diagnoses"),
            "drug_interaction_warnings": merged("drug_interaction_warnings"),
            "red_flag_symptoms": merged("red_flag_symptoms"),
            "suggested_lab_tests": _clean_list(generated.get("suggested_lab_tests")),
            "suggested_imaging": _clean_list(generated.get("suggested_imaging")),
            "clinical_guideline_summary": _text(
                generated.get("clinical_guideline_summary")
            ),
            "possible_contraindications": _clean_list(
                generated.get("possible_contraindications")
            ),
            "relevant_medical_history": merged("relevant_medical_history"),
            "medication_alerts": _clean_list(generated.get("medication_alerts")),
            "source": "groq",
            "generated": True,
            "notes": notes,
        }

    # ── Gaps ─────────────────────────────────────────────────────────────

    @staticmethod
    def _data_gaps(
        overview: dict[str, Any],
        analysis: dict[str, Any],
        evidence: dict[str, Any],
        case: Optional[Case],
    ) -> list[str]:
        """State what the record does not contain, so absence is visible."""
        gaps: list[str] = []
        if case is None:
            gaps.append("This report is not linked to a consultation case.")
        if not analysis["has_ai_intake"]:
            gaps.append("No AI intake session is on file for this case.")
        if analysis["confidence"] is None:
            gaps.append("No AI confidence score was recorded.")
        if overview["blood_group"] is None:
            gaps.append("Blood group is not recorded.")
        if overview["bmi"] is None:
            gaps.append("BMI cannot be computed: height or weight is missing.")
        if not overview["allergies"]:
            gaps.append("No allergies are recorded on the patient profile.")
        if not evidence["lab_reports"] and not evidence["imaging_and_scans"]:
            gaps.append("No laboratory or imaging results are on file.")
        gaps.extend(
            f"Not established during intake: {item}"
            for item in analysis["missing_information"]
        )
        return gaps


clinical_review_service = ClinicalReviewService()
