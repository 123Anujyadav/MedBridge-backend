"""
AI-assisted clinical report authoring for the doctor portal.

The "Issue AI Clinical Report" action used to hand the doctor a blank form and
ask them to retype the patient UUID, the patient's name, a title, a summary and
the entire report body — all of it information the platform already held, and
most of it work the AI intake pipeline had already done.

This service inverts that. `build_draft` assembles everything on file for a case
— demographics, the AI intake medical case, symptoms, uploaded reports, prior
cases and prescriptions — and returns a complete draft. The doctor reviews it
and edits only what requires clinical judgement: diagnosis, notes, prescription,
follow-up and recommendations. `issue_report` then renders the PDF, persists the
report, notifies the patient and closes the case.

Grounding rule, consistent with the rest of the platform: the LLM may only
rephrase and organise the context it is handed. Anything it cannot support from
the record is reported as a gap in `warnings`, never invented. If Groq is
unreachable the draft is still produced deterministically from stored records
and flagged `draft_source="records"`, so a doctor always knows what wrote the
text in front of them.
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
from app.models.case import Case, Symptom
from app.models.doctor import Doctor
from app.models.intake import IntakeSessionRecord
from app.models.notification import NotificationItem
from app.models.patient import Patient
from app.models.prescription import Prescription
from app.models.report import Report
from app.schemas.doctor_api import IssueAIReportRequest

logger = logging.getLogger(__name__)

DEFAULT_HOSPITAL = "MedBridge Medical Center"

# Case states that still warrant issuing a report. A case that has already been
# archived is closed to further clinical output.
DRAFTABLE_CASE_STATES = (
    "intake",
    "ai_processing",
    "routed",
    "in_consultation",
    "prescribed",
    "report_generated",
    "completed",
)

_DRAFT_SYSTEM_PROMPT = """You are a clinical documentation assistant preparing a \
draft report for a licensed physician to review, edit and sign.

ABSOLUTE RULES:
- Use ONLY the facts in the CASE RECORD provided. Never introduce a symptom, \
allergy, medication, measurement, test result or history item that is not there.
- You are drafting documentation, not diagnosing. Express the diagnosis field as \
a working clinical impression and keep it consistent with the recorded findings.
- If the record lacks something needed for a safe report, name it in "warnings" \
rather than filling the gap with a plausible guess.
- Do not invent numbers. No fabricated vitals, dosages, lab values or dates.
- Write in clear professional clinical English.

Reply with ONLY a JSON object using exactly these keys:
{
  "title": "concise report title referencing the presenting problem",
  "summary": "2-3 sentence executive summary for the patient's record",
  "clinical_findings": ["objective findings drawn from the record"],
  "diagnosis": "working clinical impression grounded in the recorded findings",
  "clinical_notes": "assessment narrative the physician can edit",
  "prescription": "medication guidance, or an explicit statement that none is \
recommended pending review",
  "follow_up_instructions": "what the patient should do and when to return",
  "recommendations": ["actionable care recommendations"],
  "recommended_tests": ["diagnostic tests worth ordering"],
  "warnings": ["information missing from the record that the physician should \
establish before issuing"]
}"""


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _clean_list(raw: Any, limit: int = 20) -> list[str]:
    """Coerce a model or DB value into a list of non-empty strings."""
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text.lower() not in {"none", "unknown", "n/a", "none reported"}:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _clean_text(raw: Any, fallback: str = "") -> str:
    text = str(raw or "").strip()
    return text if text else fallback


class AIClinicalReportService:
    """Builds and issues AI-drafted clinical reports for a doctor's own cases."""

    # ── Candidate selection ──────────────────────────────────────────────

    async def list_draft_candidates(
        self, db: AsyncSession, doctor_id: uuid.UUID, *, skip: int = 0, limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Cases this doctor may issue a report for, newest first.

        Scoped to `doctor_id` so the picker can never surface another
        clinician's patient.
        """
        result = await db.execute(
            select(Case)
            .where(Case.doctor_id == doctor_id)
            .where(Case.status.in_(DRAFTABLE_CASE_STATES))
            .order_by(Case.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        cases = list(result.scalars().all())
        if not cases:
            return []

        # One query for all intake links rather than one per case.
        case_ids = [c.id for c in cases]
        intake_rows = await db.execute(
            select(IntakeSessionRecord.routed_case_id).where(
                IntakeSessionRecord.routed_case_id.in_(case_ids)
            )
        )
        with_intake = {row for row in intake_rows.scalars().all() if row}

        return [
            {
                "case_id": c.id,
                "patient_id": c.patient_id,
                "patient_name": c.patient_name,
                "patient_age": c.patient_age,
                "patient_gender": c.patient_gender,
                "specialty": c.specialty,
                "urgency_level": c.urgency_level,
                "status": c.status,
                "chief_complaint": (c.symptom_summary or "")[:200],
                "has_ai_intake": c.id in with_intake,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in cases
        ]

    # ── Draft assembly ───────────────────────────────────────────────────

    async def build_draft(
        self, db: AsyncSession, doctor_id: uuid.UUID, case_id: uuid.UUID
    ) -> dict[str, Any]:
        """Assemble the full pre-filled draft for one case."""
        case = await self._load_owned_case(db, doctor_id, case_id)
        patient = await self._load_patient(db, case.patient_id)
        doctor = await db.get(Doctor, doctor_id)

        snapshot = await self._load_intake_snapshot(db, case_id)
        symptoms = await self._load_symptoms(db, case_id)
        prior_reports = await self._load_patient_reports(db, case.patient_id)
        prior_cases = await self._load_prior_cases(db, case.patient_id, case_id)
        prior_rx = await self._load_prescriptions(db, case.patient_id)

        context = self._build_context(
            case=case,
            patient=patient,
            snapshot=snapshot,
            symptoms=symptoms,
            prior_reports=prior_reports,
            prior_cases=prior_cases,
            prior_rx=prior_rx,
        )

        generated, source = await self._generate(context)

        doctor_name = (
            f"Dr. {doctor.first_name} {doctor.last_name}".strip()
            if doctor
            else "Attending Physician"
        )
        hospital_name = (doctor.hospital_name if doctor else None) or DEFAULT_HOSPITAL
        chief_complaint = context["chief_complaint"]

        warnings = _clean_list(generated.get("warnings"), limit=8)
        warnings.extend(self._record_gaps(context))

        return {
            "case_id": case.id,
            "patient_id": case.patient_id,
            "patient_name": context["patient_name"],
            "patient_age": case.patient_age,
            "patient_gender": case.patient_gender,
            "doctor_name": doctor_name,
            "hospital_name": hospital_name,
            "date": _today(),
            "title": _clean_text(
                generated.get("title"),
                f"AI Clinical Report - {chief_complaint[:80]}"
                if chief_complaint
                else "AI Clinical Report",
            ),
            "chief_complaint": chief_complaint,
            "ai_summary": _clean_text(
                generated.get("summary"), context["ai_summary"]
            ),
            "symptoms": context["symptoms"],
            "clinical_findings": _clean_list(generated.get("clinical_findings"))
            or context["clinical_findings"],
            "previous_history": context["previous_history"],
            "uploaded_reports": context["uploaded_reports"],
            "urgency_level": case.urgency_level or "medium",
            "red_flags": context["red_flags"],
            "ai_confidence_score": case.ai_confidence_score,
            "diagnosis": _clean_text(generated.get("diagnosis")),
            "clinical_notes": _clean_text(
                generated.get("clinical_notes"), case.notes or ""
            ),
            "prescription": _clean_text(generated.get("prescription")),
            "follow_up_instructions": _clean_text(
                generated.get("follow_up_instructions")
            ),
            "recommendations": _clean_list(generated.get("recommendations")),
            "recommended_tests": _clean_list(generated.get("recommended_tests")),
            "ai_generated": source == "groq",
            "draft_source": source,
            "warnings": list(dict.fromkeys(warnings))[:10],
        }

    # ── Issue ────────────────────────────────────────────────────────────

    async def issue_report(
        self, db: AsyncSession, doctor_id: uuid.UUID, req: IssueAIReportRequest
    ) -> Report:
        """
        Persist the doctor-approved report and dispatch it.

        Renders the PDF, stores the `Report` row against the patient and case,
        advances the case, notifies the patient and broadcasts the change so
        every open dashboard refreshes.
        """
        from app.core.websocket import websocket_manager
        from app.services.report_generator import report_generator

        case = await self._load_owned_case(db, doctor_id, req.case_id)
        patient = await self._load_patient(db, case.patient_id)
        doctor = await db.get(Doctor, doctor_id)

        doctor_name = (
            f"Dr. {doctor.first_name} {doctor.last_name}".strip()
            if doctor
            else "Attending Physician"
        )
        hospital_name = (doctor.hospital_name if doctor else None) or DEFAULT_HOSPITAL
        patient_name = f"{patient.first_name} {patient.last_name}".strip()

        clinical_notes = self._compose_notes(req)

        pdf_meta = report_generator.generate_pdf(
            patient_name=patient_name,
            patient_id=str(case.patient_id),
            doctor_name=doctor_name,
            doctor_id=str(doctor_id),
            symptoms=case.symptom_summary or "General Medical Consultation",
            diagnosis=req.diagnosis,
            clinical_notes=clinical_notes,
            medications=[],
            recommended_tests=req.recommended_tests or None,
            follow_up_date=req.follow_up_date,
            doctor_remarks=req.follow_up_instructions or None,
            hospital_name=hospital_name,
        )

        summary = req.summary.strip() or (
            f"Working impression: {req.diagnosis}. "
            f"Issued by {doctor_name} on {_today()}."
        )

        report = Report(
            patient_id=case.patient_id,
            case_id=case.id,
            patient_name=patient_name,
            type="ai_report",
            title=req.title,
            summary=summary[:2000],
            content=self._compose_content(
                req=req,
                case=case,
                doctor_name=doctor_name,
                hospital_name=hospital_name,
            ),
            doctor_name=doctor_name,
            hospital_name=hospital_name,
            date=_today(),
            status="ready",
            file_url=pdf_meta["file_url"],
            file_size=pdf_meta["file_size"],
            ai_generated=req.ai_generated,
            ai_confidence_score=req.ai_confidence_score,
            tags=["ai_report", "clinical_review", "pdf"],
            vitals={},
        )
        db.add(report)
        await db.flush()

        # Advance the case: the clinical output for it now exists.
        case.status = "report_generated"
        case.notes = clinical_notes
        case.completed_at = datetime.now(timezone.utc)
        await db.flush()

        # Deliver to the patient. Patient.id is the user id, so it addresses the
        # notification directly.
        #
        # Routed through `notification_service` rather than constructing the row
        # here: this predates that service, and building it directly meant the
        # patient's most important notification had no category (so it was
        # invisible to filters), no live WebSocket push, no deduplication and no
        # audit entry.
        from app.services.notifications import notification_service

        await notification_service.safe_notify(
            db,
            user_id=case.patient_id,
            category="report",
            type="report_issued",
            title="New Clinical Report Available",
            message=f"{doctor_name} has issued your clinical report: {req.title}.",
            priority="high" if case.urgency_level in ("high", "critical") else "medium",
            case_id=case.id,
            patient_id=case.patient_id,
            patient_name=patient_name,
            action_url=f"/patient/reports?report={report.id}",
            action_label="View Report",
            group_key="report_issued",
            dedupe_key=f"report_issued:{report.id}",
        )

        await websocket_manager.broadcast(
            {
                "type": "REPORT_ISSUED",
                "event": "ai_clinical_report_issued",
                "report_id": str(report.id),
                "case_id": str(case.id),
                "patient_id": str(case.patient_id),
                "doctor_id": str(doctor_id),
                "file_url": pdf_meta["file_url"],
            }
        )

        from app.models.user import User
        from app.services.case_timeline import case_timeline_service
        from app.services.report_versions import report_version_service

        # Version 1 of the document, carrying the same content the PDF above was
        # rendered from. The already-rendered file is reused rather than a
        # second, identical PDF being produced for the version.
        issuing_doctor = await db.get(User, doctor_id)
        version, _ = await report_version_service.create_version(
            db,
            report=report,
            snapshot={
                "title": req.title,
                "chief_complaint": case.symptom_summary or "",
                "summary": summary,
                "content": report.content,
                "diagnosis": req.diagnosis,
                "clinical_notes": req.clinical_notes,
                "prescription": req.prescription,
                "follow_up_instructions": req.follow_up_instructions,
                "ai_findings": summary if req.ai_generated else "",
                "symptoms": list(case.ai_extracted_symptoms or []),
                "recommended_tests": list(req.recommended_tests or []),
                "recommendations": list(req.recommendations or []),
            },
            author=issuing_doctor,
            author_type="doctor",
            author_name=doctor_name,
            status="approved",
            description="Report issued and approved by the treating clinician.",
            approval_note="Issued via the AI-assisted clinical report workflow.",
            render=False,
        )
        version.file_url = pdf_meta["file_url"]
        version.file_size = pdf_meta["file_size"]
        await db.flush()

        await case_timeline_service.safe_record(
            db,
            event_type="report.generated",
            description=f"Clinical report issued: {req.title}",
            actor=await db.get(User, doctor_id),
            # The narrative is AI-drafted but a clinician signs it, so the actor
            # of record is the doctor; `ai_assisted` states the AI's part.
            actor_type="doctor",
            case_id=case.id,
            patient_id=case.patient_id,
            resource="Report",
            resource_id=str(report.id),
            field="ai_assisted",
            new="true" if req.ai_generated else "false",
        )

        logger.info(
            "[AI_REPORT_ISSUED] report=%s case=%s patient=%s doctor=%s ai=%s",
            report.id,
            case.id,
            case.patient_id,
            doctor_id,
            req.ai_generated,
        )
        return report

    # ── Loading helpers ──────────────────────────────────────────────────

    async def _load_owned_case(
        self, db: AsyncSession, doctor_id: uuid.UUID, case_id: uuid.UUID
    ) -> Case:
        case = await db.get(Case, case_id)
        if case is None or case.deleted_at is not None:
            raise EntityNotFoundException("Case", str(case_id))
        if case.doctor_id != doctor_id:
            raise AuthorizationException(
                "You are not authorised to issue a report for this case."
            )
        return case

    async def _load_patient(self, db: AsyncSession, patient_id: uuid.UUID) -> Patient:
        patient = await db.get(Patient, patient_id)
        if patient is None:
            raise EntityNotFoundException("Patient", str(patient_id))
        return patient

    async def _load_intake_snapshot(
        self, db: AsyncSession, case_id: uuid.UUID
    ) -> Optional[dict[str, Any]]:
        """The AI intake medical case for this consultation, if one exists."""
        result = await db.execute(
            select(IntakeSessionRecord)
            .where(IntakeSessionRecord.routed_case_id == case_id)
            .order_by(IntakeSessionRecord.created_at.desc())
            .limit(1)
        )
        record = result.scalars().first()
        if record is None or not record.medical_case_snapshot:
            return None
        snapshot = record.medical_case_snapshot
        return snapshot if isinstance(snapshot, dict) else None

    async def _load_symptoms(
        self, db: AsyncSession, case_id: uuid.UUID
    ) -> list[Symptom]:
        result = await db.execute(select(Symptom).where(Symptom.case_id == case_id))
        return list(result.scalars().all())

    async def _load_patient_reports(
        self, db: AsyncSession, patient_id: uuid.UUID, limit: int = 10
    ) -> list[Report]:
        result = await db.execute(
            select(Report)
            .where(Report.patient_id == patient_id)
            .order_by(Report.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def _load_prior_cases(
        self,
        db: AsyncSession,
        patient_id: uuid.UUID,
        current_case_id: uuid.UUID,
        limit: int = 5,
    ) -> list[Case]:
        result = await db.execute(
            select(Case)
            .where(Case.patient_id == patient_id)
            .where(Case.id != current_case_id)
            .order_by(Case.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def _load_prescriptions(
        self, db: AsyncSession, patient_id: uuid.UUID, limit: int = 5
    ) -> list[Prescription]:
        result = await db.execute(
            select(Prescription)
            .where(Prescription.patient_id == patient_id)
            .order_by(Prescription.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    # ── Context assembly ─────────────────────────────────────────────────

    def _build_context(
        self,
        *,
        case: Case,
        patient: Patient,
        snapshot: Optional[dict[str, Any]],
        symptoms: list[Symptom],
        prior_reports: list[Report],
        prior_cases: list[Case],
        prior_rx: list[Prescription],
    ) -> dict[str, Any]:
        """
        Fold every source into one grounded record.

        The AI intake snapshot is preferred where it exists because it carries
        evidence-checked entities; the case row is the fallback.
        """
        snap = snapshot or {}

        chief_complaint = _clean_text(
            snap.get("chief_complaint"), _clean_text(case.symptom_summary)
        )

        symptom_names = _clean_list(snap.get("symptoms")) or _clean_list(
            case.ai_extracted_symptoms
        )
        if not symptom_names:
            symptom_names = _clean_list([s.name for s in symptoms])

        clinical_findings: list[str] = []
        for s in symptoms:
            parts = [s.name]
            if s.severity:
                parts.append(f"severity {s.severity}")
            if s.duration and s.duration.lower() != "unknown":
                parts.append(f"duration {s.duration}")
            if s.body_part:
                parts.append(f"site {s.body_part}")
            clinical_findings.append(" — ".join(parts))
        for key, label in (("severity", "Severity"), ("onset", "Onset"), ("duration", "Duration")):
            value = _clean_text(snap.get(key))
            if value:
                clinical_findings.append(f"{label}: {value}")

        # Previous history: profile-level facts first, then the record trail.
        previous_history: list[str] = []
        for values, label in (
            (_clean_list(patient.allergies), "Known allergies"),
            (_clean_list(patient.chronic_conditions), "Chronic conditions"),
            (_clean_list(patient.medications), "Current medications"),
        ):
            if values:
                previous_history.append(f"{label}: {', '.join(values)}")

        for key, label in (
            ("allergies", "Allergies reported at intake"),
            ("current_medications", "Medications reported at intake"),
            ("medical_history", "Past medical history"),
        ):
            values = _clean_list(snap.get(key))
            if values:
                previous_history.append(f"{label}: {', '.join(values)}")

        for prior in prior_cases:
            previous_history.append(
                f"Prior case ({prior.status}): {(prior.symptom_summary or '')[:160]}"
            )
        for rx in prior_rx:
            previous_history.append(
                f"Prior prescription: {rx.diagnosis} ({rx.status})"
            )

        uploaded_reports = [
            {
                "report_id": r.id,
                "title": r.title,
                "type": r.type,
                "date": r.date,
                "summary": (r.summary or "")[:300],
                "file_url": r.file_url,
            }
            for r in prior_reports
        ]

        ai_summary = _clean_text(
            snap.get("summary_for_doctor"),
            _clean_text(case.notes, _clean_text(case.symptom_summary)),
        )

        return {
            "patient_name": f"{patient.first_name} {patient.last_name}".strip()
            or case.patient_name,
            "patient_age": case.patient_age,
            "patient_gender": case.patient_gender,
            "blood_type": patient.blood_type,
            "specialty": case.specialty,
            "urgency": case.urgency_level,
            "chief_complaint": chief_complaint,
            "ai_summary": ai_summary,
            "symptoms": symptom_names,
            "clinical_findings": clinical_findings,
            "previous_history": previous_history,
            "uploaded_reports": uploaded_reports,
            "red_flags": _clean_list(snap.get("red_flags")),
            "differential_considerations": _clean_list(
                snap.get("differential_considerations")
            ),
            "missing_information": _clean_list(snap.get("missing_information")),
            "patient_history_note": _clean_text(case.patient_history),
            "doctor_notes": _clean_text(case.notes),
            "has_ai_intake": snapshot is not None,
        }

    def _record_gaps(self, context: dict[str, Any]) -> list[str]:
        """Gaps derivable from the record itself, independent of the LLM."""
        gaps: list[str] = []
        if not context["symptoms"]:
            gaps.append("No structured symptoms are recorded for this case.")
        if not context["chief_complaint"]:
            gaps.append("No chief complaint is recorded for this case.")
        if not context["has_ai_intake"]:
            gaps.append(
                "No AI intake session is linked to this case; the draft is built "
                "from case records only."
            )
        for item in context["missing_information"]:
            gaps.append(f"Not established during intake: {item}")
        return gaps

    # ── Generation ───────────────────────────────────────────────────────

    async def _generate(self, context: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """
        Draft the narrative fields.

        Returns `({}, "records")` when Groq is unavailable or returns nothing
        usable, so the caller falls back to the deterministic draft rather than
        showing the doctor an empty form.
        """
        client = get_groq_client()
        if not client.is_configured:
            logger.warning("[AI_REPORT] Groq unavailable — using records-only draft")
            return {}, "records"

        try:
            generated = await client.complete_json(
                system_prompt=_DRAFT_SYSTEM_PROMPT,
                user_content="CASE RECORD:\n"
                + json.dumps(self._llm_payload(context), ensure_ascii=False, indent=2),
                max_tokens=2200,
                temperature=0.2,
            )
        except Exception:
            logger.exception("[AI_REPORT] draft generation failed")
            return {}, "records"

        if not generated:
            logger.warning("[AI_REPORT] empty draft returned — using records only")
            return {}, "records"
        return generated, "groq"

    @staticmethod
    def _llm_payload(context: dict[str, Any]) -> dict[str, Any]:
        """
        The record slice handed to the model.

        Report ids and file URLs are stripped — the model has no use for them and
        they are unnecessary identifiers to put in a prompt.
        """
        payload = {
            k: v
            for k, v in context.items()
            if k not in {"uploaded_reports", "has_ai_intake"}
        }
        payload["uploaded_reports"] = [
            {"title": r["title"], "type": r["type"], "date": r["date"], "summary": r["summary"]}
            for r in context["uploaded_reports"]
        ]
        return payload

    # ── Composition ──────────────────────────────────────────────────────

    @staticmethod
    def _compose_notes(req: IssueAIReportRequest) -> str:
        """Doctor's editable clinical fields, flattened for the PDF body."""
        blocks = [req.clinical_notes.strip()]
        if req.prescription.strip():
            blocks.append(f"Prescription: {req.prescription.strip()}")
        if req.recommendations:
            blocks.append(
                "Recommendations: " + "; ".join(req.recommendations)
            )
        return "\n\n".join(b for b in blocks if b) or "No additional notes."

    @staticmethod
    def _compose_content(
        *,
        req: IssueAIReportRequest,
        case: Case,
        doctor_name: str,
        hospital_name: str,
    ) -> str:
        """The stored, human-readable report body shown in the portal."""
        sections = [
            f"CLINICAL REPORT — {req.title}",
            f"Issued by {doctor_name}, {hospital_name} on {_today()}.",
            "",
            f"Chief Complaint: {case.symptom_summary or 'Not recorded'}",
            f"Working Diagnosis: {req.diagnosis}",
        ]
        if req.clinical_notes.strip():
            sections += ["", "Clinical Notes:", req.clinical_notes.strip()]
        if req.prescription.strip():
            sections += ["", "Prescription:", req.prescription.strip()]
        if req.recommended_tests:
            sections += ["", "Recommended Tests:", "; ".join(req.recommended_tests)]
        if req.recommendations:
            sections += [
                "",
                "Recommendations:",
                "\n".join(f"- {r}" for r in req.recommendations),
            ]
        if req.follow_up_instructions.strip():
            sections += [
                "",
                "Follow-up Instructions:",
                req.follow_up_instructions.strip(),
            ]
        if req.follow_up_date:
            sections.append(f"Follow-up Date: {req.follow_up_date}")
        sections += [
            "",
            "This report was drafted with AI assistance and reviewed, edited and "
            f"approved by {doctor_name} before issue."
            if req.ai_generated
            else f"This report was authored and approved by {doctor_name}.",
        ]
        return "\n".join(sections)


ai_clinical_report_service = AIClinicalReportService()
