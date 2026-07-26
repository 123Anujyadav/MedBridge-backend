"""
SQLAlchemy adapters for clinical persistence and the intake audit trail.

`SqlCaseRepository` writes into the existing `cases`/`symptoms` tables so an
AI-generated case is indistinguishable, to the doctor portal, from any other.
`SqlIntakeAudit` writes the provenance tables introduced with this agent.

Both follow the project's transaction convention: flush here, let the
`get_db` request wrapper own the commit.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import EntityNotFoundException
from app.intake.application.dto import DoctorRef
from app.intake.domain.entities import ExtractedEntity, IntakeSession, MedicalCase
from app.intake.domain.enums import EntityKind
from app.models.case import Case, Symptom
from app.models.intake import IntakeExtractedEntity, IntakeSessionRecord
from app.models.patient import Patient

logger = logging.getLogger(__name__)

# Case.symptoms.severity is constrained to these three values.
_SEVERITY_BUCKETS = ("mild", "moderate", "severe")
_UNKNOWN = "Unknown"


def _bucket_severity(raw: str) -> str:
    """Map free-text severity onto the `symptoms.severity` CHECK constraint."""
    text = (raw or "").casefold()
    if any(token in text for token in ("mild", "slight", "low", "minor", "1", "2", "3")):
        return "mild"
    if any(
        token in text
        for token in ("severe", "critical", "extreme", "worst", "high", "8", "9", "10")
    ):
        return "severe"
    return "moderate"


def _age_from_dob(date_of_birth: str | None) -> int:
    """
    Best-effort age in years from an ISO-ish date string.

    Returns 0 rather than a plausible-looking default when the date cannot be
    parsed: an unknown age must read as unknown, not as a confident 30.
    """
    if not date_of_birth:
        return 0
    try:
        year = int(str(date_of_birth).split("-")[0])
        age = datetime.now(timezone.utc).year - year
        return age if 0 <= age <= 130 else 0
    except (ValueError, IndexError):
        return 0


class SqlCaseRepository:
    """Persists a completed intake as a clinical `Case` plus `Symptom` rows."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def persist_case(
        self,
        *,
        session: IntakeSession,
        medical_case: MedicalCase,
        doctor: DoctorRef,
    ) -> str:
        patient = await self._load_patient(session.patient_user_id)

        case = Case(
            patient_id=patient.id,
            patient_name=f"{patient.first_name} {patient.last_name}".strip(),
            patient_avatar_url=patient.avatar_url,
            patient_age=_age_from_dob(patient.date_of_birth),
            patient_gender=patient.gender or "other",
            doctor_id=uuid.UUID(doctor.doctor_id),
            doctor_name=doctor.full_name,
            specialty=doctor.specialty,
            symptom_summary=self._build_summary(medical_case),
            urgency_level=medical_case.urgency.value,
            status="routed",
            ai_extracted_symptoms=list(medical_case.symptoms),
            ai_specialty_recommendation=medical_case.recommended_specialty,
            ai_confidence_score=medical_case.overall_confidence.score,
            attachments=[],
            patient_history=self._build_history(medical_case),
            notes=medical_case.summary_for_doctor or "",
            assigned_at=datetime.now(timezone.utc),
        )
        self._db.add(case)
        await self._db.flush()

        duration = (
            medical_case.duration if medical_case.duration != _UNKNOWN else _UNKNOWN
        )
        body_site = medical_case.body_sites[0] if medical_case.body_sites else None
        for name in medical_case.symptoms:
            self._db.add(
                Symptom(
                    case_id=case.id,
                    name=name[:100],
                    severity=_bucket_severity(medical_case.severity),
                    duration=duration[:100],
                    body_part=body_site[:100] if body_site else None,
                )
            )
        await self._db.flush()

        logger.info(
            "[INTAKE_CASE_PERSISTED] case=%s patient=%s doctor=%s symptoms=%d",
            case.id,
            patient.id,
            doctor.doctor_id,
            len(medical_case.symptoms),
        )
        return str(case.id)

    async def _load_patient(self, patient_user_id: str) -> Patient:
        """
        Resolve the patient profile.

        Raises rather than creating a placeholder profile on the fly: silently
        inventing a patient record to satisfy a foreign key produces clinical
        data attached to a fictional person.
        """
        try:
            parsed = uuid.UUID(str(patient_user_id))
        except (ValueError, TypeError):
            raise EntityNotFoundException("Patient", str(patient_user_id))

        result = await self._db.execute(select(Patient).where(Patient.id == parsed))
        patient = result.scalars().first()
        if patient is None:
            raise EntityNotFoundException("Patient", str(patient_user_id))
        return patient

    @staticmethod
    def _build_summary(case: MedicalCase) -> str:
        parts = [case.chief_complaint]
        if case.symptoms:
            parts.append(f"Symptoms: {', '.join(case.symptoms)}.")
        if case.duration != _UNKNOWN:
            parts.append(f"Duration: {case.duration}.")
        if case.severity != _UNKNOWN:
            parts.append(f"Severity: {case.severity}.")
        if case.red_flags:
            parts.append(f"RED FLAGS: {'; '.join(case.red_flags)}.")
        return " ".join(p for p in parts if p).strip()

    @staticmethod
    def _build_history(case: MedicalCase) -> str:
        """
        Assemble the history block.

        Absent fields are written as explicit 'None reported' rather than being
        omitted, so a clinician can tell 'the patient denied allergies' apart
        from 'nobody asked'.
        """
        lines = [
            f"Allergies: {', '.join(case.allergies) or 'None reported'}",
            f"Current medications: {', '.join(case.current_medications) or 'None reported'}",
            f"Past history: {', '.join(case.medical_history) or 'None reported'}",
        ]
        if case.missing_information:
            lines.append(
                f"Not established during intake: {', '.join(case.missing_information)}"
            )
        if case.differential_considerations:
            lines.append(
                "For clinician consideration (not a diagnosis): "
                + ", ".join(case.differential_considerations)
            )
        return "\n".join(lines)


class SqlIntakeAudit:
    """Writes the intake provenance tables."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def record_session(
        self, session: IntakeSession, *, case_id: str | None = None
    ) -> None:
        """Insert or update the durable record for this conversation."""
        result = await self._db.execute(
            select(IntakeSessionRecord).where(
                IntakeSessionRecord.session_key == session.session_id
            )
        )
        record = result.scalars().first()

        case_snapshot = (
            session.medical_case.to_dict() if session.medical_case else None
        )
        confidence = (
            session.medical_case.overall_confidence.score
            if session.medical_case
            else 0.0
        )

        if record is None:
            record = IntakeSessionRecord(
                session_key=session.session_id,
                patient_user_id=uuid.UUID(session.patient_user_id),
            )
            self._db.add(record)

        record.status = session.status.value
        record.language = session.language.value
        record.intent = session.intent.value
        record.followup_rounds = session.followup_rounds
        record.overall_confidence = confidence
        record.red_flags = list(session.red_flags)
        record.transcript = session.patient_transcript()
        record.medical_case_snapshot = case_snapshot
        if case_id:
            record.routed_case_id = uuid.UUID(case_id)
        if session.routed_doctor_id:
            record.routed_doctor_id = uuid.UUID(session.routed_doctor_id)

        await self._db.flush()

    async def record_entities(
        self,
        session_id: str,
        entities: list[ExtractedEntity],
        *,
        rejected: list[ExtractedEntity] | None = None,
    ) -> None:
        """
        Write the per-entity provenance rows.

        Rejected extractions are stored alongside accepted ones so the
        fabrication rate is measurable rather than invisible.
        """
        result = await self._db.execute(
            select(IntakeSessionRecord).where(
                IntakeSessionRecord.session_key == session_id
            )
        )
        record = result.scalars().first()
        if record is None:
            logger.warning(
                "[INTAKE_AUDIT_NO_SESSION] session=%s — entities not recorded",
                session_id,
            )
            return

        existing = await self._db.execute(
            select(IntakeExtractedEntity).where(
                IntakeExtractedEntity.session_id == record.id
            )
        )
        for row in existing.scalars().all():
            await self._db.delete(row)

        for entity in entities:
            self._db.add(self._to_row(record.id, entity, accepted=True))
        for entity in rejected or []:
            self._db.add(self._to_row(record.id, entity, accepted=False))

        await self._db.flush()

    @staticmethod
    def _to_row(
        record_id: uuid.UUID, entity: ExtractedEntity, *, accepted: bool
    ) -> IntakeExtractedEntity:
        return IntakeExtractedEntity(
            session_id=record_id,
            kind=entity.kind.value if isinstance(entity.kind, EntityKind) else str(entity.kind),
            value=entity.value[:500],
            confidence=entity.confidence.score,
            confidence_band=entity.confidence.band.value,
            evidence_quote=entity.evidence.quote,
            evidence_turn_index=entity.evidence.turn_index,
            was_accepted=accepted,
        )
