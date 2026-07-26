"""
Use cases for the Medical Case Intake Agent.

One class per operation, each depending only on ports. These are the units the
API controllers call; they own session lifecycle, authorisation and persistence
sequencing, and delegate all AI reasoning to `IntakeWorkflowPort`.
"""

from __future__ import annotations

import logging

from app.core.exceptions import AuthorizationException, BusinessRuleValidationException
from app.intake.application.dto import (
    RoutingResult,
    SessionView,
    StartIntakeCommand,
    SubmitAnswerCommand,
)
from app.intake.application.ports import (
    CaseRepositoryPort,
    DoctorDirectoryPort,
    IntakeAuditPort,
    IntakeWorkflowPort,
    SessionStorePort,
)
from app.intake.domain.entities import IntakeSession
from app.intake.domain.enums import SessionStatus, TurnRole
from app.intake.domain.errors import InvalidSessionStateError, SessionNotFoundError

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 4000
"""Upper bound on a single patient utterance, before sanitisation."""

SESSION_TTL_SECONDS = 60 * 60
"""In-flight sessions expire after an hour of inactivity."""


def _require_text(text: str, field_name: str = "text") -> str:
    """Reject blank or oversized input before it reaches the model."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise BusinessRuleValidationException(
            f"A non-empty '{field_name}' description is required."
        )
    if len(cleaned) > MAX_INPUT_CHARS:
        raise BusinessRuleValidationException(
            f"'{field_name}' exceeds the maximum of {MAX_INPUT_CHARS} characters."
        )
    return cleaned


def _authorise(session: IntakeSession, patient_user_id: str) -> None:
    """
    Confirm the caller owns this session.

    Session ids are unguessable, but treating that as the only control would
    make them bearer tokens for another patient's clinical conversation.
    """
    if session.patient_user_id != patient_user_id:
        logger.warning(
            "[INTAKE_ACCESS_DENIED] user=%s attempted access to session=%s",
            patient_user_id,
            session.session_id,
        )
        raise AuthorizationException(
            "You are not authorised to access this intake session."
        )


class StartIntakeUseCase:
    """Open a session from the patient's first symptom description."""

    def __init__(
        self, *, sessions: SessionStorePort, workflow: IntakeWorkflowPort
    ) -> None:
        self._sessions = sessions
        self._workflow = workflow

    async def execute(self, command: StartIntakeCommand) -> SessionView:
        text = _require_text(command.text, "symptoms")

        session = IntakeSession(patient_user_id=command.patient_user_id)
        session.add_turn(TurnRole.PATIENT, text)

        logger.info(
            "[INTAKE_START] session=%s user=%s chars=%d",
            session.session_id,
            command.patient_user_id,
            len(text),
        )

        result = await self._workflow.run_detailed(session)
        await self._sessions.save(result.session, ttl_seconds=SESSION_TTL_SECONDS)

        logger.info(
            "[INTAKE_START_DONE] session=%s status=%s red_flags=%d rejected=%d",
            result.session.session_id,
            result.session.status.value,
            len(result.session.red_flags),
            result.rejected_count,
        )
        return SessionView.from_workflow(result)


class SubmitAnswerUseCase:
    """Record the patient's reply to a follow-up and re-evaluate."""

    def __init__(
        self, *, sessions: SessionStorePort, workflow: IntakeWorkflowPort
    ) -> None:
        self._sessions = sessions
        self._workflow = workflow

    async def execute(self, command: SubmitAnswerCommand) -> SessionView:
        text = _require_text(command.text, "answer")

        session = await self._sessions.get(command.session_id)
        if session is None:
            raise SessionNotFoundError(command.session_id)
        _authorise(session, command.patient_user_id)

        if session.status is not SessionStatus.COLLECTING:
            raise InvalidSessionStateError(
                session.session_id,
                current=session.status.value,
                expected=SessionStatus.COLLECTING.value,
            )

        session.add_turn(TurnRole.PATIENT, text)
        session.pending_question = None

        result = await self._workflow.run_detailed(session)
        await self._sessions.save(result.session, ttl_seconds=SESSION_TTL_SECONDS)

        logger.info(
            "[INTAKE_TURN] session=%s status=%s rounds=%d rejected=%d",
            result.session.session_id,
            result.session.status.value,
            result.session.followup_rounds,
            result.rejected_count,
        )
        return SessionView.from_workflow(result)


class GetSessionUseCase:
    """Read the current state of a session."""

    def __init__(self, *, sessions: SessionStorePort) -> None:
        self._sessions = sessions

    async def execute(self, session_id: str, patient_user_id: str) -> SessionView:
        session = await self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        _authorise(session, patient_user_id)
        return SessionView(session=session)


class SelectDoctorUseCase:
    """
    Finalise an intake: persist the clinical case and route it to the chosen
    specialist.

    Ordering matters. The case is written first so the audit trail can reference
    a real case id, and audit failures are logged without rolling back the
    clinical record — losing provenance is bad, losing the patient's case is
    worse.
    """

    def __init__(
        self,
        *,
        sessions: SessionStorePort,
        doctors: DoctorDirectoryPort,
        cases: CaseRepositoryPort,
        audit: IntakeAuditPort,
    ) -> None:
        self._sessions = sessions
        self._doctors = doctors
        self._cases = cases
        self._audit = audit

    async def execute(
        self, *, session_id: str, patient_user_id: str, doctor_id: str
    ) -> RoutingResult:
        session = await self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        _authorise(session, patient_user_id)

        if session.status is not SessionStatus.AWAITING_DOCTOR_SELECTION:
            raise InvalidSessionStateError(
                session.session_id,
                current=session.status.value,
                expected=SessionStatus.AWAITING_DOCTOR_SELECTION.value,
            )

        if session.medical_case is None:
            raise InvalidSessionStateError(
                session.session_id,
                current="no_medical_case",
                expected="medical_case_present",
            )

        doctor = await self._doctors.get(doctor_id)
        if doctor is None:
            raise BusinessRuleValidationException(
                f"Doctor '{doctor_id}' was not found or is not accepting cases."
            )

        case_id = await self._cases.persist_case(
            session=session,
            medical_case=session.medical_case,
            doctor=doctor,
        )

        session.status = SessionStatus.ROUTED
        session.routed_case_id = case_id
        session.routed_doctor_id = doctor.doctor_id
        session.touch()

        # Provenance is best-effort: a compliance-log failure must not cost the
        # patient their consultation.
        try:
            await self._audit.record_session(session, case_id=case_id)
            await self._audit.record_entities(session.session_id, session.entities)
        except Exception:
            logger.exception(
                "[INTAKE_AUDIT_FAILED] session=%s case=%s — clinical record kept",
                session.session_id,
                case_id,
            )

        await self._sessions.save(session, ttl_seconds=SESSION_TTL_SECONDS)

        logger.info(
            "[INTAKE_ROUTED] session=%s case=%s doctor=%s specialty=%s urgency=%s",
            session.session_id,
            case_id,
            doctor.doctor_id,
            doctor.specialty,
            session.medical_case.urgency.value,
        )

        return RoutingResult(
            session_id=session.session_id,
            case_id=case_id,
            doctor_id=doctor.doctor_id,
            doctor_name=doctor.full_name,
            specialty=doctor.specialty,
            urgency=session.medical_case.urgency.value,
        )
