"""
Use-case tests for the Medical Case Intake Agent.

Covers authorisation, session lifecycle rules, input validation and failure
containment, all against in-memory doubles.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import AuthorizationException, BusinessRuleValidationException
from app.intake.application.dto import (
    DoctorRef,
    StartIntakeCommand,
    SubmitAnswerCommand,
    WorkflowResult,
)
from app.intake.application.use_cases import (
    MAX_INPUT_CHARS,
    GetSessionUseCase,
    SelectDoctorUseCase,
    StartIntakeUseCase,
    SubmitAnswerUseCase,
)
from app.intake.domain.entities import IntakeSession, MedicalCase
from app.intake.domain.enums import SessionStatus, UrgencyLevel
from app.intake.domain.errors import InvalidSessionStateError, SessionNotFoundError
from tests.intake.conftest import (
    FakeDoctorDirectory,
    InMemorySessionStore,
    ScriptedLLM,
    complete_extraction,
    make_session,
)
from app.intake.workflow.graph import LangGraphIntakeWorkflow

pytestmark = pytest.mark.asyncio

DOCTOR_ID = "11111111-1111-1111-1111-111111111111"


class RecordingCaseRepo:
    """`CaseRepositoryPort` double that records what it persisted."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.persisted: list[tuple[str, str]] = []

    async def persist_case(self, *, session, medical_case, doctor) -> str:
        if self.fail:
            raise RuntimeError("database is down")
        self.persisted.append((session.session_id, doctor.doctor_id))
        return "case-abc-123"


class RecordingAudit:
    """`IntakeAuditPort` double, optionally failing."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.sessions: list[str] = []
        self.entities: list[str] = []

    async def record_session(self, session, *, case_id=None) -> None:
        if self.fail:
            raise RuntimeError("audit store unavailable")
        self.sessions.append(session.session_id)

    async def record_entities(self, session_id, entities, *, rejected=None) -> None:
        if self.fail:
            raise RuntimeError("audit store unavailable")
        self.entities.append(session_id)


class StubWorkflow:
    """`IntakeWorkflowPort` double that applies a caller-supplied mutation."""

    def __init__(self, mutate=None) -> None:
        self._mutate = mutate
        self.runs = 0

    async def run_detailed(self, session: IntakeSession) -> WorkflowResult:
        self.runs += 1
        if self._mutate:
            self._mutate(session)
        return WorkflowResult(session=session)


def _ready_session(user_id: str = "user-1") -> IntakeSession:
    session = make_session(user_id=user_id)
    session.status = SessionStatus.AWAITING_DOCTOR_SELECTION
    session.medical_case = MedicalCase(
        chief_complaint="Chest discomfort", urgency=UrgencyLevel.MEDIUM
    )
    return session


# ------------------------------------------------------------ Start intake


class TestStartIntake:
    async def test_creates_and_persists_session(self, memory_sessions):
        use_case = StartIntakeUseCase(
            sessions=memory_sessions, workflow=StubWorkflow()
        )
        view = await use_case.execute(
            StartIntakeCommand(patient_user_id="user-1", text="I have a headache")
        )

        assert view.session.patient_user_id == "user-1"
        assert view.session.session_id in memory_sessions.store
        assert view.session.turns[0].text == "I have a headache"

    @pytest.mark.parametrize("text", ["", "   ", "\n\t "])
    async def test_rejects_blank_input(self, memory_sessions, text):
        use_case = StartIntakeUseCase(
            sessions=memory_sessions, workflow=StubWorkflow()
        )
        with pytest.raises(BusinessRuleValidationException):
            await use_case.execute(
                StartIntakeCommand(patient_user_id="user-1", text=text)
            )

    async def test_rejects_oversized_input(self, memory_sessions):
        use_case = StartIntakeUseCase(
            sessions=memory_sessions, workflow=StubWorkflow()
        )
        with pytest.raises(BusinessRuleValidationException):
            await use_case.execute(
                StartIntakeCommand(
                    patient_user_id="user-1", text="a" * (MAX_INPUT_CHARS + 1)
                )
            )

    async def test_uses_real_workflow_end_to_end(self, memory_sessions):
        workflow = LangGraphIntakeWorkflow(
            llm=ScriptedLLM({"extraction": complete_extraction()}),
            doctors=FakeDoctorDirectory(),
        )
        use_case = StartIntakeUseCase(sessions=memory_sessions, workflow=workflow)
        view = await use_case.execute(
            StartIntakeCommand(
                patient_user_id="user-1",
                text="I have had moderate chest discomfort for 3 days now.",
            )
        )
        assert view.session.status is SessionStatus.AWAITING_DOCTOR_SELECTION


# ----------------------------------------------------------- Submit answer


class TestSubmitAnswer:
    async def test_appends_turn_and_reruns_workflow(self, memory_sessions):
        session = make_session()
        session.pending_question = "How long?"
        await memory_sessions.save(session)

        workflow = StubWorkflow()
        use_case = SubmitAnswerUseCase(sessions=memory_sessions, workflow=workflow)
        view = await use_case.execute(
            SubmitAnswerCommand(
                patient_user_id="user-1",
                session_id=session.session_id,
                text="about three days",
            )
        )

        assert workflow.runs == 1
        assert view.session.turns[-1].text == "about three days"
        assert view.session.pending_question is None

    async def test_unknown_session_raises_not_found(self, memory_sessions):
        use_case = SubmitAnswerUseCase(
            sessions=memory_sessions, workflow=StubWorkflow()
        )
        with pytest.raises(SessionNotFoundError):
            await use_case.execute(
                SubmitAnswerCommand(
                    patient_user_id="user-1", session_id="does-not-exist", text="hi"
                )
            )

    async def test_other_patient_cannot_advance_session(self, memory_sessions):
        """A session id must not act as a bearer token for another patient."""
        session = make_session(user_id="user-1")
        await memory_sessions.save(session)

        use_case = SubmitAnswerUseCase(
            sessions=memory_sessions, workflow=StubWorkflow()
        )
        with pytest.raises(AuthorizationException):
            await use_case.execute(
                SubmitAnswerCommand(
                    patient_user_id="attacker-9",
                    session_id=session.session_id,
                    text="hello",
                )
            )

    async def test_cannot_answer_a_finished_session(self, memory_sessions):
        session = _ready_session()
        await memory_sessions.save(session)

        use_case = SubmitAnswerUseCase(
            sessions=memory_sessions, workflow=StubWorkflow()
        )
        with pytest.raises(InvalidSessionStateError):
            await use_case.execute(
                SubmitAnswerCommand(
                    patient_user_id="user-1",
                    session_id=session.session_id,
                    text="more info",
                )
            )


# -------------------------------------------------------------- Get session


class TestGetSession:
    async def test_returns_owned_session(self, memory_sessions):
        session = make_session()
        await memory_sessions.save(session)

        view = await GetSessionUseCase(sessions=memory_sessions).execute(
            session.session_id, "user-1"
        )
        assert view.session.session_id == session.session_id

    async def test_rejects_other_patients(self, memory_sessions):
        session = make_session(user_id="user-1")
        await memory_sessions.save(session)

        with pytest.raises(AuthorizationException):
            await GetSessionUseCase(sessions=memory_sessions).execute(
                session.session_id, "user-2"
            )


# ------------------------------------------------------------ Select doctor


class TestSelectDoctor:
    @staticmethod
    def _use_case(sessions, *, repo=None, audit=None, doctors=None):
        return SelectDoctorUseCase(
            sessions=sessions,
            doctors=doctors or FakeDoctorDirectory(),
            cases=repo or RecordingCaseRepo(),
            audit=audit or RecordingAudit(),
        )

    async def test_persists_case_and_routes(self, memory_sessions):
        session = _ready_session()
        await memory_sessions.save(session)
        repo, audit = RecordingCaseRepo(), RecordingAudit()

        result = await self._use_case(
            memory_sessions, repo=repo, audit=audit
        ).execute(
            session_id=session.session_id,
            patient_user_id="user-1",
            doctor_id=DOCTOR_ID,
        )

        assert result.case_id == "case-abc-123"
        assert result.doctor_id == DOCTOR_ID
        assert repo.persisted == [(session.session_id, DOCTOR_ID)]
        assert audit.sessions == [session.session_id]

        stored = await memory_sessions.get(session.session_id)
        assert stored.status is SessionStatus.ROUTED
        assert stored.routed_case_id == "case-abc-123"

    async def test_rejects_when_case_not_ready(self, memory_sessions):
        session = make_session()  # still COLLECTING
        await memory_sessions.save(session)

        with pytest.raises(InvalidSessionStateError):
            await self._use_case(memory_sessions).execute(
                session_id=session.session_id,
                patient_user_id="user-1",
                doctor_id=DOCTOR_ID,
            )

    async def test_rejects_unknown_doctor(self, memory_sessions):
        session = _ready_session()
        await memory_sessions.save(session)

        with pytest.raises(BusinessRuleValidationException):
            await self._use_case(memory_sessions).execute(
                session_id=session.session_id,
                patient_user_id="user-1",
                doctor_id="99999999-9999-9999-9999-999999999999",
            )

    async def test_rejects_other_patients(self, memory_sessions):
        session = _ready_session(user_id="user-1")
        await memory_sessions.save(session)

        with pytest.raises(AuthorizationException):
            await self._use_case(memory_sessions).execute(
                session_id=session.session_id,
                patient_user_id="intruder",
                doctor_id=DOCTOR_ID,
            )

    async def test_database_failure_propagates(self, memory_sessions):
        """A failed clinical write must surface, not be silently swallowed."""
        session = _ready_session()
        await memory_sessions.save(session)

        with pytest.raises(RuntimeError, match="database is down"):
            await self._use_case(
                memory_sessions, repo=RecordingCaseRepo(fail=True)
            ).execute(
                session_id=session.session_id,
                patient_user_id="user-1",
                doctor_id=DOCTOR_ID,
            )

        stored = await memory_sessions.get(session.session_id)
        assert stored.status is SessionStatus.AWAITING_DOCTOR_SELECTION

    async def test_audit_failure_does_not_lose_the_case(self, memory_sessions):
        """
        Losing provenance is bad; losing the patient's consultation is worse.
        An audit write failure must not roll back a successful routing.
        """
        session = _ready_session()
        await memory_sessions.save(session)
        repo = RecordingCaseRepo()

        result = await self._use_case(
            memory_sessions, repo=repo, audit=RecordingAudit(fail=True)
        ).execute(
            session_id=session.session_id,
            patient_user_id="user-1",
            doctor_id=DOCTOR_ID,
        )

        assert result.case_id == "case-abc-123"
        stored = await memory_sessions.get(session.session_id)
        assert stored.status is SessionStatus.ROUTED
