"""
Ports: the abstract collaborators the intake use cases depend on.

Structural `Protocol`s rather than ABCs, so infrastructure adapters and test
fakes satisfy them by shape alone with no inheritance coupling. Dependencies
point inward: `application` declares what it needs, `infrastructure` conforms.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.intake.application.dto import DoctorRef, WorkflowResult
from app.intake.domain.entities import ExtractedEntity, IntakeSession, MedicalCase


@runtime_checkable
class LLMPort(Protocol):
    """A large language model that returns structured JSON."""

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 1200,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """
        Run a completion and parse the reply as a JSON object.

        Implementations must return `{}` rather than raising when the model is
        unavailable or returns unparseable output: every workflow node is
        required to degrade safely instead of failing the whole intake.
        """
        ...

    async def health(self) -> dict[str, Any]:
        """Provider reachability, for the monitoring endpoints."""
        ...


@runtime_checkable
class SessionStorePort(Protocol):
    """Persistence for in-flight intake sessions."""

    async def get(self, session_id: str) -> IntakeSession | None: ...

    async def save(self, session: IntakeSession, ttl_seconds: int | None = None) -> None: ...

    async def delete(self, session_id: str) -> None: ...


@runtime_checkable
class IntakeWorkflowPort(Protocol):
    """
    The AI orchestration graph.

    Keeping this behind a port is what lets the use cases stay free of LangGraph
    and lets tests drive the full HTTP surface with a scripted workflow.
    """

    async def run_detailed(self, session: IntakeSession) -> WorkflowResult:
        """
        Advance the session by one full pass of the intake graph.

        Must not raise: a workflow failure has to leave the session usable so
        the patient can retry, rather than collapsing the request.
        """
        ...


@runtime_checkable
class CaseRepositoryPort(Protocol):
    """Persists a completed intake as a clinical case in the main database."""

    async def persist_case(
        self,
        *,
        session: IntakeSession,
        medical_case: MedicalCase,
        doctor: DoctorRef,
    ) -> str:
        """Create the case (and its symptom rows). Returns the new case id."""
        ...


@runtime_checkable
class DoctorDirectoryPort(Protocol):
    """Looks up real clinicians available to receive a case."""

    async def find_for_specialty(
        self, specialty: str, *, limit: int = 3
    ) -> list[DoctorRef]:
        """Best-matching doctors for a specialty, ranked most suitable first."""
        ...

    async def get(self, doctor_id: str) -> DoctorRef | None: ...


@runtime_checkable
class IntakeAuditPort(Protocol):
    """
    Durable audit trail of what the agent extracted and why.

    Separate from `CaseRepositoryPort` because the clinical case and the
    extraction provenance have different consumers: clinicians read the case,
    compliance reads the audit.
    """

    async def record_session(
        self, session: IntakeSession, *, case_id: str | None = None
    ) -> None: ...

    async def record_entities(
        self,
        session_id: str,
        entities: list[ExtractedEntity],
        *,
        rejected: list[ExtractedEntity] | None = None,
    ) -> None: ...
