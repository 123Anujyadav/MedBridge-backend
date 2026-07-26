"""
Application-layer data transfer objects.

Boundary types between controllers and use cases. Deliberately not Pydantic:
HTTP validation belongs in `app/schemas/intake_api.py`, and use cases should be
callable from a worker or a test without dragging request models along.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.intake.domain.entities import IntakeSession


@dataclass(frozen=True, slots=True)
class StartIntakeCommand:
    """Open a new intake session from the patient's first description."""

    patient_user_id: str
    text: str
    age: str | None = None
    gender: str | None = None


@dataclass(frozen=True, slots=True)
class SubmitAnswerCommand:
    """Answer the agent's outstanding follow-up question."""

    patient_user_id: str
    session_id: str
    text: str


@dataclass(frozen=True, slots=True)
class DoctorRef:
    """A candidate specialist, as seen by the application layer."""

    doctor_id: str
    full_name: str
    specialty: str
    hospital_name: str | None = None
    rating: float = 0.0
    years_of_experience: int = 0
    is_available: bool = True
    is_verified: bool = False
    avatar_url: str | None = None
    """Profile photo, so the patient chooses a clinician they can see."""


@dataclass(frozen=True, slots=True)
class RoutingResult:
    """Outcome of persisting and routing a completed case."""

    session_id: str
    case_id: str
    doctor_id: str
    doctor_name: str
    specialty: str
    urgency: str


@dataclass(slots=True)
class WorkflowResult:
    """
    Outcome of one pass of the intake graph.

    Carries the advanced session plus the diagnostics from that pass, so the
    application layer can surface degradation and rejected extractions without
    re-inspecting graph internals.
    """

    session: IntakeSession
    notices: list[str] = field(default_factory=list)
    rejected_count: int = 0
    degraded: bool = False


@dataclass(slots=True)
class SessionView:
    """
    Read model returned to the presentation layer.

    Wraps the session with the derived flags a client needs to drive the UI,
    so controllers never have to re-derive domain logic.
    """

    session: IntakeSession
    rejected_extraction_count: int = 0
    notices: list[str] = field(default_factory=list)
    degraded: bool = False

    @classmethod
    def from_workflow(cls, result: WorkflowResult) -> SessionView:
        return cls(
            session=result.session,
            rejected_extraction_count=result.rejected_count,
            notices=list(result.notices),
            degraded=result.degraded,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.session.to_dict()
        payload["rejected_extraction_count"] = self.rejected_extraction_count
        payload["notices"] = list(self.notices)
        payload["degraded"] = self.degraded
        return payload
