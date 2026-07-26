"""
Pydantic v2 schemas for the AI Medical Case Intake Agent API.

HTTP-boundary types only. The domain layer uses plain dataclasses; these models
handle request validation and response shaping, and are the contract the
frontend codes against.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.intake.application.dto import SessionView
from app.intake.domain.entities import IntakeSession

# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


class StartIntakeRequest(BaseModel):
    """Opening message that begins an intake conversation."""

    symptoms: str = Field(
        min_length=1,
        max_length=4000,
        description="The patient's own description, in any supported language.",
    )
    age: str | None = Field(default=None, max_length=10)
    gender: str | None = Field(default=None, max_length=20)


class SubmitAnswerRequest(BaseModel):
    """Reply to the agent's outstanding follow-up question."""

    answer: str = Field(min_length=1, max_length=4000)


class SelectDoctorRequest(BaseModel):
    """Chosen specialist, finalising the intake."""

    doctor_id: uuid.UUID


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


class ConfidenceResponse(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    band: Literal["high", "medium", "low", "unknown"]


class EvidenceResponse(BaseModel):
    quote: str
    turn_index: int


class ExtractedEntityResponse(BaseModel):
    """One extracted clinical fact with its provenance."""

    kind: str
    value: str
    confidence: ConfidenceResponse
    evidence: EvidenceResponse
    is_unknown: bool


class ConversationTurnResponse(BaseModel):
    role: Literal["patient", "agent"]
    text: str
    timestamp: str


class SpecialistRecommendationResponse(BaseModel):
    specialty: str
    rationale: str
    match_score: float
    doctor_id: str | None = None
    doctor_name: str | None = None
    hospital_name: str | None = None
    is_available: bool = True
    avatar_url: str | None = None
    """The clinician's profile photo, when they have set one."""


class MedicalCaseResponse(BaseModel):
    """
    The structured case.

    `differential_considerations` are conditions for a clinician to evaluate and
    rule out. They are explicitly not diagnoses.
    """

    chief_complaint: str
    symptoms: list[str] = []
    duration: str
    severity: str
    onset: str
    body_sites: list[str] = []
    allergies: list[str] = []
    current_medications: list[str] = []
    medical_history: list[str] = []
    aggravating_factors: list[str] = []
    relieving_factors: list[str] = []
    urgency: Literal["low", "medium", "high", "critical"]
    red_flags: list[str] = []
    differential_considerations: list[str] = []
    missing_information: list[str] = []
    recommended_specialty: str
    overall_confidence: ConfidenceResponse
    patient_language: str
    summary_for_doctor: str
    generated_at: str


class IntakeSessionResponse(BaseModel):
    """Full state of an intake session."""

    session_id: str
    status: Literal[
        "collecting",
        "awaiting_doctor_selection",
        "routed",
        "emergency_escalated",
        "abandoned",
    ]
    language: str
    intent: str
    followup_rounds: int
    pending_question: str | None = None
    red_flags: list[str] = []
    turns: list[ConversationTurnResponse] = []
    entities: list[ExtractedEntityResponse] = []
    medical_case: MedicalCaseResponse | None = None
    recommendations: list[SpecialistRecommendationResponse] = []
    routed_case_id: str | None = None
    routed_doctor_id: str | None = None
    created_at: str
    updated_at: str

    is_emergency: bool = Field(
        default=False,
        description="True when intake was halted for emergency escalation.",
    )
    awaiting_input: bool = Field(
        default=False,
        description="True when the agent is waiting on a patient answer.",
    )
    rejected_extraction_count: int = Field(
        default=0,
        description=(
            "Extractions discarded this turn for citing text the patient never "
            "said. Non-zero indicates the model attempted to fabricate data."
        ),
    )
    degraded: bool = Field(
        default=False,
        description="True when a model call failed and a fallback was used.",
    )
    notices: list[str] = []

    @classmethod
    def from_view(cls, view: SessionView) -> IntakeSessionResponse:
        return cls.from_session(
            view.session,
            rejected_extraction_count=view.rejected_extraction_count,
            degraded=view.degraded,
            notices=view.notices,
        )

    @classmethod
    def from_session(
        cls,
        session: IntakeSession,
        *,
        rejected_extraction_count: int = 0,
        degraded: bool = False,
        notices: list[str] | None = None,
    ) -> IntakeSessionResponse:
        payload: dict[str, Any] = session.to_dict()
        payload["is_emergency"] = session.status.value == "emergency_escalated"
        payload["awaiting_input"] = bool(session.pending_question)
        payload["rejected_extraction_count"] = rejected_extraction_count
        payload["degraded"] = degraded
        payload["notices"] = list(notices or [])
        return cls.model_validate(payload)


class RoutingResponse(BaseModel):
    """Result of persisting and routing a completed case."""

    session_id: str
    case_id: str
    doctor_id: str
    doctor_name: str
    specialty: str
    urgency: str
    message: str = "Case routed to the selected specialist."


class IntakeHealthResponse(BaseModel):
    """Readiness of the intake agent's dependencies."""

    status: Literal["healthy", "degraded", "unhealthy"]
    llm: dict[str, Any]
    graph_nodes: int
    max_followup_rounds: int
    min_overall_confidence: float
