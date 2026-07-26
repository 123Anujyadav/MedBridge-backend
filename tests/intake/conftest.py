"""
Shared fixtures and test doubles for the Medical Case Intake Agent suite.

No test in this package makes a network call. `ScriptedLLM` stands in for the
model so every scenario — including fabrication attempts and total provider
failure — is deterministic and reproducible.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.intake.application.dto import DoctorRef
from app.intake.domain.entities import IntakeSession
from app.intake.domain.enums import TurnRole

# Stage identification, keyed off a distinctive phrase in each system prompt.
_STAGE_MARKERS: tuple[tuple[str, str], ...] = (
    ("extraction", "clinical intake extraction engine"),
    ("intent", "classify a patient's message"),
    ("language", "identify the language"),
    ("followup", "collecting missing information"),
    ("case", "clinical documentation engine"),
)

DEFAULT_STAGE_RESPONSES: dict[str, Any] = {
    "intent": {"intent": "symptom_report", "confidence": 0.92},
    "language": {"language": "english", "confidence": 0.9},
    "followup": {
        "question": "How long have you been feeling this way?",
        "targets": "duration",
    },
    "case": {
        "chief_complaint": "Patient-reported symptoms requiring review",
        "differential_considerations": ["Condition to rule out"],
        "recommended_specialty": "General Medicine",
        "urgency": "medium",
        "summary_for_doctor": "Structured handover summary for clinician review.",
    },
    "extraction": {"entities": []},
}


class ScriptedLLM:
    """
    Deterministic `LLMPort` double.

    Responses are configured per workflow stage. `calls` records every stage
    invoked, so tests can assert on control flow — for example that the
    emergency path never reaches the extraction or follow-up stages.
    """

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.responses: dict[str, Any] = {**DEFAULT_STAGE_RESPONSES, **(responses or {})}
        self.calls: list[str] = []

    @staticmethod
    def _stage_for(system_prompt: str) -> str:
        lowered = system_prompt.casefold()
        for stage, marker in _STAGE_MARKERS:
            if marker in lowered:
                return stage
        return "unknown"

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 1200,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        stage = self._stage_for(system_prompt)
        self.calls.append(stage)
        value = self.responses.get(stage, {})
        if isinstance(value, Exception):
            raise value
        return value

    async def health(self) -> dict[str, Any]:
        return {"status": "healthy", "provider": "scripted"}


class DeadLLM:
    """`LLMPort` double simulating total provider failure."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete_json(self, **_: Any) -> dict[str, Any]:
        self.calls += 1
        return {}

    async def health(self) -> dict[str, Any]:
        return {"status": "unhealthy", "provider": "dead", "error": "unreachable"}


class FakeDoctorDirectory:
    """In-memory `DoctorDirectoryPort` double."""

    def __init__(self, doctors: list[DoctorRef] | None = None) -> None:
        self.doctors = doctors if doctors is not None else [_default_doctor()]
        self.queried: list[str] = []

    async def find_for_specialty(
        self, specialty: str, *, limit: int = 3
    ) -> list[DoctorRef]:
        self.queried.append(specialty)
        matches = [
            d for d in self.doctors if d.specialty.casefold() == specialty.casefold()
        ]
        return matches[:limit]

    async def get(self, doctor_id: str) -> DoctorRef | None:
        return next((d for d in self.doctors if d.doctor_id == doctor_id), None)


class InMemorySessionStore:
    """In-memory `SessionStorePort` double."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}

    async def get(self, session_id: str) -> IntakeSession | None:
        raw = self.store.get(session_id)
        return IntakeSession.from_dict(raw) if raw else None

    async def save(
        self, session: IntakeSession, ttl_seconds: int | None = None
    ) -> None:
        self.store[session.session_id] = session.to_dict()

    async def delete(self, session_id: str) -> None:
        self.store.pop(session_id, None)


def _default_doctor() -> DoctorRef:
    return DoctorRef(
        doctor_id="11111111-1111-1111-1111-111111111111",
        full_name="Dr. Test Cardiologist",
        specialty="Cardiology",
        hospital_name="Test General Hospital",
        rating=4.8,
        years_of_experience=15,
        is_available=True,
        is_verified=True,
    )


def extraction(*items: tuple[str, str, float, str]) -> dict[str, Any]:
    """
    Build an extraction payload.

    Each item is `(kind, value, confidence, evidence_quote)`.
    """
    return {
        "entities": [
            {"kind": k, "value": v, "confidence": c, "evidence": e}
            for k, v, c, e in items
        ]
    }


def complete_extraction() -> dict[str, Any]:
    """Extraction satisfying every mandatory field, grounded in COMPLETE_TEXT."""
    return extraction(
        ("symptom", "chest discomfort", 0.93, "chest discomfort"),
        ("duration", "3 days", 0.91, "for 3 days"),
        ("severity", "moderate", 0.85, "moderate"),
    )


COMPLETE_TEXT = "I have had moderate chest discomfort for 3 days now."
"""Transcript that `complete_extraction()` quotes are grounded against."""


def make_session(text: str = COMPLETE_TEXT, user_id: str = "user-1") -> IntakeSession:
    session = IntakeSession(patient_user_id=user_id)
    session.add_turn(TurnRole.PATIENT, text)
    return session


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def dead_llm() -> DeadLLM:
    return DeadLLM()


@pytest.fixture
def fake_doctors() -> FakeDoctorDirectory:
    return FakeDoctorDirectory()


@pytest.fixture
def memory_sessions() -> InMemorySessionStore:
    return InMemorySessionStore()
