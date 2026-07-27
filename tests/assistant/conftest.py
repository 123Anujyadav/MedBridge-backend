"""
Fixtures and doubles for the AI Medical Assistant suite.

No test here makes a network call: the LLM is faked at the port boundary so
every scenario — including total provider failure — is deterministic.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.assistant.application.ports import RetrievedSnippet

# Stage identification, keyed off a phrase unique to each system prompt.
_MARKERS: tuple[tuple[str, str], ...] = (
    ("intent", "you classify a patient's message"),
    ("entities", "you extract clinical entities"),
    ("answer", "enterprise medical information assistant"),
    ("guard_in", "strict content safety classifier"),
    ("guard_out", "medical response safety reviewer"),
)


def answer_payload(**overrides: Any) -> dict[str, Any]:
    """A well-formed answer payload; override any field per test."""
    payload: dict[str, Any] = {
        "replyText": "Here is what I understand about your symptoms.",
        "summary": "You report a headache lasting three days.",
        "symptoms": ["headache", "nausea"],
        "causes": [
            {"cause": "Tension headache", "confidence": "High"},
            {"cause": "Dehydration", "confidence": "Medium"},
        ],
        "actions": ["Rest in a dark room", "Drink water regularly"],
        "lifestyleAdvice": [
            {"category": "Hydration", "description": "Drink 2-3 litres daily."}
        ],
        "medicines": [
            {
                "name": "Paracetamol",
                "purpose": "Pain relief",
                "sideEffects": ["Nausea"],
                "precautions": "Do not exceed 3g per day.",
                "type": "OTC",
            }
        ],
        "specialist": {
            "name": "Neurology",
            "reason": "Persistent headache",
            "priority": "Recommended",
        },
        "urgency": {"level": "Low", "explanation": "Routine management."},
        "references": ["WHO Headache Guidelines"],
        "followUpQuestions": ["How long have you had this?"],
        "conversationTitle": "Headache consultation",
        "confidence": 0.86,
    }
    payload.update(overrides)
    return payload


def entity_payload(*items: tuple[str, str, float, str]) -> dict[str, Any]:
    """Build an extraction payload from `(kind, value, confidence, evidence)`."""
    return {
        "entities": [
            {"kind": k, "value": v, "confidence": c, "evidence": e}
            for k, v, c, e in items
        ]
    }


class ScriptedLLM:
    """Deterministic `AssistantLLMPort` double."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.responses: dict[str, Any] = {
            "intent": {"intent": "symptom_report", "confidence": 0.9},
            "entities": {"entities": []},
            "answer": answer_payload(),
            "guard_in": "SAFE",
            "guard_out": "",
            **(responses or {}),
        }
        self.calls: list[str] = []
        self.prompts: list[str] = []

    @staticmethod
    def _stage(system_prompt: str, user_content: str) -> str:
        blob = f"{system_prompt}\n{user_content}".casefold()
        for stage, marker in _MARKERS:
            if marker in blob:
                return stage
        return "unknown"

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        stage = self._stage(system_prompt, user_content)
        self.calls.append(stage)
        self.prompts.append(f"{system_prompt}\n{user_content}")
        value = self.responses.get(stage, {})
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, dict) else {}

    async def complete_text(
        self,
        *,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 600,
        temperature: float = 0.2,
    ) -> str:
        stage = self._stage(system_prompt, user_content)
        self.calls.append(stage)
        value = self.responses.get(stage, "")
        if isinstance(value, Exception):
            raise value
        return str(value)

    async def health(self, *, probe: bool = False) -> dict[str, Any]:
        return {"status": "healthy", "provider": "scripted", "probe": probe}

    @property
    def last_error(self) -> str | None:
        return None


class DeadLLM:
    """Simulates total provider failure."""

    async def complete_json(self, **_: Any) -> dict[str, Any]:
        return {}

    async def complete_text(self, **_: Any) -> str:
        return ""

    async def health(self, *, probe: bool = False) -> dict[str, Any]:
        return {"status": "unhealthy", "provider": "dead", "error": "unreachable"}

    @property
    def last_error(self) -> str | None:
        return "simulated provider outage"


class FakeRetriever:
    """`KnowledgeRetrievalPort` double."""

    def __init__(self, snippets: list[RetrievedSnippet] | None = None) -> None:
        self.snippets = snippets or []
        self.queries: list[str] = []

    @property
    def is_available(self) -> bool:
        return bool(self.snippets)

    async def retrieve(self, query: str, *, limit: int = 4) -> list[RetrievedSnippet]:
        self.queries.append(query)
        return self.snippets[:limit]


class ExplodingRetriever:
    """Retriever that always fails, to prove retrieval errors are contained."""

    @property
    def is_available(self) -> bool:
        return True

    async def retrieve(self, query: str, *, limit: int = 4):
        raise RuntimeError("vector store unreachable")


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def dead_llm() -> DeadLLM:
    return DeadLLM()


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever()
