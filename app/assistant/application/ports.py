"""
Ports for the AI Medical Assistant.

Structural `Protocol`s so infrastructure adapters and test fakes satisfy them by
shape. Every port that touches an external system is required to degrade rather
than raise — a model outage or an unavailable vector store must not cost the
patient their message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.assistant.domain.entities import AssistantAnswer, Conversation


@dataclass(frozen=True, slots=True)
class RetrievedSnippet:
    """One piece of grounding context returned by knowledge retrieval."""

    text: str
    source: str = ""
    score: float = 0.0


@runtime_checkable
class AssistantLLMPort(Protocol):
    """A model that returns structured JSON."""

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Return a parsed JSON object, or `{}` when unavailable. Must not raise."""
        ...

    async def complete_text(
        self,
        *,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 600,
        temperature: float = 0.2,
    ) -> str:
        """Return plain text, or `""` when unavailable. Must not raise."""
        ...

    async def health(self, *, probe: bool = False) -> dict[str, Any]:
        """
        Report provider readiness.

        `probe=True` requests a real model call, which is the only way to detect
        a credential that is present but rejected.
        """
        ...

    @property
    def last_error(self) -> str | None:
        """Why the most recent call returned nothing, or None on success."""
        ...


@runtime_checkable
class KnowledgeRetrievalPort(Protocol):
    """
    Grounding context for an answer.

    Implementations may consult a vector store, a web search API, or nothing at
    all; the pipeline treats an empty list as "answer from model knowledge".
    """

    async def retrieve(self, query: str, *, limit: int = 4) -> list[RetrievedSnippet]:
        ...

    @property
    def is_available(self) -> bool: ...


@runtime_checkable
class GuardrailsPort(Protocol):
    """Input/output safety filtering."""

    async def check_input(self, text: str) -> tuple[bool, str]:
        """`(allowed, message)`. When blocked, `message` is shown to the patient."""
        ...

    async def check_output(self, output: str, *, user_input: str = "") -> str:
        """Return the response, revised if the safety pass required it."""
        ...


@runtime_checkable
class ConversationRepositoryPort(Protocol):
    """Persistence for assistant conversations and their messages."""

    async def get(
        self, conversation_id: str, patient_user_id: str
    ) -> Conversation | None: ...

    async def save(self, conversation: Conversation) -> None: ...

    async def list_for_patient(
        self, patient_user_id: str, *, limit: int = 50
    ) -> list[Conversation]: ...

    async def delete(self, conversation_id: str, patient_user_id: str) -> bool: ...


@runtime_checkable
class AssistantPipelinePort(Protocol):
    """The AI orchestration graph."""

    async def run(
        self, conversation: Conversation, user_text: str
    ) -> AssistantAnswer:
        """Produce one structured answer. Must not raise."""
        ...
