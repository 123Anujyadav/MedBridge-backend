"""
Graph state for the assistant pipeline.

The `Conversation` aggregate travels through the graph directly so nodes share
one source of truth for memory; everything else is per-run bookkeeping.
"""

from __future__ import annotations

from typing import TypedDict

from app.assistant.application.ports import KnowledgeRetrievalPort, RetrievedSnippet
from app.assistant.domain.entities import AssistantAnswer, Conversation, MedicalEntity
from app.assistant.domain.enums import IntentType


class AssistantState(TypedDict, total=False):
    """State passed between assistant pipeline nodes."""

    conversation: Conversation
    """The thread being continued. Carries memory across turns."""

    user_text: str
    """The sanitised message for this turn."""

    retriever: KnowledgeRetrievalPort
    """
    Request-scoped knowledge retrieval.

    Travels in state so the compiled graph stays a process-level singleton.
    """

    intent: IntentType
    entities: list[MedicalEntity]
    snippets: list[RetrievedSnippet]
    red_flags: list[str]
    answer: AssistantAnswer
    degraded: bool
    rejected_entities: int
