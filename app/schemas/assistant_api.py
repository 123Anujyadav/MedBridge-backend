"""
Pydantic v2 schemas for the AI Medical Assistant API.

Response shapes are built to drop straight into the existing React page state:
`structuredData` is the `AIResponseData` object the cards already render, and
the sidebar fields mirror the page's existing `useState` variables exactly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.assistant.domain.entities import ChatMessage, Conversation


class SendMessageRequest(BaseModel):
    """A patient turn. Omit `conversation_id` to start a new thread."""

    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=64)


class AssistantMessageResponse(BaseModel):
    """One assistant turn, shaped for the existing `MessageType`."""

    id: str
    sender: Literal["ai", "user"]
    text: str
    timestamp: str
    structured_data: dict[str, Any] = Field(default_factory=dict)
    symptoms_detected: list[str] = Field(default_factory=list)
    specialist_suggested: str | None = None
    references: list[str] = Field(default_factory=list)


class SendMessageResponse(BaseModel):
    """
    Everything the page needs after one exchange.

    The sidebar fields are returned alongside the message so the frontend can
    update all of its state from a single response without extra round trips.
    """

    conversation_id: str
    title: str
    message: AssistantMessageResponse

    conversation_summary: str | None = None
    detected_symptoms: list[str] = Field(default_factory=list)
    suggested_specialist: str | None = None
    medical_references: list[str] = Field(default_factory=list)
    emergency_risk: Literal["normal", "moderate", "critical"] = "normal"

    degraded: bool = Field(
        default=False,
        description="True when the answer was produced without a working model call.",
    )

    ai_confidence: int | None = Field(
        default=None,
        description=(
            "The model's self-reported confidence for this turn, 0-100. Null when "
            "the turn was not scored (a deterministic emergency reply or a "
            "degraded answer), so the UI can show nothing rather than a number."
        ),
    )

    @classmethod
    def build(
        cls, conversation: Conversation, message: ChatMessage
    ) -> SendMessageResponse:
        answer = message.answer
        # Exposed because the panel's confidence card had no value to render and
        # displayed a hardcoded 94% whenever any symptom was detected.
        confidence: int | None = None
        if answer and not answer.degraded and answer.confidence > 0:
            confidence = round(min(1.0, max(0.0, answer.confidence)) * 100)

        return cls(
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            message=AssistantMessageResponse(
                id=message.message_id,
                sender="ai",
                text=message.text,
                timestamp=message.created_at,
                structured_data=answer.to_ui_payload() if answer else {},
                symptoms_detected=answer.symptoms if answer else [],
                specialist_suggested=(
                    answer.specialist.name if answer and answer.specialist else None
                ),
                references=answer.references if answer else [],
            ),
            conversation_summary=conversation.summary or None,
            detected_symptoms=list(conversation.known_symptoms),
            suggested_specialist=conversation.last_specialist,
            medical_references=list(conversation.references),
            emergency_risk=conversation.emergency_risk.value,
            degraded=bool(answer.degraded) if answer else False,
            ai_confidence=confidence,
        )


class ConversationTurnResponse(BaseModel):
    """A stored turn, for replaying a conversation from history."""

    id: str
    sender: Literal["user", "ai"]
    text: str
    timestamp: str
    structured_data: dict[str, Any] = Field(default_factory=dict)


class ConversationDetailResponse(BaseModel):
    """Full transcript of one conversation."""

    conversation_id: str
    title: str
    status: str
    summary: str | None = None
    language: str
    emergency_risk: str
    detected_symptoms: list[str] = Field(default_factory=list)
    suggested_specialist: str | None = None
    medical_references: list[str] = Field(default_factory=list)
    messages: list[ConversationTurnResponse] = Field(default_factory=list)
    created_at: str
    updated_at: str

    @classmethod
    def build(cls, conversation: Conversation) -> ConversationDetailResponse:
        return cls(
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            status=conversation.status.value,
            summary=conversation.summary or None,
            language=conversation.language.value,
            emergency_risk=conversation.emergency_risk.value,
            detected_symptoms=list(conversation.known_symptoms),
            suggested_specialist=conversation.last_specialist,
            medical_references=list(conversation.references),
            messages=[
                ConversationTurnResponse(
                    id=m.message_id,
                    sender=m.role.value,
                    text=m.text,
                    timestamp=m.created_at,
                    structured_data=(
                        m.answer.to_ui_payload() if m.answer else {}
                    ),
                )
                for m in conversation.messages
            ],
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )


class ConversationSummaryResponse(BaseModel):
    """List item for the existing chat-history drawer."""

    conversation_id: str
    title: str
    preview: str
    language: str
    emergency_risk: str
    symptoms: list[str] = Field(default_factory=list)
    specialist: str | None = None
    message_count: int = 0
    updated_at: str

    @classmethod
    def build(cls, conversation: Conversation) -> ConversationSummaryResponse:
        return cls(
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            preview=(conversation.summary or "")[:180],
            language=conversation.language.value,
            emergency_risk=conversation.emergency_risk.value,
            symptoms=list(conversation.known_symptoms)[:6],
            specialist=conversation.last_specialist,
            message_count=conversation.message_count or len(conversation.messages),
            updated_at=conversation.updated_at,
        )


class AssistantHealthResponse(BaseModel):
    """Dependency health for the assistant subsystem."""

    status: Literal["healthy", "degraded", "unhealthy"]
    llm: dict[str, Any]
    environment: dict[str, Any]
    retrieval_available: bool
    guardrails_enabled: bool
