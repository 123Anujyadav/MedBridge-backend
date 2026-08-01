"""
Persistence models for the AI Medical Assistant.

Two tables following the existing conventions in `app/models/` — UUID primary
keys, audit timestamps and soft delete inherited from `Base`, CHECK constraints
for enum-like columns.

The clinical `cases` table is untouched: an assistant conversation is an
information exchange, not a routed clinical case, and conflating them would put
un-reviewed AI output into the doctor portal's case queue.
"""

import uuid

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base


class AssistantConversation(Base):
    """One patient's assistant thread."""

    __tablename__ = "assistant_conversations"
    __table_args__ = (
        Index("idx_assistant_conv_patient_updated", "patient_user_id", "updated_at"),
        CheckConstraint(
            "status IN ('active', 'archived')", name="assistant_conv_status_check"
        ),
        CheckConstraint(
            "emergency_risk IN ('normal', 'moderate', 'critical')",
            name="assistant_conv_risk_check",
        ),
    )

    conversation_key: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    """Public conversation id issued to the client (distinct from the row PK)."""

    patient_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    title: Mapped[str] = mapped_column(String(200), default="New consultation")
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    language: Mapped[str] = mapped_column(String(20), default="english")
    emergency_risk: Mapped[str] = mapped_column(String(20), default="normal")
    last_specialist: Mapped[str] = mapped_column(String(150), nullable=True)

    known_symptoms: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    """Accumulated across turns so the assistant does not re-ask."""

    asked_questions: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    """Follow-ups already put to the patient; used to suppress duplicates."""

    references: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    messages = relationship(
        "AssistantMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="AssistantMessage.created_at",
    )

# the assistant message class is for the message log that will be stored in the database that will be used for the AI Medical Assistant
class AssistantMessage(Base):
    """One turn, with the structured payload the UI rendered."""

    __tablename__ = "assistant_messages"
    __table_args__ = (
        Index("idx_assistant_msg_conversation", "conversation_id", "created_at"),
        CheckConstraint("role IN ('user', 'ai')", name="assistant_msg_role_check"),
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="assistant_msg_confidence_check",
        ),
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("assistant_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    message_key: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    text: Mapped[str] = mapped_column(Text, default="", nullable=False)

    structured: Mapped[dict] = mapped_column(JSON, nullable=True)
    """The exact `AIResponseData` payload sent to the frontend cards."""

    entities: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    """Extracted clinical entities with confidence and verbatim evidence."""

    intent: Mapped[str] = mapped_column(String(30), default="unclear")
    urgency: Mapped[str] = mapped_column(String(20), default="Low")
    knowledge_source: Mapped[str] = mapped_column(String(20), default="model_only")
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    """True when the answer was produced without a working model call."""

    conversation = relationship("AssistantConversation", back_populates="messages")
