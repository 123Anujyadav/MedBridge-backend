"""
Persistence models for the AI Medical Case Intake Agent audit trail.

Two tables, deliberately separate from `cases`:

* `intake_sessions` — a durable record of each intake conversation, retained
  after the ephemeral Redis session expires.
* `intake_extracted_entities` — per-entity provenance (value, confidence,
  evidence quote, and whether it was accepted or rejected for fabrication).

The clinical record lives in `cases`/`symptoms` and is what clinicians read.
These tables exist so a compliance reviewer can reconstruct exactly what the
model claimed, what it was allowed to keep, and why.
"""

import uuid

from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, Index, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base


class IntakeSessionRecord(Base):
    """Durable record of one AI intake conversation."""

    __tablename__ = "intake_sessions"
    __table_args__ = (
        Index("idx_intake_session_patient_status", "patient_user_id", "status"),
        CheckConstraint(
            "status IN ('collecting', 'awaiting_doctor_selection', 'routed', "
            "'emergency_escalated', 'abandoned')",
            name="intake_session_status_check",
        ),
    )

    session_key: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    """The public session id issued to the client (distinct from the row PK)."""

    patient_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    status: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    language: Mapped[str] = mapped_column(String(20), default="unknown", nullable=False)
    intent: Mapped[str] = mapped_column(String(30), default="unclear", nullable=False)

    followup_rounds: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    overall_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    red_flags: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    transcript: Mapped[str] = mapped_column(Text, default="", nullable=False)
    """Patient turns only. Agent questions are excluded, matching the domain rule."""

    medical_case_snapshot: Mapped[dict] = mapped_column(JSON, nullable=True)
    """The structured case exactly as generated, before any clinician edits."""

    routed_case_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("cases.id", ondelete="SET NULL"), nullable=True, index=True
    )
    routed_doctor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("doctors.id", ondelete="SET NULL"), nullable=True, index=True
    )

    entities = relationship(
        "IntakeExtractedEntity",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class IntakeExtractedEntity(Base):
    """
    One extracted clinical entity, with the provenance the accuracy rules require.

    Rejected extractions are stored too (`was_accepted=False`). Keeping the
    fabrications is the point: it is the only way to measure how often the model
    tries to invent clinical data, and it turns the grounding check into an
    auditable control rather than an invisible filter.
    """

    __tablename__ = "intake_extracted_entities"
    __table_args__ = (
        Index("idx_intake_entity_session_kind", "session_id", "kind"),
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="intake_entity_confidence_check",
        ),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("intake_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    kind: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    value: Mapped[str] = mapped_column(String(500), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    confidence_band: Mapped[str] = mapped_column(
        String(20), default="unknown", nullable=False
    )

    evidence_quote: Mapped[str] = mapped_column(Text, default="", nullable=False)
    """Verbatim patient text this value was drawn from."""

    evidence_turn_index: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )

    was_accepted: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    """False when the entity failed evidence grounding and was discarded."""

    session = relationship("IntakeSessionRecord", back_populates="entities")
