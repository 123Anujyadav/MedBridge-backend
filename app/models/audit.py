import uuid
from sqlalchemy import CheckConstraint, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base_class import Base

class AuditLog(Base):
    """
    HIPAA-compliant System Audit Log model.
    Logs user access, reads, and modifications of Protected Health Information (PHI).

    This table is also the case timeline's event store. Rather than introducing a
    second, parallel history table, the clinical dimensions a timeline needs —
    which case, which patient, who acted, what field moved and from what to what
    — are recorded here alongside the compliance fields.

    The table is append-only. A database trigger rejects UPDATE and DELETE, so
    an entry cannot be rewritten or removed even by code holding a session.
    """
    __tablename__ = "audit_logs"
    __table_args__ = (
        CheckConstraint("status IN ('success', 'failed', 'warning')", name="audit_log_status_check"),
        CheckConstraint(
            "actor_type IN ('patient', 'doctor', 'ai', 'admin', 'system')",
            name="audit_log_actor_type_check",
        ),
        # The timeline reads by case, newest first; this serves that directly.
        Index("idx_audit_case_created", "case_id", "created_at"),
    )


    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    user_name: Mapped[str] = mapped_column(String(200), nullable=False)
    user_role: Mapped[str] = mapped_column(String(50), nullable=False)
    
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True) # e.g., READ_PATIENT_RECORD, UPDATE_PRESCRIPTION
    resource: Mapped[str] = mapped_column(String(100), nullable=False) # e.g., Patient, Prescription
    resource_id: Mapped[str] = mapped_column(String(100), nullable=False) # UUID of target record
    
    ip_address: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="success", index=True) # success, failed, warning
    details: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # ── Clinical timeline dimensions ─────────────────────────────────────

    case_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("cases.id", ondelete="SET NULL"), nullable=True, index=True
    )
    """
    The case this event belongs to. NULL for events with no case context —
    those are simply absent from every case timeline rather than being attached
    to a plausible-looking one.
    """

    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="SET NULL"), nullable=True, index=True
    )

    actor_type: Mapped[str] = mapped_column(
        String(20), default="system", nullable=False, index=True
    )
    """patient | doctor | ai | admin | system. Always derived server-side."""

    event_type: Mapped[str] = mapped_column(String(60), nullable=True, index=True)
    """Stable semantic key, e.g. `case.created`, `ai.summary_generated`."""

    field_changed: Mapped[str] = mapped_column(String(80), nullable=True)
    previous_value: Mapped[str] = mapped_column(Text, nullable=True)
    new_value: Mapped[str] = mapped_column(Text, nullable=True)
    """
    Before/after for a single field.

    Left NULL when an event does not change a tracked value — an empty string
    would read as "changed to blank", which is a different claim.
    """

    reason: Mapped[str] = mapped_column(Text, nullable=True)

    # Relationships
    user = relationship("User", back_populates="audit_logs")
