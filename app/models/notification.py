import uuid
from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base_class import Base

class NotificationItem(Base):
    """
    User notifications and workflow alerts.

    A notification carries the clinical context it refers to — case, patient —
    by foreign key rather than by parsing it back out of the message text, so a
    card can link straight into the right workflow and can be scoped correctly.

    `dedupe_key` is what stops the same event producing the same card twice when
    a request is retried or two code paths observe the same change.
    """
    __tablename__ = "notifications"
    __table_args__ = (
        CheckConstraint(
            "priority IN ('low', 'medium', 'high', 'urgent', 'critical')",
            name="notification_priority_check",
        ),
        CheckConstraint(
            "category IN ('case', 'ai', 'appointment', 'report', 'prescription', "
            "'patient', 'system', 'security', 'general')",
            name="notification_category_check",
        ),
        # The centre reads "my unread, newest first"; this serves it directly.
        Index("idx_notification_user_read", "user_id", "read", "created_at"),
    )


    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    type: Mapped[str] = mapped_column(String(50), nullable=False) # appointment, medication, report, emergency, etc.
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(String(500), nullable=False)
    timestamp: Mapped[str] = mapped_column(String(50), nullable=False) # ISO String

    read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    priority: Mapped[str] = mapped_column(String(50), default="low") # low, medium, high, urgent, critical

    action_url: Mapped[str] = mapped_column(String(255), nullable=True)
    action_label: Mapped[str] = mapped_column(String(100), nullable=True)

    # ── Routing, grouping and clinical context ───────────────────────────

    category: Mapped[str] = mapped_column(
        String(20), default="general", nullable=False, index=True
    )
    """Filter bucket for the notification centre."""

    case_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("cases.id", ondelete="SET NULL"), nullable=True, index=True
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="SET NULL"), nullable=True, index=True
    )
    """
    Both nullable. A system or security alert has no case and no patient, and is
    left unlinked rather than attached to a plausible-looking one.
    """

    patient_name: Mapped[str] = mapped_column(String(200), nullable=True)
    """Denormalised for the card, so listing does not join to patients."""

    archived: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )

    group_key: Mapped[str] = mapped_column(String(80), nullable=True, index=True)
    """Similar notifications collapse under this, e.g. `report.uploaded`."""

    dedupe_key: Mapped[str] = mapped_column(String(160), nullable=True, index=True)
    """
    Identifies the underlying event. A second notification with the same key for
    the same user is suppressed, so a retried request cannot double-notify.
    """

    delivered_at: Mapped[str] = mapped_column(String(50), nullable=True)
    read_at: Mapped[str] = mapped_column(String(50), nullable=True)

    # Relationships
    user = relationship("User", back_populates="notifications")
