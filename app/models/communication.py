import uuid
from datetime import datetime

from sqlalchemy import (
    JSON, CheckConstraint, DateTime, Float, ForeignKey, Integer, String, Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base

COMMUNICATION_CHANNELS = ("voice", "sms", "whatsapp")

COMMUNICATION_STATUSES = (
    "queued",       # accepted by us, waiting for its first attempt
    "sending",      # claimed by a worker, request in flight
    "accepted",     # the provider took it but has not moved it yet
    "sent",         # the provider says it left their network
    "delivered",    # the handset confirmed it / the call completed
    "undelivered",  # the provider tried and the carrier refused it
    "canceled",     # withdrawn before delivery
    "failed",       # every attempt was used up, or the provider gave up
    "skipped",      # never attempted — no channel configured, or no number
)
"""
The delivery lifecycle, ours and the provider's merged into one vocabulary.

`accepted` is deliberately distinct from `sent`: a provider acknowledging a
request is not the same as a network having carried it, and an emergency
dashboard that showed the first as the second would tell a patient somebody had
been reached when nobody had. `undelivered` and `canceled` are likewise kept
apart from `failed` — a carrier refusing a number needs different action from a
transient outage.
"""

TERMINAL_COMMUNICATION_STATUSES = (
    "delivered", "undelivered", "canceled", "failed", "skipped",
)

RECIPIENT_ROLES = ("emergency_contact", "doctor", "admin")

_CHANNEL_SQL = ", ".join(f"'{c}'" for c in COMMUNICATION_CHANNELS)
_STATUS_SQL = ", ".join(f"'{s}'" for s in COMMUNICATION_STATUSES)
_ROLE_SQL = ", ".join(f"'{r}'" for r in RECIPIENT_ROLES)


class CommunicationLog(Base):
    """
    One attempt to reach one person, on one channel, about one emergency.

    A row is written *before* the provider is called, not after. That ordering
    is the whole point: if the process dies mid-request, or the vendor times
    out, the record of what was owed to whom still exists and the retry sweep
    picks it up. A log written only on success would lose exactly the messages
    worth knowing about.

    Rows are never deleted or overwritten by a retry — `attempts` counts up on
    the same row, and the last provider response replaces the previous one, so
    the sequence for an emergency reads as one line per intended contact.
    """

    __tablename__ = "communication_logs"
    __table_args__ = (
        CheckConstraint(f"channel IN ({_CHANNEL_SQL})", name="communication_channel_check"),
        CheckConstraint(f"status IN ({_STATUS_SQL})", name="communication_status_check"),
        CheckConstraint(f"recipient_role IN ({_ROLE_SQL})", name="communication_recipient_role_check"),
        CheckConstraint("attempts >= 0", name="communication_attempts_check"),
    )

    emergency_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("emergency_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    channel: Mapped[str] = mapped_column(String(20), nullable=False)
    recipient_role: Mapped[str] = mapped_column(String(30), nullable=False)
    recipient_name: Mapped[str] = mapped_column(String(200), nullable=True)

    recipient_phone: Mapped[str] = mapped_column(String(32), nullable=True)
    """
    The number dialled, stored in full.

    Responders need to know which number was actually tried when somebody was
    not reached. It is masked on the way out by the response schema rather than
    at rest, so the audit record stays complete while the API does not hand a
    third party's telephone number to every screen that lists an emergency.
    """

    status: Mapped[str] = mapped_column(String(20), nullable=False,
                                        default="queued", index=True)

    provider: Mapped[str] = mapped_column(String(30), nullable=True, default="twilio")
    provider_sid: Mapped[str] = mapped_column(String(64), nullable=True, index=True)
    """The vendor's identifier — a Twilio Call or Message SID — for reconciliation."""

    provider_status: Mapped[str] = mapped_column(String(40), nullable=True)
    error_code: Mapped[str] = mapped_column(String(40), nullable=True)
    error_message: Mapped[str] = mapped_column(String(500), nullable=True)

    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    """
    When this row becomes eligible again. Null once it is finished.

    Indexed together with `status` because the retry sweep's only query is
    "queued rows whose time has come", and it runs every thirty seconds.
    """

    template_key: Mapped[str] = mapped_column(String(60), nullable=True)
    body_preview: Mapped[str] = mapped_column(Text, nullable=True)
    """
    What was actually sent.

    Kept so an incident review can see the words a responder received rather
    than re-deriving them from a template that may since have changed.
    """

    provider_events: Mapped[list] = mapped_column(
        JSON, nullable=False, default=list,
    )
    """
    Every callback the provider has sent about this attempt, in order.

    Appended to, never replaced. `status` holds where the attempt has got to;
    this holds how it got there, which is what an incident review needs when a
    message was accepted, then sent, then refused by a carrier forty seconds
    later. Nothing is synthesised — an entry exists only because the provider
    reported it.
    """

    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=True)

    emergency = relationship("EmergencyRequest", back_populates="communications")
