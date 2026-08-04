import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime

from app.db.base_class import Base

# ── partner verification ─────────────────────────────────────────────────

PARTNER_PENDING = "pending"
PARTNER_DOCUMENT_REVIEW = "document_review"
PARTNER_APPROVED = "approved"
PARTNER_REJECTED = "rejected"
PARTNER_SUSPENDED = "suspended"

PARTNER_STATUSES = (
    PARTNER_PENDING,
    PARTNER_DOCUMENT_REVIEW,
    PARTNER_APPROVED,
    PARTNER_REJECTED,
    PARTNER_SUSPENDED,
)

PARTNER_TRANSITIONS: dict[str, tuple[str, ...]] = {
    PARTNER_PENDING: (PARTNER_DOCUMENT_REVIEW, PARTNER_REJECTED),
    PARTNER_DOCUMENT_REVIEW: (PARTNER_APPROVED, PARTNER_REJECTED),
    PARTNER_APPROVED: (PARTNER_SUSPENDED, PARTNER_REJECTED),
    PARTNER_SUSPENDED: (PARTNER_APPROVED, PARTNER_REJECTED),
    PARTNER_REJECTED: (PARTNER_PENDING,),
}

# ── assignment lifecycle ─────────────────────────────────────────────────
#
# A rider's journey is finer-grained than the order's. `medicine_orders` moves
# packed → out_for_delivery → delivered; the assignment tracks every leg in
# between so the patient's tracking screen and the rider's app can both be
# specific without changing what Phase 2 dispenses.

DELIVERY_OFFERED = "offered"
DELIVERY_ACCEPTED = "accepted"
DELIVERY_EN_ROUTE_PICKUP = "en_route_pickup"
DELIVERY_AT_PHARMACY = "at_pharmacy"
DELIVERY_PICKED_UP = "picked_up"
DELIVERY_OUT_FOR_DELIVERY = "out_for_delivery"
DELIVERY_AT_PATIENT = "at_patient"
DELIVERY_DELIVERED = "delivered"
DELIVERY_CANCELLED = "cancelled"
DELIVERY_FAILED = "failed"

DELIVERY_STATUSES = (
    DELIVERY_OFFERED,
    DELIVERY_ACCEPTED,
    DELIVERY_EN_ROUTE_PICKUP,
    DELIVERY_AT_PHARMACY,
    DELIVERY_PICKED_UP,
    DELIVERY_OUT_FOR_DELIVERY,
    DELIVERY_AT_PATIENT,
    DELIVERY_DELIVERED,
    DELIVERY_CANCELLED,
    DELIVERY_FAILED,
)

DELIVERY_TRANSITIONS: dict[str, tuple[str, ...]] = {
    DELIVERY_OFFERED: (DELIVERY_ACCEPTED, DELIVERY_CANCELLED),
    DELIVERY_ACCEPTED: (DELIVERY_EN_ROUTE_PICKUP, DELIVERY_CANCELLED),
    DELIVERY_EN_ROUTE_PICKUP: (DELIVERY_AT_PHARMACY, DELIVERY_CANCELLED, DELIVERY_FAILED),
    DELIVERY_AT_PHARMACY: (DELIVERY_PICKED_UP, DELIVERY_CANCELLED, DELIVERY_FAILED),
    # Once the medicine is in the rider's hands it cannot simply be cancelled —
    # the goods are out of the pharmacy's control, so the only exits are
    # completing the delivery or failing it, which triggers a return.
    DELIVERY_PICKED_UP: (DELIVERY_OUT_FOR_DELIVERY, DELIVERY_FAILED),
    DELIVERY_OUT_FOR_DELIVERY: (DELIVERY_AT_PATIENT, DELIVERY_FAILED),
    DELIVERY_AT_PATIENT: (DELIVERY_DELIVERED, DELIVERY_FAILED),
    DELIVERY_DELIVERED: (),
    DELIVERY_CANCELLED: (),
    DELIVERY_FAILED: (),
}

TERMINAL_STATUSES = (DELIVERY_DELIVERED, DELIVERY_CANCELLED, DELIVERY_FAILED)

# Statuses in which a rider is considered busy. Used to stop a second
# assignment landing on someone already carrying an order.
ACTIVE_STATUSES = tuple(s for s in DELIVERY_STATUSES if s not in TERMINAL_STATUSES)

VEHICLE_TYPES = ("bicycle", "motorcycle", "scooter", "car", "van", "on_foot")


class DeliveryPartner(Base):
    """
    A rider's operating profile.

    Separate from `users` for the same reason `doctors` and `patients` are: the
    account is identity, this is the professional record — vehicle, licence,
    verification, rating and lifetime statistics.
    """

    __tablename__ = "delivery_partners"
    __table_args__ = (
        CheckConstraint(
            "verification_status IN ('pending', 'document_review', 'approved', "
            "'rejected', 'suspended')",
            name="delivery_partner_verification_check",
        ),
        CheckConstraint("rating >= 0.0 AND rating <= 5.0", name="delivery_partner_rating_check"),
        Index("ix_delivery_partners_availability", "is_online", "verification_status"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    """
    The account this rider signs in with.

    Unique: one profile per account, so `user.id` resolves to exactly one rider
    and there is no ambiguity about who accepted a job.
    """

    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    phone: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    photo_url: Mapped[str] = mapped_column(String(500), nullable=True)
    date_of_birth: Mapped[str] = mapped_column(String(50), nullable=True)
    address: Mapped[str] = mapped_column(String(500), nullable=True)
    city: Mapped[str] = mapped_column(String(120), nullable=True, index=True)

    vehicle_type: Mapped[str] = mapped_column(String(30), nullable=True)
    vehicle_number: Mapped[str] = mapped_column(String(30), nullable=True, index=True)
    vehicle_model: Mapped[str] = mapped_column(String(120), nullable=True)

    driving_licence_number: Mapped[str] = mapped_column(String(60), nullable=True)
    driving_licence_expiry: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    documents: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    """[{"type": "licence", "url": "...", "expires_at": "..."}]"""

    verification_status: Mapped[str] = mapped_column(
        String(30), default=PARTNER_PENDING, nullable=False, index=True
    )
    verification_notes: Mapped[str] = mapped_column(Text, nullable=True)
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    verified_by: Mapped[uuid.UUID] = mapped_column(nullable=True)
    suspension_reason: Mapped[str] = mapped_column(String(500), nullable=True)

    is_online: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    """Whether the rider is currently accepting work. Their own toggle."""

    current_latitude: Mapped[float] = mapped_column(Float, nullable=True)
    current_longitude: Mapped[float] = mapped_column(Float, nullable=True)
    location_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    """
    Surfaced to the patient's tracking screen alongside the timestamp, so a
    stale fix is visibly stale rather than presented as the rider's position
    now.
    """

    experience_years: Mapped[int] = mapped_column(Integer, nullable=True)
    rating: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_ratings: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_deliveries: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_deliveries: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_distance_km: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_earnings: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    assignments = relationship(
        "DeliveryAssignment", back_populates="partner", cascade="all, delete-orphan"
    )

    @property
    def is_approved(self) -> bool:
        return self.verification_status == PARTNER_APPROVED

    @property
    def can_accept_work(self) -> bool:
        """Approved and clocked on. Both, or the rider gets no offers."""
        return bool(self.is_approved and self.is_online)

    @property
    def completion_rate(self) -> float:
        attempted = self.completed_deliveries + self.failed_deliveries
        if not attempted:
            return 0.0
        return round(self.completed_deliveries / attempted, 4)


class DeliveryAssignment(Base):
    """
    One order handed to one rider.

    Carries its own status machine, the delivery OTP and the proof of delivery.
    The parent `medicine_orders` row keeps the coarse status Phase 2 defined —
    this record is what makes the leg-by-leg journey answerable.
    """

    __tablename__ = "delivery_assignments"
    __table_args__ = (
        CheckConstraint(
            "status IN ('offered', 'accepted', 'en_route_pickup', 'at_pharmacy', "
            "'picked_up', 'out_for_delivery', 'at_patient', 'delivered', "
            "'cancelled', 'failed')",
            name="delivery_assignment_status_check",
        ),
        CheckConstraint("otp_attempts >= 0", name="delivery_otp_attempts_check"),
        Index("ix_delivery_assignments_partner_status", "partner_id", "status"),
        Index("ix_delivery_assignments_order", "order_id"),
    )

    order_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("medicine_orders.id", ondelete="CASCADE"), nullable=False
    )
    partner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("delivery_partners.id", ondelete="RESTRICT"), nullable=False
    )
    """
    RESTRICT: a completed delivery is the record of who handed medicine to whom.
    Removing the rider must not erase it.
    """

    pharmacy_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pharmacies.id", ondelete="RESTRICT"), nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(30), default=DELIVERY_OFFERED, nullable=False, index=True
    )

    pickup_address: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    pickup_latitude: Mapped[float] = mapped_column(Float, nullable=True)
    pickup_longitude: Mapped[float] = mapped_column(Float, nullable=True)

    drop_address: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    drop_latitude: Mapped[float] = mapped_column(Float, nullable=True)
    drop_longitude: Mapped[float] = mapped_column(Float, nullable=True)

    distance_km: Mapped[float] = mapped_column(Float, nullable=True)
    eta_minutes: Mapped[int] = mapped_column(Integer, nullable=True)
    estimated_arrival_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    delivery_fee: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    partner_earning: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # ── OTP ──────────────────────────────────────────────────────────────
    otp_hash: Mapped[str] = mapped_column(String(255), nullable=True)
    """
    Only the hash is stored. A rider who could read the code off the API would
    not need the patient to be present, which defeats the entire point of
    confirming handover.
    """

    otp_issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    otp_verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    otp_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # ── proof of delivery ────────────────────────────────────────────────
    proof_photo_url: Mapped[str] = mapped_column(String(500), nullable=True)
    proof_signature_url: Mapped[str] = mapped_column(String(500), nullable=True)
    delivery_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    proof_latitude: Mapped[float] = mapped_column(Float, nullable=True)
    proof_longitude: Mapped[float] = mapped_column(Float, nullable=True)
    proof_captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    offered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    accepted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    picked_up_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    delivered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    failure_reason: Mapped[str] = mapped_column(String(500), nullable=True)

    assigned_by: Mapped[uuid.UUID] = mapped_column(nullable=True)
    """Administrator who assigned it, when it was not self-served."""

    partner = relationship("DeliveryPartner", back_populates="assignments")
    order = relationship("MedicineOrder")
    events = relationship(
        "DeliveryEvent",
        back_populates="assignment",
        cascade="all, delete-orphan",
        order_by="DeliveryEvent.created_at",
    )

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def is_otp_verified(self) -> bool:
        return self.otp_verified_at is not None

    def can_transition_to(self, target: str) -> bool:
        return target in DELIVERY_TRANSITIONS.get(self.status, ())


class DeliveryEvent(Base):
    """
    Append-only trail of the journey, with the position it happened at.

    Kept as events rather than a mutable status so "where was the rider when
    they marked this picked up" stays answerable — the question that matters
    when a delivery is disputed.
    """

    __tablename__ = "delivery_events"

    assignment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("delivery_assignments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    note: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=True)
    longitude: Mapped[float] = mapped_column(Float, nullable=True)
    actor_type: Mapped[str] = mapped_column(String(30), default="partner", nullable=False)
    actor_id: Mapped[uuid.UUID] = mapped_column(nullable=True)

    assignment = relationship("DeliveryAssignment", back_populates="events")
