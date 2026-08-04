import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime

from app.db.base_class import Base


# ── order lifecycle ──────────────────────────────────────────────────────
#
# The transition table is the authority, not a convention. Statuses drive money
# and physical goods, so "can this order move from X to Y" is answered in one
# place and enforced in the service rather than re-derived at each call site.

ORDER_RECEIVED = "received"
ORDER_PREPARING = "preparing"
ORDER_PACKED = "packed"
ORDER_OUT_FOR_DELIVERY = "out_for_delivery"
ORDER_DELIVERED = "delivered"
ORDER_CANCELLED = "cancelled"

ORDER_STATUSES = (
    ORDER_RECEIVED,
    ORDER_PREPARING,
    ORDER_PACKED,
    ORDER_OUT_FOR_DELIVERY,
    ORDER_DELIVERED,
    ORDER_CANCELLED,
)

ORDER_TRANSITIONS: dict[str, tuple[str, ...]] = {
    ORDER_RECEIVED: (ORDER_PREPARING, ORDER_CANCELLED),
    ORDER_PREPARING: (ORDER_PACKED, ORDER_CANCELLED),
    ORDER_PACKED: (ORDER_OUT_FOR_DELIVERY, ORDER_CANCELLED),
    # Dispatch is the point of no return: the goods have left the counter, so
    # a "cancellation" after this is a return, which is a different process
    # with different money attached.
    ORDER_OUT_FOR_DELIVERY: (ORDER_DELIVERED,),
    ORDER_DELIVERED: (),
    ORDER_CANCELLED: (),
}

CANCELLABLE_STATUSES = tuple(
    status for status, nxt in ORDER_TRANSITIONS.items() if ORDER_CANCELLED in nxt
)


class MedicineOrder(Base):
    """
    A patient's order against a prescription, fulfilled by one pharmacy.

    Money is stored on the order rather than recomputed from inventory on read:
    prices change, and an invoice must show what was actually charged on the
    day, not what the same basket would cost today.
    """

    __tablename__ = "medicine_orders"
    __table_args__ = (
        CheckConstraint(
            "status IN ('received', 'preparing', 'packed', 'out_for_delivery', "
            "'delivered', 'cancelled')",
            name="medicine_order_status_check",
        ),
        CheckConstraint("total >= 0", name="medicine_order_total_non_negative"),
        Index("ix_medicine_orders_patient_status", "patient_id", "status"),
    )

    order_number: Mapped[str] = mapped_column(
        String(24), unique=True, index=True, nullable=False
    )
    """Human-quotable reference, e.g. MB-7F3A2C91. Not the UUID."""

    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    prescription_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("prescriptions.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    """
    RESTRICT, not CASCADE. A dispensing record must outlive edits to the
    prescription it came from — deleting the prescription must not silently
    erase evidence that medicines were supplied.
    """

    pharmacy_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pharmacies.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    pharmacy_name: Mapped[str] = mapped_column(String(200), nullable=False)

    status: Mapped[str] = mapped_column(
        String(30), default=ORDER_RECEIVED, nullable=False, index=True
    )

    subtotal: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    discount_total: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    delivery_fee: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="INR", nullable=False)

    delivery_address: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    delivery_latitude: Mapped[float] = mapped_column(Float, nullable=True)
    delivery_longitude: Mapped[float] = mapped_column(Float, nullable=True)
    delivery_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)

    distance_km: Mapped[float] = mapped_column(Float, nullable=True)
    eta_minutes: Mapped[int] = mapped_column(Integer, nullable=True)
    estimated_delivery_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    delivery_partner_name: Mapped[str] = mapped_column(String(150), nullable=True)
    delivery_partner_phone: Mapped[str] = mapped_column(String(50), nullable=True)

    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    dispatched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    delivered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    cancellation_reason: Mapped[str] = mapped_column(String(500), nullable=True)

    fulfilment_provider: Mapped[str] = mapped_column(
        String(40), default="local_db", nullable=False
    )
    """
    Which adapter owns this order — `local_db` today.

    Recorded per order rather than read from configuration, so orders placed
    before an external provider is switched on keep resolving through the
    adapter that actually created them.
    """

    patient = relationship("Patient")
    prescription = relationship("Prescription")
    pharmacy = relationship("Pharmacy", back_populates="orders")
    items = relationship(
        "MedicineOrderItem", back_populates="order", cascade="all, delete-orphan"
    )
    events = relationship(
        "OrderStatusEvent",
        back_populates="order",
        cascade="all, delete-orphan",
        order_by="OrderStatusEvent.created_at",
    )

    @property
    def is_cancellable(self) -> bool:
        return self.status in CANCELLABLE_STATUSES

    @property
    def is_terminal(self) -> bool:
        return not ORDER_TRANSITIONS.get(self.status, ())

    def can_transition_to(self, target: str) -> bool:
        return target in ORDER_TRANSITIONS.get(self.status, ())


class MedicineOrderItem(Base):
    """
    One line of an order.

    Product details are copied rather than referenced. An inventory row can be
    repriced, renamed or deleted; the invoice must still render years later
    exactly as it was issued.
    """

    __tablename__ = "medicine_order_items"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="order_item_quantity_positive"),
        CheckConstraint("line_total >= 0", name="order_item_total_non_negative"),
    )

    order_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("medicine_orders.id", ondelete="CASCADE"), nullable=False, index=True
    )

    inventory_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pharmacy_inventory.id", ondelete="SET NULL"), nullable=True
    )
    medication_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("medications.id", ondelete="SET NULL"), nullable=True
    )
    """The prescription line this fulfils, when it maps to one."""

    medicine_name: Mapped[str] = mapped_column(String(200), nullable=False)
    generic_name: Mapped[str] = mapped_column(String(200), nullable=True)
    brand_name: Mapped[str] = mapped_column(String(200), nullable=True)
    strength: Mapped[str] = mapped_column(String(100), nullable=True)
    rxcui: Mapped[str] = mapped_column(String(20), nullable=True)

    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    mrp: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    discount_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    line_total: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    is_generic_substitute: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    substituted_for: Mapped[str] = mapped_column(String(200), nullable=True)
    """
    The prescribed product this generic replaced.

    A substitution is only ever made when the patient explicitly accepts it —
    never silently — and this records what they agreed to.
    """

    order = relationship("MedicineOrder", back_populates="items")


class OrderStatusEvent(Base):
    """
    Append-only trail of every status the order passed through.

    Kept as events rather than a mutable `status_history` column so tracking is
    reconstructable: the current status alone cannot answer when an order was
    packed, or who cancelled it.
    """

    __tablename__ = "order_status_events"

    order_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("medicine_orders.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    note: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    actor_type: Mapped[str] = mapped_column(String(30), default="system", nullable=False)
    actor_id: Mapped[uuid.UUID] = mapped_column(nullable=True)

    order = relationship("MedicineOrder", back_populates="events")
