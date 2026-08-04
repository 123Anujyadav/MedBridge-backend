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
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime

from app.db.base_class import Base

# ── verification lifecycle ───────────────────────────────────────────────
#
# Onboarding is a review process, not a boolean. Each state is a distinct thing
# an administrator can be looking at, and the transition table is the authority
# on what may follow what — the same shape the order lifecycle uses.

VERIFICATION_PENDING = "pending"
VERIFICATION_SUBMITTED = "submitted"
VERIFICATION_DOCUMENT_REVIEW = "document_review"
VERIFICATION_APPROVED = "approved"
VERIFICATION_REJECTED = "rejected"
VERIFICATION_SUSPENDED = "suspended"

VERIFICATION_STATUSES = (
    VERIFICATION_PENDING,
    VERIFICATION_SUBMITTED,
    VERIFICATION_DOCUMENT_REVIEW,
    VERIFICATION_APPROVED,
    VERIFICATION_REJECTED,
    VERIFICATION_SUSPENDED,
)

VERIFICATION_TRANSITIONS: dict[str, tuple[str, ...]] = {
    VERIFICATION_PENDING: (VERIFICATION_SUBMITTED, VERIFICATION_REJECTED),
    VERIFICATION_SUBMITTED: (VERIFICATION_DOCUMENT_REVIEW, VERIFICATION_REJECTED),
    VERIFICATION_DOCUMENT_REVIEW: (VERIFICATION_APPROVED, VERIFICATION_REJECTED),
    # Approval is not final: a licence lapses, a complaint lands. Suspension is
    # reversible back to approved; rejection sends the applicant back to the
    # start rather than pretending the earlier submission still stands.
    VERIFICATION_APPROVED: (VERIFICATION_SUSPENDED, VERIFICATION_REJECTED),
    VERIFICATION_SUSPENDED: (VERIFICATION_APPROVED, VERIFICATION_REJECTED),
    VERIFICATION_REJECTED: (VERIFICATION_PENDING,),
}

DOCUMENT_TYPES = (
    "drug_license",
    "gst_certificate",
    "pan_card",
    "business_registration",
    "store_image",
    "owner_id",
    "pharmacist_certificate",
    "digital_signature",
)

DOCUMENT_STATUSES = ("uploaded", "under_review", "approved", "rejected", "expired")


class Pharmacy(Base):
    """
    A dispensing partner in the MedBridge network.

    Two kinds of row live here and the difference matters:

    * **Onboarded partners** — `is_partner=True`. MedBridge holds their
      inventory, so stock, price and ordering all work.
    * **Discovered places** — `is_partner=False`, populated from Google Places
      so a patient can still see and route to a nearby chemist. These have no
      inventory and cannot take an order; `can_fulfil` is the guard.

    Conflating the two would let the UI offer "Order now" against a shop that
    has never heard of MedBridge.
    """

    __tablename__ = "pharmacies"
    __table_args__ = (
        CheckConstraint("rating >= 0.0 AND rating <= 5.0", name="pharmacy_rating_check"),
        CheckConstraint(
            "latitude >= -90.0 AND latitude <= 90.0", name="pharmacy_latitude_check"
        ),
        CheckConstraint(
            "longitude >= -180.0 AND longitude <= 180.0", name="pharmacy_longitude_check"
        ),
        Index("ix_pharmacies_geo", "latitude", "longitude"),
    )

    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    address: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    city: Mapped[str] = mapped_column(String(120), nullable=True)
    postal_code: Mapped[str] = mapped_column(String(20), nullable=True)
    phone: Mapped[str] = mapped_column(String(50), nullable=True)

    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)

    google_place_id: Mapped[str] = mapped_column(
        String(255), nullable=True, unique=True, index=True
    )
    """
    Set for rows discovered through Places.

    Unique so repeated discovery sweeps update the existing row rather than
    accumulating a duplicate of the same shop on every search.
    """

    licence_number: Mapped[str] = mapped_column(String(100), nullable=True)
    rating: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_ratings: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    is_partner: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_24x7: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    opens_at: Mapped[str] = mapped_column(String(5), nullable=True)   # "09:00"
    closes_at: Mapped[str] = mapped_column(String(5), nullable=True)  # "22:00"

    delivers: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    delivery_radius_km: Mapped[float] = mapped_column(Float, default=8.0, nullable=False)
    delivery_fee: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    free_delivery_above: Mapped[float] = mapped_column(Float, nullable=True)
    min_order_value: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    avg_prep_minutes: Mapped[int] = mapped_column(Integer, default=15, nullable=False)
    """Counter time before a rider leaves. Feeds the delivery estimate."""

    # ── business identity ────────────────────────────────────────────────
    owner_name: Mapped[str] = mapped_column(String(200), nullable=True)
    business_name: Mapped[str] = mapped_column(String(250), nullable=True)
    gst_number: Mapped[str] = mapped_column(String(20), nullable=True, index=True)
    drug_license_number: Mapped[str] = mapped_column(String(100), nullable=True, index=True)
    drug_license_expiry: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    email: Mapped[str] = mapped_column(String(255), nullable=True)
    whatsapp: Mapped[str] = mapped_column(String(50), nullable=True)
    emergency_phone: Mapped[str] = mapped_column(String(50), nullable=True)

    # ── branding ─────────────────────────────────────────────────────────
    logo_url: Mapped[str] = mapped_column(String(500), nullable=True)
    banner_url: Mapped[str] = mapped_column(String(500), nullable=True)
    store_images: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    # ── fulfilment options ───────────────────────────────────────────────
    express_delivery: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    express_delivery_radius_km: Mapped[float] = mapped_column(Float, nullable=True)
    pickup_available: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    holiday_dates: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    """ISO dates the store is closed regardless of `opens_at`/`closes_at`."""

    # ── settlement ───────────────────────────────────────────────────────
    upi_id: Mapped[str] = mapped_column(String(120), nullable=True)
    bank_account_name: Mapped[str] = mapped_column(String(200), nullable=True)
    bank_account_number: Mapped[str] = mapped_column(String(60), nullable=True)
    bank_ifsc: Mapped[str] = mapped_column(String(20), nullable=True)
    platform_commission_percent: Mapped[float] = mapped_column(
        Float, default=0.0, nullable=False
    )

    # ── verification ─────────────────────────────────────────────────────
    verification_status: Mapped[str] = mapped_column(
        String(30), default=VERIFICATION_PENDING, nullable=False, index=True
    )
    verification_notes: Mapped[str] = mapped_column(Text, nullable=True)
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    verified_by: Mapped[uuid.UUID] = mapped_column(nullable=True)
    rejection_reason: Mapped[str] = mapped_column(String(500), nullable=True)
    suspended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    suspension_reason: Mapped[str] = mapped_column(String(500), nullable=True)

    inventory = relationship(
        "PharmacyInventory", back_populates="pharmacy", cascade="all, delete-orphan"
    )
    orders = relationship("MedicineOrder", back_populates="pharmacy")
    documents = relationship(
        "PharmacyDocument", back_populates="pharmacy", cascade="all, delete-orphan"
    )
    verification_events = relationship(
        "PharmacyVerificationEvent",
        back_populates="pharmacy",
        cascade="all, delete-orphan",
        order_by="PharmacyVerificationEvent.created_at",
    )

    @property
    def can_fulfil(self) -> bool:
        """
        Whether an order may be placed here at all. Unchanged from Phase 2.

        Verification deliberately does *not* appear as a third condition here.
        Approval is what grants `is_partner`, and rejection or suspension is
        what withdraws it — so the dispensing gate stays a single fact rather
        than two that can disagree. Adding `verification_status == approved`
        alongside would also have silently stopped every pharmacy onboarded
        before this module existed, since they predate the column.
        """
        return bool(self.is_partner and self.is_active)


class PharmacyInventory(Base):
    """
    One stocked product at one pharmacy.

    Joined on `rxcui` wherever possible. Matching a prescription to stock by
    drug name does not work — "Crocin", "Paracetamol" and "Acetaminophen" are
    one ingredient under three labels — so the RxNorm concept id resolved
    during safety verification is the real key. `medicine_name` is retained for
    display and for the fallback path when a prescription line never resolved.
    """

    __tablename__ = "pharmacy_inventory"
    __table_args__ = (
        UniqueConstraint(
            "pharmacy_id", "sku", name="uq_pharmacy_inventory_pharmacy_sku"
        ),
        CheckConstraint("stock_quantity >= 0", name="inventory_stock_non_negative"),
        CheckConstraint("mrp >= 0 AND selling_price >= 0", name="inventory_price_non_negative"),
        CheckConstraint(
            "discount_percent >= 0 AND discount_percent <= 100",
            name="inventory_discount_range",
        ),
        Index("ix_pharmacy_inventory_rxcui", "rxcui"),
        Index("ix_pharmacy_inventory_lookup", "pharmacy_id", "rxcui"),
    )

    pharmacy_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pharmacies.id", ondelete="CASCADE"), nullable=False, index=True
    )

    sku: Mapped[str] = mapped_column(String(64), nullable=False)
    """The pharmacy's own product code. Unique per pharmacy."""

    rxcui: Mapped[str] = mapped_column(String(20), nullable=True)
    medicine_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    generic_name: Mapped[str] = mapped_column(String(200), nullable=True, index=True)
    brand_name: Mapped[str] = mapped_column(String(200), nullable=True)
    manufacturer: Mapped[str] = mapped_column(String(200), nullable=True)

    strength: Mapped[str] = mapped_column(String(100), nullable=True)
    form: Mapped[str] = mapped_column(String(50), nullable=True)  # tablet, syrup, ...
    pack_size: Mapped[str] = mapped_column(String(50), nullable=True)

    is_generic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    requires_prescription: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    mrp: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    selling_price: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    discount_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    stock_quantity: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    low_stock_threshold: Mapped[int] = mapped_column(Integer, default=10, nullable=False)
    restock_expected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    stock_synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    """
    When the partner last confirmed this figure.

    Surfaced to the caller so a stale count is visibly stale. A quantity nobody
    has touched for a week is a guess, and presenting it as live availability is
    how a patient ends up at a counter that has none.
    """

    # ── catalogue detail ─────────────────────────────────────────────────
    composition: Mapped[str] = mapped_column(String(500), nullable=True)
    drug_schedule: Mapped[str] = mapped_column(String(20), nullable=True)
    """Regulatory schedule — H, H1, X, OTC. Drives dispensing restrictions."""

    category: Mapped[str] = mapped_column(String(120), nullable=True, index=True)
    barcode: Mapped[str] = mapped_column(String(64), nullable=True, index=True)
    storage_instructions: Mapped[str] = mapped_column(String(300), nullable=True)

    # ── batch and shelf life ─────────────────────────────────────────────
    batch_number: Mapped[str] = mapped_column(String(60), nullable=True)
    manufacturing_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expiry_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    # ── replenishment thresholds ─────────────────────────────────────────
    min_stock: Mapped[int] = mapped_column(Integer, nullable=True)
    max_stock: Mapped[int] = mapped_column(Integer, nullable=True)
    reorder_level: Mapped[int] = mapped_column(Integer, nullable=True)

    gst_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    pharmacy = relationship("Pharmacy", back_populates="inventory")

    @property
    def availability(self) -> str:
        """
        `available` | `limited` | `out_of_stock`. Unchanged from Phase 2.

        Expiry deliberately does not appear here. This property is what the
        patient-facing search reads, and quietly folding a new condition into
        it would change what Phase 2 dispenses. Shelf life is exposed
        separately through `stock_state`, which the admin console uses.
        """
        if self.stock_quantity <= 0:
            return "out_of_stock"
        if self.stock_quantity <= self.low_stock_threshold:
            return "limited"
        return "available"

    def can_supply(self, quantity: int) -> bool:
        return self.stock_quantity >= max(1, quantity)

    # ── admin-facing stock health ────────────────────────────────────────

    NEAR_EXPIRY_DAYS = 90

    @property
    def stock_state(self) -> str:
        """
        Richer state for the admin console:
        `expired` | `near_expiry` | `out_of_stock` | `critical` | `low` | `available`.

        Expiry outranks quantity — a full shelf of expired stock is a problem to
        act on, not availability. Read only by the admin module; the patient
        path continues to use `availability`.
        """
        from datetime import datetime as _dt, timedelta, timezone as _tz

        if self.expiry_date:
            now = _dt.now(_tz.utc)
            expiry = self.expiry_date
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=_tz.utc)
            if expiry <= now:
                return "expired"
            if expiry <= now + timedelta(days=self.NEAR_EXPIRY_DAYS):
                return "near_expiry"

        if self.stock_quantity <= 0:
            return "out_of_stock"
        if self.reorder_level is not None and self.stock_quantity <= self.reorder_level:
            return "critical"
        if self.stock_quantity <= self.low_stock_threshold:
            return "low"
        return "available"

    @property
    def inventory_value(self) -> float:
        """Stock at cost-to-sell. Feeds the inventory-value analytic."""
        return round((self.selling_price or self.mrp or 0.0) * self.stock_quantity, 2)


class PharmacyDocument(Base):
    """
    A compliance document supporting a pharmacy's verification.

    `expires_at` is the load-bearing field: a drug licence that lapsed last
    month is not a valid document, and the expiry sweep is what turns an
    approved pharmacy back into one needing review.
    """

    __tablename__ = "pharmacy_documents"
    __table_args__ = (
        CheckConstraint(
            "status IN ('uploaded', 'under_review', 'approved', 'rejected', 'expired')",
            name="pharmacy_document_status_check",
        ),
        Index("ix_pharmacy_documents_expiry", "expires_at"),
    )

    pharmacy_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pharmacies.id", ondelete="CASCADE"), nullable=False, index=True
    )

    doc_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    file_url: Mapped[str] = mapped_column(String(500), nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    document_number: Mapped[str] = mapped_column(String(120), nullable=True)

    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    status: Mapped[str] = mapped_column(String(30), default="uploaded", nullable=False)
    review_notes: Mapped[str] = mapped_column(String(500), nullable=True)
    reviewed_by: Mapped[uuid.UUID] = mapped_column(nullable=True)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    pharmacy = relationship("Pharmacy", back_populates="documents")

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        from datetime import datetime as _dt, timezone as _tz

        expiry = self.expires_at
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=_tz.utc)
        return expiry <= _dt.now(_tz.utc)

    @property
    def days_to_expiry(self) -> int | None:
        if not self.expires_at:
            return None
        from datetime import datetime as _dt, timezone as _tz

        expiry = self.expires_at
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=_tz.utc)
        return (expiry - _dt.now(_tz.utc)).days


class PharmacyVerificationEvent(Base):
    """
    Append-only trail of the verification workflow.

    Stored as events rather than mutating a status column alone, because
    "who approved this pharmacy, when, and on what note" is a compliance
    question that the current status cannot answer.
    """

    __tablename__ = "pharmacy_verification_events"

    pharmacy_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pharmacies.id", ondelete="CASCADE"), nullable=False, index=True
    )

    from_status: Mapped[str] = mapped_column(String(30), nullable=True)
    to_status: Mapped[str] = mapped_column(String(30), nullable=False)
    note: Mapped[str] = mapped_column(String(1000), default="", nullable=False)

    actor_id: Mapped[uuid.UUID] = mapped_column(nullable=True)
    actor_name: Mapped[str] = mapped_column(String(200), default="", nullable=False)

    pharmacy = relationship("Pharmacy", back_populates="verification_events")
