"""Wire format for the pharmacy administration module."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

VerificationStatus = Literal[
    "pending", "submitted", "document_review", "approved", "rejected", "suspended"
]
DocumentType = Literal[
    "drug_license", "gst_certificate", "pan_card", "business_registration",
    "store_image", "owner_id", "pharmacist_certificate", "digital_signature",
]
DocumentStatus = Literal["uploaded", "under_review", "approved", "rejected", "expired"]
StockState = Literal[
    "available", "low", "critical", "out_of_stock", "expired", "near_expiry"
]


# ── pharmacy ─────────────────────────────────────────────────────────────


class PharmacyBase(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    address: str = Field(default="", max_length=500)
    city: Optional[str] = Field(None, max_length=120)
    postal_code: Optional[str] = Field(None, max_length=20)
    phone: Optional[str] = Field(None, max_length=50)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    owner_name: Optional[str] = Field(None, max_length=200)
    business_name: Optional[str] = Field(None, max_length=250)
    gst_number: Optional[str] = Field(None, max_length=20)
    drug_license_number: Optional[str] = Field(None, max_length=100)
    drug_license_expiry: Optional[datetime] = None
    licence_number: Optional[str] = Field(None, max_length=100)

    email: Optional[str] = Field(None, max_length=255)
    whatsapp: Optional[str] = Field(None, max_length=50)
    emergency_phone: Optional[str] = Field(None, max_length=50)

    logo_url: Optional[str] = Field(None, max_length=500)
    banner_url: Optional[str] = Field(None, max_length=500)
    store_images: List[str] = Field(default_factory=list)

    is_24x7: bool = False
    opens_at: Optional[str] = Field(None, max_length=5)
    closes_at: Optional[str] = Field(None, max_length=5)
    holiday_dates: List[str] = Field(default_factory=list)

    delivers: bool = True
    express_delivery: bool = False
    express_delivery_radius_km: Optional[float] = Field(None, ge=0, le=100)
    pickup_available: bool = False
    delivery_radius_km: float = Field(default=8.0, ge=0, le=100)
    delivery_fee: float = Field(default=0.0, ge=0)
    free_delivery_above: Optional[float] = Field(None, ge=0)
    min_order_value: float = Field(default=0.0, ge=0)
    avg_prep_minutes: int = Field(default=15, ge=0, le=600)

    upi_id: Optional[str] = Field(None, max_length=120)
    bank_account_name: Optional[str] = Field(None, max_length=200)
    bank_account_number: Optional[str] = Field(None, max_length=60)
    bank_ifsc: Optional[str] = Field(None, max_length=20)
    platform_commission_percent: float = Field(default=0.0, ge=0, le=100)


class PharmacyCreateRequest(PharmacyBase):
    """A new pharmacy always starts unverified; the workflow grants partnership."""


class PharmacyUpdateRequest(BaseModel):
    """Every field optional — a PATCH-style partial update."""

    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    address: Optional[str] = Field(None, max_length=500)
    city: Optional[str] = Field(None, max_length=120)
    postal_code: Optional[str] = Field(None, max_length=20)
    phone: Optional[str] = Field(None, max_length=50)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    owner_name: Optional[str] = Field(None, max_length=200)
    business_name: Optional[str] = Field(None, max_length=250)
    gst_number: Optional[str] = Field(None, max_length=20)
    drug_license_number: Optional[str] = Field(None, max_length=100)
    drug_license_expiry: Optional[datetime] = None
    email: Optional[str] = Field(None, max_length=255)
    whatsapp: Optional[str] = Field(None, max_length=50)
    emergency_phone: Optional[str] = Field(None, max_length=50)
    logo_url: Optional[str] = Field(None, max_length=500)
    banner_url: Optional[str] = Field(None, max_length=500)
    store_images: Optional[List[str]] = None
    is_24x7: Optional[bool] = None
    opens_at: Optional[str] = Field(None, max_length=5)
    closes_at: Optional[str] = Field(None, max_length=5)
    holiday_dates: Optional[List[str]] = None
    delivers: Optional[bool] = None
    express_delivery: Optional[bool] = None
    express_delivery_radius_km: Optional[float] = Field(None, ge=0, le=100)
    pickup_available: Optional[bool] = None
    delivery_radius_km: Optional[float] = Field(None, ge=0, le=100)
    delivery_fee: Optional[float] = Field(None, ge=0)
    free_delivery_above: Optional[float] = Field(None, ge=0)
    min_order_value: Optional[float] = Field(None, ge=0)
    avg_prep_minutes: Optional[int] = Field(None, ge=0, le=600)
    upi_id: Optional[str] = Field(None, max_length=120)
    bank_account_name: Optional[str] = Field(None, max_length=200)
    bank_account_number: Optional[str] = Field(None, max_length=60)
    bank_ifsc: Optional[str] = Field(None, max_length=20)
    platform_commission_percent: Optional[float] = Field(None, ge=0, le=100)


class PharmacyDocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    doc_type: DocumentType
    file_url: str
    file_name: str = ""
    document_number: Optional[str] = None
    issued_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    status: DocumentStatus
    review_notes: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    is_expired: bool = False
    days_to_expiry: Optional[int] = None
    created_at: datetime


class VerificationEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    from_status: Optional[str] = None
    to_status: str
    note: str = ""
    actor_name: str = ""
    created_at: datetime


class PharmacyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    address: str
    city: Optional[str] = None
    postal_code: Optional[str] = None
    phone: Optional[str] = None
    latitude: float
    longitude: float
    owner_name: Optional[str] = None
    business_name: Optional[str] = None
    gst_number: Optional[str] = None
    drug_license_number: Optional[str] = None
    drug_license_expiry: Optional[datetime] = None
    email: Optional[str] = None
    whatsapp: Optional[str] = None
    emergency_phone: Optional[str] = None
    logo_url: Optional[str] = None
    banner_url: Optional[str] = None
    store_images: List[str] = Field(default_factory=list)
    rating: float = 0.0
    total_ratings: int = 0
    is_partner: bool
    is_active: bool
    is_24x7: bool
    opens_at: Optional[str] = None
    closes_at: Optional[str] = None
    holiday_dates: List[str] = Field(default_factory=list)
    delivers: bool
    express_delivery: bool = False
    express_delivery_radius_km: Optional[float] = None
    pickup_available: bool = False
    delivery_radius_km: float
    delivery_fee: float
    free_delivery_above: Optional[float] = None
    min_order_value: float
    avg_prep_minutes: int
    platform_commission_percent: float = 0.0
    verification_status: VerificationStatus
    verification_notes: Optional[str] = None
    verified_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    suspended_at: Optional[datetime] = None
    suspension_reason: Optional[str] = None
    can_fulfil: bool = False
    created_at: datetime
    updated_at: datetime


class PharmacyDetailResponse(PharmacyResponse):
    """
    Full record for the detail screen.

    Bank and UPI details are deliberately absent from every response model.
    They are stored and auditable but never returned over the API — an admin
    console does not need to display settlement credentials to operate, and not
    serialising them removes a whole class of accidental exposure.
    """

    documents: List[PharmacyDocumentResponse] = Field(default_factory=list)
    verification_events: List[VerificationEventResponse] = Field(default_factory=list)


class PharmacyListResponse(BaseModel):
    items: List[PharmacyResponse] = Field(default_factory=list)
    total: int = 0
    skip: int = 0
    limit: int = 50


# ── workflow ─────────────────────────────────────────────────────────────


class VerificationTransitionRequest(BaseModel):
    to_status: VerificationStatus
    note: str = Field(default="", max_length=1000)


class SetActiveRequest(BaseModel):
    active: bool
    reason: str = Field(default="", max_length=500)


class BulkStatusRequest(BaseModel):
    pharmacy_ids: List[uuid.UUID] = Field(min_length=1, max_length=500)
    active: bool
    reason: str = Field(default="", max_length=500)


class DocumentCreateRequest(BaseModel):
    doc_type: DocumentType
    file_url: str = Field(min_length=1, max_length=500)
    file_name: str = Field(default="", max_length=255)
    document_number: Optional[str] = Field(None, max_length=120)
    issued_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class DocumentReviewRequest(BaseModel):
    status: DocumentStatus
    notes: str = Field(default="", max_length=500)


# ── inventory ────────────────────────────────────────────────────────────


class InventoryUpsertRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sku: Optional[str] = Field(None, max_length=64)
    rxcui: Optional[str] = Field(None, max_length=20)
    medicine_name: Optional[str] = Field(None, max_length=200)
    generic_name: Optional[str] = Field(None, max_length=200)
    brand_name: Optional[str] = Field(None, max_length=200)
    manufacturer: Optional[str] = Field(None, max_length=200)
    composition: Optional[str] = Field(None, max_length=500)
    strength: Optional[str] = Field(None, max_length=100)
    form: Optional[str] = Field(None, max_length=50)
    pack_size: Optional[str] = Field(None, max_length=50)
    drug_schedule: Optional[str] = Field(None, max_length=20)
    category: Optional[str] = Field(None, max_length=120)
    barcode: Optional[str] = Field(None, max_length=64)
    storage_instructions: Optional[str] = Field(None, max_length=300)
    batch_number: Optional[str] = Field(None, max_length=60)
    manufacturing_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    is_generic: Optional[bool] = None
    requires_prescription: Optional[bool] = None
    mrp: Optional[float] = Field(None, ge=0)
    selling_price: Optional[float] = Field(None, ge=0)
    discount_percent: Optional[float] = Field(None, ge=0, le=100)
    gst_percent: Optional[float] = Field(None, ge=0, le=100)
    stock_quantity: Optional[int] = Field(None, ge=0)
    low_stock_threshold: Optional[int] = Field(None, ge=0)
    min_stock: Optional[int] = Field(None, ge=0)
    max_stock: Optional[int] = Field(None, ge=0)
    reorder_level: Optional[int] = Field(None, ge=0)
    restock_expected_at: Optional[datetime] = None


class InventoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    pharmacy_id: uuid.UUID
    sku: str
    rxcui: Optional[str] = None
    medicine_name: str
    generic_name: Optional[str] = None
    brand_name: Optional[str] = None
    manufacturer: Optional[str] = None
    composition: Optional[str] = None
    strength: Optional[str] = None
    form: Optional[str] = None
    pack_size: Optional[str] = None
    drug_schedule: Optional[str] = None
    category: Optional[str] = None
    barcode: Optional[str] = None
    storage_instructions: Optional[str] = None
    batch_number: Optional[str] = None
    manufacturing_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    is_generic: bool = False
    requires_prescription: bool = True
    mrp: float = 0.0
    selling_price: float = 0.0
    discount_percent: float = 0.0
    gst_percent: float = 0.0
    stock_quantity: int = 0
    low_stock_threshold: int = 10
    min_stock: Optional[int] = None
    max_stock: Optional[int] = None
    reorder_level: Optional[int] = None
    restock_expected_at: Optional[datetime] = None
    stock_synced_at: Optional[datetime] = None
    availability: str = "unknown"
    stock_state: StockState = "available"
    inventory_value: float = 0.0
    created_at: datetime
    updated_at: datetime


class InventoryListResponse(BaseModel):
    items: List[InventoryResponse] = Field(default_factory=list)
    total: int = 0
    skip: int = 0
    limit: int = 50


class ImportResultResponse(BaseModel):
    created: int = 0
    updated: int = 0
    errors: List[dict] = Field(default_factory=list)


class StatusResponse(BaseModel):
    """Envelope for actions whose only result is success or failure."""

    status: str = "ok"
    message: str = ""


class BulkStatusResponse(StatusResponse):
    """
    Bulk result.

    `updated` and `requested` are reported separately because they legitimately
    differ — an id that no longer exists is skipped rather than aborting the
    batch, and the caller needs to see that happened.
    """

    updated: int = 0
    requested: int = 0


# ── analytics & audit ────────────────────────────────────────────────────


class PharmacyAnalyticsResponse(BaseModel):
    window_days: int
    orders_total: int
    orders_delivered: int
    orders_cancelled: int
    revenue_total: float
    revenue_delivered: float
    average_delivery_minutes: float
    conversion_rate: float
    orders_by_status: dict = Field(default_factory=dict)
    top_pharmacies: List[dict] = Field(default_factory=list)
    top_cities: List[dict] = Field(default_factory=list)
    top_medicines: List[dict] = Field(default_factory=list)
    inventory_value: float
    pharmacies_total: int
    pharmacies_partner: int


class AuditEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_name: str
    user_role: str
    action: str
    resource: str
    resource_id: str
    ip_address: str
    details: str = ""
    field_changed: Optional[str] = None
    previous_value: Optional[str] = None
    new_value: Optional[str] = None
    created_at: datetime


class AuditListResponse(BaseModel):
    items: List[AuditEntryResponse] = Field(default_factory=list)
    total: int = 0
    skip: int = 0
    limit: int = 50


# ── owner provisioning ───────────────────────────────────────────────────


class PharmacyOwnerResponse(BaseModel):
    """
    A pharmacy owner account.

    Carries no credential field of any kind — a temporary password is returned
    only by the two endpoints that mint one, and never by a read.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    role: str
    is_active: bool
    is_verified: bool
    pharmacy_id: Optional[uuid.UUID] = None
    created_at: datetime
    updated_at: datetime


class AssignOwnerRequest(BaseModel):
    """Link an existing account. Only unassigned or pharmacy-role users qualify."""

    user_id: uuid.UUID


class CreateOwnerRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    # Optional: omitted means the server generates one. A supplied password is
    # still hashed and never stored or echoed in plaintext.
    password: Optional[str] = Field(None, min_length=8, max_length=128)


class ChangeOwnerRequest(BaseModel):
    user_id: uuid.UUID
    reason: str = Field(default="", max_length=500)


class RemoveOwnerRequest(BaseModel):
    reason: str = Field(default="", max_length=500)


class OwnerStatusRequest(BaseModel):
    active: bool
    reason: str = Field(default="", max_length=500)


class OwnerCredentialResponse(BaseModel):
    """
    Returned once, by the endpoints that mint a credential.

    The plaintext exists only in this response; the database holds a hash and
    the audit trail records that a reset occurred without the value.
    """

    owner: PharmacyOwnerResponse
    temporary_password: str
    message: str = (
        "Share this password with the owner over a trusted channel. "
        "It cannot be retrieved again."
    )


class OwnerInvitationResponse(BaseModel):
    user_id: str
    email: str
    pharmacy_name: str
    temporary_password: str
    portal_url: str
    delivery: str
    email_sent: bool
