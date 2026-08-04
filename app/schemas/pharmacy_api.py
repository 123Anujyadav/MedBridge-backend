"""Wire format for pharmacy discovery, offers, ordering and tracking."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Availability = Literal["available", "limited", "out_of_stock", "unknown"]
OrderStatus = Literal[
    "received", "preparing", "packed", "out_for_delivery", "delivered", "cancelled"
]


class MedicineAlternativeResponse(BaseModel):
    inventory_id: str
    name: str
    generic_name: Optional[str] = None
    brand_name: Optional[str] = None
    strength: Optional[str] = None
    is_generic: bool = False
    unit_price: float = 0.0
    mrp: float = 0.0
    discount_percent: float = 0.0
    stock_quantity: int = 0
    availability: Availability = "unknown"
    saving_per_unit: float = 0.0


class MedicineAvailabilityResponse(BaseModel):
    requested_name: str
    rxcui: Optional[str] = None
    requested_quantity: int
    status: Availability
    inventory_id: Optional[str] = None
    matched_name: Optional[str] = None
    generic_name: Optional[str] = None
    brand_name: Optional[str] = None
    strength: Optional[str] = None
    is_generic: bool = False
    mrp: float = 0.0
    unit_price: float = 0.0
    discount_percent: float = 0.0
    stock_quantity: int = 0
    restock_expected_at: Optional[str] = None
    stock_synced_at: Optional[str] = None
    line_total: float = 0.0
    savings: float = 0.0
    alternatives: List[MedicineAlternativeResponse] = Field(default_factory=list)


class PharmacyOfferResponse(BaseModel):
    pharmacy_id: str
    name: str
    address: str
    phone: Optional[str] = None
    latitude: float
    longitude: float
    rating: float
    total_ratings: int
    is_partner: bool
    is_24x7: bool
    is_open_now: bool
    delivers: bool

    distance_km: float
    travel_minutes: int
    eta_minutes: int
    distance_source: str

    delivery_fee: float
    min_order_value: float
    subtotal: float
    total_savings: float
    grand_total: float

    items: List[MedicineAvailabilityResponse] = Field(default_factory=list)
    can_order: bool
    fully_available: bool
    fulfilment_ratio: float
    unavailable_items: List[str] = Field(default_factory=list)
    badges: List[str] = Field(default_factory=list)
    score: float
    map_url: str
    directions_url: str


class PharmacySearchResponse(BaseModel):
    """
    The pharmacy step of the workflow, in one payload.

    `maps_enabled` is surfaced so the client can explain *why* a result set is
    empty. Without it, "no pharmacies nearby" and "Maps is not configured" look
    identical to a patient.
    """

    prescription_id: uuid.UUID
    latitude: float
    longitude: float
    radius_km: float
    offers: List[PharmacyOfferResponse] = Field(default_factory=list)
    assistant_summary: str = ""
    maps_enabled: bool = False
    provider: str = "local_db"


class OrderSelectionItem(BaseModel):
    inventory_id: uuid.UUID
    quantity: int = Field(ge=1, le=1000)
    medication_id: Optional[uuid.UUID] = None
    is_generic_substitute: bool = False
    substituted_for: Optional[str] = Field(None, max_length=200)


class PlaceOrderRequest(BaseModel):
    prescription_id: uuid.UUID
    pharmacy_id: uuid.UUID
    items: List[OrderSelectionItem] = Field(min_length=1)
    delivery_address: str = Field(min_length=1, max_length=500)
    delivery_latitude: Optional[float] = Field(None, ge=-90, le=90)
    delivery_longitude: Optional[float] = Field(None, ge=-180, le=180)
    delivery_notes: str = Field(default="", max_length=1000)
    distance_km: Optional[float] = Field(None, ge=0)
    eta_minutes: Optional[int] = Field(None, ge=0, le=10_080)


class OrderItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    medicine_name: str
    generic_name: Optional[str] = None
    brand_name: Optional[str] = None
    strength: Optional[str] = None
    quantity: int
    unit_price: float
    mrp: float
    discount_percent: float
    line_total: float
    is_generic_substitute: bool
    substituted_for: Optional[str] = None


class OrderEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    status: OrderStatus
    note: str = ""
    actor_type: str = "system"
    created_at: datetime


class MedicineOrderResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    order_number: str
    prescription_id: uuid.UUID
    pharmacy_id: uuid.UUID
    pharmacy_name: str
    status: OrderStatus
    subtotal: float
    discount_total: float
    delivery_fee: float
    total: float
    currency: str
    delivery_address: str
    delivery_notes: str = ""
    distance_km: Optional[float] = None
    eta_minutes: Optional[int] = None
    estimated_delivery_at: Optional[datetime] = None
    delivery_partner_name: Optional[str] = None
    delivery_partner_phone: Optional[str] = None
    placed_at: Optional[datetime] = None
    dispatched_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    cancellation_reason: Optional[str] = None
    is_cancellable: bool = False
    created_at: datetime
    items: List[OrderItemResponse] = Field(default_factory=list)
    events: List[OrderEventResponse] = Field(default_factory=list)


class CancelOrderRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=500)


class AdvanceOrderRequest(BaseModel):
    """Pharmacy-side status update. Rejected if the transition is not allowed."""

    status: OrderStatus
    note: str = Field(default="", max_length=500)
    delivery_partner_name: Optional[str] = Field(None, max_length=150)
    delivery_partner_phone: Optional[str] = Field(None, max_length=50)


# ── geocoding ────────────────────────────────────────────────────────────


class GeocodeResultResponse(BaseModel):
    """One place match from Nominatim."""

    display_name: str
    latitude: float
    longitude: float
    type: Optional[str] = None
    importance: float = 0.0


class ReverseGeocodeResponse(BaseModel):
    latitude: float
    longitude: float
    # Null when the address is unknown — never a fabricated placeholder.
    address: Optional[str] = None
