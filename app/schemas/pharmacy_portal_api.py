"""Wire format for the Pharmacy Owner Portal."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.pharmacy_api import OrderEventResponse, OrderItemResponse, OrderStatus

OrderAction = Literal["accept", "prepare", "ready", "pack", "dispatch", "deliver", "reject"]
ReviewOutcome = Literal["approved", "clarification_requested", "rejected"]


class PortalDashboardResponse(BaseModel):
    pharmacy_id: str
    pharmacy_name: str
    orders_today: int = 0
    orders_by_status: dict = Field(default_factory=dict)
    orders_active: int = 0
    revenue_today: float = 0.0
    revenue_week: float = 0.0
    revenue_month: float = 0.0
    average_delivery_minutes: float = 0.0
    average_prep_minutes: float = 0.0
    orders_delivered_total: int = 0
    customer_rating: float = 0.0
    total_ratings: int = 0
    pending_prescriptions: int = 0
    stock_low: int = 0
    stock_critical: int = 0
    stock_out: int = 0
    stock_near_expiry: int = 0
    stock_expired: int = 0
    catalogue_size: int = 0
    inventory_value: float = 0.0


class PortalOrderResponse(BaseModel):
    """
    An order as the dispensing counter sees it.

    Deliberately omits the patient's name, phone and full identity — the store
    needs the delivery address and the prescription, not the medical identity of
    the person behind it. `patient_id` is carried for the customer analytics
    join only.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    order_number: str
    prescription_id: uuid.UUID
    patient_id: uuid.UUID
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


class PortalOrderListResponse(BaseModel):
    items: List[PortalOrderResponse] = Field(default_factory=list)
    total: int = 0
    skip: int = 0
    limit: int = 50


class OrderActionRequest(BaseModel):
    action: OrderAction
    note: str = Field(default="", max_length=500)
    delivery_partner_name: Optional[str] = Field(None, max_length=150)
    delivery_partner_phone: Optional[str] = Field(None, max_length=50)


class PrescriberSummary(BaseModel):
    doctor_name: str
    specialty: Optional[str] = None
    qualification: Optional[str] = None
    hospital: Optional[str] = None
    registration_number: Optional[str] = None
    experience_years: Optional[int] = None
    avatar_url: Optional[str] = None


class ReviewMedication(BaseModel):
    name: str
    generic_name: Optional[str] = None
    brand_name: Optional[str] = None
    strength: Optional[str] = None
    dosage: str
    frequency: str
    duration: str
    food_instruction: Optional[str] = None
    route: Optional[str] = None
    quantity: Optional[int] = None
    special_instructions: str = ""


class ReviewFinding(BaseModel):
    category: str
    severity: str
    title: str
    detail: str = ""
    recommendation: str = ""
    medications_involved: List[str] = Field(default_factory=list)
    source: str = ""
    evidence: List[dict] = Field(default_factory=list)


class ReviewVerification(BaseModel):
    verdict: str
    status: str
    confidence: float
    summary: str = ""
    unchecked_medications: List[str] = Field(default_factory=list)
    findings: List[ReviewFinding] = Field(default_factory=list)


class ExpiryAlert(BaseModel):
    medicine_name: str
    batch_number: Optional[str] = None
    expiry_date: Optional[str] = None
    state: str


class PrescriptionReviewResponse(BaseModel):
    """Read-only dispensing pack. Nothing here is writable through the portal."""

    order_id: str
    order_number: str
    prescription_id: str
    diagnosis: str
    notes: str = ""
    issued_at: str
    signed_at: Optional[str] = None
    pdf_url: Optional[str] = None
    prescription_image_url: Optional[str] = None
    prescriber: PrescriberSummary
    patient_name: str
    patient_allergies: List[str] = Field(default_factory=list)
    medications: List[ReviewMedication] = Field(default_factory=list)
    verification: Optional[ReviewVerification] = None
    expiry_alerts: List[ExpiryAlert] = Field(default_factory=list)


class PrescriptionReviewRequest(BaseModel):
    outcome: ReviewOutcome
    note: str = Field(default="", max_length=1000)


class PortalAlertResponse(BaseModel):
    type: str
    severity: Literal["info", "warning", "critical"]
    title: str
    detail: str = ""
    reference_id: str
    created_at: str


class PortalAnalyticsResponse(BaseModel):
    window_days: int
    orders: int
    revenue: float
    average_basket: float
    fastest_moving: List[dict] = Field(default_factory=list)
    slowest_moving: List[dict] = Field(default_factory=list)
    peak_hours: List[dict] = Field(default_factory=list)
    top_customers: List[dict] = Field(default_factory=list)
    inventory_value: float
    expiry_loss: float
    catalogue_size: int


class PortalCustomerResponse(BaseModel):
    patient_id: str
    name: str
    orders: int
    total_spend: float
    average_spend: float
    last_order_at: Optional[str] = None
