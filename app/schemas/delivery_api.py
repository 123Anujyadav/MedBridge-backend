"""Wire format for the Delivery & Logistics platform."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

PartnerStatus = Literal[
    "pending", "document_review", "approved", "rejected", "suspended"
]
DeliveryStatus = Literal[
    "offered", "accepted", "en_route_pickup", "at_pharmacy", "picked_up",
    "out_for_delivery", "at_patient", "delivered", "cancelled", "failed",
]
# `delivered` is absent by design: a delivery is completed by verifying the
# patient's OTP, never by a rider asserting it.
AdvanceTarget = Literal[
    "accepted", "en_route_pickup", "at_pharmacy", "picked_up",
    "out_for_delivery", "at_patient",
]
VehicleType = Literal["bicycle", "motorcycle", "scooter", "car", "van", "on_foot"]


class DeliveryPartnerResponse(BaseModel):
    """
    A rider's own profile.

    Returned to the rider and to administrators. The patient-facing tracking
    payload is a separate, much narrower model.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    full_name: str
    phone: str
    photo_url: Optional[str] = None
    city: Optional[str] = None
    vehicle_type: Optional[str] = None
    vehicle_number: Optional[str] = None
    vehicle_model: Optional[str] = None
    driving_licence_number: Optional[str] = None
    driving_licence_expiry: Optional[datetime] = None
    verification_status: PartnerStatus
    verification_notes: Optional[str] = None
    suspension_reason: Optional[str] = None
    is_online: bool = False
    experience_years: Optional[int] = None
    rating: float = 0.0
    total_ratings: int = 0
    completed_deliveries: int = 0
    failed_deliveries: int = 0
    completion_rate: float = 0.0
    total_distance_km: float = 0.0
    total_earnings: float = 0.0
    current_latitude: Optional[float] = None
    current_longitude: Optional[float] = None
    location_updated_at: Optional[datetime] = None
    created_at: datetime


class PartnerCreateRequest(BaseModel):
    user_id: uuid.UUID
    full_name: str = Field(min_length=1, max_length=200)
    phone: str = Field(min_length=3, max_length=50)
    photo_url: Optional[str] = Field(None, max_length=500)
    city: Optional[str] = Field(None, max_length=120)
    address: Optional[str] = Field(None, max_length=500)
    vehicle_type: Optional[VehicleType] = None
    vehicle_number: Optional[str] = Field(None, max_length=30)
    vehicle_model: Optional[str] = Field(None, max_length=120)
    driving_licence_number: Optional[str] = Field(None, max_length=60)
    driving_licence_expiry: Optional[datetime] = None
    experience_years: Optional[int] = Field(None, ge=0, le=70)


class PartnerVerificationRequest(BaseModel):
    to_status: PartnerStatus
    note: str = Field(default="", max_length=1000)


class OnlineToggleRequest(BaseModel):
    online: bool


class LocationUpdateRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class DeliveryEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    status: DeliveryStatus
    note: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    actor_type: str = "partner"
    created_at: datetime


class DeliveryAssignmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    order_id: uuid.UUID
    partner_id: uuid.UUID
    pharmacy_id: uuid.UUID
    status: DeliveryStatus
    pickup_address: str = ""
    pickup_latitude: Optional[float] = None
    pickup_longitude: Optional[float] = None
    drop_address: str = ""
    drop_latitude: Optional[float] = None
    drop_longitude: Optional[float] = None
    distance_km: Optional[float] = None
    eta_minutes: Optional[int] = None
    estimated_arrival_at: Optional[datetime] = None
    delivery_fee: float = 0.0
    partner_earning: float = 0.0
    # The OTP itself never appears — only whether it has been satisfied.
    otp_verified: bool = False
    otp_attempts: int = 0
    proof_photo_url: Optional[str] = None
    proof_signature_url: Optional[str] = None
    delivery_notes: str = ""
    proof_captured_at: Optional[datetime] = None
    offered_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None
    picked_up_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    created_at: datetime
    events: List[DeliveryEventResponse] = Field(default_factory=list)


class AssignmentListResponse(BaseModel):
    items: List[DeliveryAssignmentResponse] = Field(default_factory=list)
    total: int = 0
    skip: int = 0
    limit: int = 50


class AdvanceRequest(BaseModel):
    target: AdvanceTarget
    note: str = Field(default="", max_length=500)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)


class FailRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=500)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)


class VerifyOtpRequest(BaseModel):
    code: str = Field(min_length=4, max_length=10)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)


class ProofRequest(BaseModel):
    photo_url: Optional[str] = Field(None, max_length=500)
    signature_url: Optional[str] = Field(None, max_length=500)
    notes: str = Field(default="", max_length=2000)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)


class RouteResponse(BaseModel):
    destination_label: str = ""
    destination_latitude: Optional[float] = None
    destination_longitude: Optional[float] = None
    heading_to: Literal["pharmacy", "patient"]
    # Reported so the client can explain a missing distance rather than
    # rendering a blank where a figure should be.
    maps_enabled: bool = False
    distance_km: Optional[float] = None
    eta_minutes: Optional[int] = None
    distance_text: Optional[str] = None
    duration_text: Optional[str] = None
    navigation_url: Optional[str] = None
    map_url: Optional[str] = None


class PartnerDashboardResponse(BaseModel):
    partner_id: str
    full_name: str
    is_online: bool
    verification_status: PartnerStatus
    deliveries_today: int = 0
    by_status: dict = Field(default_factory=dict)
    active_count: int = 0
    earnings_today: float = 0.0
    distance_today_km: float = 0.0
    delivered_today: int = 0
    average_delivery_minutes: float = 0.0
    completion_rate: float = 0.0
    rating: float = 0.0
    total_ratings: int = 0
    lifetime_deliveries: int = 0
    lifetime_distance_km: float = 0.0
    lifetime_earnings: float = 0.0


class FleetAnalyticsResponse(BaseModel):
    window_days: int
    assignments: int
    delivered: int
    failed: int
    cancelled: int
    success_rate: float
    total_distance_km: float
    delivery_revenue: float
    average_eta_minutes: float
    by_status: dict = Field(default_factory=dict)
    top_partners: List[dict] = Field(default_factory=list)
    partners_approved: int
    partners_online: int


class AssignOrderRequest(BaseModel):
    order_id: uuid.UUID
    partner_id: uuid.UUID


class TrackingEventResponse(BaseModel):
    status: str
    note: str = ""
    created_at: str


class DeliveryTrackingResponse(BaseModel):
    """
    What a patient may see about their rider.

    Deliberately narrow — name, photo, vehicle, rating, position and ETA. The
    rider's licence, address, earnings and other jobs are theirs, and none of
    it helps someone waiting at a door.
    """

    assignment_id: str
    status: DeliveryStatus
    partner_name: str
    partner_photo_url: Optional[str] = None
    partner_phone: Optional[str] = None
    partner_rating: float = 0.0
    vehicle_type: Optional[str] = None
    vehicle_number: Optional[str] = None
    current_latitude: Optional[float] = None
    current_longitude: Optional[float] = None
    location_updated_at: Optional[str] = None
    eta_minutes: Optional[int] = None
    distance_km: Optional[float] = None
    estimated_arrival_at: Optional[str] = None
    otp_required: bool = False
    delivered_at: Optional[str] = None
    events: List[TrackingEventResponse] = Field(default_factory=list)
