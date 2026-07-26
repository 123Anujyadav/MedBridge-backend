import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field

class UserStatusUpdateRequest(BaseModel):
    is_active: bool

class VerifyDoctorRequest(BaseModel):
    verification_status: str = Field(pattern="^(verified|rejected|under_review)$")

class HospitalCoordinates(BaseModel):
    lat: float = Field(ge=-90.0, le=90.0)
    lng: float = Field(ge=-180.0, le=180.0)

class CreateHospitalRequest(BaseModel):
    """
    Hospital onboarding payload.

    `email` and `coordinates` were previously required but the admin
    registration form never collected them, so every submission failed
    validation. They are optional now, and `total_beds`/`emergency_services` —
    which the form *does* send — are accepted instead of being silently
    discarded as unknown fields.
    """

    name: str = Field(min_length=1, max_length=150)
    address: str = Field(min_length=1, max_length=255)
    city: str = Field(min_length=1, max_length=100)
    state: str = Field(min_length=1, max_length=100)
    phone: str = Field(min_length=5, max_length=50)
    email: Optional[EmailStr] = None
    coordinates: Optional[HospitalCoordinates] = None
    services: List[str] = Field(default_factory=list)
    emergency_capacity: str = Field(default="available", pattern="^(available|limited|full)$")
    total_beds: int = Field(default=0, ge=0)
    available_beds: int = Field(default=0, ge=0)
    ambulance_count: int = Field(default=0, ge=0)
    emergency_services: bool = Field(default=False)

class HospitalVerificationRequest(BaseModel):
    verification_status: str = Field(pattern="^(verified|pending|rejected|under_review)$")

class AuditLogResponse(BaseModel):
    id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    user_name: str
    user_role: str
    action: str
    resource: str
    resource_id: str
    status: str
    details: Optional[str] = None
    ip_address: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class AdminDashboardResponse(BaseModel):
    """
    System overview counters.

    The extra breakdowns below were already being rendered by
    `AdminDashboard.tsx` but were never returned by the API, so those tiles
    displayed `undefined`. They are all real counts, not estimates.
    """

    total_users: int
    total_doctors: int
    total_hospitals: int
    active_emergencies: int
    system_status: str

    total_patients: int = 0
    total_cases: int = 0
    active_patients: int = 0
    active_doctors: int = 0
    active_hospitals: int = 0
    pending_doctor_verifications: int = 0

class ServiceStatus(BaseModel):
    status: str
    latency_ms: Optional[float] = None
    error: Optional[str] = None

class SystemAlertRequest(BaseModel):
    """An operational announcement broadcast to a role."""

    title: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=500)
    severity: str = Field(default="high", pattern="^(low|medium|high|critical)$")
    category: str = Field(default="system", pattern="^(system|security)$")
    """`security` alerts cannot be muted by recipient preferences."""
    audience: List[str] = Field(default_factory=lambda: ["doctor"])
    action_url: Optional[str] = Field(None, max_length=255)
    action_label: Optional[str] = Field(None, max_length=100)


class SystemAlertResponse(BaseModel):
    delivered: int
    audience: List[str]
    category: str
    severity: str


class SystemMonitorResponse(BaseModel):
    database: ServiceStatus
    redis: ServiceStatus
    celery: ServiceStatus
    cpu_usage: float
    memory_usage: float

class AdminAnalyticsResponse(BaseModel):
    users_by_role: dict
    hospitals_by_capacity: dict
    emergency_success_ratio: float
    ai_reports_summary: Optional[dict] = None

    avg_case_resolution_hours: float = 0.0
    """Mean hours from case creation to completion, over completed cases only."""

    avg_ai_confidence: float = 0.0
    """
    Mean AI confidence across generated reports, as a percentage.

    This is the model's own reported confidence, aggregated from
    `reports.ai_confidence_score`. It is deliberately NOT called accuracy:
    accuracy requires a validated ground-truth evaluation pipeline, which this
    system does not have.
    """


class HospitalResponse(BaseModel):
    id: uuid.UUID
    name: str
    address: str
    city: str
    state: str
    phone: str
    email: str
    services: List[str]
    ambulance_linked: bool
    ambulance_count: int
    emergency_capacity: str
    total_doctors: int
    total_beds: int
    available_beds: int
    rating: float
    coordinates: dict
    logo_url: Optional[str] = None
    verification_status: str

    class Config:
        from_attributes = True

