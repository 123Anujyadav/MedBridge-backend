import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field

class UserStatusUpdateRequest(BaseModel):
    is_active: bool

class VerifyDoctorRequest(BaseModel):
    """
    An administrator's decision on a clinician's credentials.

    `pending` is the "unverify" action: it returns an approved clinician to the
    queue and, because every doctor route re-reads this status per request,
    ends their access immediately rather than at their next sign-in.
    """

    verification_status: str = Field(
        pattern="^(verified|rejected|under_review|pending)$"
    )


class AdminDoctorResponse(BaseModel):
    """
    Everything an administrator needs in order to judge a clinician.

    Wider than `DoctorResponse` on purpose — it carries the account-level facts
    that live on `users` (email, whether the account is suspended, when they
    registered) alongside the clinical profile, so the verification queue is one
    request rather than one per doctor.

    `doctor_code` is included because the administrator is the only person who
    can tell a newly approved clinician what their Doctor ID is. This response
    is reachable only behind `RoleChecker(["admin"])`.
    """

    id: uuid.UUID
    doctor_code: Optional[str] = None

    email: Optional[str] = None
    is_active: bool = True
    account_verified: bool = False
    registered_at: Optional[datetime] = None

    first_name: str
    last_name: str
    phone: str
    specialty: str
    sub_specialties: List[str] = Field(default_factory=list)
    hospital_id: Optional[uuid.UUID] = None
    hospital_name: Optional[str] = None
    license_number: str
    years_of_experience: int = 0
    education: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    rating: float = 0.0
    total_patients: int = 0
    total_cases: int = 0
    availability: Optional[str] = None
    consultation_fee: float = 0.0
    verification_status: str
    verified_date: Optional[str] = None

    class Config:
        from_attributes = True


class PaginatedAdminDoctors(BaseModel):
    """
    One page of the clinician roster, with enough context to navigate it.

    The list version of this endpoint silently returned only the first 100
    clinicians — an administrator on a platform with more than that simply could
    not see, and therefore could not approve, the rest. Carrying `total` and
    `pages` is what makes the truncation visible instead of invisible.
    """

    items: List[AdminDoctorResponse]
    total: int
    """Clinicians matching the filter, across every page."""

    page: int
    size: int
    pages: int
    has_next: bool
    has_prev: bool


class AdminAccountCapResponse(BaseModel):
    """How many administrator slots are in use, and the ceiling."""

    in_use: int
    maximum: int
    slots_available: int

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

