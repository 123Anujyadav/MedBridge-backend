import uuid
from datetime import date as _date, datetime, timezone
from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator
from app.schemas.patient import ConsentFlagsSchema, PatientResponse


class _AppointmentSlotFields(BaseModel):
    """
    Shared calendar validation for the two routes that set an appointment slot.

    The field patterns only constrain the *shape* of the strings, so
    `9999-99-99` and `25:99` both satisfy them and were being stored verbatim.
    These validators parse the values, which is what actually rejects a date
    or a time that does not exist, and refuse a day that has already passed —
    a slot in the past is never a booking anyone can attend.

    Only whole days strictly before today (UTC) are refused. The stored value
    carries no timezone, so rejecting by time-of-day as well would turn a
    legitimate same-day booking into an error for anyone east of UTC.
    """

    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")  # YYYY-MM-DD
    time: str = Field(pattern=r"^\d{2}:\d{2}$")        # HH:MM

    @field_validator("date")
    @classmethod
    def _validate_date(cls, v: str) -> str:
        try:
            parsed = _date.fromisoformat(v)
        except ValueError:
            raise ValueError("date must be a real calendar date in YYYY-MM-DD form.")
        if parsed < datetime.now(timezone.utc).date():
            raise ValueError("An appointment cannot be booked in the past.")
        return v

    @field_validator("time")
    @classmethod
    def _validate_time(cls, v: str) -> str:
        hour, _, minute = v.partition(":")
        if not (0 <= int(hour) <= 23 and 0 <= int(minute) <= 59):
            raise ValueError("time must be a real 24-hour time in HH:MM form.")
        return v


class AppointmentCreateRequest(_AppointmentSlotFields):
    doctor_id: uuid.UUID
    specialty: str = Field(min_length=1, max_length=100)
    hospital_name: str = Field(min_length=1, max_length=150)
    type: str = Field(default="in_person", pattern="^(in_person|video|phone|ai_triage)$")
    reason: str = Field(min_length=1, max_length=255)

class BookableDoctorResponse(BaseModel):
    """
    A verified clinician a patient may book.

    Deliberately narrow: the booking form needs an id, a name, a specialty, a
    hospital and a fee. License numbers, contact details and case counts are not
    a patient's to read, so they are not exposed here.
    """

    id: uuid.UUID
    name: str
    specialty: str
    hospital_name: Optional[str] = None
    consultation_fee: float = 0.0
    rating: float = 0.0
    years_of_experience: int = 0
    availability: str = "available"
    avatar_url: Optional[str] = None

    class Config:
        from_attributes = True

class AppointmentRescheduleRequest(_AppointmentSlotFields):
    """
    New slot for an existing appointment.

    Same field rules as `AppointmentCreateRequest` — inherited rather than
    restated — so a date or time that booking would reject cannot enter
    through the reschedule path instead.
    """

class AppointmentResponse(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    doctor_id: uuid.UUID
    patient_name: str
    doctor_name: str
    # Profile photos of the two parties. Stored on the row and kept current by
    # `AvatarService`; exposed so appointment lists can show a face without a
    # per-row profile lookup. Null when the person has not set a photo.
    patient_avatar_url: Optional[str] = None
    doctor_avatar_url: Optional[str] = None
    specialty: str
    hospital_name: str
    date: str
    time: str
    duration: int
    type: str
    status: str
    reason: str
    notes: str
    room_number: Optional[str] = None
    video_call_link: Optional[str] = None
    case_id: Optional[uuid.UUID] = None

    class Config:
        from_attributes = True

class MedicationAdherenceTrack(BaseModel):
    status: str = Field(pattern="^(pending|taken|missed|snoozed|active)$")

class EmergencyLocation(BaseModel):
    lat: float = Field(ge=-90.0, le=90.0)
    lng: float = Field(ge=-180.0, le=180.0)
    address: str = Field(min_length=1, max_length=255)

class EmergencyPanicRequest(BaseModel):
    location: EmergencyLocation

class EmergencyPanicResponse(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    patient_name: str
    patient_phone: str
    location: Any
    hospital_id: Optional[uuid.UUID] = None
    hospital_name: Optional[str] = None
    ambulance_dispatched: bool
    ambulance_id: Optional[str] = None
    status: str
    eta: Optional[int] = None

    class Config:
        from_attributes = True

class MedicationResponse(BaseModel):
    id: uuid.UUID
    name: str
    generic_name: Optional[str] = None
    dosage: str
    frequency: str
    duration: str
    special_instructions: str
    status: str
    scheduled_times: List[str]
    taken_doses: int
    total_doses: int
    start_date: str
    end_date: str
    side_effects: List[str]
    interactions: List[str]

    class Config:
        from_attributes = True

class ReportSummaryResponse(BaseModel):
    id: uuid.UUID
    type: str
    title: str
    summary: str
    date: str
    status: str
    ai_generated: bool

    class Config:
        from_attributes = True

class PatientDashboardResponse(BaseModel):
    patient_id: uuid.UUID
    health_score: int
    upcoming_appointments: List[AppointmentResponse]
    today_medications: List[MedicationResponse]
    recent_reports: List[ReportSummaryResponse]
    unread_notifications_count: int

class PrescriptionResponse(BaseModel):
    id: uuid.UUID
    case_id: uuid.UUID
    patient_id: uuid.UUID
    patient_name: str
    doctor_id: uuid.UUID
    doctor_name: str
    diagnosis: str
    notes: str
    status: str
    follow_up_date: Optional[str] = None
    attachment_url: Optional[str] = None
    medications: List[MedicationResponse] = []

    class Config:
        from_attributes = True

class ReportResponse(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    case_id: Optional[uuid.UUID] = None
    patient_name: str
    type: str
    title: str
    summary: str
    content: str
    doctor_name: Optional[str] = None
    hospital_name: Optional[str] = None
    date: str
    status: str
    file_url: Optional[str] = None
    file_size: Optional[str] = None
    ai_generated: bool
    ai_confidence_score: Optional[float] = None
    tags: List[str] = []
    vitals: Optional[dict] = None

    class Config:
        from_attributes = True



class NearbyHospitalItem(BaseModel):
    """
    One facility from the emergency hospital search.

    Every optional field is genuinely unknown when it is null, never a
    placeholder. OpenStreetMap carries no phone number or live opening state
    for most facilities, and `distance_km`/`eta_minutes` are null whenever
    routing could not be reached — an invented ETA on an emergency screen is
    somewhere an ambulance gets sent.
    """

    place_id: str
    name: str
    address: Optional[str] = None
    latitude: float
    longitude: float
    distance_km: Optional[float] = None
    eta_minutes: Optional[int] = None
    distance_text: Optional[str] = None
    duration_text: Optional[str] = None
    phone: Optional[str] = None
    directions_url: str


class NearbyHospitalsResponse(BaseModel):
    """
    The facilities near a point, nearest first.

    `available` is false whenever there is nothing real to report, with
    `reason` saying why in language safe to show a patient. An empty list
    means "we could not find out", never "there are none nearby", so the
    client must not present it as the latter.
    """

    available: bool
    reason: Optional[str] = None
    latitude: float
    longitude: float
    hospitals: List[NearbyHospitalItem] = []
