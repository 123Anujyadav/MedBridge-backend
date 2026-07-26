import uuid
from typing import Any, List, Optional
from pydantic import BaseModel, Field
from app.schemas.patient import ConsentFlagsSchema, PatientResponse

class AppointmentCreateRequest(BaseModel):
    doctor_id: uuid.UUID
    specialty: str = Field(min_length=1, max_length=100)
    hospital_name: str = Field(min_length=1, max_length=150)
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")  # YYYY-MM-DD
    time: str = Field(pattern=r"^\d{2}:\d{2}$")        # HH:MM
    type: str = Field(default="in_person", pattern="^(in_person|video|phone|ai_triage)$")
    reason: str = Field(min_length=1, max_length=255)

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

