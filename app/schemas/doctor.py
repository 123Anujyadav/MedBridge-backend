import uuid
from typing import List, Optional
from pydantic import BaseModel, Field

class DoctorBase(BaseModel):
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    phone: str = Field(min_length=5, max_length=50)
    specialty: str = Field(min_length=1, max_length=100)
    sub_specialties: List[str] = Field(default_factory=list)
    hospital_id: Optional[uuid.UUID] = None
    hospital_name: Optional[str] = Field(None, max_length=150)
    license_number: str = Field(min_length=2, max_length=100)
    years_of_experience: int = Field(default=0, ge=0)
    availability: str = Field(default="available", pattern="^(available|busy|offline|on_leave)$")
    next_available: Optional[str] = Field(None, max_length=100)
    consultation_fee: float = Field(default=0.0, ge=0.0)
    education: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    avatar_url: Optional[str] = Field(None, max_length=255)
    bio: Optional[str] = Field(None, max_length=1000)

class DoctorCreate(DoctorBase):
    pass

class DoctorUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    phone: Optional[str] = Field(None, min_length=5, max_length=50)
    specialty: Optional[str] = Field(None, min_length=1, max_length=100)
    sub_specialties: Optional[List[str]] = None
    hospital_id: Optional[uuid.UUID] = None
    hospital_name: Optional[str] = Field(None, max_length=150)
    license_number: Optional[str] = Field(None, min_length=2, max_length=100)
    years_of_experience: Optional[int] = Field(None, ge=0)
    availability: Optional[str] = Field(None, pattern="^(available|busy|offline|on_leave)$")
    next_available: Optional[str] = Field(None, max_length=100)
    consultation_fee: Optional[float] = Field(None, ge=0.0)
    education: Optional[List[str]] = None
    certifications: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    avatar_url: Optional[str] = Field(None, max_length=255)
    bio: Optional[str] = Field(None, max_length=1000)

class DoctorResponse(DoctorBase):
    id: uuid.UUID
    rating: float
    total_patients: int
    total_cases: int
    verification_status: str
    verified_date: Optional[str]

    class Config:
        from_attributes = True
