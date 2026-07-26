import uuid
from typing import List, Optional
from pydantic import BaseModel, Field

class EmergencyContactSchema(BaseModel):
    name: str = Field(default="")
    phone: str = Field(default="")
    relationship: str = Field(default="")

class ConsentFlagsSchema(BaseModel):
    dataSharing: bool = Field(default=True)
    researchParticipation: bool = Field(default=False)
    emergencyAccess: bool = Field(default=True)
    aiProcessing: bool = Field(default=True)

class PatientBase(BaseModel):
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    phone: str = Field(min_length=5, max_length=50)
    date_of_birth: str = Field(min_length=4, max_length=50)
    gender: str = Field(pattern="^(male|female|other)$")
    blood_type: Optional[str] = Field(None, max_length=20)
    height: Optional[float] = Field(None, gt=0)
    weight: Optional[float] = Field(None, gt=0)
    address: Optional[str] = Field(None, max_length=255)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=100)
    emergency_contact: EmergencyContactSchema = Field(default_factory=EmergencyContactSchema)
    allergies: List[str] = Field(default_factory=list)
    chronic_conditions: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    insurance_provider: Optional[str] = Field(None, max_length=150)
    insurance_number: Optional[str] = Field(None, max_length=100)
    avatar_url: Optional[str] = Field(None, max_length=255)
    consent_flags: ConsentFlagsSchema = Field(default_factory=ConsentFlagsSchema)

class PatientCreate(PatientBase):
    pass

class PatientUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    phone: Optional[str] = Field(None, min_length=5, max_length=50)
    date_of_birth: Optional[str] = Field(None, min_length=4, max_length=50)
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$")
    blood_type: Optional[str] = Field(None, max_length=20)
    height: Optional[float] = Field(None, gt=0)
    weight: Optional[float] = Field(None, gt=0)
    address: Optional[str] = Field(None, max_length=255)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=100)
    emergency_contact: Optional[EmergencyContactSchema] = None
    allergies: Optional[List[str]] = None
    chronic_conditions: Optional[List[str]] = None
    medications: Optional[List[str]] = None
    insurance_provider: Optional[str] = Field(None, max_length=150)
    insurance_number: Optional[str] = Field(None, max_length=100)
    avatar_url: Optional[str] = Field(None, max_length=255)
    consent_flags: Optional[ConsentFlagsSchema] = None

class PatientResponse(PatientBase):
    id: uuid.UUID
    health_score: int

    class Config:
        from_attributes = True
