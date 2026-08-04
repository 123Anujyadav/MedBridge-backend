"""Wire format for prescription safety reviews and the prescriber card."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Verdict = Literal["safe", "warning", "critical", "unknown"]


class EvidenceResponse(BaseModel):
    source: str = ""
    section: str = ""
    excerpt: str = ""
    reference: str = ""


class VerificationFindingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    category: str
    severity: Verdict
    confidence: float
    title: str
    detail: str = ""
    recommendation: str = ""
    medications_involved: List[str] = Field(default_factory=list)
    source: str = ""
    evidence: List[EvidenceResponse] = Field(default_factory=list)


class PrescriptionVerificationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    prescription_id: uuid.UUID
    status: Literal["pending", "completed", "failed", "degraded"]
    verdict: Verdict
    confidence: float
    summary: str = ""
    checked_medication_count: int = 0
    unchecked_medications: List[str] = Field(default_factory=list)
    sources_used: List[str] = Field(default_factory=list)
    engine_version: str = ""
    model_used: Optional[str] = None
    duration_ms: int = 0
    completed_at: Optional[datetime] = None
    created_at: datetime
    findings: List[VerificationFindingResponse] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.status == "completed"


class PrescriberCardResponse(BaseModel):
    """
    The prescriber as recorded on the prescription.

    Served from the snapshot columns, not the live doctor profile, so this is
    who signed on the day — not who that clinician is today.
    """

    model_config = ConfigDict(from_attributes=True)

    doctor_id: uuid.UUID
    doctor_name: str
    specialty: Optional[str] = None
    qualification: Optional[str] = None
    hospital: Optional[str] = None
    registration_number: Optional[str] = None
    experience_years: Optional[int] = None
    avatar_url: Optional[str] = None
    consultation_date: Optional[datetime] = None
    signed_at: Optional[datetime] = None
    signature_url: Optional[str] = None
    consultation_completed: bool = False
    prescription_signed: bool = False


class PrescriptionDocumentResponse(BaseModel):
    """Everything the patient's prescription screen needs in one call."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: str
    diagnosis: str
    notes: str = ""
    follow_up_date: Optional[str] = None
    created_at: datetime
    prescriber: PrescriberCardResponse
    medications: List["MedicationLineResponse"] = Field(default_factory=list)
    verification: Optional[PrescriptionVerificationResponse] = None
    pdf_url: Optional[str] = None
    prescription_image_url: Optional[str] = None


class MedicationLineResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
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
    rxcui: Optional[str] = None
    special_instructions: str = ""
    scheduled_times: List[str] = Field(default_factory=list)
    start_date: str
    end_date: str


PrescriptionDocumentResponse.model_rebuild()
