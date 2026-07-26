"""
Response contract for the Clinical Review Workspace.

This is a read-only projection over data that already exists in `patients`,
`cases`, `symptoms`, `reports`, `prescriptions`, `appointments` and
`intake_sessions`. Nothing here introduces new storage.

Every field is optional or defaults to empty. That is deliberate: the workspace
must be able to say "not recorded" rather than render a plausible-looking value
a clinician could mistake for a measurement.
"""

from __future__ import annotations

import uuid
from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ── Section 1: Patient Overview ──────────────────────────────────────────────


class PatientOverview(BaseModel):
    patient_id: uuid.UUID
    patient_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    blood_group: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None
    """Computed only when both height and weight are on file. Never estimated."""
    bmi_category: Optional[str] = None

    allergies: List[str] = Field(default_factory=list)
    chronic_conditions: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)

    previous_visits: int = 0
    appointment_date: Optional[str] = None
    appointment_status: Optional[str] = None
    assigned_doctor: Optional[str] = None
    assigned_doctor_specialty: Optional[str] = None


# ── Section 2: AI Clinical Analysis ──────────────────────────────────────────


class SymptomTimelineEntry(BaseModel):
    name: str
    severity: Optional[str] = None
    duration: Optional[str] = None
    body_part: Optional[str] = None


class ConfidenceReading(BaseModel):
    """Present only when the pipeline actually recorded a score."""

    score: float = Field(ge=0.0, le=1.0)
    percentage: int = Field(ge=0, le=100)
    level: str
    """High (>= 0.8), Medium (>= 0.5), Low."""


class AIClinicalAnalysis(BaseModel):
    chief_complaint: str = ""
    ai_summary: str = ""
    extracted_symptoms: List[str] = Field(default_factory=list)
    symptom_timeline: List[SymptomTimelineEntry] = Field(default_factory=list)
    possible_causes: List[str] = Field(default_factory=list)
    """`differential_considerations` from intake. Prompts for a clinician, not conclusions."""
    severity: Optional[str] = None
    onset: Optional[str] = None
    duration: Optional[str] = None
    urgency_level: Optional[str] = None
    confidence: Optional[ConfidenceReading] = None
    recommended_specialist: Optional[str] = None
    recommendation_reason: Optional[str] = None
    emergency_indicators: List[str] = Field(default_factory=list)
    """Red flags detected deterministically during intake."""
    language_detected: Optional[str] = None
    conversation_summary: str = ""
    missing_information: List[str] = Field(default_factory=list)
    has_ai_intake: bool = False


# ── Section 3: Medical Evidence ──────────────────────────────────────────────


class EvidenceDocument(BaseModel):
    report_id: uuid.UUID
    title: str
    type: str
    category: str
    """lab | imaging | ai_analysis | clinical | other — derived from `type`."""
    date: str
    summary: str = ""
    status: str = ""
    doctor_name: Optional[str] = None
    file_url: Optional[str] = None
    downloadable: bool = False
    ai_generated: bool = False
    ai_confidence_score: Optional[float] = None


class CaseAttachment(BaseModel):
    name: str = ""
    type: str = ""
    url: Optional[str] = None


class MedicationLine(BaseModel):
    name: str
    generic_name: Optional[str] = None
    dosage: str = ""
    frequency: str = ""
    duration: str = ""
    special_instructions: str = ""
    status: str = ""
    side_effects: List[str] = Field(default_factory=list)
    interactions: List[str] = Field(default_factory=list)
    """Stored per-medication interactions, surfaced as recorded fact."""


class PrescriptionSummary(BaseModel):
    prescription_id: uuid.UUID
    diagnosis: str
    notes: str = ""
    status: str
    doctor_name: str
    follow_up_date: Optional[str] = None
    created_at: Optional[str] = None
    medications: List[MedicationLine] = Field(default_factory=list)


class MedicalEvidence(BaseModel):
    uploaded_reports: List[EvidenceDocument] = Field(default_factory=list)
    lab_reports: List[EvidenceDocument] = Field(default_factory=list)
    imaging_and_scans: List[EvidenceDocument] = Field(default_factory=list)
    ai_report_analysis: List[EvidenceDocument] = Field(default_factory=list)
    historical_reports: List[EvidenceDocument] = Field(default_factory=list)
    case_attachments: List[CaseAttachment] = Field(default_factory=list)
    doctor_notes: str = ""
    previous_prescriptions: List[PrescriptionSummary] = Field(default_factory=list)


# ── AI Assistance ────────────────────────────────────────────────────────────


class AISuggestions(BaseModel):
    """
    Advisory output. Every field is a prompt for clinician evaluation and is
    labelled as such in the UI — never a confirmed diagnosis.
    """

    differential_diagnoses: List[str] = Field(default_factory=list)
    drug_interaction_warnings: List[str] = Field(default_factory=list)
    red_flag_symptoms: List[str] = Field(default_factory=list)
    suggested_lab_tests: List[str] = Field(default_factory=list)
    suggested_imaging: List[str] = Field(default_factory=list)
    clinical_guideline_summary: str = ""
    possible_contraindications: List[str] = Field(default_factory=list)
    relevant_medical_history: List[str] = Field(default_factory=list)
    medication_alerts: List[str] = Field(default_factory=list)

    source: str = "records"
    """`groq` when the model produced these, `records` when it was unavailable."""
    generated: bool = False
    notes: List[str] = Field(default_factory=list)
    """Why a section is empty, e.g. no medications on file to interact."""


# ── Patient Timeline ─────────────────────────────────────────────────────────


class TimelineEvent(BaseModel):
    key: str
    label: str
    status: str
    """`completed` only when a real row or timestamp backs it, else `pending`."""
    timestamp: Optional[str] = None
    detail: str = ""


# ── Report Review comparison ─────────────────────────────────────────────────


class DecisionComparison(BaseModel):
    """Patient input -> AI interpretation -> doctor's final decision."""

    patient_input: str = ""
    patient_input_source: str = ""
    ai_interpretation: str = ""
    ai_interpretation_source: str = ""
    doctor_decision: str = ""
    doctor_decision_source: str = ""
    doctor_has_decided: bool = False


# ── Envelope ─────────────────────────────────────────────────────────────────


class ClinicalReviewResponse(BaseModel):
    report_id: uuid.UUID
    report_title: str
    report_status: str
    report_content: str
    report_file_url: Optional[str] = None
    case_id: Optional[uuid.UUID] = None
    case_status: Optional[str] = None

    patient_overview: PatientOverview
    ai_analysis: AIClinicalAnalysis
    medical_evidence: MedicalEvidence
    ai_suggestions: AISuggestions
    timeline: List[TimelineEvent] = Field(default_factory=list)
    comparison: DecisionComparison

    data_gaps: List[str] = Field(default_factory=list)
    """Explicit statements of what the record does not contain."""


class SaveConsultationRequest(BaseModel):
    """
    Persist the doctor's working decision without closing the case.

    Writes only to columns that already exist on `cases`.
    """

    case_id: uuid.UUID
    clinical_notes: str = Field(default="")
    diagnosis: Optional[str] = Field(None, max_length=255)
    complete_case: bool = False
    """When true the case moves to `completed`; otherwise `in_consultation`."""


class SaveConsultationResponse(BaseModel):
    case_id: uuid.UUID
    status: str
    notes: str
    saved_at: str
    timeline: List[TimelineEvent] = Field(default_factory=list)


class ReviewActionResponse(BaseModel):
    """Result of approving the AI summary onto the case record."""

    case_id: uuid.UUID
    status: str
    approved_summary: str
    approved_at: str


class ApproveAISummaryRequest(BaseModel):
    case_id: uuid.UUID
    summary: str = Field(min_length=1)
    """The AI summary as reviewed — the doctor may have edited it."""
