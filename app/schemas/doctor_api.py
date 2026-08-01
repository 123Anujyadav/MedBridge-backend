import uuid
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field, model_validator
from app.schemas.clinical_review import ConfidenceReading
from app.schemas.patient_api import AppointmentResponse, ReportResponse

class UpdateAvailabilityRequest(BaseModel):
    availability: str = Field(pattern="^(available|busy|offline|on_leave)$")
    next_available: Optional[str] = Field(None, max_length=100)

class UpdateCaseNotesRequest(BaseModel):
    notes: str

class DiagnoseCaseRequest(BaseModel):
    diagnosis: str = Field(min_length=1, max_length=255)
    notes: str = Field(default="")

class CreateMedicationItem(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    generic_name: Optional[str] = Field(None, max_length=150)
    dosage: str = Field(min_length=1, max_length=100)
    frequency: str = Field(min_length=1, max_length=100)
    duration: str = Field(min_length=1, max_length=100)
    special_instructions: str = Field(default="")
    scheduled_times: List[str] = Field(default_factory=list)
    start_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    side_effects: List[str] = Field(default_factory=list)
    interactions: List[str] = Field(default_factory=list)

class CreatePrescriptionRequest(BaseModel):
    case_id: uuid.UUID
    patient_id: uuid.UUID
    diagnosis: str = Field(min_length=1, max_length=255)
    notes: str = Field(default="")
    follow_up_date: Optional[str] = Field(None, max_length=100)
    attachment_url: Optional[str] = Field(None, max_length=255)
    medications: List[CreateMedicationItem] = Field(default_factory=list)

class CreateReportRequest(BaseModel):
    patient_id: uuid.UUID
    case_id: Optional[uuid.UUID] = None
    """
    Consultation this document belongs to.

    Optional, and validated against the patient before use: a report attached to
    a case that is not theirs would put one patient's result on another's
    timeline. Left NULL when the document belongs to no particular case.
    """
    patient_name: str = Field(min_length=1, max_length=200)
    type: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(default="")
    content: str = Field(min_length=1)
    hospital_name: Optional[str] = Field(None, max_length=150)
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    file_url: Optional[str] = Field(None, max_length=255)
    file_size: Optional[str] = Field(None, max_length=50)
    ai_generated: bool = Field(default=False)
    ai_confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    vitals: Optional[dict] = None

class CaseResponse(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    patient_name: str
    patient_avatar_url: Optional[str] = None
    patient_age: int
    patient_gender: str
    doctor_id: Optional[uuid.UUID] = None
    doctor_name: Optional[str] = None
    specialty: str
    symptom_summary: str
    urgency_level: str
    status: str
    ai_extracted_symptoms: List[str] = []
    ai_specialty_recommendation: Optional[str] = None
    ai_confidence_score: float
    attachments: List[Any] = []
    patient_history: Optional[str] = None
    notes: str
    # Null until a clinician completes the consultation.
    diagnosis: Optional[str] = None

    class Config:
        from_attributes = True

class DoctorDashboardResponse(BaseModel):
    doctor_id: uuid.UUID
    total_patients: int
    total_consultations_week: int
    rating: float
    today_appointments: List[AppointmentResponse]
    pending_cases: List[CaseResponse]

class DoctorAnalyticsResponse(BaseModel):
    age_distribution: dict
    status_distribution: dict
    adherence_rate: float

    case_trend: List[dict] = Field(default_factory=list)
    """Monthly `{month, period, cases, resolved}` derived from real case rows."""

    specialty_distribution: List[dict] = Field(default_factory=list)
    """`{name, value}` case counts per specialty. Empty when there are no cases."""

    # ── Enterprise analytics (additive; legacy fields above are unchanged) ──
    #
    # Every figure is a real aggregate over this doctor's own rows. A metric the
    # data model cannot support is `None` and is named in `unavailable_metrics`,
    # never rendered as a zero a clinician could mistake for a measurement.

    range: dict = Field(default_factory=dict)
    summary: dict = Field(default_factory=dict)
    workload: dict = Field(default_factory=dict)
    patients: dict = Field(default_factory=dict)
    ai: dict = Field(default_factory=dict)
    reports: dict = Field(default_factory=dict)
    prescriptions: dict = Field(default_factory=dict)
    appointments: dict = Field(default_factory=dict)
    activity: List[dict] = Field(default_factory=list)
    """Recent events across this doctor's cases, from the existing audit trail."""
    unavailable_metrics: List[dict] = Field(default_factory=list)

# ── AI-assisted clinical report workflow ──────────────────────────────────────
#
# The doctor no longer authors a report from a blank form. The backend assembles
# everything already known about the case (patient demographics, the AI intake
# case, uploaded reports, prior history) into a draft, and the doctor reviews and
# edits the clinical judgement fields only.


class ReportDraftCandidate(BaseModel):
    """One case the calling doctor may issue an AI clinical report for."""

    case_id: uuid.UUID
    patient_id: uuid.UUID
    patient_name: str
    patient_age: int
    patient_gender: str
    specialty: str
    urgency_level: str
    status: str
    chief_complaint: str
    has_ai_intake: bool
    """True when an AI intake session is on file, so the draft is AI-grounded."""
    created_at: Optional[str] = None


class DraftAttachment(BaseModel):
    """An existing report on the patient's record, surfaced into the draft."""

    report_id: uuid.UUID
    title: str
    type: str
    date: str
    summary: str
    file_url: Optional[str] = None


class AIReportDraftRequest(BaseModel):
    case_id: uuid.UUID


class AIReportDraftResponse(BaseModel):
    """
    A complete pre-filled clinical report awaiting doctor review.

    Everything above `diagnosis` is auto-populated from the record and is not
    for the doctor to retype. The last five fields are seeded suggestions the
    doctor is expected to confirm or rewrite.
    """

    # ── Identity, auto-filled ────────────────────────────────────────────
    case_id: uuid.UUID
    patient_id: uuid.UUID
    patient_name: str
    patient_age: int
    patient_gender: str
    doctor_name: str
    hospital_name: str
    date: str
    title: str

    # ── Clinical context, auto-filled ────────────────────────────────────
    chief_complaint: str
    ai_summary: str
    symptoms: List[str] = Field(default_factory=list)
    clinical_findings: List[str] = Field(default_factory=list)
    previous_history: List[str] = Field(default_factory=list)
    uploaded_reports: List[DraftAttachment] = Field(default_factory=list)
    urgency_level: str = "medium"
    red_flags: List[str] = Field(default_factory=list)
    ai_confidence_score: Optional[float] = None

    # ── Doctor-editable, pre-seeded ──────────────────────────────────────
    diagnosis: str = ""
    clinical_notes: str = ""
    prescription: str = ""
    follow_up_instructions: str = ""
    recommendations: List[str] = Field(default_factory=list)
    recommended_tests: List[str] = Field(default_factory=list)

    # ── Provenance ───────────────────────────────────────────────────────
    ai_generated: bool = False
    """False when the LLM was unreachable and the draft came from records only."""
    draft_source: str = "records"
    """`groq` or `records`. Surfaced so a doctor knows what produced the text."""
    warnings: List[str] = Field(default_factory=list)
    """Gaps in the record the doctor should resolve before issuing."""


class IssueAIReportRequest(BaseModel):
    """The doctor's reviewed and approved version of the draft."""

    case_id: uuid.UUID
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(default="", max_length=2000)

    diagnosis: str = Field(min_length=1, max_length=2000)
    """Required: a report is not issuable until a clinician states a diagnosis."""

    clinical_notes: str = Field(default="")
    prescription: str = Field(default="")
    follow_up_instructions: str = Field(default="")
    recommendations: List[str] = Field(default_factory=list)
    recommended_tests: List[str] = Field(default_factory=list)
    follow_up_date: Optional[str] = Field(None, max_length=100)

    ai_generated: bool = Field(default=True)
    ai_confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)


# ── Doctor report cards ───────────────────────────────────────────────────────
#
# `GET /doctor/reports` returns these instead of bare `ReportResponse` rows so a
# clinician can triage the list without opening every report. It is a strict
# superset of `ReportResponse`, so existing consumers are unaffected.
#
# Every added field is optional or defaults to empty: the card hides what the
# record does not hold rather than rendering a placeholder.


class ReportCardIndicator(BaseModel):
    """A compact badge derived deterministically from stored values."""

    label: str
    tone: str
    """success | warning | error | info | neutral — the StatusBadge vocabulary."""


class DoctorReportCard(ReportResponse):
    # ── Patient ──────────────────────────────────────────────────────────
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    patient_short_id: str = ""
    """First segment of the patient UUID — enough to disambiguate on a card."""
    appointment_date: Optional[str] = None
    assigned_doctor: Optional[str] = None

    # ── Case ─────────────────────────────────────────────────────────────
    case_status: Optional[str] = None
    chief_complaint: Optional[str] = None
    extracted_symptoms: List[str] = Field(default_factory=list)
    specialty: Optional[str] = None
    urgency_level: Optional[str] = None
    ai_confidence: Optional[ConfidenceReading] = None
    """Present only when a score was actually recorded."""
    language_detected: Optional[str] = None
    case_created_at: Optional[str] = None
    case_updated_at: Optional[str] = None

    # ── Medical ──────────────────────────────────────────────────────────
    allergies: List[str] = Field(default_factory=list)
    chronic_conditions: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    uploaded_reports_count: int = 0
    previous_visits_count: int = 0
    previous_prescriptions_count: int = 0

    # ── AI ───────────────────────────────────────────────────────────────
    ai_summary: str = ""
    """The summary already on file. Never regenerated for the list view."""
    indicators: List[ReportCardIndicator] = Field(default_factory=list)
    flagged_for_follow_up: bool = False


# ── Bulk report actions ───────────────────────────────────────────────────────
#
# Only clinically safe, reversible-in-spirit operations are exposed. Prescribing
# and diagnosis finalisation are deliberately absent: both require per-patient
# judgement and must never be applied to a selection.


class BulkReportAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ASSIGN_SPECIALIST = "assign_specialist"
    FLAG_FOLLOW_UP = "flag_follow_up"
    ARCHIVE = "archive"
    MARK_REVIEWED = "mark_reviewed"
    REMOVE_REVIEW_FLAG = "remove_review_flag"


ACTIONS_REQUIRING_REASON = frozenset(
    {BulkReportAction.REJECT, BulkReportAction.ARCHIVE}
)
"""Rejection and archival remove a report from the working set; the record
should say why."""


class BulkReportActionRequest(BaseModel):
    action: BulkReportAction
    report_ids: List[uuid.UUID] = Field(min_length=1, max_length=500)
    reason: Optional[str] = Field(None, max_length=1000)
    target_doctor_id: Optional[uuid.UUID] = None
    """Required for `assign_specialist`: the receiving clinician."""

    @model_validator(mode="after")
    def _check_action_requirements(self) -> "BulkReportActionRequest":
        if self.action in ACTIONS_REQUIRING_REASON and not (self.reason or "").strip():
            raise ValueError(f"A reason is required to {self.action.value} reports.")
        if self.action is BulkReportAction.ASSIGN_SPECIALIST and not self.target_doctor_id:
            raise ValueError("target_doctor_id is required to assign a specialist.")
        # De-duplicate while preserving order: a repeated id must not be counted
        # twice in the result totals.
        seen: dict[uuid.UUID, None] = {}
        for rid in self.report_ids:
            seen.setdefault(rid, None)
        self.report_ids = list(seen)
        return self


class BulkItemOutcome(BaseModel):
    report_id: uuid.UUID
    outcome: str
    """completed | skipped | failed"""
    detail: str = ""


class BulkJobStatus(BaseModel):
    job_id: str
    action: str
    status: str
    """queued | running | completed"""
    total: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    items: List[BulkItemOutcome] = Field(default_factory=list)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    message: str = ""


class BulkSelectionResponse(BaseModel):
    """Every report id matching the caller's current filters."""

    report_ids: List[uuid.UUID] = Field(default_factory=list)
    total: int = 0


class BulkExportRequest(BaseModel):
    report_ids: List[uuid.UUID] = Field(min_length=1, max_length=500)


# ── Clinical document versions ────────────────────────────────────────────────


class ReportVersionSummary(BaseModel):
    version_number: int
    created_at: Optional[str] = None
    author_name: str
    author_type: str
    """doctor | ai | system"""
    status: str
    description: str = ""
    file_url: Optional[str] = None
    file_size: Optional[str] = None
    content_hash: str
    approval_note: Optional[str] = None
    rejection_reason: Optional[str] = None
    approved_by_name: Optional[str] = None
    approved_at: Optional[str] = None
    restored_from_version: Optional[int] = None
    is_latest: bool = False
    is_editable: bool = False
    """False for historical versions — the database rejects changes to them."""


class ReportVersionListResponse(BaseModel):
    report_id: uuid.UUID
    report_status: str
    current_version: int
    total: int
    skip: int
    limit: int
    has_more: bool
    versions: List[ReportVersionSummary] = Field(default_factory=list)


class DiffSegment(BaseModel):
    type: str
    """equal | added | removed"""
    text: str


class VersionFieldDiff(BaseModel):
    field: str
    label: str
    change: str
    """added | removed | modified"""
    previous_value: str = ""
    new_value: str = ""
    added_items: List[str] = Field(default_factory=list)
    removed_items: List[str] = Field(default_factory=list)
    segments: List[DiffSegment] = Field(default_factory=list)


class VersionComparisonResponse(BaseModel):
    report_id: uuid.UUID
    version_a: ReportVersionSummary
    version_b: ReportVersionSummary
    changed_by_type: str
    """Author of the newer version — who made the changes listed here."""
    changed_by_name: str
    identical: bool
    fields: List[VersionFieldDiff] = Field(default_factory=list)
    added_count: int = 0
    removed_count: int = 0
    modified_count: int = 0


class CreateReportVersionRequest(BaseModel):
    """
    Append a revision.

    Either supply the document fields, or set `restore_from_version` to bring an
    earlier version's content forward as a new one.
    """

    title: Optional[str] = Field(None, max_length=200)
    chief_complaint: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    diagnosis: Optional[str] = None
    clinical_notes: Optional[str] = None
    prescription: Optional[str] = None
    follow_up_instructions: Optional[str] = None
    ai_findings: Optional[str] = None
    symptoms: List[str] = Field(default_factory=list)
    recommended_tests: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

    author_type: str = Field(default="doctor", pattern="^(doctor|ai)$")
    """`ai` records a regenerated draft; it never supersedes an approved version."""
    status: str = Field(
        default="draft",
        pattern="^(draft|ai_draft|under_review|approved|rejected)$",
    )
    description: str = Field(default="", max_length=500)
    approval_note: Optional[str] = Field(None, max_length=2000)
    rejection_reason: Optional[str] = Field(None, max_length=2000)
    restore_from_version: Optional[int] = Field(None, ge=1)


class CompleteConsultationRequest(BaseModel):
    case_id: uuid.UUID
    diagnosis: str = Field(min_length=1, max_length=255)
    clinical_notes: str = Field(min_length=1)
    medications: List[dict] = Field(default_factory=list)
    recommended_tests: List[str] = Field(default_factory=list)
    follow_up_date: Optional[str] = Field(None, max_length=100)
    doctor_remarks: Optional[str] = Field(None)


