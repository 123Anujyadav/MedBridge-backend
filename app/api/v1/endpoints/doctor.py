import logging
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional
from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Query, Request, UploadFile, status,
)
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import (
    get_db, get_current_active_user, get_redis, require_approved_doctor,
)
from app.api.pagination import Pagination, pagination_params
from app.core.exceptions import AuthorizationException, EntityNotFoundException
from app.models.user import User
from app.repositories.appointment import appointment_repository
from app.repositories.case import case_repository
from app.repositories.prescription import prescription_repository
from app.repositories.patient import patient_repository
from app.schemas.patient import PatientResponse
from app.schemas.patient_api import AppointmentResponse, PrescriptionResponse, ReportResponse
from app.schemas.doctor import DoctorResponse
from app.schemas.doctor_api import (
    AIReportDraftRequest,
    AIReportDraftResponse,
    BulkExportRequest,
    BulkJobStatus,
    BulkReportActionRequest,
    BulkSelectionResponse,
    CaseResponse,
    CreatePrescriptionRequest,
    CreateReportRequest,
    DiagnoseCaseRequest,
    DoctorAnalyticsResponse,
    DoctorDashboardResponse,
    CreateReportVersionRequest,
    DoctorReportCard,
    IssueAIReportRequest,
    ReportDraftCandidate,
    ReportVersionListResponse,
    ReportVersionSummary,
    VersionComparisonResponse,
    UpdateAvailabilityRequest,
    UpdateCaseNotesRequest,
    CompleteConsultationRequest,
)
from app.schemas.clinical_review import (
    ApproveAISummaryRequest,
    ClinicalReviewResponse,
    ReviewActionResponse,
    SaveConsultationRequest,
    SaveConsultationResponse,
)
from app.services.ai_report import ai_clinical_report_service
from app.services.avatar import avatar_service
from app.services.case_timeline import case_timeline_service
from app.services.clinical_review import clinical_review_service
from app.services.report_bulk import report_bulk_service
from app.services.report_cards import report_card_service
from app.services.report_versions import report_version_service
from app.api.v1.endpoints.shared import _client_ip
from app.services.doctor import doctor_service
from app.services.doctor_analytics import doctor_analytics_service
from app.services.consultation import consultation_service
from app.core.websocket import websocket_manager


logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_approved_doctor)])
"""
Every doctor API requires the doctor role *and* a current administrator
approval. `require_approved_doctor` performs the role check itself, so this
replaces the plain `RoleChecker(["doctor"])` rather than sitting alongside it.
"""

@router.get("/profile", response_model=DoctorResponse)
async def get_profile(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Retrieve doctor clinical profile.
    """
    return await doctor_service.get_profile(db, current_user.id)

@router.post("/profile/avatar", response_model=DoctorResponse)
async def upload_profile_photo(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Upload or replace the doctor's own profile photo.

    The stored image is a re-encoded copy, so nothing executable and no camera
    metadata is retained. Patient-facing surfaces that carry a copy of this
    avatar are refreshed in the same transaction.
    """
    return await avatar_service.set_avatar(db, current_user, file)


@router.delete("/profile/avatar", response_model=DoctorResponse)
async def remove_profile_photo(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Remove the doctor's profile photo and delete the stored image."""
    return await avatar_service.remove_avatar(db, current_user)


@router.get("/dashboard", response_model=DoctorDashboardResponse)
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Fetch doctor portal dashboard metrics, appointments, and pending cases.
    """
    return await doctor_service.get_dashboard(db, current_user.id)

@router.get("/analytics", response_model=DoctorAnalyticsResponse)
async def get_analytics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    range_preset: Optional[str] = Query(
        None, alias="range", pattern="^(today|yesterday|7d|30d|90d|custom)$"
    ),
    date_from: Optional[str] = Query(None, max_length=10),
    date_to: Optional[str] = Query(None, max_length=10),
) -> Any:
    """
    Operational and clinical analytics for the calling doctor.

    Extends the original response rather than replacing it: the legacy
    demographic fields are unchanged, and the enterprise sections are additive,
    so existing consumers keep working.

    Every figure is a real aggregate over this doctor's own rows. Metrics the
    data model cannot support are returned as null and named in
    `unavailable_metrics`.
    """
    legacy = await doctor_service.get_analytics(db, current_user.id)
    enriched = await doctor_analytics_service.build(
        db, current_user.id,
        preset=range_preset, date_from=date_from, date_to=date_to,
    )
    activity = await case_timeline_service.recent_for_doctor(db, current_user.id)
    return {**legacy, **enriched, "activity": activity}


@router.get("/analytics/export")
async def export_analytics(
    export_format: str = Query("csv", alias="format", pattern="^(csv|pdf)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    range_preset: Optional[str] = Query(
        None, alias="range", pattern="^(today|yesterday|7d|30d|90d|custom)$"
    ),
    date_from: Optional[str] = Query(None, max_length=10),
    date_to: Optional[str] = Query(None, max_length=10),
) -> Any:
    """
    Export the same analytics as CSV or PDF.

    Renders through the platform's existing ReportLab generator so there is one
    document pipeline, and reads the same service the dashboard reads so an
    export can never disagree with what is on screen.
    """
    from fastapi.responses import Response

    from app.services.report_generator import report_generator

    data = await doctor_analytics_service.build(
        db, current_user.id,
        preset=range_preset, date_from=date_from, date_to=date_to,
    )
    doctor = await doctor_service.get_profile(db, current_user.id)
    doctor_name = f"Dr. {doctor.first_name} {doctor.last_name}".strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    if export_format == "csv":
        body = doctor_analytics_service.to_csv(data)
        return Response(
            content=body,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="analytics-{stamp}.csv"'
            },
        )

    meta = report_generator.generate_analytics_pdf(
        title="Clinical Analytics Summary",
        doctor_name=doctor_name,
        hospital_name=doctor.hospital_name or "MedBridge Medical Center",
        period=f"{data['range']['date_from']} to {data['range']['date_to']}",
        sections=doctor_analytics_service.to_sections(data),
        filename_stem=f"analytics_{str(current_user.id)[:8]}_{stamp}",
    )
    with open(meta["dest_path"], "rb") as handle:
        content = handle.read()
    return Response(
        content=content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="analytics-{stamp}.pdf"'
        },
    )

@router.get("/patients", response_model=List[PatientResponse])
async def list_patients(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
) -> Any:
    """
    List all unique patients consulting under this doctor.
    """
    return await case_repository.get_patients_by_doctor(
        db, current_user.id, skip=page.skip, limit=page.limit
    )

@router.get("/patients/{id}", response_model=PatientResponse)
async def get_patient_profile(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Retrieve clinical profile for an assigned patient.
    """
    # Authorisation is a targeted existence check, not a scan of the doctor's
    # patient list: a paginated list would silently deny access to any patient
    # beyond the first page, and scanning is O(n) on every profile view.
    from sqlalchemy import select
    from app.models.case import Case

    linked = await db.scalar(
        select(Case.id)
        .where(Case.doctor_id == current_user.id)
        .where(Case.patient_id == id)
        .limit(1)
    )
    if linked is None:
        raise AuthorizationException("You are not authorized to view this patient profile.")

    patient = await patient_repository.get(db, id)
    if not patient:
        raise EntityNotFoundException("Patient", str(id))
    return patient

@router.get("/appointments", response_model=List[AppointmentResponse])
async def list_appointments(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
) -> Any:
    """
    List all appointments scheduled with this doctor.
    """
    from sqlalchemy import select
    from app.models.appointment import Appointment
    result = await db.execute(
        select(Appointment)
        .where(Appointment.doctor_id == current_user.id)
        .order_by(Appointment.date.desc(), Appointment.time.desc())
        .offset(page.skip)
        .limit(page.limit)
    )
    return list(result.scalars().all())

@router.put("/appointments/{id}/status", response_model=AppointmentResponse)
async def update_appointment_status(
    id: uuid.UUID,
    status_str: str = Query(..., pattern="^(scheduled|confirmed|in_progress|completed|cancelled)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Update status of a scheduled appointment.
    """
    appt = await appointment_repository.get(db, id)
    if not appt:
        raise EntityNotFoundException("Appointment", str(id))
    if appt.doctor_id != current_user.id:
        raise AuthorizationException("Access denied to this appointment.")
        
    appt.status = status_str
    await db.flush()
    
    # Broadcast real-time event to role dashboards
    await websocket_manager.broadcast({
        "type": "APPOINTMENT_STATUS_UPDATED",
        "appointment_id": str(appt.id),
        "patient_id": str(appt.patient_id),
        "doctor_id": str(appt.doctor_id),
        "status": status_str
    })
    return appt

@router.get("/cases", response_model=List[CaseResponse])
async def list_cases(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
) -> Any:
    """
    List all consultation cases assigned to this doctor.
    """
    return await case_repository.get_by_doctor(
        db, current_user.id, skip=page.skip, limit=page.limit
    )

@router.get("/cases/{id}", response_model=CaseResponse)
async def get_case(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Retrieve details of a specific assigned consultation case.
    """
    case = await consultation_service.get_case(db, id)
    if case.doctor_id != current_user.id:
        raise AuthorizationException("Access denied to this case.")
    return case

@router.put("/cases/{id}/notes", response_model=CaseResponse)
async def update_case_notes(
    id: uuid.UUID,
    notes_in: UpdateCaseNotesRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Save or append progress notes to a consultation case.
    """
    return await consultation_service.update_case_notes(db, current_user.id, id, notes_in)

@router.post("/cases/{id}/diagnose", response_model=CaseResponse)
async def diagnose_case(
    id: uuid.UUID,
    diagnose_in: DiagnoseCaseRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Diagnose and finalize a consultation case.
    """
    return await consultation_service.diagnose_case(db, current_user.id, id, diagnose_in)

@router.post("/cases/complete", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
async def complete_consultation(
    req: CompleteConsultationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Complete consultation case, generate an official PDF Medical Report, save it in the database,
    and send real-time WebSocket notifications across Patient, Doctor, and Admin dashboards.
    """
    meds_dicts = [m.model_dump() for m in req.medications] if req.medications else []
    return await consultation_service.complete_consultation(
        db=db,
        doctor_id=current_user.id,
        case_id=req.case_id,
        diagnosis=req.diagnosis,
        clinical_notes=req.clinical_notes,
        medications=meds_dicts,
        recommended_tests=req.recommended_tests,
        follow_up_date=req.follow_up_date,
        doctor_remarks=req.doctor_remarks
    )


@router.post("/prescriptions", response_model=PrescriptionResponse, status_code=status.HTTP_201_CREATED)
async def write_prescription(
    prescription_in: CreatePrescriptionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Issue a new medical prescription for a patient consultation case.
    """
    return await consultation_service.write_prescription(db, current_user.id, prescription_in)

@router.get("/prescriptions", response_model=List[PrescriptionResponse])
async def list_prescriptions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    List all prescriptions issued by this doctor.
    """
    try:
        prescriptions = await prescription_repository.get_by_doctor(db, current_user.id)
        return prescriptions or []
    except Exception as e:
        logger.exception(f"Error retrieving prescriptions for doctor user {current_user.id}: {e}")
        return []


@router.post("/reports", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
async def write_report(
    report_in: CreateReportRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Create a new lab report or clinical summary for a patient.
    """
    return await consultation_service.write_report(db, current_user.id, report_in)

@router.put("/schedule/availability", response_model=DoctorResponse)
async def update_availability(
    availability_in: UpdateAvailabilityRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Update availability status and next available slot parameters.
    """
    return await doctor_service.update_availability(db, current_user.id, availability_in)

@router.get("/reports", response_model=List[DoctorReportCard])
async def list_doctor_reports(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
    status_filter: Optional[str] = Query(
        None, alias="status",
        pattern="^(pending|ready|reviewed|shared|pending_review|approved|rejected|needs_revision|archived)$",
    ),
    urgency: Optional[str] = Query(None, pattern="^(low|medium|high|critical)$"),
    flagged: Optional[bool] = Query(None),
) -> Any:
    """
    List clinical and AI reports for patients under this doctor's care.

    Each row is a compact clinical summary — patient, case, AI intake and record
    counts — so the list can be triaged without opening every report. The
    response is a strict superset of `ReportResponse`.

    Scoped to patients who have a case assigned to the caller. Previously this
    returned every `Report` row in the system to any authenticated doctor, which
    leaked one patient's PHI to unrelated clinicians.
    """
    return await report_card_service.list_cards(
        db, current_user.id, skip=page.skip, limit=page.limit,
        status=status_filter, urgency=urgency, flagged=flagged,
    )


@router.get("/reports/ids", response_model=BulkSelectionResponse)
async def list_report_ids(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    status_filter: Optional[str] = Query(
        None, alias="status",
        pattern="^(pending|ready|reviewed|shared|pending_review|approved|rejected|needs_revision|archived)$",
    ),
    urgency: Optional[str] = Query(None, pattern="^(low|medium|high|critical)$"),
    flagged: Optional[bool] = Query(None),
) -> Any:
    """
    Every report id matching the caller's filters.

    Backs "select all matching current filters" without making the client page
    through the list, and resolves the set server-side so it can never include
    a report the caller may not act on.
    """
    ids = await report_bulk_service.owned_report_ids(
        db, current_user.id, status=status_filter, urgency=urgency, flagged=flagged
    )
    return {"report_ids": ids, "total": len(ids)}


@router.post("/reports/bulk", response_model=BulkJobStatus)
async def run_bulk_report_action(
    req: BulkReportActionRequest,
    request: Request,
    background: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    redis: Redis = Depends(get_redis),
) -> Any:
    """
    Apply one clinically safe action to a selection of reports.

    Small batches complete inline; larger ones return a queued job to poll.
    Either way the response carries per-item completed/skipped/failed outcomes —
    one bad item never aborts the rest. Every batch writes a HIPAA audit entry.
    """
    return await report_bulk_service.start(
        db,
        user=current_user,
        action=req.action,
        report_ids=req.report_ids,
        reason=req.reason,
        target_doctor_id=req.target_doctor_id,
        ip_address=_client_ip(request),
        redis=redis,
        background=background,
    )


@router.get("/reports/bulk/{job_id}", response_model=BulkJobStatus)
async def get_bulk_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    redis: Redis = Depends(get_redis),
) -> Any:
    """Progress for a queued bulk action."""
    return await report_bulk_service.get_job(redis, job_id)


@router.post("/reports/bulk/export")
async def export_selected_reports(
    req: BulkExportRequest,
    export_format: str = Query("csv", alias="format", pattern="^(csv|pdf)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Export a selection as CSV metadata or a ZIP of the stored PDFs.

    Scoped to the caller's own reports; anything else in the selection is simply
    absent from the export rather than raising.
    """
    from fastapi.responses import Response

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    if export_format == "csv":
        body = await report_bulk_service.export_csv(db, current_user.id, req.report_ids)
        return Response(
            content=body,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="reports-{stamp}.csv"'
            },
        )

    archive, skipped = await report_bulk_service.export_pdf_bundle(
        db, current_user.id, req.report_ids
    )
    return Response(
        content=archive,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="reports-{stamp}.zip"',
            # Surfaced so the UI can say "3 had no PDF" instead of shipping a
            # quietly smaller bundle than the doctor selected.
            "X-Skipped-Count": str(len(skipped)),
        },
    )


@router.get("/reports/draft-candidates", response_model=List[ReportDraftCandidate])
async def list_report_draft_candidates(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
) -> Any:
    """
    Cases this doctor can issue an AI clinical report for.

    Backs the case picker in the Issue AI Clinical Report modal, replacing the
    hand-typed patient UUID. Scoped to the caller's own cases.
    """
    return await ai_clinical_report_service.list_draft_candidates(
        db, current_user.id, skip=page.skip, limit=page.limit
    )


@router.post("/reports/ai-draft", response_model=AIReportDraftResponse)
async def generate_ai_report_draft(
    req: AIReportDraftRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Assemble a complete draft clinical report for a case.

    Loads the patient, the AI intake case, uploaded reports, extracted symptoms
    and prior history, and returns them pre-filled alongside AI-drafted clinical
    text for the doctor to review and edit. Nothing here is persisted — the
    draft only becomes a record when the doctor issues it.
    """
    return await ai_clinical_report_service.build_draft(
        db, current_user.id, req.case_id
    )


@router.post(
    "/reports/issue",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
)
async def issue_ai_clinical_report(
    req: IssueAIReportRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Issue the doctor-approved clinical report.

    Generates the PDF, stores the report against the patient and case, advances
    the case, notifies the patient and broadcasts the update to open dashboards.
    """
    return await ai_clinical_report_service.issue_report(db, current_user.id, req)


@router.get("/reports/{id}/versions", response_model=ReportVersionListResponse)
async def list_report_versions(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
) -> Any:
    """
    Version history for a clinical document, newest first.

    A report created before version tracking has version 1 materialised from its
    own stored content on first access, so history always starts from what the
    record actually says.
    """
    report = await report_version_service.load_readable_report(db, id, current_user)
    await report_version_service.ensure_initial_version(db, report, current_user)
    return await report_version_service.list_versions(db, report, skip=skip, limit=limit)


@router.get("/reports/{id}/versions/compare", response_model=VersionComparisonResponse)
async def compare_report_versions(
    id: uuid.UUID,
    a: int = Query(..., ge=1),
    b: int = Query(..., ge=1),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Field-level differences between two versions of the same document.

    Changes are attributed to the newer version's author, which is what
    distinguishes an AI redraft from a clinician's edit.
    """
    report = await report_version_service.load_readable_report(db, id, current_user)
    return await report_version_service.compare(db, report, a, b)


@router.post(
    "/reports/{id}/versions",
    response_model=ReportVersionSummary,
    status_code=status.HTTP_201_CREATED,
)
async def create_report_version(
    id: uuid.UUID,
    req: CreateReportVersionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Append a revision — never an in-place edit.

    Existing versions are untouched. A snapshot identical to the newest version
    returns that version rather than writing a duplicate or re-rendering a PDF.
    """
    report = await report_version_service.load_editable_report(db, id, current_user)
    await report_version_service.ensure_initial_version(db, report, current_user)

    if req.restore_from_version is not None:
        version, created = await report_version_service.restore(
            db, report=report, user=current_user,
            source_number=req.restore_from_version,
        )
        event, description = "report.version_restored", (
            f"Restored document content from version {req.restore_from_version}."
        )
    else:
        snapshot = {
            "title": req.title or report.title,
            "chief_complaint": req.chief_complaint,
            "summary": req.summary,
            "content": req.content,
            "diagnosis": req.diagnosis,
            "clinical_notes": req.clinical_notes,
            "prescription": req.prescription,
            "follow_up_instructions": req.follow_up_instructions,
            "ai_findings": req.ai_findings,
            "symptoms": req.symptoms,
            "recommended_tests": req.recommended_tests,
            "recommendations": req.recommendations,
        }
        # An AI regeneration is always a draft awaiting review. It is appended
        # above an approved version, never in place of it.
        version_status = "ai_draft" if req.author_type == "ai" else req.status
        version, created = await report_version_service.create_version(
            db,
            report=report,
            snapshot=snapshot,
            author=None if req.author_type == "ai" else current_user,
            author_type=req.author_type,
            author_name="AI Assistant" if req.author_type == "ai" else current_user.email,
            status=version_status,
            description=req.description,
            approval_note=req.approval_note,
            rejection_reason=req.rejection_reason,
        )
        event, description = "report.version_created", (
            f"Document version {version.version_number} created "
            f"({version.author_type})."
        )

    if created:
        await case_timeline_service.safe_record(
            db,
            event_type=event,
            description=description,
            actor=current_user,
            actor_type="doctor",
            case_id=report.case_id,
            patient_id=report.patient_id,
            resource="ReportVersion",
            resource_id=str(report.id),
            field="document_version",
            previous=str(version.version_number - 1) if version.version_number > 1 else None,
            new=str(version.version_number),
            reason=req.approval_note or req.rejection_reason,
            ip_address=_client_ip(request),
        )

    return report_version_service._version_payload(version, is_latest=True)


@router.get("/reports/{id}/clinical-review", response_model=ClinicalReviewResponse)
async def get_clinical_review(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Assemble the Clinical Review Workspace for a report.

    A read-only projection over `patients`, `cases`, `symptoms`, `reports`,
    `prescriptions`, `appointments` and `intake_sessions`. Creates no new
    storage and duplicates no business logic — report authoring stays in
    `AIClinicalReportService`, prescriptions in `ConsultationService`.
    """
    return await clinical_review_service.build_review(db, current_user.id, id)


@router.post("/cases/review/save", response_model=SaveConsultationResponse)
async def save_consultation_review(
    req: SaveConsultationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Save the doctor's working consultation notes, optionally completing the case.

    Writes only to columns that already exist on `cases`.
    """
    return await clinical_review_service.save_consultation(
        db,
        current_user.id,
        req.case_id,
        clinical_notes=req.clinical_notes,
        diagnosis=req.diagnosis,
        complete_case=req.complete_case,
    )


@router.post("/cases/review/approve-summary", response_model=ReviewActionResponse)
async def approve_ai_summary(
    req: ApproveAISummaryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Record the doctor's sign-off on the AI summary against the case.
    """
    return await clinical_review_service.approve_ai_summary(
        db, current_user.id, req.case_id, req.summary
    )


async def _load_report_for_doctor(db: AsyncSession, report_id: uuid.UUID, doctor_id: uuid.UUID):
    """
    Fetch a report only if it belongs to one of this doctor's patients.

    Shared by the status and content mutation routes, which previously let any
    doctor edit any patient's report.
    """
    from sqlalchemy import select
    from app.models.case import Case
    from app.models.report import Report

    res = await db.execute(select(Report).where(Report.id == report_id))
    report = res.scalars().first()
    if not report:
        raise EntityNotFoundException("Report", str(report_id))

    linked = await db.scalar(
        select(Case.id)
        .where(Case.doctor_id == doctor_id)
        .where(Case.patient_id == report.patient_id)
        .limit(1)
    )
    if linked is None:
        raise AuthorizationException(
            "You are not authorised to modify a report for this patient."
        )
    return report

@router.put("/reports/{id}/status", response_model=ReportResponse)
async def update_report_status(
    id: uuid.UUID,
    request: Request,
    status_str: str = Query(..., pattern="^(pending_review|approved|rejected|needs_revision|ready|pending|archived|shared)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Approve, reject, request revision, archive, or share a clinical report.

    `shared` releases the document to the patient and notifies them; this reuses
    the existing status route rather than adding a separate share endpoint.

    Only for reports belonging to this doctor's own patients. The transition is
    appended to the case timeline with its before and after values.
    """
    report = await _load_report_for_doctor(db, id, current_user.id)
    previous_status = report.status
    report.status = status_str
    report.doctor_name = f"Dr. {current_user.email.split('@')[0].capitalize()}"
    await db.flush()

    if status_str == "shared" and previous_status != "shared":
        # Through the notification service, not a direct row: otherwise this
        # carries no category, gets no live push, and is not deduplicated —
        # which made it invisible to the patient's category filters.
        from app.services.notifications import notification_service

        await notification_service.safe_notify(
            db,
            user_id=report.patient_id,
            category="report",
            type="report_shared",
            title="Clinical Report Shared With You",
            message=f"{report.doctor_name} shared the report '{report.title}' with you.",
            priority="medium",
            case_id=report.case_id,
            patient_id=report.patient_id,
            patient_name=report.patient_name,
            action_url=f"/patient/reports?report={report.id}",
            action_label="View Report",
            group_key="report_shared",
            dedupe_key=f"report_shared:{report.id}",
        )

    if previous_status != status_str:
        event = {
            "ready": "report.approved",
            "approved": "report.approved",
            "rejected": "report.rejected",
            "archived": "report.archived",
            "shared": "report.shared_with_patient",
        }.get(status_str, "report.status_changed")
        await case_timeline_service.safe_record(
            db,
            event_type=event,
            description=f"Report '{report.title}' moved to {status_str}.",
            actor=current_user,
            actor_type="doctor",
            # Only linked when the report actually carries a case FK; an
            # unlinked report's events belong to no case timeline.
            case_id=report.case_id,
            patient_id=report.patient_id,
            resource="Report",
            resource_id=str(report.id),
            field="status",
            previous=previous_status,
            new=status_str,
            ip_address=_client_ip(request),
        )
    return report

@router.put("/reports/{id}/content", response_model=ReportResponse)
async def update_report_content(
    id: uuid.UUID,
    content: str = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Edit clinical content of an AI report prior to final doctor approval.

    Only for reports belonging to this doctor's own patients.
    """
    report = await _load_report_for_doctor(db, id, current_user.id)
    report.content = content
    await db.flush()
    return report


