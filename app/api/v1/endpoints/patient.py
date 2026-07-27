import logging
import uuid
from datetime import date
from typing import Any, List, Optional
from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db, get_current_active_user, RoleChecker
from app.api.pagination import Pagination, pagination_params
from app.core.exceptions import AuthorizationException, EntityNotFoundException
from app.models.user import User
from app.repositories.appointment import appointment_repository
from app.repositories.prescription import prescription_repository, medication_repository
from app.repositories.report import report_repository
from app.repositories.notification import notification_repository
from app.schemas.patient import ConsentFlagsSchema, PatientResponse, PatientUpdate
from app.schemas.shared_api import NotificationResponse
from app.schemas.patient_api import (
    AppointmentCreateRequest,
    AppointmentRescheduleRequest,
    AppointmentResponse,
    BookableDoctorResponse,
    EmergencyPanicRequest,
    EmergencyPanicResponse,
    MedicationAdherenceTrack,
    MedicationResponse,
    PatientDashboardResponse,
    PrescriptionResponse,
    ReportResponse,
    ReportSummaryResponse,
)
from app.services.avatar import avatar_service
from app.services.patient import patient_service
from app.services.shared import shared_service
from app.services.appointment import appointment_service
from app.services.emergency import emergency_service
from app.services.vitals import vitals_service
from app.repositories.vital_reading import vital_reading_repository
from app.schemas.vitals_api import (
    VitalReadingCreate,
    VitalReadingResponse,
    VitalType,
    VitalsDashboardResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(RoleChecker(["patient"]))])

@router.get("/dashboard", response_model=PatientDashboardResponse)
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Fetch patient summary dashboard (appointments, active meds, recent reports, notifications count).
    """
    return await patient_service.get_dashboard(db, current_user.id)

@router.get("/profile", response_model=PatientResponse)
async def get_profile(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Retrieve clinical patient profile details.
    """
    return await patient_service.get_profile(db, current_user.id)

@router.put("/profile", response_model=PatientResponse)
async def update_profile(
    profile_in: PatientUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Update patient demographic and clinical fields.
    """
    return await patient_service.update_profile(db, current_user.id, profile_in)

@router.post("/profile/avatar", response_model=PatientResponse)
async def upload_profile_photo(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Upload or replace the patient's own profile photo.

    The photo is decoded and re-encoded before storage, so metadata (including
    EXIF GPS) is dropped and a non-image payload cannot survive. The record
    written is always the caller's own; there is no target-user parameter.
    """
    return await avatar_service.set_avatar(db, current_user, file)


@router.delete("/profile/avatar", response_model=PatientResponse)
async def remove_profile_photo(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Remove the patient's profile photo and delete the stored image."""
    return await avatar_service.remove_avatar(db, current_user)


@router.put("/consent", response_model=PatientResponse)
async def update_consent(
    consent_in: ConsentFlagsSchema,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Modify HIPAA consent authorization settings.
    """
    return await patient_service.update_consent(db, current_user.id, consent_in)

@router.get("/doctors", response_model=List[BookableDoctorResponse])
async def list_bookable_doctors(
    specialty: Optional[str] = Query(None, max_length=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
) -> Any:
    """
    List verified clinicians a patient can book.

    The booking form had no source of real doctors and shipped with five
    hardcoded UUIDs in its `<option>` values, so every booking attempt referred
    to a doctor that does not exist in the database. Only `verified` doctors are
    returned — an unverified or rejected clinician must not be bookable.
    """
    from sqlalchemy import select

    from app.models.doctor import Doctor

    stmt = (
        select(Doctor)
        .where(Doctor.verification_status == "verified")
        .where(Doctor.deleted_at.is_(None))
    )
    if specialty:
        stmt = stmt.where(Doctor.specialty.ilike(specialty))

    result = await db.execute(
        stmt.order_by(Doctor.rating.desc()).offset(page.skip).limit(page.limit)
    )

    return [
        BookableDoctorResponse(
            id=d.id,
            name=f"Dr. {d.first_name} {d.last_name}".strip(),
            specialty=d.specialty,
            hospital_name=d.hospital_name,
            consultation_fee=d.consultation_fee or 0.0,
            rating=d.rating or 0.0,
            years_of_experience=d.years_of_experience or 0,
            availability=d.availability or "available",
            avatar_url=d.avatar_url,
        )
        for d in result.scalars().all()
    ]

@router.get("/appointments", response_model=List[AppointmentResponse])
async def list_appointments(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
) -> Any:
    """
    List scheduled and past appointments, newest first.
    """
    return await appointment_repository.get_by_patient(
        db, current_user.id, skip=page.skip, limit=page.limit
    )

@router.post("/appointments", response_model=AppointmentResponse, status_code=status.HTTP_201_CREATED)
async def book_appointment(
    appt_in: AppointmentCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Schedule a new consultation slot.
    """
    return await appointment_service.book_appointment(db, current_user.id, appt_in)

@router.put("/appointments/{id}/reschedule", response_model=AppointmentResponse)
async def reschedule_appointment(
    id: uuid.UUID,
    reschedule_in: AppointmentRescheduleRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Move a scheduled consultation to a new date and time.
    """
    return await appointment_service.reschedule_appointment(
        db, current_user.id, id, reschedule_in.date, reschedule_in.time
    )

@router.put("/appointments/{id}/cancel", response_model=AppointmentResponse)
async def cancel_appointment(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Cancel a scheduled consultation.
    """
    return await appointment_service.cancel_appointment(db, current_user.id, id)

@router.get("/prescriptions", response_model=List[PrescriptionResponse])
async def list_prescriptions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
) -> Any:
    """
    List patient prescriptions, newest first.
    """
    return await prescription_repository.get_by_patient(
        db, current_user.id, skip=page.skip, limit=page.limit
    )

@router.get("/prescriptions/{id}", response_model=PrescriptionResponse)
async def get_prescription(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get detailed prescription and medication item specs.
    """
    rx = await prescription_repository.get(db, id)
    if not rx:
        raise EntityNotFoundException("Prescription", str(id))
    if rx.patient_id != current_user.id:
        raise AuthorizationException("Access denied to this prescription.")
    return rx

@router.put("/medications/{id}/track", response_model=MedicationResponse)
async def track_medication(
    id: uuid.UUID,
    track_in: MedicationAdherenceTrack,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Update dose status (taken/missed/snoozed) for clinical adherence tracking.
    """
    med = await medication_repository.get(db, id)
    if not med:
        raise EntityNotFoundException("Medication", str(id))
        
    # Get prescription to verify owner
    rx = await prescription_repository.get(db, med.prescription_id)
    if not rx or rx.patient_id != current_user.id:
        raise AuthorizationException("Access denied to this medication record.")

    # Update adherence details
    update_fields = {"status": track_in.status}
    if track_in.status == "taken":
        update_fields["taken_doses"] = med.taken_doses + 1

    updated_med = await medication_repository.update(db, db_obj=med, obj_in=update_fields)
    return updated_med

@router.get("/reports", response_model=List[ReportSummaryResponse])
async def list_reports(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
) -> Any:
    """
    List report and lab-result summaries, newest first.
    """
    return await report_repository.get_by_patient(
        db, current_user.id, skip=page.skip, limit=page.limit
    )

@router.post(
    "/records",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_medical_record(
    file: UploadFile = File(...),
    title: Optional[str] = Query(None, max_length=200),
    type: str = Query("upload", max_length=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Upload a medical document and register it as a clinical record.

    `/shared/upload` stores bytes on disk and returns metadata, but persists
    nothing — so an uploaded document never appeared in the patient's records,
    could not survive a refresh, and could not be downloaded or deleted. This
    route writes the file *and* commits the owning `Report` row, which is what
    makes the record real.
    """
    stored = await shared_service.save_uploaded_file(file)

    patient = await patient_service.get_profile(db, current_user.id)
    display_name = f"{patient.first_name} {patient.last_name}".strip()

    record = await report_repository.create(
        db,
        obj_in={
            "patient_id": current_user.id,
            "patient_name": display_name or "Patient",
            "type": type,
            "title": title or stored["filename"],
            "summary": f"Patient-uploaded document ({stored['filename']}).",
            "content": (
                "This record was uploaded by the patient. The original document "
                "is available for download."
            ),
            "date": date.today().isoformat(),
            "status": "ready",
            "file_url": stored["file_url"],
            "file_size": f"{max(1, stored['size_bytes'] // 1024)} KB",
            "ai_generated": False,
            "tags": ["patient-upload"],
        },
    )

    logger.info(
        "[RECORD_UPLOADED] patient=%s report=%s file=%s",
        current_user.id,
        record.id,
        stored["file_url"],
    )
    return record


@router.get("/reports/{id}", response_model=ReportResponse)
async def get_report(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get detailed report contents, clinical tags, and vitals snapshot.
    """
    report = await report_repository.get(db, id)
    if not report:
        raise EntityNotFoundException("Report", str(id))
    if report.patient_id != current_user.id:
        raise AuthorizationException("Access denied to this clinical report.")
    return report


@router.delete("/reports/{id}", status_code=status.HTTP_200_OK)
async def delete_report(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Delete one of the patient's own records.

    Ownership is verified before the delete so a guessed id cannot remove
    another patient's record. The stored file is removed alongside the row —
    leaving orphaned bytes on disk after the metadata is gone would make the
    upload directory grow without bound and defeat the deletion the patient
    asked for.
    """
    report = await report_repository.get(db, id)
    if not report:
        raise EntityNotFoundException("Report", str(id))
    if report.patient_id != current_user.id:
        raise AuthorizationException("Access denied to this clinical report.")

    stored_url = report.file_url
    await report_repository.remove(db, id=id)

    if stored_url:
        shared_service.delete_stored_file(stored_url)

    logger.info("[RECORD_DELETED] patient=%s report=%s", current_user.id, id)
    return {"message": "Record deleted.", "id": str(id)}

@router.get("/notifications", response_model=List[NotificationResponse])
async def list_notifications(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
) -> Any:
    """
    List user notifications, newest first.
    """
    return await notification_repository.get_by_user(
        db, current_user.id, skip=page.skip, limit=page.limit
    )

@router.put("/notifications/{id}/read")
async def mark_notification_read(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Mark alert notification as read.
    """
    notif = await notification_repository.get(db, id)
    if not notif:
        raise EntityNotFoundException("Notification", str(id))
    if notif.user_id != current_user.id:
        raise AuthorizationException("Access denied to this alert.")
        
    await notification_repository.update(db, db_obj=notif, obj_in={"read": True})
    return {"message": "Notification marked as read."}

@router.post("/emergency", response_model=EmergencyPanicResponse, status_code=status.HTTP_201_CREATED)
async def trigger_emergency(
    panic_in: EmergencyPanicRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Trigger emergency panic route, matching patient to nearest available hospital.
    """
    return await emergency_service.trigger_panic(db, current_user.id, panic_in.location)

@router.get("/emergency/{id}", response_model=EmergencyPanicResponse)
async def track_emergency(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Track ETA and dispatcher status of ambulance.
    """
    return await emergency_service.track_emergency(db, current_user.id, id)


# ---------------------------------------------------------------------------
# Vitals & adherence — the data behind the dashboard charts.
# Every value is read from `vital_readings` / `medications`. When a patient has
# no history the endpoints return empty collections; they never synthesise
# placeholder health data.
# ---------------------------------------------------------------------------


@router.get("/vitals/dashboard", response_model=VitalsDashboardResponse)
async def get_vitals_dashboard(
    days: int = Query(7, ge=1, le=365, description="Trailing window in days."),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Chart-ready vitals and medication adherence for the patient dashboard.

    Returns blood pressure, heart rate, temperature, oxygen saturation, weight,
    BMI (only when height is known) and glucose, bucketed per day, plus daily
    adherence. Empty arrays mean "no readings recorded", not an error.
    """
    series, latest = await vitals_service.build_series(db, current_user.id, days=days)
    adherence = await vitals_service.build_adherence(db, current_user.id, days=days)

    return VitalsDashboardResponse(
        series=series,
        adherence=adherence,
        has_vitals_data=bool(series),
        has_adherence_data=bool(adherence),
        latest=latest,
        days=days,
    )


@router.get("/vitals", response_model=List[VitalReadingResponse])
async def list_vitals(
    type: Optional[VitalType] = Query(None, description="Filter by reading type."),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
) -> Any:
    """List this patient's raw vital readings, oldest first."""
    return await vital_reading_repository.get_by_patient(
        db,
        current_user.id,
        types=[type.value] if type else None,
        skip=page.skip,
        limit=page.limit,
    )


@router.post(
    "/vitals",
    response_model=VitalReadingResponse,
    status_code=status.HTTP_201_CREATED,
)
async def record_vital(
    reading_in: VitalReadingCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Record a vital reading for the authenticated patient.

    The reading is always attributed to the caller, so one patient cannot write
    into another's chart. Unit and physiological plausibility are validated
    before the row is stored.
    """
    return await vitals_service.record(db, current_user.id, reading_in)
