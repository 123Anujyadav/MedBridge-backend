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
from app.schemas.emergency_profile import (
    EmergencyLocationUpdate,
    EmergencyProfileResponse,
    EmergencyProfileUpsert,
)
from app.schemas.sos import (
    SOSActiveResponse,
    SOSCancelRequest,
    SOSCommunicationsResponse,
    SOSEmergencyResponse,
    SOSHospitalResponse,
    SOSTimelineResponse,
    SOSTriggerRequest,
)
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
from app.services.emergency_profile import emergency_profile_service
from app.services.emergency_comms import (
    emergency_comms_service,
    spawn_communications,
)
from app.services.sos import sos_service
from app.services.sos_timeline import sos_timeline_service
from app.services.sos_notifications import get_emergency_notifier
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

# ---------------------------------------------------------------------------
# Emergency Profile — the standing record of who to call, where the patient
# lives, and where they last were.
#
# Registered before `/emergency/{id}` only for readability; the paths do not
# overlap, since `/emergency-profile` has no slash where that route needs one.
#
# Every handler keys off `current_user.id`. None of them accepts a patient
# identifier, so there is no parameter a caller could change in order to reach
# somebody else's record.
# ---------------------------------------------------------------------------

@router.get("/emergency-profile", response_model=Optional[EmergencyProfileResponse])
async def get_emergency_profile(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Retrieve the caller's emergency profile.

    Returns `null` rather than 404 when none exists yet: a patient who has not
    filled the form in is the ordinary starting state, and the page renders an
    empty form for it rather than an error.
    """
    return await emergency_profile_service.get_profile(db, current_user.id)


@router.put("/emergency-profile", response_model=EmergencyProfileResponse)
async def save_emergency_profile(
    payload: EmergencyProfileUpsert,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Create or update the caller's emergency contact and registered address.

    One endpoint for both, because the form has one Save button. Stored
    coordinates are left alone — editing an address must not silently discard a
    position the patient has already captured.
    """
    return await emergency_profile_service.upsert_profile(
        db, current_user.id, payload
    )


@router.delete("/emergency-profile", status_code=status.HTTP_200_OK)
async def delete_emergency_profile(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Delete the caller's emergency profile in full."""
    return await emergency_profile_service.delete_profile(db, current_user.id)


@router.put("/emergency-profile/location", response_model=EmergencyProfileResponse)
async def update_emergency_location(
    payload: EmergencyLocationUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Record coordinates captured from the browser.

    Only latitude and longitude are accepted; the Google Maps link is derived
    from them on the server, so the stored link can only ever point at the
    stored position.
    """
    return await emergency_profile_service.update_location(
        db, current_user.id, payload
    )


@router.delete("/emergency-profile/location", response_model=EmergencyProfileResponse)
async def clear_emergency_location(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Forget the stored coordinates, keeping the contact and address.

    Separate from deleting the profile so a patient can withdraw the most
    sensitive part of the record without losing the rest of it.
    """
    return await emergency_profile_service.clear_location(db, current_user.id)


# ---------------------------------------------------------------------------
# SOS emergency workflow.
#
# `/sos/active` is declared before `/sos/{id}` so the literal wins the match.
# No handler takes a patient identifier: the emergency is always resolved from
# `current_user.id`, so there is no parameter a caller could change to reach
# somebody else's.
#
# Each write commits before it announces. The `get_db` dependency would
# otherwise commit *after* the handler returns, so a responder could be alerted
# to an emergency that a later failure rolled back.
# ---------------------------------------------------------------------------

@router.post("/sos", response_model=SOSEmergencyResponse,
             status_code=status.HTTP_201_CREATED)
async def trigger_sos(
    payload: SOSTriggerRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Raise an emergency.

    Refuses, with a message the patient can act on, when the emergency profile
    is incomplete, when no position is available, or when one of their
    emergencies is already open.

    Telephone calls, SMS and WhatsApp are handed to a background task rather
    than awaited here. Three calls and six messages are seconds of network I/O
    against an external vendor; a patient pressing an SOS button must not watch
    a spinner for them. The emergency is committed and broadcast to the
    dashboards first, so responders see it before the first handset rings, and
    the background task works from the durable communication log — nothing is
    lost if it fails.
    """
    response, emergency = await sos_service.trigger(db, current_user.id, payload)
    await db.commit()

    await get_emergency_notifier().emergency_raised(response)

    spawn_communications(emergency.id)
    return response


@router.get("/sos/active", response_model=SOSActiveResponse)
async def get_active_sos(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Whether this patient has an emergency open, and which one.

    One request, so the SOS button can render the right state on load without
    fetching a list and reasoning about it.
    """
    from app.services.sos import to_response

    emergency = await sos_service.get_active_for_patient(db, current_user.id)
    if emergency is None:
        return {"active": False, "emergency": None}

    patient = await patient_service.get_profile(db, current_user.id)
    return {"active": True, "emergency": to_response(emergency, patient)}


@router.get("/sos", response_model=List[SOSEmergencyResponse])
async def list_my_sos(
    active_only: bool = Query(False, description="Only emergencies still running."),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """This patient's emergency history, newest first."""
    return await sos_service.list_for_actor(db, current_user, active_only=active_only)


@router.get("/sos/{id}", response_model=SOSEmergencyResponse)
async def get_my_sos(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """One of this patient's emergencies, with its timeline."""
    return await sos_service.get_one(db, id, current_user)


@router.post("/sos/{id}/cancel", response_model=SOSEmergencyResponse)
async def cancel_my_sos(
    id: uuid.UUID,
    payload: SOSCancelRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Stand down an emergency this patient raised.

    A false alarm should be withdrawable by the person who raised it, without
    waiting for staff to do it for them.
    """
    response, _ = await sos_service.cancel(db, id, current_user, payload.reason)
    await db.commit()

    await get_emergency_notifier().emergency_updated(response)
    return response


@router.get("/sos/{id}/communications", response_model=SOSCommunicationsResponse)
async def get_sos_communications(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Every call and message raised for this emergency, with its outcome.

    Reachability is decided by `sos_service` before anything is read, so a
    patient can only ever see the communication log of their own emergency.
    Telephone numbers are masked in the response — the full number stays in the
    record for an incident review, but the API does not hand a third party's
    number to every screen.
    """
    await sos_service.get_one(db, id, current_user)  # authorises, or raises
    return await emergency_comms_service.summarise(db, id)


@router.get("/sos/{id}/timeline", response_model=SOSTimelineResponse)
async def get_sos_timeline(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    The whole history of an emergency on one axis.

    Status changes and communication attempts are stored in separate tables —
    one is clinical state, the other is delivery — and merged here in
    timestamp order, so the patient's screen shows "Doctor assigned" and "SMS
    sent to your emergency contact" in the order they actually happened.
    """
    emergency = await sos_service.get_one(db, id, current_user)
    return await sos_timeline_service.build(db, id, emergency)


@router.get("/sos/{id}/hospital", response_model=SOSHospitalResponse)
async def get_sos_hospital(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    The nearest facility found for this emergency, if one was.

    Answers with `available: false` and a reason when no Google Maps key is
    configured or the search returned nothing. It never guesses a hospital or
    an ETA — an invented address on an emergency screen is somewhere an
    ambulance gets sent.
    """
    await sos_service.get_one(db, id, current_user)
    return await sos_timeline_service.hospital_summary(db, id)


@router.post("/emergency", response_model=EmergencyPanicResponse, status_code=status.HTTP_201_CREATED)
async def trigger_emergency(
    panic_in: EmergencyPanicRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Legacy panic route, kept for compatibility and now backed by the SOS service.

    The request and response shapes are unchanged, so existing callers keep
    working. What changed is that it no longer invents its outcome: the previous
    implementation set `ambulance_dispatched` true with a hard-coded unit id, a
    twelve-minute ETA and a hospital name chosen when no hospital had capacity.
    Telling a patient that help is on its way when nothing has been arranged is
    the most harmful thing this endpoint could do, so it now records the same
    honest `pending` state the SOS workflow does, and a responder moves it on.

    New clients should use `POST /patient/sos`.
    """
    response, emergency = await sos_service.trigger(
        db, current_user.id,
        SOSTriggerRequest(latitude=panic_in.location.lat,
                          longitude=panic_in.location.lng),
        # This route predates the emergency profile and carries its own
        # coordinates and address, so it keeps the preconditions it shipped
        # with. `POST /patient/sos` is the one that requires the profile.
        require_profile=False,
        address_override=panic_in.location.address,
    )
    await db.commit()

    await get_emergency_notifier().emergency_raised(response)
    return emergency

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
