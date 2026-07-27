import logging
import uuid
from typing import Any, List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis
from app.api.deps import get_db, get_redis, get_current_active_user, RoleChecker
from app.api.pagination import Pagination, pagination_params
from app.core.exceptions import EntityNotFoundException
from app.models.user import User
from app.repositories.user import user_repository
from app.repositories.doctor import doctor_repository
from app.repositories.hospital import hospital_repository
from app.repositories.audit import audit_log_repository
from app.schemas.user import UserResponse
from app.schemas.doctor import DoctorResponse
from app.schemas.doctor_api import CaseResponse
from app.schemas.patient_api import AppointmentResponse
from app.schemas.admin_api import (
    AdminAccountCapResponse,
    AdminAnalyticsResponse,
    AdminDashboardResponse,
    AdminDoctorResponse,
    PaginatedAdminDoctors,
    AuditLogResponse,
    CreateHospitalRequest,
    HospitalVerificationRequest,
    SystemAlertRequest,
    SystemAlertResponse,
    SystemMonitorResponse,
    UserStatusUpdateRequest,
    VerifyDoctorRequest,
    HospitalResponse,
)
from app.services.admin import admin_service

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(RoleChecker(["admin"]))])

@router.get("/dashboard", response_model=AdminDashboardResponse)
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Fetch global system dashboard overview stats.
    """
    return await admin_service.get_dashboard(db)

@router.get("/analytics", response_model=AdminAnalyticsResponse)
async def get_analytics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Compute global breakdown statistics on system users, hospitals, and emergencies.
    """
    return await admin_service.get_analytics(db)

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    role: Optional[str] = Query(None, pattern="^(patient|doctor|admin)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
) -> Any:
    """
    List all user logins on the platform, filtered optionally by role.
    """
    from sqlalchemy import select
    stmt = select(User).order_by(User.created_at.desc())
    if role:
        stmt = stmt.where(User.role == role)
    stmt = stmt.offset(page.skip).limit(page.limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())

@router.put("/users/{id}/status", response_model=UserResponse)
async def update_user_status(
    id: uuid.UUID,
    status_in: UserStatusUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Deactivate or reactivate a user account's platform access.
    """
    res = await admin_service.update_user_status(db, id, status_in.is_active)
    from app.core.websocket import websocket_manager
    await websocket_manager.broadcast({
        "type": "USER_STATUS_UPDATED",
        "user_id": str(id),
        "is_active": status_in.is_active
    })
    return res

@router.delete("/users/{id}", status_code=status.HTTP_200_OK)
async def delete_user(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Permanently delete a user account and associated clinical profile.
    """
    res = await admin_service.delete_user(db, id)
    from app.core.websocket import websocket_manager
    await websocket_manager.broadcast({
        "type": "USER_DELETED",
        "user_id": str(id)
    })
    return res

@router.get("/appointments", response_model=List[AppointmentResponse])
async def list_all_appointments(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
) -> Any:
    """
    List all appointments across all doctors and patients in the system.
    """
    from sqlalchemy import select
    from app.models.appointment import Appointment
    result = await db.execute(
        select(Appointment)
        .order_by(Appointment.date.desc(), Appointment.time.desc())
        .offset(page.skip)
        .limit(page.limit)
    )
    return list(result.scalars().all())

@router.put("/appointments/{id}/status", response_model=AppointmentResponse)
async def admin_update_appointment_status(
    id: uuid.UUID,
    status_str: str = Query(..., pattern="^(scheduled|confirmed|in_progress|completed|cancelled)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Admin override for appointment status.
    """
    from sqlalchemy import select
    from app.models.appointment import Appointment
    from app.core.exceptions import EntityNotFoundException
    from app.core.websocket import websocket_manager
    
    res = await db.execute(select(Appointment).where(Appointment.id == id))
    appt = res.scalars().first()
    if not appt:
        raise EntityNotFoundException("Appointment", str(id))
        
    appt.status = status_str
    await db.flush()
    
    await websocket_manager.broadcast({
        "type": "APPOINTMENT_STATUS_UPDATED",
        "appointment_id": str(id),
        "patient_id": str(appt.patient_id),
        "doctor_id": str(appt.doctor_id),
        "status": status_str
    })
    return appt

@router.get("/cases", response_model=List[CaseResponse])
async def list_all_cases(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
) -> Any:
    """
    List consultation cases across the platform.

    The admin Cases screen previously called the doctor-scoped case list, which
    is gated to the doctor role — so it returned 403 and the page could only
    ever render its error state.
    """
    from sqlalchemy import select
    from app.models.case import Case

    result = await db.execute(
        select(Case)
        .where(Case.deleted_at.is_(None))
        .order_by(Case.created_at.desc())
        .offset(page.skip)
        .limit(page.limit)
    )
    return list(result.scalars().all())


@router.get("/doctors/pending", response_model=List[DoctorResponse])
async def list_pending_doctors(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    List all doctors currently awaiting registration credentials approval.
    """
    from sqlalchemy import select
    from app.models.doctor import Doctor
    result = await db.execute(
        select(Doctor)
        .where(Doctor.verification_status.in_(["pending", "under_review"]))
        .order_by(Doctor.created_at.desc())
    )
    return list(result.scalars().all())

@router.get("/doctors", response_model=PaginatedAdminDoctors)
async def list_doctors_for_review(
    verification_status: Optional[str] = Query(
        None, pattern="^(verified|pending|rejected|expired|under_review)$"
    ),
    page: int = Query(1, ge=1, description="1-based page number."),
    size: int = Query(25, ge=1, le=200, description="Clinicians per page."),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    The clinician roster for the verification centre, one page at a time.

    Unlike `/doctors/pending` this returns every clinician whatever their
    status, joined to their account, so an administrator can review a decision
    they have already made — unverify an approved doctor, or reconsider a
    rejection — instead of only ever seeing the inbound queue.

    Paged rather than capped. The earlier version returned a bare array bounded
    at 100 rows with nothing to say it had stopped, so on a platform with more
    clinicians than that the surplus were invisible to the only people who can
    approve them.
    """
    return await admin_service.list_doctors_for_review(
        db, verification_status, page=page, size=size
    )

@router.get("/doctors/{id}", response_model=AdminDoctorResponse)
async def get_doctor_for_review(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    One clinician's complete record, including their Doctor ID.

    Behind the admin role guard on this router: the Doctor ID is a sign-in
    factor, so it is never served to anyone but an administrator.
    """
    return await admin_service.get_doctor_for_review(db, id)

@router.put("/doctors/{id}/verify", response_model=AdminDoctorResponse)
async def verify_doctor_profile(
    id: uuid.UUID,
    verify_in: VerifyDoctorRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Approve, reject or unverify a clinician's registration.

    Approving allocates the clinician's Doctor ID if they do not already hold
    one; the response carries it so the administrator can pass it on. Every
    doctor route re-reads the verification status per request, so unverifying
    or rejecting ends an active session immediately rather than at its next
    refresh.
    """
    doctor = await admin_service.verify_doctor(db, id, verify_in.verification_status)
    decided_status = doctor.verification_status
    payload = await admin_service.get_doctor_for_review(db, id)

    # Commit before announcing. The session is otherwise committed by the `get_db`
    # dependency *after* this function returns, so broadcasting here would tell
    # every listener about a decision that a later failure could still roll back.
    await db.commit()

    from app.core.websocket import websocket_manager
    await websocket_manager.broadcast_to_role({
        "type": "DOCTOR_VERIFICATION_UPDATED",
        "doctor_id": str(id),
        "verification_status": decided_status,
    }, "admin")
    logger.info(
        "[DOCTOR_VERIFICATION] admin=%s doctor=%s -> %s",
        current_user.id, id, decided_status,
    )
    return payload

@router.get("/admin-accounts", response_model=AdminAccountCapResponse)
async def get_admin_account_cap(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    How many of the platform's administrator slots are in use.

    The cap is enforced in the service layer and by a database trigger; this
    route exists so the admin console can show the limit rather than let
    somebody discover it by hitting an error.
    """
    from app.services.admin_accounts import MAX_ADMIN_ACCOUNTS, count_admin_accounts

    in_use = await count_admin_accounts(db)
    return {
        "in_use": in_use,
        "maximum": MAX_ADMIN_ACCOUNTS,
        "slots_available": max(0, MAX_ADMIN_ACCOUNTS - in_use),
    }

@router.get("/hospitals", response_model=List[HospitalResponse])
async def list_hospitals(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
) -> Any:
    """
    List all clinics and hospitals.
    """
    return await hospital_repository.get_multi(
        db, skip=page.skip, limit=page.limit
    )

@router.post("/hospitals", response_model=HospitalResponse, status_code=status.HTTP_201_CREATED)
async def register_hospital(
    hospital_in: CreateHospitalRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Onboard and register a new clinic or hospital facility.
    """
    from app.models.hospital import Hospital
    hospital = Hospital(
        name=hospital_in.name,
        address=hospital_in.address,
        city=hospital_in.city,
        state=hospital_in.state,
        phone=hospital_in.phone,
        email=hospital_in.email or "",
        coordinates=(
            hospital_in.coordinates.model_dump()
            if hospital_in.coordinates
            else {"lat": 0.0, "lng": 0.0}
        ),
        services=hospital_in.services,
        emergency_capacity=hospital_in.emergency_capacity,
        total_beds=hospital_in.total_beds,
        available_beds=hospital_in.available_beds or hospital_in.total_beds,
        ambulance_count=hospital_in.ambulance_count,
        ambulance_linked=hospital_in.emergency_services,
        verification_status="pending",
        rating=5.0
    )
    db.add(hospital)
    await db.flush()
    return hospital

@router.put("/hospitals/{id}/verify", response_model=HospitalResponse)
async def verify_hospital_facility(
    id: uuid.UUID,
    verify_in: HospitalVerificationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Verify or update a hospital's platform status.
    """
    return await admin_service.verify_hospital(db, id, verify_in.verification_status)


@router.get("/audit-logs", response_model=List[AuditLogResponse])
async def get_audit_logs(
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Retrieve HIPAA compliance audit trails tracking database read/write actions of patient data.
    """
    return await audit_log_repository.get_logs(db, limit)

@router.post("/system-alerts", response_model=SystemAlertResponse,
             status_code=status.HTTP_201_CREATED)
async def broadcast_system_alert(
    req: SystemAlertRequest,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Broadcast a maintenance window, degraded-service notice or security alert.

    Admin-only, and delivered as individual notifications through the existing
    service — so each recipient gets a row they can read, dismiss and audit,
    with the same deduplication and live delivery as any clinical alert.
    """
    from datetime import datetime, timezone

    from app.services.notifications import notification_service

    audience = tuple(
        role for role in req.audience if role in ("doctor", "patient", "admin")
    ) or ("doctor",)

    # Deduped per minute so a double-submitted announcement does not reach
    # every clinician twice.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    delivered = await notification_service.broadcast_system_alert(
        db,
        title=req.title,
        message=req.message,
        priority=req.severity,
        category=req.category,
        roles=audience,
        action_url=req.action_url,
        action_label=req.action_label,
        dedupe_key=f"{req.category}_alert:{req.title[:60]}:{stamp}",
        redis=redis,
    )
    logger.info(
        "[SYSTEM_ALERT] admin=%s category=%s severity=%s delivered=%d",
        current_user.id, req.category, req.severity, delivered,
    )
    return {
        "delivered": delivered,
        "audience": list(audience),
        "category": req.category,
        "severity": req.severity,
    }


@router.get("/monitor", response_model=SystemMonitorResponse)
async def system_monitoring(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Perform live health ping diagnostics of core services (database, Redis, task queue).
    """
    return await admin_service.get_system_status(db, redis)
