import logging
import uuid
from typing import Any, List, Optional
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis
from app.api.deps import get_db, get_redis, get_current_active_user
from app.api.pagination import Pagination, pagination_params
from app.core.exceptions import AuthorizationException, EntityNotFoundException
from app.models.user import User
from app.models.audit import AuditLog
from app.repositories.notification import notification_repository
from app.schemas.shared_api import (
    CaseTimelineResponse,
    MarkNotificationsRequest,
    NotificationCenterResponse,
    CalendarEventResponse,
    ClientAuditLogRequest,
    FeedbackRequest,
    NotificationResponse,
    SearchResult,
    SettingsResponse,
    SettingsUpdateRequest,
    TimelineEventResponse,
    UploadResponse,
)
from app.services.case_timeline import case_timeline_service
from app.services.notifications import notification_service
from app.services.shared import shared_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Number of right-most X-Forwarded-For hops that belong to our own proxies.
# Only the entries our infrastructure appended can be trusted; anything further
# left was supplied by the client and is forgeable.
TRUSTED_PROXY_HOPS = 1


def _client_ip(request: Request) -> str:
    """
    Resolve the caller's source IP.

    Behind a reverse proxy the socket peer is the proxy, so `X-Forwarded-For` is
    consulted — but only the hop our own proxy appended. Taking the left-most
    entry (the common mistake) would let a client spoof any address simply by
    sending its own `X-Forwarded-For` header.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        hops = [h.strip() for h in forwarded.split(",") if h.strip()]
        if hops:
            index = max(0, len(hops) - TRUSTED_PROXY_HOPS)
            return hops[index][:50]

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()[:50]

    return request.client.host[:50] if request.client else "unknown"

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Upload clinical images, laboratory reports, or general profile attachments.
    """
    return await shared_service.save_uploaded_file(file)

@router.get("/avatars/{filename}")
async def serve_avatar(filename: str) -> Any:
    """
    Serve a stored profile photo.

    Deliberately unauthenticated: a browser `<img>` tag cannot attach a bearer
    token, and every avatar surface in the SPA is an `<img>`. Two things make
    that acceptable here, and neither applies to clinical documents:

    * this route can only ever read `uploads/avatars/`, which holds re-encoded
      profile photos and nothing else — reports stay behind
      `/shared/reports/{id}/download`, which checks ownership;
    * stored names are 128 bits of randomness, so a URL cannot be guessed or
      enumerated from a patient id.

    Names are matched against the exact pattern this server writes, so a
    traversal attempt is rejected on its shape before the disk is touched.
    """
    from fastapi.responses import FileResponse

    from app.core.avatar import resolve_stored_avatar

    path = resolve_stored_avatar(filename)
    if path is None:
        raise EntityNotFoundException("Avatar", filename)

    return FileResponse(
        path,
        media_type="image/webp",
        headers={
            # A new upload gets a new filename, so a stored image never changes
            # and can be cached hard.
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/notifications", response_model=List[NotificationResponse])
async def list_notifications(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: Pagination = Depends(pagination_params),
) -> Any:
    """
    Retrieve all alert notifications addressed to the current active user.

    Kept as a plain list for existing callers. The notification centre uses
    `/notifications/center`, which adds filters, search, grouping and counts.
    """
    return await notification_repository.get_by_user(
        db, current_user.id, skip=page.skip, limit=page.limit
    )


@router.get("/notifications/center", response_model=NotificationCenterResponse)
async def notification_center(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    category: Optional[str] = Query(None, max_length=20),
    unread_only: bool = Query(False),
    critical_only: bool = Query(False),
    include_archived: bool = Query(False),
    search: Optional[str] = Query(None, max_length=200),
    date_from: Optional[str] = Query(None, max_length=40),
    date_to: Optional[str] = Query(None, max_length=40),
    skip: int = Query(0, ge=0),
    limit: int = Query(30, ge=1, le=100),
) -> Any:
    """
    The notification centre feed: filtered, searchable, grouped and counted.

    Scoped to the caller's own rows by `user_id`, so a doctor can never be
    handed another clinician's alerts. Critical items sort to the top.
    """
    return await notification_service.list_for_user(
        db, current_user,
        category=category, unread_only=unread_only, critical_only=critical_only,
        include_archived=include_archived, search=search,
        date_from=date_from, date_to=date_to, skip=skip, limit=limit,
    )


@router.put("/notifications/read-all")
async def mark_all_notifications_read(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Mark every unread, unarchived notification for the caller as read."""
    updated = await notification_service.mark_all_read(db, current_user)
    return {"status": "success", "updated": updated}


@router.put("/notifications/read-selected")
async def mark_selected_notifications_read(
    req: MarkNotificationsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Mark a chosen set as read. Ids the caller does not own are ignored."""
    updated = await notification_service.mark_selected_read(
        db, current_user, req.notification_ids
    )
    return {"status": "success", "updated": updated}


@router.put("/notifications/{id}/archive", response_model=NotificationResponse)
async def archive_notification(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Dismiss a notification. Recorded in the audit trail."""
    return await notification_service.archive(db, current_user, id)


@router.put("/notifications/{id}/opened", response_model=NotificationResponse)
async def record_notification_opened(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Record that the doctor followed a notification's action link.

    Marks it read and appends an `opened` event, so the audit trail
    distinguishes a notification that was acted on from one merely seen.
    """
    return await notification_service.record_opened(db, current_user, id)

@router.get("/notifications/unread-count")
async def unread_count(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Retrieve the number of unread notifications for the current active user.
    """
    count = await notification_repository.get_unread_count(db, current_user.id)
    return {"unread_count": count}

@router.put("/notifications/{id}/read", response_model=NotificationResponse)
async def mark_notification_as_read(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Mark an unread alert notification as read.
    """
    item = await notification_repository.get(db, id)
    if not item or item.user_id != current_user.id:
        raise EntityNotFoundException("Notification", str(id))

    item.read = True
    await db.flush()
    await db.refresh(item)
    return item

@router.get("/search", response_model=List[SearchResult])
async def search(
    q: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Perform a unified, case-insensitive query across doctors and hospital clinics.
    """
    return await shared_service.search_entities(db, q)

@router.get("/calendar", response_model=List[CalendarEventResponse])
async def calendar_schedule(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Aggregate all appointment schedules for the current user into a calendar feed.
    """
    return await shared_service.get_calendar_events(db, current_user)

@router.get("/timeline", response_model=CaseTimelineResponse)
async def case_timeline(
    case_id: uuid.UUID = Query(...),
    category: List[str] = Query(default_factory=list),
    search: Optional[str] = Query(None, max_length=200),
    date_from: Optional[str] = Query(None, max_length=40),
    date_to: Optional[str] = Query(None, max_length=40),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    The full chronological history of one case.

    Merges recorded audit events — which carry the actor, the reason and the
    before/after values — with milestones derived from the case's own rows.
    Supports category filters, keyword search, a date range and pagination.

    Restricted to the case's own patient, its assigned doctor, or an admin.
    """
    return await case_timeline_service.build(
        db, case_id, current_user,
        categories=category, search=search,
        date_from=date_from, date_to=date_to,
        skip=skip, limit=limit,
    )

@router.post("/audit-logs", status_code=status.HTTP_201_CREATED)
async def record_client_audit(
    audit_in: ClientAuditLogRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Record a PHI-access event in the HIPAA audit trail.

    Only the *subject* of the event comes from the client (which resource was
    viewed). Actor identity, role, source IP, timestamp and outcome are all
    derived server-side from the authenticated session and the connection, so a
    caller cannot attribute an action to another user, spoof an IP, backdate an
    entry, or mark a failed access as successful.
    """
    audit = AuditLog(
        user_id=current_user.id,
        user_name=current_user.email,
        user_role=current_user.role,
        action=audit_in.action.value,
        resource=audit_in.resource,
        resource_id=audit_in.resource_id,
        # Server-determined: never trusted from the request body.
        status="success",
        details=audit_in.details or "",
        ip_address=_client_ip(request),
    )
    db.add(audit)
    await db.flush()

    logger.info(
        "[AUDIT] user=%s role=%s action=%s resource=%s:%s ip=%s",
        current_user.id,
        current_user.role,
        audit.action,
        audit.resource,
        audit.resource_id,
        audit.ip_address,
    )
    return {"status": "success", "message": "Audit event recorded successfully."}

@router.get("/settings", response_model=SettingsResponse)
async def get_settings(
    redis: Redis = Depends(get_redis),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Retrieve personalized theme and notification preferences.
    """
    return await shared_service.get_user_settings(redis, current_user.id)

@router.put("/settings", response_model=SettingsResponse)
async def update_settings(
    settings_in: SettingsUpdateRequest,
    redis: Redis = Depends(get_redis),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Modify personalized theme and notification preferences.
    """
    return await shared_service.update_user_settings(
        redis, current_user.id, settings_in.model_dump(exclude_unset=True)
    )

@router.post("/feedback", status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    feedback_in: FeedbackRequest,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Record user satisfaction feedback, bug reports, and app reviews.
    """
    logger.info(
        f"Feedback received from User {current_user.id} ({current_user.role}): "
        f"Category: {feedback_in.category}, Rating: {feedback_in.rating}/5. "
        f"Subject: {feedback_in.subject}"
    )
    return {"status": "success", "message": "Feedback submitted successfully."}

@router.get("/reports/{id}/download")
async def download_report_pdf(
    id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    disposition: str = Query("attachment", pattern="^(attachment|inline)$"),
    version: Optional[int] = Query(None, ge=1),
) -> Any:
    """
    Serve a clinical report PDF.

    `disposition=inline` backs the in-app preview and `version=N` serves a
    specific revision — both on this one route, so the preview a doctor reads
    and the file a patient receives are always the same bytes from the same
    generator.

    Access is restricted to the owning patient, a doctor with a case for that
    patient, or an admin. Previously any authenticated user could download any
    patient's report by id.
    """
    import os
    from fastapi.responses import FileResponse
    from sqlalchemy import select
    from app.models.case import Case
    from app.models.report import Report

    res = await db.execute(select(Report).where(Report.id == id))
    report = res.scalars().first()
    if not report:
        raise EntityNotFoundException("Report PDF", str(id))

    # --- authorisation ---------------------------------------------------
    if current_user.role == "patient":
        if report.patient_id != current_user.id:
            raise AuthorizationException("Access denied to this clinical report.")
    elif current_user.role == "doctor":
        linked = await db.scalar(
            select(Case.id)
            .where(Case.doctor_id == current_user.id)
            .where(Case.patient_id == report.patient_id)
            .limit(1)
        )
        if linked is None:
            raise AuthorizationException("Access denied to this clinical report.")
    elif current_user.role != "admin":
        raise AuthorizationException("Access denied to this clinical report.")

    # --- version resolution ----------------------------------------------
    # A specific revision is served from its own frozen file; without a version
    # the caller gets the document as it currently stands.
    from app.services.report_versions import report_version_service

    if version is not None:
        target = await report_version_service.get_version(db, report.id, version)
        if not target.file_url:
            await report_version_service.render_version(db, report, target)
        source_url = target.file_url
        label = f"-v{version}"
    else:
        source_url = report.file_url
        label = ""

    if not source_url:
        raise EntityNotFoundException("Report PDF", str(id))

    # --- path resolution (constrained to the uploads directory) ----------
    # Shared with the writers via core.upload. This route used to compute its
    # own relative path and landed on app/uploads instead of Backend/uploads,
    # which made every generated PDF undownloadable.
    from app.core.upload import UPLOADS_ROOT

    uploads_root = UPLOADS_ROOT
    stored = source_url or ""
    if not stored.startswith("/uploads/"):
        # Never treat an arbitrary stored string as a filesystem path.
        logger.warning(f"Report {id} has a non-uploads file_url; refusing to serve.")
        raise EntityNotFoundException("Report File", str(id))

    file_path = os.path.abspath(
        os.path.join(uploads_root, stored[len("/uploads/"):])
    )
    if not file_path.startswith(uploads_root + os.sep):
        logger.error(f"Path traversal attempt blocked for report {id}")
        raise AuthorizationException("Invalid report path.")

    if not os.path.exists(file_path):
        raise EntityNotFoundException("Report File", str(id))

    # HIPAA access logging: a patient retrieving their own report is a
    # disclosure event and belongs on the case history.
    if current_user.role == "patient":
        await case_timeline_service.safe_record(
            db,
            event_type="patient.report_downloaded",
            description=f"Patient downloaded report '{report.title}'.",
            actor=current_user,
            actor_type="patient",
            case_id=report.case_id,
            patient_id=report.patient_id,
            resource="Report",
            resource_id=str(report.id),
            ip_address=_client_ip(request),
        )

    filename = os.path.basename(file_path)
    if disposition == "inline":
        # FileResponse always sets `attachment` when given a filename, so the
        # header is written explicitly for the preview case.
        return FileResponse(
            path=file_path,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                # Versions are immutable, so their rendering can be cached hard.
                "Cache-Control": "private, max-age=3600" if version else "private, no-cache",
                "X-Document-Version": str(version or report.current_version or 1),
            },
        )

    safe_title = "".join(c for c in report.title if c.isalnum() or c in " -_").strip()
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=f"{safe_title or 'clinical-report'}{label}.pdf",
    )

