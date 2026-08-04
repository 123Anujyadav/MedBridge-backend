"""
Delivery & Logistics endpoints.

Three audiences, three gates, one module:

* `/api/v1/delivery/*` — the rider. Gated by `require_approved_delivery_partner`,
  which re-checks the profile per request, so a rider suspended mid-shift stops
  being able to act immediately.
* `/api/v1/delivery/admin/*` — the fleet. Gated by `RoleChecker(["admin"])`.
* `/api/v1/delivery/tracking/{order_id}` — the patient, read-only, scoped to
  their own order.

No rider endpoint accepts a partner id: the profile comes from the auth gate,
so there is no parameter to change to reach another rider's work.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    RoleChecker,
    get_current_active_user,
    get_db,
    require_approved_delivery_partner,
)
from app.api.pagination import Pagination, pagination_params
from app.core.exceptions import AuthorizationException, EntityNotFoundException
from app.models.delivery import DeliveryPartner
from app.models.user import User
from app.schemas.delivery_api import (
    AdvanceRequest,
    AssignOrderRequest,
    AssignmentListResponse,
    DeliveryAssignmentResponse,
    DeliveryPartnerResponse,
    DeliveryTrackingResponse,
    FailRequest,
    FleetAnalyticsResponse,
    LocationUpdateRequest,
    OnlineToggleRequest,
    PartnerCreateRequest,
    PartnerDashboardResponse,
    PartnerVerificationRequest,
    ProofRequest,
    RouteResponse,
    VerifyOtpRequest,
)
from app.services.delivery import delivery_service as svc

logger = logging.getLogger(__name__)

router = APIRouter()
admin_router = APIRouter(dependencies=[Depends(RoleChecker(["admin"]))])


def _partner_payload(partner: DeliveryPartner) -> DeliveryPartnerResponse:
    data = {
        field: getattr(partner, field)
        for field in DeliveryPartnerResponse.model_fields
        if hasattr(partner, field)
    }
    data["completion_rate"] = partner.completion_rate
    return DeliveryPartnerResponse(**data)


def _assignment_payload(assignment) -> DeliveryAssignmentResponse:
    data = {
        field: getattr(assignment, field)
        for field in DeliveryAssignmentResponse.model_fields
        if hasattr(assignment, field) and field not in ("otp_verified", "events")
    }
    data["otp_verified"] = assignment.is_otp_verified
    data["events"] = list(getattr(assignment, "events", []) or [])
    return DeliveryAssignmentResponse(**data)


# ── rider ────────────────────────────────────────────────────────────────


@router.get("/me", response_model=DeliveryPartnerResponse)
async def my_profile(partner: DeliveryPartner = Depends(require_approved_delivery_partner)):
    return _partner_payload(partner)


@router.get("/dashboard", response_model=PartnerDashboardResponse)
async def dashboard(
    db: AsyncSession = Depends(get_db),
    partner: DeliveryPartner = Depends(require_approved_delivery_partner),
):
    """Today's deliveries, earnings, distance and completion rate."""
    return PartnerDashboardResponse(**await svc.dashboard(db, partner))


@router.post("/online", response_model=DeliveryPartnerResponse)
async def set_online(
    payload: OnlineToggleRequest,
    db: AsyncSession = Depends(get_db),
    partner: DeliveryPartner = Depends(require_approved_delivery_partner),
):
    """
    Clock on or off.

    Going offline stops new offers; it does not drop work already accepted — a
    rider carrying medicine still has to deliver it.
    """
    await svc.set_online(db, partner, online=payload.online)
    await db.commit()
    return _partner_payload(partner)


@router.post("/location", response_model=DeliveryPartnerResponse)
async def update_location(
    payload: LocationUpdateRequest,
    db: AsyncSession = Depends(get_db),
    partner: DeliveryPartner = Depends(require_approved_delivery_partner),
):
    """Push the rider's position for live tracking."""
    await svc.update_location(
        db, partner, latitude=payload.latitude, longitude=payload.longitude
    )
    await db.commit()
    return _partner_payload(partner)


@router.get("/orders", response_model=AssignmentListResponse)
async def my_deliveries(
    status_filter: Optional[str] = Query(None, alias="status"),
    page: Pagination = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    partner: DeliveryPartner = Depends(require_approved_delivery_partner),
):
    """This rider's assignments. `status=active` returns everything in flight."""
    items, total = await svc.list_for_partner(
        db, partner, status=status_filter, skip=page.skip, limit=page.limit
    )
    return AssignmentListResponse(
        items=[_assignment_payload(a) for a in items],
        total=total, skip=page.skip, limit=page.limit,
    )


@router.get("/orders/{assignment_id}", response_model=DeliveryAssignmentResponse)
async def get_delivery(
    assignment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    partner: DeliveryPartner = Depends(require_approved_delivery_partner),
):
    return _assignment_payload(await svc._owned(db, assignment_id, partner))


@router.post("/orders/{assignment_id}/advance", response_model=DeliveryAssignmentResponse)
async def advance_delivery(
    assignment_id: uuid.UUID,
    payload: AdvanceRequest,
    db: AsyncSession = Depends(get_db),
    partner: DeliveryPartner = Depends(require_approved_delivery_partner),
):
    """
    Move to the next leg: accept → en route → at pharmacy → picked up →
    out for delivery → at patient.

    `delivered` is not reachable here — completion requires the patient's OTP.
    Picking up and delivering also move the parent order, routed through
    `ordering_service` so its own transition table stays the authority.
    """
    assignment = await svc.advance(
        db, partner, assignment_id,
        target=payload.target, note=payload.note,
        latitude=payload.latitude, longitude=payload.longitude,
    )
    await db.commit()
    return _assignment_payload(await svc._owned(db, assignment.id, partner))


@router.post("/orders/{assignment_id}/fail", response_model=DeliveryAssignmentResponse)
async def fail_delivery(
    assignment_id: uuid.UUID,
    payload: FailRequest,
    db: AsyncSession = Depends(get_db),
    partner: DeliveryPartner = Depends(require_approved_delivery_partner),
):
    """
    Abandon a delivery that could not be completed.

    The order is left where it is: whether that becomes a return, a re-dispatch
    or a refund is a commercial call for the pharmacy, not the rider.
    """
    assignment = await svc.fail(
        db, partner, assignment_id, reason=payload.reason,
        latitude=payload.latitude, longitude=payload.longitude,
    )
    await db.commit()
    return _assignment_payload(await svc._owned(db, assignment.id, partner))


@router.post("/orders/{assignment_id}/verify-otp", response_model=DeliveryAssignmentResponse)
async def verify_otp(
    assignment_id: uuid.UUID,
    payload: VerifyOtpRequest,
    db: AsyncSession = Depends(get_db),
    partner: DeliveryPartner = Depends(require_approved_delivery_partner),
):
    """
    Confirm handover with the patient's code. The only path to `delivered`.

    Attempts are bounded and the code expires, so a rider cannot brute-force it
    at the doorstep.
    """
    assignment = await svc.verify_otp(
        db, partner, assignment_id, code=payload.code,
        latitude=payload.latitude, longitude=payload.longitude,
    )
    await db.commit()
    return _assignment_payload(await svc._owned(db, assignment.id, partner))


@router.post("/orders/{assignment_id}/proof", response_model=DeliveryAssignmentResponse)
async def capture_proof(
    assignment_id: uuid.UUID,
    payload: ProofRequest,
    db: AsyncSession = Depends(get_db),
    partner: DeliveryPartner = Depends(require_approved_delivery_partner),
):
    """Photo, signature, notes and the GPS fix they were captured at."""
    assignment = await svc.capture_proof(
        db, partner, assignment_id,
        photo_url=payload.photo_url, signature_url=payload.signature_url,
        notes=payload.notes, latitude=payload.latitude, longitude=payload.longitude,
    )
    await db.commit()
    return _assignment_payload(await svc._owned(db, assignment.id, partner))


@router.get("/orders/{assignment_id}/route", response_model=RouteResponse)
async def delivery_route(
    assignment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    partner: DeliveryPartner = Depends(require_approved_delivery_partner),
):
    """
    Navigation for the current leg — pharmacy before pickup, patient after.

    Live distance and ETA come from Distance Matrix when a Maps key is
    configured; `maps_enabled` says whether that happened.
    """
    assignment = await svc._owned(db, assignment_id, partner)
    return RouteResponse(**await svc.route(assignment))


# ── patient tracking ─────────────────────────────────────────────────────


@router.get("/tracking/{order_id}", response_model=Optional[DeliveryTrackingResponse])
async def track_order(
    order_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Live tracking for the patient who placed this order.

    Returns null when no rider is assigned yet — a normal state, not an error.
    The payload is deliberately narrow: name, photo, vehicle, rating, position
    and ETA, and nothing else about the rider.
    """
    if getattr(current_user, "role", None) != "patient":
        raise AuthorizationException("Only the patient may track their own order.")

    data = await svc.tracking_for_order(db, order_id, current_user.id)
    return DeliveryTrackingResponse(**data) if data else None


# ── admin fleet management ───────────────────────────────────────────────


@admin_router.get("/partners", response_model=list[DeliveryPartnerResponse])
async def list_partners(
    verification_status: Optional[str] = Query(None),
    online_only: bool = Query(False),
    page: Pagination = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    query = select(DeliveryPartner).where(DeliveryPartner.deleted_at.is_(None))
    if verification_status:
        query = query.where(DeliveryPartner.verification_status == verification_status)
    if online_only:
        query = query.where(DeliveryPartner.is_online.is_(True))

    rows = await db.execute(
        query.order_by(DeliveryPartner.created_at.desc())
        .offset(page.skip)
        .limit(page.limit)
    )
    return [_partner_payload(p) for p in rows.scalars().all()]


@admin_router.get("/partners/available", response_model=list[DeliveryPartnerResponse])
async def available_partners(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Approved, online, and not already carrying an order."""
    return [_partner_payload(p) for p in await svc.available_partners(db)]


@admin_router.post(
    "/partners", response_model=DeliveryPartnerResponse, status_code=status.HTTP_201_CREATED
)
async def create_partner(
    payload: PartnerCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a rider profile against an existing account.

    The account must already hold the `delivery_partner` role — provisioning
    the login itself is account administration, not fleet management.
    """
    user = await db.get(User, payload.user_id)
    if not user:
        raise EntityNotFoundException("User", str(payload.user_id))
    if user.role != "delivery_partner":
        raise AuthorizationException(
            f"{user.email} is a '{user.role}' account. Only a delivery_partner "
            "account can have a rider profile."
        )

    existing = (
        await db.execute(
            select(DeliveryPartner).where(DeliveryPartner.user_id == payload.user_id)
        )
    ).scalar_one_or_none()
    if existing:
        raise AuthorizationException("This account already has a rider profile.")

    partner = DeliveryPartner(**payload.model_dump())
    db.add(partner)
    await db.commit()
    await db.refresh(partner)
    logger.info("[DELIVERY_PARTNER_CREATED] partner=%s by=%s", partner.id, current_user.id)
    return _partner_payload(partner)


@admin_router.post(
    "/partners/{partner_id}/verification", response_model=DeliveryPartnerResponse
)
async def transition_partner(
    partner_id: uuid.UUID,
    payload: PartnerVerificationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Advance a rider's verification. Anything but approval takes them off the
    road immediately.
    """
    partner = await svc.transition_partner(
        db, partner_id, to_status=payload.to_status, note=payload.note, actor=current_user
    )
    await db.commit()
    return _partner_payload(partner)


@admin_router.post(
    "/assign", response_model=DeliveryAssignmentResponse, status_code=status.HTTP_201_CREATED
)
async def assign_order(
    payload: AssignOrderRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Hand a packed order to a rider.

    Refused unless the order is packed, the rider is approved, and neither is
    already committed elsewhere. Reassignment is the same call once the previous
    assignment has been cancelled or failed.
    """
    assignment = await svc.create_assignment(
        db, order_id=payload.order_id, partner_id=payload.partner_id,
        assigned_by=current_user.id,
    )
    await db.commit()
    return _assignment_payload(await svc._assignment(db, assignment.id))


@admin_router.get("/analytics", response_model=FleetAnalyticsResponse)
async def fleet_analytics(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return FleetAnalyticsResponse(**await svc.network_analytics(db, days=days))
