"""
Pharmacy administration endpoints.

Mounted at `/api/v1/admin/pharmacies`. The router carries
`RoleChecker(["admin"])`, matching the existing admin router, so every route
below is administrator-only by construction rather than by each handler
remembering to check — a patient or doctor token cannot reach any of them.

Nothing here is reachable from the patient or doctor portals, and no route
alters the dispensing path: `can_fulfil` and `availability` keep the semantics
Phase 2 shipped.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import RoleChecker, get_current_active_user, get_db, get_redis
from redis.asyncio import Redis
from app.api.pagination import Pagination, pagination_params
from app.core.exceptions import BusinessRuleValidationException
from app.models.user import User
from app.schemas.pharmacy_admin_api import (
    AuditEntryResponse,
    AuditListResponse,
    BulkStatusRequest,
    DocumentCreateRequest,
    DocumentReviewRequest,
    ImportResultResponse,
    InventoryListResponse,
    InventoryResponse,
    InventoryUpsertRequest,
    PharmacyAnalyticsResponse,
    PharmacyCreateRequest,
    PharmacyDetailResponse,
    PharmacyDocumentResponse,
    PharmacyListResponse,
    PharmacyResponse,
    PharmacyUpdateRequest,
    BulkStatusResponse,
    AssignOwnerRequest,
    ChangeOwnerRequest,
    CreateOwnerRequest,
    OwnerCredentialResponse,
    OwnerInvitationResponse,
    OwnerStatusRequest,
    PharmacyOwnerResponse,
    RemoveOwnerRequest,
    SetActiveRequest,
    StatusResponse,
    VerificationTransitionRequest,
)
from app.services.pharmacy_admin import pharmacy_admin_service
from app.services.pharmacy_owner import pharmacy_owner_service

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(RoleChecker(["admin"]))])

MAX_IMPORT_BYTES = 2 * 1024 * 1024
"""A CSV import is parsed into memory inside one transaction."""


def _client_ip(request: Request) -> str:
    """
    Caller IP for the audit trail.

    Only the hop our own proxy appended is trusted; anything further left in
    X-Forwarded-For was supplied by the client and is forgeable, so recording it
    as fact would put attacker-controlled text in the compliance log.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _pharmacy_payload(pharmacy) -> dict:
    data = {
        column: getattr(pharmacy, column)
        for column in PharmacyResponse.model_fields
        if hasattr(pharmacy, column)
    }
    data["can_fulfil"] = pharmacy.can_fulfil
    return data


# ── pharmacy CRUD ────────────────────────────────────────────────────────


@router.get("", response_model=PharmacyListResponse)
async def list_pharmacies(
    search: Optional[str] = Query(None, max_length=200),
    verification_status: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    is_partner: Optional[bool] = Query(None),
    city: Optional[str] = Query(None, max_length=120),
    page: Pagination = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Paged, filterable directory of the partner network."""
    items, total = await pharmacy_admin_service.list_pharmacies(
        db,
        skip=page.skip,
        limit=page.limit,
        search=search,
        verification_status=verification_status,
        is_active=is_active,
        is_partner=is_partner,
        city=city,
    )
    return PharmacyListResponse(
        items=[PharmacyResponse(**_pharmacy_payload(p)) for p in items],
        total=total,
        skip=page.skip,
        limit=page.limit,
    )


@router.post("", response_model=PharmacyDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_pharmacy(
    payload: PharmacyCreateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Onboard a pharmacy. It starts unverified and invisible to patients."""
    pharmacy = await pharmacy_admin_service.create(
        db, payload=payload.model_dump(exclude_unset=True),
        actor=current_user, ip=_client_ip(request),
    )
    await db.commit()
    fresh = await pharmacy_admin_service.get(db, pharmacy.id)
    return PharmacyDetailResponse(
        **_pharmacy_payload(fresh),
        documents=list(fresh.documents),
        verification_events=list(fresh.verification_events),
    )


@router.get("/analytics", response_model=PharmacyAnalyticsResponse)
async def pharmacy_analytics(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Network-wide metrics.

    Declared before `/{pharmacy_id}` so the literal path is matched first —
    otherwise "analytics" would be parsed as a UUID and 422.
    """
    data = await pharmacy_admin_service.analytics(db, days=days)
    data["top_medicines"] = await pharmacy_admin_service.top_medicines(db, days=days)
    return PharmacyAnalyticsResponse(**data)


@router.get("/audit", response_model=AuditListResponse)
async def pharmacy_audit_trail(
    resource_id: Optional[str] = Query(None, max_length=100),
    action: Optional[str] = Query(None, max_length=100),
    page: Pagination = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Every administrator action against the pharmacy network."""
    items, total = await pharmacy_admin_service.audit_trail(
        db, resource_id=resource_id, action=action, skip=page.skip, limit=page.limit
    )
    return AuditListResponse(
        items=[AuditEntryResponse.model_validate(entry) for entry in items],
        total=total,
        skip=page.skip,
        limit=page.limit,
    )


@router.get("/documents/expiring", response_model=list[PharmacyDocumentResponse])
async def expiring_documents(
    within_days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Compliance documents already expired or lapsing soon."""
    documents = await pharmacy_admin_service.expiring_documents(db, within_days=within_days)
    return [
        PharmacyDocumentResponse(
            **{
                field: getattr(doc, field)
                for field in PharmacyDocumentResponse.model_fields
                if hasattr(doc, field)
            }
        )
        for doc in documents
    ]


@router.get("/inventory", response_model=InventoryListResponse)
async def search_catalogue(
    search: Optional[str] = Query(None, max_length=200),
    pharmacy_id: Optional[uuid.UUID] = Query(None),
    category: Optional[str] = Query(None, max_length=120),
    manufacturer: Optional[str] = Query(None, max_length=200),
    stock_state: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    page: Pagination = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Medicine catalogue search across the whole network, or one pharmacy."""
    items, total = await pharmacy_admin_service.list_inventory(
        db, pharmacy_id,
        search=search, category=category, manufacturer=manufacturer,
        stock_state=stock_state, min_price=min_price, max_price=max_price,
        skip=page.skip, limit=page.limit,
    )
    return InventoryListResponse(
        items=[_inventory_payload(item) for item in items],
        total=total, skip=page.skip, limit=page.limit,
    )


@router.get("/{pharmacy_id}", response_model=PharmacyDetailResponse)
async def get_pharmacy(
    pharmacy_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    pharmacy = await pharmacy_admin_service.get(db, pharmacy_id)
    return PharmacyDetailResponse(
        **_pharmacy_payload(pharmacy),
        documents=list(pharmacy.documents),
        verification_events=list(pharmacy.verification_events),
    )


@router.put("/{pharmacy_id}", response_model=PharmacyDetailResponse)
async def update_pharmacy(
    pharmacy_id: uuid.UUID,
    payload: PharmacyUpdateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    await pharmacy_admin_service.update(
        db, pharmacy_id, payload=payload.model_dump(exclude_unset=True),
        actor=current_user, ip=_client_ip(request),
    )
    await db.commit()
    fresh = await pharmacy_admin_service.get(db, pharmacy_id)
    return PharmacyDetailResponse(
        **_pharmacy_payload(fresh),
        documents=list(fresh.documents),
        verification_events=list(fresh.verification_events),
    )


@router.post("/{pharmacy_id}/status", response_model=PharmacyResponse)
async def set_pharmacy_active(
    pharmacy_id: uuid.UUID,
    payload: SetActiveRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Suspend or reactivate. Suspension removes it from patient search."""
    pharmacy = await pharmacy_admin_service.set_active(
        db, pharmacy_id, active=payload.active, reason=payload.reason,
        actor=current_user, ip=_client_ip(request),
    )
    await db.commit()
    return PharmacyResponse(**_pharmacy_payload(pharmacy))


@router.post("/bulk/status", response_model=BulkStatusResponse)
async def bulk_set_status(
    payload: BulkStatusRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    changed = await pharmacy_admin_service.bulk_set_status(
        db, payload.pharmacy_ids, active=payload.active, reason=payload.reason,
        actor=current_user, ip=_client_ip(request),
    )
    await db.commit()
    return {"status": "ok", "updated": changed, "requested": len(payload.pharmacy_ids)}


@router.delete("/{pharmacy_id}", response_model=StatusResponse)
async def delete_pharmacy(
    pharmacy_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Retire a pharmacy. Soft delete — dispensing history is preserved."""
    await pharmacy_admin_service.soft_delete(
        db, pharmacy_id, actor=current_user, ip=_client_ip(request)
    )
    await db.commit()
    return {"status": "ok", "message": "Pharmacy retired."}


# ── verification ─────────────────────────────────────────────────────────


@router.post("/{pharmacy_id}/verification", response_model=PharmacyDetailResponse)
async def transition_verification(
    pharmacy_id: uuid.UUID,
    payload: VerificationTransitionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Advance the verification workflow.

    Approval grants partner status; rejection and suspension withdraw it, so the
    review state and the dispensing gate cannot disagree.
    """
    await pharmacy_admin_service.transition_verification(
        db, pharmacy_id, to_status=payload.to_status, note=payload.note,
        actor=current_user, ip=_client_ip(request),
    )
    await db.commit()
    fresh = await pharmacy_admin_service.get(db, pharmacy_id)
    return PharmacyDetailResponse(
        **_pharmacy_payload(fresh),
        documents=list(fresh.documents),
        verification_events=list(fresh.verification_events),
    )


@router.post(
    "/{pharmacy_id}/documents",
    response_model=PharmacyDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_document(
    pharmacy_id: uuid.UUID,
    payload: DocumentCreateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    document = await pharmacy_admin_service.add_document(
        db, pharmacy_id, payload=payload.model_dump(), actor=current_user,
        ip=_client_ip(request),
    )
    await db.commit()
    return PharmacyDocumentResponse(
        **{
            field: getattr(document, field)
            for field in PharmacyDocumentResponse.model_fields
            if hasattr(document, field)
        }
    )


@router.post("/documents/{document_id}/review", response_model=PharmacyDocumentResponse)
async def review_document(
    document_id: uuid.UUID,
    payload: DocumentReviewRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    document = await pharmacy_admin_service.review_document(
        db, document_id, status=payload.status, notes=payload.notes,
        actor=current_user, ip=_client_ip(request),
    )
    await db.commit()
    return PharmacyDocumentResponse(
        **{
            field: getattr(document, field)
            for field in PharmacyDocumentResponse.model_fields
            if hasattr(document, field)
        }
    )


# ── inventory ────────────────────────────────────────────────────────────


def _inventory_payload(item) -> InventoryResponse:
    data = {
        field: getattr(item, field)
        for field in InventoryResponse.model_fields
        if hasattr(item, field)
    }
    data["availability"] = item.availability
    data["stock_state"] = item.stock_state
    data["inventory_value"] = item.inventory_value
    return InventoryResponse(**data)


@router.post(
    "/{pharmacy_id}/inventory",
    response_model=InventoryResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_inventory_item(
    pharmacy_id: uuid.UUID,
    payload: InventoryUpsertRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    item = await pharmacy_admin_service.upsert_inventory(
        db, pharmacy_id, payload=payload.model_dump(exclude_unset=True),
        actor=current_user, ip=_client_ip(request),
    )
    await db.commit()
    return _inventory_payload(item)


@router.put("/{pharmacy_id}/inventory/{item_id}", response_model=InventoryResponse)
async def update_inventory_item(
    pharmacy_id: uuid.UUID,
    item_id: uuid.UUID,
    payload: InventoryUpsertRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    item = await pharmacy_admin_service.upsert_inventory(
        db, pharmacy_id, payload=payload.model_dump(exclude_unset=True),
        actor=current_user, ip=_client_ip(request), item_id=item_id,
    )
    await db.commit()
    return _inventory_payload(item)


@router.delete("/inventory/{item_id}", response_model=StatusResponse)
async def delete_inventory_item(
    item_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    await pharmacy_admin_service.delete_inventory(
        db, item_id, actor=current_user, ip=_client_ip(request)
    )
    await db.commit()
    return {"status": "ok", "message": "Inventory item removed."}


@router.get(
    "/{pharmacy_id}/inventory/export",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}, "description": "Inventory CSV"}},
)
async def export_inventory(
    pharmacy_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Inventory as CSV.

    Served as text/csv rather than JSON so a spreadsheet opens it directly; the
    same column set is what `import` accepts, making export→edit→import a
    round trip rather than two different formats.
    """
    items, _ = await pharmacy_admin_service.list_inventory(
        db, pharmacy_id, skip=0, limit=5000
    )
    body = pharmacy_admin_service.export_inventory_csv(items)
    return Response(
        content=body,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="inventory-{pharmacy_id}.csv"'
        },
    )


@router.post("/{pharmacy_id}/inventory/import", response_model=ImportResultResponse)
async def import_inventory(
    pharmacy_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Bulk-load stock from a CSV body, matched on `sku`.

    Bad rows are reported by line number and skipped; the rest still load.
    """
    raw = await request.body()
    if len(raw) > MAX_IMPORT_BYTES:
        raise BusinessRuleValidationException(
            f"The import file exceeds {MAX_IMPORT_BYTES // (1024 * 1024)} MB."
        )

    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise BusinessRuleValidationException(
            "The import file must be UTF-8 encoded CSV."
        ) from None

    result = await pharmacy_admin_service.import_inventory_csv(
        db, pharmacy_id, content=content, actor=current_user, ip=_client_ip(request)
    )
    await db.commit()
    return ImportResultResponse(**result)


# ── owner provisioning ───────────────────────────────────────────────────
#
# The last link between a verified pharmacy and a person who can operate it.
# Everything below reuses the existing authentication: the same `users` table,
# the same password hashing, the same tokens. No JWT change, no second identity
# system, no bypass.


def _owner_payload(user) -> PharmacyOwnerResponse:
    return PharmacyOwnerResponse(
        id=user.id,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        is_verified=user.is_verified,
        pharmacy_id=user.pharmacy_id,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


@router.get("/{pharmacy_id}/owners", response_model=list[PharmacyOwnerResponse])
async def list_pharmacy_owners(
    pharmacy_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Everyone ever linked to this store, active or revoked."""
    owners = await pharmacy_owner_service.list_owners(db, pharmacy_id)
    return [_owner_payload(owner) for owner in owners]


@router.post("/{pharmacy_id}/owners/assign", response_model=PharmacyOwnerResponse)
async def assign_pharmacy_owner(
    pharmacy_id: uuid.UUID,
    payload: AssignOwnerRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Link an existing account to this pharmacy.

    Refused if the store is unverified or suspended, if it already has an
    active owner, or if the account holds a patient/doctor/admin role.
    """
    owner = await pharmacy_owner_service.assign_existing_user(
        db, pharmacy_id, payload.user_id, actor=current_user, ip=_client_ip(request)
    )
    await db.commit()
    return _owner_payload(owner)


@router.post(
    "/{pharmacy_id}/owners",
    response_model=OwnerCredentialResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_pharmacy_owner(
    pharmacy_id: uuid.UUID,
    payload: CreateOwnerRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a dedicated owner account for this pharmacy.

    The temporary password is returned exactly once, here. Only its hash is
    stored, so there is no endpoint that can read it back.
    """
    owner, temporary = await pharmacy_owner_service.create_owner(
        db, pharmacy_id, email=payload.email, password=payload.password,
        actor=current_user, ip=_client_ip(request),
    )
    await db.commit()
    return OwnerCredentialResponse(
        owner=_owner_payload(owner), temporary_password=temporary
    )


@router.post("/{pharmacy_id}/owners/change", response_model=PharmacyOwnerResponse)
async def change_pharmacy_owner(
    pharmacy_id: uuid.UUID,
    payload: ChangeOwnerRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Hand the store to a different operator.

    Remove-then-assign in one transaction, so the pharmacy is never left with
    two active owners or none.
    """
    owner = await pharmacy_owner_service.change_owner(
        db, pharmacy_id, payload.user_id, actor=current_user,
        reason=payload.reason, ip=_client_ip(request),
    )
    await db.commit()
    return _owner_payload(owner)


@router.post("/{pharmacy_id}/owners/remove", response_model=StatusResponse)
async def remove_pharmacy_owner(
    pharmacy_id: uuid.UUID,
    payload: RemoveOwnerRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Revoke the current owner.

    Refused while orders are in flight — removing the operator then would leave
    nobody able to dispatch medicine a patient is waiting on.
    """
    await pharmacy_owner_service.remove_owner(
        db, pharmacy_id, actor=current_user, reason=payload.reason,
        ip=_client_ip(request),
    )
    await db.commit()
    return StatusResponse(status="ok", message="Pharmacy access revoked.")


@router.post("/owners/{user_id}/status", response_model=PharmacyOwnerResponse)
async def set_owner_status(
    user_id: uuid.UUID,
    payload: OwnerStatusRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Suspend or reactivate an owner without unlinking them.

    Takes effect on the owner's next request: `require_verified_pharmacy` reads
    `is_active` live rather than trusting the session.
    """
    owner = await pharmacy_owner_service.set_owner_active(
        db, user_id, active=payload.active, actor=current_user,
        reason=payload.reason, ip=_client_ip(request),
    )
    await db.commit()
    return _owner_payload(owner)


@router.post("/owners/{user_id}/reset-password", response_model=OwnerCredentialResponse)
async def reset_owner_password(
    user_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Issue a new temporary password. Returned once; only the hash is stored."""
    owner, temporary = await pharmacy_owner_service.reset_password(
        db, user_id, actor=current_user, ip=_client_ip(request)
    )
    await db.commit()
    return OwnerCredentialResponse(
        owner=_owner_payload(owner), temporary_password=temporary
    )


@router.post("/owners/{user_id}/invite", response_model=OwnerInvitationResponse)
async def invite_owner(
    user_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    redis: Redis = Depends(get_redis),
):
    """
    Invite an owner into the portal.

    Delivered as an in-app notification and a one-time password returned here.
    `email_sent` is reported as false because the platform's email task is a
    stub that never contacts a mail server — claiming otherwise would be false.
    """
    result = await pharmacy_owner_service.send_invitation(
        db, user_id, actor=current_user, ip=_client_ip(request), redis=redis
    )
    await db.commit()
    return OwnerInvitationResponse(**result)
