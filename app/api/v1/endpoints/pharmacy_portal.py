"""
Pharmacy Owner Portal endpoints.

Mounted at `/api/v1/pharmacy-portal`. Every route depends on
`require_verified_pharmacy`, which checks role, store link and live approval
per request — so a suspended store stops working immediately rather than at the
next token refresh.

No route accepts a pharmacy id. The store is derived from the signed-in owner's
own user row, which means there is no parameter to tamper with to reach another
pharmacy's orders, stock or customers. Patients, doctors and administrators are
all refused by the same gate.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, require_verified_pharmacy
from app.api.pagination import Pagination, pagination_params
from app.core.exceptions import BusinessRuleValidationException, EntityNotFoundException
from app.models.user import User
from app.schemas.pharmacy_admin_api import (
    ImportResultResponse,
    InventoryListResponse,
    InventoryResponse,
    InventoryUpsertRequest,
    StatusResponse,
)
from app.schemas.pharmacy_portal_api import (
    OrderActionRequest,
    PortalAlertResponse,
    PortalAnalyticsResponse,
    PortalCustomerResponse,
    PortalDashboardResponse,
    PortalOrderListResponse,
    PortalOrderResponse,
    PrescriptionReviewRequest,
    PrescriptionReviewResponse,
)
from app.services.pharmacy_portal import pharmacy_portal_service as portal

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_IMPORT_BYTES = 2 * 1024 * 1024


def _client_ip(request: Request) -> str:
    """Only the proxy-appended hop is trusted; the rest is client-supplied."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _order_payload(order) -> PortalOrderResponse:
    return PortalOrderResponse(
        id=order.id,
        order_number=order.order_number,
        prescription_id=order.prescription_id,
        patient_id=order.patient_id,
        status=order.status,
        subtotal=order.subtotal,
        discount_total=order.discount_total,
        delivery_fee=order.delivery_fee,
        total=order.total,
        currency=order.currency,
        delivery_address=order.delivery_address,
        delivery_notes=order.delivery_notes or "",
        distance_km=order.distance_km,
        eta_minutes=order.eta_minutes,
        estimated_delivery_at=order.estimated_delivery_at,
        delivery_partner_name=order.delivery_partner_name,
        delivery_partner_phone=order.delivery_partner_phone,
        placed_at=order.placed_at,
        dispatched_at=order.dispatched_at,
        delivered_at=order.delivered_at,
        cancelled_at=order.cancelled_at,
        cancellation_reason=order.cancellation_reason,
        is_cancellable=order.is_cancellable,
        created_at=order.created_at,
        items=list(order.items),
        events=list(getattr(order, "events", []) or []),
    )


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


# ── dashboard ────────────────────────────────────────────────────────────


@router.get("/dashboard", response_model=PortalDashboardResponse)
async def dashboard(
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    """Today's operating picture for the signed-in store."""
    return PortalDashboardResponse(**await portal.dashboard(db, owner))


@router.get("/alerts", response_model=list[PortalAlertResponse])
async def alerts(
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    """
    Live operational alerts.

    Derived from current state rather than stored, so a resolved shortage stops
    appearing the moment stock is corrected — there is nothing to mark as read.
    """
    return [PortalAlertResponse(**alert) for alert in await portal.alerts(db, owner)]


# ── orders ───────────────────────────────────────────────────────────────


@router.get("/orders", response_model=PortalOrderListResponse)
async def list_orders(
    status_filter: Optional[str] = Query(None, alias="status"),
    search: Optional[str] = Query(None, max_length=200),
    page: Pagination = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    """The store's order queue. `status=active` returns everything in flight."""
    items, total = await portal.list_orders(
        db, owner, status=status_filter, search=search,
        skip=page.skip, limit=page.limit,
    )
    return PortalOrderListResponse(
        items=[_order_payload(o) for o in items],
        total=total, skip=page.skip, limit=page.limit,
    )


@router.get("/orders/{order_id}", response_model=PortalOrderResponse)
async def get_order(
    order_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    return _order_payload(await portal.get_order(db, owner, order_id))


@router.post("/orders/{order_id}/action", response_model=PortalOrderResponse)
async def act_on_order(
    order_id: uuid.UUID,
    payload: OrderActionRequest,
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    """
    Advance or reject an order.

    Actions map onto the shared lifecycle rather than adding states:
    accept→preparing, ready→packed, dispatch→out_for_delivery, reject→cancelled
    (which returns the reserved stock). The patient's tracking timeline is the
    same machine, so both sides always agree.
    """
    await portal.act_on_order(
        db, owner, order_id,
        action=payload.action, note=payload.note,
        delivery_partner_name=payload.delivery_partner_name,
        delivery_partner_phone=payload.delivery_partner_phone,
    )
    await db.commit()
    return _order_payload(await portal.get_order(db, owner, order_id))


# ── prescription review ──────────────────────────────────────────────────


@router.get("/orders/{order_id}/prescription", response_model=PrescriptionReviewResponse)
async def prescription_for_order(
    order_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    """
    Everything the counter needs before dispensing: prescriber, medicines, the
    AI safety review with evidence, recorded allergies and batch expiry alerts.

    Strictly read-only — no field returned here is writable through this portal.
    """
    return PrescriptionReviewResponse(**await portal.prescription_for_order(db, owner, order_id))


@router.post("/orders/{order_id}/prescription/review", response_model=PortalOrderResponse)
async def review_prescription(
    order_id: uuid.UUID,
    payload: PrescriptionReviewRequest,
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    """
    Record the dispensing decision.

    Written to the order's event trail, never onto the prescription: a refusal
    is a fact about this dispensing attempt, not an amendment to what the
    doctor wrote.
    """
    await portal.record_prescription_review(
        db, owner, order_id, outcome=payload.outcome, note=payload.note
    )
    await db.commit()
    return _order_payload(await portal.get_order(db, owner, order_id))


# ── inventory ────────────────────────────────────────────────────────────


@router.get("/inventory", response_model=InventoryListResponse)
async def list_inventory(
    search: Optional[str] = Query(None, max_length=200),
    category: Optional[str] = Query(None, max_length=120),
    manufacturer: Optional[str] = Query(None, max_length=200),
    stock_state: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    page: Pagination = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    items, total = await portal.list_inventory(
        db, owner, search=search, category=category, manufacturer=manufacturer,
        stock_state=stock_state, min_price=min_price, max_price=max_price,
        skip=page.skip, limit=page.limit,
    )
    return InventoryListResponse(
        items=[_inventory_payload(i) for i in items],
        total=total, skip=page.skip, limit=page.limit,
    )


@router.get("/inventory/lookup", response_model=InventoryResponse)
async def lookup_by_code(
    code: str = Query(min_length=1, max_length=64),
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    """Barcode or QR scan at the counter. Matches barcode, then SKU."""
    item = await portal.find_by_code(db, owner, code)
    if not item:
        raise EntityNotFoundException("Inventory item", code)
    return _inventory_payload(item)


@router.post(
    "/inventory", response_model=InventoryResponse, status_code=status.HTTP_201_CREATED
)
async def create_inventory(
    payload: InventoryUpsertRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    item = await portal.upsert_inventory(
        db, owner, payload=payload.model_dump(exclude_unset=True), ip=_client_ip(request)
    )
    await db.commit()
    return _inventory_payload(item)


@router.put("/inventory/{item_id}", response_model=InventoryResponse)
async def update_inventory(
    item_id: uuid.UUID,
    payload: InventoryUpsertRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    """
    Update stock, batch, expiry, price, discount or GST.

    The write lands on the same `pharmacy_inventory` row patient search reads,
    so a stock correction is reflected in availability and nearest-pharmacy
    results on their next query with no extra propagation step.
    """
    item = await portal.upsert_inventory(
        db, owner, payload=payload.model_dump(exclude_unset=True),
        item_id=item_id, ip=_client_ip(request),
    )
    await db.commit()
    return _inventory_payload(item)


@router.delete("/inventory/{item_id}", response_model=StatusResponse)
async def delete_inventory(
    item_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    await portal.delete_inventory(db, owner, item_id, ip=_client_ip(request))
    await db.commit()
    return StatusResponse(status="ok", message="Item removed from your catalogue.")


@router.get(
    "/inventory/export",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}, "description": "Inventory CSV"}},
)
async def export_inventory(
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    body, name = await portal.export_inventory(db, owner)
    safe = "".join(c for c in name if c.isalnum() or c in " -_").strip() or "inventory"
    return Response(
        content=body,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{safe}-inventory.csv"'},
    )


@router.post("/inventory/import", response_model=ImportResultResponse)
async def import_inventory(
    request: Request,
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    """Bulk stock load. Bad rows are reported by line number and skipped."""
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

    result = await portal.import_inventory(db, owner, content=content, ip=_client_ip(request))
    await db.commit()
    return ImportResultResponse(**result)


# ── analytics, customers, reports ────────────────────────────────────────


@router.get("/analytics", response_model=PortalAnalyticsResponse)
async def analytics(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    return PortalAnalyticsResponse(**await portal.analytics(db, owner, days=days))


@router.get("/customers", response_model=list[PortalCustomerResponse])
async def customers(
    limit: int = Query(25, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    return [
        PortalCustomerResponse(**row)
        for row in await portal.customers(db, owner, limit=limit)
    ]


SALES_COLUMNS = (
    "order_number", "date", "status", "items",
    "subtotal", "discount", "delivery_fee", "gst", "total",
)


@router.get(
    "/reports/sales",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}, "description": "Sales report CSV"}},
)
async def sales_report(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    owner: User = Depends(require_verified_pharmacy),
):
    """
    Sales with per-line GST, as CSV.

    Tax is computed from each item's own `gst_percent` rather than a flat store
    rate, because different drug schedules carry different rates.
    """
    rows = await portal.sales_report(db, owner, days=days)
    body = portal.report_to_csv(rows, SALES_COLUMNS)
    return Response(
        content=body,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="sales-{days}d.csv"'},
    )
