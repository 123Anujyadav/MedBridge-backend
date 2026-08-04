"""
Pharmacy search, ordering and delivery tracking.

Mounted at `/api/v1/pharmacy`. Every route is patient-scoped from the bearer
token: the signed-in patient's id is taken from the session, never from the
request body, so there is no field to tamper with to reach another patient's
prescription or order.
"""

from __future__ import annotations

import logging
import uuid
from typing import List

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_user, get_db
from app.core.exceptions import AuthorizationException, EntityNotFoundException
from app.models.medicine_order import ORDER_OUT_FOR_DELIVERY, MedicineOrder
from app.models.user import User
from app.pharmacy.application.agent import pharmacy_agent
from app.pharmacy.application.ordering import ordering_service
from app.pharmacy.factory import get_search_service
from app.schemas.pharmacy_api import (
    AdvanceOrderRequest,
    GeocodeResultResponse,
    ReverseGeocodeResponse,
    CancelOrderRequest,
    MedicineOrderResponse,
    PharmacyOfferResponse,
    PharmacySearchResponse,
    PlaceOrderRequest,
)
from app.services.maps import get_maps_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _require_patient(user: User) -> uuid.UUID:
    if getattr(user, "role", None) != "patient":
        raise AuthorizationException("Only patients can use the pharmacy workflow.")
    return user.id


def _offer_payload(offer) -> PharmacyOfferResponse:
    return PharmacyOfferResponse(
        pharmacy_id=offer.pharmacy_id,
        name=offer.name,
        address=offer.address,
        phone=offer.phone,
        latitude=offer.latitude,
        longitude=offer.longitude,
        rating=offer.rating,
        total_ratings=offer.total_ratings,
        is_partner=offer.is_partner,
        is_24x7=offer.is_24x7,
        is_open_now=offer.is_open_now,
        delivers=offer.delivers,
        distance_km=offer.distance_km,
        travel_minutes=offer.travel_minutes,
        eta_minutes=offer.eta_minutes,
        distance_source=offer.distance_source,
        delivery_fee=offer.delivery_fee,
        min_order_value=offer.min_order_value,
        subtotal=offer.subtotal,
        total_savings=offer.total_savings,
        grand_total=offer.grand_total,
        items=[
            {
                **vars(item),
                "line_total": item.line_total,
                "savings": item.savings,
                "alternatives": [vars(a) for a in item.alternatives],
            }
            for item in offer.items
        ],
        can_order=offer.can_order,
        fully_available=offer.fully_available,
        fulfilment_ratio=round(offer.fulfilment_ratio, 2),
        unavailable_items=offer.unavailable_items,
        badges=offer.badges,
        score=offer.score,
        map_url=offer.map_url,
        directions_url=offer.directions_url,
    )


@router.get("/search", response_model=PharmacySearchResponse)
async def search_pharmacies(
    prescription_id: uuid.UUID,
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(10.0, gt=0, le=50),
    limit: int = Query(5, ge=1, le=5),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Nearby pharmacies ranked for this prescription.

    Availability, distance, ETA, rating and price are all computed before the
    assistant is asked to describe them — the summary explains a result, it
    never produces one.
    """
    patient_id = _require_patient(current_user)
    service = get_search_service()

    offers = await service.offers_for_prescription(
        db,
        prescription_id=prescription_id,
        patient_id=patient_id,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        limit=limit,
    )

    medicine_count = len(offers[0].items) if offers else 0
    summary = await pharmacy_agent.summarise_offers(offers, medicine_count=medicine_count)

    return PharmacySearchResponse(
        prescription_id=prescription_id,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        offers=[_offer_payload(o) for o in offers],
        assistant_summary=summary,
        maps_enabled=get_maps_service().is_enabled(),
        provider=service._provider.name,  # noqa: SLF001 - reported for diagnostics
    )


@router.post("/orders", response_model=MedicineOrderResponse, status_code=status.HTTP_201_CREATED)
async def place_order(
    payload: PlaceOrderRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Place an order and reserve its stock.

    Stock is decremented in the same transaction, so two patients cannot both
    buy the last pack.
    """
    patient_id = _require_patient(current_user)

    order = await ordering_service.place_order(
        db,
        patient_id=patient_id,
        prescription_id=payload.prescription_id,
        pharmacy_id=payload.pharmacy_id,
        selections=[item.model_dump() for item in payload.items],
        delivery_address=payload.delivery_address,
        delivery_latitude=payload.delivery_latitude,
        delivery_longitude=payload.delivery_longitude,
        delivery_notes=payload.delivery_notes,
        distance_km=payload.distance_km,
        eta_minutes=payload.eta_minutes,
    )
    await db.commit()

    fresh = await ordering_service.get_for_patient(db, order.id, patient_id)
    return _order_response(fresh)


@router.get("/orders", response_model=List[MedicineOrderResponse])
async def list_orders(
    limit: int = Query(25, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """The signed-in patient's order history, newest first."""
    patient_id = _require_patient(current_user)
    orders = await ordering_service.list_for_patient(db, patient_id, limit=limit)
    return [_order_response(o, include_events=False) for o in orders]


@router.get("/orders/{order_id}", response_model=MedicineOrderResponse)
async def track_order(
    order_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """One order with its full status trail."""
    patient_id = _require_patient(current_user)
    order = await ordering_service.get_for_patient(db, order_id, patient_id)
    return _order_response(order)


@router.post("/orders/{order_id}/cancel", response_model=MedicineOrderResponse)
async def cancel_order(
    order_id: uuid.UUID,
    payload: CancelOrderRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Cancel before dispatch. Reserved stock is returned to the pharmacy.

    Refused once the order is out for delivery — at that point the goods have
    left the counter and it is a return, not a cancellation.
    """
    patient_id = _require_patient(current_user)
    order = await ordering_service.get_for_patient(db, order_id, patient_id)

    await ordering_service.cancel_order(
        db, order, reason=payload.reason, actor_id=patient_id
    )
    await db.commit()

    fresh = await ordering_service.get_for_patient(db, order_id, patient_id)
    return _order_response(fresh)


@router.post("/orders/{order_id}/status", response_model=MedicineOrderResponse)
async def advance_order_status(
    order_id: uuid.UUID,
    payload: AdvanceOrderRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Move an order along its lifecycle.

    Admin-only for now. Partner pharmacies get their own authenticated channel
    when the pharmacy portal ships; until then a human operator advances the
    status, and letting patients do it would let them mark their own order
    delivered.
    """
    if getattr(current_user, "role", None) != "admin":
        raise AuthorizationException(
            "Only an administrator can change an order's fulfilment status."
        )

    order = await db.get(MedicineOrder, order_id)
    if not order:
        raise EntityNotFoundException("Order", str(order_id))

    await ordering_service.advance_status(
        db, order, payload.status, note=payload.note, actor_type="admin",
        actor_id=current_user.id,
    )
    if payload.status == ORDER_OUT_FOR_DELIVERY:
        order.delivery_partner_name = payload.delivery_partner_name
        order.delivery_partner_phone = payload.delivery_partner_phone

    await db.commit()

    fresh = await ordering_service.get_for_patient(db, order_id, order.patient_id)
    return _order_response(fresh)


def _order_response(order, include_events: bool = True) -> MedicineOrderResponse:
    return MedicineOrderResponse(
        id=order.id,
        order_number=order.order_number,
        prescription_id=order.prescription_id,
        pharmacy_id=order.pharmacy_id,
        pharmacy_name=order.pharmacy_name,
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
        events=list(order.events) if include_events else [],
    )


# ── geocoding (OpenStreetMap) ────────────────────────────────────────────
#
# Address search and reverse lookup, proxied through the backend rather than
# called from the browser. Nominatim's usage policy requires a real User-Agent
# and rate limiting; a browser cannot honour either, and thousands of clients
# hitting it directly would get the platform blocked. The shared limiter lives
# in `MapsService`.


@router.get("/geocode/search", response_model=list[GeocodeResultResponse])
async def geocode_search(
    q: str = Query(min_length=3, max_length=200),
    limit: int = Query(5, ge=1, le=10),
    current_user: User = Depends(get_current_active_user),
):
    """
    Forward geocoding for address autocomplete.

    Requires three characters — below that Nominatim returns noise, and
    querying per keystroke is what its policy asks callers not to do.
    Returns an empty list on failure so a typeahead cannot break its form.
    """
    results = await get_maps_service().forward_geocode(q, limit=limit)
    return [GeocodeResultResponse(**item) for item in results]


@router.get("/geocode/reverse", response_model=ReverseGeocodeResponse)
async def geocode_reverse(
    latitude: float = Query(ge=-90, le=90),
    longitude: float = Query(ge=-180, le=180),
    current_user: User = Depends(get_current_active_user),
):
    """The street address for a coordinate pair. `address` is null if unknown."""
    address = await get_maps_service().reverse_geocode(latitude, longitude)
    return ReverseGeocodeResponse(
        latitude=latitude, longitude=longitude, address=address
    )
