"""
Delivery and logistics.

The final leg: a rider carries medicine from a verified pharmacy to the
patient, and the handover is confirmed with an OTP the patient holds.

Four rules run through the module:

1. **The order's own lifecycle is not re-implemented.** `medicine_orders` keeps
   the coarse status Phase 2 defined and moves through `ordering_service`; the
   assignment tracks the finer legs beside it. Both sides stay in step because
   only one of them owns each fact.

2. **A delivery is not complete until the OTP verifies.** The rider cannot mark
   it delivered on their own say-so — that is the whole point of the code.

3. **Only the hash of the OTP is stored.** A rider able to read the code from
   the API would not need the patient present.

4. **Assignment is scoped structurally.** Every rider-facing method takes the
   `DeliveryPartner` resolved by the auth gate, so there is no id in a request
   that could reach another rider's job.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from sqlalchemy import Integer, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import (
    AuthorizationException,
    BusinessRuleValidationException,
    EntityNotFoundException,
)
from app.core.security import get_password_hash, verify_password
from app.models.delivery import (
    ACTIVE_STATUSES,
    DELIVERY_ACCEPTED,
    DELIVERY_AT_PATIENT,
    DELIVERY_AT_PHARMACY,
    DELIVERY_CANCELLED,
    DELIVERY_DELIVERED,
    DELIVERY_EN_ROUTE_PICKUP,
    DELIVERY_FAILED,
    DELIVERY_OFFERED,
    DELIVERY_OUT_FOR_DELIVERY,
    DELIVERY_PICKED_UP,
    DELIVERY_TRANSITIONS,
    DeliveryAssignment,
    DeliveryEvent,
    DeliveryPartner,
    PARTNER_APPROVED,
    PARTNER_STATUSES,
    PARTNER_TRANSITIONS,
)
from app.models.medicine_order import (
    MedicineOrder,
    ORDER_DELIVERED,
    ORDER_OUT_FOR_DELIVERY,
    ORDER_PACKED,
)
from app.models.pharmacy import Pharmacy
from app.models.user import User
from app.pharmacy.application.ordering import ordering_service
from app.services.maps import get_maps_service
from app.services.notifications import notification_service

logger = logging.getLogger(__name__)

OTP_LENGTH = 6
OTP_VALIDITY_MINUTES = 30
MAX_OTP_ATTEMPTS = 5
"""
Bounded so a rider cannot brute-force a six-digit code at the doorstep. Five
attempts against a million possibilities is not a meaningful attack surface,
and it is generous for someone reading a code aloud.
"""

# Legs that move the parent order. Everything else is finer-grained progress
# the order does not model, so it stays purely on the assignment.
ORDER_SYNC = {
    DELIVERY_PICKED_UP: ORDER_OUT_FOR_DELIVERY,
    DELIVERY_DELIVERED: ORDER_DELIVERED,
}

DEFAULT_PARTNER_SHARE = 0.8
"""Share of the delivery fee the rider earns. Platform keeps the remainder."""


def generate_otp() -> str:
    """A numeric code the patient can read aloud."""
    return "".join(secrets.choice("0123456789") for _ in range(OTP_LENGTH))


class DeliveryService:
    # ── partner profile ──────────────────────────────────────────────────

    async def get_partner_by_user(
        self, db: AsyncSession, user_id: uuid.UUID
    ) -> DeliveryPartner:
        result = await db.execute(
            select(DeliveryPartner).where(
                DeliveryPartner.user_id == user_id,
                DeliveryPartner.deleted_at.is_(None),
            )
        )
        partner = result.scalar_one_or_none()
        if not partner:
            raise EntityNotFoundException("DeliveryPartner", str(user_id))
        return partner

    async def set_online(
        self, db: AsyncSession, partner: DeliveryPartner, *, online: bool
    ) -> DeliveryPartner:
        """
        Clock on or off.

        Going offline does not drop work already accepted — a rider carrying
        medicine still has to deliver it. It only stops new offers.
        """
        partner.is_online = online
        await db.flush()
        logger.info(
            "[DELIVERY_PARTNER_%s] partner=%s",
            "ONLINE" if online else "OFFLINE", partner.id,
        )
        return partner

    async def update_location(
        self,
        db: AsyncSession,
        partner: DeliveryPartner,
        *,
        latitude: float,
        longitude: float,
    ) -> DeliveryPartner:
        """
        Record the rider's position for live tracking.

        The timestamp is stored with it and surfaced to the patient, so a stale
        fix reads as stale rather than as the rider's position now.
        """
        partner.current_latitude = latitude
        partner.current_longitude = longitude
        partner.location_updated_at = datetime.now(timezone.utc)
        await db.flush()
        return partner

    # ── assignment ───────────────────────────────────────────────────────

    async def _assignment(
        self, db: AsyncSession, assignment_id: uuid.UUID
    ) -> DeliveryAssignment:
        result = await db.execute(
            select(DeliveryAssignment)
            .where(DeliveryAssignment.id == assignment_id)
            .options(selectinload(DeliveryAssignment.events))
            # Refreshes the event trail rather than returning the identity
            # map's copy — without it, reading an assignment back in the same
            # transaction that just appended a leg omits it.
            .execution_options(populate_existing=True)
        )
        assignment = result.scalar_one_or_none()
        if not assignment:
            raise EntityNotFoundException("DeliveryAssignment", str(assignment_id))
        return assignment

    async def _owned(
        self, db: AsyncSession, assignment_id: uuid.UUID, partner: DeliveryPartner
    ) -> DeliveryAssignment:
        assignment = await self._assignment(db, assignment_id)
        if assignment.partner_id != partner.id:
            logger.warning(
                "[DELIVERY_ACCESS_DENIED] partner=%s tried assignment=%s of partner=%s",
                partner.id, assignment_id, assignment.partner_id,
            )
            raise AuthorizationException("This delivery belongs to another partner.")
        return assignment

    def _record(
        self,
        db: AsyncSession,
        assignment: DeliveryAssignment,
        *,
        status: str,
        note: str = "",
        latitude: float | None = None,
        longitude: float | None = None,
        actor_type: str = "partner",
        actor_id: uuid.UUID | None = None,
    ) -> None:
        db.add(
            DeliveryEvent(
                assignment_id=assignment.id,
                status=status,
                note=note[:500],
                latitude=latitude,
                longitude=longitude,
                actor_type=actor_type,
                actor_id=actor_id,
            )
        )

    async def create_assignment(
        self,
        db: AsyncSession,
        *,
        order_id: uuid.UUID,
        partner_id: uuid.UUID,
        assigned_by: uuid.UUID | None = None,
    ) -> DeliveryAssignment:
        """
        Offer an order to a rider.

        Refused unless the order is packed and ready to leave the counter, and
        unless the rider is approved and not already carrying something.
        """
        order = await db.get(MedicineOrder, order_id)
        if not order:
            raise EntityNotFoundException("Order", str(order_id))

        if order.status != ORDER_PACKED:
            raise BusinessRuleValidationException(
                f"Order {order.order_number} is '{order.status}'. Only a packed order "
                "can be handed to a rider."
            )

        existing = await db.scalar(
            select(func.count()).select_from(DeliveryAssignment).where(
                DeliveryAssignment.order_id == order_id,
                DeliveryAssignment.status.in_(ACTIVE_STATUSES),
            )
        )
        if existing:
            raise BusinessRuleValidationException(
                f"Order {order.order_number} already has an active delivery assignment."
            )

        partner = await db.get(DeliveryPartner, partner_id)
        if not partner or partner.deleted_at is not None:
            raise EntityNotFoundException("DeliveryPartner", str(partner_id))
        if not partner.is_approved:
            raise BusinessRuleValidationException(
                f"{partner.full_name} is '{partner.verification_status}' and cannot "
                "take deliveries."
            )

        busy = await db.scalar(
            select(func.count()).select_from(DeliveryAssignment).where(
                DeliveryAssignment.partner_id == partner_id,
                DeliveryAssignment.status.in_(ACTIVE_STATUSES),
            )
        )
        if busy:
            raise BusinessRuleValidationException(
                f"{partner.full_name} is already carrying a delivery."
            )

        pharmacy = await db.get(Pharmacy, order.pharmacy_id)
        fee = order.delivery_fee or 0.0

        assignment = DeliveryAssignment(
            order_id=order_id,
            partner_id=partner_id,
            pharmacy_id=order.pharmacy_id,
            status=DELIVERY_OFFERED,
            pickup_address=(pharmacy.address if pharmacy else "") or "",
            pickup_latitude=pharmacy.latitude if pharmacy else None,
            pickup_longitude=pharmacy.longitude if pharmacy else None,
            drop_address=order.delivery_address or "",
            drop_latitude=order.delivery_latitude,
            drop_longitude=order.delivery_longitude,
            distance_km=order.distance_km,
            eta_minutes=order.eta_minutes,
            delivery_fee=fee,
            partner_earning=round(fee * DEFAULT_PARTNER_SHARE, 2),
            offered_at=datetime.now(timezone.utc),
            assigned_by=assigned_by,
        )
        db.add(assignment)
        await db.flush()

        self._record(
            db, assignment, status=DELIVERY_OFFERED,
            note=f"Offered to {partner.full_name}.",
            actor_type="admin" if assigned_by else "system", actor_id=assigned_by,
        )

        await notification_service.notify(
            db,
            user_id=partner.user_id,
            category="system",
            type="delivery_assigned",
            title="New delivery offered",
            message=f"Pickup from {pharmacy.name if pharmacy else 'a pharmacy'}.",
            priority="high",
            action_url="/delivery/orders",
            action_label="View delivery",
            dedupe_key=f"delivery-offer-{assignment.id}",
        )

        logger.info(
            "[DELIVERY_ASSIGNED] assignment=%s order=%s partner=%s",
            assignment.id, order_id, partner_id,
        )
        return assignment

    async def list_for_partner(
        self,
        db: AsyncSession,
        partner: DeliveryPartner,
        *,
        status: str | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[list[DeliveryAssignment], int]:
        query = select(DeliveryAssignment).where(
            DeliveryAssignment.partner_id == partner.id,
            DeliveryAssignment.deleted_at.is_(None),
        )
        if status == "active":
            query = query.where(DeliveryAssignment.status.in_(ACTIVE_STATUSES))
        elif status:
            query = query.where(DeliveryAssignment.status == status)

        total = await db.scalar(select(func.count()).select_from(query.subquery()))
        rows = await db.execute(
            query.order_by(DeliveryAssignment.created_at.desc()).offset(skip).limit(limit)
        )
        return list(rows.scalars().all()), int(total or 0)

    # ── the journey ──────────────────────────────────────────────────────

    async def advance(
        self,
        db: AsyncSession,
        partner: DeliveryPartner,
        assignment_id: uuid.UUID,
        *,
        target: str,
        note: str = "",
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> DeliveryAssignment:
        """
        Move a delivery to its next leg.

        `delivered` is deliberately not reachable here — it is only produced by
        `verify_otp`, so a rider cannot mark medicine handed over without the
        patient confirming it.
        """
        assignment = await self._owned(db, assignment_id, partner)

        if target == DELIVERY_DELIVERED:
            raise BusinessRuleValidationException(
                "A delivery is completed by verifying the patient's OTP, not by "
                "marking it delivered."
            )

        if not assignment.can_transition_to(target):
            allowed = DELIVERY_TRANSITIONS.get(assignment.status, ())
            raise BusinessRuleValidationException(
                f"A delivery that is '{assignment.status}' cannot become '{target}'. "
                f"Allowed: {', '.join(allowed) if allowed else 'none — this is final'}."
            )

        assignment.status = target
        now = datetime.now(timezone.utc)

        if target == DELIVERY_ACCEPTED:
            assignment.accepted_at = now
        elif target == DELIVERY_PICKED_UP:
            assignment.picked_up_at = now

        if latitude is not None and longitude is not None:
            await self.update_location(db, partner, latitude=latitude, longitude=longitude)

        self._record(
            db, assignment, status=target, note=note,
            latitude=latitude, longitude=longitude, actor_id=partner.user_id,
        )

        # Keep the parent order in step for the two legs it models. Routed
        # through `ordering_service` so the order's own transition table stays
        # the single authority on what the order may do.
        order_target = ORDER_SYNC.get(target)
        if order_target:
            order = await db.get(MedicineOrder, assignment.order_id)
            if order and order.can_transition_to(order_target):
                await ordering_service.advance_status(
                    db, order, order_target,
                    note=f"Rider {partner.full_name}: {target.replace('_', ' ')}.",
                    actor_type="delivery_partner", actor_id=partner.user_id,
                )
                order.delivery_partner_name = partner.full_name
                order.delivery_partner_phone = partner.phone

        # The OTP is issued when the rider sets off, not at the doorstep, so
        # the patient has it in hand before the rider arrives.
        if target == DELIVERY_OUT_FOR_DELIVERY and not assignment.otp_hash:
            await self.issue_otp(db, assignment)

        await db.flush()
        logger.info(
            "[DELIVERY_STATUS] assignment=%s -> %s partner=%s",
            assignment.id, target, partner.id,
        )
        return assignment

    async def fail(
        self,
        db: AsyncSession,
        partner: DeliveryPartner,
        assignment_id: uuid.UUID,
        *,
        reason: str,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> DeliveryAssignment:
        """
        Abandon a delivery that could not be completed.

        The order is deliberately left where it is rather than cancelled: the
        medicine has usually left the pharmacy, and deciding whether that is a
        return, a re-dispatch or a refund is a commercial call for the pharmacy
        and the patient, not the rider.
        """
        assignment = await self._owned(db, assignment_id, partner)
        if not assignment.can_transition_to(DELIVERY_FAILED):
            raise BusinessRuleValidationException(
                f"A delivery that is '{assignment.status}' cannot be failed."
            )

        assignment.status = DELIVERY_FAILED
        assignment.failure_reason = reason[:500]
        assignment.cancelled_at = datetime.now(timezone.utc)
        partner.failed_deliveries += 1

        self._record(
            db, assignment, status=DELIVERY_FAILED, note=reason,
            latitude=latitude, longitude=longitude, actor_id=partner.user_id,
        )
        await db.flush()
        logger.info("[DELIVERY_FAILED] assignment=%s reason=%s", assignment.id, reason[:80])
        return assignment

    # ── OTP ──────────────────────────────────────────────────────────────

    async def issue_otp(
        self, db: AsyncSession, assignment: DeliveryAssignment
    ) -> str:
        """
        Mint the handover code and send it to the patient.

        Returns the plaintext for the caller to deliver; only the hash is
        stored. Reissuing resets the attempt counter, so a patient who mistyped
        five times is not locked out of their own delivery.
        """
        code = generate_otp()
        assignment.otp_hash = get_password_hash(code)
        assignment.otp_issued_at = datetime.now(timezone.utc)
        assignment.otp_attempts = 0
        await db.flush()

        order = await db.get(MedicineOrder, assignment.order_id)
        if order:
            await notification_service.notify(
                db,
                user_id=order.patient_id,
                category="system",
                type="delivery_otp",
                title="Your delivery code",
                message=(
                    f"Share {code} with the delivery partner when they arrive. "
                    "Do not share it before your medicine is handed over."
                ),
                priority="high",
                action_url=f"/patient/orders/{order.id}",
                action_label="Track delivery",
                dedupe_key=f"delivery-otp-{assignment.id}-{assignment.otp_issued_at}",
            )

        logger.info("[DELIVERY_OTP_ISSUED] assignment=%s", assignment.id)
        return code

    async def verify_otp(
        self,
        db: AsyncSession,
        partner: DeliveryPartner,
        assignment_id: uuid.UUID,
        *,
        code: str,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> DeliveryAssignment:
        """
        Confirm handover and complete the delivery.

        This is the only path to `delivered`. The order follows through
        `ordering_service`, and the rider's statistics are credited here.
        """
        assignment = await self._owned(db, assignment_id, partner)

        if assignment.status != DELIVERY_AT_PATIENT:
            raise BusinessRuleValidationException(
                "Mark yourself at the patient's address before verifying the code."
            )
        if not assignment.otp_hash:
            raise BusinessRuleValidationException(
                "No delivery code has been issued for this order yet."
            )
        if assignment.is_otp_verified:
            raise BusinessRuleValidationException("This delivery is already confirmed.")

        if assignment.otp_attempts >= MAX_OTP_ATTEMPTS:
            raise BusinessRuleValidationException(
                "Too many incorrect attempts. Ask support to reissue the code."
            )

        issued = assignment.otp_issued_at
        if issued:
            if issued.tzinfo is None:
                issued = issued.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - issued > timedelta(minutes=OTP_VALIDITY_MINUTES):
                raise BusinessRuleValidationException(
                    "This delivery code has expired. Ask for a new one to be sent."
                )

        if not verify_password(code.strip(), assignment.otp_hash):
            # Counted before raising, so a wrong code costs an attempt whether
            # or not the caller retries.
            assignment.otp_attempts += 1
            await db.flush()
            remaining = MAX_OTP_ATTEMPTS - assignment.otp_attempts
            raise BusinessRuleValidationException(
                f"That code is not correct. {max(0, remaining)} attempt(s) remaining."
            )

        now = datetime.now(timezone.utc)
        assignment.otp_verified_at = now
        assignment.status = DELIVERY_DELIVERED
        assignment.delivered_at = now

        partner.completed_deliveries += 1
        partner.total_distance_km = round(
            (partner.total_distance_km or 0.0) + (assignment.distance_km or 0.0), 2
        )
        partner.total_earnings = round(
            (partner.total_earnings or 0.0) + (assignment.partner_earning or 0.0), 2
        )

        self._record(
            db, assignment, status=DELIVERY_DELIVERED,
            note="Handover confirmed with the patient's code.",
            latitude=latitude, longitude=longitude, actor_id=partner.user_id,
        )

        order = await db.get(MedicineOrder, assignment.order_id)
        if order and order.can_transition_to(ORDER_DELIVERED):
            await ordering_service.advance_status(
                db, order, ORDER_DELIVERED,
                note=f"Delivered by {partner.full_name}, confirmed by code.",
                actor_type="delivery_partner", actor_id=partner.user_id,
            )

        await db.flush()
        logger.info("[DELIVERY_COMPLETED] assignment=%s partner=%s", assignment.id, partner.id)
        return assignment

    # ── proof of delivery ────────────────────────────────────────────────

    async def capture_proof(
        self,
        db: AsyncSession,
        partner: DeliveryPartner,
        assignment_id: uuid.UUID,
        *,
        photo_url: str | None = None,
        signature_url: str | None = None,
        notes: str = "",
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> DeliveryAssignment:
        """
        Attach evidence of the handover.

        Accepted from `at_patient` onward: a rider photographs the doorstep as
        they arrive, and requiring completion first would mean capturing proof
        after the moment it documents.
        """
        assignment = await self._owned(db, assignment_id, partner)

        if assignment.status not in (DELIVERY_AT_PATIENT, DELIVERY_DELIVERED):
            raise BusinessRuleValidationException(
                "Proof of delivery can only be captured at the patient's address."
            )

        if photo_url:
            assignment.proof_photo_url = photo_url
        if signature_url:
            assignment.proof_signature_url = signature_url
        if notes:
            assignment.delivery_notes = notes[:2000]

        assignment.proof_latitude = latitude
        assignment.proof_longitude = longitude
        assignment.proof_captured_at = datetime.now(timezone.utc)

        self._record(
            db, assignment, status=assignment.status,
            note="Proof of delivery captured.",
            latitude=latitude, longitude=longitude, actor_id=partner.user_id,
        )
        await db.flush()
        return assignment

    # ── routing ──────────────────────────────────────────────────────────

    async def route(self, assignment: DeliveryAssignment) -> dict:
        """
        Navigation for the current leg.

        Before pickup the rider heads to the pharmacy; afterwards to the
        patient. Distance and ETA come from Distance Matrix when a Maps key is
        configured, and `maps_enabled` is reported so the client can explain a
        missing figure rather than showing a blank.
        """
        heading_to_pharmacy = assignment.status in (
            DELIVERY_OFFERED, DELIVERY_ACCEPTED, DELIVERY_EN_ROUTE_PICKUP, DELIVERY_AT_PHARMACY
        )
        if heading_to_pharmacy:
            destination = (assignment.pickup_latitude, assignment.pickup_longitude)
            label = assignment.pickup_address
        else:
            destination = (assignment.drop_latitude, assignment.drop_longitude)
            label = assignment.drop_address

        partner = assignment.partner
        origin = (
            (partner.current_latitude, partner.current_longitude)
            if partner and partner.current_latitude is not None
            else (assignment.pickup_latitude, assignment.pickup_longitude)
        )

        maps = get_maps_service()
        payload: dict[str, Any] = {
            "destination_label": label,
            "destination_latitude": destination[0],
            "destination_longitude": destination[1],
            "heading_to": "pharmacy" if heading_to_pharmacy else "patient",
            "maps_enabled": maps.is_enabled(),
            "distance_km": assignment.distance_km,
            "eta_minutes": assignment.eta_minutes,
            "navigation_url": None,
            "map_url": None,
        }

        if destination[0] is not None and destination[1] is not None:
            payload["map_url"] = (
                f"https://www.openstreetmap.org/?mlat={destination[0]}"
                f"&mlon={destination[1]}#map=17/{destination[0]}/{destination[1]}"
            )
            if origin[0] is not None and origin[1] is not None:
                payload["navigation_url"] = (
                    "https://www.openstreetmap.org/directions"
                    "?engine=fossgis_osrm_car"
                    f"&route={origin[0]}%2C{origin[1]}"
                    f"%3B{destination[0]}%2C{destination[1]}"
                )
                live = await maps.distance_matrix(origin, destination)
                if live:
                    payload["distance_km"] = live["distance_km"]
                    payload["eta_minutes"] = live["duration_minutes"]
                    payload["distance_text"] = live.get("distance_text")
                    payload["duration_text"] = live.get("duration_text")

        return payload

    # ── dashboard & analytics ────────────────────────────────────────────

    async def dashboard(self, db: AsyncSession, partner: DeliveryPartner) -> dict:
        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        mine = DeliveryAssignment.partner_id == partner.id

        by_status = {
            row[0]: row[1]
            for row in (
                await db.execute(
                    select(DeliveryAssignment.status, func.count(DeliveryAssignment.id))
                    .where(mine, DeliveryAssignment.created_at >= start_of_day)
                    .group_by(DeliveryAssignment.status)
                )
            ).all()
        }

        today = (
            await db.execute(
                select(
                    func.coalesce(func.sum(DeliveryAssignment.partner_earning), 0.0),
                    func.coalesce(func.sum(DeliveryAssignment.distance_km), 0.0),
                    func.count(DeliveryAssignment.id),
                ).where(
                    mine,
                    DeliveryAssignment.delivered_at >= start_of_day,
                    DeliveryAssignment.status == DELIVERY_DELIVERED,
                )
            )
        ).one()

        # Average delivery time is measured from acceptance to handover, which
        # is the span the rider actually controls.
        spans = (
            await db.execute(
                select(DeliveryAssignment.accepted_at, DeliveryAssignment.delivered_at)
                .where(
                    mine,
                    DeliveryAssignment.status == DELIVERY_DELIVERED,
                    DeliveryAssignment.accepted_at.isnot(None),
                    DeliveryAssignment.delivered_at.isnot(None),
                )
                .limit(200)
            )
        ).all()
        minutes = [
            (delivered - accepted).total_seconds() / 60
            for accepted, delivered in spans
            if delivered and accepted and delivered > accepted
        ]

        return {
            "partner_id": str(partner.id),
            "full_name": partner.full_name,
            "is_online": partner.is_online,
            "verification_status": partner.verification_status,
            "deliveries_today": sum(by_status.values()),
            "by_status": by_status,
            "active_count": await db.scalar(
                select(func.count()).select_from(DeliveryAssignment).where(
                    mine, DeliveryAssignment.status.in_(ACTIVE_STATUSES)
                )
            ) or 0,
            "earnings_today": round(float(today[0] or 0), 2),
            "distance_today_km": round(float(today[1] or 0), 2),
            "delivered_today": int(today[2] or 0),
            "average_delivery_minutes": (
                round(sum(minutes) / len(minutes), 1) if minutes else 0.0
            ),
            "completion_rate": partner.completion_rate,
            "rating": round(partner.rating or 0.0, 2),
            "total_ratings": partner.total_ratings,
            "lifetime_deliveries": partner.completed_deliveries,
            "lifetime_distance_km": round(partner.total_distance_km or 0.0, 2),
            "lifetime_earnings": round(partner.total_earnings or 0.0, 2),
        }

    async def network_analytics(self, db: AsyncSession, *, days: int = 30) -> dict:
        """Fleet-wide metrics for the administrator."""
        since = datetime.now(timezone.utc) - timedelta(days=days)
        recent = DeliveryAssignment.created_at >= since

        totals = (
            await db.execute(
                select(
                    func.count(DeliveryAssignment.id),
                    func.coalesce(func.sum(DeliveryAssignment.distance_km), 0.0),
                    func.coalesce(func.sum(DeliveryAssignment.delivery_fee), 0.0),
                    func.coalesce(func.avg(DeliveryAssignment.eta_minutes), 0.0),
                ).where(recent)
            )
        ).one()

        by_status = {
            row[0]: row[1]
            for row in (
                await db.execute(
                    select(DeliveryAssignment.status, func.count(DeliveryAssignment.id))
                    .where(recent)
                    .group_by(DeliveryAssignment.status)
                )
            ).all()
        }

        delivered = by_status.get(DELIVERY_DELIVERED, 0)
        failed = by_status.get(DELIVERY_FAILED, 0)
        cancelled = by_status.get(DELIVERY_CANCELLED, 0)
        attempted = delivered + failed

        top = [
            {
                "partner": row[0],
                "deliveries": row[1],
                "distance_km": round(float(row[2] or 0), 2),
            }
            for row in (
                await db.execute(
                    select(
                        DeliveryPartner.full_name,
                        func.count(DeliveryAssignment.id),
                        func.sum(DeliveryAssignment.distance_km),
                    )
                    .join(
                        DeliveryAssignment,
                        DeliveryAssignment.partner_id == DeliveryPartner.id,
                    )
                    .where(recent, DeliveryAssignment.status == DELIVERY_DELIVERED)
                    .group_by(DeliveryPartner.full_name)
                    .order_by(func.count(DeliveryAssignment.id).desc())
                    .limit(10)
                )
            ).all()
        ]

        fleet = (
            await db.execute(
                select(
                    func.count(DeliveryPartner.id),
                    func.sum(func.cast(DeliveryPartner.is_online, Integer)),
                ).where(
                    DeliveryPartner.deleted_at.is_(None),
                    DeliveryPartner.verification_status == PARTNER_APPROVED,
                )
            )
        ).one()

        return {
            "window_days": days,
            "assignments": int(totals[0] or 0),
            "delivered": delivered,
            "failed": failed,
            "cancelled": cancelled,
            "success_rate": round(delivered / attempted, 4) if attempted else 0.0,
            "total_distance_km": round(float(totals[1] or 0), 2),
            "delivery_revenue": round(float(totals[2] or 0), 2),
            "average_eta_minutes": round(float(totals[3] or 0), 1),
            "by_status": by_status,
            "top_partners": top,
            "partners_approved": int(fleet[0] or 0),
            "partners_online": int(fleet[1] or 0),
        }

    # ── patient-facing tracking ──────────────────────────────────────────

    async def tracking_for_order(
        self, db: AsyncSession, order_id: uuid.UUID, patient_id: uuid.UUID
    ) -> dict | None:
        """
        What the patient may see about their rider.

        Deliberately narrow: name, photo, vehicle, rating, live position and
        ETA. The rider's licence number, address, earnings and other
        assignments are theirs, and none of it helps a patient waiting at a
        door.

        Returns None when no rider has been assigned yet — a normal state, not
        an error.
        """
        order = await db.get(MedicineOrder, order_id)
        if not order:
            raise EntityNotFoundException("Order", str(order_id))
        if order.patient_id != patient_id:
            raise AuthorizationException("This order belongs to another patient.")

        result = await db.execute(
            select(DeliveryAssignment)
            .where(DeliveryAssignment.order_id == order_id)
            .options(selectinload(DeliveryAssignment.events))
            .order_by(DeliveryAssignment.created_at.desc())
            .limit(1)
        )
        assignment = result.scalar_one_or_none()
        if not assignment:
            return None

        partner = await db.get(DeliveryPartner, assignment.partner_id)

        return {
            "assignment_id": str(assignment.id),
            "status": assignment.status,
            "partner_name": partner.full_name if partner else "Assigned rider",
            "partner_photo_url": partner.photo_url if partner else None,
            "partner_phone": partner.phone if partner else None,
            "partner_rating": round(partner.rating, 2) if partner else 0.0,
            "vehicle_type": partner.vehicle_type if partner else None,
            "vehicle_number": partner.vehicle_number if partner else None,
            "current_latitude": partner.current_latitude if partner else None,
            "current_longitude": partner.current_longitude if partner else None,
            # Paired with the position so a stale fix is visibly stale.
            "location_updated_at": (
                partner.location_updated_at.isoformat()
                if partner and partner.location_updated_at
                else None
            ),
            "eta_minutes": assignment.eta_minutes,
            "distance_km": assignment.distance_km,
            "estimated_arrival_at": (
                assignment.estimated_arrival_at.isoformat()
                if assignment.estimated_arrival_at
                else None
            ),
            "otp_required": assignment.status == DELIVERY_AT_PATIENT
            and not assignment.is_otp_verified,
            "delivered_at": (
                assignment.delivered_at.isoformat() if assignment.delivered_at else None
            ),
            "events": [
                {
                    "status": event.status,
                    "note": event.note,
                    "created_at": event.created_at.isoformat(),
                }
                for event in assignment.events
            ],
        }

    # ── admin fleet management ───────────────────────────────────────────

    async def transition_partner(
        self,
        db: AsyncSession,
        partner_id: uuid.UUID,
        *,
        to_status: str,
        note: str,
        actor: User,
    ) -> DeliveryPartner:
        """Advance a rider's verification, refusing illegal transitions."""
        if to_status not in PARTNER_STATUSES:
            raise BusinessRuleValidationException(f"Unknown status '{to_status}'.")

        partner = await db.get(DeliveryPartner, partner_id)
        if not partner or partner.deleted_at is not None:
            raise EntityNotFoundException("DeliveryPartner", str(partner_id))

        current = partner.verification_status
        allowed = PARTNER_TRANSITIONS.get(current, ())
        if to_status not in allowed:
            raise BusinessRuleValidationException(
                f"A partner that is '{current}' cannot become '{to_status}'. "
                f"Allowed: {', '.join(allowed) if allowed else 'none'}."
            )

        partner.verification_status = to_status
        partner.verification_notes = note[:1000] or partner.verification_notes

        if to_status == PARTNER_APPROVED:
            partner.verified_at = datetime.now(timezone.utc)
            partner.verified_by = actor.id
            partner.suspension_reason = None
        else:
            # Anything other than approval takes them off the road immediately.
            partner.is_online = False
            if to_status == "suspended":
                partner.suspension_reason = note[:500]

        await db.flush()
        logger.info(
            "[DELIVERY_PARTNER_VERIFICATION] partner=%s %s -> %s by=%s",
            partner.id, current, to_status, actor.id,
        )
        return partner

    async def available_partners(
        self, db: AsyncSession, *, limit: int = 50
    ) -> list[DeliveryPartner]:
        """
        Approved riders who are online and not already carrying something.

        The busy check is a subquery rather than a Python filter so a growing
        fleet does not mean loading every rider to answer one dispatch screen.
        """
        busy = (
            select(DeliveryAssignment.partner_id)
            .where(DeliveryAssignment.status.in_(ACTIVE_STATUSES))
            .subquery()
        )
        result = await db.execute(
            select(DeliveryPartner)
            .where(
                DeliveryPartner.deleted_at.is_(None),
                DeliveryPartner.verification_status == PARTNER_APPROVED,
                DeliveryPartner.is_online.is_(True),
                DeliveryPartner.id.notin_(select(busy.c.partner_id)),
            )
            .limit(limit)
        )
        return list(result.scalars().all())


delivery_service = DeliveryService()
