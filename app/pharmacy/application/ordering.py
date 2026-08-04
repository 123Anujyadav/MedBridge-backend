"""
Medicine ordering and the delivery lifecycle.

Two rules run through everything here:

1. **Stock is decremented inside the same transaction that creates the order.**
   Checking availability and then writing the order in separate steps lets two
   patients buy the last box of the same medicine; the reservation is part of
   placing the order or it does not happen.

2. **Status changes go through `ORDER_TRANSITIONS`.** An order moving from
   `received` straight to `delivered`, or being cancelled after dispatch, is
   rejected rather than accepted-and-logged. These transitions move money and
   physical goods.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import (
    AuthorizationException,
    EntityNotFoundException,
    BusinessRuleValidationException,
)
from app.models.medicine_order import (
    CANCELLABLE_STATUSES,
    MedicineOrder,
    MedicineOrderItem,
    ORDER_CANCELLED,
    ORDER_DELIVERED,
    ORDER_OUT_FOR_DELIVERY,
    ORDER_RECEIVED,
    ORDER_TRANSITIONS,
    OrderStatusEvent,
)
from app.models.pharmacy import Pharmacy, PharmacyInventory
from app.models.prescription import Prescription

logger = logging.getLogger(__name__)


def generate_order_number() -> str:
    """
    A short, human-quotable reference.

    `secrets` rather than a sequence: an incrementing order number tells any
    customer roughly how many orders the platform has taken, and lets them
    guess their neighbour's.
    """
    return f"MB-{secrets.token_hex(4).upper()}"


class OrderingService:
    # ── placing ──────────────────────────────────────────────────────────

    async def place_order(
        self,
        db: AsyncSession,
        *,
        patient_id: uuid.UUID,
        prescription_id: uuid.UUID,
        pharmacy_id: uuid.UUID,
        selections: Sequence[dict],
        delivery_address: str,
        delivery_latitude: float | None = None,
        delivery_longitude: float | None = None,
        delivery_notes: str = "",
        distance_km: float | None = None,
        eta_minutes: int | None = None,
    ) -> MedicineOrder:
        """
        Create an order and reserve its stock.

        `selections` is a list of {inventory_id, quantity, medication_id?,
        is_generic_substitute?}. The client sends explicit inventory ids because
        a substitution must be something the patient chose, not something the
        server picked on their behalf.
        """
        prescription = await db.get(Prescription, prescription_id)
        if not prescription:
            raise EntityNotFoundException("Prescription", str(prescription_id))
        if prescription.patient_id != patient_id:
            # Ordering against someone else's prescription would both dispense
            # to the wrong person and expose what they were prescribed.
            logger.warning(
                "[ORDER_DENIED] patient=%s tried prescription=%s", patient_id, prescription_id
            )
            raise AuthorizationException(
                "You cannot order against a prescription that is not yours."
            )

        pharmacy = await db.get(Pharmacy, pharmacy_id)
        if not pharmacy:
            raise EntityNotFoundException("Pharmacy", str(pharmacy_id))
        if not pharmacy.can_fulfil:
            raise BusinessRuleValidationException(
                f"{pharmacy.name} is not accepting orders through MedBridge."
            )

        if not selections:
            raise BusinessRuleValidationException("An order must contain at least one medicine.")

        inventory_ids = [uuid.UUID(str(s["inventory_id"])) for s in selections]
        rows = (
            await db.execute(
                select(PharmacyInventory)
                .where(
                    PharmacyInventory.id.in_(inventory_ids),
                    PharmacyInventory.deleted_at.is_(None),
                )
                .with_for_update()
            )
        ).scalars().all()
        # SELECT ... FOR UPDATE holds these rows for the rest of the
        # transaction, so a concurrent order cannot pass the stock check between
        # our read and our decrement.

        by_id = {str(row.id): row for row in rows}

        order = MedicineOrder(
            order_number=generate_order_number(),
            patient_id=patient_id,
            prescription_id=prescription_id,
            pharmacy_id=pharmacy_id,
            pharmacy_name=pharmacy.name,
            status=ORDER_RECEIVED,
            delivery_address=delivery_address,
            delivery_latitude=delivery_latitude,
            delivery_longitude=delivery_longitude,
            delivery_notes=delivery_notes or "",
            distance_km=distance_km,
            eta_minutes=eta_minutes,
            placed_at=datetime.now(timezone.utc),
            fulfilment_provider="local_db",
        )
        if eta_minutes:
            order.estimated_delivery_at = datetime.now(timezone.utc) + timedelta(
                minutes=eta_minutes
            )

        db.add(order)
        await db.flush()

        subtotal = 0.0
        discount_total = 0.0

        for selection in selections:
            row = by_id.get(str(selection["inventory_id"]))
            if row is None:
                raise BusinessRuleValidationException(
                    "One of the selected medicines is no longer listed."
                )
            if row.pharmacy_id != pharmacy_id:
                raise BusinessRuleValidationException(
                    "All medicines in an order must come from the same pharmacy."
                )

            quantity = max(1, int(selection.get("quantity") or 1))
            if not row.can_supply(quantity):
                raise BusinessRuleValidationException(
                    f"{row.medicine_name} has only {row.stock_quantity} left; "
                    f"{quantity} were requested."
                )

            unit_price = round(row.selling_price or row.mrp, 2)
            line_total = round(unit_price * quantity, 2)
            subtotal += line_total
            discount_total += round(max(0.0, row.mrp - unit_price) * quantity, 2)

            medication_id = selection.get("medication_id")
            db.add(
                MedicineOrderItem(
                    order_id=order.id,
                    inventory_id=row.id,
                    medication_id=uuid.UUID(str(medication_id)) if medication_id else None,
                    medicine_name=row.medicine_name,
                    generic_name=row.generic_name,
                    brand_name=row.brand_name,
                    strength=row.strength,
                    rxcui=row.rxcui,
                    quantity=quantity,
                    unit_price=unit_price,
                    mrp=round(row.mrp, 2),
                    discount_percent=round(row.discount_percent, 2),
                    line_total=line_total,
                    is_generic_substitute=bool(selection.get("is_generic_substitute")),
                    substituted_for=selection.get("substituted_for"),
                )
            )

            row.stock_quantity -= quantity

        fee = pharmacy.delivery_fee or 0.0
        if pharmacy.free_delivery_above and subtotal >= pharmacy.free_delivery_above:
            fee = 0.0
        if subtotal < (pharmacy.min_order_value or 0.0):
            raise BusinessRuleValidationException(
                f"{pharmacy.name} has a minimum order of {pharmacy.min_order_value:.2f}."
            )

        order.subtotal = round(subtotal, 2)
        order.discount_total = round(discount_total, 2)
        order.delivery_fee = round(fee, 2)
        order.total = round(subtotal + fee, 2)

        db.add(
            OrderStatusEvent(
                order_id=order.id,
                status=ORDER_RECEIVED,
                note=f"Order placed with {pharmacy.name}.",
                actor_type="patient",
                actor_id=patient_id,
            )
        )

        logger.info(
            "[ORDER_PLACED] order=%s patient=%s pharmacy=%s items=%d total=%.2f",
            order.order_number, patient_id, pharmacy_id, len(selections), order.total,
        )
        return order

    # ── lifecycle ────────────────────────────────────────────────────────

    async def advance_status(
        self,
        db: AsyncSession,
        order: MedicineOrder,
        target: str,
        *,
        note: str = "",
        actor_type: str = "pharmacy",
        actor_id: uuid.UUID | None = None,
    ) -> MedicineOrder:
        """Move an order forward, refusing any transition the table forbids."""
        if not order.can_transition_to(target):
            allowed = ORDER_TRANSITIONS.get(order.status, ())
            raise BusinessRuleValidationException(
                f"An order that is '{order.status}' cannot become '{target}'. "
                f"Allowed: {', '.join(allowed) if allowed else 'none — this is a final state'}."
            )

        order.status = target
        now = datetime.now(timezone.utc)
        if target == ORDER_OUT_FOR_DELIVERY:
            order.dispatched_at = now
        elif target == ORDER_DELIVERED:
            order.delivered_at = now

        db.add(
            OrderStatusEvent(
                order_id=order.id,
                status=target,
                note=note or f"Order {target.replace('_', ' ')}.",
                actor_type=actor_type,
                actor_id=actor_id,
            )
        )
        logger.info("[ORDER_STATUS] order=%s -> %s", order.order_number, target)
        return order

    async def cancel_order(
        self,
        db: AsyncSession,
        order: MedicineOrder,
        *,
        reason: str,
        actor_id: uuid.UUID | None = None,
    ) -> MedicineOrder:
        """
        Cancel before dispatch and return the stock.

        Restoring the reserved quantity is the half that is easy to forget:
        without it every cancellation permanently removes stock the pharmacy
        still physically holds.
        """
        if order.status not in CANCELLABLE_STATUSES:
            raise BusinessRuleValidationException(
                f"This order is already {order.status.replace('_', ' ')} and can no "
                "longer be cancelled."
            )

        item_result = await db.execute(
            select(MedicineOrderItem).where(MedicineOrderItem.order_id == order.id)
        )
        for item in item_result.scalars().all():
            if not item.inventory_id:
                continue
            row = await db.get(PharmacyInventory, item.inventory_id)
            if row:
                row.stock_quantity += item.quantity

        order.status = ORDER_CANCELLED
        order.cancelled_at = datetime.now(timezone.utc)
        order.cancellation_reason = reason[:500]

        db.add(
            OrderStatusEvent(
                order_id=order.id,
                status=ORDER_CANCELLED,
                note=reason[:500],
                actor_type="patient" if actor_id else "system",
                actor_id=actor_id,
            )
        )
        logger.info("[ORDER_CANCELLED] order=%s reason=%s", order.order_number, reason[:80])
        return order

    # ── reads ────────────────────────────────────────────────────────────

    async def get_for_patient(
        self, db: AsyncSession, order_id: uuid.UUID, patient_id: uuid.UUID
    ) -> MedicineOrder:
        result = await db.execute(
            select(MedicineOrder)
            .where(MedicineOrder.id == order_id)
            .options(
                selectinload(MedicineOrder.items),
                selectinload(MedicineOrder.events),
            )
        )
        order = result.scalar_one_or_none()
        if not order:
            raise EntityNotFoundException("Order", str(order_id))
        if order.patient_id != patient_id:
            raise AuthorizationException("This order belongs to another patient.")
        return order

    async def list_for_patient(
        self, db: AsyncSession, patient_id: uuid.UUID, *, limit: int = 25
    ) -> list[MedicineOrder]:
        result = await db.execute(
            select(MedicineOrder)
            .where(
                MedicineOrder.patient_id == patient_id,
                MedicineOrder.deleted_at.is_(None),
            )
            .options(selectinload(MedicineOrder.items))
            .order_by(MedicineOrder.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


ordering_service = OrderingService()
