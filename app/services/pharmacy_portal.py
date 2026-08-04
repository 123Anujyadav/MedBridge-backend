"""
Pharmacy Owner Portal.

Everything a verified partner does to run their own store: the order queue,
prescription review before dispensing, inventory, analytics and reports.

Three rules shape the module:

1. **Store scoping is structural, not a filter.** Every method takes the owner's
   `User` and derives the pharmacy from `user.pharmacy_id`. No endpoint accepts
   a store id, so there is no parameter an owner can change to reach another
   store's orders or stock.

2. **Nothing here duplicates Phase 2 or Phase 4.** Order transitions go through
   `ordering_service`, which owns the state machine and the stock arithmetic.
   Inventory goes through `pharmacy_admin_service`, which owns validation, CSV
   handling and the audit trail. This module scopes and presents; it does not
   re-implement.

3. **The prescription is read-only.** A pharmacy may accept, query or refuse to
   dispense. It may never alter what a doctor wrote — there is no code path
   here that writes to `prescriptions` or `medications`.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from sqlalchemy import Integer, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import (
    AuthorizationException,
    BusinessRuleValidationException,
    EntityNotFoundException,
)
from app.models.medicine_order import (
    MedicineOrder,
    MedicineOrderItem,
    ORDER_CANCELLED,
    ORDER_DELIVERED,
    ORDER_OUT_FOR_DELIVERY,
    ORDER_PACKED,
    ORDER_PREPARING,
    ORDER_RECEIVED,
)
from app.models.patient import Patient
from app.models.pharmacy import Pharmacy, PharmacyInventory
from app.models.prescription import Prescription
from app.models.rx_verification import PrescriptionVerification
from app.models.user import User
from app.pharmacy.application.ordering import ordering_service
from app.services.pharmacy_admin import pharmacy_admin_service

logger = logging.getLogger(__name__)

# The portal's vocabulary mapped onto the existing lifecycle. These are labels
# over `ORDER_TRANSITIONS`, not new states: adding states would change what the
# patient's tracking timeline renders, which is Phase 2 behaviour.
#
#   Accept  → preparing          Reject → cancelled (stock returned)
#   Ready   → packed             Dispatch → out_for_delivery
#
ACTION_TO_STATUS = {
    "accept": ORDER_PREPARING,
    "prepare": ORDER_PREPARING,
    "ready": ORDER_PACKED,
    "pack": ORDER_PACKED,
    "dispatch": ORDER_OUT_FOR_DELIVERY,
    "deliver": ORDER_DELIVERED,
}

ACTIVE_STATUSES = (ORDER_RECEIVED, ORDER_PREPARING, ORDER_PACKED, ORDER_OUT_FOR_DELIVERY)

# Prescription outcomes a pharmacy may record. Deliberately separate from the
# order status: refusing to dispense is a clinical judgement about the
# prescription, while cancelling is a commercial action on the order.
RX_REVIEW_OUTCOMES = ("approved", "clarification_requested", "rejected")


class PharmacyPortalService:
    # ── scoping ──────────────────────────────────────────────────────────

    async def _store(self, db: AsyncSession, owner: User) -> Pharmacy:
        """
        The store this owner operates.

        `require_verified_pharmacy` has already checked role, link and approval
        before any endpoint reaches here; this re-reads the row because the
        service is also called from tests and background jobs.
        """
        if not owner.pharmacy_id:
            raise AuthorizationException("This account is not linked to a pharmacy.")
        pharmacy = await db.get(Pharmacy, owner.pharmacy_id)
        if not pharmacy or pharmacy.deleted_at is not None:
            raise EntityNotFoundException("Pharmacy", str(owner.pharmacy_id))
        return pharmacy

    async def _owned_order(
        self, db: AsyncSession, owner: User, order_id: uuid.UUID
    ) -> MedicineOrder:
        result = await db.execute(
            select(MedicineOrder)
            .where(MedicineOrder.id == order_id)
            .options(
                selectinload(MedicineOrder.items),
                selectinload(MedicineOrder.events),
            )
            # Forces the eager collections to refresh instead of returning
            # whatever the identity map already held. Without it, reading an
            # order back inside the same transaction that just appended a
            # status event returns the stale trail — so the counter would not
            # see the clarification note it had just recorded.
            .execution_options(populate_existing=True)
        )
        order = result.scalar_one_or_none()
        if not order:
            raise EntityNotFoundException("Order", str(order_id))
        if order.pharmacy_id != owner.pharmacy_id:
            # Logged: an authenticated owner reaching for another store's order
            # is worth seeing whether it is a bug or a probe.
            logger.warning(
                "[PORTAL_ORDER_DENIED] owner=%s store=%s tried order=%s of store=%s",
                owner.id, owner.pharmacy_id, order_id, order.pharmacy_id,
            )
            raise AuthorizationException("This order belongs to another pharmacy.")
        return order

    # ── dashboard ────────────────────────────────────────────────────────

    async def dashboard(self, db: AsyncSession, owner: User) -> dict:
        """
        Today's operating picture, computed in SQL.

        Runs on every dashboard load, so counts are aggregated rather than
        fetched and tallied in Python.
        """
        store = await self._store(db, owner)
        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

        mine = MedicineOrder.pharmacy_id == store.id

        by_status = {
            row[0]: row[1]
            for row in (
                await db.execute(
                    select(MedicineOrder.status, func.count(MedicineOrder.id))
                    .where(mine, MedicineOrder.created_at >= start_of_day)
                    .group_by(MedicineOrder.status)
                )
            ).all()
        }

        async def revenue_since(moment: datetime) -> float:
            value = await db.scalar(
                select(func.coalesce(func.sum(MedicineOrder.total), 0.0)).where(
                    mine,
                    MedicineOrder.created_at >= moment,
                    MedicineOrder.status != ORDER_CANCELLED,
                )
            )
            return round(float(value or 0), 2)

        delivery = (
            await db.execute(
                select(
                    func.coalesce(func.avg(MedicineOrder.eta_minutes), 0.0),
                    func.count(MedicineOrder.id),
                ).where(mine, MedicineOrder.status == ORDER_DELIVERED)
            )
        ).one()

        # Preparation time is measured, not estimated: the gap between the order
        # arriving and the counter marking it dispatched.
        prep_rows = (
            await db.execute(
                select(MedicineOrder.placed_at, MedicineOrder.dispatched_at).where(
                    mine,
                    MedicineOrder.dispatched_at.isnot(None),
                    MedicineOrder.placed_at.isnot(None),
                )
                .limit(200)
            )
        ).all()
        prep_minutes = [
            (dispatched - placed).total_seconds() / 60
            for placed, dispatched in prep_rows
            if dispatched and placed and dispatched > placed
        ]

        stock_rows = (
            await db.execute(
                select(PharmacyInventory).where(
                    PharmacyInventory.pharmacy_id == store.id,
                    PharmacyInventory.deleted_at.is_(None),
                )
            )
        ).scalars().all()

        buckets: dict[str, int] = {}
        inventory_value = 0.0
        for row in stock_rows:
            buckets[row.stock_state] = buckets.get(row.stock_state, 0) + 1
            inventory_value += row.inventory_value

        pending_rx = await db.scalar(
            select(func.count()).select_from(MedicineOrder).where(
                mine, MedicineOrder.status == ORDER_RECEIVED
            )
        )

        return {
            "pharmacy_id": str(store.id),
            "pharmacy_name": store.name,
            "orders_today": sum(by_status.values()),
            "orders_by_status": by_status,
            "orders_active": await db.scalar(
                select(func.count()).select_from(MedicineOrder).where(
                    mine, MedicineOrder.status.in_(ACTIVE_STATUSES)
                )
            ) or 0,
            "revenue_today": await revenue_since(start_of_day),
            "revenue_week": await revenue_since(now - timedelta(days=7)),
            "revenue_month": await revenue_since(now - timedelta(days=30)),
            "average_delivery_minutes": round(float(delivery[0] or 0), 1),
            "orders_delivered_total": int(delivery[1] or 0),
            "average_prep_minutes": (
                round(sum(prep_minutes) / len(prep_minutes), 1) if prep_minutes else 0.0
            ),
            "customer_rating": round(store.rating or 0.0, 2),
            "total_ratings": store.total_ratings or 0,
            "pending_prescriptions": int(pending_rx or 0),
            "stock_low": buckets.get("low", 0),
            "stock_critical": buckets.get("critical", 0),
            "stock_out": buckets.get("out_of_stock", 0),
            "stock_near_expiry": buckets.get("near_expiry", 0),
            "stock_expired": buckets.get("expired", 0),
            "catalogue_size": len(stock_rows),
            "inventory_value": round(inventory_value, 2),
        }

    # ── order queue ──────────────────────────────────────────────────────

    async def list_orders(
        self,
        db: AsyncSession,
        owner: User,
        *,
        status: str | None = None,
        search: str | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[list[MedicineOrder], int]:
        store = await self._store(db, owner)
        query = (
            select(MedicineOrder)
            .where(
                MedicineOrder.pharmacy_id == store.id,
                MedicineOrder.deleted_at.is_(None),
            )
            .options(selectinload(MedicineOrder.items))
        )

        if status == "active":
            query = query.where(MedicineOrder.status.in_(ACTIVE_STATUSES))
        elif status:
            query = query.where(MedicineOrder.status == status)

        if search:
            term = f"%{search.strip().lower()}%"
            query = query.where(
                or_(
                    func.lower(MedicineOrder.order_number).like(term),
                    func.lower(MedicineOrder.delivery_address).like(term),
                )
            )

        total = await db.scalar(select(func.count()).select_from(query.subquery()))
        rows = await db.execute(
            query.order_by(MedicineOrder.created_at.desc()).offset(skip).limit(limit)
        )
        return list(rows.scalars().all()), int(total or 0)

    async def get_order(
        self, db: AsyncSession, owner: User, order_id: uuid.UUID
    ) -> MedicineOrder:
        return await self._owned_order(db, owner, order_id)

    async def act_on_order(
        self,
        db: AsyncSession,
        owner: User,
        order_id: uuid.UUID,
        *,
        action: str,
        note: str = "",
        delivery_partner_name: str | None = None,
        delivery_partner_phone: str | None = None,
    ) -> MedicineOrder:
        """
        Advance an order through the shared lifecycle.

        Delegates to `ordering_service`, which owns the transition table and
        (for rejection) the stock restoration. Re-implementing either here
        would give the portal and the patient's tracking screen two different
        ideas of what an order is doing.
        """
        order = await self._owned_order(db, owner, order_id)
        normalised = (action or "").strip().lower()

        if normalised == "reject":
            if order.status != ORDER_RECEIVED:
                raise BusinessRuleValidationException(
                    "An order can only be rejected before it is accepted."
                )
            await ordering_service.cancel_order(
                db, order, reason=note or "Rejected by pharmacy", actor_id=owner.id
            )
            logger.info("[PORTAL_ORDER_REJECTED] order=%s by=%s", order.order_number, owner.id)
            return order

        target = ACTION_TO_STATUS.get(normalised)
        if not target:
            raise BusinessRuleValidationException(
                f"'{action}' is not a recognised order action."
            )

        await ordering_service.advance_status(
            db, order, target, note=note, actor_type="pharmacy", actor_id=owner.id
        )

        if target == ORDER_OUT_FOR_DELIVERY:
            order.delivery_partner_name = delivery_partner_name
            order.delivery_partner_phone = delivery_partner_phone

        return order

    # ── prescription review ──────────────────────────────────────────────

    async def prescription_for_order(
        self, db: AsyncSession, owner: User, order_id: uuid.UUID
    ) -> dict:
        """
        Everything the counter needs before dispensing.

        Read-only throughout: the prescriber snapshot, the medication lines, the
        AI safety review with its findings, and the patient's recorded
        allergies. No field returned here is writable through this portal.
        """
        order = await self._owned_order(db, owner, order_id)

        result = await db.execute(
            select(Prescription)
            .where(Prescription.id == order.prescription_id)
            .options(selectinload(Prescription.medications))
        )
        prescription = result.scalar_one_or_none()
        if not prescription:
            raise EntityNotFoundException("Prescription", str(order.prescription_id))

        verification = (
            await db.execute(
                select(PrescriptionVerification)
                .where(PrescriptionVerification.prescription_id == prescription.id)
                .options(selectinload(PrescriptionVerification.findings))
                .order_by(PrescriptionVerification.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        patient = await db.get(Patient, prescription.patient_id)

        # Expiry is checked against what this store would actually hand over,
        # which the prescription itself cannot tell you.
        expiry_alerts: list[dict] = []
        for item in order.items:
            if not item.inventory_id:
                continue
            stock = await db.get(PharmacyInventory, item.inventory_id)
            if stock and stock.stock_state in ("expired", "near_expiry"):
                expiry_alerts.append(
                    {
                        "medicine_name": stock.medicine_name,
                        "batch_number": stock.batch_number,
                        "expiry_date": (
                            stock.expiry_date.isoformat() if stock.expiry_date else None
                        ),
                        "state": stock.stock_state,
                    }
                )

        return {
            "order_id": str(order.id),
            "order_number": order.order_number,
            "prescription_id": str(prescription.id),
            "diagnosis": prescription.diagnosis,
            "notes": prescription.notes or "",
            "issued_at": (
                prescription.consultation_date.isoformat()
                if prescription.consultation_date
                else prescription.created_at.isoformat()
            ),
            "signed_at": (
                prescription.signed_at.isoformat() if prescription.signed_at else None
            ),
            "pdf_url": prescription.pdf_url,
            "prescription_image_url": prescription.prescription_image_url,
            "prescriber": {
                "doctor_name": prescription.doctor_name,
                "specialty": prescription.doctor_specialty,
                "qualification": prescription.doctor_qualification,
                "hospital": prescription.doctor_hospital,
                "registration_number": prescription.doctor_registration_number,
                "experience_years": prescription.doctor_experience_years,
                "avatar_url": prescription.doctor_avatar_url,
            },
            "patient_name": prescription.patient_name,
            "patient_allergies": list(getattr(patient, "allergies", None) or []),
            "medications": [
                {
                    "name": m.name,
                    "generic_name": m.generic_name,
                    "brand_name": m.brand_name,
                    "strength": m.strength,
                    "dosage": m.dosage,
                    "frequency": m.frequency,
                    "duration": m.duration,
                    "food_instruction": m.food_instruction,
                    "route": m.route,
                    "quantity": m.quantity,
                    "special_instructions": m.special_instructions,
                }
                for m in prescription.medications
            ],
            "verification": (
                {
                    "verdict": verification.verdict,
                    "status": verification.status,
                    "confidence": verification.confidence,
                    "summary": verification.summary,
                    "unchecked_medications": verification.unchecked_medications,
                    "findings": [
                        {
                            "category": f.category,
                            "severity": f.severity,
                            "title": f.title,
                            "detail": f.detail,
                            "recommendation": f.recommendation,
                            "medications_involved": f.medications_involved,
                            "source": f.source,
                            "evidence": f.evidence,
                        }
                        for f in verification.findings
                    ],
                }
                if verification
                else None
            ),
            "expiry_alerts": expiry_alerts,
        }

    async def record_prescription_review(
        self,
        db: AsyncSession,
        owner: User,
        order_id: uuid.UUID,
        *,
        outcome: str,
        note: str,
    ) -> MedicineOrder:
        """
        Record the counter's dispensing decision.

        Written onto the order's event trail rather than onto the prescription:
        a pharmacy's refusal is a fact about this dispensing attempt, not an
        amendment to what the doctor prescribed. The prescription is never
        touched.
        """
        if outcome not in RX_REVIEW_OUTCOMES:
            raise BusinessRuleValidationException(
                f"'{outcome}' is not a valid review outcome."
            )

        order = await self._owned_order(db, owner, order_id)

        if outcome == "approved":
            if order.status == ORDER_RECEIVED:
                await ordering_service.advance_status(
                    db, order, ORDER_PREPARING,
                    note=note or "Prescription verified at the counter.",
                    actor_type="pharmacy", actor_id=owner.id,
                )
            return order

        if outcome == "rejected":
            if order.status not in (ORDER_RECEIVED, ORDER_PREPARING, ORDER_PACKED):
                raise BusinessRuleValidationException(
                    "This order has already been dispatched and cannot be refused."
                )
            await ordering_service.cancel_order(
                db, order,
                reason=note or "Prescription refused by pharmacist",
                actor_id=owner.id,
            )
            return order

        # clarification_requested — recorded, order left where it is.
        from app.models.medicine_order import OrderStatusEvent

        db.add(
            OrderStatusEvent(
                order_id=order.id,
                status=order.status,
                note=f"Clarification requested: {note}"[:500],
                actor_type="pharmacy",
                actor_id=owner.id,
            )
        )
        await db.flush()
        return order

    # ── inventory (delegated) ────────────────────────────────────────────

    async def list_inventory(self, db: AsyncSession, owner: User, **filters):
        store = await self._store(db, owner)
        return await pharmacy_admin_service.list_inventory(db, store.id, **filters)

    async def upsert_inventory(
        self,
        db: AsyncSession,
        owner: User,
        *,
        payload: dict,
        item_id: uuid.UUID | None = None,
        ip: str = "",
    ):
        store = await self._store(db, owner)
        if item_id:
            existing = await db.get(PharmacyInventory, item_id)
            if not existing or existing.pharmacy_id != store.id:
                raise AuthorizationException("That item belongs to another pharmacy.")
        return await pharmacy_admin_service.upsert_inventory(
            db, store.id, payload=payload, actor=owner, ip=ip, item_id=item_id
        )

    async def delete_inventory(
        self, db: AsyncSession, owner: User, item_id: uuid.UUID, *, ip: str = ""
    ) -> None:
        store = await self._store(db, owner)
        item = await db.get(PharmacyInventory, item_id)
        if not item or item.pharmacy_id != store.id:
            raise AuthorizationException("That item belongs to another pharmacy.")
        await pharmacy_admin_service.delete_inventory(db, item_id, actor=owner, ip=ip)

    async def import_inventory(
        self, db: AsyncSession, owner: User, *, content: str, ip: str = ""
    ) -> dict:
        store = await self._store(db, owner)
        return await pharmacy_admin_service.import_inventory_csv(
            db, store.id, content=content, actor=owner, ip=ip
        )

    async def export_inventory(self, db: AsyncSession, owner: User) -> tuple[str, str]:
        store = await self._store(db, owner)
        items, _ = await pharmacy_admin_service.list_inventory(db, store.id, limit=5000)
        return pharmacy_admin_service.export_inventory_csv(items), store.name

    async def find_by_code(
        self, db: AsyncSession, owner: User, code: str
    ) -> PharmacyInventory | None:
        """
        Barcode / QR lookup.

        Matched against barcode first and SKU second — a scanner emits the
        barcode, while a human typing a code at the counter usually means the
        SKU.
        """
        store = await self._store(db, owner)
        cleaned = (code or "").strip()
        if not cleaned:
            return None

        result = await db.execute(
            select(PharmacyInventory).where(
                PharmacyInventory.pharmacy_id == store.id,
                PharmacyInventory.deleted_at.is_(None),
                or_(
                    PharmacyInventory.barcode == cleaned,
                    PharmacyInventory.sku == cleaned,
                ),
            ).limit(1)
        )
        return result.scalar_one_or_none()

    # ── analytics ────────────────────────────────────────────────────────

    async def analytics(self, db: AsyncSession, owner: User, *, days: int = 30) -> dict:
        """Store-scoped performance. Aggregated in SQL, never in Python."""
        store = await self._store(db, owner)
        since = datetime.now(timezone.utc) - timedelta(days=days)
        mine = MedicineOrder.pharmacy_id == store.id
        recent = MedicineOrder.created_at >= since

        totals = (
            await db.execute(
                select(
                    func.count(MedicineOrder.id),
                    func.coalesce(func.sum(MedicineOrder.total), 0.0),
                    func.coalesce(func.avg(MedicineOrder.total), 0.0),
                ).where(mine, recent, MedicineOrder.status != ORDER_CANCELLED)
            )
        ).one()

        movers = (
            await db.execute(
                select(
                    MedicineOrderItem.medicine_name,
                    func.sum(MedicineOrderItem.quantity),
                    func.sum(MedicineOrderItem.line_total),
                )
                .join(MedicineOrder, MedicineOrder.id == MedicineOrderItem.order_id)
                .where(mine, recent)
                .group_by(MedicineOrderItem.medicine_name)
                .order_by(func.sum(MedicineOrderItem.quantity).desc())
            )
        ).all()

        # Peak hours drive staffing, so they are bucketed from the actual
        # placement timestamps rather than assumed.
        peak: dict[int, int] = {}
        placed_rows = (
            await db.execute(
                select(MedicineOrder.placed_at).where(
                    mine, recent, MedicineOrder.placed_at.isnot(None)
                )
            )
        ).scalars().all()
        for placed in placed_rows:
            peak[placed.hour] = peak.get(placed.hour, 0) + 1

        stock_rows = (
            await db.execute(
                select(PharmacyInventory).where(
                    PharmacyInventory.pharmacy_id == store.id,
                    PharmacyInventory.deleted_at.is_(None),
                )
            )
        ).scalars().all()

        inventory_value = round(sum(r.inventory_value for r in stock_rows), 2)
        expiry_loss = round(
            sum(r.inventory_value for r in stock_rows if r.stock_state == "expired"), 2
        )

        top_customers = [
            {"patient_id": str(row[0]), "orders": row[1], "spend": round(float(row[2] or 0), 2)}
            for row in (
                await db.execute(
                    select(
                        MedicineOrder.patient_id,
                        func.count(MedicineOrder.id),
                        func.sum(MedicineOrder.total),
                    )
                    .where(mine, recent, MedicineOrder.status != ORDER_CANCELLED)
                    .group_by(MedicineOrder.patient_id)
                    .order_by(func.sum(MedicineOrder.total).desc())
                    .limit(10)
                )
            ).all()
        ]

        return {
            "window_days": days,
            "orders": int(totals[0] or 0),
            "revenue": round(float(totals[1] or 0), 2),
            "average_basket": round(float(totals[2] or 0), 2),
            "fastest_moving": [
                {"medicine": r[0], "units": int(r[1] or 0), "revenue": round(float(r[2] or 0), 2)}
                for r in movers[:10]
            ],
            # Slowest movers are the tail of the same ranking, reversed — the
            # medicines tying up shelf space.
            "slowest_moving": [
                {"medicine": r[0], "units": int(r[1] or 0), "revenue": round(float(r[2] or 0), 2)}
                for r in list(reversed(movers))[:10]
            ],
            "peak_hours": [
                {"hour": hour, "orders": count} for hour, count in sorted(peak.items())
            ],
            "top_customers": top_customers,
            "inventory_value": inventory_value,
            "expiry_loss": expiry_loss,
            "catalogue_size": len(stock_rows),
        }

    async def customers(
        self, db: AsyncSession, owner: User, *, limit: int = 25
    ) -> list[dict]:
        """Who buys here, ranked by spend."""
        store = await self._store(db, owner)
        rows = (
            await db.execute(
                select(
                    MedicineOrder.patient_id,
                    func.count(MedicineOrder.id),
                    func.coalesce(func.sum(MedicineOrder.total), 0.0),
                    func.max(MedicineOrder.created_at),
                )
                .where(
                    MedicineOrder.pharmacy_id == store.id,
                    MedicineOrder.status != ORDER_CANCELLED,
                )
                .group_by(MedicineOrder.patient_id)
                .order_by(func.sum(MedicineOrder.total).desc())
                .limit(limit)
            )
        ).all()

        customers: list[dict] = []
        for patient_id, order_count, spend, last_order in rows:
            patient = await db.get(Patient, patient_id)
            customers.append(
                {
                    "patient_id": str(patient_id),
                    "name": (
                        f"{patient.first_name} {patient.last_name}" if patient else "Unknown"
                    ),
                    "orders": int(order_count or 0),
                    "total_spend": round(float(spend or 0), 2),
                    "average_spend": round(float(spend or 0) / max(1, order_count), 2),
                    "last_order_at": last_order.isoformat() if last_order else None,
                }
            )
        return customers

    # ── notifications feed ───────────────────────────────────────────────

    async def alerts(self, db: AsyncSession, owner: User) -> list[dict]:
        """
        Operational alerts, derived rather than stored.

        Computed from live state so an alert cannot go stale: a resolved
        shortage stops appearing the moment stock is updated, with nothing to
        mark as read.
        """
        store = await self._store(db, owner)
        alerts: list[dict] = []

        waiting = (
            await db.execute(
                select(MedicineOrder)
                .where(
                    MedicineOrder.pharmacy_id == store.id,
                    MedicineOrder.status == ORDER_RECEIVED,
                )
                .order_by(MedicineOrder.created_at)
                .limit(20)
            )
        ).scalars().all()
        for order in waiting:
            alerts.append(
                {
                    "type": "new_order",
                    "severity": "info",
                    "title": f"New order {order.order_number}",
                    "detail": f"{len(order.items) if order.items else 0} item(s) awaiting acceptance.",
                    "reference_id": str(order.id),
                    "created_at": order.created_at.isoformat(),
                }
            )

        stock_rows = (
            await db.execute(
                select(PharmacyInventory).where(
                    PharmacyInventory.pharmacy_id == store.id,
                    PharmacyInventory.deleted_at.is_(None),
                )
            )
        ).scalars().all()

        for row in stock_rows:
            state = row.stock_state
            if state in ("out_of_stock", "critical", "low"):
                alerts.append(
                    {
                        "type": "low_stock",
                        "severity": "warning" if state == "low" else "critical",
                        "title": f"{row.medicine_name} is {state.replace('_', ' ')}",
                        "detail": f"{row.stock_quantity} unit(s) remaining.",
                        "reference_id": str(row.id),
                        "created_at": (
                            row.stock_synced_at or row.updated_at
                        ).isoformat(),
                    }
                )
            elif state in ("expired", "near_expiry"):
                alerts.append(
                    {
                        "type": "expiry",
                        "severity": "critical" if state == "expired" else "warning",
                        "title": f"{row.medicine_name} {state.replace('_', ' ')}",
                        "detail": (
                            f"Batch {row.batch_number or '—'} expires "
                            f"{row.expiry_date.date().isoformat() if row.expiry_date else 'unknown'}."
                        ),
                        "reference_id": str(row.id),
                        "created_at": (row.expiry_date or row.updated_at).isoformat(),
                    }
                )

        severity_rank = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda a: (severity_rank.get(a["severity"], 3), a["created_at"]))
        return alerts

    # ── reports ──────────────────────────────────────────────────────────

    async def sales_report(
        self, db: AsyncSession, owner: User, *, days: int = 30
    ) -> list[dict]:
        """
        Per-order sales rows for CSV/PDF export, including GST.

        Tax is computed from the inventory row's `gst_percent` at line level
        rather than a flat store rate, because different schedules carry
        different rates.
        """
        store = await self._store(db, owner)
        since = datetime.now(timezone.utc) - timedelta(days=days)

        orders = (
            await db.execute(
                select(MedicineOrder)
                .where(
                    MedicineOrder.pharmacy_id == store.id,
                    MedicineOrder.created_at >= since,
                )
                .options(selectinload(MedicineOrder.items))
                .order_by(MedicineOrder.created_at.desc())
            )
        ).scalars().all()

        rows: list[dict] = []
        for order in orders:
            gst_total = 0.0
            for item in order.items:
                stock = (
                    await db.get(PharmacyInventory, item.inventory_id)
                    if item.inventory_id
                    else None
                )
                rate = (stock.gst_percent if stock else 0.0) or 0.0
                gst_total += item.line_total * rate / 100.0

            rows.append(
                {
                    "order_number": order.order_number,
                    "date": order.created_at.date().isoformat(),
                    "status": order.status,
                    "items": len(order.items),
                    "subtotal": round(order.subtotal, 2),
                    "discount": round(order.discount_total, 2),
                    "delivery_fee": round(order.delivery_fee, 2),
                    "gst": round(gst_total, 2),
                    "total": round(order.total, 2),
                }
            )
        return rows

    @staticmethod
    def report_to_csv(rows: Sequence[dict], columns: Sequence[str]) -> str:
        import csv
        import io

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()


pharmacy_portal_service = PharmacyPortalService()
