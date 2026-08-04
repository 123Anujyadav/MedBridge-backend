"""
Pharmacy administration.

Everything an administrator does to the partner network: onboarding, the
verification workflow, document review, catalogue and stock management,
pricing, delivery configuration, and the analytics over all of it.

Two rules run through the module:

1. **Every mutation is audited.** `AuditLog` already carries actor, resource,
   field, previous value, new value and IP; this service writes to it rather
   than inventing a parallel trail.

2. **Nothing here touches the dispensing path.** Patient-facing search and
   ordering read `Pharmacy.can_fulfil` and `PharmacyInventory.availability`,
   whose semantics are unchanged. Admin state — verification, expiry, reorder
   levels — is additive and read only by this module.
"""

from __future__ import annotations

import csv
import io
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from sqlalchemy import Integer, Select, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import (
    BusinessRuleValidationException,
    EntityNotFoundException,
)
from app.models.audit import AuditLog
from app.models.medicine_order import MedicineOrder
from app.models.pharmacy import (
    DOCUMENT_TYPES,
    Pharmacy,
    PharmacyDocument,
    PharmacyInventory,
    PharmacyVerificationEvent,
    VERIFICATION_APPROVED,
    VERIFICATION_PENDING,
    VERIFICATION_REJECTED,
    VERIFICATION_STATUSES,
    VERIFICATION_SUSPENDED,
    VERIFICATION_TRANSITIONS,
)
from app.models.user import User

logger = logging.getLogger(__name__)

# Columns an administrator may set directly. Anything outside this set is
# ignored rather than blindly applied, so a stray key in a request body cannot
# reach a column the workflow is supposed to own — `verification_status`,
# `verified_at` and `is_partner` are all changed through the workflow only.
EDITABLE_FIELDS = frozenset({
    "name", "address", "city", "postal_code", "phone", "latitude", "longitude",
    "licence_number", "is_active", "is_24x7", "opens_at", "closes_at",
    "delivers", "delivery_radius_km", "delivery_fee", "free_delivery_above",
    "min_order_value", "avg_prep_minutes", "owner_name", "business_name",
    "gst_number", "drug_license_number", "drug_license_expiry", "email",
    "whatsapp", "emergency_phone", "logo_url", "banner_url", "store_images",
    "express_delivery", "express_delivery_radius_km", "pickup_available",
    "holiday_dates", "upi_id", "bank_account_name", "bank_account_number",
    "bank_ifsc", "platform_commission_percent", "rating", "total_ratings",
})

INVENTORY_EDITABLE_FIELDS = frozenset({
    "sku", "rxcui", "medicine_name", "generic_name", "brand_name", "manufacturer",
    "strength", "form", "pack_size", "is_generic", "requires_prescription",
    "mrp", "selling_price", "discount_percent", "stock_quantity",
    "low_stock_threshold", "restock_expected_at", "composition", "drug_schedule",
    "category", "barcode", "storage_instructions", "batch_number",
    "manufacturing_date", "expiry_date", "min_stock", "max_stock",
    "reorder_level", "gst_percent",
})

CSV_COLUMNS = (
    "sku", "medicine_name", "generic_name", "brand_name", "manufacturer",
    "strength", "form", "category", "batch_number", "expiry_date",
    "mrp", "selling_price", "discount_percent", "gst_percent",
    "stock_quantity", "low_stock_threshold", "reorder_level", "barcode",
)

MAX_IMPORT_ROWS = 5_000
"""A CSV import runs in one transaction; an unbounded file would hold locks on
the inventory table for as long as it takes to parse."""


def _mask(value: Any) -> str:
    """
    Render a value for the audit trail without leaking a full secret.

    Bank and UPI details are audited — an administrator changing where a
    pharmacy's money goes is exactly what an audit log is for — but the log is
    itself readable by other admins, so only the tail is retained.
    """
    text = "" if value is None else str(value)
    return text if len(text) <= 4 else f"••••{text[-4:]}"


SENSITIVE_FIELDS = frozenset({"bank_account_number", "upi_id", "bank_ifsc"})


class PharmacyAdminService:
    # ── audit ────────────────────────────────────────────────────────────

    def _audit(
        self,
        db: AsyncSession,
        *,
        actor: User,
        action: str,
        resource_id: str,
        details: str = "",
        field: str | None = None,
        previous: Any = None,
        new: Any = None,
        ip: str = "",
        resource: str = "Pharmacy",
    ) -> None:
        if field in SENSITIVE_FIELDS:
            previous, new = _mask(previous), _mask(new)

        db.add(
            AuditLog(
                user_id=actor.id,
                user_name=getattr(actor, "email", "") or "",
                user_role=getattr(actor, "role", "admin") or "admin",
                action=action,
                resource=resource,
                resource_id=resource_id,
                ip_address=ip or "unknown",
                status="success",
                details=details,
                actor_type="admin",
                event_type=action.lower(),
                field_changed=field,
                previous_value=None if previous is None else str(previous)[:2000],
                new_value=None if new is None else str(new)[:2000],
            )
        )

    # ── pharmacy CRUD ────────────────────────────────────────────────────

    async def get(self, db: AsyncSession, pharmacy_id: uuid.UUID) -> Pharmacy:
        result = await db.execute(
            select(Pharmacy)
            .where(Pharmacy.id == pharmacy_id, Pharmacy.deleted_at.is_(None))
            .options(
                selectinload(Pharmacy.documents),
                selectinload(Pharmacy.verification_events),
            )
            # `populate_existing` forces the eager collections to refresh rather
            # than returning whatever the identity map already held. Without it,
            # reading a pharmacy back inside the same transaction that just
            # added a document or a verification event returns the stale
            # collection — so the detail screen would omit the very change the
            # administrator had just made.
            .execution_options(populate_existing=True)
        )
        pharmacy = result.scalar_one_or_none()
        if not pharmacy:
            raise EntityNotFoundException("Pharmacy", str(pharmacy_id))
        return pharmacy

    def _list_query(
        self,
        *,
        search: str | None = None,
        verification_status: str | None = None,
        is_active: bool | None = None,
        is_partner: bool | None = None,
        city: str | None = None,
    ) -> Select:
        query = select(Pharmacy).where(Pharmacy.deleted_at.is_(None))

        if search:
            term = f"%{search.strip().lower()}%"
            query = query.where(
                or_(
                    func.lower(Pharmacy.name).like(term),
                    func.lower(Pharmacy.business_name).like(term),
                    func.lower(Pharmacy.owner_name).like(term),
                    func.lower(Pharmacy.city).like(term),
                    func.lower(Pharmacy.gst_number).like(term),
                    func.lower(Pharmacy.drug_license_number).like(term),
                    func.lower(Pharmacy.phone).like(term),
                )
            )
        if verification_status:
            query = query.where(Pharmacy.verification_status == verification_status)
        if is_active is not None:
            query = query.where(Pharmacy.is_active.is_(is_active))
        if is_partner is not None:
            query = query.where(Pharmacy.is_partner.is_(is_partner))
        if city:
            query = query.where(func.lower(Pharmacy.city) == city.strip().lower())

        return query

    async def list_pharmacies(
        self, db: AsyncSession, *, skip: int = 0, limit: int = 50, **filters
    ) -> tuple[list[Pharmacy], int]:
        """Page of pharmacies plus the unpaginated total, for the table footer."""
        query = self._list_query(**filters)

        total = await db.scalar(
            select(func.count()).select_from(query.subquery())
        )
        rows = await db.execute(
            query.order_by(Pharmacy.created_at.desc()).offset(skip).limit(limit)
        )
        return list(rows.scalars().all()), int(total or 0)

    async def create(
        self, db: AsyncSession, *, payload: dict, actor: User, ip: str = ""
    ) -> Pharmacy:
        """
        Onboard a pharmacy.

        Created as a non-partner in `pending`: a new record must clear
        verification before it can appear in patient search, and making that
        the default means no administrator has to remember to withhold it.
        """
        fields = {k: v for k, v in payload.items() if k in EDITABLE_FIELDS}

        existing_gst = fields.get("gst_number")
        if existing_gst:
            clash = await db.scalar(
                select(func.count())
                .select_from(Pharmacy)
                .where(
                    Pharmacy.gst_number == existing_gst,
                    Pharmacy.deleted_at.is_(None),
                )
            )
            if clash:
                raise BusinessRuleValidationException(
                    f"A pharmacy with GST number {existing_gst} already exists."
                )

        pharmacy = Pharmacy(
            **fields,
            is_partner=False,
            verification_status=VERIFICATION_PENDING,
        )
        db.add(pharmacy)
        await db.flush()

        db.add(
            PharmacyVerificationEvent(
                pharmacy_id=pharmacy.id,
                from_status=None,
                to_status=VERIFICATION_PENDING,
                note="Pharmacy record created.",
                actor_id=actor.id,
                actor_name=getattr(actor, "email", "") or "",
            )
        )
        self._audit(
            db, actor=actor, action="PHARMACY_CREATED", resource_id=str(pharmacy.id),
            details=f"Onboarded {pharmacy.name}.", ip=ip,
        )
        logger.info("[PHARMACY_ADMIN_CREATED] id=%s by=%s", pharmacy.id, actor.id)
        return pharmacy

    async def update(
        self,
        db: AsyncSession,
        pharmacy_id: uuid.UUID,
        *,
        payload: dict,
        actor: User,
        ip: str = "",
    ) -> Pharmacy:
        """Apply an edit, auditing each changed field individually."""
        pharmacy = await self.get(db, pharmacy_id)

        for field, value in payload.items():
            if field not in EDITABLE_FIELDS:
                continue
            previous = getattr(pharmacy, field, None)
            if previous == value:
                continue
            setattr(pharmacy, field, value)
            # One audit row per field rather than a single diff blob: "who
            # changed the commission rate" has to be answerable without parsing
            # a JSON payload out of a details column.
            self._audit(
                db, actor=actor, action="PHARMACY_UPDATED", resource_id=str(pharmacy.id),
                details=f"Updated {field}.", field=field, previous=previous,
                new=value, ip=ip,
            )

        await db.flush()
        return pharmacy

    async def set_active(
        self,
        db: AsyncSession,
        pharmacy_id: uuid.UUID,
        *,
        active: bool,
        reason: str,
        actor: User,
        ip: str = "",
    ) -> Pharmacy:
        """Suspend or reactivate. Suspension removes it from patient search."""
        pharmacy = await self.get(db, pharmacy_id)
        previous = pharmacy.is_active
        pharmacy.is_active = active

        if not active:
            pharmacy.suspended_at = datetime.now(timezone.utc)
            pharmacy.suspension_reason = reason[:500]
        else:
            pharmacy.suspended_at = None
            pharmacy.suspension_reason = None

        self._audit(
            db, actor=actor,
            action="PHARMACY_ACTIVATED" if active else "PHARMACY_SUSPENDED",
            resource_id=str(pharmacy.id), details=reason[:500],
            field="is_active", previous=previous, new=active, ip=ip,
        )
        await db.flush()
        return pharmacy

    async def soft_delete(
        self, db: AsyncSession, pharmacy_id: uuid.UUID, *, actor: User, ip: str = ""
    ) -> None:
        """
        Retire a pharmacy.

        Soft delete, always. Orders reference `pharmacies` with RESTRICT and a
        dispensing record must outlive the shop that filled it, so the row is
        marked rather than removed.
        """
        pharmacy = await self.get(db, pharmacy_id)

        open_orders = await db.scalar(
            select(func.count())
            .select_from(MedicineOrder)
            .where(
                MedicineOrder.pharmacy_id == pharmacy_id,
                MedicineOrder.status.notin_(["delivered", "cancelled"]),
            )
        )
        if open_orders:
            raise BusinessRuleValidationException(
                f"{pharmacy.name} has {open_orders} order(s) still in progress and "
                "cannot be removed until they are delivered or cancelled."
            )

        pharmacy.soft_delete()
        pharmacy.is_active = False
        pharmacy.is_partner = False

        self._audit(
            db, actor=actor, action="PHARMACY_DELETED", resource_id=str(pharmacy.id),
            details=f"Retired {pharmacy.name}.", ip=ip,
        )
        await db.flush()

    async def bulk_set_status(
        self,
        db: AsyncSession,
        pharmacy_ids: Sequence[uuid.UUID],
        *,
        active: bool,
        reason: str,
        actor: User,
        ip: str = "",
    ) -> int:
        """Suspend or reactivate many at once. Returns how many changed."""
        changed = 0
        for pharmacy_id in pharmacy_ids:
            try:
                await self.set_active(
                    db, pharmacy_id, active=active, reason=reason, actor=actor, ip=ip
                )
                changed += 1
            except EntityNotFoundException:
                # One bad id in a bulk selection should not abandon the rest.
                logger.warning("[PHARMACY_BULK_SKIP] unknown id=%s", pharmacy_id)
        return changed

    # ── verification workflow ────────────────────────────────────────────

    async def transition_verification(
        self,
        db: AsyncSession,
        pharmacy_id: uuid.UUID,
        *,
        to_status: str,
        note: str,
        actor: User,
        ip: str = "",
    ) -> Pharmacy:
        """
        Advance the verification workflow, refusing illegal transitions.

        Approval is also what grants partner status, and rejection or suspension
        is what withdraws it — so the dispensing gate and the review state can
        never disagree.
        """
        if to_status not in VERIFICATION_STATUSES:
            raise BusinessRuleValidationException(f"Unknown status '{to_status}'.")

        pharmacy = await self.get(db, pharmacy_id)
        current = pharmacy.verification_status

        allowed = VERIFICATION_TRANSITIONS.get(current, ())
        if to_status not in allowed:
            raise BusinessRuleValidationException(
                f"A pharmacy that is '{current}' cannot become '{to_status}'. "
                f"Allowed: {', '.join(allowed) if allowed else 'none'}."
            )

        pharmacy.verification_status = to_status
        pharmacy.verification_notes = note[:1000] or pharmacy.verification_notes

        if to_status == VERIFICATION_APPROVED:
            pharmacy.is_partner = True
            pharmacy.is_active = True
            pharmacy.verified_at = datetime.now(timezone.utc)
            pharmacy.verified_by = actor.id
            pharmacy.rejection_reason = None
            pharmacy.suspended_at = None
            pharmacy.suspension_reason = None
        elif to_status == VERIFICATION_SUSPENDED:
            pharmacy.is_partner = False
            pharmacy.suspended_at = datetime.now(timezone.utc)
            pharmacy.suspension_reason = note[:500]
        elif to_status == VERIFICATION_REJECTED:
            pharmacy.is_partner = False
            pharmacy.rejection_reason = note[:500]

        db.add(
            PharmacyVerificationEvent(
                pharmacy_id=pharmacy.id,
                from_status=current,
                to_status=to_status,
                note=note[:1000],
                actor_id=actor.id,
                actor_name=getattr(actor, "email", "") or "",
            )
        )
        self._audit(
            db, actor=actor, action="PHARMACY_VERIFICATION", resource_id=str(pharmacy.id),
            details=note[:500], field="verification_status", previous=current,
            new=to_status, ip=ip,
        )
        await db.flush()
        logger.info(
            "[PHARMACY_VERIFICATION] id=%s %s -> %s by=%s",
            pharmacy.id, current, to_status, actor.id,
        )
        return pharmacy

    # ── documents ────────────────────────────────────────────────────────

    async def add_document(
        self,
        db: AsyncSession,
        pharmacy_id: uuid.UUID,
        *,
        payload: dict,
        actor: User,
        ip: str = "",
    ) -> PharmacyDocument:
        await self.get(db, pharmacy_id)

        doc_type = payload.get("doc_type")
        if doc_type not in DOCUMENT_TYPES:
            raise BusinessRuleValidationException(
                f"'{doc_type}' is not a recognised document type."
            )

        document = PharmacyDocument(
            pharmacy_id=pharmacy_id,
            doc_type=doc_type,
            file_url=payload["file_url"],
            file_name=payload.get("file_name", "") or "",
            document_number=payload.get("document_number"),
            issued_at=payload.get("issued_at"),
            expires_at=payload.get("expires_at"),
            status="uploaded",
        )
        db.add(document)
        await db.flush()

        self._audit(
            db, actor=actor, action="PHARMACY_DOCUMENT_ADDED",
            resource="PharmacyDocument", resource_id=str(document.id),
            details=f"{doc_type} uploaded for pharmacy {pharmacy_id}.", ip=ip,
        )
        return document

    async def review_document(
        self,
        db: AsyncSession,
        document_id: uuid.UUID,
        *,
        status: str,
        notes: str,
        actor: User,
        ip: str = "",
    ) -> PharmacyDocument:
        document = await db.get(PharmacyDocument, document_id)
        if not document:
            raise EntityNotFoundException("PharmacyDocument", str(document_id))

        previous = document.status
        document.status = status
        document.review_notes = notes[:500]
        document.reviewed_by = actor.id
        document.reviewed_at = datetime.now(timezone.utc)

        self._audit(
            db, actor=actor, action="PHARMACY_DOCUMENT_REVIEWED",
            resource="PharmacyDocument", resource_id=str(document.id),
            details=notes[:500], field="status", previous=previous, new=status, ip=ip,
        )
        await db.flush()
        return document

    async def expiring_documents(
        self, db: AsyncSession, *, within_days: int = 30
    ) -> list[PharmacyDocument]:
        """
        Documents already expired or expiring soon.

        This is the alert feed. Expired ones are included rather than filtered
        out — a licence that lapsed last week is more urgent than one lapsing
        next month, and an alert list that drops it is worse than useless.
        """
        horizon = datetime.now(timezone.utc) + timedelta(days=within_days)
        result = await db.execute(
            select(PharmacyDocument)
            .where(
                PharmacyDocument.deleted_at.is_(None),
                PharmacyDocument.expires_at.isnot(None),
                PharmacyDocument.expires_at <= horizon,
                PharmacyDocument.status != "rejected",
            )
            .order_by(PharmacyDocument.expires_at)
        )
        return list(result.scalars().all())

    # ── inventory ────────────────────────────────────────────────────────

    async def list_inventory(
        self,
        db: AsyncSession,
        pharmacy_id: uuid.UUID | None = None,
        *,
        search: str | None = None,
        category: str | None = None,
        manufacturer: str | None = None,
        stock_state: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[list[PharmacyInventory], int]:
        query = select(PharmacyInventory).where(PharmacyInventory.deleted_at.is_(None))

        if pharmacy_id:
            query = query.where(PharmacyInventory.pharmacy_id == pharmacy_id)
        if search:
            term = f"%{search.strip().lower()}%"
            query = query.where(
                or_(
                    func.lower(PharmacyInventory.medicine_name).like(term),
                    func.lower(PharmacyInventory.generic_name).like(term),
                    func.lower(PharmacyInventory.brand_name).like(term),
                    func.lower(PharmacyInventory.manufacturer).like(term),
                    func.lower(PharmacyInventory.barcode).like(term),
                    func.lower(PharmacyInventory.sku).like(term),
                )
            )
        if category:
            query = query.where(func.lower(PharmacyInventory.category) == category.lower())
        if manufacturer:
            query = query.where(
                func.lower(PharmacyInventory.manufacturer) == manufacturer.lower()
            )
        if min_price is not None:
            query = query.where(PharmacyInventory.selling_price >= min_price)
        if max_price is not None:
            query = query.where(PharmacyInventory.selling_price <= max_price)

        # Expiry and out-of-stock are expressible in SQL and filtered there so
        # paging stays correct. The finer bands (critical vs low) depend on
        # per-row thresholds and are applied after fetching.
        now = datetime.now(timezone.utc)
        if stock_state == "expired":
            query = query.where(
                PharmacyInventory.expiry_date.isnot(None),
                PharmacyInventory.expiry_date <= now,
            )
        elif stock_state == "near_expiry":
            query = query.where(
                PharmacyInventory.expiry_date.isnot(None),
                PharmacyInventory.expiry_date > now,
                PharmacyInventory.expiry_date
                <= now + timedelta(days=PharmacyInventory.NEAR_EXPIRY_DAYS),
            )
        elif stock_state == "out_of_stock":
            query = query.where(PharmacyInventory.stock_quantity <= 0)

        total = await db.scalar(select(func.count()).select_from(query.subquery()))
        rows = await db.execute(
            query.order_by(PharmacyInventory.medicine_name).offset(skip).limit(limit)
        )
        items = list(rows.scalars().all())

        if stock_state in ("critical", "low", "available"):
            items = [i for i in items if i.stock_state == stock_state]

        return items, int(total or 0)

    async def upsert_inventory(
        self,
        db: AsyncSession,
        pharmacy_id: uuid.UUID,
        *,
        payload: dict,
        actor: User,
        ip: str = "",
        item_id: uuid.UUID | None = None,
    ) -> PharmacyInventory:
        await self.get(db, pharmacy_id)
        fields = {k: v for k, v in payload.items() if k in INVENTORY_EDITABLE_FIELDS}

        if item_id:
            item = await db.get(PharmacyInventory, item_id)
            if not item or item.pharmacy_id != pharmacy_id:
                raise EntityNotFoundException("PharmacyInventory", str(item_id))

            for field, value in fields.items():
                previous = getattr(item, field, None)
                if previous == value:
                    continue
                setattr(item, field, value)
                # Price and stock are the fields that move money and dispensing,
                # so they are audited per change rather than in aggregate.
                if field in ("selling_price", "mrp", "stock_quantity", "discount_percent"):
                    self._audit(
                        db, actor=actor, action="INVENTORY_UPDATED",
                        resource="PharmacyInventory", resource_id=str(item.id),
                        details=f"{item.medicine_name}: {field}", field=field,
                        previous=previous, new=value, ip=ip,
                    )
            item.stock_synced_at = datetime.now(timezone.utc)
            await db.flush()
            return item

        item = PharmacyInventory(
            pharmacy_id=pharmacy_id,
            stock_synced_at=datetime.now(timezone.utc),
            **fields,
        )
        db.add(item)
        await db.flush()
        self._audit(
            db, actor=actor, action="INVENTORY_CREATED",
            resource="PharmacyInventory", resource_id=str(item.id),
            details=f"Added {item.medicine_name}.", ip=ip,
        )
        return item

    async def delete_inventory(
        self, db: AsyncSession, item_id: uuid.UUID, *, actor: User, ip: str = ""
    ) -> None:
        item = await db.get(PharmacyInventory, item_id)
        if not item:
            raise EntityNotFoundException("PharmacyInventory", str(item_id))
        item.soft_delete()
        self._audit(
            db, actor=actor, action="INVENTORY_DELETED",
            resource="PharmacyInventory", resource_id=str(item.id),
            details=f"Removed {item.medicine_name}.", ip=ip,
        )
        await db.flush()

    # ── CSV import / export ──────────────────────────────────────────────

    @staticmethod
    def export_inventory_csv(items: Iterable[PharmacyInventory]) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    column: (
                        getattr(item, column).isoformat()
                        if isinstance(getattr(item, column, None), datetime)
                        else getattr(item, column, "")
                    )
                    for column in CSV_COLUMNS
                }
            )
        return buffer.getvalue()

    async def import_inventory_csv(
        self,
        db: AsyncSession,
        pharmacy_id: uuid.UUID,
        *,
        content: str,
        actor: User,
        ip: str = "",
    ) -> dict:
        """
        Bulk-load stock from CSV, matched on `sku`.

        Rows are validated individually and failures are reported by line
        number rather than aborting the batch — a 500-row file with three bad
        prices should load 497 rows and tell the administrator which three to
        fix, not reject the lot.
        """
        await self.get(db, pharmacy_id)

        reader = csv.DictReader(io.StringIO(content))
        created = updated = 0
        errors: list[dict] = []

        existing = {
            row.sku: row
            for row in (
                await db.execute(
                    select(PharmacyInventory).where(
                        PharmacyInventory.pharmacy_id == pharmacy_id,
                        PharmacyInventory.deleted_at.is_(None),
                    )
                )
            ).scalars().all()
        }

        for line_number, raw in enumerate(reader, start=2):
            if line_number - 1 > MAX_IMPORT_ROWS:
                errors.append(
                    {"line": line_number, "error": f"File exceeds {MAX_IMPORT_ROWS} rows."}
                )
                break
            try:
                fields = self._parse_csv_row(raw)
            except ValueError as exc:
                errors.append({"line": line_number, "error": str(exc)})
                continue

            sku = fields.get("sku")
            if not sku:
                errors.append({"line": line_number, "error": "sku is required."})
                continue

            item = existing.get(sku)
            if item:
                for field, value in fields.items():
                    setattr(item, field, value)
                item.stock_synced_at = datetime.now(timezone.utc)
                updated += 1
            else:
                item = PharmacyInventory(
                    pharmacy_id=pharmacy_id,
                    stock_synced_at=datetime.now(timezone.utc),
                    **fields,
                )
                db.add(item)
                existing[sku] = item
                created += 1

        await db.flush()
        self._audit(
            db, actor=actor, action="INVENTORY_IMPORTED", resource_id=str(pharmacy_id),
            details=f"CSV import: {created} created, {updated} updated, {len(errors)} rejected.",
            ip=ip,
        )
        logger.info(
            "[PHARMACY_INVENTORY_IMPORT] pharmacy=%s created=%d updated=%d errors=%d",
            pharmacy_id, created, updated, len(errors),
        )
        return {"created": created, "updated": updated, "errors": errors}

    @staticmethod
    def _parse_csv_row(raw: dict) -> dict:
        """Coerce one CSV row, raising ValueError with a readable message."""
        fields: dict[str, Any] = {}

        for column in CSV_COLUMNS:
            value = (raw.get(column) or "").strip()
            if value == "":
                continue

            if column in ("mrp", "selling_price", "discount_percent", "gst_percent"):
                try:
                    fields[column] = float(value)
                except ValueError:
                    raise ValueError(f"{column} '{value}' is not a number.") from None
            elif column in ("stock_quantity", "low_stock_threshold", "reorder_level"):
                try:
                    fields[column] = int(float(value))
                except ValueError:
                    raise ValueError(f"{column} '{value}' is not a whole number.") from None
            elif column == "expiry_date":
                try:
                    fields[column] = datetime.fromisoformat(value[:10]).replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    raise ValueError(
                        f"expiry_date '{value}' is not an ISO date (YYYY-MM-DD)."
                    ) from None
            else:
                fields[column] = value

        if fields.get("stock_quantity", 0) < 0:
            raise ValueError("stock_quantity cannot be negative.")
        if fields.get("selling_price", 0) < 0 or fields.get("mrp", 0) < 0:
            raise ValueError("Prices cannot be negative.")

        return fields

    # ── analytics ────────────────────────────────────────────────────────

    async def analytics(self, db: AsyncSession, *, days: int = 30) -> dict:
        """
        Network-wide metrics for the admin dashboard.

        Computed in SQL rather than by loading orders into Python: this runs on
        every dashboard load and the order table only grows.
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)

        delivered = MedicineOrder.status == "delivered"
        recent = MedicineOrder.created_at >= since

        totals = (
            await db.execute(
                select(
                    func.count(MedicineOrder.id),
                    func.coalesce(func.sum(MedicineOrder.total), 0.0),
                ).where(recent)
            )
        ).one()

        delivered_row = (
            await db.execute(
                select(
                    func.count(MedicineOrder.id),
                    func.coalesce(func.sum(MedicineOrder.total), 0.0),
                    func.coalesce(func.avg(MedicineOrder.eta_minutes), 0.0),
                ).where(recent, delivered)
            )
        ).one()

        cancelled = await db.scalar(
            select(func.count()).select_from(MedicineOrder).where(
                recent, MedicineOrder.status == "cancelled"
            )
        )

        by_status = {
            row[0]: row[1]
            for row in (
                await db.execute(
                    select(MedicineOrder.status, func.count(MedicineOrder.id))
                    .where(recent)
                    .group_by(MedicineOrder.status)
                )
            ).all()
        }

        top_pharmacies = [
            {"pharmacy": row[0], "orders": row[1], "revenue": round(float(row[2] or 0), 2)}
            for row in (
                await db.execute(
                    select(
                        MedicineOrder.pharmacy_name,
                        func.count(MedicineOrder.id),
                        func.sum(MedicineOrder.total),
                    )
                    .where(recent)
                    .group_by(MedicineOrder.pharmacy_name)
                    .order_by(func.sum(MedicineOrder.total).desc())
                    .limit(10)
                )
            ).all()
        ]

        top_cities = [
            {"city": row[0] or "Unknown", "pharmacies": row[1]}
            for row in (
                await db.execute(
                    select(Pharmacy.city, func.count(Pharmacy.id))
                    .where(Pharmacy.deleted_at.is_(None), Pharmacy.is_partner.is_(True))
                    .group_by(Pharmacy.city)
                    .order_by(func.count(Pharmacy.id).desc())
                    .limit(10)
                )
            ).all()
        ]

        inventory_value = await db.scalar(
            select(
                func.coalesce(
                    func.sum(
                        PharmacyInventory.selling_price * PharmacyInventory.stock_quantity
                    ),
                    0.0,
                )
            ).where(PharmacyInventory.deleted_at.is_(None))
        )

        network = (
            await db.execute(
                select(
                    func.count(Pharmacy.id),
                    func.sum(func.cast(Pharmacy.is_partner, Integer)),
                ).where(Pharmacy.deleted_at.is_(None))
            )
        ).one()

        order_count = int(totals[0] or 0)
        delivered_count = int(delivered_row[0] or 0)

        return {
            "window_days": days,
            "orders_total": order_count,
            "orders_delivered": delivered_count,
            "orders_cancelled": int(cancelled or 0),
            "revenue_total": round(float(totals[1] or 0), 2),
            "revenue_delivered": round(float(delivered_row[1] or 0), 2),
            "average_delivery_minutes": round(float(delivered_row[2] or 0), 1),
            # Conversion here is delivery completion, not checkout: an order
            # that was placed and then cancelled did not convert into medicine
            # reaching a patient.
            "conversion_rate": round(delivered_count / order_count, 4) if order_count else 0.0,
            "orders_by_status": by_status,
            "top_pharmacies": top_pharmacies,
            "top_cities": top_cities,
            "inventory_value": round(float(inventory_value or 0), 2),
            "pharmacies_total": int(network[0] or 0),
            "pharmacies_partner": int(network[1] or 0),
        }

    async def top_medicines(self, db: AsyncSession, *, days: int = 30, limit: int = 10):
        from app.models.medicine_order import MedicineOrderItem

        since = datetime.now(timezone.utc) - timedelta(days=days)
        rows = await db.execute(
            select(
                MedicineOrderItem.medicine_name,
                func.sum(MedicineOrderItem.quantity),
                func.sum(MedicineOrderItem.line_total),
            )
            .join(MedicineOrder, MedicineOrder.id == MedicineOrderItem.order_id)
            .where(MedicineOrder.created_at >= since)
            .group_by(MedicineOrderItem.medicine_name)
            .order_by(func.sum(MedicineOrderItem.quantity).desc())
            .limit(limit)
        )
        return [
            {"medicine": row[0], "units": int(row[1] or 0), "revenue": round(float(row[2] or 0), 2)}
            for row in rows.all()
        ]

    # ── audit feed ───────────────────────────────────────────────────────

    async def audit_trail(
        self,
        db: AsyncSession,
        *,
        resource_id: str | None = None,
        action: str | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[list[AuditLog], int]:
        # `User` is included because owner provisioning audits the account it
        # acted on as well as the store. Omitting it made every assignment,
        # suspension and password reset invisible in the console's audit feed
        # even though the rows were being written.
        query = select(AuditLog).where(
            AuditLog.resource.in_(
                ["Pharmacy", "PharmacyInventory", "PharmacyDocument", "User"]
            ),
            # Scoped to pharmacy actions so unrelated user administration does
            # not leak into the pharmacy audit view.
            AuditLog.action.like("PHARMACY%") | AuditLog.action.like("INVENTORY%"),
        )
        if resource_id:
            query = query.where(AuditLog.resource_id == resource_id)
        if action:
            query = query.where(AuditLog.action == action)

        total = await db.scalar(select(func.count()).select_from(query.subquery()))
        rows = await db.execute(
            query.order_by(AuditLog.created_at.desc()).offset(skip).limit(limit)
        )
        return list(rows.scalars().all()), int(total or 0)


pharmacy_admin_service = PharmacyAdminService()
