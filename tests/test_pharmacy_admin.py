"""
Pharmacy administration module.

The properties pinned here are the ones that would let an unverified pharmacy
dispense, lose a compliance trail, or corrupt stock: the verification state
machine, the audit trail, CSV import validation, and the guarantee that none of
this changed what Phase 2 dispenses.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BusinessRuleValidationException
from app.models.audit import AuditLog
from app.models.pharmacy import (
    Pharmacy,
    PharmacyDocument,
    PharmacyInventory,
    VERIFICATION_APPROVED,
    VERIFICATION_DOCUMENT_REVIEW,
    VERIFICATION_PENDING,
    VERIFICATION_REJECTED,
    VERIFICATION_STATUSES,
    VERIFICATION_SUBMITTED,
    VERIFICATION_SUSPENDED,
    VERIFICATION_TRANSITIONS,
)
from app.models.user import User
from app.services.pharmacy_admin import (
    CSV_COLUMNS,
    EDITABLE_FIELDS,
    PharmacyAdminService,
    pharmacy_admin_service as service,
)


async def _admin(db: AsyncSession) -> User:
    user = User(
        id=uuid.uuid4(),
        email=f"admin-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password="x",
        role="admin",
        is_active=True,
    )
    db.add(user)
    await db.flush()
    return user


def _payload(**overrides) -> dict:
    base = {
        "name": "Apex Chemist",
        "address": "1 Connaught Place",
        "city": "New Delhi",
        "latitude": 28.6315,
        "longitude": 77.2167,
        "phone": "+911100000000",
        "owner_name": "R. Sharma",
        "business_name": "Apex Health Pvt Ltd",
    }
    base.update(overrides)
    return base


# ── the state machine ────────────────────────────────────────────────────


def test_every_verification_status_is_in_the_transition_table():
    """A status with no entry would silently be a dead end."""
    assert set(VERIFICATION_TRANSITIONS) == set(VERIFICATION_STATUSES)


def test_the_approval_path_is_reachable_end_to_end():
    path = [
        VERIFICATION_PENDING,
        VERIFICATION_SUBMITTED,
        VERIFICATION_DOCUMENT_REVIEW,
        VERIFICATION_APPROVED,
    ]
    for current, nxt in zip(path, path[1:]):
        assert nxt in VERIFICATION_TRANSITIONS[current], f"{current} -> {nxt} blocked"


def test_approval_cannot_skip_document_review():
    """The whole point of the workflow is that documents are seen first."""
    assert VERIFICATION_APPROVED not in VERIFICATION_TRANSITIONS[VERIFICATION_PENDING]
    assert VERIFICATION_APPROVED not in VERIFICATION_TRANSITIONS[VERIFICATION_SUBMITTED]


def test_suspension_is_reversible_and_rejection_restarts():
    assert VERIFICATION_APPROVED in VERIFICATION_TRANSITIONS[VERIFICATION_SUSPENDED]
    assert VERIFICATION_TRANSITIONS[VERIFICATION_REJECTED] == (VERIFICATION_PENDING,)


def test_workflow_owned_columns_are_not_directly_editable():
    """
    An administrator must not be able to PUT `verification_status` or
    `is_partner` straight onto a pharmacy — that would bypass document review
    entirely and is exactly what the allow-list exists to prevent.
    """
    for field in ("verification_status", "is_partner", "verified_at", "verified_by"):
        assert field not in EDITABLE_FIELDS


# ── onboarding ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_new_pharmacy_starts_unverified_and_cannot_dispense(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await service.create(db, payload=_payload(), actor=admin)
    await db.flush()

    assert pharmacy.verification_status == VERIFICATION_PENDING
    assert pharmacy.is_partner is False
    # can_fulfil is what patient search reads. A brand-new record must not be
    # dispensable before anyone has looked at its licence.
    assert pharmacy.can_fulfil is False


@pytest.mark.asyncio
async def test_duplicate_gst_number_is_refused(db: AsyncSession):
    admin = await _admin(db)
    gst = f"GST{uuid.uuid4().hex[:10].upper()}"
    await service.create(db, payload=_payload(gst_number=gst), actor=admin)
    await db.flush()

    with pytest.raises(BusinessRuleValidationException) as exc:
        await service.create(db, payload=_payload(gst_number=gst), actor=admin)
    assert gst in str(exc.value)


@pytest.mark.asyncio
async def test_approval_grants_partner_status(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await service.create(db, payload=_payload(), actor=admin)
    await db.flush()

    for stage in (
        VERIFICATION_SUBMITTED,
        VERIFICATION_DOCUMENT_REVIEW,
        VERIFICATION_APPROVED,
    ):
        await service.transition_verification(
            db, pharmacy.id, to_status=stage, note=f"moved to {stage}", actor=admin
        )

    assert pharmacy.verification_status == VERIFICATION_APPROVED
    assert pharmacy.is_partner is True
    assert pharmacy.is_active is True
    assert pharmacy.verified_at is not None
    # Approval is what opens the dispensing gate.
    assert pharmacy.can_fulfil is True


@pytest.mark.asyncio
async def test_suspension_withdraws_partner_status(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await service.create(db, payload=_payload(), actor=admin)
    await db.flush()
    for stage in (
        VERIFICATION_SUBMITTED,
        VERIFICATION_DOCUMENT_REVIEW,
        VERIFICATION_APPROVED,
    ):
        await service.transition_verification(db, pharmacy.id, to_status=stage, note="", actor=admin)

    await service.transition_verification(
        db, pharmacy.id, to_status=VERIFICATION_SUSPENDED,
        note="Drug licence lapsed", actor=admin,
    )

    assert pharmacy.is_partner is False
    assert pharmacy.can_fulfil is False
    assert pharmacy.suspension_reason == "Drug licence lapsed"


@pytest.mark.asyncio
async def test_illegal_transition_is_refused(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await service.create(db, payload=_payload(), actor=admin)
    await db.flush()

    with pytest.raises(BusinessRuleValidationException) as exc:
        await service.transition_verification(
            db, pharmacy.id, to_status=VERIFICATION_APPROVED, note="", actor=admin
        )
    assert "cannot become" in str(exc.value)


@pytest.mark.asyncio
async def test_verification_trail_records_who_and_when(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await service.create(db, payload=_payload(), actor=admin)
    await db.flush()
    await service.transition_verification(
        db, pharmacy.id, to_status=VERIFICATION_SUBMITTED, note="Docs in", actor=admin
    )

    fresh = await service.get(db, pharmacy.id)
    trail = [(e.from_status, e.to_status) for e in fresh.verification_events]
    assert (None, VERIFICATION_PENDING) in trail
    assert (VERIFICATION_PENDING, VERIFICATION_SUBMITTED) in trail
    assert all(event.actor_id == admin.id for event in fresh.verification_events)


# ── audit ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_field_changes_are_audited_individually(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await service.create(db, payload=_payload(), actor=admin)
    await db.flush()

    await service.update(
        db, pharmacy.id,
        payload={"delivery_fee": 49.0, "min_order_value": 199.0},
        actor=admin, ip="203.0.113.9",
    )
    await db.flush()

    entries, _ = await service.audit_trail(db, resource_id=str(pharmacy.id))
    changed = {e.field_changed for e in entries if e.field_changed}
    assert {"delivery_fee", "min_order_value"} <= changed

    fee_entry = next(e for e in entries if e.field_changed == "delivery_fee")
    assert fee_entry.new_value == "49.0"
    assert fee_entry.ip_address == "203.0.113.9"
    assert fee_entry.user_role == "admin"


@pytest.mark.asyncio
async def test_bank_details_are_masked_in_the_audit_trail(db: AsyncSession):
    """
    The audit log is readable by other administrators, so a settlement account
    number must not sit in it in full.
    """
    admin = await _admin(db)
    pharmacy = await service.create(db, payload=_payload(), actor=admin)
    await db.flush()

    await service.update(
        db, pharmacy.id, payload={"bank_account_number": "123456789012"}, actor=admin
    )
    await db.flush()

    entries, _ = await service.audit_trail(db, resource_id=str(pharmacy.id))
    entry = next(e for e in entries if e.field_changed == "bank_account_number")
    assert entry.new_value == "••••9012"
    assert "123456789012" not in (entry.new_value or "")


# ── deletion safety ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deletion_is_soft_and_preserves_the_row(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await service.create(db, payload=_payload(), actor=admin)
    await db.flush()
    pharmacy_id = pharmacy.id

    await service.soft_delete(db, pharmacy_id, actor=admin)
    await db.flush()

    row = await db.get(Pharmacy, pharmacy_id)
    assert row is not None
    assert row.deleted_at is not None
    assert row.is_partner is False
    assert row.can_fulfil is False


# ── inventory & stock health ─────────────────────────────────────────────


def _inventory(**overrides) -> PharmacyInventory:
    base = dict(
        pharmacy_id=uuid.uuid4(), sku="SKU1", medicine_name="Metformin 500mg",
        mrp=100.0, selling_price=80.0, stock_quantity=50, low_stock_threshold=10,
    )
    base.update(overrides)
    return PharmacyInventory(**base)


def test_expired_stock_outranks_quantity():
    """A full shelf of expired stock is a problem, not availability."""
    item = _inventory(
        stock_quantity=500,
        expiry_date=datetime.now(timezone.utc) - timedelta(days=1),
    )
    assert item.stock_state == "expired"
    # The patient-facing property is deliberately untouched by expiry.
    assert item.availability == "available"


def test_near_expiry_is_flagged_before_it_lapses():
    item = _inventory(expiry_date=datetime.now(timezone.utc) + timedelta(days=30))
    assert item.stock_state == "near_expiry"


def test_reorder_level_produces_critical_before_low():
    item = _inventory(stock_quantity=3, reorder_level=5, low_stock_threshold=10)
    assert item.stock_state == "critical"


def test_stock_state_falls_back_to_available():
    item = _inventory(stock_quantity=500, reorder_level=5)
    assert item.stock_state == "available"


def test_inventory_value_uses_selling_price():
    assert _inventory(selling_price=80.0, stock_quantity=10).inventory_value == 800.0


def test_patient_facing_availability_is_unchanged_by_this_module():
    """
    Phase 2's dispensing semantics must not have moved. These are the exact
    bands the patient search relies on.
    """
    assert _inventory(stock_quantity=0).availability == "out_of_stock"
    assert _inventory(stock_quantity=5, low_stock_threshold=10).availability == "limited"
    assert _inventory(stock_quantity=50, low_stock_threshold=10).availability == "available"


# ── CSV round trip ───────────────────────────────────────────────────────


def test_export_emits_the_documented_columns():
    csv_text = PharmacyAdminService.export_inventory_csv([_inventory()])
    header = csv_text.splitlines()[0].split(",")
    assert header == list(CSV_COLUMNS)


def test_csv_row_parsing_coerces_numbers_and_dates():
    parsed = PharmacyAdminService._parse_csv_row(
        {
            "sku": "A1", "medicine_name": "Metformin", "mrp": "100.50",
            "stock_quantity": "40", "expiry_date": "2027-01-31",
        }
    )
    assert parsed["mrp"] == 100.5
    assert parsed["stock_quantity"] == 40
    assert parsed["expiry_date"].year == 2027


def test_csv_row_rejects_a_bad_number_with_a_readable_message():
    with pytest.raises(ValueError) as exc:
        PharmacyAdminService._parse_csv_row({"sku": "A1", "mrp": "free"})
    assert "not a number" in str(exc.value)


def test_csv_row_rejects_a_bad_date():
    with pytest.raises(ValueError) as exc:
        PharmacyAdminService._parse_csv_row({"sku": "A1", "expiry_date": "31/01/2027"})
    assert "ISO date" in str(exc.value)


def test_csv_row_rejects_negative_stock():
    with pytest.raises(ValueError):
        PharmacyAdminService._parse_csv_row({"sku": "A1", "stock_quantity": "-5"})


@pytest.mark.asyncio
async def test_import_loads_good_rows_and_reports_bad_ones(db: AsyncSession):
    """
    A 3-row file with one bad price should load 2 rows and name the bad line —
    rejecting the whole batch would make large imports unusable.
    """
    admin = await _admin(db)
    pharmacy = await service.create(db, payload=_payload(), actor=admin)
    await db.flush()

    csv_text = (
        "sku,medicine_name,mrp,selling_price,stock_quantity\n"
        "A1,Metformin,100,80,50\n"
        "A2,Atorvastatin,notanumber,90,20\n"
        "A3,Amlodipine,60,50,30\n"
    )
    result = await service.import_inventory_csv(
        db, pharmacy.id, content=csv_text, actor=admin
    )

    assert result["created"] == 2
    assert len(result["errors"]) == 1
    assert result["errors"][0]["line"] == 3


@pytest.mark.asyncio
async def test_import_updates_existing_rows_by_sku(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await service.create(db, payload=_payload(), actor=admin)
    await db.flush()

    first = "sku,medicine_name,mrp,selling_price,stock_quantity\nA1,Metformin,100,80,50\n"
    await service.import_inventory_csv(db, pharmacy.id, content=first, actor=admin)

    second = "sku,medicine_name,mrp,selling_price,stock_quantity\nA1,Metformin,100,75,120\n"
    result = await service.import_inventory_csv(db, pharmacy.id, content=second, actor=admin)

    assert result["created"] == 0
    assert result["updated"] == 1

    row = (
        await db.execute(
            select(PharmacyInventory).where(
                PharmacyInventory.pharmacy_id == pharmacy.id,
                PharmacyInventory.sku == "A1",
            )
        )
    ).scalar_one()
    assert row.stock_quantity == 120
    assert row.selling_price == 75


# ── documents ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_document_type_is_refused(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await service.create(db, payload=_payload(), actor=admin)
    await db.flush()

    with pytest.raises(BusinessRuleValidationException):
        await service.add_document(
            db, pharmacy.id,
            payload={"doc_type": "selfie", "file_url": "/x.png"},
            actor=admin,
        )


@pytest.mark.asyncio
async def test_expiring_documents_include_already_expired(db: AsyncSession):
    """
    A licence that lapsed last week is more urgent than one lapsing next month;
    an alert feed that filters it out is worse than useless.
    """
    admin = await _admin(db)
    pharmacy = await service.create(db, payload=_payload(), actor=admin)
    await db.flush()

    await service.add_document(
        db, pharmacy.id,
        payload={
            "doc_type": "drug_license",
            "file_url": "/lapsed.pdf",
            "expires_at": datetime.now(timezone.utc) - timedelta(days=7),
        },
        actor=admin,
    )
    await db.flush()

    alerts = await service.expiring_documents(db, within_days=30)
    assert any(str(d.pharmacy_id) == str(pharmacy.id) and d.is_expired for d in alerts)


def test_document_expiry_helpers():
    past = PharmacyDocument(
        pharmacy_id=uuid.uuid4(), doc_type="drug_license", file_url="/x.pdf",
        expires_at=datetime.now(timezone.utc) - timedelta(days=2),
    )
    assert past.is_expired is True
    assert past.days_to_expiry is not None and past.days_to_expiry < 0

    undated = PharmacyDocument(
        pharmacy_id=uuid.uuid4(), doc_type="pan_card", file_url="/y.pdf"
    )
    assert undated.is_expired is False
    assert undated.days_to_expiry is None


# ── filters & paging ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_matches_business_identity_fields(db: AsyncSession):
    admin = await _admin(db)
    unique = uuid.uuid4().hex[:8]
    await service.create(
        db, payload=_payload(name=f"Zeta {unique}", business_name=f"Zeta Health {unique}"),
        actor=admin,
    )
    await db.flush()

    found, total = await service.list_pharmacies(db, search=unique)
    assert total >= 1
    assert any(unique in (p.name or "") for p in found)


@pytest.mark.asyncio
async def test_pagination_returns_the_unpaginated_total(db: AsyncSession):
    admin = await _admin(db)
    city = f"City{uuid.uuid4().hex[:6]}"
    for index in range(3):
        await service.create(db, payload=_payload(name=f"P{index}", city=city), actor=admin)
    await db.flush()

    page, total = await service.list_pharmacies(db, city=city, skip=0, limit=2)
    assert len(page) == 2
    assert total == 3


@pytest.mark.asyncio
async def test_verification_status_filter(db: AsyncSession):
    admin = await _admin(db)
    city = f"City{uuid.uuid4().hex[:6]}"
    await service.create(db, payload=_payload(city=city), actor=admin)
    await db.flush()

    pending, _ = await service.list_pharmacies(
        db, city=city, verification_status=VERIFICATION_PENDING
    )
    approved, _ = await service.list_pharmacies(
        db, city=city, verification_status=VERIFICATION_APPROVED
    )
    assert len(pending) == 1
    assert approved == []


# ── analytics ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_analytics_returns_a_complete_shape(db: AsyncSession):
    """
    Runs against whatever is in the test database. The assertion is on shape
    and invariants, not on figures that depend on seeding order.
    """
    data = await service.analytics(db, days=30)

    for key in (
        "orders_total", "orders_delivered", "revenue_total", "conversion_rate",
        "inventory_value", "pharmacies_total", "pharmacies_partner",
        "top_pharmacies", "top_cities", "orders_by_status",
    ):
        assert key in data

    assert 0.0 <= data["conversion_rate"] <= 1.0
    assert data["orders_delivered"] <= data["orders_total"]
    assert data["pharmacies_partner"] <= data["pharmacies_total"]
    assert data["inventory_value"] >= 0
