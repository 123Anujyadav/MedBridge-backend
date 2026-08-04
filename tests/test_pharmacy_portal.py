"""
Pharmacy Owner Portal.

The properties pinned here are the ones that would let one store see or alter
another store's business, let an unapproved store operate, or let the portal
rewrite a doctor's prescription.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    AuthorizationException,
    BusinessRuleValidationException,
)
from app.models.case import Case
from app.models.doctor import Doctor
from app.models.medicine_order import (
    ORDER_CANCELLED,
    ORDER_DELIVERED,
    ORDER_OUT_FOR_DELIVERY,
    ORDER_PACKED,
    ORDER_PREPARING,
    ORDER_RECEIVED,
    ORDER_TRANSITIONS,
)
from app.models.patient import Patient
from app.models.pharmacy import (
    Pharmacy,
    PharmacyInventory,
    VERIFICATION_APPROVED,
    VERIFICATION_PENDING,
)
from app.models.prescription import Medication, Prescription
from app.models.user import User
from app.pharmacy.application.ordering import ordering_service
from app.services.pharmacy_portal import (
    ACTION_TO_STATUS,
    RX_REVIEW_OUTCOMES,
    pharmacy_portal_service as portal,
)

ORIGIN_LAT, ORIGIN_LNG = 28.6315, 77.2167


async def _store(db: AsyncSession, *, approved: bool = True) -> Pharmacy:
    pharmacy = Pharmacy(
        name=f"Store-{uuid.uuid4().hex[:6]}",
        address="1 Connaught Place",
        latitude=ORIGIN_LAT,
        longitude=ORIGIN_LNG,
        is_partner=approved,
        is_active=True,
        is_24x7=True,
        delivers=True,
        delivery_radius_km=15.0,
        verification_status=VERIFICATION_APPROVED if approved else VERIFICATION_PENDING,
    )
    db.add(pharmacy)
    await db.flush()
    return pharmacy


async def _owner(db: AsyncSession, pharmacy: Pharmacy | None) -> User:
    user = User(
        id=uuid.uuid4(),
        email=f"owner-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password="x",
        role="pharmacy",
        is_active=True,
        pharmacy_id=pharmacy.id if pharmacy else None,
    )
    db.add(user)
    await db.flush()
    return user


async def _stock(db: AsyncSession, pharmacy: Pharmacy, **overrides) -> PharmacyInventory:
    base = dict(
        pharmacy_id=pharmacy.id,
        sku=f"SKU-{uuid.uuid4().hex[:6]}",
        rxcui="6809",
        medicine_name="Metformin 500mg",
        generic_name="metformin",
        mrp=100.0,
        selling_price=80.0,
        stock_quantity=200,
        low_stock_threshold=10,
        stock_synced_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    item = PharmacyInventory(**base)
    db.add(item)
    await db.flush()
    return item


async def _patient(db: AsyncSession) -> Patient:
    user = User(
        id=uuid.uuid4(),
        email=f"pat-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password="x",
        role="patient",
        is_active=True,
    )
    db.add(user)
    await db.flush()
    patient = Patient(
        id=user.id, first_name="Test", last_name="Patient", phone="+910000000000",
        date_of_birth="1990-01-01", gender="male", allergies=["Penicillin"],
    )
    db.add(patient)
    await db.flush()
    return patient


async def _prescription(db: AsyncSession, patient: Patient) -> Prescription:
    doctor_user = User(
        id=uuid.uuid4(), email=f"dr-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password="x", role="doctor", is_active=True,
    )
    db.add(doctor_user)
    await db.flush()
    doctor = Doctor(
        id=doctor_user.id, first_name="Ananya", last_name="Iyer", phone="+910000000001",
        specialty="Cardiology", license_number=f"LIC-{uuid.uuid4().hex[:6]}",
    )
    db.add(doctor)
    case = Case(
        patient_id=patient.id, patient_name="Test Patient", patient_age=36,
        patient_gender="male", doctor_id=doctor.id, doctor_name="Dr. Ananya Iyer",
        specialty="Cardiology", symptom_summary="Follow-up", status="prescribed",
    )
    db.add(case)
    await db.flush()

    rx = Prescription(
        case_id=case.id, patient_id=patient.id, patient_name="Test Patient",
        doctor_id=doctor.id, doctor_name="Dr. Ananya Iyer",
        diagnosis="Type 2 Diabetes", status="active",
        doctor_specialty="Cardiology", doctor_registration_number="MCI-1234",
        consultation_date=datetime.now(timezone.utc),
        signed_at=datetime.now(timezone.utc),
    )
    db.add(rx)
    await db.flush()
    db.add(
        Medication(
            prescription_id=rx.id, name="Metformin", generic_name="metformin",
            rxcui="6809", dosage="1 tablet", frequency="twice daily",
            duration="30 days", quantity=60,
            start_date="2026-08-03", end_date="2026-09-02",
        )
    )
    await db.flush()
    return rx


async def _order(db, patient, rx, pharmacy, item, quantity=10):
    order = await ordering_service.place_order(
        db,
        patient_id=patient.id, prescription_id=rx.id, pharmacy_id=pharmacy.id,
        selections=[{"inventory_id": str(item.id), "quantity": quantity}],
        delivery_address="221B Baker Street", eta_minutes=25,
    )
    await db.flush()
    return order


# ── store isolation ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_owner_cannot_read_another_stores_order(db: AsyncSession):
    """
    The single most important property: two stores must not see each other's
    business. Scoping is derived from the owner's own user row, so there is no
    request parameter to tamper with.
    """
    store_a, store_b = await _store(db), await _store(db)
    owner_a, owner_b = await _owner(db, store_a), await _owner(db, store_b)

    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item_a = await _stock(db, store_a)
    order = await _order(db, patient, rx, store_a, item_a)

    assert (await portal.get_order(db, owner_a, order.id)).id == order.id
    with pytest.raises(AuthorizationException):
        await portal.get_order(db, owner_b, order.id)


@pytest.mark.asyncio
async def test_owner_cannot_edit_another_stores_inventory(db: AsyncSession):
    store_a, store_b = await _store(db), await _store(db)
    owner_b = await _owner(db, store_b)
    item_a = await _stock(db, store_a)

    with pytest.raises(AuthorizationException):
        await portal.upsert_inventory(
            db, owner_b, payload={"stock_quantity": 0}, item_id=item_a.id
        )
    with pytest.raises(AuthorizationException):
        await portal.delete_inventory(db, owner_b, item_a.id)


@pytest.mark.asyncio
async def test_inventory_listing_is_scoped_to_the_owners_store(db: AsyncSession):
    store_a, store_b = await _store(db), await _store(db)
    owner_a = await _owner(db, store_a)
    await _stock(db, store_a, medicine_name="Mine")
    await _stock(db, store_b, medicine_name="Theirs")

    items, _ = await portal.list_inventory(db, owner_a)
    assert items
    assert all(i.pharmacy_id == store_a.id for i in items)


@pytest.mark.asyncio
async def test_account_without_a_store_is_refused(db: AsyncSession):
    orphan = await _owner(db, None)
    with pytest.raises(AuthorizationException):
        await portal.dashboard(db, orphan)


# ── order workflow ───────────────────────────────────────────────────────


def test_portal_actions_map_onto_the_existing_lifecycle():
    """
    The portal's vocabulary is labels over `ORDER_TRANSITIONS`, not new states.
    Adding states would change what the patient's tracking timeline renders.
    """
    assert ACTION_TO_STATUS["accept"] == ORDER_PREPARING
    assert ACTION_TO_STATUS["ready"] == ORDER_PACKED
    assert ACTION_TO_STATUS["dispatch"] == ORDER_OUT_FOR_DELIVERY
    for target in set(ACTION_TO_STATUS.values()):
        assert target in ORDER_TRANSITIONS


@pytest.mark.asyncio
async def test_accept_moves_the_order_to_preparing(db: AsyncSession):
    store = await _store(db)
    owner = await _owner(db, store)
    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item = await _stock(db, store)
    order = await _order(db, patient, rx, store, item)

    await portal.act_on_order(db, owner, order.id, action="accept", note="Stock confirmed")
    await db.flush()
    assert order.status == ORDER_PREPARING


@pytest.mark.asyncio
async def test_full_workflow_reaches_delivered(db: AsyncSession):
    store = await _store(db)
    owner = await _owner(db, store)
    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item = await _stock(db, store)
    order = await _order(db, patient, rx, store, item)

    for action in ("accept", "ready", "dispatch", "deliver"):
        await portal.act_on_order(db, owner, order.id, action=action)
        await db.flush()

    assert order.status == ORDER_DELIVERED
    assert order.dispatched_at is not None
    assert order.delivered_at is not None


@pytest.mark.asyncio
async def test_rejecting_a_new_order_returns_the_stock(db: AsyncSession):
    """Refusing an order must not destroy inventory the store still holds."""
    store = await _store(db)
    owner = await _owner(db, store)
    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item = await _stock(db, store, stock_quantity=100)
    order = await _order(db, patient, rx, store, item, quantity=30)
    await db.flush()
    assert item.stock_quantity == 70

    await portal.act_on_order(db, owner, order.id, action="reject", note="Out of stock")
    await db.flush()

    assert order.status == ORDER_CANCELLED
    assert item.stock_quantity == 100


@pytest.mark.asyncio
async def test_rejection_is_refused_once_accepted(db: AsyncSession):
    store = await _store(db)
    owner = await _owner(db, store)
    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item = await _stock(db, store)
    order = await _order(db, patient, rx, store, item)

    await portal.act_on_order(db, owner, order.id, action="accept")
    await db.flush()

    with pytest.raises(BusinessRuleValidationException):
        await portal.act_on_order(db, owner, order.id, action="reject")


@pytest.mark.asyncio
async def test_unknown_action_is_refused(db: AsyncSession):
    store = await _store(db)
    owner = await _owner(db, store)
    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item = await _stock(db, store)
    order = await _order(db, patient, rx, store, item)

    with pytest.raises(BusinessRuleValidationException):
        await portal.act_on_order(db, owner, order.id, action="teleport")


@pytest.mark.asyncio
async def test_stage_skipping_is_refused(db: AsyncSession):
    """The shared state machine still governs, even through the portal."""
    store = await _store(db)
    owner = await _owner(db, store)
    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item = await _stock(db, store)
    order = await _order(db, patient, rx, store, item)

    with pytest.raises(BusinessRuleValidationException):
        await portal.act_on_order(db, owner, order.id, action="deliver")


# ── prescription review ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_review_pack_exposes_prescriber_allergies_and_findings(db: AsyncSession):
    store = await _store(db)
    owner = await _owner(db, store)
    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item = await _stock(db, store)
    order = await _order(db, patient, rx, store, item)

    pack = await portal.prescription_for_order(db, owner, order.id)

    assert pack["prescriber"]["doctor_name"] == "Dr. Ananya Iyer"
    assert pack["prescriber"]["registration_number"] == "MCI-1234"
    assert pack["patient_allergies"] == ["Penicillin"]
    assert pack["medications"][0]["name"] == "Metformin"
    assert pack["signed_at"] is not None


@pytest.mark.asyncio
async def test_review_flags_expired_batches(db: AsyncSession):
    """
    Expiry is checked against what this store would actually hand over — the
    prescription itself cannot tell you the batch is out of date.
    """
    store = await _store(db)
    owner = await _owner(db, store)
    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item = await _stock(
        db, store,
        batch_number="B-001",
        expiry_date=datetime.now(timezone.utc) - timedelta(days=5),
    )
    order = await _order(db, patient, rx, store, item)

    pack = await portal.prescription_for_order(db, owner, order.id)
    assert any(a["state"] == "expired" for a in pack["expiry_alerts"])
    assert pack["expiry_alerts"][0]["batch_number"] == "B-001"


@pytest.mark.asyncio
async def test_review_never_modifies_the_prescription(db: AsyncSession):
    """
    A pharmacy may refuse to dispense. It may never rewrite what the doctor
    ordered — the decision lands on the order's event trail instead.
    """
    store = await _store(db)
    owner = await _owner(db, store)
    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item = await _stock(db, store)
    order = await _order(db, patient, rx, store, item)

    # Medication count is read through an explicit query rather than the lazy
    # relationship — touching `rx.medications` here would emit IO outside the
    # async context and raise MissingGreenlet.
    from sqlalchemy import func as sa_func, select as sa_select

    async def medication_count() -> int:
        return int(
            await db.scalar(
                sa_select(sa_func.count())
                .select_from(Medication)
                .where(Medication.prescription_id == rx.id)
            )
            or 0
        )

    before = (rx.diagnosis, rx.status, rx.notes, await medication_count())

    await portal.record_prescription_review(
        db, owner, order.id, outcome="rejected", note="Illegible signature"
    )
    await db.flush()

    assert (rx.diagnosis, rx.status, rx.notes, await medication_count()) == before
    assert order.status == ORDER_CANCELLED


@pytest.mark.asyncio
async def test_approving_the_prescription_starts_preparation(db: AsyncSession):
    store = await _store(db)
    owner = await _owner(db, store)
    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item = await _stock(db, store)
    order = await _order(db, patient, rx, store, item)

    await portal.record_prescription_review(
        db, owner, order.id, outcome="approved", note="Verified"
    )
    await db.flush()
    assert order.status == ORDER_PREPARING


@pytest.mark.asyncio
async def test_clarification_records_a_note_without_moving_the_order(db: AsyncSession):
    store = await _store(db)
    owner = await _owner(db, store)
    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item = await _stock(db, store)
    order = await _order(db, patient, rx, store, item)

    await portal.record_prescription_review(
        db, owner, order.id, outcome="clarification_requested", note="Dose unclear"
    )
    await db.flush()

    assert order.status == ORDER_RECEIVED
    fresh = await portal.get_order(db, owner, order.id)
    assert any("Clarification requested" in e.note for e in fresh.events)


def test_review_outcomes_are_a_closed_set():
    assert set(RX_REVIEW_OUTCOMES) == {"approved", "clarification_requested", "rejected"}


@pytest.mark.asyncio
async def test_invalid_review_outcome_is_refused(db: AsyncSession):
    store = await _store(db)
    owner = await _owner(db, store)
    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item = await _stock(db, store)
    order = await _order(db, patient, rx, store, item)

    with pytest.raises(BusinessRuleValidationException):
        await portal.record_prescription_review(
            db, owner, order.id, outcome="maybe", note=""
        )


# ── inventory reaches patient search ─────────────────────────────────────


@pytest.mark.asyncio
async def test_stock_update_is_visible_to_patient_search_immediately(db: AsyncSession):
    """
    The portal writes to the same `pharmacy_inventory` row the patient-facing
    provider reads, so there is no propagation step that could drift.
    """
    from app.pharmacy.domain.ports import MedicineRequirement
    from app.pharmacy.infrastructure.local_db_provider import LocalDbPharmacyProvider

    store = await _store(db)
    owner = await _owner(db, store)
    item = await _stock(db, store, stock_quantity=50)

    # Moved well away from the shared origin every other fixture uses. The `db`
    # fixture commits, so earlier tests leave a growing pile of pharmacies at
    # that point — searching there would rank this store out of the window and
    # fail on ordering rather than on the availability being asserted.
    store.latitude = 19.0760
    store.longitude = 72.8777
    await db.flush()

    requirement = [MedicineRequirement(name="Metformin", rxcui="6809", quantity=10)]
    provider = LocalDbPharmacyProvider()

    async def offer_for_store():
        offers = await provider.find_offers(
            db=db,
            latitude=store.latitude,
            longitude=store.longitude,
            requirements=requirement,
            radius_km=2.0,
            limit=10,
        )
        return next((o for o in offers if o.pharmacy_id == str(store.id)), None)

    before = await offer_for_store()
    assert before is not None, "the store should be found at its own coordinates"
    assert before.items[0].status == "available"

    await portal.upsert_inventory(
        db, owner, payload={"stock_quantity": 0}, item_id=item.id
    )
    await db.flush()

    after = await offer_for_store()
    assert after is not None
    assert after.items[0].status == "out_of_stock"
    assert after.can_order is False


@pytest.mark.asyncio
async def test_barcode_lookup_is_store_scoped(db: AsyncSession):
    store_a, store_b = await _store(db), await _store(db)
    owner_a, owner_b = await _owner(db, store_a), await _owner(db, store_b)
    code = f"BAR{uuid.uuid4().hex[:8]}"
    await _stock(db, store_a, barcode=code)

    assert (await portal.find_by_code(db, owner_a, code)) is not None
    # The same barcode must not resolve for a different store.
    assert (await portal.find_by_code(db, owner_b, code)) is None


@pytest.mark.asyncio
async def test_blank_code_lookup_returns_nothing(db: AsyncSession):
    store = await _store(db)
    owner = await _owner(db, store)
    assert (await portal.find_by_code(db, owner, "   ")) is None


# ── dashboard & analytics ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dashboard_reports_store_scoped_figures(db: AsyncSession):
    store = await _store(db)
    owner = await _owner(db, store)
    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item = await _stock(db, store, stock_quantity=5, low_stock_threshold=10)
    await _order(db, patient, rx, store, item, quantity=1)

    data = await portal.dashboard(db, owner)

    assert data["pharmacy_id"] == str(store.id)
    assert data["orders_today"] >= 1
    assert data["pending_prescriptions"] >= 1
    assert data["stock_low"] >= 1
    assert data["inventory_value"] >= 0


@pytest.mark.asyncio
async def test_analytics_shape_and_invariants(db: AsyncSession):
    store = await _store(db)
    owner = await _owner(db, store)
    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item = await _stock(db, store)
    await _order(db, patient, rx, store, item)

    data = await portal.analytics(db, owner, days=30)

    for key in (
        "orders", "revenue", "average_basket", "fastest_moving", "slowest_moving",
        "peak_hours", "top_customers", "inventory_value", "expiry_loss",
    ):
        assert key in data
    assert data["revenue"] >= 0
    assert data["expiry_loss"] >= 0


@pytest.mark.asyncio
async def test_alerts_surface_low_stock_and_expiry(db: AsyncSession):
    store = await _store(db)
    owner = await _owner(db, store)
    await _stock(db, store, medicine_name="Low one", stock_quantity=2, low_stock_threshold=10)
    await _stock(
        db, store, medicine_name="Old one",
        expiry_date=datetime.now(timezone.utc) - timedelta(days=1),
    )

    alerts = await portal.alerts(db, owner)
    kinds = {a["type"] for a in alerts}
    assert "low_stock" in kinds
    assert "expiry" in kinds
    # Critical items must sort above warnings.
    severities = [a["severity"] for a in alerts]
    assert severities == sorted(severities, key=lambda s: {"critical": 0, "warning": 1, "info": 2}[s])


@pytest.mark.asyncio
async def test_sales_report_rows_carry_gst(db: AsyncSession):
    store = await _store(db)
    owner = await _owner(db, store)
    patient = await _patient(db)
    rx = await _prescription(db, patient)
    item = await _stock(db, store, gst_percent=12.0)
    await _order(db, patient, rx, store, item, quantity=10)

    rows = await portal.sales_report(db, owner, days=30)
    assert rows
    assert rows[0]["gst"] > 0
    assert rows[0]["total"] >= rows[0]["subtotal"]


def test_report_csv_emits_requested_columns():
    columns = ("order_number", "total")
    csv_text = portal.report_to_csv([{"order_number": "MB-1", "total": 10.0}], columns)
    assert csv_text.splitlines()[0] == "order_number,total"
