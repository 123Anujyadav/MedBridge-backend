"""
End-to-end prescription → pharmacy workflow against the database.

Covers the parts that only break when real rows and a real transaction are
involved: stock actually moving, a cancellation actually returning it, and the
geo/inventory queries actually finding what they should.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthorizationException, BusinessRuleValidationException
from app.models.case import Case
from app.models.doctor import Doctor
from app.models.medicine_order import (
    ORDER_DELIVERED,
    ORDER_OUT_FOR_DELIVERY,
    ORDER_PACKED,
    ORDER_PREPARING,
)
from app.models.patient import Patient
from app.models.pharmacy import Pharmacy, PharmacyInventory
from app.models.prescription import Medication, Prescription
from app.models.user import User
from app.pharmacy.application.ordering import ordering_service
from app.pharmacy.domain.ports import MedicineRequirement
from app.pharmacy.infrastructure.local_db_provider import LocalDbPharmacyProvider

# Connaught Place, New Delhi — the origin every fixture is positioned around.
ORIGIN_LAT, ORIGIN_LNG = 28.6315, 77.2167


async def _seed_patient(db: AsyncSession) -> Patient:
    user = User(
        id=uuid.uuid4(),
        email=f"rx-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password="x",
        role="patient",
        is_active=True,
    )
    db.add(user)
    await db.flush()

    patient = Patient(
        id=user.id,
        first_name="Test",
        last_name="Patient",
        phone="+910000000000",
        date_of_birth="1990-01-01",
        gender="male",
        blood_type="O+",
    )
    db.add(patient)
    await db.flush()
    return patient


async def _seed_prescription(db: AsyncSession, patient: Patient) -> Prescription:
    doctor_user = User(
        id=uuid.uuid4(),
        email=f"dr-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password="x",
        role="doctor",
        is_active=True,
    )
    db.add(doctor_user)
    await db.flush()

    doctor = Doctor(
        id=doctor_user.id,
        first_name="Ananya",
        last_name="Iyer",
        phone="+910000000001",
        specialty="Cardiology",
        license_number=f"LIC-{uuid.uuid4().hex[:6]}",
    )
    db.add(doctor)

    case = Case(
        patient_id=patient.id,
        patient_name="Test Patient",
        patient_age=36,
        patient_gender="male",
        doctor_id=doctor.id,
        doctor_name="Dr. Ananya Iyer",
        specialty="Cardiology",
        symptom_summary="Routine diabetes follow-up",
        status="prescribed",
    )
    db.add(case)
    await db.flush()

    rx = Prescription(
        case_id=case.id,
        patient_id=patient.id,
        patient_name="Test Patient",
        doctor_id=doctor.id,
        doctor_name="Dr. Ananya Iyer",
        diagnosis="Type 2 Diabetes",
        status="active",
        consultation_date=datetime.now(timezone.utc),
        signed_at=datetime.now(timezone.utc),
    )
    db.add(rx)
    await db.flush()

    db.add(
        Medication(
            prescription_id=rx.id,
            name="Metformin",
            generic_name="metformin",
            rxcui="6809",
            dosage="1 tablet",
            frequency="twice daily",
            duration="30 days",
            quantity=60,
            start_date="2026-08-03",
            end_date="2026-09-02",
        )
    )
    await db.flush()
    return rx


async def _seed_pharmacy(
    db: AsyncSession, *, stock: int = 100, price: float = 80.0, distance_offset: float = 0.01
) -> tuple[Pharmacy, PharmacyInventory]:
    pharmacy = Pharmacy(
        name="Apex Chemist",
        address="1 Connaught Place",
        latitude=ORIGIN_LAT + distance_offset,
        longitude=ORIGIN_LNG,
        is_partner=True,
        is_active=True,
        is_24x7=True,
        delivers=True,
        delivery_radius_km=15.0,
        delivery_fee=25.0,
        avg_prep_minutes=10,
        rating=4.5,
        total_ratings=200,
    )
    db.add(pharmacy)
    await db.flush()

    item = PharmacyInventory(
        pharmacy_id=pharmacy.id,
        sku=f"SKU-{uuid.uuid4().hex[:6]}",
        rxcui="6809",
        medicine_name="Metformin 500mg",
        generic_name="metformin",
        strength="500 mg",
        is_generic=True,
        mrp=100.0,
        selling_price=price,
        discount_percent=20.0,
        stock_quantity=stock,
        low_stock_threshold=10,
        stock_synced_at=datetime.now(timezone.utc),
    )
    db.add(item)
    await db.flush()
    return pharmacy, item


@pytest.mark.asyncio
async def test_offers_find_a_nearby_partner_with_stock(db: AsyncSession):
    """The discovery → availability → pricing path, end to end."""
    pharmacy, item = await _seed_pharmacy(db)

    offers = await LocalDbPharmacyProvider().find_offers(
        db=db,
        latitude=ORIGIN_LAT,
        longitude=ORIGIN_LNG,
        requirements=[MedicineRequirement(name="Metformin", rxcui="6809", quantity=60)],
        # The `db` fixture commits, so every earlier test leaves pharmacies at
        # these coordinates behind. A top-5 window would rank this one out and
        # fail on ordering rather than on the availability being asserted.
        radius_km=10.0,
        limit=50,
    )

    assert offers, "the seeded partner pharmacy should have been found"
    offer = next(o for o in offers if o.pharmacy_id == str(pharmacy.id))
    assert offer.can_order is True
    assert offer.fully_available is True
    assert offer.items[0].unit_price == 80.0
    assert offer.items[0].line_total == 4800.0
    assert offer.subtotal == 4800.0
    assert offer.grand_total == 4825.0  # includes the 25.00 delivery fee
    assert offer.eta_minutes > 0


@pytest.mark.asyncio
async def test_pharmacy_beyond_the_radius_is_excluded(db: AsyncSession):
    # ~0.5 degrees of latitude is roughly 55km — well outside a 10km search.
    far, _ = await _seed_pharmacy(db, distance_offset=0.5)

    offers = await LocalDbPharmacyProvider().find_offers(
        db=db,
        latitude=ORIGIN_LAT,
        longitude=ORIGIN_LNG,
        requirements=[MedicineRequirement(name="Metformin", rxcui="6809", quantity=1)],
        radius_km=10.0,
        limit=5,
    )

    # Asserted against this pharmacy specifically rather than an empty result:
    # the `db` fixture commits, so rows seeded by earlier tests are still
    # present and a global emptiness check would be testing the wrong thing.
    assert all(o.pharmacy_id != str(far.id) for o in offers)


@pytest.mark.asyncio
async def test_non_partner_pharmacy_is_never_offered(db: AsyncSession):
    """
    Places-discovered shops have no inventory and no relationship with us.
    Offering "Order now" against one would be a button that cannot work.
    """
    pharmacy, _ = await _seed_pharmacy(db)
    pharmacy.is_partner = False
    await db.flush()

    offers = await LocalDbPharmacyProvider().find_offers(
        db=db,
        latitude=ORIGIN_LAT,
        longitude=ORIGIN_LNG,
        requirements=[MedicineRequirement(name="Metformin", rxcui="6809", quantity=1)],
        radius_km=10.0,
        limit=5,
    )
    assert all(o.pharmacy_id != str(pharmacy.id) for o in offers)


@pytest.mark.asyncio
async def test_placing_an_order_reserves_stock(db: AsyncSession):
    patient = await _seed_patient(db)
    rx = await _seed_prescription(db, patient)
    pharmacy, item = await _seed_pharmacy(db, stock=100)

    order = await ordering_service.place_order(
        db,
        patient_id=patient.id,
        prescription_id=rx.id,
        pharmacy_id=pharmacy.id,
        selections=[{"inventory_id": str(item.id), "quantity": 60}],
        delivery_address="221B Baker Street",
        eta_minutes=25,
    )
    await db.flush()

    assert order.status == "received"
    assert order.subtotal == 4800.0
    assert order.total == 4825.0
    assert order.order_number.startswith("MB-")
    # The reservation is the point: stock moved in the same transaction.
    assert item.stock_quantity == 40
    assert order.estimated_delivery_at is not None


@pytest.mark.asyncio
async def test_order_cannot_exceed_available_stock(db: AsyncSession):
    patient = await _seed_patient(db)
    rx = await _seed_prescription(db, patient)
    pharmacy, item = await _seed_pharmacy(db, stock=5)

    with pytest.raises(BusinessRuleValidationException) as exc:
        await ordering_service.place_order(
            db,
            patient_id=patient.id,
            prescription_id=rx.id,
            pharmacy_id=pharmacy.id,
            selections=[{"inventory_id": str(item.id), "quantity": 50}],
            delivery_address="221B Baker Street",
        )
    assert "only 5 left" in str(exc.value)
    assert item.stock_quantity == 5  # untouched


@pytest.mark.asyncio
async def test_cancelling_returns_the_reserved_stock(db: AsyncSession):
    """
    The half that is easy to forget: without it, every cancellation permanently
    removes stock the pharmacy still physically holds.
    """
    patient = await _seed_patient(db)
    rx = await _seed_prescription(db, patient)
    pharmacy, item = await _seed_pharmacy(db, stock=100)

    order = await ordering_service.place_order(
        db,
        patient_id=patient.id,
        prescription_id=rx.id,
        pharmacy_id=pharmacy.id,
        selections=[{"inventory_id": str(item.id), "quantity": 30}],
        delivery_address="221B Baker Street",
    )
    await db.flush()
    assert item.stock_quantity == 70

    await ordering_service.cancel_order(
        db, order, reason="Changed my mind", actor_id=patient.id
    )
    await db.flush()

    assert order.status == "cancelled"
    assert order.cancellation_reason == "Changed my mind"
    assert item.stock_quantity == 100


@pytest.mark.asyncio
async def test_full_delivery_lifecycle_records_every_stage(db: AsyncSession):
    patient = await _seed_patient(db)
    rx = await _seed_prescription(db, patient)
    pharmacy, item = await _seed_pharmacy(db)

    order = await ordering_service.place_order(
        db,
        patient_id=patient.id,
        prescription_id=rx.id,
        pharmacy_id=pharmacy.id,
        selections=[{"inventory_id": str(item.id), "quantity": 10}],
        delivery_address="221B Baker Street",
    )
    await db.flush()

    for stage in (ORDER_PREPARING, ORDER_PACKED, ORDER_OUT_FOR_DELIVERY, ORDER_DELIVERED):
        await ordering_service.advance_status(db, order, stage)
        await db.flush()

    assert order.status == ORDER_DELIVERED
    assert order.dispatched_at is not None
    assert order.delivered_at is not None

    fresh = await ordering_service.get_for_patient(db, order.id, patient.id)
    recorded = [event.status for event in fresh.events]
    assert recorded == ["received", "preparing", "packed", "out_for_delivery", "delivered"]


@pytest.mark.asyncio
async def test_cancellation_after_dispatch_is_refused(db: AsyncSession):
    patient = await _seed_patient(db)
    rx = await _seed_prescription(db, patient)
    pharmacy, item = await _seed_pharmacy(db)

    order = await ordering_service.place_order(
        db,
        patient_id=patient.id,
        prescription_id=rx.id,
        pharmacy_id=pharmacy.id,
        selections=[{"inventory_id": str(item.id), "quantity": 5}],
        delivery_address="221B Baker Street",
    )
    await db.flush()

    for stage in (ORDER_PREPARING, ORDER_PACKED, ORDER_OUT_FOR_DELIVERY):
        await ordering_service.advance_status(db, order, stage)
    await db.flush()

    with pytest.raises(BusinessRuleValidationException):
        await ordering_service.cancel_order(db, order, reason="too late")


@pytest.mark.asyncio
async def test_skipping_a_lifecycle_stage_is_refused(db: AsyncSession):
    patient = await _seed_patient(db)
    rx = await _seed_prescription(db, patient)
    pharmacy, item = await _seed_pharmacy(db)

    order = await ordering_service.place_order(
        db,
        patient_id=patient.id,
        prescription_id=rx.id,
        pharmacy_id=pharmacy.id,
        selections=[{"inventory_id": str(item.id), "quantity": 5}],
        delivery_address="221B Baker Street",
    )
    await db.flush()

    with pytest.raises(BusinessRuleValidationException) as exc:
        await ordering_service.advance_status(db, order, ORDER_DELIVERED)
    assert "cannot become" in str(exc.value)


@pytest.mark.asyncio
async def test_ordering_against_another_patients_prescription_is_refused(db: AsyncSession):
    """Would both dispense to the wrong person and leak what they were prescribed."""
    owner = await _seed_patient(db)
    intruder = await _seed_patient(db)
    rx = await _seed_prescription(db, owner)
    pharmacy, item = await _seed_pharmacy(db)

    with pytest.raises(AuthorizationException):
        await ordering_service.place_order(
            db,
            patient_id=intruder.id,
            prescription_id=rx.id,
            pharmacy_id=pharmacy.id,
            selections=[{"inventory_id": str(item.id), "quantity": 1}],
            delivery_address="Elsewhere",
        )


@pytest.mark.asyncio
async def test_reading_another_patients_order_is_refused(db: AsyncSession):
    owner = await _seed_patient(db)
    intruder = await _seed_patient(db)
    rx = await _seed_prescription(db, owner)
    pharmacy, item = await _seed_pharmacy(db)

    order = await ordering_service.place_order(
        db,
        patient_id=owner.id,
        prescription_id=rx.id,
        pharmacy_id=pharmacy.id,
        selections=[{"inventory_id": str(item.id), "quantity": 1}],
        delivery_address="221B Baker Street",
    )
    await db.flush()

    with pytest.raises(AuthorizationException):
        await ordering_service.get_for_patient(db, order.id, intruder.id)
