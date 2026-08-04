"""
Delivery & Logistics.

The properties pinned here are the ones that would let a rider mark medicine
delivered without handing it over, see another rider's work, or leave the
order and the assignment telling two different stories.
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
from app.core.security import verify_password
from app.models.case import Case
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
    DELIVERY_STATUSES,
    DELIVERY_TRANSITIONS,
    DeliveryPartner,
    PARTNER_APPROVED,
    PARTNER_DOCUMENT_REVIEW,
    PARTNER_PENDING,
    PARTNER_STATUSES,
    PARTNER_SUSPENDED,
    PARTNER_TRANSITIONS,
    TERMINAL_STATUSES,
)
from app.models.doctor import Doctor
from app.models.medicine_order import (
    ORDER_DELIVERED,
    ORDER_OUT_FOR_DELIVERY,
    ORDER_PACKED,
    ORDER_PREPARING,
)
from app.models.patient import Patient
from app.models.pharmacy import Pharmacy, PharmacyInventory, VERIFICATION_APPROVED
from app.models.prescription import Medication, Prescription
from app.models.user import User
from app.pharmacy.application.ordering import ordering_service
from app.services.delivery import (
    MAX_OTP_ATTEMPTS,
    OTP_LENGTH,
    delivery_service as svc,
    generate_otp,
)

LAT, LNG = 28.6315, 77.2167


# ── fixtures ─────────────────────────────────────────────────────────────


async def _user(db: AsyncSession, role: str) -> User:
    user = User(
        id=uuid.uuid4(), email=f"{role}-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password="x", role=role, is_active=True,
    )
    db.add(user)
    await db.flush()
    return user


async def _partner(db: AsyncSession, *, approved: bool = True, online: bool = True):
    user = await _user(db, "delivery_partner")
    partner = DeliveryPartner(
        user_id=user.id,
        full_name=f"Rider {uuid.uuid4().hex[:4]}",
        phone="+919000000000",
        vehicle_type="motorcycle",
        vehicle_number=f"DL{uuid.uuid4().hex[:6].upper()}",
        verification_status=PARTNER_APPROVED if approved else PARTNER_PENDING,
        is_online=online,
    )
    db.add(partner)
    await db.flush()
    return partner


async def _packed_order(db: AsyncSession):
    """A full chain: patient → prescription → pharmacy → packed order."""
    patient_user = await _user(db, "patient")
    patient = Patient(
        id=patient_user.id, first_name="P", last_name="Q", phone="+910000000000",
        date_of_birth="1990-01-01", gender="male",
    )
    db.add(patient)

    doctor_user = await _user(db, "doctor")
    doctor = Doctor(
        id=doctor_user.id, first_name="D", last_name="R", phone="+910000000001",
        specialty="General", license_number=f"LIC-{uuid.uuid4().hex[:6]}",
    )
    db.add(doctor)
    await db.flush()

    case = Case(
        patient_id=patient.id, patient_name="P Q", patient_age=30,
        patient_gender="male", doctor_id=doctor.id, doctor_name="Dr R",
        specialty="General", symptom_summary="x", status="prescribed",
    )
    db.add(case)
    await db.flush()

    rx = Prescription(
        case_id=case.id, patient_id=patient.id, patient_name="P Q",
        doctor_id=doctor.id, doctor_name="Dr R", diagnosis="x", status="active",
    )
    db.add(rx)
    await db.flush()
    db.add(
        Medication(
            prescription_id=rx.id, name="Metformin", rxcui="6809", dosage="1",
            frequency="bd", duration="30 days", quantity=30,
            start_date="2026-08-03", end_date="2026-09-02",
        )
    )

    pharmacy = Pharmacy(
        name=f"Chem-{uuid.uuid4().hex[:5]}", address="1 CP", latitude=LAT, longitude=LNG,
        is_partner=True, is_active=True, is_24x7=True, delivers=True,
        delivery_fee=40.0, verification_status=VERIFICATION_APPROVED,
    )
    db.add(pharmacy)
    await db.flush()

    item = PharmacyInventory(
        pharmacy_id=pharmacy.id, sku=f"S-{uuid.uuid4().hex[:5]}", rxcui="6809",
        medicine_name="Metformin 500mg", mrp=100.0, selling_price=80.0,
        stock_quantity=100, low_stock_threshold=10,
    )
    db.add(item)
    await db.flush()

    order = await ordering_service.place_order(
        db, patient_id=patient.id, prescription_id=rx.id, pharmacy_id=pharmacy.id,
        selections=[{"inventory_id": str(item.id), "quantity": 10}],
        delivery_address="221B Baker Street", eta_minutes=25,
    )
    await db.flush()

    for stage in (ORDER_PREPARING, ORDER_PACKED):
        await ordering_service.advance_status(db, order, stage)
    await db.flush()
    return order, pharmacy, patient


# ── state machines ───────────────────────────────────────────────────────


def test_every_delivery_status_is_in_the_transition_table():
    assert set(DELIVERY_TRANSITIONS) == set(DELIVERY_STATUSES)


def test_the_full_journey_is_reachable():
    path = [
        DELIVERY_OFFERED, DELIVERY_ACCEPTED, DELIVERY_EN_ROUTE_PICKUP,
        DELIVERY_AT_PHARMACY, DELIVERY_PICKED_UP, DELIVERY_OUT_FOR_DELIVERY,
        DELIVERY_AT_PATIENT, DELIVERY_DELIVERED,
    ]
    for current, nxt in zip(path, path[1:]):
        assert nxt in DELIVERY_TRANSITIONS[current], f"{current} -> {nxt} blocked"


def test_terminal_statuses_have_no_exits():
    for status in TERMINAL_STATUSES:
        assert DELIVERY_TRANSITIONS[status] == ()


def test_cancellation_is_impossible_once_medicine_is_carried():
    """
    After pickup the goods have left the pharmacy's control, so the only exits
    are completing or failing — not a quiet cancel.
    """
    for status in (DELIVERY_PICKED_UP, DELIVERY_OUT_FOR_DELIVERY, DELIVERY_AT_PATIENT):
        assert DELIVERY_CANCELLED not in DELIVERY_TRANSITIONS[status]
        assert DELIVERY_FAILED in DELIVERY_TRANSITIONS[status]


def test_active_statuses_exclude_the_terminal_ones():
    assert set(ACTIVE_STATUSES).isdisjoint(TERMINAL_STATUSES)


def test_partner_verification_cannot_skip_document_review():
    assert PARTNER_APPROVED not in PARTNER_TRANSITIONS[PARTNER_PENDING]
    assert PARTNER_APPROVED in PARTNER_TRANSITIONS[PARTNER_DOCUMENT_REVIEW]
    assert set(PARTNER_TRANSITIONS) == set(PARTNER_STATUSES)


# ── assignment guards ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_only_a_packed_order_can_be_assigned(db: AsyncSession):
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)

    # Move it on, then it is no longer assignable.
    assignment = await svc.create_assignment(
        db, order_id=order.id, partner_id=partner.id
    )
    await db.flush()
    assert assignment.status == DELIVERY_OFFERED

    other = await _partner(db)
    with pytest.raises(BusinessRuleValidationException) as exc:
        await svc.create_assignment(db, order_id=order.id, partner_id=other.id)
    assert "already has an active delivery" in str(exc.value)


@pytest.mark.asyncio
async def test_unapproved_partner_cannot_be_assigned(db: AsyncSession):
    order, _, _ = await _packed_order(db)
    partner = await _partner(db, approved=False)

    with pytest.raises(BusinessRuleValidationException) as exc:
        await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    assert "cannot take deliveries" in str(exc.value)


@pytest.mark.asyncio
async def test_busy_partner_cannot_take_a_second_order(db: AsyncSession):
    order_a, _, _ = await _packed_order(db)
    order_b, _, _ = await _packed_order(db)
    partner = await _partner(db)

    await svc.create_assignment(db, order_id=order_a.id, partner_id=partner.id)
    await db.flush()

    with pytest.raises(BusinessRuleValidationException) as exc:
        await svc.create_assignment(db, order_id=order_b.id, partner_id=partner.id)
    assert "already carrying" in str(exc.value)


@pytest.mark.asyncio
async def test_an_unpacked_order_is_refused(db: AsyncSession):
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    order.status = ORDER_PREPARING
    await db.flush()

    with pytest.raises(BusinessRuleValidationException) as exc:
        await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    assert "Only a packed order" in str(exc.value)


# ── rider isolation ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_rider_cannot_touch_another_riders_delivery(db: AsyncSession):
    """The single most important property of the rider API."""
    order, _, _ = await _packed_order(db)
    mine = await _partner(db)
    theirs = await _partner(db)

    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=mine.id)
    await db.flush()

    with pytest.raises(AuthorizationException):
        await svc._owned(db, assignment.id, theirs)
    with pytest.raises(AuthorizationException):
        await svc.advance(db, theirs, assignment.id, target=DELIVERY_ACCEPTED)


@pytest.mark.asyncio
async def test_listing_is_scoped_to_the_rider(db: AsyncSession):
    order, _, _ = await _packed_order(db)
    mine = await _partner(db)
    theirs = await _partner(db)
    await svc.create_assignment(db, order_id=order.id, partner_id=mine.id)
    await db.flush()

    ours, _ = await svc.list_for_partner(db, mine)
    others, total = await svc.list_for_partner(db, theirs)
    assert ours
    assert all(a.partner_id == mine.id for a in ours)
    assert others == [] and total == 0


# ── the journey, end to end ──────────────────────────────────────────────


async def _to_at_patient(db, partner, assignment):
    for target in (
        DELIVERY_ACCEPTED, DELIVERY_EN_ROUTE_PICKUP, DELIVERY_AT_PHARMACY,
        DELIVERY_PICKED_UP, DELIVERY_OUT_FOR_DELIVERY, DELIVERY_AT_PATIENT,
    ):
        await svc.advance(db, partner, assignment.id, target=target)
    await db.flush()


@pytest.mark.asyncio
async def test_pickup_moves_the_parent_order_out_for_delivery(db: AsyncSession):
    """
    The assignment and the order must never tell two different stories — the
    order follows through `ordering_service`, which owns its transitions.
    """
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()

    for target in (
        DELIVERY_ACCEPTED, DELIVERY_EN_ROUTE_PICKUP, DELIVERY_AT_PHARMACY, DELIVERY_PICKED_UP
    ):
        await svc.advance(db, partner, assignment.id, target=target)
    await db.flush()

    assert order.status == ORDER_OUT_FOR_DELIVERY
    assert order.delivery_partner_name == partner.full_name
    assert order.delivery_partner_phone == partner.phone


@pytest.mark.asyncio
async def test_stage_skipping_is_refused(db: AsyncSession):
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()

    with pytest.raises(BusinessRuleValidationException) as exc:
        await svc.advance(db, partner, assignment.id, target=DELIVERY_AT_PATIENT)
    assert "cannot become" in str(exc.value)


@pytest.mark.asyncio
async def test_a_rider_cannot_mark_delivered_directly(db: AsyncSession):
    """
    The whole point of the OTP. If `advance` accepted `delivered`, a rider could
    complete a job without ever meeting the patient.
    """
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()
    await _to_at_patient(db, partner, assignment)

    with pytest.raises(BusinessRuleValidationException) as exc:
        await svc.advance(db, partner, assignment.id, target=DELIVERY_DELIVERED)
    assert "verifying the patient's OTP" in str(exc.value)
    assert order.status != ORDER_DELIVERED


# ── OTP ──────────────────────────────────────────────────────────────────


def test_otp_is_numeric_and_unpredictable():
    codes = {generate_otp() for _ in range(300)}
    assert len(codes) > 250  # collisions are possible but must be rare
    assert all(len(c) == OTP_LENGTH and c.isdigit() for c in codes)


@pytest.mark.asyncio
async def test_otp_is_issued_on_dispatch_and_only_stored_hashed(db: AsyncSession):
    """
    Issued when the rider sets off, so the patient holds it before arrival —
    and stored as a hash, or a rider could read it from the API.
    """
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()

    for target in (
        DELIVERY_ACCEPTED, DELIVERY_EN_ROUTE_PICKUP, DELIVERY_AT_PHARMACY,
        DELIVERY_PICKED_UP, DELIVERY_OUT_FOR_DELIVERY,
    ):
        await svc.advance(db, partner, assignment.id, target=target)
    await db.flush()

    assert assignment.otp_hash is not None
    assert assignment.otp_issued_at is not None
    assert len(assignment.otp_hash) > OTP_LENGTH  # a hash, not the code


@pytest.mark.asyncio
async def test_correct_otp_completes_the_delivery_and_the_order(db: AsyncSession):
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()
    await _to_at_patient(db, partner, assignment)

    code = await svc.issue_otp(db, assignment)
    await db.flush()

    await svc.verify_otp(db, partner, assignment.id, code=code)
    await db.flush()

    assert assignment.status == DELIVERY_DELIVERED
    assert assignment.is_otp_verified
    assert order.status == ORDER_DELIVERED
    assert partner.completed_deliveries == 1
    assert partner.total_earnings > 0


@pytest.mark.asyncio
async def test_wrong_otp_is_refused_and_counted(db: AsyncSession):
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()
    await _to_at_patient(db, partner, assignment)
    await svc.issue_otp(db, assignment)
    await db.flush()

    with pytest.raises(BusinessRuleValidationException) as exc:
        await svc.verify_otp(db, partner, assignment.id, code="000000")
    assert "not correct" in str(exc.value)
    assert assignment.otp_attempts == 1
    assert assignment.status == DELIVERY_AT_PATIENT
    assert order.status != ORDER_DELIVERED


@pytest.mark.asyncio
async def test_otp_attempts_are_bounded(db: AsyncSession):
    """A six-digit code must not be brute-forceable at the doorstep."""
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()
    await _to_at_patient(db, partner, assignment)
    code = await svc.issue_otp(db, assignment)
    await db.flush()

    for _ in range(MAX_OTP_ATTEMPTS):
        with pytest.raises(BusinessRuleValidationException):
            await svc.verify_otp(db, partner, assignment.id, code="111111")

    # Even the right code is now refused until it is reissued.
    with pytest.raises(BusinessRuleValidationException) as exc:
        await svc.verify_otp(db, partner, assignment.id, code=code)
    assert "Too many incorrect attempts" in str(exc.value)


@pytest.mark.asyncio
async def test_expired_otp_is_refused(db: AsyncSession):
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()
    await _to_at_patient(db, partner, assignment)
    code = await svc.issue_otp(db, assignment)
    assignment.otp_issued_at = datetime.now(timezone.utc) - timedelta(hours=2)
    await db.flush()

    with pytest.raises(BusinessRuleValidationException) as exc:
        await svc.verify_otp(db, partner, assignment.id, code=code)
    assert "expired" in str(exc.value)


@pytest.mark.asyncio
async def test_otp_cannot_be_verified_before_reaching_the_patient(db: AsyncSession):
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()
    await svc.advance(db, partner, assignment.id, target=DELIVERY_ACCEPTED)
    code = await svc.issue_otp(db, assignment)
    await db.flush()

    with pytest.raises(BusinessRuleValidationException) as exc:
        await svc.verify_otp(db, partner, assignment.id, code=code)
    assert "at the patient's address" in str(exc.value)


@pytest.mark.asyncio
async def test_reissuing_resets_the_attempt_counter(db: AsyncSession):
    """A patient who mistyped must not be locked out of their own delivery."""
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()
    await _to_at_patient(db, partner, assignment)
    await svc.issue_otp(db, assignment)
    await db.flush()

    with pytest.raises(BusinessRuleValidationException):
        await svc.verify_otp(db, partner, assignment.id, code="999999")
    assert assignment.otp_attempts == 1

    fresh = await svc.issue_otp(db, assignment)
    await db.flush()
    assert assignment.otp_attempts == 0
    assert verify_password(fresh, assignment.otp_hash)


# ── proof of delivery ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_proof_captures_evidence_and_position(db: AsyncSession):
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()
    await _to_at_patient(db, partner, assignment)

    await svc.capture_proof(
        db, partner, assignment.id,
        photo_url="/uploads/proof.jpg", signature_url="/uploads/sig.png",
        notes="Handed to patient", latitude=LAT, longitude=LNG,
    )
    await db.flush()

    assert assignment.proof_photo_url == "/uploads/proof.jpg"
    assert assignment.proof_signature_url == "/uploads/sig.png"
    assert assignment.delivery_notes == "Handed to patient"
    assert assignment.proof_latitude == LAT
    assert assignment.proof_captured_at is not None


@pytest.mark.asyncio
async def test_proof_is_refused_before_arrival(db: AsyncSession):
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()
    await svc.advance(db, partner, assignment.id, target=DELIVERY_ACCEPTED)
    await db.flush()

    with pytest.raises(BusinessRuleValidationException):
        await svc.capture_proof(db, partner, assignment.id, photo_url="/x.jpg")


# ── failure ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_failing_leaves_the_order_for_the_pharmacy_to_decide(db: AsyncSession):
    """
    The medicine has left the counter. Whether that becomes a return, a
    re-dispatch or a refund is not the rider's call.
    """
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()
    await _to_at_patient(db, partner, assignment)
    before = order.status

    await svc.fail(db, partner, assignment.id, reason="Nobody at the address")
    await db.flush()

    assert assignment.status == DELIVERY_FAILED
    assert assignment.failure_reason == "Nobody at the address"
    assert partner.failed_deliveries == 1
    assert order.status == before


# ── tracking ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tracking_exposes_only_what_the_patient_needs(db: AsyncSession):
    order, _, patient = await _packed_order(db)
    partner = await _partner(db)
    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()
    await svc.update_location(db, partner, latitude=LAT, longitude=LNG)
    await db.flush()

    data = await svc.tracking_for_order(db, order.id, patient.id)

    assert data["partner_name"] == partner.full_name
    assert data["vehicle_number"] == partner.vehicle_number
    assert data["current_latitude"] == LAT
    # Paired with the fix so a stale position reads as stale.
    assert data["location_updated_at"] is not None
    # Nothing private about the rider leaks.
    for leaked in ("driving_licence_number", "total_earnings", "address", "otp_hash"):
        assert leaked not in data


@pytest.mark.asyncio
async def test_another_patient_cannot_track_this_order(db: AsyncSession):
    order, _, _ = await _packed_order(db)
    intruder = await _user(db, "patient")

    with pytest.raises(AuthorizationException):
        await svc.tracking_for_order(db, order.id, intruder.id)


@pytest.mark.asyncio
async def test_tracking_is_null_before_a_rider_is_assigned(db: AsyncSession):
    """A normal state, not an error."""
    order, _, patient = await _packed_order(db)
    assert await svc.tracking_for_order(db, order.id, patient.id) is None


# ── fleet ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_available_partners_excludes_busy_and_offline(db: AsyncSession):
    order, _, _ = await _packed_order(db)
    free = await _partner(db)
    busy = await _partner(db)
    offline = await _partner(db, online=False)
    unapproved = await _partner(db, approved=False)

    await svc.create_assignment(db, order_id=order.id, partner_id=busy.id)
    await db.flush()

    available = {p.id for p in await svc.available_partners(db)}
    assert free.id in available
    assert busy.id not in available
    assert offline.id not in available
    assert unapproved.id not in available


@pytest.mark.asyncio
async def test_suspension_takes_a_rider_off_the_road(db: AsyncSession):
    admin = await _user(db, "admin")
    partner = await _partner(db)

    await svc.transition_partner(
        db, partner.id, to_status=PARTNER_SUSPENDED, note="Complaint", actor=admin
    )
    await db.flush()

    assert partner.verification_status == PARTNER_SUSPENDED
    assert partner.is_online is False
    assert partner.can_accept_work is False


@pytest.mark.asyncio
async def test_illegal_partner_transition_is_refused(db: AsyncSession):
    admin = await _user(db, "admin")
    partner = await _partner(db, approved=False)

    with pytest.raises(BusinessRuleValidationException):
        await svc.transition_partner(
            db, partner.id, to_status=PARTNER_APPROVED, note="", actor=admin
        )


@pytest.mark.asyncio
async def test_going_offline_does_not_drop_accepted_work(db: AsyncSession):
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    assignment = await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()
    await svc.advance(db, partner, assignment.id, target=DELIVERY_ACCEPTED)

    await svc.set_online(db, partner, online=False)
    await db.flush()

    assert partner.is_online is False
    assert assignment.status == DELIVERY_ACCEPTED
    assert assignment.is_active


@pytest.mark.asyncio
async def test_dashboard_and_analytics_shapes(db: AsyncSession):
    order, _, _ = await _packed_order(db)
    partner = await _partner(db)
    await svc.create_assignment(db, order_id=order.id, partner_id=partner.id)
    await db.flush()

    board = await svc.dashboard(db, partner)
    for key in (
        "deliveries_today", "earnings_today", "distance_today_km",
        "average_delivery_minutes", "completion_rate", "rating",
    ):
        assert key in board

    fleet = await svc.network_analytics(db, days=30)
    for key in (
        "assignments", "delivered", "failed", "success_rate",
        "total_distance_km", "average_eta_minutes", "top_partners",
    ):
        assert key in fleet
    assert 0.0 <= fleet["success_rate"] <= 1.0


def test_completion_rate_handles_a_new_rider():
    partner = DeliveryPartner(
        user_id=uuid.uuid4(), full_name="New", phone="+91",
        completed_deliveries=0, failed_deliveries=0,
    )
    assert partner.completion_rate == 0.0
    partner.completed_deliveries = 3
    partner.failed_deliveries = 1
    assert partner.completion_rate == 0.75
