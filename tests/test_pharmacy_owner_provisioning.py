"""
Pharmacy owner provisioning.

The properties pinned here are the ones that would hand portal access to the
wrong person, leave a store with two operators or none, or strand orders nobody
can dispatch.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    BusinessRuleValidationException,
    EntityNotFoundException,
)
from app.core.security import verify_password
from app.models.medicine_order import ORDER_RECEIVED
from app.models.pharmacy import (
    Pharmacy,
    VERIFICATION_APPROVED,
    VERIFICATION_PENDING,
)
from app.models.user import User
from app.services.pharmacy_owner import (
    TEMP_PASSWORD_LENGTH,
    generate_temporary_password,
    pharmacy_owner_service as owners,
)


async def _admin(db: AsyncSession) -> User:
    user = User(
        id=uuid.uuid4(), email=f"admin-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password="x", role="admin", is_active=True,
    )
    db.add(user)
    await db.flush()
    return user


async def _pharmacy(db: AsyncSession, *, approved: bool = True, active: bool = True) -> Pharmacy:
    pharmacy = Pharmacy(
        name=f"Store-{uuid.uuid4().hex[:6]}",
        address="1 Connaught Place",
        latitude=28.6315, longitude=77.2167,
        is_partner=approved, is_active=active, is_24x7=True, delivers=True,
        verification_status=VERIFICATION_APPROVED if approved else VERIFICATION_PENDING,
    )
    db.add(pharmacy)
    await db.flush()
    return pharmacy


async def _user(db: AsyncSession, *, role: str = "pharmacy") -> User:
    user = User(
        id=uuid.uuid4(), email=f"u-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password="x", role=role, is_active=True,
    )
    db.add(user)
    await db.flush()
    return user


# ── credential generation ────────────────────────────────────────────────


def test_temporary_password_is_long_and_unpredictable():
    passwords = {generate_temporary_password() for _ in range(200)}
    assert len(passwords) == 200
    assert all(len(p) == TEMP_PASSWORD_LENGTH for p in passwords)


def test_temporary_password_omits_ambiguous_characters():
    """
    A credential is dictated over the phone or copied off a screen. O/0 and
    l/1/I are the characters that get transcribed wrongly.
    """
    combined = "".join(generate_temporary_password() for _ in range(50))
    for ambiguous in ("O", "0", "l", "I", "1"):
        assert ambiguous not in combined


# ── creating an owner ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_owner_links_role_and_pharmacy(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await _pharmacy(db)

    owner, temporary = await owners.create_owner(
        db, pharmacy.id, email="Owner@Example.com", actor=admin
    )
    await db.flush()

    assert owner.role == "pharmacy"
    assert owner.pharmacy_id == pharmacy.id
    assert owner.is_active is True
    # Email normalised, so a duplicate cannot slip in on casing alone.
    assert owner.email == "owner@example.com"


@pytest.mark.asyncio
async def test_created_password_is_hashed_not_stored_plaintext(db: AsyncSession):
    """The plaintext exists only in the return value; the row holds a hash."""
    admin = await _admin(db)
    pharmacy = await _pharmacy(db)

    owner, temporary = await owners.create_owner(
        db, pharmacy.id, email=f"o-{uuid.uuid4().hex[:6]}@example.com", actor=admin
    )
    await db.flush()

    assert owner.hashed_password != temporary
    assert temporary not in owner.hashed_password
    # And it is the real credential — the existing verifier accepts it.
    assert verify_password(temporary, owner.hashed_password) is True


@pytest.mark.asyncio
async def test_duplicate_email_is_refused(db: AsyncSession):
    admin = await _admin(db)
    pharmacy_a, pharmacy_b = await _pharmacy(db), await _pharmacy(db)
    email = f"dup-{uuid.uuid4().hex[:6]}@example.com"

    await owners.create_owner(db, pharmacy_a.id, email=email, actor=admin)
    await db.flush()

    with pytest.raises(BusinessRuleValidationException) as exc:
        await owners.create_owner(db, pharmacy_b.id, email=email, actor=admin)
    assert "already exists" in str(exc.value)


# ── the staffability gate ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unverified_pharmacy_cannot_be_staffed(db: AsyncSession):
    """
    Access before approval would let a store dispense before anyone checked its
    drug licence.
    """
    admin = await _admin(db)
    pharmacy = await _pharmacy(db, approved=False)

    with pytest.raises(BusinessRuleValidationException) as exc:
        await owners.create_owner(
            db, pharmacy.id, email=f"x-{uuid.uuid4().hex[:6]}@example.com", actor=admin
        )
    assert "not been approved" in str(exc.value)


@pytest.mark.asyncio
async def test_suspended_pharmacy_cannot_be_staffed(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await _pharmacy(db, active=False)

    with pytest.raises(BusinessRuleValidationException) as exc:
        await owners.create_owner(
            db, pharmacy.id, email=f"y-{uuid.uuid4().hex[:6]}@example.com", actor=admin
        )
    assert "suspended" in str(exc.value)


@pytest.mark.asyncio
async def test_assignment_also_respects_the_gate(db: AsyncSession):
    """The rule lives in the service, so every path is covered — not just create."""
    admin = await _admin(db)
    pharmacy = await _pharmacy(db, approved=False)
    candidate = await _user(db)

    with pytest.raises(BusinessRuleValidationException):
        await owners.assign_existing_user(db, pharmacy.id, candidate.id, actor=admin)


# ── one active owner per pharmacy ────────────────────────────────────────


@pytest.mark.asyncio
async def test_second_active_owner_is_refused(db: AsyncSession):
    """Two people holding one store's inventory is an accountability hole."""
    admin = await _admin(db)
    pharmacy = await _pharmacy(db)
    await owners.create_owner(
        db, pharmacy.id, email=f"first-{uuid.uuid4().hex[:6]}@example.com", actor=admin
    )
    await db.flush()

    second = await _user(db)
    with pytest.raises(BusinessRuleValidationException) as exc:
        await owners.assign_existing_user(db, pharmacy.id, second.id, actor=admin)
    assert "already has an active owner" in str(exc.value)


@pytest.mark.asyncio
async def test_current_owner_returns_only_the_active_one(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await _pharmacy(db)
    owner, _ = await owners.create_owner(
        db, pharmacy.id, email=f"a-{uuid.uuid4().hex[:6]}@example.com", actor=admin
    )
    await db.flush()

    assert (await owners.current_owner(db, pharmacy.id)).id == owner.id

    await owners.set_owner_active(db, owner.id, active=False, actor=admin)
    await db.flush()
    assert await owners.current_owner(db, pharmacy.id) is None


# ── role protection ──────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["patient", "doctor", "admin"])
async def test_other_roles_cannot_be_converted(db: AsyncSession, role: str):
    """
    Silently re-roling a doctor would orphan their cases, and an admin their
    audit trail. Never what an operator meant by "link this account".
    """
    admin = await _admin(db)
    pharmacy = await _pharmacy(db)
    candidate = await _user(db, role=role)

    with pytest.raises(BusinessRuleValidationException) as exc:
        await owners.assign_existing_user(db, pharmacy.id, candidate.id, actor=admin)
    assert "cannot be converted" in str(exc.value)
    # And the account is untouched.
    assert candidate.role == role


@pytest.mark.asyncio
async def test_user_already_running_another_store_is_refused(db: AsyncSession):
    admin = await _admin(db)
    store_a, store_b = await _pharmacy(db), await _pharmacy(db)
    owner, _ = await owners.create_owner(
        db, store_a.id, email=f"b-{uuid.uuid4().hex[:6]}@example.com", actor=admin
    )
    await db.flush()

    with pytest.raises(BusinessRuleValidationException) as exc:
        await owners.assign_existing_user(db, store_b.id, owner.id, actor=admin)
    assert "another pharmacy" in str(exc.value)


# ── change & remove ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_change_owner_swaps_in_one_transaction(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await _pharmacy(db)
    outgoing, _ = await owners.create_owner(
        db, pharmacy.id, email=f"out-{uuid.uuid4().hex[:6]}@example.com", actor=admin
    )
    await db.flush()
    incoming = await _user(db)

    result = await owners.change_owner(db, pharmacy.id, incoming.id, actor=admin)
    await db.flush()

    assert result.id == incoming.id
    assert incoming.pharmacy_id == pharmacy.id
    assert incoming.is_active is True
    # The outgoing owner loses access immediately.
    assert outgoing.pharmacy_id is None
    assert outgoing.is_active is False
    # Exactly one active owner remains.
    assert (await owners.current_owner(db, pharmacy.id)).id == incoming.id


@pytest.mark.asyncio
async def test_change_to_the_same_owner_is_refused(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await _pharmacy(db)
    owner, _ = await owners.create_owner(
        db, pharmacy.id, email=f"same-{uuid.uuid4().hex[:6]}@example.com", actor=admin
    )
    await db.flush()

    with pytest.raises(BusinessRuleValidationException):
        await owners.change_owner(db, pharmacy.id, owner.id, actor=admin)


@pytest.mark.asyncio
async def test_removal_revokes_access_without_deleting_the_account(db: AsyncSession):
    """
    The row is referenced by audit entries and order events it recorded, so it
    is unlinked and deactivated rather than deleted.
    """
    admin = await _admin(db)
    pharmacy = await _pharmacy(db)
    owner, _ = await owners.create_owner(
        db, pharmacy.id, email=f"rm-{uuid.uuid4().hex[:6]}@example.com", actor=admin
    )
    await db.flush()
    owner_id = owner.id

    await owners.remove_owner(db, pharmacy.id, actor=admin, reason="Left the business")
    await db.flush()

    still_there = await db.get(User, owner_id)
    assert still_there is not None
    assert still_there.pharmacy_id is None
    assert still_there.is_active is False


@pytest.mark.asyncio
async def test_removal_is_refused_while_orders_are_in_flight(db: AsyncSession):
    """
    Removing the operator mid-delivery would leave nobody able to dispatch
    medicine a patient is waiting on.
    """
    from app.models.medicine_order import MedicineOrder
    from app.models.prescription import Prescription
    from app.models.case import Case
    from app.models.doctor import Doctor
    from app.models.patient import Patient

    admin = await _admin(db)
    pharmacy = await _pharmacy(db)
    owner, _ = await owners.create_owner(
        db, pharmacy.id, email=f"busy-{uuid.uuid4().hex[:6]}@example.com", actor=admin
    )
    await db.flush()

    patient_user = await _user(db, role="patient")
    patient = Patient(
        id=patient_user.id, first_name="P", last_name="Q", phone="+910000000000",
        date_of_birth="1990-01-01", gender="male",
    )
    db.add(patient)
    doctor_user = await _user(db, role="doctor")
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
        MedicineOrder(
            order_number=f"MB-{uuid.uuid4().hex[:8].upper()}",
            patient_id=patient.id, prescription_id=rx.id, pharmacy_id=pharmacy.id,
            pharmacy_name=pharmacy.name, status=ORDER_RECEIVED, total=100.0,
            delivery_address="somewhere",
        )
    )
    await db.flush()

    with pytest.raises(BusinessRuleValidationException) as exc:
        await owners.remove_owner(db, pharmacy.id, actor=admin)
    assert "in progress" in str(exc.value)
    # Access is untouched by the refusal.
    assert owner.pharmacy_id == pharmacy.id


@pytest.mark.asyncio
async def test_removing_when_there_is_no_owner_is_refused(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await _pharmacy(db)
    with pytest.raises(BusinessRuleValidationException):
        await owners.remove_owner(db, pharmacy.id, actor=admin)


# ── suspend & reactivate ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_suspend_then_reactivate_keeps_the_link(db: AsyncSession):
    """Distinct from removal: the store link survives a suspension."""
    admin = await _admin(db)
    pharmacy = await _pharmacy(db)
    owner, _ = await owners.create_owner(
        db, pharmacy.id, email=f"s-{uuid.uuid4().hex[:6]}@example.com", actor=admin
    )
    await db.flush()

    await owners.set_owner_active(db, owner.id, active=False, actor=admin, reason="Leave")
    await db.flush()
    assert owner.is_active is False
    assert owner.pharmacy_id == pharmacy.id

    await owners.set_owner_active(db, owner.id, active=True, actor=admin)
    await db.flush()
    assert owner.is_active is True


@pytest.mark.asyncio
async def test_reactivation_cannot_create_a_second_active_owner(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await _pharmacy(db)
    first, _ = await owners.create_owner(
        db, pharmacy.id, email=f"f-{uuid.uuid4().hex[:6]}@example.com", actor=admin
    )
    await db.flush()

    await owners.set_owner_active(db, first.id, active=False, actor=admin)
    await db.flush()

    replacement = await _user(db)
    await owners.assign_existing_user(db, pharmacy.id, replacement.id, actor=admin)
    await db.flush()

    # Re-enabling the original would now give the store two operators.
    with pytest.raises(BusinessRuleValidationException) as exc:
        await owners.set_owner_active(db, first.id, active=True, actor=admin)
    assert "already has an active owner" in str(exc.value)


@pytest.mark.asyncio
async def test_status_change_on_a_non_owner_is_refused(db: AsyncSession):
    admin = await _admin(db)
    patient = await _user(db, role="patient")
    with pytest.raises(EntityNotFoundException):
        await owners.set_owner_active(db, patient.id, active=False, actor=admin)


# ── credentials & invitation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_password_reset_issues_a_working_credential(db: AsyncSession):
    admin = await _admin(db)
    pharmacy = await _pharmacy(db)
    owner, original = await owners.create_owner(
        db, pharmacy.id, email=f"pw-{uuid.uuid4().hex[:6]}@example.com", actor=admin
    )
    await db.flush()

    _, fresh = await owners.reset_password(db, owner.id, actor=admin)
    await db.flush()

    assert fresh != original
    assert verify_password(fresh, owner.hashed_password) is True
    # The superseded password stops working.
    assert verify_password(original, owner.hashed_password) is False


@pytest.mark.asyncio
async def test_invitation_reports_email_was_not_sent(db: AsyncSession):
    """
    The platform's email task is a stub that never contacts a mail server.
    Reporting `email_sent: true` would be a lie the operator would act on.
    """
    admin = await _admin(db)
    pharmacy = await _pharmacy(db)
    owner, _ = await owners.create_owner(
        db, pharmacy.id, email=f"inv-{uuid.uuid4().hex[:6]}@example.com", actor=admin
    )
    await db.flush()

    result = await owners.send_invitation(db, owner.id, actor=admin)
    await db.flush()

    assert result["email_sent"] is False
    assert result["delivery"] == "in_app_notification"
    assert result["temporary_password"]
    assert verify_password(result["temporary_password"], owner.hashed_password) is True


@pytest.mark.asyncio
async def test_invitation_requires_a_linked_pharmacy(db: AsyncSession):
    admin = await _admin(db)
    orphan = await _user(db)
    with pytest.raises(BusinessRuleValidationException):
        await owners.send_invitation(db, orphan.id, actor=admin)


# ── audit ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_provisioning_action_is_audited(db: AsyncSession):
    from app.services.pharmacy_admin import pharmacy_admin_service

    admin = await _admin(db)
    pharmacy = await _pharmacy(db)

    owner, _ = await owners.create_owner(
        db, pharmacy.id, email=f"aud-{uuid.uuid4().hex[:6]}@example.com", actor=admin
    )
    await db.flush()
    await owners.set_owner_active(db, owner.id, active=False, actor=admin, reason="x")
    await owners.set_owner_active(db, owner.id, active=True, actor=admin)
    await owners.reset_password(db, owner.id, actor=admin)
    await db.flush()

    entries, _ = await pharmacy_admin_service.audit_trail(db, resource_id=str(owner.id))
    actions = {e.action for e in entries}
    assert "PHARMACY_OWNER_SUSPENDED" in actions
    assert "PHARMACY_OWNER_ACTIVATED" in actions
    assert "PHARMACY_OWNER_PASSWORD_RESET" in actions

    store_entries, _ = await pharmacy_admin_service.audit_trail(
        db, resource_id=str(pharmacy.id)
    )
    assert "PHARMACY_OWNER_CREATED" in {e.action for e in store_entries}


@pytest.mark.asyncio
async def test_audit_never_records_a_password(db: AsyncSession):
    """The audit trail is readable by other administrators."""
    from app.services.pharmacy_admin import pharmacy_admin_service

    admin = await _admin(db)
    pharmacy = await _pharmacy(db)
    owner, temporary = await owners.create_owner(
        db, pharmacy.id, email=f"sec-{uuid.uuid4().hex[:6]}@example.com", actor=admin
    )
    await db.flush()
    await owners.reset_password(db, owner.id, actor=admin)
    await db.flush()

    entries, _ = await pharmacy_admin_service.audit_trail(db, resource_id=str(owner.id))
    for entry in entries:
        blob = " ".join(
            filter(None, [entry.details, entry.previous_value, entry.new_value])
        )
        assert temporary not in blob
        assert owner.hashed_password not in blob
