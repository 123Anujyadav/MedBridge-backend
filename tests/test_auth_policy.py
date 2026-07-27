"""
The platform's authentication and authorisation policy, enforced.

Three roles, three rules:

* **Patient** — may self-register, active immediately.
* **Doctor** — may self-register, but holds no session and reaches no clinical
  data until an administrator approves the account.
* **Admin** — cannot be registered at all. Accounts are created by an existing
  administrator or directly in the database.

Underlying all of it: a token establishes *which* account is calling and
nothing more. Role, active status and clinical approval are read from the
database on every request, so a forged or stale claim grants nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from httpx import AsyncClient
from sqlalchemy import select, text

from app.core.config import settings
from app.core.security import get_password_hash
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.user import User
from app.services.doctor_access import AWAITING_APPROVAL_MESSAGE
from conftest import login_payload

pytestmark = pytest.mark.asyncio

PW = "policy-password-123"
ADMIN = "policy.admin@aronofy.com"
PATIENT = "policy.patient@aronofy.com"
DOCTOR = "policy.doctor@aronofy.com"

PATIENT_PROFILE = {
    "first_name": "Asha", "last_name": "Verma", "phone": "+919812345670",
    "date_of_birth": "1991-04-17", "gender": "female",
}
DOCTOR_PROFILE = {
    "first_name": "Neha", "last_name": "Kulkarni", "phone": "+919812349999",
    "specialty": "Cardiology", "license_number": "MH-POLICY-1",
}


@pytest.fixture
async def estate(db):
    """Only a pre-created administrator — the roles under test register."""
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("appointments", "cases", "doctors", "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))

    admin = User(email=ADMIN, hashed_password=get_password_hash(PW),
                 role="admin", is_verified=True)
    db.add(admin)
    await db.commit()
    return {"admin": admin.id}


async def login(client: AsyncClient, email: str, password: str = PW):
    return await client.post("/api/v1/auth/login",
                             json=await login_payload(email, password))


async def auth(client: AsyncClient, email: str) -> dict[str, str]:
    resp = await login(client, email)
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def register_patient(client: AsyncClient, email: str = PATIENT):
    return await client.post("/api/v1/auth/signup/patient", json={
        "email": email, "password": PW, "profile": PATIENT_PROFILE})


async def register_doctor(client: AsyncClient, email: str = DOCTOR, licence="MH-POLICY-1"):
    profile = {**DOCTOR_PROFILE, "license_number": licence}
    return await client.post("/api/v1/auth/signup/doctor", json={
        "email": email, "password": PW, "profile": profile})


class TestPatientRegistration:
    async def test_patient_can_self_register_and_sign_in(self, client, estate):
        assert (await register_patient(client)).status_code == 201
        resp = await login(client, PATIENT)
        assert resp.status_code == 200, resp.text
        assert resp.json()["access_token"]

    async def test_registration_assigns_the_patient_role_server_side(
        self, client, estate, db
    ):
        await register_patient(client)
        role = await db.scalar(select(User.role).where(User.email == PATIENT))
        assert role == "patient"

    async def test_email_must_be_unique(self, client, estate):
        assert (await register_patient(client)).status_code == 201
        assert (await register_patient(client)).status_code in (400, 409, 422)

    async def test_patient_reaches_only_patient_apis(self, client, estate):
        await register_patient(client)
        headers = await auth(client, PATIENT)
        assert (await client.get("/api/v1/patient/dashboard",
                                 headers=headers)).status_code == 200
        assert (await client.get("/api/v1/doctor/dashboard",
                                 headers=headers)).status_code == 403
        assert (await client.get("/api/v1/admin/dashboard",
                                 headers=headers)).status_code == 403


class TestDoctorRegistrationAndApproval:
    async def test_signup_creates_a_pending_doctor(self, client, estate, db):
        assert (await register_doctor(client)).status_code == 201
        user = await db.scalar(select(User).where(User.email == DOCTOR))
        doctor = await db.get(Doctor, user.id)
        assert user.role == "doctor"
        assert doctor.verification_status == "pending"

    async def test_unapproved_doctor_cannot_sign_in(self, client, estate):
        await register_doctor(client)
        resp = await login(client, DOCTOR)
        assert resp.status_code == 403, resp.text
        assert resp.json()["message"] == AWAITING_APPROVAL_MESSAGE

    async def test_under_review_doctor_still_cannot_sign_in(self, client, estate, db):
        await register_doctor(client)
        user = await db.scalar(select(User).where(User.email == DOCTOR))
        doctor = await db.get(Doctor, user.id)
        doctor.verification_status = "under_review"
        await db.commit()

        resp = await login(client, DOCTOR)
        assert resp.status_code == 403
        assert resp.json()["message"] == AWAITING_APPROVAL_MESSAGE

    async def test_rejected_doctor_is_told_the_decision(self, client, estate, db):
        await register_doctor(client)
        user = await db.scalar(select(User).where(User.email == DOCTOR))
        doctor = await db.get(Doctor, user.id)
        doctor.verification_status = "rejected"
        await db.commit()

        resp = await login(client, DOCTOR)
        assert resp.status_code == 403
        # A rejected clinician must not be told to keep waiting.
        assert resp.json()["message"] != AWAITING_APPROVAL_MESSAGE

    async def test_approved_doctor_can_sign_in_and_work(self, client, estate, db):
        await register_doctor(client)
        user = await db.scalar(select(User).where(User.email == DOCTOR))

        admin_headers = await auth(client, ADMIN)
        approved = await client.put(f"/api/v1/admin/doctors/{user.id}/verify",
                                    json={"verification_status": "verified"},
                                    headers=admin_headers)
        assert approved.status_code == 200, approved.text

        headers = await auth(client, DOCTOR)
        assert (await client.get("/api/v1/doctor/dashboard",
                                 headers=headers)).status_code == 200

    async def test_approval_withdrawn_mid_session_locks_the_doctor_out(
        self, client, estate, db
    ):
        """
        A token issued while approved must stop working the moment approval is
        revoked — the guard reads the current row, it does not trust the session.
        """
        await register_doctor(client)
        user = await db.scalar(select(User).where(User.email == DOCTOR))
        doctor = await db.get(Doctor, user.id)
        doctor.verification_status = "verified"
        await db.commit()

        headers = await auth(client, DOCTOR)
        assert (await client.get("/api/v1/doctor/dashboard",
                                 headers=headers)).status_code == 200

        admin_headers = await auth(client, ADMIN)
        await client.put(f"/api/v1/admin/doctors/{user.id}/verify",
                         json={"verification_status": "rejected"},
                         headers=admin_headers)

        assert (await client.get("/api/v1/doctor/dashboard",
                                 headers=headers)).status_code == 403

    async def test_suspended_doctor_loses_access_and_reactivation_restores_it(
        self, client, estate, db
    ):
        await register_doctor(client)
        user = await db.scalar(select(User).where(User.email == DOCTOR))
        doctor = await db.get(Doctor, user.id)
        doctor.verification_status = "verified"
        await db.commit()

        headers = await auth(client, DOCTOR)
        admin_headers = await auth(client, ADMIN)

        suspended = await client.put(f"/api/v1/admin/users/{user.id}/status",
                                     json={"is_active": False},
                                     headers=admin_headers)
        assert suspended.status_code == 200
        assert (await client.get("/api/v1/doctor/dashboard",
                                 headers=headers)).status_code == 403
        assert (await login(client, DOCTOR)).status_code == 403

        await client.put(f"/api/v1/admin/users/{user.id}/status",
                         json={"is_active": True}, headers=admin_headers)
        assert (await login(client, DOCTOR)).status_code == 200


class TestAdminRegistrationIsImpossible:
    async def test_no_admin_signup_route_exists(self, client, estate):
        for path in ("/api/v1/auth/signup/admin", "/api/v1/admin/signup",
                     "/api/v1/auth/signup"):
            resp = await client.post(path, json={
                "email": "intruder@aronofy.com", "password": PW,
                "profile": PATIENT_PROFILE})
            assert resp.status_code in (404, 405), f"{path} -> {resp.status_code}"

    async def test_openapi_exposes_only_patient_and_doctor_signup(self, client):
        from app.main import app

        signup_paths = {p for p in app.openapi()["paths"] if "signup" in p}
        assert signup_paths == {
            "/api/v1/auth/signup/patient",
            "/api/v1/auth/signup/doctor",
        }

    async def test_role_cannot_be_smuggled_through_signup(self, client, estate, db):
        """An extra `role` field in the payload must not become the account's role."""
        resp = await client.post("/api/v1/auth/signup/patient", json={
            "email": "escalate@aronofy.com", "password": PW,
            "role": "admin", "is_active": True,
            "profile": {**PATIENT_PROFILE, "role": "admin"}})
        assert resp.status_code in (201, 422)

        if resp.status_code == 201:
            role = await db.scalar(
                select(User.role).where(User.email == "escalate@aronofy.com"))
            assert role == "patient"

    async def test_patient_cannot_promote_themselves(self, client, estate, db):
        await register_patient(client)
        headers = await auth(client, PATIENT)

        # The profile update surface must not carry role or status.
        await client.put("/api/v1/patient/profile",
                         json={"role": "admin", "is_active": True, "city": "Pune"},
                         headers=headers)

        role = await db.scalar(select(User.role).where(User.email == PATIENT))
        assert role == "patient"
        assert (await client.get("/api/v1/admin/dashboard",
                                 headers=headers)).status_code == 403

    async def test_patient_cannot_use_admin_user_management(self, client, estate):
        await register_patient(client)
        headers = await auth(client, PATIENT)
        me = await client.get("/api/v1/auth/me", headers=headers)
        user_id = me.json()["id"]

        assert (await client.put(f"/api/v1/admin/users/{user_id}/status",
                                 json={"is_active": True},
                                 headers=headers)).status_code == 403


class TestRoleComesFromTheDatabase:
    async def test_forged_admin_claim_is_ignored(self, client, estate, db):
        """
        A token is signed by this server, so its *claims* can be crafted by
        anyone able to mint one. Authorisation reads the database instead, so an
        `admin` claim on a patient's token buys nothing.
        """
        await register_patient(client)
        user = await db.scalar(select(User).where(User.email == PATIENT))

        forged = jwt.encode(
            {
                "sub": str(user.id),
                "type": "access",
                "role": "admin",           # ignored by design
                "is_active": True,
                "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
            },
            settings.JWT_SECRET,
            algorithm=settings.JWT_ALGORITHM,
        )
        headers = {"Authorization": f"Bearer {forged}"}

        assert (await client.get("/api/v1/admin/dashboard",
                                 headers=headers)).status_code == 403
        assert (await client.get("/api/v1/doctor/dashboard",
                                 headers=headers)).status_code == 403
        # The same token still works for what the account actually is.
        assert (await client.get("/api/v1/patient/dashboard",
                                 headers=headers)).status_code == 200

    async def test_refresh_token_is_not_accepted_as_an_access_token(
        self, client, estate
    ):
        await register_patient(client)
        resp = await login(client, PATIENT)
        refresh = resp.json()["refresh_token"]

        assert (await client.get(
            "/api/v1/patient/dashboard",
            headers={"Authorization": f"Bearer {refresh}"})).status_code == 401

    async def test_token_for_a_deleted_account_is_refused(self, client, estate, db):
        await register_patient(client)
        headers = await auth(client, PATIENT)
        user = await db.scalar(select(User).where(User.email == PATIENT))

        await db.execute(text("PRAGMA foreign_keys = OFF;"))
        # Matched on email: SQLite stores the UUID primary key as undashed hex,
        # so comparing against `str(uuid)` silently matches nothing.
        await db.execute(
            text("DELETE FROM patients WHERE id IN "
                 "(SELECT id FROM users WHERE email = :e)"), {"e": PATIENT})
        await db.execute(text("DELETE FROM users WHERE email = :e"), {"e": PATIENT})
        await db.commit()
        # Tests share one session with the app, and a raw DELETE does not evict
        # the identity map. Production gives every request a fresh session, so
        # the cached instance is dropped here to reproduce that.
        db.expunge_all()

        assert (await client.get("/api/v1/patient/dashboard",
                                 headers=headers)).status_code == 401

    async def test_unsigned_and_malformed_tokens_are_refused(self, client, estate):
        for token in ("", "not-a-token", "a.b.c",
                      jwt.encode({"sub": "x", "type": "access"},
                                 "the-wrong-secret", algorithm="HS256")):
            resp = await client.get("/api/v1/patient/dashboard",
                                    headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code in (401, 403)


class TestAdminAccess:
    async def test_admin_signs_in_and_reaches_only_admin_apis(self, client, estate):
        headers = await auth(client, ADMIN)
        assert (await client.get("/api/v1/admin/dashboard",
                                 headers=headers)).status_code == 200
        assert (await client.get("/api/v1/patient/dashboard",
                                 headers=headers)).status_code == 403
        assert (await client.get("/api/v1/doctor/dashboard",
                                 headers=headers)).status_code == 403

    async def test_admin_sees_pending_doctors_and_can_action_them(
        self, client, estate, db
    ):
        await register_doctor(client)
        headers = await auth(client, ADMIN)

        pending = await client.get("/api/v1/admin/doctors/pending", headers=headers)
        assert pending.status_code == 200
        assert any(d["verification_status"] == "pending" for d in pending.json())

        user = await db.scalar(select(User).where(User.email == DOCTOR))
        for target in ("verified", "rejected", "under_review"):
            resp = await client.put(f"/api/v1/admin/doctors/{user.id}/verify",
                                    json={"verification_status": target},
                                    headers=headers)
            assert resp.status_code == 200
            assert resp.json()["verification_status"] == target


class TestSessionPersistence:
    async def test_session_survives_token_rotation(self, client, estate):
        await register_patient(client)
        signed_in = await login(client, PATIENT)
        refresh = signed_in.json()["refresh_token"]

        rotated = await client.post("/api/v1/auth/refresh",
                                    json={"refresh_token": refresh})
        assert rotated.status_code == 200, rotated.text
        headers = {"Authorization": f"Bearer {rotated.json()['access_token']}"}
        assert (await client.get("/api/v1/auth/me", headers=headers)).status_code == 200

    async def test_logout_ends_the_session(self, client, estate):
        await register_patient(client)
        signed_in = await login(client, PATIENT)
        refresh = signed_in.json()["refresh_token"]

        assert (await client.post("/api/v1/auth/logout",
                                  json={"refresh_token": refresh},
                                  headers={"Authorization":
                                           f"Bearer {signed_in.json()['access_token']}"}
                                  )).status_code == 200
        assert (await client.post("/api/v1/auth/refresh",
                                  json={"refresh_token": refresh})).status_code == 401
