"""
The clinician's Doctor ID, and the ceiling on administrator accounts.

Two rules that are only worth having if they cannot be talked around, so most
of what is here is the ways round them:

* signing in as a doctor with the right password and no Doctor ID, the wrong
  one, another doctor's, or one in the wrong case;
* an administrator's approval being what issues the ID, and withdrawing it
  taking effect on the next request rather than the next sign-in;
* a third administrator, by any route the application offers.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text

from app.core.doctor_code import (
    DOCTOR_CODE_LENGTH,
    doctor_codes_match,
    generate_doctor_code,
    is_valid_doctor_code,
    normalise_doctor_code,
)
from app.core.security import get_password_hash
from app.models.doctor import Doctor
from app.models.user import User
from app.services.admin_accounts import (
    MAX_ADMIN_ACCOUNTS,
    assert_admin_slot_available,
    count_admin_accounts,
)
from conftest import login_payload

pytestmark = pytest.mark.asyncio

PW = "password123"
ADMIN = "id.admin@aronofy.com"
DOCTOR = "id.doctor@aronofy.com"
OTHER_DOCTOR = "id.other@aronofy.com"
PATIENT = "id.patient@aronofy.com"


@pytest.fixture
async def estate(db):
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("appointments", "cases", "doctors", "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))

    ids = {}
    for email, role in ((ADMIN, "admin"), (DOCTOR, "doctor"),
                        (OTHER_DOCTOR, "doctor"), (PATIENT, "patient")):
        user = User(email=email, hashed_password=get_password_hash(PW),
                    role=role, is_verified=True, is_active=True)
        db.add(user)
        await db.flush()
        ids[email] = user.id

    for email, licence in ((DOCTOR, "LIC-ID-1"), (OTHER_DOCTOR, "LIC-ID-2")):
        db.add(Doctor(
            id=ids[email], first_name="Test", last_name="Clinician",
            phone="+10000000000", specialty="Cardiology",
            license_number=licence, verification_status="verified",
            # Stated explicitly because a Doctor ID is issued by an
            # administrator's approval and by nothing else. These clinicians are
            # already approved, so they hold one; a pending clinician would not.
            doctor_code=generate_doctor_code(),
        ))
    await db.commit()
    return ids


async def sign_in(client: AsyncClient, email: str, doctor_id=..., password=PW):
    """
    Post a sign-in. `doctor_id` defaults to the account's real one; pass an
    explicit value (including None) to send something else.
    """
    body = await login_payload(email, password)
    if doctor_id is not ...:
        body.pop("doctor_id", None)
        if doctor_id is not None:
            body["doctor_id"] = doctor_id
    return await client.post("/api/v1/auth/login", json=body)


# ── the identifier itself ────────────────────────────────────────────────

class TestDoctorCodeGeneration:
    def test_is_eight_upper_case_alphanumerics(self):
        for _ in range(200):
            code = generate_doctor_code()
            assert len(code) == DOCTOR_CODE_LENGTH == 8
            assert code.isupper() or code.isdigit()
            assert is_valid_doctor_code(code)

    def test_successive_draws_differ(self):
        """
        A predictable ID would collapse clinician sign-in back to two factors,
        so this is a security property rather than a quality-of-output one.
        """
        drawn = {generate_doctor_code() for _ in range(500)}
        assert len(drawn) == 500

    def test_normalisation_accepts_how_people_actually_type_it(self):
        assert normalise_doctor_code(" dr8a9xq2 ") == "DR8A9XQ2"
        assert normalise_doctor_code("DR8A 9XQ2") == "DR8A9XQ2"
        assert normalise_doctor_code(None) == ""

    def test_a_missing_stored_code_never_matches(self):
        """The check that stops an un-issued profile being signed into."""
        assert doctor_codes_match("DR8A9XQ2", None) is False
        assert doctor_codes_match("DR8A9XQ2", "") is False
        assert doctor_codes_match(None, "DR8A9XQ2") is False
        assert doctor_codes_match("", "") is False

    def test_rejects_anything_outside_the_alphabet(self):
        assert not is_valid_doctor_code("dr8a9xq2")   # lower case
        assert not is_valid_doctor_code("DR8A9XQ")    # seven
        assert not is_valid_doctor_code("DR8A9XQ23")  # nine
        assert not is_valid_doctor_code("DR8A-9XQ")   # punctuation

    async def test_a_new_doctor_row_holds_no_id_until_approval(self, db, estate):
        """
        The workflow is signup → pending → approval → Doctor ID issued.

        A clinician who has not been approved must have no ID at all: there is
        nothing to leak and nothing to sign in with. The IDs in this fixture
        exist only because it approves its doctors explicitly.
        """
        user = User(email="fresh.doctor@aronofy.com",
                    hashed_password=get_password_hash(PW),
                    role="doctor", is_active=True, is_verified=True)
        db.add(user)
        await db.flush()
        db.add(Doctor(
            id=user.id, first_name="Fresh", last_name="Signup",
            phone="+10000000001", specialty="Cardiology",
            license_number="LIC-FRESH-1", verification_status="pending",
        ))
        await db.flush()

        issued = await db.scalar(
            select(Doctor.doctor_code).where(Doctor.id == user.id)
        )
        assert issued is None, "a pending clinician was issued a Doctor ID"

    async def test_approved_doctors_hold_unique_valid_ids(self, db, estate):
        codes = [c for c in
                 (await db.execute(select(Doctor.doctor_code))).scalars()
                 if c is not None]
        assert all(is_valid_doctor_code(c) for c in codes)
        assert len(set(codes)) == len(codes)


# ── sign-in ──────────────────────────────────────────────────────────────

class TestDoctorSignIn:
    async def test_correct_three_factors_succeed(self, client, estate):
        resp = await sign_in(client, DOCTOR)
        assert resp.status_code == 200, resp.text
        assert resp.json()["access_token"]

    async def test_email_and_password_alone_are_refused(self, client, estate):
        """The rule the whole feature exists for."""
        resp = await sign_in(client, DOCTOR, doctor_id=None)
        assert resp.status_code == 401, resp.text

    async def test_wrong_doctor_id_is_refused(self, client, estate):
        resp = await sign_in(client, DOCTOR, doctor_id="AAAA1111")
        assert resp.status_code == 401

    async def test_another_clinicians_doctor_id_is_refused(self, client, estate, db):
        other = await db.scalar(
            select(Doctor.doctor_code).where(Doctor.id == estate[OTHER_DOCTOR])
        )
        resp = await sign_in(client, DOCTOR, doctor_id=other)
        assert resp.status_code == 401

    async def test_correct_doctor_id_with_wrong_password_is_refused(
        self, client, estate
    ):
        resp = await sign_in(client, DOCTOR, password="not-the-password")
        assert resp.status_code == 401

    async def test_lower_case_and_spacing_are_accepted(self, client, estate, db):
        code = await db.scalar(
            select(Doctor.doctor_code).where(Doctor.id == estate[DOCTOR])
        )
        resp = await sign_in(client, DOCTOR, doctor_id=f" {code.lower()} ")
        assert resp.status_code == 200, resp.text

    async def test_every_failure_reads_the_same(self, client, estate, db):
        """
        A distinguishable message would turn the clinician endpoint into an
        oracle: confirm a stolen password, then brute-force the Doctor ID with
        a clear signal on every guess.
        """
        code = await db.scalar(
            select(Doctor.doctor_code).where(Doctor.id == estate[DOCTOR])
        )
        attempts = [
            {"doctor_id": "AAAA1111", "email": DOCTOR, "password": PW},
            {"doctor_id": code, "email": DOCTOR, "password": "wrong"},
            {"doctor_id": "AAAA1111", "email": DOCTOR, "password": "wrong"},
            {"doctor_id": code, "email": "nobody@aronofy.com", "password": PW},
        ]
        results = set()
        for body in attempts:
            resp = await client.post("/api/v1/auth/login/doctor", json=body)
            results.add((resp.status_code, resp.json()["message"]))
        assert len(results) == 1, results

    async def test_a_doctor_id_does_not_let_a_patient_in_by_the_doctor_route(
        self, client, estate, db
    ):
        code = await db.scalar(
            select(Doctor.doctor_code).where(Doctor.id == estate[DOCTOR])
        )
        resp = await client.post("/api/v1/auth/login/doctor", json={
            "doctor_id": code, "email": PATIENT, "password": PW,
        })
        assert resp.status_code == 401

    async def test_dedicated_route_requires_the_field(self, client, estate):
        resp = await client.post("/api/v1/auth/login/doctor", json={
            "email": DOCTOR, "password": PW,
        })
        assert resp.status_code == 422

    async def test_patient_sign_in_is_unchanged(self, client, estate):
        """The patient module must not have acquired a third factor."""
        resp = await client.post("/api/v1/auth/login",
                                 json={"email": PATIENT, "password": PW})
        assert resp.status_code == 200, resp.text

    async def test_admin_sign_in_is_unchanged(self, client, estate):
        resp = await client.post("/api/v1/auth/login",
                                 json={"email": ADMIN, "password": PW})
        assert resp.status_code == 200, resp.text

    async def test_unverified_doctor_is_refused_even_with_the_right_id(
        self, client, estate, db
    ):
        doctor = await db.get(Doctor, estate[DOCTOR])
        doctor.verification_status = "pending"
        await db.commit()

        resp = await sign_in(client, DOCTOR)
        assert resp.status_code == 403, resp.text


# ── the administrator's decisions ────────────────────────────────────────

class TestAdminVerification:
    async def admin_headers(self, client):
        resp = await client.post("/api/v1/auth/login",
                                 json={"email": ADMIN, "password": PW})
        assert resp.status_code == 200, resp.text
        return {"Authorization": f"Bearer {resp.json()['access_token']}"}

    async def test_approval_issues_an_id_when_the_profile_has_none(
        self, client, estate, db
    ):
        doctor = await db.get(Doctor, estate[DOCTOR])
        doctor.doctor_code = None
        doctor.verification_status = "pending"
        await db.commit()

        resp = await client.put(
            f"/api/v1/admin/doctors/{estate[DOCTOR]}/verify",
            json={"verification_status": "verified"},
            headers=await self.admin_headers(client),
        )
        assert resp.status_code == 200, resp.text
        assert is_valid_doctor_code(resp.json()["doctor_code"])

    async def test_re_approval_keeps_the_id_already_handed_out(
        self, client, estate, db
    ):
        """
        Reissuing on every approval would lock out a clinician who had already
        been told theirs.
        """
        before = await db.scalar(
            select(Doctor.doctor_code).where(Doctor.id == estate[DOCTOR])
        )
        headers = await self.admin_headers(client)
        for _ in range(2):
            resp = await client.put(
                f"/api/v1/admin/doctors/{estate[DOCTOR]}/verify",
                json={"verification_status": "verified"}, headers=headers,
            )
            assert resp.status_code == 200
            assert resp.json()["doctor_code"] == before

    async def test_unverifying_ends_access_on_the_next_request(
        self, client, estate, db
    ):
        signed_in = await sign_in(client, DOCTOR)
        headers = {"Authorization": f"Bearer {signed_in.json()['access_token']}"}
        assert (await client.get("/api/v1/doctor/dashboard",
                                 headers=headers)).status_code == 200

        await client.put(
            f"/api/v1/admin/doctors/{estate[DOCTOR]}/verify",
            json={"verification_status": "pending"},
            headers=await self.admin_headers(client),
        )
        assert (await client.get("/api/v1/doctor/dashboard",
                                 headers=headers)).status_code == 403
        assert (await sign_in(client, DOCTOR)).status_code == 403

    async def test_rejecting_ends_access(self, client, estate):
        await client.put(
            f"/api/v1/admin/doctors/{estate[DOCTOR]}/verify",
            json={"verification_status": "rejected"},
            headers=await self.admin_headers(client),
        )
        assert (await sign_in(client, DOCTOR)).status_code == 403

    async def test_suspending_the_account_ends_access(self, client, estate):
        await client.put(
            f"/api/v1/admin/users/{estate[DOCTOR]}/status",
            json={"is_active": False},
            headers=await self.admin_headers(client),
        )
        assert (await sign_in(client, DOCTOR)).status_code == 403

    async def test_review_list_carries_what_the_screen_shows(self, client, estate):
        resp = await client.get("/api/v1/admin/doctors",
                                headers=await self.admin_headers(client))
        assert resp.status_code == 200, resp.text
        # Paginated envelope since the audit found the bare array silently
        # truncated at 100 rows; see test_doctor_identity_hardening.py.
        row = next(r for r in resp.json()["items"] if r["id"] == str(estate[DOCTOR]))
        for field in ("doctor_code", "email", "phone", "specialty",
                      "sub_specialties", "hospital_name", "license_number",
                      "years_of_experience", "education", "certifications",
                      "languages", "verification_status", "registered_at",
                      "is_active"):
            assert field in row, f"{field} missing from the verification queue"

    async def test_the_doctor_id_is_not_served_to_a_doctor_or_a_patient(
        self, client, estate
    ):
        """It is a sign-in factor, so only an administrator may read it."""
        signed_in = await sign_in(client, DOCTOR)
        headers = {"Authorization": f"Bearer {signed_in.json()['access_token']}"}
        assert (await client.get("/api/v1/admin/doctors",
                                 headers=headers)).status_code == 403

        patient = await client.post("/api/v1/auth/login",
                                    json={"email": PATIENT, "password": PW})
        p_headers = {"Authorization": f"Bearer {patient.json()['access_token']}"}
        assert (await client.get("/api/v1/admin/doctors",
                                 headers=p_headers)).status_code == 403


# ── the administrator ceiling ────────────────────────────────────────────

class TestAdminAccountCap:
    async def test_the_cap_is_two(self):
        assert MAX_ADMIN_ACCOUNTS == 2

    async def test_a_third_administrator_is_refused(self, db, estate):
        from app.core.exceptions import BusinessRuleValidationException

        assert await count_admin_accounts(db) == 1
        await assert_admin_slot_available(db)  # room for one more

        db.add(User(email="second.admin@aronofy.com",
                    hashed_password=get_password_hash(PW),
                    role="admin", is_active=True))
        await db.flush()
        assert await count_admin_accounts(db) == 2

        with pytest.raises(BusinessRuleValidationException):
            await assert_admin_slot_available(db)

    async def test_a_retired_administrator_gives_up_their_slot(self, db, estate):
        db.add(User(email="third.admin@aronofy.com",
                    hashed_password=get_password_hash(PW),
                    role="admin", is_active=True))
        await db.flush()
        assert await count_admin_accounts(db) == 2

        retiring = await db.scalar(select(User).where(User.email == ADMIN))
        retiring.is_active = False
        await db.flush()

        assert await count_admin_accounts(db) == 1
        await assert_admin_slot_available(db)  # does not raise

    async def test_deactivated_and_soft_deleted_admins_are_not_counted(
        self, db, estate
    ):
        from datetime import datetime, timezone

        db.add(User(email="gone.admin@aronofy.com",
                    hashed_password=get_password_hash(PW), role="admin",
                    is_active=False, deleted_at=datetime.now(timezone.utc)))
        await db.flush()
        assert await count_admin_accounts(db) == 1

    async def test_reinstating_an_administrator_respects_the_cap(self, db, estate):
        """
        Reactivating is a creation as far as the cap is concerned — the slot may
        have been taken while the account was retired.
        """
        from app.core.exceptions import BusinessRuleValidationException
        from app.services.admin import admin_service

        retired = User(email="back.admin@aronofy.com",
                       hashed_password=get_password_hash(PW), role="admin",
                       is_active=False)
        db.add(retired)
        db.add(User(email="filled.admin@aronofy.com",
                    hashed_password=get_password_hash(PW),
                    role="admin", is_active=True))
        await db.flush()
        assert await count_admin_accounts(db) == 2

        with pytest.raises(BusinessRuleValidationException):
            await admin_service.update_user_status(db, retired.id, True)

    async def test_the_cap_endpoint_reports_the_real_numbers(self, client, estate):
        resp = await client.post("/api/v1/auth/login",
                                 json={"email": ADMIN, "password": PW})
        headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}

        cap = await client.get("/api/v1/admin/admin-accounts", headers=headers)
        assert cap.status_code == 200, cap.text
        body = cap.json()
        assert body["maximum"] == MAX_ADMIN_ACCOUNTS
        assert body["in_use"] == 1
        assert body["slots_available"] == 1
