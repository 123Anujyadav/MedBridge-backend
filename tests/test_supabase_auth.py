"""
Supabase as the identity provider, with this database still the source of truth.

Supabase is stubbed in-process. That is deliberate rather than a shortcut: these
tests must assert *our* behaviour — that a rejected clinician is refused, that a
role claim is ignored, that an existing account is linked and never duplicated —
and they must do so deterministically, without a network or a live project.

The stub speaks the parts of the GoTrue contract the client actually uses, and
signs tokens the way a Supabase project does, so `SupabaseJWTVerifier` performs
a real signature check against them.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from httpx import AsyncClient
from sqlalchemy import select, text

from app.core import identity as identity_module
from app.core.config import settings
from app.core.identity import LocalJWTVerifier, SupabaseJWTVerifier, set_token_verifier
from app.core.security import get_password_hash
from app.core.supabase import (
    SupabaseAuthClient,
    SupabaseAuthError,
    set_supabase_auth_client,
)
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.user import User
from app.services.doctor_access import AWAITING_APPROVAL_MESSAGE
from conftest import login_payload

pytestmark = pytest.mark.asyncio

PW = "supabase-password-123"
JWT_SECRET = "stub-supabase-jwt-secret-value-0123456789"

ADMIN = "sb.admin@aronofy.com"
PATIENT = "sb.patient@aronofy.com"
DOCTOR = "sb.doctor@aronofy.com"
LEGACY = "sb.legacy@aronofy.com"

PATIENT_PROFILE = {
    "first_name": "Asha", "last_name": "Verma", "phone": "+919812345670",
    "date_of_birth": "1991-04-17", "gender": "female",
}
DOCTOR_PROFILE = {
    "first_name": "Neha", "last_name": "Kulkarni", "phone": "+919812349999",
    "specialty": "Cardiology", "license_number": "MH-SB-1",
}


class FakeSupabase(SupabaseAuthClient):
    """
    An in-process GoTrue.

    Holds identities in a dict and mints HS256 access tokens exactly as a
    Supabase project with a JWT secret does, so token verification in the code
    under test is genuine.
    """

    def __init__(self) -> None:
        super().__init__(base_url="https://stub.supabase.co",
                         anon_key="anon", service_role_key="service", timeout=5)
        self.users: dict[str, dict] = {}          # id -> record
        self.by_email: dict[str, str] = {}        # email -> id
        self.sessions: dict[str, str] = {}        # refresh token -> id
        self.revoked: set[str] = set()
        self.recovery_emails: list[str] = []
        self.verification_emails: list[str] = []

    # -- helpers
    def _token_for(self, user_id: str) -> str:
        record = self.users[user_id]
        return jwt.encode(
            {
                "sub": user_id,
                "email": record["email"],
                "role": "authenticated",
                "aud": "authenticated",
                "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            },
            JWT_SECRET, algorithm="HS256",
        )

    def _session(self, user_id: str) -> dict:
        refresh = uuid.uuid4().hex
        self.sessions[refresh] = user_id
        return {
            "access_token": self._token_for(user_id),
            "refresh_token": refresh,
            "token_type": "bearer",
            "user": self.users[user_id],
        }

    def seed(self, email: str, password: str, confirmed: bool = True) -> str:
        user_id = str(uuid.uuid4())
        self.users[user_id] = {
            "id": user_id, "email": email,
            "email_confirmed_at": datetime.now(timezone.utc).isoformat()
            if confirmed else None,
        }
        self.by_email[email.lower()] = user_id
        self.users[user_id]["_password"] = password
        return user_id

    # -- GoTrue surface
    async def sign_in(self, email: str, password: str) -> dict:
        user_id = self.by_email.get(email.lower())
        if user_id is None or self.users[user_id]["_password"] != password:
            raise SupabaseAuthError("Invalid login credentials", 400)
        return self._session(user_id)

    async def refresh_session(self, refresh_token: str) -> dict:
        if refresh_token in self.revoked:
            raise SupabaseAuthError("Invalid Refresh Token", 401)
        user_id = self.sessions.get(refresh_token)
        if user_id is None:
            raise SupabaseAuthError("Invalid Refresh Token", 401)
        self.revoked.add(refresh_token)          # Supabase rotates
        return self._session(user_id)

    async def sign_out(self, access_token: str) -> None:
        claims = jwt.decode(access_token, JWT_SECRET, algorithms=["HS256"],
                            options={"verify_aud": False})
        user_id = claims["sub"]
        for token, owner in list(self.sessions.items()):
            if owner == user_id:
                self.revoked.add(token)

    async def get_user(self, access_token: str) -> dict:
        try:
            claims = jwt.decode(access_token, JWT_SECRET, algorithms=["HS256"],
                                options={"verify_aud": False})
        except jwt.PyJWTError as exc:
            raise SupabaseAuthError("invalid claim", 401) from exc
        return self.users[claims["sub"]]

    async def send_password_reset(self, email: str, redirect_to=None) -> None:
        self.recovery_emails.append(email)

    async def resend_verification(self, email: str) -> None:
        self.verification_emails.append(email)

    async def admin_create_user(self, email, password, *, email_confirm=False,
                                user_metadata=None) -> dict:
        if email.lower() in self.by_email:
            raise SupabaseAuthError(
                "A user with this email address has already been registered", 422)
        user_id = self.seed(email, password, confirmed=email_confirm)
        return self.users[user_id]

    async def admin_get_user_by_email(self, email: str):
        user_id = self.by_email.get(email.lower())
        return self.users[user_id] if user_id else None

    async def admin_delete_user(self, supabase_user_id: str) -> None:
        record = self.users.pop(supabase_user_id, None)
        if record:
            self.by_email.pop(record["email"].lower(), None)

    async def admin_update_user(self, supabase_user_id: str, **fields) -> dict:
        record = self.users[supabase_user_id]
        if "password" in fields:
            record["_password"] = fields["password"]
        record.update({k: v for k, v in fields.items() if k != "password"})
        return record


@pytest.fixture
async def supabase(monkeypatch):
    """Run the whole stack with Supabase as the provider."""
    stub = FakeSupabase()
    set_supabase_auth_client(stub)
    monkeypatch.setattr(settings, "AUTH_PROVIDER", "supabase")
    monkeypatch.setattr(settings, "SUPABASE_URL", "https://stub.supabase.co")
    monkeypatch.setattr(settings, "SUPABASE_ANON_KEY", "anon")
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "service")
    monkeypatch.setattr(settings, "SUPABASE_JWT_SECRET", JWT_SECRET)
    set_token_verifier(SupabaseJWTVerifier())
    yield stub
    set_supabase_auth_client(None)
    set_token_verifier(LocalJWTVerifier())


@pytest.fixture
async def estate(db):
    """A pre-created administrator and a legacy account with no Supabase link."""
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("appointments", "cases", "doctors", "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))

    admin = User(email=ADMIN, hashed_password=get_password_hash(PW),
                 role="admin", is_verified=True)
    legacy = User(email=LEGACY, hashed_password=get_password_hash(PW),
                  role="patient", is_verified=True)
    db.add_all([admin, legacy])
    await db.flush()
    db.add(Patient(id=legacy.id, first_name="Legacy", last_name="User",
                   phone="+9100", date_of_birth="1980-01-01", gender="male"))
    await db.commit()
    return {"admin": admin.id, "legacy": legacy.id}


async def signup_patient(client: AsyncClient, email=PATIENT):
    return await client.post("/api/v1/auth/signup/patient", json={
        "email": email, "password": PW, "profile": PATIENT_PROFILE})


async def signup_doctor(client: AsyncClient, email=DOCTOR):
    return await client.post("/api/v1/auth/signup/doctor", json={
        "email": email, "password": PW, "profile": DOCTOR_PROFILE})


async def login(client: AsyncClient, email, password=PW):
    return await client.post("/api/v1/auth/login",
                             json=await login_payload(email, password))


class TestSignup:
    async def test_patient_signup_creates_identity_and_local_profile(
        self, client, estate, supabase, db
    ):
        resp = await signup_patient(client)
        assert resp.status_code == 201, resp.text

        user = await db.scalar(select(User).where(User.email == PATIENT))
        assert user.role == "patient"
        # Linked to the identity, and the identity exists at the provider.
        assert user.supabase_user_id in supabase.users
        assert supabase.users[user.supabase_user_id]["email"] == PATIENT

    async def test_doctor_signup_is_pending_and_cannot_sign_in(
        self, client, estate, supabase, db
    ):
        assert (await signup_doctor(client)).status_code == 201

        user = await db.scalar(select(User).where(User.email == DOCTOR))
        doctor = await db.get(Doctor, user.id)
        assert doctor.verification_status == "pending"

        resp = await login(client, DOCTOR)
        assert resp.status_code == 403
        assert resp.json()["message"] == AWAITING_APPROVAL_MESSAGE

    async def test_duplicate_registration_is_refused(self, client, estate, supabase):
        assert (await signup_patient(client)).status_code == 201
        assert (await signup_patient(client)).status_code in (400, 409, 422)

    async def test_local_failure_does_not_strand_an_identity(
        self, client, estate, supabase, db
    ):
        """A rolled-back signup must leave the address free to register again."""
        resp = await client.post("/api/v1/auth/signup/patient", json={
            "email": "rollback@aronofy.com", "password": PW,
            "profile": {**PATIENT_PROFILE, "gender": "not-a-gender"}})
        assert resp.status_code == 422
        assert "rollback@aronofy.com" not in supabase.by_email

    async def test_signup_sends_verification_through_supabase(
        self, client, estate, supabase
    ):
        await signup_patient(client)
        assert PATIENT in supabase.verification_emails


class TestLogin:
    async def test_patient_login_returns_supabase_tokens(
        self, client, estate, supabase
    ):
        await signup_patient(client)
        resp = await login(client, PATIENT)
        assert resp.status_code == 200, resp.text

        body = resp.json()
        # Contract unchanged: same three fields the frontend already reads.
        assert set(body) >= {"access_token", "refresh_token", "token_type"}
        claims = jwt.decode(body["access_token"], JWT_SECRET,
                            algorithms=["HS256"], options={"verify_aud": False})
        assert claims["email"] == PATIENT

    async def test_wrong_password_is_refused(self, client, estate, supabase):
        await signup_patient(client)
        assert (await login(client, PATIENT, "wrong-password")).status_code == 401

    async def test_supabase_token_authorises_platform_apis(
        self, client, estate, supabase
    ):
        await signup_patient(client)
        token = (await login(client, PATIENT)).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        assert (await client.get("/api/v1/patient/dashboard",
                                 headers=headers)).status_code == 200
        assert (await client.get("/api/v1/doctor/dashboard",
                                 headers=headers)).status_code == 403
        assert (await client.get("/api/v1/admin/dashboard",
                                 headers=headers)).status_code == 403

    async def test_approved_doctor_can_sign_in(self, client, estate, supabase, db):
        await signup_doctor(client)
        user = await db.scalar(select(User).where(User.email == DOCTOR))
        doctor = await db.get(Doctor, user.id)
        doctor.verification_status = "verified"
        await db.commit()

        token = (await login(client, DOCTOR)).json()["access_token"]
        assert (await client.get(
            "/api/v1/doctor/dashboard",
            headers={"Authorization": f"Bearer {token}"})).status_code == 200

    async def test_suspended_account_cannot_sign_in(
        self, client, estate, supabase, db
    ):
        await signup_patient(client)
        user = await db.scalar(select(User).where(User.email == PATIENT))
        user.is_active = False
        await db.commit()

        assert (await login(client, PATIENT)).status_code == 403


class TestExistingUsersAreLinkedNotDuplicated:
    async def test_legacy_account_is_linked_by_email_on_first_sign_in(
        self, client, estate, supabase, db
    ):
        """
        Someone who has used the platform for months, then signs in through the
        new provider, must keep the same account — same id, same records.
        """
        supabase.seed(LEGACY, PW)
        before = await db.scalar(select(User).where(User.email == LEGACY))
        assert before.supabase_user_id is None
        original_id = before.id

        resp = await login(client, LEGACY)
        assert resp.status_code == 200, resp.text

        db.expunge_all()
        after = await db.scalar(select(User).where(User.email == LEGACY))
        assert after.id == original_id, "the existing account must be reused"
        assert after.supabase_user_id == supabase.by_email[LEGACY.lower()]

        total = await db.scalar(
            select(text("count(*)")).select_from(User).where(User.email == LEGACY))
        assert total == 1, "linking must never create a second account"

    async def test_identity_without_a_local_account_is_refused(
        self, client, estate, supabase
    ):
        """Authenticating at the provider does not make someone a user here."""
        supabase.seed("stranger@elsewhere.com", PW)
        assert (await login(client, "stranger@elsewhere.com")).status_code == 401


class TestSessionLifecycle:
    async def test_session_restore_uses_the_access_token(
        self, client, estate, supabase
    ):
        await signup_patient(client)
        token = (await login(client, PATIENT)).json()["access_token"]
        me = await client.get("/api/v1/auth/me",
                              headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["email"] == PATIENT

    async def test_refresh_rotates_and_keeps_working(self, client, estate, supabase):
        await signup_patient(client)
        first = (await login(client, PATIENT)).json()

        rotated = await client.post("/api/v1/auth/refresh",
                                    json={"refresh_token": first["refresh_token"]})
        assert rotated.status_code == 200, rotated.text
        assert rotated.json()["refresh_token"] != first["refresh_token"]

        assert (await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {rotated.json()['access_token']}"}
        )).status_code == 200

    async def test_rotated_refresh_token_cannot_be_replayed(
        self, client, estate, supabase
    ):
        await signup_patient(client)
        first = (await login(client, PATIENT)).json()
        await client.post("/api/v1/auth/refresh",
                          json={"refresh_token": first["refresh_token"]})
        replay = await client.post("/api/v1/auth/refresh",
                                   json={"refresh_token": first["refresh_token"]})
        assert replay.status_code == 401

    async def test_logout_revokes_the_supabase_session(
        self, client, estate, supabase
    ):
        await signup_patient(client)
        session = (await login(client, PATIENT)).json()

        out = await client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": session["refresh_token"]},
            headers={"Authorization": f"Bearer {session['access_token']}"})
        assert out.status_code == 200

        resumed = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": session["refresh_token"]})
        assert resumed.status_code == 401

    async def test_refresh_re_checks_local_authorisation(
        self, client, estate, supabase, db
    ):
        """Suspension between refreshes must not be papered over by a new token."""
        await signup_patient(client)
        session = (await login(client, PATIENT)).json()

        user = await db.scalar(select(User).where(User.email == PATIENT))
        user.is_active = False
        await db.commit()

        assert (await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": session["refresh_token"]})).status_code == 401


class TestTokenVerification:
    async def test_token_signed_with_another_key_is_refused(
        self, client, estate, supabase
    ):
        forged = jwt.encode(
            {"sub": str(uuid.uuid4()), "email": PATIENT, "aud": "authenticated",
             "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            "an-attackers-secret", algorithm="HS256")
        assert (await client.get(
            "/api/v1/patient/dashboard",
            headers={"Authorization": f"Bearer {forged}"})).status_code == 401

    async def test_expired_token_is_refused(self, client, estate, supabase, db):
        await signup_patient(client)
        user = await db.scalar(select(User).where(User.email == PATIENT))
        expired = jwt.encode(
            {"sub": user.supabase_user_id, "email": PATIENT,
             "aud": "authenticated",
             "exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
            JWT_SECRET, algorithm="HS256")
        assert (await client.get(
            "/api/v1/patient/dashboard",
            headers={"Authorization": f"Bearer {expired}"})).status_code == 401

    async def test_role_claim_in_a_supabase_token_is_ignored(
        self, client, estate, supabase, db
    ):
        """
        The provider controls the claims, so authorisation reads this database.
        A token asserting `role: admin` must gain nothing.
        """
        await signup_patient(client)
        user = await db.scalar(select(User).where(User.email == PATIENT))
        escalated = jwt.encode(
            {"sub": user.supabase_user_id, "email": PATIENT, "role": "admin",
             "app_metadata": {"role": "admin"}, "aud": "authenticated",
             "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            JWT_SECRET, algorithm="HS256")
        headers = {"Authorization": f"Bearer {escalated}"}

        assert (await client.get("/api/v1/admin/dashboard",
                                 headers=headers)).status_code == 403
        assert (await client.get("/api/v1/patient/dashboard",
                                 headers=headers)).status_code == 200

    async def test_unauthenticated_access_is_refused(self, client, estate, supabase):
        assert (await client.get("/api/v1/patient/dashboard")).status_code == 401


class TestRecovery:
    async def test_password_reset_is_sent_by_supabase(self, client, estate, supabase):
        await signup_patient(client)
        resp = await client.post("/api/v1/auth/forgot-password",
                                 json={"email": PATIENT})
        assert resp.status_code == 200
        assert PATIENT in supabase.recovery_emails

    async def test_reset_completes_and_changes_the_password(
        self, client, estate, supabase
    ):
        await signup_patient(client)
        recovery_token = (await login(client, PATIENT)).json()["access_token"]

        resp = await client.post("/api/v1/auth/reset-password", json={
            "token": recovery_token, "new_password": "brand-new-password-9"})
        assert resp.status_code == 200, resp.text

        assert (await login(client, PATIENT)).status_code == 401
        assert (await login(client, PATIENT,
                            "brand-new-password-9")).status_code == 200

    async def test_reset_with_a_bad_token_is_refused(self, client, estate, supabase):
        resp = await client.post("/api/v1/auth/reset-password", json={
            "token": "not-a-supabase-token", "new_password": "whatever-1234"})
        assert resp.status_code in (400, 422)

    async def test_forgot_password_does_not_disclose_registration(
        self, client, estate, supabase
    ):
        known = await client.post("/api/v1/auth/forgot-password",
                                  json={"email": LEGACY})
        unknown = await client.post("/api/v1/auth/forgot-password",
                                    json={"email": "nobody@nowhere.com"})
        assert known.status_code == unknown.status_code == 200
        assert known.text == unknown.text


class TestAdminPolicy:
    async def test_no_admin_signup_exists(self, client, estate, supabase):
        for path in ("/api/v1/auth/signup/admin", "/api/v1/admin/signup"):
            resp = await client.post(path, json={"email": "x@y.com", "password": PW})
            assert resp.status_code in (404, 405)

    async def test_pre_created_admin_signs_in_and_is_scoped(
        self, client, estate, supabase
    ):
        supabase.seed(ADMIN, PW)
        token = (await login(client, ADMIN)).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        assert (await client.get("/api/v1/admin/dashboard",
                                 headers=headers)).status_code == 200
        assert (await client.get("/api/v1/patient/dashboard",
                                 headers=headers)).status_code == 403
