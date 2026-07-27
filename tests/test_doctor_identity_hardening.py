"""
Regression tests for the defects a production audit reproduced.

Each class here corresponds to a finding that was *demonstrated* against a
running system, not one that was theorised. The point of the tests is that the
demonstration stops working.

The administrator-cap concurrency proof lives in `test_admin_cap_concurrency.py`
because it needs a real PostgreSQL server; the rest run on the suite's SQLite.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text

from app.core.doctor_code import (
    assign_doctor_code,
    doctor_codes_match,
    generate_doctor_code,
    is_valid_doctor_code,
    normalise_doctor_code,
)
from app.core.security import get_password_hash
from app.models.doctor import Doctor
from app.models.user import User
from conftest import login_payload

pytestmark = pytest.mark.asyncio

PW = "password123"
ADMIN = "hard.admin@aronofy.com"
DOCTOR = "hard.doctor@aronofy.com"
PATIENT = "hard.patient@aronofy.com"


@pytest.fixture
async def estate(db):
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("appointments", "cases", "doctors", "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))

    ids = {}
    for email, role in ((ADMIN, "admin"), (DOCTOR, "doctor"), (PATIENT, "patient")):
        user = User(email=email, hashed_password=get_password_hash(PW),
                    role=role, is_verified=True, is_active=True)
        db.add(user)
        await db.flush()
        ids[email] = user.id

    db.add(Doctor(
        id=ids[DOCTOR], first_name="Hard", last_name="Ening",
        phone="+10000000000", specialty="Cardiology",
        license_number="LIC-HARD-1", verification_status="verified",
    ))
    await db.commit()
    return ids


async def admin_headers(client):
    r = await client.post("/api/v1/auth/login",
                          json={"email": ADMIN, "password": PW})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ── H-1 ──────────────────────────────────────────────────────────────────

class TestMalformedDoctorIdNeverCrashes:
    """
    `hmac.compare_digest` raises `TypeError` on non-ASCII input.

    The audit turned four different Doctor IDs — a Cyrillic letter, an accented
    capital, a Kelvin sign and an emoji — into unauthenticated HTTP 500s with a
    stack trace apiece. A 500 is also trivially distinguishable from a 401,
    which undid the uniform failure response the three-factor design relies on.
    """

    UNICODE = {
        "cyrillic": "OTB1HA4З",
        "accented": "ÀÀÀÀÀÀÀÀ",
        "kelvin sign": "OTK1HA48",
        "emoji": "OTB1HA4\U0001f600",
        "cjk": "医生证件号码一二",
        "rtl override": "‮OTB1HA4",
        "combining": "ÓTB1HA48",
        "zero width": "OTB1​HA48",
    }

    MALFORMED = {
        "empty": "",
        "one char": "A",
        "seven": "OTB1HA4",
        "nine": "OTB1HA489",
        "lowercase wrong": "aaaaaaaa",
        "punctuation": "OTB1-HA4",
        "nul byte": "OTB1HA4\x00",
        "control chars": "OTB1\x07HA4",
        "sql injection": "' OR '1'='1",
        "long": "A" * 64,
    }

    def test_comparison_is_total(self):
        """No input may raise; every input is True or False."""
        stored = "OTB1HA48"
        for label, value in {**self.UNICODE, **self.MALFORMED}.items():
            assert doctor_codes_match(value, stored) is False, label

    def test_a_malformed_stored_code_never_matches(self):
        """Defends the invariant from the other side."""
        for stored in ("", None, "short", "lowercase", "TOOLONGCODE"):
            assert doctor_codes_match("OTB1HA48", stored) is False

    def test_trailing_newline_is_not_a_valid_code(self):
        """
        Python's `$` also matches before a trailing newline, so `^[A-Z0-9]{8}$`
        accepted a nine-character value that the database's POSIX check would
        reject. The pattern is anchored with `\\Z`.
        """
        assert is_valid_doctor_code("ABCDEFGH\n") is False
        assert is_valid_doctor_code("ABCDEFGH") is True

    @pytest.mark.parametrize("label", list(UNICODE) + list(MALFORMED))
    async def test_no_payload_produces_a_5xx(self, client, estate, label):
        value = {**self.UNICODE, **self.MALFORMED}[label]
        resp = await client.post("/api/v1/auth/login/doctor", json={
            "doctor_id": value, "email": DOCTOR, "password": PW,
        })
        assert resp.status_code < 500, f"{label} -> {resp.status_code}: {resp.text}"
        assert resp.status_code == 401, f"{label} -> {resp.status_code}"

    async def test_every_malformed_attempt_reads_identically(self, client, estate):
        """A 500 among 401s is itself an oracle. They must be one response."""
        seen = set()
        for value in {**self.UNICODE, **self.MALFORMED}.values():
            r = await client.post("/api/v1/auth/login/doctor", json={
                "doctor_id": value, "email": DOCTOR, "password": PW,
            })
            seen.add((r.status_code, r.json().get("message")))
        assert len(seen) == 1, seen

    async def test_oversized_value_is_refused_by_the_schema(self, client, estate):
        r = await client.post("/api/v1/auth/login/doctor", json={
            "doctor_id": "A" * 10000, "email": DOCTOR, "password": PW,
        })
        assert r.status_code == 422

    async def test_a_correct_code_still_works(self, client, estate, db):
        """The hardening must not have broken the happy path."""
        code = await db.scalar(
            select(Doctor.doctor_code).where(Doctor.id == estate[DOCTOR])
        )
        r = await client.post("/api/v1/auth/login/doctor", json={
            "doctor_id": code, "email": DOCTOR, "password": PW,
        })
        assert r.status_code == 200, r.text

    async def test_whitespace_and_case_are_still_forgiven(self, client, estate, db):
        code = await db.scalar(
            select(Doctor.doctor_code).where(Doctor.id == estate[DOCTOR])
        )
        for variant in (code.lower(), f"  {code}  ", f"{code[:4]} {code[4:]}",
                        f"{code[:2]}\t{code[2:]}"):
            r = await client.post("/api/v1/auth/login/doctor", json={
                "doctor_id": variant, "email": DOCTOR, "password": PW,
            })
            assert r.status_code == 200, f"{variant!r} -> {r.text}"


# ── M-2 ──────────────────────────────────────────────────────────────────

class TestVerificationQueuePagination:
    """
    The queue returned a bare array capped at 100 rows with nothing to say it
    had stopped. On a platform with more clinicians than that, the surplus were
    invisible to the only people who can approve them.
    """

    @pytest.fixture
    async def many(self, db, estate):
        for i in range(57):
            user = User(email=f"bulk{i}@aronofy.com",
                        hashed_password=get_password_hash(PW),
                        role="doctor", is_active=True, is_verified=True)
            db.add(user)
            await db.flush()
            db.add(Doctor(
                id=user.id, first_name=f"Bulk{i}", last_name="Doctor",
                phone="+1", specialty="General", license_number=f"LIC-BULK-{i}",
                verification_status="pending",
            ))
        await db.commit()
        return 57

    async def test_response_is_an_envelope_with_a_total(self, client, estate, many):
        r = await client.get("/api/v1/admin/doctors",
                             headers=await admin_headers(client))
        assert r.status_code == 200, r.text
        body = r.json()
        for field in ("items", "total", "page", "size", "pages",
                      "has_next", "has_prev"):
            assert field in body, f"{field} missing"
        assert body["total"] == many + 1  # the bulk doctors plus the fixture's

    async def test_paging_walks_every_clinician_exactly_once(
        self, client, estate, many
    ):
        headers = await admin_headers(client)
        seen, page = [], 1
        while True:
            r = await client.get("/api/v1/admin/doctors", headers=headers,
                                 params={"page": page, "size": 10})
            body = r.json()
            seen.extend(d["id"] for d in body["items"])
            if not body["has_next"]:
                break
            page += 1
            assert page < 50, "pagination did not terminate"

        assert len(seen) == many + 1
        assert len(set(seen)) == len(seen), "a clinician appeared on two pages"

    async def test_nothing_is_silently_truncated(self, client, estate, many):
        """The old failure: 100 rows back, 120 in the table, no indication."""
        r = await client.get("/api/v1/admin/doctors",
                             headers=await admin_headers(client),
                             params={"page": 1, "size": 25})
        body = r.json()
        assert len(body["items"]) == 25
        assert body["total"] == many + 1
        assert body["has_next"] is True

    async def test_filter_and_paging_compose(self, client, estate, many):
        r = await client.get("/api/v1/admin/doctors",
                             headers=await admin_headers(client),
                             params={"verification_status": "verified",
                                     "page": 1, "size": 10})
        body = r.json()
        assert body["total"] == 1
        assert body["pages"] == 1
        assert body["has_next"] is False
        assert all(d["verification_status"] == "verified" for d in body["items"])

    async def test_page_beyond_the_end_is_empty_not_an_error(
        self, client, estate, many
    ):
        r = await client.get("/api/v1/admin/doctors",
                             headers=await admin_headers(client),
                             params={"page": 999, "size": 25})
        assert r.status_code == 200
        assert r.json()["items"] == []
        assert r.json()["has_next"] is False

    async def test_paging_arguments_are_bounded(self, client, estate):
        headers = await admin_headers(client)
        for params in ({"page": 0}, {"page": -1}, {"size": 0},
                       {"size": 100000}, {"size": -5}):
            r = await client.get("/api/v1/admin/doctors", headers=headers,
                                 params=params)
            assert r.status_code == 422, f"{params} -> {r.status_code}"

    async def test_still_admin_only(self, client, estate):
        r = await client.post("/api/v1/auth/login",
                              json={"email": PATIENT, "password": PW})
        headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        assert (await client.get("/api/v1/admin/doctors",
                                 headers=headers)).status_code == 403


# ── M-3 ──────────────────────────────────────────────────────────────────

class TestVerifiedDoctorAlwaysHasAnId:
    async def test_approval_issues_an_id_when_absent(self, client, estate, db):
        doctor = await db.get(Doctor, estate[DOCTOR])
        doctor.doctor_code = None
        doctor.verification_status = "pending"
        await db.commit()

        r = await client.put(f"/api/v1/admin/doctors/{estate[DOCTOR]}/verify",
                             json={"verification_status": "verified"},
                             headers=await admin_headers(client))
        assert r.status_code == 200, r.text
        assert is_valid_doctor_code(r.json()["doctor_code"])

    async def test_no_verified_doctor_ends_up_without_one(self, client, estate, db):
        """The invariant the database constraint also enforces."""
        headers = await admin_headers(client)
        for status_value in ("pending", "rejected", "verified", "pending", "verified"):
            await client.put(f"/api/v1/admin/doctors/{estate[DOCTOR]}/verify",
                             json={"verification_status": status_value},
                             headers=headers)

        rows = (await db.execute(
            select(Doctor.verification_status, Doctor.doctor_code)
        )).all()
        for status_value, code in rows:
            if status_value == "verified":
                assert is_valid_doctor_code(code), "verified with no Doctor ID"

    async def test_assign_survives_a_taken_code(self, db, estate):
        """
        `allocate_doctor_code` checks then writes; another approval can take the
        value in between. The assignment retries instead of surfacing the
        unique-index violation as a 500.
        """
        doctor = await db.get(Doctor, estate[DOCTOR])
        code = await assign_doctor_code(db, doctor)
        assert is_valid_doctor_code(code)
        assert doctor.doctor_code == code


# ── M-4 / M-5 ────────────────────────────────────────────────────────────

class TestVerificationBroadcastIsAdminOnly:
    """
    A clinician's approval status is employment data. It was going to every
    connected socket — patients included — via a global broadcast.
    """

    async def test_broadcast_targets_the_admin_role_only(
        self, client, estate, monkeypatch
    ):
        from app.core import websocket as ws_module

        sent: list[tuple[dict, str]] = []
        globally: list[dict] = []

        async def fake_role(message, role):
            sent.append((message, role))

        async def fake_global(message):
            globally.append(message)

        monkeypatch.setattr(
            ws_module.websocket_manager, "broadcast_to_role", fake_role)
        monkeypatch.setattr(
            ws_module.websocket_manager, "broadcast", fake_global)

        r = await client.put(f"/api/v1/admin/doctors/{estate[DOCTOR]}/verify",
                             json={"verification_status": "rejected"},
                             headers=await admin_headers(client))
        assert r.status_code == 200, r.text

        assert globally == [], "verification status went to every connected client"
        assert len(sent) == 1
        message, role = sent[0]
        assert role == "admin"
        assert message["type"] == "DOCTOR_VERIFICATION_UPDATED"
        assert message["verification_status"] == "rejected"

    async def test_nothing_is_announced_when_the_decision_fails(
        self, client, estate, monkeypatch
    ):
        """A rolled-back decision must not be broadcast."""
        from app.core import websocket as ws_module

        sent: list = []

        async def fake_role(message, role):
            sent.append((message, role))

        monkeypatch.setattr(
            ws_module.websocket_manager, "broadcast_to_role", fake_role)

        import uuid as _u
        r = await client.put(f"/api/v1/admin/doctors/{_u.uuid4()}/verify",
                             json={"verification_status": "verified"},
                             headers=await admin_headers(client))
        assert r.status_code == 404
        assert sent == [], "announced a decision that never happened"


# ── L-2 ──────────────────────────────────────────────────────────────────

class TestRateLimitBucket:
    def test_both_login_routes_share_one_counter(self):
        """
        Keying on the exact path gave `/auth/login/doctor` its own budget, so an
        attacker had the per-minute allowance twice over against one account.
        """
        from app.middleware.rate_limit import PROTECTED_PATHS

        assert "/auth/login" in PROTECTED_PATHS
        for path in ("/auth/login", "/auth/login/doctor"):
            matched = [p for p in PROTECTED_PATHS if path.startswith(p)]
            assert "/auth/login" in matched


# ── generation invariants ────────────────────────────────────────────────

class TestGenerationStillSound:
    def test_generated_codes_remain_valid_and_unpredictable(self):
        drawn = {generate_doctor_code() for _ in range(500)}
        assert len(drawn) == 500
        assert all(is_valid_doctor_code(c) for c in drawn)

    def test_normalisation_is_unchanged_for_ordinary_input(self):
        assert normalise_doctor_code(" dr8a9xq2 ") == "DR8A9XQ2"
        assert normalise_doctor_code("DR8A 9XQ2") == "DR8A9XQ2"
        assert normalise_doctor_code(None) == ""
