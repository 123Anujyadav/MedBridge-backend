"""
Regression guards for defects found during end-to-end platform verification.

Each test pins a bug that was reachable from the running product:

* **Token rotation returned an already-revoked token.** Access and refresh
  tokens carried only `sub`, `type` and a whole-second `exp`, so two tokens
  minted for one user inside the same second encoded identically. The refresh
  flow blacklists the presented token by its own string, so an identical
  replacement came back dead and the session ended on the next refresh.
* **Two live appointments could occupy one clinician slot.** The service checks
  for a conflict and then inserts, which cannot hold when requests overlap.
* **System health reported an unreachable Redis as "online".** The resilient
  client signals failure by returning False rather than raising, and the
  monitor only guarded against an exception.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from sqlalchemy import func, select, text

from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_password_hash,
)
from app.models.appointment import Appointment
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.user import User
from app.schemas.patient_api import AppointmentCreateRequest
from app.services.admin import admin_service
from app.services.appointment import appointment_service
from app.core.exceptions import BusinessRuleValidationException

pytestmark = pytest.mark.asyncio

PW = "password123"
SLOT_DATE = "2031-05-14"
SLOT_TIME = "09:30"


class TestTokenIdentity:
    """Every issued token must be distinct, or rotation revokes its successor."""

    async def test_refresh_tokens_minted_together_are_distinct(self):
        subject = "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"
        first = create_refresh_token(subject=subject)
        second = create_refresh_token(subject=subject)

        assert first != second, (
            "two refresh tokens for one subject encoded identically; the refresh "
            "flow would hand back a token it had just blacklisted"
        )

    async def test_access_tokens_minted_together_are_distinct(self):
        subject = "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"
        assert create_access_token(subject=subject) != create_access_token(subject=subject)

    async def test_tokens_stay_decodable_and_keep_their_claims(self):
        subject = "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"
        payload = decode_token(create_refresh_token(subject=subject))

        assert payload["sub"] == subject
        assert payload["type"] == "refresh"
        assert payload["jti"], "a unique token id is what makes each token distinct"

    async def test_identical_claims_still_differ_by_token_id(self):
        """Same subject, same expiry second — only the jti separates them."""
        subject = "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"
        expires = timedelta(days=7)
        a = decode_token(create_refresh_token(subject=subject, expires_delta=expires))
        b = decode_token(create_refresh_token(subject=subject, expires_delta=expires))

        assert a["jti"] != b["jti"]

    async def test_rotation_returns_a_token_that_still_works(self, client, db):
        """The end-to-end shape of the bug: refresh, then refresh again."""
        email = "rotate.regression@aronofy.com"
        db.add(User(email=email, hashed_password=get_password_hash(PW),
                    role="patient", is_verified=True))
        await db.commit()

        login = await client.post("/api/v1/auth/login",
                                  json={"email": email, "password": PW})
        assert login.status_code == 200, login.text

        first = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": login.json()["refresh_token"]})
        assert first.status_code == 200, first.text

        # Immediately reuse what the server just issued.
        second = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": first.json()["refresh_token"]})
        assert second.status_code == 200, (
            "the refresh token just issued was already revoked: " + second.text
        )


SLOT_EMAILS = ("slot.doc@aronofy.com", "slot.pata@aronofy.com",
               "slot.patb@aronofy.com")


@pytest.fixture
async def booking_estate(db):
    """One doctor and two patients, with no appointments on the books."""
    # The test database persists across tests, so this estate is rebuilt from
    # scratch each time rather than colliding with the previous test's rows.
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    await db.execute(text("DELETE FROM appointments;"))
    for email in SLOT_EMAILS:
        await db.execute(
            text("DELETE FROM doctors WHERE id IN "
                 "(SELECT id FROM users WHERE email = :e)"), {"e": email})
        await db.execute(
            text("DELETE FROM patients WHERE id IN "
                 "(SELECT id FROM users WHERE email = :e)"), {"e": email})
        await db.execute(text("DELETE FROM users WHERE email = :e"), {"e": email})
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    ids: dict[str, object] = {}
    for key, email, role in (("doctor", SLOT_EMAILS[0], "doctor"),
                             ("patient_a", SLOT_EMAILS[1], "patient"),
                             ("patient_b", SLOT_EMAILS[2], "patient")):
        user = User(email=email, hashed_password=get_password_hash(PW),
                    role=role, is_verified=True)
        db.add(user)
        await db.flush()
        ids[key] = user.id

    db.add(Doctor(id=ids["doctor"], first_name="Vikram", last_name="Sen",
                  phone="+9111", specialty="Cardiology", hospital_name="Central",
                  license_number="LIC-SLOT-1", verification_status="verified"))
    for key, first in (("patient_a", "Asha"), ("patient_b", "Ravi")):
        db.add(Patient(id=ids[key], first_name=first, last_name="Test",
                       phone="+9122", date_of_birth="1990-01-01",
                       gender="female", allergies=[], chronic_conditions=[],
                       medications=[]))
    await db.commit()
    return ids


def _booking(doctor_id, time: str = SLOT_TIME) -> AppointmentCreateRequest:
    return AppointmentCreateRequest(
        doctor_id=doctor_id, specialty="Cardiology",
        hospital_name="Central", date=SLOT_DATE, time=time,
        type="in_person", reason="Chest tightness",
    )


class TestAppointmentSlotIntegrity:
    """One live booking per clinician slot, enforced by the database."""

    async def test_same_patient_cannot_book_one_slot_twice(self, db, booking_estate):
        req = _booking(booking_estate["doctor"])
        await appointment_service.book_appointment(db, booking_estate["patient_a"], req)
        await db.commit()

        with pytest.raises(BusinessRuleValidationException):
            await appointment_service.book_appointment(
                db, booking_estate["patient_a"], req)

    async def test_second_patient_cannot_take_an_occupied_slot(self, db, booking_estate):
        req = _booking(booking_estate["doctor"])
        await appointment_service.book_appointment(db, booking_estate["patient_a"], req)
        await db.commit()

        with pytest.raises(BusinessRuleValidationException):
            await appointment_service.book_appointment(
                db, booking_estate["patient_b"], req)

    async def test_database_refuses_a_duplicate_the_service_check_missed(
        self, db, booking_estate
    ):
        """
        The guarantee must not depend on the pre-insert check.

        Rows are added directly, bypassing `book_appointment`, so this fails if
        the constraint is ever dropped and only the application check remains.
        """
        from sqlalchemy.exc import IntegrityError

        common = dict(
            doctor_id=booking_estate["doctor"], doctor_name="Dr. Vikram Sen",
            specialty="Cardiology", hospital_name="Central", date=SLOT_DATE,
            time="11:45", duration=30, type="in_person", status="scheduled",
            reason="Direct insert", notes="",
        )
        db.add(Appointment(patient_id=booking_estate["patient_a"],
                           patient_name="Asha Test", **common))
        await db.commit()

        db.add(Appointment(patient_id=booking_estate["patient_b"],
                           patient_name="Ravi Test", **common))
        with pytest.raises(IntegrityError):
            await db.commit()
        await db.rollback()

    async def test_cancelled_slot_can_be_rebooked(self, db, booking_estate):
        req = _booking(booking_estate["doctor"], time="14:00")
        appt = await appointment_service.book_appointment(
            db, booking_estate["patient_a"], req)
        await db.commit()

        await appointment_service.cancel_appointment(
            db, booking_estate["patient_a"], appt.id)
        await db.commit()

        # Releasing a slot must make it available again, not permanently burn it.
        rebooked = await appointment_service.book_appointment(
            db, booking_estate["patient_b"], req)
        await db.commit()
        assert rebooked.status == "scheduled"

    async def test_only_one_active_row_survives_per_slot(self, db, booking_estate):
        req = _booking(booking_estate["doctor"], time="16:20")
        await appointment_service.book_appointment(db, booking_estate["patient_a"], req)
        await db.commit()
        with pytest.raises(BusinessRuleValidationException):
            await appointment_service.book_appointment(
                db, booking_estate["patient_b"], req)
        await db.rollback()

        count = await db.scalar(
            select(func.count(Appointment.id))
            .where(Appointment.doctor_id == booking_estate["doctor"])
            .where(Appointment.date == SLOT_DATE)
            .where(Appointment.time == "16:20")
            .where(Appointment.status.in_(["scheduled", "confirmed", "in_progress"]))
        )
        assert count == 1


class TestSystemHealthReporting:
    """An unreachable dependency must be reported as down."""

    async def test_monitor_reports_unreachable_redis_as_offline(self, db):
        class DownRedis:
            async def ping(self) -> bool:
                # ResilientRedisClient returns False rather than raising, so
                # every other call site can fall back to the in-memory store.
                return False

        status = await admin_service.get_system_status(db, DownRedis())
        assert status["redis"]["status"] == "offline"

    async def test_monitor_reports_reachable_redis_as_online(self, db):
        class UpRedis:
            async def ping(self) -> bool:
                return True

        status = await admin_service.get_system_status(db, UpRedis())
        assert status["redis"]["status"] == "online"
        assert status["database"]["status"] == "online"
