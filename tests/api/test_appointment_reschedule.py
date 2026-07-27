"""
Appointment rescheduling and the bookable-doctor directory.

Both were missing rather than broken. There was no reschedule route at all, and
no endpoint listing doctors a patient may book — the booking form carried five
hardcoded doctor UUIDs that existed in no environment, so every booking failed
on "Doctor not found".
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.core.security import get_password_hash
from app.models.appointment import Appointment
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.user import User


@pytest.fixture
async def booking_world(db):
    """One patient, one verified doctor, one unverified doctor, one appointment."""
    from sqlalchemy import text

    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("appointments", "reports", "cases", "doctors", "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    patient_user = User(
        email="appt.patient@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="patient",
        is_verified=True,
    )
    verified_user = User(
        email="appt.verified@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="doctor",
        is_verified=True,
    )
    unverified_user = User(
        email="appt.unverified@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="doctor",
        is_verified=True,
    )
    db.add_all([patient_user, verified_user, unverified_user])
    await db.flush()

    db.add(
        Patient(
            id=patient_user.id,
            first_name="Ana",
            last_name="Booker",
            phone="+1234500010",
            date_of_birth="1991-09-09",
            gender="female",
        )
    )
    db.add_all(
        [
            Doctor(
                id=verified_user.id,
                first_name="Vera",
                last_name="Verified",
                phone="+1234500011",
                specialty="Cardiology",
                hospital_name="MedBridge General",
                license_number="LIC-VERIFIED-1",
                consultation_fee=150.0,
                rating=4.9,
                verification_status="verified",
            ),
            Doctor(
                id=unverified_user.id,
                first_name="Uma",
                last_name="Unverified",
                phone="+1234500012",
                specialty="Neurology",
                hospital_name="MedBridge General",
                license_number="LIC-UNVERIFIED-1",
                verification_status="pending",
            ),
        ]
    )
    await db.flush()

    appt = Appointment(
        patient_id=patient_user.id,
        doctor_id=verified_user.id,
        patient_name="Ana Booker",
        doctor_name="Vera Verified",
        specialty="Cardiology",
        hospital_name="MedBridge General",
        date="2026-09-01",
        time="10:00",
        type="in_person",
        status="confirmed",
        reason="Follow-up",
        notes="",
    )
    db.add(appt)
    await db.flush()

    return {
        "patient_id": patient_user.id,
        "verified_doctor_id": verified_user.id,
        "unverified_doctor_id": unverified_user.id,
        "appointment_id": appt.id,
    }


async def _patient_headers(client: AsyncClient) -> dict[str, str]:
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "appt.patient@aronofy.com", "password": "password123"},
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.mark.asyncio
class TestBookableDoctorDirectory:
    async def test_lists_only_verified_doctors(
        self, client: AsyncClient, booking_world
    ):
        headers = await _patient_headers(client)
        resp = await client.get("/api/v1/patient/doctors", headers=headers)
        assert resp.status_code == 200, resp.text

        body = resp.json()
        ids = {d["id"] for d in body}
        assert str(booking_world["verified_doctor_id"]) in ids
        assert str(booking_world["unverified_doctor_id"]) not in ids, (
            "an unverified clinician must not be bookable"
        )

    async def test_returned_doctor_id_is_actually_bookable(
        self, client: AsyncClient, booking_world
    ):
        """The directory and the booking route must agree on doctor ids."""
        headers = await _patient_headers(client)
        listed = (
            await client.get("/api/v1/patient/doctors", headers=headers)
        ).json()
        doctor = listed[0]

        resp = await client.post(
            "/api/v1/patient/appointments",
            json={
                "doctor_id": doctor["id"],
                "specialty": doctor["specialty"],
                "hospital_name": doctor["hospital_name"],
                "date": "2026-10-05",
                "time": "14:30",
                "type": "in_person",
                "reason": "Chest discomfort",
            },
            headers=headers,
        )
        assert resp.status_code == 201, resp.text

    async def test_filters_by_specialty(self, client: AsyncClient, booking_world):
        headers = await _patient_headers(client)
        resp = await client.get(
            "/api/v1/patient/doctors", params={"specialty": "Cardiology"}, headers=headers
        )
        assert resp.status_code == 200
        assert all(d["specialty"] == "Cardiology" for d in resp.json())

    async def test_does_not_leak_license_numbers(
        self, client: AsyncClient, booking_world
    ):
        headers = await _patient_headers(client)
        body = (await client.get("/api/v1/patient/doctors", headers=headers)).text
        assert "LIC-VERIFIED" not in body
        assert "license" not in body.lower()


@pytest.mark.asyncio
class TestAppointmentReschedule:
    async def test_reschedule_persists_the_new_slot(
        self, client: AsyncClient, booking_world, db
    ):
        headers = await _patient_headers(client)
        appt_id = booking_world["appointment_id"]

        resp = await client.put(
            f"/api/v1/patient/appointments/{appt_id}/reschedule",
            json={"date": "2026-09-15", "time": "16:00"},
            headers=headers,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["date"] == "2026-09-15"
        assert resp.json()["time"] == "16:00"

        await db.refresh(
            (await db.execute(select(Appointment).where(Appointment.id == appt_id)))
            .scalars()
            .first()
        )
        row = (
            await db.execute(select(Appointment).where(Appointment.id == appt_id))
        ).scalars().first()
        assert (row.date, row.time) == ("2026-09-15", "16:00")

    async def test_reschedule_resets_status_to_scheduled(
        self, client: AsyncClient, booking_world
    ):
        """A doctor who confirmed the old slot has not agreed to the new one."""
        headers = await _patient_headers(client)
        resp = await client.put(
            f"/api/v1/patient/appointments/{booking_world['appointment_id']}/reschedule",
            json={"date": "2026-09-16", "time": "09:30"},
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "scheduled"

    async def test_rejects_an_already_booked_slot(
        self, client: AsyncClient, booking_world, db
    ):
        headers = await _patient_headers(client)

        # A second appointment occupies the target slot.
        blocker = Appointment(
            patient_id=booking_world["patient_id"],
            doctor_id=booking_world["verified_doctor_id"],
            patient_name="Ana Booker",
            doctor_name="Vera Verified",
            specialty="Cardiology",
            hospital_name="MedBridge General",
            date="2026-11-01",
            time="11:00",
            type="in_person",
            status="scheduled",
            reason="Other",
            notes="",
        )
        db.add(blocker)
        await db.flush()

        resp = await client.put(
            f"/api/v1/patient/appointments/{booking_world['appointment_id']}/reschedule",
            json={"date": "2026-11-01", "time": "11:00"},
            headers=headers,
        )
        # 422 is this platform's status for a business-rule violation; booking a
        # taken slot answers the same way.
        assert resp.status_code == 422, resp.text
        assert "already booked" in resp.text

    async def test_rejects_malformed_date_and_time(
        self, client: AsyncClient, booking_world
    ):
        headers = await _patient_headers(client)
        for payload in (
            {"date": "01-09-2026", "time": "16:00"},
            {"date": "2026-09-15", "time": "4pm"},
            {"date": "2026-09-15"},
        ):
            resp = await client.put(
                f"/api/v1/patient/appointments/{booking_world['appointment_id']}/reschedule",
                json=payload,
                headers=headers,
            )
            assert resp.status_code == 422, f"{payload} should be rejected"

    async def test_cannot_reschedule_a_completed_appointment(
        self, client: AsyncClient, booking_world, db
    ):
        headers = await _patient_headers(client)
        appt = (
            await db.execute(
                select(Appointment).where(
                    Appointment.id == booking_world["appointment_id"]
                )
            )
        ).scalars().first()
        appt.status = "completed"
        await db.flush()

        resp = await client.put(
            f"/api/v1/patient/appointments/{appt.id}/reschedule",
            json={"date": "2026-09-20", "time": "10:00"},
            headers=headers,
        )
        assert resp.status_code == 422
        assert "cannot be rescheduled" in resp.text

    async def test_cannot_reschedule_someone_elses_appointment(
        self, client: AsyncClient, booking_world, db
    ):
        other = User(
            email="appt.other@aronofy.com",
            hashed_password=get_password_hash("password123"),
            role="patient",
            is_verified=True,
        )
        db.add(other)
        await db.flush()
        db.add(
            Patient(
                id=other.id,
                first_name="Other",
                last_name="Patient",
                phone="+1234500099",
                date_of_birth="1995-05-05",
                gender="male",
            )
        )
        await db.flush()

        login = await client.post(
            "/api/v1/auth/login",
            json={"email": "appt.other@aronofy.com", "password": "password123"},
        )
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        resp = await client.put(
            f"/api/v1/patient/appointments/{booking_world['appointment_id']}/reschedule",
            json={"date": "2026-09-21", "time": "10:00"},
            headers=headers,
        )
        assert resp.status_code in (403, 404)

    async def test_missing_appointment_is_not_found(
        self, client: AsyncClient, booking_world
    ):
        headers = await _patient_headers(client)
        resp = await client.put(
            f"/api/v1/patient/appointments/{uuid.uuid4()}/reschedule",
            json={"date": "2026-09-22", "time": "10:00"},
            headers=headers,
        )
        assert resp.status_code == 404
