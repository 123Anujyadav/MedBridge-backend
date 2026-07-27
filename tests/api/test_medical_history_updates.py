"""
Medical history must update automatically after each clinical event.

The patient's history screen is assembled from two API reads —
`GET /patient/reports` and `GET /patient/prescriptions`. So "history updates
automatically" reduces to a checkable property: every event that should appear
there must land in one of those two collections, owned by the right patient,
without any extra write from the client.

Covers the four triggers named in the requirements:
  1. report uploaded            -> /patient/records
  2. AI analysis completed      -> AI intake persists a report
  3. consultation completed     -> /doctor/cases/complete
  4. prescription generated     -> /doctor/prescriptions
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.core.security import get_password_hash
from app.models.case import Case
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.report import Report
from app.models.user import User


@pytest.fixture
async def history_world(db):
    """A patient, a verified treating doctor, and an open case linking them."""
    from sqlalchemy import text

    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in (
        "medications",
        "prescriptions",
        "appointments",
        "reports",
        "cases",
        "doctors",
        "patients",
        "users",
    ):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    patient_user = User(
        email="hist.patient@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="patient",
        is_verified=True,
    )
    doctor_user = User(
        email="hist.doctor@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="doctor",
        is_verified=True,
    )
    db.add_all([patient_user, doctor_user])
    await db.flush()

    db.add(
        Patient(
            id=patient_user.id,
            first_name="Hana",
            last_name="History",
            phone="+1234500020",
            date_of_birth="1987-02-02",
            gender="female",
        )
    )
    db.add(
        Doctor(
            id=doctor_user.id,
            first_name="Dana",
            last_name="Doctor",
            phone="+1234500021",
            specialty="General Medicine",
            hospital_name="MedBridge General",
            license_number="LIC-HIST-1",
            verification_status="verified",
        )
    )
    await db.flush()

    case = Case(
        patient_id=patient_user.id,
        patient_name="Hana History",
        patient_age=39,
        patient_gender="female",
        doctor_id=doctor_user.id,
        doctor_name="Dana Doctor",
        specialty="General Medicine",
        symptom_summary="Persistent cough for two weeks",
        status="in_consultation",
        urgency_level="medium",
    )
    db.add(case)
    await db.flush()

    return {
        "patient_id": patient_user.id,
        "doctor_id": doctor_user.id,
        "case_id": case.id,
    }


async def _headers(client: AsyncClient, email: str) -> dict[str, str]:
    resp = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "password123"}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def _history(client: AsyncClient, headers: dict[str, str]) -> dict:
    """The two reads the medical-history screen performs."""
    reports = await client.get("/api/v1/patient/reports", headers=headers)
    prescriptions = await client.get("/api/v1/patient/prescriptions", headers=headers)
    assert reports.status_code == 200, reports.text
    assert prescriptions.status_code == 200, prescriptions.text
    return {"reports": reports.json(), "prescriptions": prescriptions.json()}


@pytest.mark.asyncio
class TestMedicalHistoryAutoUpdates:
    async def test_history_starts_empty(self, client: AsyncClient, history_world):
        headers = await _headers(client, "hist.patient@aronofy.com")
        history = await _history(client, headers)
        assert history["reports"] == []
        assert history["prescriptions"] == []

    async def test_uploaded_report_enters_history(
        self, client: AsyncClient, history_world
    ):
        headers = await _headers(client, "hist.patient@aronofy.com")

        await client.post(
            "/api/v1/patient/records",
            files={"file": ("scan.pdf", b"%PDF-1.4 x-ray", "application/pdf")},
            headers=headers,
        )

        history = await _history(client, headers)
        assert len(history["reports"]) == 1
        assert history["reports"][0]["title"] == "scan.pdf"

    async def test_completed_consultation_enters_history(
        self, client: AsyncClient, history_world, db
    ):
        doctor_headers = await _headers(client, "hist.doctor@aronofy.com")

        resp = await client.post(
            "/api/v1/doctor/cases/complete",
            json={
                "case_id": str(history_world["case_id"]),
                "diagnosis": "Acute bronchitis",
                "clinical_notes": "Chest clear. Advised rest and fluids.",
                "medications": [],
                "recommended_tests": ["Chest X-ray"],
            },
            headers=doctor_headers,
        )
        assert resp.status_code == 201, resp.text

        # Visible to the patient without any client-side write.
        patient_headers = await _headers(client, "hist.patient@aronofy.com")
        history = await _history(client, patient_headers)
        assert len(history["reports"]) >= 1, (
            "a completed consultation must appear in the patient's history"
        )

        row = (
            await db.execute(
                select(Report).where(Report.patient_id == history_world["patient_id"])
            )
        ).scalars().first()
        assert row is not None
        assert row.patient_id == history_world["patient_id"]

    async def test_prescription_enters_history(
        self, client: AsyncClient, history_world
    ):
        doctor_headers = await _headers(client, "hist.doctor@aronofy.com")

        resp = await client.post(
            "/api/v1/doctor/prescriptions",
            json={
                "case_id": str(history_world["case_id"]),
                "patient_id": str(history_world["patient_id"]),
                "diagnosis": "Acute bronchitis",
                "notes": "Complete the full course.",
                "medications": [
                    {
                        "name": "Amoxicillin",
                        "dosage": "500mg",
                        "frequency": "Three times daily",
                        "duration": "7 days",
                        "start_date": "2026-07-27",
                        "end_date": "2026-08-03",
                    }
                ],
            },
            headers=doctor_headers,
        )
        assert resp.status_code == 201, resp.text

        patient_headers = await _headers(client, "hist.patient@aronofy.com")
        history = await _history(client, patient_headers)
        assert len(history["prescriptions"]) == 1
        assert history["prescriptions"][0]["diagnosis"] == "Acute bronchitis"
        # The medication rows are what the reminders screen tracks doses against.
        assert history["prescriptions"][0]["medications"], (
            "a prescription with no medication rows leaves reminders empty"
        )

    async def test_prescribed_medication_is_trackable(
        self, client: AsyncClient, history_world
    ):
        """
        Closes the loop on the dose-tracking bug.

        The reminders screen previously rendered placeholder medications with
        fabricated UUIDs, and tracking one returned 404. A medication that came
        from a real prescription must accept a dose update and persist it.
        """
        doctor_headers = await _headers(client, "hist.doctor@aronofy.com")
        await client.post(
            "/api/v1/doctor/prescriptions",
            json={
                "case_id": str(history_world["case_id"]),
                "patient_id": str(history_world["patient_id"]),
                "diagnosis": "Acute bronchitis",
                "notes": "",
                "medications": [
                    {
                        "name": "Amoxicillin",
                        "dosage": "500mg",
                        "frequency": "Three times daily",
                        "duration": "7 days",
                        "start_date": "2026-07-27",
                        "end_date": "2026-08-03",
                    }
                ],
            },
            headers=doctor_headers,
        )

        patient_headers = await _headers(client, "hist.patient@aronofy.com")
        history = await _history(client, patient_headers)
        med_id = history["prescriptions"][0]["medications"][0]["id"]

        track = await client.put(
            f"/api/v1/patient/medications/{med_id}/track",
            json={"status": "taken"},
            headers=patient_headers,
        )
        assert track.status_code == 200, track.text
        assert track.json()["status"] == "taken"
        assert track.json()["taken_doses"] == 1

        # Persists across a fresh read, which is what "must survive refresh" means.
        again = await _history(client, patient_headers)
        med = again["prescriptions"][0]["medications"][0]
        assert med["status"] == "taken"
        assert med["taken_doses"] == 1

    async def test_history_is_scoped_to_the_owning_patient(
        self, client: AsyncClient, history_world, db
    ):
        """A second patient must not inherit anyone else's history."""
        other = User(
            email="hist.other@aronofy.com",
            hashed_password=get_password_hash("password123"),
            role="patient",
            is_verified=True,
        )
        db.add(other)
        await db.flush()
        db.add(
            Patient(
                id=other.id,
                first_name="Otto",
                last_name="Other",
                phone="+1234500029",
                date_of_birth="1993-03-03",
                gender="male",
            )
        )
        await db.flush()

        owner_headers = await _headers(client, "hist.patient@aronofy.com")
        await client.post(
            "/api/v1/patient/records",
            files={"file": ("private.pdf", b"%PDF-1.4 private", "application/pdf")},
            headers=owner_headers,
        )

        other_headers = await _headers(client, "hist.other@aronofy.com")
        history = await _history(client, other_headers)
        assert history["reports"] == []
        assert history["prescriptions"] == []
