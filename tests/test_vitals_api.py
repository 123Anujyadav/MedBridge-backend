"""
Tests for the vitals and adherence APIs backing the dashboard charts.

The central guarantee under test: with no readings the API returns empty
collections. A fabricated vital sign is indistinguishable from a real one once
it reaches a clinician, so "no data" must never be filled in.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from app.core.security import get_password_hash
from app.models.patient import Patient
from app.models.user import User
from conftest import login_payload

pytestmark = pytest.mark.asyncio

PW = "password123"
EMAIL = "vitals.patient@aronofy.com"
OTHER = "vitals.other@aronofy.com"


@pytest.fixture
async def vitals_patient(db):
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("vital_readings", "medications", "prescriptions", "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    ids = {}
    for key, email in (("main", EMAIL), ("other", OTHER)):
        user = User(
            email=email,
            hashed_password=get_password_hash(PW),
            role="patient",
            is_verified=True,
        )
        db.add(user)
        await db.flush()
        ids[key] = user.id
        db.add(
            Patient(
                id=user.id,
                first_name=key,
                last_name="Test",
                phone="+910000000000",
                date_of_birth="1990-01-01",
                gender="female",
                # Height is present so BMI is computable for the main patient.
                height=170.0 if key == "main" else None,
            )
        )
    await db.commit()
    return ids


async def _login(client: AsyncClient, email: str) -> dict[str, str]:
    resp = await client.post(
        "/api/v1/auth/login", json=await login_payload(email, PW)
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


class TestEmptyState:
    """No readings must yield empty arrays, never placeholder values."""

    async def test_dashboard_is_empty_without_readings(self, client, vitals_patient):
        headers = await _login(client, EMAIL)
        resp = await client.get("/api/v1/patient/vitals/dashboard", headers=headers)

        assert resp.status_code == 200
        body = resp.json()
        assert body["series"] == []
        assert body["adherence"] == []
        assert body["has_vitals_data"] is False
        assert body["has_adherence_data"] is False
        assert body["latest"] == {}

    async def test_list_is_empty_without_readings(self, client, vitals_patient):
        headers = await _login(client, EMAIL)
        resp = await client.get("/api/v1/patient/vitals", headers=headers)
        assert resp.status_code == 200
        assert resp.json() == []


class TestRecordingReadings:
    async def test_records_and_classifies(self, client, vitals_patient):
        headers = await _login(client, EMAIL)
        resp = await client.post(
            "/api/v1/patient/vitals",
            json={"type": "heart_rate", "value": 72, "unit": "bpm"},
            headers=headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["value"] == 72
        assert body["status"] == "normal"

    async def test_flags_out_of_band_reading(self, client, vitals_patient):
        headers = await _login(client, EMAIL)
        resp = await client.post(
            "/api/v1/patient/vitals",
            json={"type": "oxygen_saturation", "value": 88, "unit": "%"},
            headers=headers,
        )
        assert resp.status_code == 201
        assert resp.json()["status"] == "critical"

    @pytest.mark.parametrize(
        "payload",
        [
            {"type": "heart_rate", "value": 9999, "unit": "bpm"},   # implausible
            {"type": "heart_rate", "value": 70, "unit": "mmHg"},    # wrong unit
            {"type": "not_a_vital", "value": 1, "unit": "x"},       # unknown type
        ],
    )
    async def test_rejects_invalid_readings(self, client, vitals_patient, payload):
        headers = await _login(client, EMAIL)
        resp = await client.post(
            "/api/v1/patient/vitals", json=payload, headers=headers
        )
        assert resp.status_code == 422


class TestChartSeries:
    async def test_new_readings_appear_in_the_series(self, client, vitals_patient):
        headers = await _login(client, EMAIL)
        for kind, value, unit in [
            ("blood_pressure_systolic", 128, "mmHg"),
            ("blood_pressure_diastolic", 82, "mmHg"),
            ("heart_rate", 72, "bpm"),
            ("weight", 68.5, "kg"),
        ]:
            resp = await client.post(
                "/api/v1/patient/vitals",
                json={"type": kind, "value": value, "unit": unit},
                headers=headers,
            )
            assert resp.status_code == 201

        resp = await client.get("/api/v1/patient/vitals/dashboard", headers=headers)
        body = resp.json()
        assert body["has_vitals_data"] is True
        assert len(body["series"]) == 1

        point = body["series"][0]
        # Keys must match the Recharts `dataKey` props already in the page.
        assert "day" in point and "date" in point
        assert point["systolic"] == 128
        assert point["diastolic"] == 82
        assert point["heartRate"] == 72
        assert point["weight"] == 68.5

    async def test_bmi_computed_only_when_height_known(self, client, vitals_patient):
        headers = await _login(client, EMAIL)
        await client.post(
            "/api/v1/patient/vitals",
            json={"type": "weight", "value": 68.5, "unit": "kg"},
            headers=headers,
        )
        resp = await client.get("/api/v1/patient/vitals/dashboard", headers=headers)
        # 68.5 kg at 1.70 m -> 23.7
        assert resp.json()["series"][0]["bmi"] == 23.7

    async def test_bmi_omitted_when_height_unknown(self, client, vitals_patient):
        """Without a height, BMI is left null rather than estimated."""
        headers = await _login(client, OTHER)
        await client.post(
            "/api/v1/patient/vitals",
            json={"type": "weight", "value": 68.5, "unit": "kg"},
            headers=headers,
        )
        resp = await client.get("/api/v1/patient/vitals/dashboard", headers=headers)
        assert resp.json()["series"][0]["bmi"] is None

    async def test_unrecorded_measures_are_null_not_zero(self, client, vitals_patient):
        """A missing measure must be null; zero would plot as a real reading."""
        headers = await _login(client, EMAIL)
        await client.post(
            "/api/v1/patient/vitals",
            json={"type": "heart_rate", "value": 72, "unit": "bpm"},
            headers=headers,
        )
        point = (
            await client.get("/api/v1/patient/vitals/dashboard", headers=headers)
        ).json()["series"][0]
        assert point["heartRate"] == 72
        assert point["systolic"] is None
        assert point["temperature"] is None


class TestAccessControl:
    async def test_requires_authentication(self, client, vitals_patient):
        assert (await client.get("/api/v1/patient/vitals/dashboard")).status_code == 401

    async def test_readings_are_scoped_to_the_caller(self, client, vitals_patient):
        """One patient's readings must never surface in another's chart."""
        main = await _login(client, EMAIL)
        await client.post(
            "/api/v1/patient/vitals",
            json={"type": "heart_rate", "value": 72, "unit": "bpm"},
            headers=main,
        )

        other = await _login(client, OTHER)
        resp = await client.get("/api/v1/patient/vitals", headers=other)
        assert resp.json() == []

        dashboard = await client.get(
            "/api/v1/patient/vitals/dashboard", headers=other
        )
        assert dashboard.json()["series"] == []

    async def test_doctors_cannot_use_the_patient_route(self, client, vitals_patient, db):
        doctor = User(
            email="vitals.doc@aronofy.com",
            hashed_password=get_password_hash(PW),
            role="doctor",
            is_verified=True,
        )
        db.add(doctor)
        await db.flush()
        # An approved clinician, so this proves the patient route rejects the
        # doctor *role* rather than incidentally rejecting an unapproved account.
        from app.models.doctor import Doctor

        db.add(Doctor(
            id=doctor.id, first_name="Vitals", last_name="Doc", phone="+9100",
            specialty="Cardiology", license_number="LIC-VITALS-1",
            verification_status="verified",
        ))
        await db.commit()

        resp = await client.post(
            "/api/v1/auth/login",
            json=await login_payload("vitals.doc@aronofy.com", PW),
        )
        headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
        assert (
            await client.get("/api/v1/patient/vitals/dashboard", headers=headers)
        ).status_code == 403
