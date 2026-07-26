"""
Regression tests for Patient Module PHI-isolation and upload-validation fixes.

Each test here corresponds to a vulnerability that was confirmed exploitable
against the live database during the Patient Module audit. They exist so the
holes cannot silently reopen.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from app.core.security import get_password_hash
from app.models.case import Case
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.report import Report
from app.models.user import User

pytestmark = pytest.mark.asyncio

PW = "password123"


@pytest.fixture
async def two_doctors_two_patients(db):
    """Patient A with a report and a case under doctor A; doctor B unrelated."""
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("reports", "cases", "doctors", "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    ids: dict[str, uuid.UUID] = {}
    for key, role in (("pA", "patient"), ("pB", "patient"), ("dA", "doctor"), ("dB", "doctor")):
        user = User(
            email=f"{key}.sec@aronofy.com",
            hashed_password=get_password_hash(PW),
            role=role,
            is_verified=True,
        )
        db.add(user)
        await db.flush()
        ids[key] = user.id

    for key in ("pA", "pB"):
        db.add(
            Patient(
                id=ids[key],
                first_name=key,
                last_name="Test",
                phone="+910000000000",
                date_of_birth="1990-01-01",
                gender="female",
            )
        )
    for key in ("dA", "dB"):
        db.add(
            Doctor(
                id=ids[key],
                first_name=key,
                last_name="Doc",
                phone="+919999999999",
                specialty="General Medicine",
                license_number=f"LIC-{key}",
                availability="available",
                verification_status="verified",
            )
        )
    await db.flush()

    report = Report(
        patient_id=ids["pA"],
        patient_name="pA Test",
        type="lab_result",
        title="PA CONFIDENTIAL BLOODWORK",
        summary="confidential",
        content="confidential",
        date="2026-07-26",
        status="ready",
        file_url="/uploads/pa-secret.pdf",
        tags=[],
    )
    db.add(report)
    db.add(
        Case(
            patient_id=ids["pA"],
            patient_name="pA Test",
            patient_age=30,
            patient_gender="female",
            doctor_id=ids["dA"],
            doctor_name="dA Doc",
            specialty="General Medicine",
            symptom_summary="test",
            urgency_level="low",
            status="routed",
            notes="",
        )
    )
    await db.commit()

    ids["report_id"] = report.id
    return ids


async def _login(client: AsyncClient, key: str) -> dict[str, str]:
    resp = await client.post(
        "/api/v1/auth/login", json={"email": f"{key}.sec@aronofy.com", "password": PW}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


class TestDoctorReportIsolation:
    """`GET /doctor/reports` previously returned every report to every doctor."""

    async def test_doctor_sees_only_their_own_patients_reports(
        self, client, two_doctors_two_patients
    ):
        headers = await _login(client, "dA")
        resp = await client.get("/api/v1/doctor/reports", headers=headers)
        assert resp.status_code == 200
        titles = [r["title"] for r in resp.json()]
        assert "PA CONFIDENTIAL BLOODWORK" in titles

    async def test_unrelated_doctor_sees_no_reports(
        self, client, two_doctors_two_patients
    ):
        headers = await _login(client, "dB")
        resp = await client.get("/api/v1/doctor/reports", headers=headers)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_unrelated_doctor_cannot_change_report_status(
        self, client, two_doctors_two_patients
    ):
        headers = await _login(client, "dB")
        report_id = two_doctors_two_patients["report_id"]
        resp = await client.put(
            f"/api/v1/doctor/reports/{report_id}/status",
            params={"status_str": "approved"},
            headers=headers,
        )
        assert resp.status_code == 403

    async def test_unrelated_doctor_cannot_edit_report_content(
        self, client, two_doctors_two_patients
    ):
        headers = await _login(client, "dB")
        report_id = two_doctors_two_patients["report_id"]
        resp = await client.put(
            f"/api/v1/doctor/reports/{report_id}/content",
            params={"content": "tampered"},
            headers=headers,
        )
        assert resp.status_code == 403

    async def test_treating_doctor_can_change_report_status(
        self, client, two_doctors_two_patients
    ):
        headers = await _login(client, "dA")
        report_id = two_doctors_two_patients["report_id"]
        resp = await client.put(
            f"/api/v1/doctor/reports/{report_id}/status",
            params={"status_str": "approved"},
            headers=headers,
        )
        assert resp.status_code == 200


class TestReportDownloadAuthorisation:
    """The download route previously performed no ownership check at all."""

    async def test_other_patient_is_denied(self, client, two_doctors_two_patients):
        headers = await _login(client, "pB")
        report_id = two_doctors_two_patients["report_id"]
        resp = await client.get(
            f"/api/v1/shared/reports/{report_id}/download", headers=headers
        )
        assert resp.status_code == 403

    async def test_unrelated_doctor_is_denied(self, client, two_doctors_two_patients):
        headers = await _login(client, "dB")
        report_id = two_doctors_two_patients["report_id"]
        resp = await client.get(
            f"/api/v1/shared/reports/{report_id}/download", headers=headers
        )
        assert resp.status_code == 403

    async def test_owner_passes_authorisation(self, client, two_doctors_two_patients):
        """
        Owner clears the authorisation gate; the file itself is absent in tests,
        so a 404 here proves access was granted and only the file was missing.
        """
        headers = await _login(client, "pA")
        report_id = two_doctors_two_patients["report_id"]
        resp = await client.get(
            f"/api/v1/shared/reports/{report_id}/download", headers=headers
        )
        assert resp.status_code == 404


class TestUploadValidation:
    """`POST /shared/upload` previously accepted any file of any type or size."""

    PDF = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF"

    async def test_accepts_a_valid_pdf(self, client, two_doctors_two_patients):
        headers = await _login(client, "pA")
        resp = await client.post(
            "/api/v1/shared/upload",
            files={"file": ("report.pdf", self.PDF, "application/pdf")},
            headers=headers,
        )
        assert resp.status_code == 201
        assert resp.json()["file_url"].startswith("/uploads/")

    async def test_rejects_executable(self, client, two_doctors_two_patients):
        headers = await _login(client, "pA")
        resp = await client.post(
            "/api/v1/shared/upload",
            files={"file": ("evil.exe", b"MZ\x90\x00", "application/x-msdownload")},
            headers=headers,
        )
        assert resp.status_code == 400

    async def test_rejects_disallowed_extension_with_spoofed_mime(
        self, client, two_doctors_two_patients
    ):
        """A hostile client can set any Content-Type, so extension is checked too."""
        headers = await _login(client, "pA")
        resp = await client.post(
            "/api/v1/shared/upload",
            files={"file": ("payload.sh", b"#!/bin/sh\nrm -rf /", "application/pdf")},
            headers=headers,
        )
        assert resp.status_code == 400

    async def test_rejects_oversized_file(self, client, two_doctors_two_patients):
        headers = await _login(client, "pA")
        resp = await client.post(
            "/api/v1/shared/upload",
            files={"file": ("big.pdf", b"x" * (11 * 1024 * 1024), "application/pdf")},
            headers=headers,
        )
        assert resp.status_code == 413

    async def test_stored_filename_does_not_use_client_input(
        self, client, two_doctors_two_patients
    ):
        """A traversal-style filename must not influence the path on disk."""
        headers = await _login(client, "pA")
        resp = await client.post(
            "/api/v1/shared/upload",
            files={"file": ("../../etc/passwd.pdf", self.PDF, "application/pdf")},
            headers=headers,
        )
        assert resp.status_code == 201
        assert ".." not in resp.json()["file_url"]
        assert "/uploads/" in resp.json()["file_url"]
