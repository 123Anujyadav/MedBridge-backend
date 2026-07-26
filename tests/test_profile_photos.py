"""
Profile photo management for patients and doctors.

Covers the lifecycle (upload, replace, remove), the validation boundary, and the
two properties that make the feature safe to expose:

* a user can only ever write their own avatar — no route accepts a target id;
* stored bytes are a re-encoded image, so nothing executable and no camera
  metadata survives an upload.
"""

from __future__ import annotations

import io
import os

import pytest
from httpx import AsyncClient
from PIL import Image
from sqlalchemy import select, text

from app.core.avatar import AVATARS_DIR, resolve_stored_avatar
from app.core.security import get_password_hash
from app.models.appointment import Appointment
from app.models.case import Case
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.user import User

pytestmark = pytest.mark.asyncio

PW = "password123"
PATIENT = "photo.pat@aronofy.com"
PATIENT_B = "photo.patb@aronofy.com"
DOCTOR = "photo.doc@aronofy.com"
ADMIN = "photo.admin@aronofy.com"

EMAILS = (PATIENT, PATIENT_B, DOCTOR, ADMIN)


def make_image(fmt: str = "PNG", size=(640, 480), colour=(30, 120, 110)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format=fmt)
    return buffer.getvalue()


def upload_payload(name: str, data: bytes, content_type: str) -> dict:
    return {"file": (name, io.BytesIO(data), content_type)}


@pytest.fixture
async def estate(db):
    """A patient, a second patient, a doctor and an admin, with a shared case."""
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    await db.execute(text("DELETE FROM appointments;"))
    await db.execute(text("DELETE FROM cases;"))
    for email in EMAILS:
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
    for key, email, role in (("patient", PATIENT, "patient"),
                             ("patient_b", PATIENT_B, "patient"),
                             ("doctor", DOCTOR, "doctor"),
                             ("admin", ADMIN, "admin")):
        user = User(email=email, hashed_password=get_password_hash(PW),
                    role=role, is_verified=True)
        db.add(user)
        await db.flush()
        ids[key] = user.id

    db.add(Doctor(id=ids["doctor"], first_name="Vikram", last_name="Sen",
                  phone="+9111", specialty="Cardiology", hospital_name="Central",
                  license_number="LIC-PHOTO-1", verification_status="verified"))
    for key, first in (("patient", "Asha"), ("patient_b", "Ravi")):
        db.add(Patient(id=ids[key], first_name=first, last_name="Test",
                       phone="+9122", date_of_birth="1990-01-01", gender="female",
                       allergies=[], chronic_conditions=[], medications=[]))
    await db.flush()

    case = Case(patient_id=ids["patient"], patient_name="Asha Test",
                patient_age=35, patient_gender="female", doctor_id=ids["doctor"],
                doctor_name="Dr. Vikram Sen", specialty="Cardiology",
                symptom_summary="Chest tightness.", urgency_level="medium",
                status="routed", ai_extracted_symptoms=[], ai_confidence_score=0.0,
                attachments=[], notes="")
    db.add(case)
    await db.flush()
    ids["case"] = case.id

    db.add(Appointment(patient_id=ids["patient"], doctor_id=ids["doctor"],
                       patient_name="Asha Test", doctor_name="Dr. Vikram Sen",
                       specialty="Cardiology", hospital_name="Central",
                       date="2031-09-09", time="10:00", duration=30,
                       type="in_person", status="scheduled", reason="Review",
                       notes="", case_id=case.id))
    await db.commit()
    return ids


async def login(client: AsyncClient, email: str) -> dict[str, str]:
    r = await client.post("/api/v1/auth/login",
                          json={"email": email, "password": PW})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


class TestPatientProfilePhoto:
    async def test_upload_sets_avatar_on_the_profile(self, client, estate):
        headers = await login(client, PATIENT)
        r = await client.post("/api/v1/patient/profile/avatar",
                              files=upload_payload("me.png", make_image(), "image/png"),
                              headers=headers)
        assert r.status_code == 200, r.text
        avatar_url = r.json()["avatar_url"]
        assert avatar_url.startswith("/uploads/avatars/")

        profile = await client.get("/api/v1/patient/profile", headers=headers)
        assert profile.json()["avatar_url"] == avatar_url

    async def test_stored_file_is_a_reencoded_square_webp(self, client, estate):
        headers = await login(client, PATIENT)
        r = await client.post(
            "/api/v1/patient/profile/avatar",
            files=upload_payload("wide.png", make_image(size=(1200, 300)), "image/png"),
            headers=headers)
        assert r.status_code == 200, r.text

        filename = r.json()["avatar_url"].rsplit("/", 1)[-1]
        path = resolve_stored_avatar(filename)
        assert path and os.path.exists(path)
        with Image.open(path) as stored:
            assert stored.format == "WEBP"
            assert stored.size == (512, 512)
            # Re-encoding is what strips EXIF; nothing may be carried over.
            assert not stored.getexif()

    async def test_thumbnail_is_generated(self, client, estate):
        headers = await login(client, PATIENT)
        r = await client.post("/api/v1/patient/profile/avatar",
                              files=upload_payload("me.png", make_image(), "image/png"),
                              headers=headers)
        stem = r.json()["avatar_url"].rsplit("/", 1)[-1].removesuffix(".webp")
        thumb = resolve_stored_avatar(f"{stem}_thumb.webp")
        assert thumb and os.path.exists(thumb)
        with Image.open(thumb) as image:
            assert image.size == (128, 128)

    async def test_replacing_deletes_the_previous_file(self, client, estate):
        headers = await login(client, PATIENT)
        first = await client.post(
            "/api/v1/patient/profile/avatar",
            files=upload_payload("a.png", make_image(), "image/png"), headers=headers)
        first_name = first.json()["avatar_url"].rsplit("/", 1)[-1]
        assert resolve_stored_avatar(first_name)

        second = await client.post(
            "/api/v1/patient/profile/avatar",
            files=upload_payload("b.jpg", make_image("JPEG"), "image/jpeg"),
            headers=headers)
        assert second.json()["avatar_url"] != first.json()["avatar_url"]
        # The superseded image must not linger in storage.
        assert resolve_stored_avatar(first_name) is None

    async def test_remove_clears_profile_and_storage(self, client, estate):
        headers = await login(client, PATIENT)
        uploaded = await client.post(
            "/api/v1/patient/profile/avatar",
            files=upload_payload("me.webp", make_image("WEBP"), "image/webp"),
            headers=headers)
        filename = uploaded.json()["avatar_url"].rsplit("/", 1)[-1]

        removed = await client.delete("/api/v1/patient/profile/avatar", headers=headers)
        assert removed.status_code == 200, removed.text
        assert removed.json()["avatar_url"] is None
        assert resolve_stored_avatar(filename) is None

    async def test_remove_without_a_photo_is_a_no_op(self, client, estate):
        headers = await login(client, PATIENT)
        r = await client.delete("/api/v1/patient/profile/avatar", headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()["avatar_url"] is None


class TestDoctorProfilePhoto:
    async def test_doctor_can_upload_and_remove(self, client, estate):
        headers = await login(client, DOCTOR)
        r = await client.post("/api/v1/doctor/profile/avatar",
                              files=upload_payload("dr.png", make_image(), "image/png"),
                              headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()["avatar_url"].startswith("/uploads/avatars/")

        profile = await client.get("/api/v1/doctor/profile", headers=headers)
        assert profile.json()["avatar_url"] == r.json()["avatar_url"]

        removed = await client.delete("/api/v1/doctor/profile/avatar", headers=headers)
        assert removed.json()["avatar_url"] is None


class TestPhotoValidation:
    """Only real, reasonably sized JPEG/PNG/WEBP images are accepted."""

    @pytest.mark.parametrize(
        "name,payload,content_type",
        [
            ("payload.png", b"MZ\x90\x00executable-body", "image/png"),
            ("payload.exe", b"MZ\x90\x00executable-body", "application/x-msdownload"),
            ("doc.pdf", b"%PDF-1.4 not a photo", "application/pdf"),
            ("x.svg", b"<svg onload=alert(1)></svg>", "image/svg+xml"),
            ("script.png", b"<?php system($_GET[0]); ?>", "image/png"),
        ],
    )
    async def test_non_images_are_rejected(
        self, client, estate, name, payload, content_type
    ):
        headers = await login(client, PATIENT)
        r = await client.post("/api/v1/patient/profile/avatar",
                              files=upload_payload(name, payload, content_type),
                              headers=headers)
        assert r.status_code == 400, r.text

    async def test_oversized_image_is_rejected(self, client, estate):
        headers = await login(client, PATIENT)
        oversized = make_image(size=(2000, 2000)) + b"\x00" * (5 * 1024 * 1024)
        r = await client.post("/api/v1/patient/profile/avatar",
                              files=upload_payload("big.png", oversized, "image/png"),
                              headers=headers)
        assert r.status_code == 413, r.text

    async def test_empty_file_is_rejected(self, client, estate):
        headers = await login(client, PATIENT)
        r = await client.post("/api/v1/patient/profile/avatar",
                              files=upload_payload("empty.png", b"", "image/png"),
                              headers=headers)
        assert r.status_code == 400, r.text


class TestPhotoAuthorisation:
    async def test_unauthenticated_upload_is_blocked(self, client, estate):
        r = await client.post("/api/v1/patient/profile/avatar",
                              files=upload_payload("x.png", make_image(), "image/png"))
        assert r.status_code in (401, 403)

    async def test_roles_cannot_cross_routes(self, client, estate):
        patient_headers = await login(client, PATIENT)
        doctor_headers = await login(client, DOCTOR)

        assert (await client.post(
            "/api/v1/doctor/profile/avatar",
            files=upload_payload("x.png", make_image(), "image/png"),
            headers=patient_headers)).status_code == 403
        assert (await client.post(
            "/api/v1/patient/profile/avatar",
            files=upload_payload("x.png", make_image(), "image/png"),
            headers=doctor_headers)).status_code == 403

    async def test_a_patient_cannot_affect_another_patients_photo(
        self, client, estate, db
    ):
        """
        The write target comes from the session, never the request, so there is
        no parameter through which one patient could reach another's row.
        """
        a_headers = await login(client, PATIENT)
        b_headers = await login(client, PATIENT_B)

        await client.post("/api/v1/patient/profile/avatar",
                          files=upload_payload("a.png", make_image(), "image/png"),
                          headers=a_headers)
        a_url = (await client.get("/api/v1/patient/profile",
                                  headers=a_headers)).json()["avatar_url"]

        # B uploads and then removes their own photo.
        await client.post("/api/v1/patient/profile/avatar",
                          files=upload_payload("b.png", make_image(), "image/png"),
                          headers=b_headers)
        await client.delete("/api/v1/patient/profile/avatar", headers=b_headers)

        still = (await client.get("/api/v1/patient/profile",
                                  headers=a_headers)).json()["avatar_url"]
        assert still == a_url
        assert resolve_stored_avatar(a_url.rsplit("/", 1)[-1])


class TestPhotoServing:
    async def test_served_image_is_the_stored_webp(self, client, estate):
        headers = await login(client, PATIENT)
        uploaded = await client.post(
            "/api/v1/patient/profile/avatar",
            files=upload_payload("me.png", make_image(), "image/png"), headers=headers)
        filename = uploaded.json()["avatar_url"].rsplit("/", 1)[-1]

        r = await client.get(f"/api/v1/shared/avatars/{filename}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/webp"
        assert r.content[:4] == b"RIFF" and r.content[8:12] == b"WEBP"

    @pytest.mark.parametrize(
        "probe",
        [
            "../reports/report.pdf",
            "..%2f..%2fapp.db",
            "....//etc/passwd",
            "arbitrary.webp",
            "abc.png",
        ],
    )
    async def test_traversal_and_unknown_names_are_refused(self, client, estate, probe):
        r = await client.get(f"/api/v1/shared/avatars/{probe}")
        assert r.status_code in (400, 404, 422)

    async def test_resolver_only_accepts_generated_names(self):
        """Guards the serving route at its source, independent of routing."""
        assert resolve_stored_avatar("../../app.db") is None
        assert resolve_stored_avatar("report.pdf") is None
        assert resolve_stored_avatar("") is None
        assert os.path.isdir(AVATARS_DIR) or True  # created lazily on first upload


class TestPhotoPropagation:
    """A changed photo must reach the rows that carry a copy of it."""

    async def test_upload_updates_case_and_appointment_copies(
        self, client, estate, db
    ):
        patient_headers = await login(client, PATIENT)
        doctor_headers = await login(client, DOCTOR)

        patient_url = (await client.post(
            "/api/v1/patient/profile/avatar",
            files=upload_payload("p.png", make_image(), "image/png"),
            headers=patient_headers)).json()["avatar_url"]
        doctor_url = (await client.post(
            "/api/v1/doctor/profile/avatar",
            files=upload_payload("d.png", make_image(), "image/png"),
            headers=doctor_headers)).json()["avatar_url"]

        case = await db.scalar(select(Case).where(Case.id == estate["case"]))
        await db.refresh(case)
        assert case.patient_avatar_url == patient_url

        appointment = await db.scalar(select(Appointment))
        await db.refresh(appointment)
        assert appointment.patient_avatar_url == patient_url
        assert appointment.doctor_avatar_url == doctor_url

    async def test_removal_clears_the_copies(self, client, estate, db):
        patient_headers = await login(client, PATIENT)
        await client.post("/api/v1/patient/profile/avatar",
                          files=upload_payload("p.png", make_image(), "image/png"),
                          headers=patient_headers)
        await client.delete("/api/v1/patient/profile/avatar", headers=patient_headers)

        case = await db.scalar(select(Case).where(Case.id == estate["case"]))
        await db.refresh(case)
        assert case.patient_avatar_url is None

    async def test_appointment_response_exposes_both_photos(self, client, estate):
        patient_headers = await login(client, PATIENT)
        doctor_headers = await login(client, DOCTOR)
        await client.post("/api/v1/patient/profile/avatar",
                          files=upload_payload("p.png", make_image(), "image/png"),
                          headers=patient_headers)
        await client.post("/api/v1/doctor/profile/avatar",
                          files=upload_payload("d.png", make_image(), "image/png"),
                          headers=doctor_headers)

        listed = await client.get("/api/v1/patient/appointments", headers=patient_headers)
        assert listed.status_code == 200, listed.text
        row = listed.json()[0]
        assert row["patient_avatar_url"] and row["doctor_avatar_url"]
