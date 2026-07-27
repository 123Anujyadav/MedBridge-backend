"""
Medical records lifecycle: upload -> persist -> list -> refetch -> delete.

These cover the gap that made patient uploads useless in production. The only
upload route was `/shared/upload`, which wrote bytes to disk and returned
metadata without committing a row, so an uploaded document never appeared in the
records list, did not survive a reload, and could be neither downloaded nor
deleted. Each requirement gets its own assertion here so a regression names
exactly which one broke.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.core.security import get_password_hash
from app.models.patient import Patient
from app.models.report import Report
from app.models.user import User


@pytest.fixture
async def records_patient(db):
    """A verified patient with no reports, plus a second patient for isolation."""
    from sqlalchemy import text

    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("reports", "cases", "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    owner = User(
        email="records.owner@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="patient",
        is_verified=True,
    )
    intruder = User(
        email="records.intruder@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="patient",
        is_verified=True,
    )
    db.add_all([owner, intruder])
    await db.flush()

    db.add_all(
        [
            Patient(
                id=owner.id,
                first_name="Rita",
                last_name="Owner",
                phone="+1234500001",
                date_of_birth="1988-04-02",
                gender="female",
            ),
            Patient(
                id=intruder.id,
                first_name="Mal",
                last_name="Intruder",
                phone="+1234500002",
                date_of_birth="1990-01-01",
                gender="male",
            ),
        ]
    )
    await db.flush()
    return {"owner_id": owner.id, "intruder_id": intruder.id}


async def _login(client: AsyncClient, email: str) -> dict[str, str]:
    resp = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "password123"}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _pdf_upload(name: str = "bloodwork.pdf"):
    """A minimal file that satisfies the MIME and extension allowlists."""
    return {"file": (name, b"%PDF-1.4 synthetic test document", "application/pdf")}


@pytest.mark.asyncio
class TestMedicalRecordUpload:
    async def test_upload_persists_metadata_to_the_database(
        self, client: AsyncClient, records_patient, db
    ):
        headers = await _login(client, "records.owner@aronofy.com")

        resp = await client.post(
            "/api/v1/patient/records", files=_pdf_upload(), headers=headers
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()

        # The row itself, not just the HTTP echo.
        row = (
            await db.execute(
                select(Report).where(Report.id == uuid.UUID(body["id"]))
            )
        ).scalars().first()
        assert row is not None, "upload returned 201 but committed no report row"
        assert row.patient_id == records_patient["owner_id"]
        assert row.file_url, "no stored file reference was recorded"
        assert row.title == "bloodwork.pdf"

    async def test_uploaded_record_appears_in_the_list_immediately(
        self, client: AsyncClient, records_patient
    ):
        headers = await _login(client, "records.owner@aronofy.com")
        assert (await client.get("/api/v1/patient/reports", headers=headers)).json() == []

        await client.post(
            "/api/v1/patient/records", files=_pdf_upload(), headers=headers
        )

        listed = (await client.get("/api/v1/patient/reports", headers=headers)).json()
        assert len(listed) == 1
        assert listed[0]["title"] == "bloodwork.pdf"

    async def test_record_survives_a_refetch(
        self, client: AsyncClient, records_patient
    ):
        """Stands in for the browser reload: a fresh GET must still see it."""
        headers = await _login(client, "records.owner@aronofy.com")
        created = (
            await client.post(
                "/api/v1/patient/records", files=_pdf_upload(), headers=headers
            )
        ).json()

        again = await client.get(
            f"/api/v1/patient/reports/{created['id']}", headers=headers
        )
        assert again.status_code == 200
        assert again.json()["id"] == created["id"]

    async def test_rejects_a_disallowed_file_type(
        self, client: AsyncClient, records_patient
    ):
        headers = await _login(client, "records.owner@aronofy.com")
        resp = await client.post(
            "/api/v1/patient/records",
            files={"file": ("payload.exe", b"MZ\x90\x00", "application/x-msdownload")},
            headers=headers,
        )
        assert resp.status_code == 400


@pytest.mark.asyncio
class TestMedicalRecordDelete:
    async def test_delete_removes_the_database_row(
        self, client: AsyncClient, records_patient, db
    ):
        headers = await _login(client, "records.owner@aronofy.com")
        created = (
            await client.post(
                "/api/v1/patient/records", files=_pdf_upload(), headers=headers
            )
        ).json()

        resp = await client.delete(
            f"/api/v1/patient/reports/{created['id']}", headers=headers
        )
        assert resp.status_code == 200

        row = (
            await db.execute(
                select(Report).where(Report.id == uuid.UUID(created["id"]))
            )
        ).scalars().first()
        assert row is None, "delete returned 200 but the row is still present"

        listed = (await client.get("/api/v1/patient/reports", headers=headers)).json()
        assert listed == []

    async def test_cannot_delete_another_patients_record(
        self, client: AsyncClient, records_patient
    ):
        owner = await _login(client, "records.owner@aronofy.com")
        created = (
            await client.post(
                "/api/v1/patient/records", files=_pdf_upload(), headers=owner
            )
        ).json()

        intruder = await _login(client, "records.intruder@aronofy.com")
        resp = await client.delete(
            f"/api/v1/patient/reports/{created['id']}", headers=intruder
        )
        assert resp.status_code in (403, 404)

        # Still there for its owner.
        still = await client.get(
            f"/api/v1/patient/reports/{created['id']}", headers=owner
        )
        assert still.status_code == 200

    async def test_deleting_a_missing_record_is_not_found(
        self, client: AsyncClient, records_patient
    ):
        headers = await _login(client, "records.owner@aronofy.com")
        resp = await client.delete(
            f"/api/v1/patient/reports/{uuid.uuid4()}", headers=headers
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestMedicalRecordDownload:
    async def test_owner_can_download_an_uploaded_record(
        self, client: AsyncClient, records_patient
    ):
        headers = await _login(client, "records.owner@aronofy.com")
        created = (
            await client.post(
                "/api/v1/patient/records", files=_pdf_upload(), headers=headers
            )
        ).json()

        resp = await client.get(
            f"/api/v1/shared/reports/{created['id']}/download", headers=headers
        )
        assert resp.status_code == 200
        assert resp.content

    async def test_download_requires_authentication(
        self, client: AsyncClient, records_patient
    ):
        """
        The UI used a plain `<a href download>`, which sends no Authorization
        header. This asserts the route really does reject that, so the anchor
        cannot be reintroduced without a failing test.
        """
        headers = await _login(client, "records.owner@aronofy.com")
        created = (
            await client.post(
                "/api/v1/patient/records", files=_pdf_upload(), headers=headers
            )
        ).json()

        resp = await client.get(f"/api/v1/shared/reports/{created['id']}/download")
        assert resp.status_code in (401, 403)

    async def test_cannot_download_another_patients_record(
        self, client: AsyncClient, records_patient
    ):
        owner = await _login(client, "records.owner@aronofy.com")
        created = (
            await client.post(
                "/api/v1/patient/records", files=_pdf_upload(), headers=owner
            )
        ).json()

        intruder = await _login(client, "records.intruder@aronofy.com")
        resp = await client.get(
            f"/api/v1/shared/reports/{created['id']}/download", headers=intruder
        )
        assert resp.status_code in (403, 404)
