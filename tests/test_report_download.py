"""
Report download.

The route served `report.file_url` and nothing else. Only the consultation flow
writes that column, so every report produced by the AI intake pipeline, every
uploaded record and every lab result had no file — and the route answered 404
for them while the UI offered a Download button for all of them. In the live
database that was 51 of 63 reports.

The version branch of the very same handler had always rendered a missing
document on demand; only the live document did not.
"""

from __future__ import annotations

import os
import uuid

import pytest
from httpx import AsyncClient

from app.models.report import Report
from app.models.user import User
from app.models.patient import Patient
from app.core.security import get_password_hash
from conftest import login_payload

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def patient_ctx(db):
    """A patient with one report of each interesting shape."""
    suffix = uuid.uuid4().hex[:8]
    email = f"dl.pat.{suffix}@example.com"
    user = User(
        email=email,
        hashed_password=get_password_hash("password123"),
        role="patient",
        is_active=True,
        is_verified=True,
    )
    db.add(user)
    await db.flush()

    db.add(Patient(
        id=user.id, first_name="Dana", last_name="Lowe",
        phone="+911234500001",
        date_of_birth="1990-04-02", gender="female",
    ))
    await db.flush()

    reports = {}
    for key, kwargs in {
        # What the AI intake pipeline writes: narrative text, no file.
        "content_only": dict(
            type="ai_symptom_intake",
            title="AI Clinical Decision Support Report",
            summary="Symptom intake: persistent headache",
            content="### 1. Chief Complaint\nPersistent one-sided headache.",
        ),
        # A record with only a summary line.
        "summary_only": dict(
            type="lab_result", title="Complete Blood Count",
            summary="Slightly low iron levels.", content="",
        ),
        # The row exists but carries no document at all.
        "empty": dict(
            type="lab", title="Imaging Order", summary="", content="",
        ),
    }.items():
        report = Report(
            patient_id=user.id, patient_name="Dana Lowe",
            date="2026-08-01", status="ready", **kwargs,
        )
        db.add(report)
        await db.flush()
        # Plain ids, not ORM instances: the objects expire as the session is
        # used by the request under test.
        reports[key] = report.id

    await db.flush()

    body = await login_payload(email, "password123")
    return {"email": email, "user_id": user.id, "password_body": body, "reports": reports}


async def _auth(client: AsyncClient, ctx) -> dict:
    resp = await client.post("/api/v1/auth/login", json=ctx["password_body"])
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


class TestReportsWithoutAStoredFile:
    """The reports that used to 404."""

    async def test_a_content_only_report_downloads(self, client, patient_ctx):
        headers = await _auth(client, patient_ctx)
        rid = patient_ctx["reports"]["content_only"]

        resp = await client.get(f"/api/v1/shared/reports/{rid}/download", headers=headers)

        assert resp.status_code == 200, resp.text
        assert resp.content[:4] == b"%PDF"
        assert resp.headers["content-type"] == "application/pdf"

    async def test_a_summary_only_report_downloads(self, client, patient_ctx):
        headers = await _auth(client, patient_ctx)
        rid = patient_ctx["reports"]["summary_only"]

        resp = await client.get(f"/api/v1/shared/reports/{rid}/download", headers=headers)

        assert resp.status_code == 200
        assert resp.content[:4] == b"%PDF"

    async def test_the_filename_comes_from_the_report_title(self, client, patient_ctx):
        headers = await _auth(client, patient_ctx)
        rid = patient_ctx["reports"]["content_only"]

        resp = await client.get(f"/api/v1/shared/reports/{rid}/download", headers=headers)

        disposition = resp.headers["content-disposition"]
        assert "attachment" in disposition
        assert "AI" in disposition and ".pdf" in disposition

    async def test_inline_preview_is_not_an_attachment(self, client, patient_ctx):
        headers = await _auth(client, patient_ctx)
        rid = patient_ctx["reports"]["content_only"]

        resp = await client.get(
            f"/api/v1/shared/reports/{rid}/download?disposition=inline", headers=headers
        )

        assert resp.status_code == 200
        assert resp.headers["content-disposition"].startswith("inline")
        assert resp.headers["content-type"] == "application/pdf"

    async def test_rendering_is_repeatable(self, client, patient_ctx):
        """Two downloads of the same report both succeed and agree."""
        headers = await _auth(client, patient_ctx)
        rid = patient_ctx["reports"]["content_only"]
        url = f"/api/v1/shared/reports/{rid}/download"

        first = await client.get(url, headers=headers)
        second = await client.get(url, headers=headers)

        assert first.status_code == second.status_code == 200
        assert first.content[:4] == second.content[:4] == b"%PDF"


class TestNothingToServe:
    """
    A report with no document behind it.

    It must be a clean, typed refusal — the client turns this into
    "Report not available", and nothing internal may appear in the body.
    """

    async def test_an_empty_report_is_refused_cleanly(self, client, patient_ctx):
        headers = await _auth(client, patient_ctx)
        rid = patient_ctx["reports"]["empty"]

        resp = await client.get(f"/api/v1/shared/reports/{rid}/download", headers=headers)

        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] == "ENTITY_NOT_FOUND"
        assert body["success"] is False

    async def test_the_refusal_leaks_nothing_internal(self, client, patient_ctx):
        headers = await _auth(client, patient_ctx)
        rid = patient_ctx["reports"]["empty"]

        resp = await client.get(f"/api/v1/shared/reports/{rid}/download", headers=headers)

        text = resp.text.lower()
        for leak in ("traceback", "sqlalchemy", "select ", "/uploads/", "c:\\", "site-packages"):
            assert leak not in text, f"response leaked {leak!r}"

    async def test_an_unknown_report_is_still_404(self, client, patient_ctx):
        headers = await _auth(client, patient_ctx)
        resp = await client.get(
            f"/api/v1/shared/reports/{uuid.uuid4()}/download", headers=headers
        )
        assert resp.status_code == 404


class TestOnDemandRenderingDoesNotReleaseDrafts:
    """
    Rendering makes a document exist where none did, so it must not hand a
    patient one a clinician has not released.

    Before on-demand rendering, a patient asking for a report with no stored
    file simply got a 404 — and that accidental 404 was the only thing keeping
    a clinician's working draft out of their hands.
    """

    @pytest.mark.parametrize("draft_status", ["pending_review", "needs_revision", "rejected"])
    async def test_a_patient_cannot_pull_a_clinician_draft(
        self, client, patient_ctx, db, draft_status
    ):
        report = Report(
            patient_id=patient_ctx["user_id"], patient_name="Dana Lowe",
            type="ai_report", title="Draft Assessment",
            summary="Working notes.", content="Not finished yet.",
            date="2026-08-01", status=draft_status,
        )
        db.add(report)
        await db.flush()

        headers = await _auth(client, patient_ctx)
        resp = await client.get(
            f"/api/v1/shared/reports/{report.id}/download", headers=headers
        )

        assert resp.status_code == 404, (
            f"a {draft_status} draft was rendered and released to the patient"
        )

    @pytest.mark.parametrize("released_status", ["ready", "approved", "shared", "pending"])
    async def test_a_patient_receives_a_released_document(
        self, client, patient_ctx, db, released_status
    ):
        """
        `pending` is included deliberately: an AI intake summary is written
        from the patient's own words and its full text is already served to
        them by `GET /patient/reports/{id}`, so withholding the same text as a
        PDF would protect nothing.
        """
        report = Report(
            patient_id=patient_ctx["user_id"], patient_name="Dana Lowe",
            type="ai_symptom_intake", title="Intake Summary",
            summary="Reported headache.", content="Patient reports a headache.",
            date="2026-08-01", status=released_status,
        )
        db.add(report)
        await db.flush()

        headers = await _auth(client, patient_ctx)
        resp = await client.get(
            f"/api/v1/shared/reports/{report.id}/download", headers=headers
        )

        assert resp.status_code == 200
        assert resp.content[:4] == b"%PDF"


class TestContentTypeMatchesTheStoredFile:
    """
    A report is not always a PDF.

    `POST /patient/records` accepts JPEG, PNG, plain text and Word documents and
    stores the upload as the report's file. The route answered every download
    with `application/pdf` and a `.pdf` filename, so a patient who uploaded a
    scan got it back under a name that would not open.
    """

    @pytest.mark.parametrize(
        "extension,expected_media",
        [
            (".pdf", "application/pdf"),
            (".jpg", "image/jpeg"),
            (".png", "image/png"),
            (".txt", "text/plain"),
        ],
    )
    async def test_stored_upload_keeps_its_own_type(
        self, client, patient_ctx, db, tmp_path, extension, expected_media
    ):
        from app.core.upload import UPLOADS_ROOT

        reports_dir = os.path.join(UPLOADS_ROOT, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        name = f"upload_test_{uuid.uuid4().hex[:8]}{extension}"
        with open(os.path.join(reports_dir, name), "wb") as handle:
            handle.write(b"%PDF-1.4 stub" if extension == ".pdf" else b"stub-bytes")

        report = Report(
            patient_id=patient_ctx["user_id"], patient_name="Dana Lowe",
            type="upload", title="Uploaded Record", summary="", content="",
            date="2026-08-01", status="ready", file_url=f"/uploads/reports/{name}",
        )
        db.add(report)
        await db.flush()

        headers = await _auth(client, patient_ctx)
        resp = await client.get(
            f"/api/v1/shared/reports/{report.id}/download", headers=headers
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].split(";")[0] == expected_media
        assert extension in resp.headers["content-disposition"]

    async def test_generated_documents_are_still_pdf(self, client, patient_ctx):
        headers = await _auth(client, patient_ctx)
        rid = patient_ctx["reports"]["content_only"]

        resp = await client.get(f"/api/v1/shared/reports/{rid}/download", headers=headers)

        assert resp.headers["content-type"] == "application/pdf"
        assert ".pdf" in resp.headers["content-disposition"]


class TestAuthorisationIsUnchanged:
    """The fix must not widen who can read a report."""

    async def test_another_patient_is_refused(self, client, patient_ctx, db):
        suffix = uuid.uuid4().hex[:8]
        other_email = f"dl.other.{suffix}@example.com"
        other = User(
            email=other_email, hashed_password=get_password_hash("password123"),
            role="patient", is_active=True, is_verified=True,
        )
        db.add(other)
        await db.flush()
        db.add(Patient(
            id=other.id, first_name="Sam", last_name="Reed",
            phone="+911234500002",
            date_of_birth="1988-01-01", gender="male",
        ))
        await db.flush()

        body = await login_payload(other_email, "password123")
        resp = await client.post("/api/v1/auth/login", json=body)
        headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}

        rid = patient_ctx["reports"]["content_only"]
        denied = await client.get(f"/api/v1/shared/reports/{rid}/download", headers=headers)
        assert denied.status_code == 403

    async def test_anonymous_is_refused(self, client, patient_ctx):
        rid = patient_ctx["reports"]["content_only"]
        resp = await client.get(f"/api/v1/shared/reports/{rid}/download")
        assert resp.status_code in (401, 403)
