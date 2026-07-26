"""
Tests for the clinical document lifecycle: versions, preview and comparison.

The guarantees that matter most, in order of how much damage a regression does:

* **Nothing is overwritten.** A revision appends. An approved version is still
  readable, byte-for-byte, after ten more revisions.
* **AI never supersedes a clinician.** An AI regeneration lands as `ai_draft`;
  the doctor-approved version it followed is untouched and still present.
* **Restore appends, it does not rewind.** Restoring v1 creates a new version
  carrying v1's content — v1 itself, and everything after it, survives.
* **Identical content is not a new version**, so the system does not accumulate
  duplicate revisions or re-render identical PDFs.
* **Preview is the real document**, served from the same route as the download.
* **Patients see only what was shared with them.**
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text

from app.core.security import get_password_hash
from app.models.case import Case
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.report import Report
from app.models.report_version import ReportVersion
from app.models.user import User

pytestmark = pytest.mark.asyncio

PW = "password123"
DOC_A = "ver.doca@aronofy.com"
DOC_B = "ver.docb@aronofy.com"
PAT_A = "ver.pata@aronofy.com"
PAT_B = "ver.patb@aronofy.com"


@pytest.fixture
async def estate(db):
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("audit_logs", "report_versions", "notifications", "medications",
                  "prescriptions", "appointments", "reports", "symptoms",
                  "cases", "doctors", "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    ids: dict[str, Any] = {}
    for k, e, r in (("doc_a", DOC_A, "doctor"), ("doc_b", DOC_B, "doctor"),
                    ("pat_a", PAT_A, "patient"), ("pat_b", PAT_B, "patient")):
        u = User(email=e, hashed_password=get_password_hash(PW), role=r, is_verified=True)
        db.add(u); await db.flush(); ids[k] = u.id

    db.add(Doctor(id=ids["doc_a"], first_name="Asha", last_name="Rao", phone="+911",
                  specialty="Neurology", hospital_name="Central",
                  license_number="LIC-VER-A", verification_status="verified"))
    db.add(Doctor(id=ids["doc_b"], first_name="Vikram", last_name="Sen", phone="+912",
                  specialty="Cardiology", hospital_name="East",
                  license_number="LIC-VER-B", verification_status="verified"))
    db.add(Patient(id=ids["pat_a"], first_name="Meera", last_name="Iyer", phone="+913",
                   date_of_birth="1992-03-14", gender="female",
                   allergies=[], chronic_conditions=[], medications=[]))
    db.add(Patient(id=ids["pat_b"], first_name="Rahul", last_name="Nair", phone="+914",
                   date_of_birth="1988-06-02", gender="male",
                   allergies=[], chronic_conditions=[], medications=[]))
    await db.flush()

    case = Case(patient_id=ids["pat_a"], patient_name="Meera Iyer", patient_age=34,
                patient_gender="female", doctor_id=ids["doc_a"], doctor_name="Dr. Asha Rao",
                specialty="Neurology", symptom_summary="Severe headache.",
                urgency_level="medium", status="routed", ai_extracted_symptoms=["headache"],
                ai_confidence_score=0.8, attachments=[], notes="")
    db.add(case); await db.flush()
    ids["case"] = case.id

    report = Report(patient_id=ids["pat_a"], case_id=case.id, patient_name="Meera Iyer",
                    type="ai_report", title="Headache Report", summary="Assessment.",
                    content="Original body.", doctor_name="Dr. Asha Rao",
                    hospital_name="Central", date="2026-07-20", status="pending_review",
                    ai_generated=True, tags=[], vitals={})
    db.add(report); await db.flush()
    ids["report"] = report.id
    await db.commit()
    return ids


async def _login(client: AsyncClient, email: str) -> dict[str, str]:
    r = await client.post("/api/v1/auth/login", json={"email": email, "password": PW})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def _versions(client, headers, report_id) -> dict:
    r = await client.get(f"/api/v1/doctor/reports/{report_id}/versions", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


async def _create(client, headers, report_id, **payload):
    return await client.post(
        f"/api/v1/doctor/reports/{report_id}/versions", json=payload, headers=headers
    )


class TestInitialVersion:
    async def test_legacy_report_gets_version_one_from_its_own_content(
        self, client, estate
    ):
        """History starts from what the record says, not from a guess."""
        headers = await _login(client, DOC_A)
        body = await _versions(client, headers, estate["report"])

        assert body["total"] == 1
        v1 = body["versions"][0]
        assert v1["version_number"] == 1
        assert v1["author_type"] == "system"
        assert v1["is_latest"] is True

    async def test_initial_version_is_idempotent(self, client, estate):
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        second = await _versions(client, headers, estate["report"])
        assert second["total"] == 1


class TestAppendOnlyRevisions:
    async def test_new_version_does_not_alter_the_previous_one(self, client, estate, db):
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])

        r = await _create(client, headers, estate["report"],
                          diagnosis="Migraine", content="Revised body.",
                          description="Doctor revision")
        assert r.status_code == 201, r.text
        assert r.json()["version_number"] == 2

        v1 = (await db.execute(
            select(ReportVersion)
            .where(ReportVersion.report_id == estate["report"])
            .where(ReportVersion.version_number == 1)
        )).scalars().first()
        assert v1.content == "Original body."
        assert v1.diagnosis == ""

    async def test_identical_content_creates_no_new_version(self, client, estate):
        """A save that changes nothing must not manufacture a revision."""
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])

        payload = {"diagnosis": "Migraine", "content": "Revised body."}
        first = await _create(client, headers, estate["report"], **payload)
        second = await _create(client, headers, estate["report"], **payload)

        assert first.json()["version_number"] == 2
        assert second.json()["version_number"] == 2
        assert (await _versions(client, headers, estate["report"]))["total"] == 2

    async def test_identical_content_reuses_the_rendered_file(self, client, estate):
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        payload = {"diagnosis": "Migraine", "content": "Body."}
        first = await _create(client, headers, estate["report"], **payload)
        second = await _create(client, headers, estate["report"], **payload)
        assert first.json()["file_url"] == second.json()["file_url"]

    async def test_historical_versions_are_marked_read_only(self, client, estate):
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        await _create(client, headers, estate["report"], diagnosis="Migraine")

        body = await _versions(client, headers, estate["report"])
        by_number = {v["version_number"]: v for v in body["versions"]}
        assert by_number[2]["is_editable"] is True
        assert by_number[1]["is_editable"] is False


class TestAIVersioning:
    async def test_ai_regeneration_becomes_an_ai_draft(self, client, estate):
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        r = await _create(client, headers, estate["report"],
                          author_type="ai", content="AI redraft.",
                          description="AI regenerated the report")
        assert r.status_code == 201
        assert r.json()["author_type"] == "ai"
        assert r.json()["status"] == "ai_draft"
        assert r.json()["author_name"] == "AI Assistant"

    async def test_ai_draft_never_replaces_an_approved_version(self, client, estate, db):
        """The approved document must still exist, unchanged, afterwards."""
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])

        approved = await _create(client, headers, estate["report"],
                                 diagnosis="Confirmed migraine", status="approved",
                                 content="Doctor approved body.",
                                 approval_note="Reviewed and signed.")
        approved_number = approved.json()["version_number"]

        await _create(client, headers, estate["report"],
                      author_type="ai", content="AI rewrote everything.")

        stored = (await db.execute(
            select(ReportVersion)
            .where(ReportVersion.report_id == estate["report"])
            .where(ReportVersion.version_number == approved_number)
        )).scalars().first()
        assert stored.status == "approved"
        assert stored.content == "Doctor approved body."
        assert stored.approval_note == "Reviewed and signed."

    async def test_ai_authored_version_has_no_user_attribution(self, client, estate, db):
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        r = await _create(client, headers, estate["report"],
                          author_type="ai", content="AI body.")
        version = (await db.execute(
            select(ReportVersion)
            .where(ReportVersion.version_number == r.json()["version_number"])
        )).scalars().first()
        assert version.author_id is None
        assert version.author_type == "ai"


class TestRestore:
    async def test_restore_appends_rather_than_rewinding(self, client, estate, db):
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        await _create(client, headers, estate["report"], content="Second body.")
        await _create(client, headers, estate["report"], content="Third body.")

        r = await _create(client, headers, estate["report"], restore_from_version=2)
        assert r.status_code == 201
        restored = r.json()
        assert restored["version_number"] == 4
        assert restored["restored_from_version"] == 2

        # Every earlier version still exists, untouched.
        rows = (await db.execute(
            select(ReportVersion)
            .where(ReportVersion.report_id == estate["report"])
            .order_by(ReportVersion.version_number)
        )).scalars().all()
        assert [v.version_number for v in rows] == [1, 2, 3, 4]
        assert rows[1].content == "Second body."
        assert rows[2].content == "Third body."
        assert rows[3].content == "Second body."

    async def test_restore_blocked_once_shared_with_the_patient(self, client, estate, db):
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        await _create(client, headers, estate["report"], content="Second body.")

        report = await db.get(Report, estate["report"])
        report.status = "shared"
        await db.commit()

        r = await _create(client, headers, estate["report"], restore_from_version=1)
        assert r.status_code == 422
        assert r.json()["code"] == "BUSINESS_RULE_VALIDATION_FAILED"


class TestComparison:
    async def test_reports_field_level_changes(self, client, estate):
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        await _create(client, headers, estate["report"],
                      diagnosis="Migraine", content="Revised body.",
                      recommended_tests=["MRI"])

        r = await client.get(
            f"/api/v1/doctor/reports/{estate['report']}/versions/compare",
            params={"a": 1, "b": 2}, headers=headers,
        )
        assert r.status_code == 200, r.text
        body = r.json()

        assert body["identical"] is False
        fields = {f["field"]: f for f in body["fields"]}
        assert fields["diagnosis"]["change"] == "added"
        assert fields["diagnosis"]["new_value"] == "Migraine"
        assert fields["content"]["change"] == "modified"
        assert "MRI" in fields["recommended_tests"]["added_items"]

    async def test_attributes_changes_to_the_newer_versions_author(self, client, estate):
        """This is what separates an AI redraft from a doctor's edit."""
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        await _create(client, headers, estate["report"], author_type="ai",
                      content="AI body.")

        r = await client.get(
            f"/api/v1/doctor/reports/{estate['report']}/versions/compare",
            params={"a": 1, "b": 2}, headers=headers,
        )
        assert r.json()["changed_by_type"] == "ai"
        assert r.json()["changed_by_name"] == "AI Assistant"

    async def test_word_level_segments_mark_additions_and_removals(self, client, estate):
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        await _create(client, headers, estate["report"], content="Original revised body.")

        r = await client.get(
            f"/api/v1/doctor/reports/{estate['report']}/versions/compare",
            params={"a": 1, "b": 2}, headers=headers,
        )
        content = next(f for f in r.json()["fields"] if f["field"] == "content")
        kinds = {s["type"] for s in content["segments"]}
        assert "added" in kinds

    async def test_comparing_a_version_with_itself_is_rejected(self, client, estate):
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        r = await client.get(
            f"/api/v1/doctor/reports/{estate['report']}/versions/compare",
            params={"a": 1, "b": 1}, headers=headers,
        )
        assert r.status_code == 422

    async def test_unknown_version_is_404(self, client, estate):
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        r = await client.get(
            f"/api/v1/doctor/reports/{estate['report']}/versions/compare",
            params={"a": 1, "b": 99}, headers=headers,
        )
        assert r.status_code == 404


class TestPreviewAndDownload:
    async def test_preview_uses_the_same_route_as_download(self, client, estate):
        """One generator, one route — a preview cannot drift from the file."""
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        await _create(client, headers, estate["report"], diagnosis="Migraine")

        preview = await client.get(
            f"/api/v1/shared/reports/{estate['report']}/download",
            params={"disposition": "inline", "version": 2}, headers=headers,
        )
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "application/pdf"
        assert preview.headers["content-disposition"].startswith("inline")
        assert preview.content[:5] == b"%PDF-"

        download = await client.get(
            f"/api/v1/shared/reports/{estate['report']}/download",
            params={"version": 2}, headers=headers,
        )
        assert download.headers["content-disposition"].startswith("attachment")
        # Same version, same stored file: byte-identical.
        assert download.content == preview.content

    async def test_each_version_renders_its_own_document(self, client, estate):
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        await _create(client, headers, estate["report"], diagnosis="Migraine")

        v1 = await client.get(f"/api/v1/shared/reports/{estate['report']}/download",
                              params={"version": 1}, headers=headers)
        v2 = await client.get(f"/api/v1/shared/reports/{estate['report']}/download",
                              params={"version": 2}, headers=headers)
        assert v1.status_code == 200 and v2.status_code == 200
        assert v1.content != v2.content


class TestSecurity:
    async def test_other_doctor_cannot_read_versions(self, client, estate):
        headers = await _login(client, DOC_B)
        r = await client.get(f"/api/v1/doctor/reports/{estate['report']}/versions",
                             headers=headers)
        assert r.status_code == 403

    async def test_other_doctor_cannot_create_a_version(self, client, estate):
        headers = await _login(client, DOC_B)
        r = await _create(client, headers, estate["report"], content="Injected.")
        assert r.status_code == 403

    async def test_patient_cannot_reach_the_doctor_version_routes(self, client, estate):
        headers = await _login(client, PAT_A)
        r = await client.get(f"/api/v1/doctor/reports/{estate['report']}/versions",
                             headers=headers)
        assert r.status_code == 403

    async def test_patient_cannot_download_an_unshared_report(self, client, estate):
        """A draft is not the patient's document until it is released."""
        headers = await _login(client, PAT_A)
        r = await client.get(f"/api/v1/shared/reports/{estate['report']}/download",
                             headers=headers)
        assert r.status_code in (403, 404)

    async def test_other_patient_never_reaches_the_document(self, client, estate, db):
        report = await db.get(Report, estate["report"])
        report.status = "shared"
        await db.commit()

        headers = await _login(client, PAT_B)
        r = await client.get(f"/api/v1/shared/reports/{estate['report']}/download",
                             headers=headers)
        assert r.status_code == 403


class TestIntegrity:
    async def test_versions_belong_to_their_own_report(self, client, estate, db):
        headers = await _login(client, DOC_A)

        other = Report(patient_id=estate["pat_a"], case_id=estate["case"],
                       patient_name="Meera Iyer", type="ai_report",
                       title="Second Report", summary="s", content="Other body.",
                       doctor_name="Dr. Asha Rao", date="2026-07-21",
                       status="pending_review", ai_generated=True, tags=[], vitals={})
        db.add(other); await db.commit()

        await _versions(client, headers, estate["report"])
        await _create(client, headers, estate["report"], content="Report one revision.")
        await _versions(client, headers, other.id)

        first = await _versions(client, headers, estate["report"])
        second = await _versions(client, headers, other.id)

        assert first["total"] == 2
        assert second["total"] == 1
        assert "Report one revision" not in str(second["versions"])

    async def test_version_numbers_are_sequential_per_report(self, client, estate, db):
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        for i in range(3):
            await _create(client, headers, estate["report"], content=f"Body {i}.")

        rows = (await db.execute(
            select(ReportVersion.version_number)
            .where(ReportVersion.report_id == estate["report"])
            .order_by(ReportVersion.version_number)
        )).scalars().all()
        assert list(rows) == [1, 2, 3, 4]

    async def test_report_tracks_its_current_version(self, client, estate, db):
        headers = await _login(client, DOC_A)
        await _versions(client, headers, estate["report"])
        await _create(client, headers, estate["report"], content="Newer body.")

        db.expire_all()
        report = await db.get(Report, estate["report"])
        assert report.current_version == 2

    async def test_unknown_report_is_404(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.get(f"/api/v1/doctor/reports/{uuid.uuid4()}/versions",
                             headers=headers)
        assert r.status_code == 404
