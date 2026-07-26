"""
Tests for bulk actions on the AI Reports list.

The properties that matter most here are the ones a happy-path test would miss:

* **Partial success.** A batch routinely contains items already in the target
  state and items the caller cannot touch. Neither may abort the rest, and both
  must be reported per-item rather than silently folded into a success count.
* **Ownership by intersection.** Another doctor's report in the selection is
  skipped with a non-committal reason — never acted on, and never confirmed to
  exist.
* **Audit completeness.** Every batch writes exactly one `audit_logs` row
  carrying the actor, action, affected ids and reason.
* **Constant query count**, asserted directly, because a regression to per-item
  updates would pass every content assertion.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import event, select, text

from app.core.security import get_password_hash
from app.models.audit import AuditLog
from app.models.case import Case
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.report import Report
from app.models.user import User

pytestmark = pytest.mark.asyncio

PW = "password123"
DOC_A = "bulk.doca@aronofy.com"
DOC_B = "bulk.docb@aronofy.com"
PAT_A = "bulk.pata@aronofy.com"
PAT_B = "bulk.patb@aronofy.com"


@pytest.fixture
async def estate(db):
    """
    Doctor A owns six reports in varied states; doctor B owns one.

    The varied states are what make the skip paths real: approving a batch that
    already contains an approved report must skip it, not double-apply.
    """
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("audit_logs", "intake_extracted_entities", "intake_sessions",
                  "notifications", "medications", "prescriptions", "appointments",
                  "reports", "symptoms", "cases", "doctors", "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    ids: dict[str, Any] = {}
    for key, email, role in (("doc_a", DOC_A, "doctor"), ("doc_b", DOC_B, "doctor"),
                             ("pat_a", PAT_A, "patient"), ("pat_b", PAT_B, "patient")):
        u = User(email=email, hashed_password=get_password_hash(PW), role=role, is_verified=True)
        db.add(u)
        await db.flush()
        ids[key] = u.id

    db.add(Doctor(id=ids["doc_a"], first_name="Asha", last_name="Rao", phone="+911",
                  specialty="Neurology", hospital_name="Central",
                  license_number="LIC-BULK-A", verification_status="verified"))
    db.add(Doctor(id=ids["doc_b"], first_name="Vikram", last_name="Sen", phone="+912",
                  specialty="Cardiology", hospital_name="East",
                  license_number="LIC-BULK-B", verification_status="verified"))
    db.add(Patient(id=ids["pat_a"], first_name="Meera", last_name="Iyer", phone="+913",
                   date_of_birth="1992-03-14", gender="female",
                   allergies=[], chronic_conditions=[], medications=[]))
    db.add(Patient(id=ids["pat_b"], first_name="Rahul", last_name="Nair", phone="+914",
                   date_of_birth="1988-06-02", gender="male",
                   allergies=[], chronic_conditions=[], medications=[]))
    await db.flush()

    def mk_case(pid, did, dname, spec):
        return Case(patient_id=pid, patient_name="P", patient_age=34,
                    patient_gender="female", doctor_id=did, doctor_name=dname,
                    specialty=spec, symptom_summary="s", urgency_level="medium",
                    status="routed", ai_extracted_symptoms=[], ai_confidence_score=0.0,
                    attachments=[], notes="")

    case_a = mk_case(ids["pat_a"], ids["doc_a"], "Dr. Asha Rao", "Neurology")
    case_b = mk_case(ids["pat_b"], ids["doc_b"], "Dr. Vikram Sen", "Cardiology")
    db.add_all([case_a, case_b])
    await db.flush()
    ids["case_a"], ids["case_b"] = case_a.id, case_b.id

    # Five pending + one already approved, to exercise the skip path.
    ids["pending"] = []
    for i in range(5):
        r = Report(patient_id=ids["pat_a"], case_id=case_a.id, patient_name="Meera Iyer",
                   type="ai_report", title=f"Pending {i}", summary="s", content="c",
                   date="2026-07-20", status="pending_review", ai_generated=True,
                   tags=[], vitals={})
        db.add(r)
        await db.flush()
        ids["pending"].append(r.id)

    already = Report(patient_id=ids["pat_a"], case_id=case_a.id, patient_name="Meera Iyer",
                     type="ai_report", title="Already Approved", summary="s", content="c",
                     date="2026-07-20", status="ready", ai_generated=True,
                     tags=[], vitals={})
    # Belongs to doctor B — must never be touched by doctor A.
    foreign = Report(patient_id=ids["pat_b"], case_id=case_b.id, patient_name="Rahul Nair",
                     type="ai_report", title="Foreign", summary="s", content="c",
                     date="2026-07-20", status="pending_review", ai_generated=True,
                     tags=[], vitals={})
    # No linked case — assign_specialist must skip it.
    orphan = Report(patient_id=ids["pat_a"], patient_name="Meera Iyer",
                    type="lab_result", title="Orphan", summary="s", content="c",
                    date="2026-07-20", status="pending_review", ai_generated=False,
                    tags=[], vitals={})
    db.add_all([already, foreign, orphan])
    await db.flush()
    ids["already"], ids["foreign"], ids["orphan"] = already.id, foreign.id, orphan.id

    await db.commit()
    return ids


async def _login(client: AsyncClient, email: str) -> dict[str, str]:
    r = await client.post("/api/v1/auth/login", json={"email": email, "password": PW})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def _bulk(client, headers, **payload):
    return await client.post("/api/v1/doctor/reports/bulk", json=payload, headers=headers)


class TestBulkOutcomes:
    async def test_approves_and_reports_totals(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await _bulk(client, headers, action="approve",
                        report_ids=[str(i) for i in estate["pending"]])
        assert r.status_code == 200, r.text
        job = r.json()

        assert job["status"] == "completed"
        assert job["total"] == 5
        assert job["completed"] == 5
        assert job["skipped"] == 0
        assert job["failed"] == 0

    async def test_partial_success_skips_no_ops(self, client, estate):
        """An already-approved report is skipped; the rest still complete."""
        headers = await _login(client, DOC_A)
        ids = [str(i) for i in estate["pending"]] + [str(estate["already"])]
        job = (await _bulk(client, headers, action="approve", report_ids=ids)).json()

        assert job["completed"] == 5
        assert job["skipped"] == 1
        assert job["failed"] == 0
        skipped = next(i for i in job["items"] if i["outcome"] == "skipped")
        assert skipped["report_id"] == str(estate["already"])
        assert "Already ready" in skipped["detail"]

    async def test_one_bad_item_does_not_abort_the_batch(self, client, estate):
        """A foreign id and a nonexistent id must not stop the real work."""
        headers = await _login(client, DOC_A)
        ids = ([str(i) for i in estate["pending"]]
               + [str(estate["foreign"]), str(uuid.uuid4())])
        job = (await _bulk(client, headers, action="approve", report_ids=ids)).json()

        assert job["completed"] == 5
        assert job["skipped"] == 2
        assert job["failed"] == 0

    async def test_duplicate_ids_counted_once(self, client, estate):
        headers = await _login(client, DOC_A)
        first = str(estate["pending"][0])
        job = (await _bulk(client, headers, action="approve",
                           report_ids=[first, first, first])).json()
        assert job["total"] == 1
        assert job["completed"] == 1


class TestPermissions:
    async def test_foreign_report_is_never_modified(self, client, estate, db):
        headers = await _login(client, DOC_A)
        await _bulk(client, headers, action="approve", report_ids=[str(estate["foreign"])])

        db.expire_all()
        foreign = await db.get(Report, estate["foreign"])
        assert foreign.status == "pending_review"

    async def test_foreign_report_skip_reason_is_non_committal(self, client, estate):
        """The reason must not confirm the report exists."""
        headers = await _login(client, DOC_A)
        job = (await _bulk(client, headers, action="approve",
                           report_ids=[str(estate["foreign"])])).json()
        detail = job["items"][0]["detail"]
        assert job["items"][0]["outcome"] == "skipped"
        assert "not accessible" in detail.lower()
        assert "exist" not in detail.lower()

    async def test_unknown_id_looks_identical_to_foreign_id(self, client, estate):
        headers = await _login(client, DOC_A)
        job = (await _bulk(client, headers, action="approve",
                           report_ids=[str(uuid.uuid4())])).json()
        assert job["items"][0]["detail"] == (
            await _bulk(client, await _login(client, DOC_A), action="approve",
                        report_ids=[str(estate["foreign"])])
        ).json()["items"][0]["detail"]

    async def test_patient_cannot_run_bulk_actions(self, client, estate):
        headers = await _login(client, PAT_A)
        r = await _bulk(client, headers, action="approve",
                        report_ids=[str(estate["pending"][0])])
        assert r.status_code == 403

    async def test_select_all_ids_is_scoped(self, client, estate):
        headers = await _login(client, DOC_B)
        r = await client.get("/api/v1/doctor/reports/ids", headers=headers)
        assert r.status_code == 200
        assert str(estate["pending"][0]) not in r.json()["report_ids"]
        assert str(estate["foreign"]) in r.json()["report_ids"]


class TestActionSafety:
    async def test_reject_requires_a_reason(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await _bulk(client, headers, action="reject",
                        report_ids=[str(estate["pending"][0])])
        assert r.status_code == 422

    async def test_archive_requires_a_reason(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await _bulk(client, headers, action="archive",
                        report_ids=[str(estate["pending"][0])])
        assert r.status_code == 422

    async def test_reject_with_reason_succeeds(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await _bulk(client, headers, action="reject", reason="Insufficient evidence",
                        report_ids=[str(estate["pending"][0])])
        assert r.status_code == 200
        assert r.json()["completed"] == 1

    async def test_assign_requires_target_doctor(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await _bulk(client, headers, action="assign_specialist",
                        report_ids=[str(estate["pending"][0])])
        assert r.status_code == 422

    async def test_assign_rejects_unverified_doctor(self, client, estate, db):
        unverified = await db.get(Doctor, estate["doc_b"])
        unverified.verification_status = "pending"
        await db.commit()

        headers = await _login(client, DOC_A)
        r = await _bulk(client, headers, action="assign_specialist",
                        target_doctor_id=str(estate["doc_b"]),
                        report_ids=[str(estate["pending"][0])])
        # The platform maps business-rule failures to 422; the code
        # distinguishes this from a schema rejection.
        assert r.status_code == 422
        assert r.json()["code"] == "BUSINESS_RULE_VALIDATION_FAILED"

    async def test_assign_skips_reports_without_a_case(self, client, estate):
        headers = await _login(client, DOC_A)
        job = (await _bulk(client, headers, action="assign_specialist",
                           target_doctor_id=str(estate["doc_b"]),
                           report_ids=[str(estate["orphan"])])).json()
        assert job["skipped"] == 1
        assert "No linked case" in job["items"][0]["detail"]

    async def test_assign_reassigns_the_case(self, client, estate, db):
        headers = await _login(client, DOC_A)
        job = (await _bulk(client, headers, action="assign_specialist",
                           target_doctor_id=str(estate["doc_b"]),
                           report_ids=[str(estate["pending"][0])])).json()
        assert job["completed"] == 1

        db.expire_all()
        case = await db.get(Case, estate["case_a"])
        assert case.doctor_id == estate["doc_b"]
        assert case.specialty == "Cardiology"

    async def test_no_bulk_prescribe_or_diagnose_action_exists(self, client, estate):
        """Unsafe operations must be rejected by the schema, not merely absent."""
        headers = await _login(client, DOC_A)
        for unsafe in ("prescribe", "finalize_diagnosis", "delete"):
            r = await _bulk(client, headers, action=unsafe,
                            report_ids=[str(estate["pending"][0])])
            assert r.status_code == 422, f"{unsafe} was accepted"


class TestFlagsAndStatuses:
    async def test_flag_follow_up_sets_the_flag(self, client, estate):
        headers = await _login(client, DOC_A)
        job = (await _bulk(client, headers, action="flag_follow_up",
                           report_ids=[str(estate["pending"][0])])).json()
        assert job["completed"] == 1

        cards = (await client.get("/api/v1/doctor/reports", headers=headers)).json()
        card = next(c for c in cards if c["id"] == str(estate["pending"][0]))
        assert card["flagged_for_follow_up"] is True
        assert "Follow-up Required" in {i["label"] for i in card["indicators"]}

    async def test_flag_is_idempotent(self, client, estate):
        headers = await _login(client, DOC_A)
        rid = [str(estate["pending"][0])]
        await _bulk(client, headers, action="flag_follow_up", report_ids=rid)
        job = (await _bulk(client, headers, action="flag_follow_up", report_ids=rid)).json()
        assert job["skipped"] == 1
        assert job["completed"] == 0

    async def test_mark_reviewed_then_remove_review_flag(self, client, estate):
        headers = await _login(client, DOC_A)
        rid = [str(estate["pending"][0])]

        await _bulk(client, headers, action="mark_reviewed", report_ids=rid)
        cards = (await client.get("/api/v1/doctor/reports", headers=headers)).json()
        assert next(c for c in cards if c["id"] == rid[0])["status"] == "reviewed"

        await _bulk(client, headers, action="remove_review_flag", report_ids=rid)
        cards = (await client.get("/api/v1/doctor/reports", headers=headers)).json()
        assert next(c for c in cards if c["id"] == rid[0])["status"] == "pending_review"

    async def test_archive_removes_from_default_filter(self, client, estate):
        headers = await _login(client, DOC_A)
        rid = [str(estate["pending"][0])]
        await _bulk(client, headers, action="archive", reason="Duplicate", report_ids=rid)

        archived = (await client.get("/api/v1/doctor/reports", headers=headers,
                                     params={"status": "archived"})).json()
        assert rid[0] in {c["id"] for c in archived}


class TestAuditLog:
    async def test_writes_one_audit_row_per_batch(self, client, estate, db):
        headers = await _login(client, DOC_A)
        await _bulk(client, headers, action="approve",
                    report_ids=[str(i) for i in estate["pending"]])

        rows = (await db.execute(
            select(AuditLog).where(AuditLog.resource == "ReportBatch")
        )).scalars().all()
        assert len(rows) == 1

        entry = rows[0]
        assert entry.user_id == estate["doc_a"]
        assert entry.user_role == "doctor"
        assert entry.action == "BULK_APPROVE_REPORTS"
        assert entry.ip_address

        details = json.loads(entry.details)
        assert details["completed"] == 5
        assert set(details["affected_report_ids"]) == {str(i) for i in estate["pending"]}

    async def test_records_the_reason(self, client, estate, db):
        headers = await _login(client, DOC_A)
        await _bulk(client, headers, action="reject", reason="Duplicate submission",
                    report_ids=[str(estate["pending"][0])])

        entry = (await db.execute(
            select(AuditLog).where(AuditLog.action == "BULK_REJECT_REPORTS")
        )).scalars().first()
        assert json.loads(entry.details)["reason"] == "Duplicate submission"

    async def test_partial_batch_is_flagged_warning_only_on_failure(
        self, client, estate, db
    ):
        """Skips are normal; only real failures downgrade the audit status."""
        headers = await _login(client, DOC_A)
        await _bulk(client, headers, action="approve",
                    report_ids=[str(estate["pending"][0]), str(estate["foreign"])])

        entry = (await db.execute(
            select(AuditLog).where(AuditLog.resource == "ReportBatch")
        )).scalars().first()
        assert entry.status == "success"
        assert json.loads(entry.details)["skipped"] == 1


class TestExport:
    async def test_csv_contains_only_owned_reports(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.post(
            "/api/v1/doctor/reports/bulk/export",
            params={"format": "csv"},
            json={"report_ids": [str(estate["pending"][0]), str(estate["foreign"])]},
            headers=headers,
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        body = r.text
        assert "Pending 0" in body
        assert "Foreign" not in body

    async def test_csv_excludes_report_bodies(self, client, estate):
        """A CSV of free-text clinical narrative is a PHI spill risk."""
        headers = await _login(client, DOC_A)
        r = await client.post(
            "/api/v1/doctor/reports/bulk/export", params={"format": "csv"},
            json={"report_ids": [str(estate["pending"][0])]}, headers=headers,
        )
        assert "content" not in r.text.splitlines()[0]

    async def test_pdf_bundle_reports_skips(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.post(
            "/api/v1/doctor/reports/bulk/export", params={"format": "pdf"},
            json={"report_ids": [str(i) for i in estate["pending"]]}, headers=headers,
        )
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/zip"
        # None of these fixtures carry a stored PDF.
        assert int(r.headers["x-skipped-count"]) == 5


class TestQueryEfficiency:
    async def test_query_count_does_not_grow_with_batch_size(self, client, estate, db):
        """
        The bulk-UPDATE guarantee, asserted directly.

        Every content assertion above would still pass if this regressed to a
        per-item load-and-save loop, so the query count is measured instead.
        """
        headers = await _login(client, DOC_A)
        from app.core.database import engine

        counter = {"n": 0}

        def _count(conn, cursor, statement, params, context, executemany):
            counter["n"] += 1

        sync_engine = engine.sync_engine
        event.listen(sync_engine, "before_cursor_execute", _count)
        try:
            counter["n"] = 0
            await _bulk(client, headers, action="flag_follow_up",
                        report_ids=[str(estate["pending"][0])])
            one = counter["n"]

            counter["n"] = 0
            await _bulk(client, headers, action="mark_reviewed",
                        report_ids=[str(i) for i in estate["pending"]])
            five = counter["n"]
        finally:
            event.remove(sync_engine, "before_cursor_execute", _count)

        assert five <= one, (
            f"a 5-report batch used {five} queries vs {one} for a single report — "
            "the bulk path is no longer batched"
        )
