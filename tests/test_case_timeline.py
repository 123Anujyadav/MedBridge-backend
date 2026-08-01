"""
Tests for the case timeline and audit trail.

The properties under test, in order of how badly a regression would hurt:

* **Ownership.** A case history is the densest PHI surface in the product — it
  narrates the whole consultation. Only the case's patient, its assigned doctor
  and an admin may read it.
* **Case scoping.** Events are attached by foreign key. An event belonging to
  another case, or to no case, never appears on a case's timeline.
* **No fabrication.** A milestone with no timestamp behind it is omitted rather
  than dated by guesswork, and a value the backend never stored is NULL rather
  than an empty string.
* **Attribution.** AI, doctor, patient and system events are distinguishable,
  and recorded events carry the actor the platform authenticated — never one
  supplied by a caller.
* **Append-only.** Nothing in the API surface can edit or delete history.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text

from app.core.security import get_password_hash
from app.models.audit import AuditLog
from app.models.case import Case, Symptom
from app.models.doctor import Doctor
from app.models.intake import IntakeExtractedEntity, IntakeSessionRecord
from app.models.patient import Patient
from app.models.prescription import Prescription
from app.models.report import Report
from app.models.user import User
from app.services import clinical_review as review_module
from conftest import login_payload

pytestmark = pytest.mark.asyncio

PW = "password123"
DOC_A = "tl.doca@aronofy.com"
DOC_B = "tl.docb@aronofy.com"
PAT_A = "tl.pata@aronofy.com"
PAT_B = "tl.patb@aronofy.com"


class _FakeGroq:
    is_configured = False

    async def complete_json(self, **_):
        return {}


@pytest.fixture(autouse=True)
def no_groq(monkeypatch):
    monkeypatch.setattr(review_module, "get_groq_client", lambda: _FakeGroq())


@pytest.fixture
async def estate(db):
    """
    One patient with TWO cases under the SAME doctor.

    Both pass every ownership check, so only foreign-key scoping can keep their
    histories apart — which is exactly the failure mode worth testing.
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
                  license_number="LIC-TL-A", verification_status="verified"))
    db.add(Doctor(id=ids["doc_b"], first_name="Vikram", last_name="Sen", phone="+912",
                  specialty="Cardiology", hospital_name="East",
                  license_number="LIC-TL-B", verification_status="verified"))
    db.add(Patient(id=ids["pat_a"], first_name="Meera", last_name="Iyer", phone="+913",
                   date_of_birth="1992-03-14", gender="female",
                   allergies=[], chronic_conditions=[], medications=[]))
    db.add(Patient(id=ids["pat_b"], first_name="Rahul", last_name="Nair", phone="+914",
                   date_of_birth="1988-06-02", gender="male",
                   allergies=[], chronic_conditions=[], medications=[]))
    await db.flush()

    def mk(pid, did, dname, spec, summary):
        return Case(patient_id=pid, patient_name="P", patient_age=34,
                    patient_gender="female", doctor_id=did, doctor_name=dname,
                    specialty=spec, symptom_summary=summary, urgency_level="medium",
                    status="routed", ai_extracted_symptoms=[], ai_confidence_score=0.0,
                    attachments=[], notes="")

    headache = mk(ids["pat_a"], ids["doc_a"], "Dr. Asha Rao", "Neurology",
                  "Severe headache with photophobia.")
    ankle = mk(ids["pat_a"], ids["doc_a"], "Dr. Asha Rao", "Orthopedics",
               "Twisted ankle after a fall.")
    other = mk(ids["pat_b"], ids["doc_b"], "Dr. Vikram Sen", "Cardiology",
               "Chest tightness.")
    db.add_all([headache, ankle, other])
    await db.flush()
    ids.update(headache=headache.id, ankle=ankle.id, other=other.id)

    db.add(Symptom(case_id=headache.id, name="headache", severity="moderate",
                   duration="5 days", body_part="head"))

    intake = IntakeSessionRecord(
        session_key="tl-headache", patient_user_id=ids["pat_a"], status="routed",
        language="hinglish", intent="symptom_report", followup_rounds=2,
        overall_confidence=0.9, red_flags=["Thunderclap onset"],
        transcript="Sar dard hai.",
        medical_case_snapshot={
            "chief_complaint": "Severe headache",
            "summary_for_doctor": "HEADACHE_MARKER",
            "urgency": "high",
            "recommended_specialty": "Neurology",
        },
        routed_case_id=headache.id, routed_doctor_id=ids["doc_a"],
    )
    db.add(intake)
    await db.flush()
    db.add(IntakeExtractedEntity(
        session_id=intake.id, kind="symptom", value="headache", confidence=0.9,
        confidence_band="high", evidence_quote="sar dard", evidence_turn_index=0,
        was_accepted=True,
    ))

    for case, title in ((headache, "Headache Report"), (ankle, "Ankle Report")):
        db.add(Report(patient_id=ids["pat_a"], case_id=case.id,
                      patient_name="Meera Iyer", type="ai_report", title=title,
                      summary="s", content="c", doctor_name="Dr. Asha Rao",
                      date="2026-07-20", status="pending_review", ai_generated=True,
                      tags=[], vitals={}))

    db.add(Prescription(case_id=ankle.id, patient_id=ids["pat_a"],
                        patient_name="Meera Iyer", doctor_id=ids["doc_a"],
                        doctor_name="Dr. Asha Rao", diagnosis="ANKLE_RX_MARKER",
                        notes="", status="active"))
    await db.commit()
    return ids


async def _login(client: AsyncClient, email: str) -> dict[str, str]:
    r = await client.post("/api/v1/auth/login", json=await login_payload(email, PW))
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def _timeline(client, headers, case_id, **params) -> dict:
    r = await client.get("/api/v1/shared/timeline",
                         params={"case_id": str(case_id), **params}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


class TestAuthorization:
    async def test_assigned_doctor_may_read(self, client, estate):
        headers = await _login(client, DOC_A)
        assert (await _timeline(client, headers, estate["headache"]))["total"] > 0

    async def test_owning_patient_may_read(self, client, estate):
        headers = await _login(client, PAT_A)
        assert (await _timeline(client, headers, estate["headache"]))["total"] > 0

    async def test_unassigned_doctor_denied(self, client, estate):
        headers = await _login(client, DOC_B)
        r = await client.get("/api/v1/shared/timeline",
                             params={"case_id": str(estate["headache"])}, headers=headers)
        assert r.status_code == 403

    async def test_other_patient_denied(self, client, estate):
        headers = await _login(client, PAT_B)
        r = await client.get("/api/v1/shared/timeline",
                             params={"case_id": str(estate["headache"])}, headers=headers)
        assert r.status_code == 403

    async def test_unknown_case_404(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.get("/api/v1/shared/timeline",
                             params={"case_id": str(uuid.uuid4())}, headers=headers)
        assert r.status_code == 404


class TestDerivedMilestones:
    async def test_includes_real_lifecycle_events(self, client, estate):
        headers = await _login(client, DOC_A)
        body = await _timeline(client, headers, estate["headache"])
        types = {e["event_type"] for e in body["events"]}

        assert "case.created" in types
        assert "patient.symptoms_submitted" in types
        assert "ai.intake_started" in types
        assert "ai.entities_extracted" in types
        assert "ai.summary_generated" in types
        assert "ai.urgency_assessed" in types
        assert "ai.specialist_recommended" in types
        assert "report.generated" in types

    async def test_omits_milestones_with_no_timestamp(self, client, estate):
        """The case was never assigned or closed, so neither may appear."""
        headers = await _login(client, DOC_A)
        types = {e["event_type"] for e in
                 (await _timeline(client, headers, estate["headache"]))["events"]}
        assert "case.assigned" not in types
        assert "case.closed" not in types

    async def test_every_event_has_a_real_timestamp(self, client, estate):
        headers = await _login(client, DOC_A)
        for e in (await _timeline(client, headers, estate["headache"]))["events"]:
            assert e["timestamp"], f"{e['event_type']} has no timestamp"

    async def test_events_are_reverse_chronological(self, client, estate):
        headers = await _login(client, DOC_A)
        stamps = [e["timestamp"] for e in
                  (await _timeline(client, headers, estate["headache"]))["events"]]
        assert stamps == sorted(stamps, reverse=True)

    async def test_derived_events_are_labelled(self, client, estate):
        headers = await _login(client, DOC_A)
        events = (await _timeline(client, headers, estate["headache"]))["events"]
        assert all(e["source"] in {"recorded", "derived"} for e in events)
        assert any(e["source"] == "derived" for e in events)

class TestCaseScoping:
    async def test_other_cases_events_are_absent(self, client, estate):
        """Both cases share a patient and a doctor; only the FK separates them."""
        headers = await _login(client, DOC_A)
        headache = await _timeline(client, headers, estate["headache"])
        ankle = await _timeline(client, headers, estate["ankle"])

        headache_text = str(headache["events"])
        ankle_text = str(ankle["events"])

        assert "HEADACHE_MARKER" in headache_text
        assert "HEADACHE_MARKER" not in ankle_text
        assert "ANKLE_RX_MARKER" in ankle_text
        assert "ANKLE_RX_MARKER" not in headache_text

    async def test_ankle_case_has_no_ai_intake_events(self, client, estate):
        headers = await _login(client, DOC_A)
        types = {e["event_type"] for e in
                 (await _timeline(client, headers, estate["ankle"]))["events"]}
        assert "ai.intake_started" not in types
        assert "ai.summary_generated" not in types

    async def test_audit_row_without_case_id_is_on_no_timeline(
        self, client, estate, db
    ):
        """A NULL case_id must not be attached to a plausible-looking case."""
        db.add(AuditLog(
            user_id=estate["doc_a"], user_name="Dr. Asha Rao", user_role="doctor",
            action="ORPHAN_EVENT", resource="Report", resource_id="x",
            ip_address="127.0.0.1", status="success", details="ORPHAN_MARKER",
            actor_type="doctor", event_type="report.status_changed",
        ))
        await db.commit()

        headers = await _login(client, DOC_A)
        for case_key in ("headache", "ankle"):
            body = await _timeline(client, headers, estate[case_key])
            assert "ORPHAN_MARKER" not in str(body["events"])


class TestRecordedEvents:
    async def test_notes_and_status_changes_capture_before_and_after(
        self, client, estate
    ):
        headers = await _login(client, DOC_A)
        r = await client.post(
            "/api/v1/doctor/cases/review/save",
            json={"case_id": str(estate["headache"]),
                  "clinical_notes": "Reviewed; consistent with migraine.",
                  "diagnosis": "Migraine without aura"},
            headers=headers,
        )
        assert r.status_code == 200, r.text

        events = (await _timeline(client, headers, estate["headache"]))["events"]
        by_type = {e["event_type"]: e for e in events}

        status_event = by_type["case.status_changed"]
        assert status_event["previous_value"] == "routed"
        assert status_event["new_value"] == "in_consultation"
        assert status_event["field_changed"] == "status"
        assert status_event["source"] == "recorded"
        assert status_event["actor_type"] == "doctor"

        assert by_type["case.diagnosis_updated"]["new_value"] == "Migraine without aura"
        assert by_type["case.notes_added"]["field_changed"] == "notes"

    async def test_report_status_transition_is_recorded(self, client, estate, db):
        headers = await _login(client, DOC_A)
        report = (await db.execute(
            select(Report).where(Report.case_id == estate["headache"])
        )).scalars().first()

        r = await client.put(f"/api/v1/doctor/reports/{report.id}/status",
                             params={"status_str": "ready"}, headers=headers)
        assert r.status_code == 200, r.text

        events = (await _timeline(client, headers, estate["headache"]))["events"]
        approved = next(e for e in events if e["event_type"] == "report.approved")
        assert approved["previous_value"] == "pending_review"
        assert approved["new_value"] == "ready"
        assert approved["actor_type"] == "doctor"
        assert approved["source"] == "recorded"

    async def test_ai_summary_approval_is_attributed_to_the_doctor(
        self, client, estate
    ):
        headers = await _login(client, DOC_A)
        r = await client.post(
            "/api/v1/doctor/cases/review/approve-summary",
            json={"case_id": str(estate["headache"]), "summary": "Reviewed summary."},
            headers=headers,
        )
        assert r.status_code == 200

        events = (await _timeline(client, headers, estate["headache"]))["events"]
        event = next(e for e in events if e["event_type"] == "ai.summary_approved")
        assert event["actor_type"] == "doctor"
        assert event["actor_label"] == "Doctor"

    async def test_unchanged_values_record_nothing(self, client, estate):
        """Saving identical notes twice must not manufacture a second change."""
        headers = await _login(client, DOC_A)
        payload = {"case_id": str(estate["headache"]), "clinical_notes": "Same text."}
        await client.post("/api/v1/doctor/cases/review/save", json=payload, headers=headers)
        first = len([e for e in (await _timeline(client, headers, estate["headache"]))["events"]
                     if e["event_type"] == "case.notes_added"])

        await client.post("/api/v1/doctor/cases/review/save", json=payload, headers=headers)
        second = len([e for e in (await _timeline(client, headers, estate["headache"]))["events"]
                      if e["event_type"] == "case.notes_added"])
        assert second == first

    async def test_null_change_values_stay_null(self, client, estate):
        """An event that changes no tracked value must not report empty strings."""
        headers = await _login(client, DOC_A)
        events = (await _timeline(client, headers, estate["headache"]))["events"]
        created = next(e for e in events if e["event_type"] == "case.created")
        assert created["previous_value"] is None
        assert created["new_value"] is None


class TestActorAttribution:
    async def test_ai_and_human_events_are_distinguishable(self, client, estate):
        headers = await _login(client, DOC_A)
        events = (await _timeline(client, headers, estate["headache"]))["events"]

        ai = [e for e in events if e["actor_type"] == "ai"]
        assert ai, "expected AI events"
        assert all(e["actor_label"] == "AI Assistant" for e in ai)

        patient = [e for e in events if e["actor_type"] == "patient"]
        assert patient
        assert all(e["actor_label"] == "Patient" for e in patient)

    async def test_actor_types_are_from_the_closed_vocabulary(self, client, estate):
        headers = await _login(client, DOC_A)
        allowed = {"patient", "doctor", "ai", "admin", "system"}
        for e in (await _timeline(client, headers, estate["headache"]))["events"]:
            assert e["actor_type"] in allowed


class TestFiltersAndSearch:
    async def test_category_filter(self, client, estate):
        headers = await _login(client, DOC_A)
        body = await _timeline(client, headers, estate["headache"], category="ai")
        assert body["events"]
        assert all(e["category"] == "ai" for e in body["events"])

    async def test_multiple_categories(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.get(
            "/api/v1/shared/timeline",
            params=[("case_id", str(estate["headache"])),
                    ("category", "ai"), ("category", "reports")],
            headers=headers,
        )
        assert r.status_code == 200
        assert {e["category"] for e in r.json()["events"]} <= {"ai", "reports"}

    async def test_keyword_search(self, client, estate):
        headers = await _login(client, DOC_A)
        body = await _timeline(client, headers, estate["headache"],
                               search="HEADACHE_MARKER")
        assert body["total"] >= 1
        assert all("HEADACHE_MARKER" in str(e) for e in body["events"])

    async def test_search_with_no_match_returns_empty(self, client, estate):
        headers = await _login(client, DOC_A)
        body = await _timeline(client, headers, estate["headache"],
                               search="zzz-no-such-thing")
        assert body["total"] == 0
        assert body["events"] == []

    async def test_date_range_excludes_out_of_window(self, client, estate):
        headers = await _login(client, DOC_A)
        body = await _timeline(client, headers, estate["headache"],
                               date_from="2000-01-01", date_to="2000-01-02")
        assert body["total"] == 0


class TestPagination:
    async def test_paginates_and_reports_has_more(self, client, estate):
        headers = await _login(client, DOC_A)
        first = await _timeline(client, headers, estate["headache"], limit=2)
        assert first["returned"] == 2
        assert first["has_more"] is True
        assert first["total"] > 2

        second = await _timeline(client, headers, estate["headache"], limit=2, skip=2)
        assert {e["id"] for e in second["events"]}.isdisjoint(
            {e["id"] for e in first["events"]}
        )

    async def test_last_page_reports_no_more(self, client, estate):
        headers = await _login(client, DOC_A)
        body = await _timeline(client, headers, estate["headache"], limit=200)
        assert body["has_more"] is False
        assert body["returned"] == body["total"]


class TestAppendOnlySurface:
    async def test_no_api_can_mutate_audit_history(self, client, estate):
        """
        The only audit route accepts POST. Nothing exposes update or delete.

        Database-level enforcement is verified separately against PostgreSQL;
        this asserts the HTTP surface offers no way in.
        """
        from app.main import app

        for path, spec in app.openapi()["paths"].items():
            if "audit" not in path and "timeline" not in path:
                continue
            for method in spec:
                assert method.lower() in {"get", "post"}, (
                    f"{method.upper()} {path} can mutate audit history"
                )

    async def test_client_cannot_set_actor_or_case(self, client, estate, db):
        """The client audit route derives the actor; the body cannot override it."""
        headers = await _login(client, DOC_A)
        r = await client.post(
            "/api/v1/shared/audit-logs",
            json={"action": "VIEW_REPORT", "resource": "Report",
                  "resource_id": str(uuid.uuid4()),
                  "user_id": str(estate["doc_b"]), "actor_type": "admin"},
            headers=headers,
        )
        assert r.status_code == 201

        entry = (await db.execute(
            select(AuditLog).where(AuditLog.action == "VIEW_REPORT")
        )).scalars().first()
        assert entry.user_id == estate["doc_a"]
        assert entry.user_role == "doctor"
