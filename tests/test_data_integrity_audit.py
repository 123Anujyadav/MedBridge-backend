"""
Regression tests for the Doctor Portal data-integrity audit.

Every test here fails against the pre-audit code. They fall into two families:

**Ownership** — a resource may only be read or written by the patient it belongs
to, the doctor assigned to its case, or an admin. Anything else is a PHI leak.

**Case scoping** — where a record carries a `case_id`, consumers must key off
that FK and nothing else. The dangerous failure mode is not a crash, it is a
screen that quietly answers about the wrong case: an unrelated lab result
rendered with another consultation's chief complaint, or a case timeline
showing a prescription written for a different visit. When `case_id` is NULL
the correct behaviour is to report no case context, never to infer one.

The fixture deliberately gives ONE patient TWO cases under the SAME doctor.
That is the configuration in which cross-case bleed is invisible to ownership
checks — both cases pass every authorisation rule, so only correct FK scoping
keeps them apart.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from app.core.security import get_password_hash
from app.models.case import Case
from app.models.doctor import Doctor
from app.models.intake import IntakeSessionRecord
from app.models.patient import Patient
from app.models.prescription import Medication, Prescription
from app.models.report import Report
from app.models.user import User
from app.services import clinical_review as review_module

pytestmark = pytest.mark.asyncio

PW = "password123"
DOC_A = "audit.doca@aronofy.com"
DOC_B = "audit.docb@aronofy.com"
PAT_A = "audit.pata@aronofy.com"
PAT_B = "audit.patb@aronofy.com"


class _FakeGroq:
    """Keeps the AI suggestion pass out of these tests."""

    is_configured = False

    async def complete_json(self, **_):
        return {}


@pytest.fixture(autouse=True)
def no_groq(monkeypatch):
    monkeypatch.setattr(review_module, "get_groq_client", lambda: _FakeGroq())


@pytest.fixture
async def estate(db):
    """
    Two doctors, two patients, and — critically — two cases for patient A
    under doctor A, each with its own report and prescription.
    """
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in (
        "intake_extracted_entities", "intake_sessions", "notifications",
        "medications", "prescriptions", "appointments", "reports",
        "symptoms", "cases", "doctors", "patients", "users",
    ):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    ids: dict[str, Any] = {}
    for key, email, role in (
        ("doc_a", DOC_A, "doctor"), ("doc_b", DOC_B, "doctor"),
        ("pat_a", PAT_A, "patient"), ("pat_b", PAT_B, "patient"),
    ):
        u = User(email=email, hashed_password=get_password_hash(PW), role=role, is_verified=True)
        db.add(u)
        await db.flush()
        ids[key] = u.id

    db.add(Doctor(id=ids["doc_a"], first_name="Asha", last_name="Rao",
                  phone="+911", specialty="Neurology", hospital_name="Central",
                  license_number="LIC-AUD-A", verification_status="verified"))
    db.add(Doctor(id=ids["doc_b"], first_name="Vikram", last_name="Sen",
                  phone="+912", specialty="Cardiology", hospital_name="East",
                  license_number="LIC-AUD-B", verification_status="verified"))
    db.add(Patient(id=ids["pat_a"], first_name="Meera", last_name="Iyer",
                   phone="+913", date_of_birth="1992-03-14", gender="female",
                   allergies=["Penicillin"], chronic_conditions=[], medications=[]))
    db.add(Patient(id=ids["pat_b"], first_name="Rahul", last_name="Nair",
                   phone="+914", date_of_birth="1988-06-02", gender="male",
                   allergies=[], chronic_conditions=[], medications=[]))
    await db.flush()

    def make_case(patient_id, doctor_id, doctor_name, specialty, summary, urgency):
        return Case(
            patient_id=patient_id, patient_name="Patient", patient_age=34,
            patient_gender="female", doctor_id=doctor_id, doctor_name=doctor_name,
            specialty=specialty, symptom_summary=summary, urgency_level=urgency,
            status="routed", ai_extracted_symptoms=[], ai_confidence_score=0.0,
            attachments=[], notes="",
        )

    # Both belong to patient A under doctor A — ownership cannot separate them.
    headache = make_case(ids["pat_a"], ids["doc_a"], "Dr. Asha Rao", "Neurology",
                         "Severe headache with photophobia.", "high")
    ankle = make_case(ids["pat_a"], ids["doc_a"], "Dr. Asha Rao", "Orthopedics",
                      "Twisted ankle after a fall.", "low")
    # Patient B, doctor B — used for cross-doctor and cross-patient checks.
    cardiac = make_case(ids["pat_b"], ids["doc_b"], "Dr. Vikram Sen", "Cardiology",
                        "Chest tightness on exertion.", "critical")
    db.add_all([headache, ankle, cardiac])
    await db.flush()
    ids["headache_case"], ids["ankle_case"] = headache.id, ankle.id
    ids["cardiac_case"] = cardiac.id

    db.add(IntakeSessionRecord(
        session_key="audit-headache", patient_user_id=ids["pat_a"], status="routed",
        language="english", intent="symptom_report", overall_confidence=0.9,
        red_flags=["Thunderclap onset"], transcript="My head is pounding.",
        medical_case_snapshot={
            "chief_complaint": "Severe headache with photophobia",
            "symptoms": ["headache", "photophobia"],
            "red_flags": ["Thunderclap onset"],
            "summary_for_doctor": "HEADACHE CASE SUMMARY",
            "overall_confidence": {"score": 0.9, "band": "high"},
        },
        routed_case_id=headache.id, routed_doctor_id=ids["doc_a"],
    ))

    headache_report = Report(
        patient_id=ids["pat_a"], case_id=headache.id, patient_name="Meera Iyer",
        type="ai_report", title="Headache Report", summary="Headache assessment.",
        content="Headache text.", doctor_name="Dr. Asha Rao", date="2026-07-20",
        status="ready", ai_generated=True, ai_confidence_score=0.9, tags=[], vitals={},
    )
    ankle_report = Report(
        patient_id=ids["pat_a"], case_id=ankle.id, patient_name="Meera Iyer",
        type="ai_report", title="Ankle Report", summary="Ankle assessment.",
        content="Ankle text.", doctor_name="Dr. Asha Rao", date="2026-07-25",
        status="ready", ai_generated=True, tags=[], vitals={},
    )
    # No case_id at all — the "never infer" case.
    orphan_report = Report(
        patient_id=ids["pat_a"], patient_name="Meera Iyer", type="lab_result",
        title="Orphan Lab Result", summary="CBC within limits.",
        content="Lab text.", doctor_name="Dr. Asha Rao", date="2026-07-26",
        status="ready", ai_generated=False, tags=[], vitals={},
    )
    db.add_all([headache_report, ankle_report, orphan_report])
    await db.flush()
    ids["headache_report"] = headache_report.id
    ids["ankle_report"] = ankle_report.id
    ids["orphan_report"] = orphan_report.id

    for case_key, diagnosis in ((headache, "Migraine"), (ankle, "Ankle sprain")):
        rx = Prescription(case_id=case_key.id, patient_id=ids["pat_a"],
                          patient_name="Meera Iyer", doctor_id=ids["doc_a"],
                          doctor_name="Dr. Asha Rao", diagnosis=diagnosis,
                          notes="", status="active")
        db.add(rx)
        await db.flush()
        db.add(Medication(prescription_id=rx.id, name=f"Med for {diagnosis}",
                          dosage="1", frequency="daily", duration="5 days",
                          status="active", scheduled_times=[], taken_doses=0,
                          total_doses=5, start_date="2026-07-20",
                          end_date="2026-07-25", side_effects=[], interactions=[]))

    await db.commit()
    return ids


async def _login(client: AsyncClient, email: str) -> dict[str, str]:
    r = await client.post("/api/v1/auth/login", json={"email": email, "password": PW})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ── Timeline ownership ───────────────────────────────────────────────────────


class TestCaseTimelineOwnership:
    """`GET /shared/timeline` performed no ownership check whatsoever."""

    async def test_owning_patient_may_read(self, client, estate):
        headers = await _login(client, PAT_A)
        r = await client.get("/api/v1/shared/timeline",
                             params={"case_id": str(estate["headache_case"])},
                             headers=headers)
        assert r.status_code == 200

    async def test_other_patient_denied(self, client, estate):
        headers = await _login(client, PAT_B)
        r = await client.get("/api/v1/shared/timeline",
                             params={"case_id": str(estate["headache_case"])},
                             headers=headers)
        assert r.status_code == 403

    async def test_unassigned_doctor_denied(self, client, estate):
        headers = await _login(client, DOC_B)
        r = await client.get("/api/v1/shared/timeline",
                             params={"case_id": str(estate["headache_case"])},
                             headers=headers)
        assert r.status_code == 403

    async def test_assigned_doctor_may_read(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.get("/api/v1/shared/timeline",
                             params={"case_id": str(estate["headache_case"])},
                             headers=headers)
        assert r.status_code == 200

    async def test_completed_case_does_not_crash(self, client, estate):
        """`case.diagnosis` is not a mapped column; reading it raised AttributeError."""
        headers = await _login(client, DOC_A)
        closed = await client.post(
            "/api/v1/doctor/cases/review/save",
            json={"case_id": str(estate["headache_case"]),
                  "clinical_notes": "Closing out.", "complete_case": True},
            headers=headers,
        )
        assert closed.status_code == 200 and closed.json()["status"] == "completed"

        r = await client.get("/api/v1/shared/timeline",
                             params={"case_id": str(estate["headache_case"])},
                             headers=headers)
        assert r.status_code == 200, r.text
        assert any(e["event_type"] == "case.closed" for e in r.json()["events"])

    async def test_timeline_contains_only_its_own_case_prescriptions(
        self, client, estate
    ):
        headers = await _login(client, DOC_A)
        r = await client.get("/api/v1/shared/timeline",
                             params={"case_id": str(estate["headache_case"])},
                             headers=headers)
        body = str(r.json()["events"])
        assert "Migraine" in body
        assert "Ankle sprain" not in body


# ── Report authoring ownership ───────────────────────────────────────────────


class TestReportAuthoringOwnership:
    """`POST /doctor/reports` accepted any patient_id with no check."""

    @staticmethod
    def _payload(patient_id) -> dict[str, Any]:
        return {
            "patient_id": str(patient_id), "patient_name": "Injected",
            "type": "ai_report", "title": "Injected Report",
            "summary": "s", "content": "c", "date": "2026-07-26",
        }

    async def test_cannot_author_for_unrelated_patient(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.post("/api/v1/doctor/reports",
                              json=self._payload(estate["pat_b"]), headers=headers)
        assert r.status_code == 403

    async def test_injected_report_does_not_reach_the_patient(self, client, estate):
        headers = await _login(client, DOC_A)
        await client.post("/api/v1/doctor/reports",
                          json=self._payload(estate["pat_b"]), headers=headers)

        victim = await _login(client, PAT_B)
        listing = await client.get("/api/v1/patient/reports", headers=victim)
        assert "Injected Report" not in {r["title"] for r in listing.json()}

    async def test_own_patient_still_permitted(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.post("/api/v1/doctor/reports",
                              json=self._payload(estate["pat_a"]), headers=headers)
        assert r.status_code == 201, r.text


# ── Prescription ↔ case consistency ──────────────────────────────────────────


class TestPrescriptionCaseConsistency:
    """
    `patient_id` came from the request while `patient_name` came from the case,
    so a prescription could be filed under one patient carrying another's name.
    """

    @staticmethod
    def _payload(case_id, patient_id) -> dict[str, Any]:
        return {
            "case_id": str(case_id), "patient_id": str(patient_id),
            "diagnosis": "Mismatched", "notes": "",
            "medications": [{
                "name": "Test", "dosage": "1", "frequency": "daily",
                "duration": "5 days", "special_instructions": "",
                "scheduled_times": [], "start_date": "2026-07-26",
                "end_date": "2026-07-31", "side_effects": [], "interactions": [],
            }],
        }

    async def test_rejects_patient_not_on_the_case(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.post(
            "/api/v1/doctor/prescriptions",
            json=self._payload(estate["headache_case"], estate["pat_b"]),
            headers=headers,
        )
        assert r.status_code == 403

    async def test_mismatched_prescription_never_persists(self, client, estate):
        headers = await _login(client, DOC_A)
        await client.post(
            "/api/v1/doctor/prescriptions",
            json=self._payload(estate["headache_case"], estate["pat_b"]),
            headers=headers,
        )
        victim = await _login(client, PAT_B)
        listing = await client.get("/api/v1/patient/prescriptions", headers=victim)
        assert "Mismatched" not in {p["diagnosis"] for p in listing.json()}

    async def test_matching_patient_accepted(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.post(
            "/api/v1/doctor/prescriptions",
            json=self._payload(estate["headache_case"], estate["pat_a"]),
            headers=headers,
        )
        assert r.status_code == 201, r.text


# ── Case inference ───────────────────────────────────────────────────────────


class TestNoCaseInference:
    """A NULL case_id must produce no case context, never an inferred one."""

    async def test_orphan_report_has_no_case_context(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.get(
            f"/api/v1/doctor/reports/{estate['orphan_report']}/clinical-review",
            headers=headers,
        )
        assert r.status_code == 200, r.text
        body = r.json()

        assert body["case_id"] is None
        assert body["case_status"] is None
        ai = body["ai_analysis"]
        assert ai["has_ai_intake"] is False
        assert ai["urgency_level"] is None
        assert ai["confidence"] is None
        assert ai["emergency_indicators"] == []
        # The most dangerous single symptom of the old fallback.
        assert "HEADACHE CASE SUMMARY" not in ai["ai_summary"]
        assert "photophobia" not in (ai["chief_complaint"] or "").lower()

    async def test_orphan_report_states_the_gap(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.get(
            f"/api/v1/doctor/reports/{estate['orphan_report']}/clinical-review",
            headers=headers,
        )
        gaps = r.json()["data_gaps"]
        assert any("not linked to a consultation case" in g for g in gaps)

    async def test_orphan_report_keeps_patient_level_truth(self, client, estate):
        """Dropping case inference must not drop correct patient-level data."""
        headers = await _login(client, DOC_A)
        r = await client.get(
            f"/api/v1/doctor/reports/{estate['orphan_report']}/clinical-review",
            headers=headers,
        )
        po = r.json()["patient_overview"]
        assert po["patient_name"] == "Meera Iyer"
        assert po["allergies"] == ["Penicillin"]

    async def test_orphan_card_borrows_nothing(self, client, estate):
        headers = await _login(client, DOC_A)
        cards = (await client.get("/api/v1/doctor/reports", headers=headers)).json()
        orphan = next(c for c in cards if c["title"] == "Orphan Lab Result")

        assert orphan["case_id"] is None
        assert orphan["chief_complaint"] is None
        assert orphan["urgency_level"] is None
        assert orphan["ai_confidence"] is None
        assert "HEADACHE" not in orphan["ai_summary"].upper()


# ── Cross-case bleed ─────────────────────────────────────────────────────────


class TestCrossCaseScoping:
    """
    Both cases belong to the same patient and the same doctor, so every
    ownership check passes for both. Only FK scoping keeps them apart.
    """

    async def test_timeline_uses_only_this_cases_prescription(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.get(
            f"/api/v1/doctor/reports/{estate['ankle_report']}/clinical-review",
            headers=headers,
        )
        timeline = {e["key"]: e for e in r.json()["timeline"]}
        assert timeline["prescription"]["detail"] == "Ankle sprain"
        assert "Migraine" not in timeline["prescription"]["detail"]

    async def test_timeline_report_milestone_is_this_cases_report(
        self, client, estate
    ):
        headers = await _login(client, DOC_A)
        save = await client.post(
            "/api/v1/doctor/cases/review/save",
            json={"case_id": str(estate["ankle_case"]),
                  "clinical_notes": "Ankle reviewed."},
            headers=headers,
        )
        assert save.status_code == 200, save.text
        timeline = {e["key"]: e for e in save.json()["timeline"]}
        assert timeline["report_issued"]["detail"] == "Ankle Report"
        assert timeline["report_issued"]["detail"] != "Headache Report"

    async def test_ai_summary_belongs_to_its_own_case(self, client, estate):
        """The intake snapshot is attached to the headache case only."""
        headers = await _login(client, DOC_A)

        headache = await client.get(
            f"/api/v1/doctor/reports/{estate['headache_report']}/clinical-review",
            headers=headers,
        )
        ankle = await client.get(
            f"/api/v1/doctor/reports/{estate['ankle_report']}/clinical-review",
            headers=headers,
        )
        assert headache.json()["ai_analysis"]["ai_summary"] == "HEADACHE CASE SUMMARY"
        assert ankle.json()["ai_analysis"]["ai_summary"] != "HEADACHE CASE SUMMARY"
        assert ankle.json()["ai_analysis"]["emergency_indicators"] == []

    async def test_evidence_panel_is_still_patient_wide(self, client, estate):
        """
        Case scoping must not over-correct.

        The evidence panel is explicitly the patient's prescribing history, so
        it should still show both cases' prescriptions.
        """
        headers = await _login(client, DOC_A)
        r = await client.get(
            f"/api/v1/doctor/reports/{estate['ankle_report']}/clinical-review",
            headers=headers,
        )
        diagnoses = {
            p["diagnosis"] for p in r.json()["medical_evidence"]["previous_prescriptions"]
        }
        assert {"Migraine", "Ankle sprain"} <= diagnoses


# ── Consultation → report linkage ────────────────────────────────────────────


class TestConsultationReportLinkage:
    async def test_completed_consultation_links_its_report(self, client, estate):
        """A consultation report used to be created with a NULL case_id."""
        headers = await _login(client, DOC_A)
        r = await client.post(
            "/api/v1/doctor/cases/complete",
            json={
                "case_id": str(estate["ankle_case"]),
                "diagnosis": "Ankle sprain, grade I",
                "clinical_notes": "Rest and ice.",
                "medications": [], "recommended_tests": [],
            },
            headers=headers,
        )
        assert r.status_code == 201, r.text
        assert r.json()["case_id"] == str(estate["ankle_case"])


# ── Cross-doctor isolation on every read surface ─────────────────────────────


class TestCrossDoctorIsolation:
    async def test_doctor_b_cannot_reach_doctor_a_surfaces(self, client, estate):
        headers = await _login(client, DOC_B)

        for path in (
            f"/api/v1/doctor/reports/{estate['headache_report']}/clinical-review",
            f"/api/v1/doctor/cases/{estate['headache_case']}",
            f"/api/v1/doctor/patients/{estate['pat_a']}",
        ):
            r = await client.get(path, headers=headers)
            assert r.status_code == 403, f"{path} returned {r.status_code}"

    async def test_doctor_b_report_list_excludes_patient_a(self, client, estate):
        headers = await _login(client, DOC_B)
        cards = (await client.get("/api/v1/doctor/reports", headers=headers)).json()
        assert all(c["patient_id"] != str(estate["pat_a"]) for c in cards)

    async def test_doctor_b_cannot_write_to_doctor_a_case(self, client, estate):
        headers = await _login(client, DOC_B)
        for path, payload in (
            ("/api/v1/doctor/cases/review/save",
             {"case_id": str(estate["headache_case"]), "clinical_notes": "x"}),
            ("/api/v1/doctor/cases/review/approve-summary",
             {"case_id": str(estate["headache_case"]), "summary": "x"}),
        ):
            r = await client.post(path, json=payload, headers=headers)
            assert r.status_code == 403, f"{path} returned {r.status_code}"

    async def test_unknown_ids_are_not_oracles(self, client, estate):
        """A random id must 404, not reveal existence via a different code."""
        headers = await _login(client, DOC_A)
        r = await client.get(
            f"/api/v1/doctor/reports/{uuid.uuid4()}/clinical-review", headers=headers
        )
        assert r.status_code == 404
