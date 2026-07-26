"""
Tests for the Clinical Review Workspace projection.

The workspace is where a clinician forms an opinion, so the assertions here are
mostly about *absence*: a value the record does not hold must render as absent
and be named in `data_gaps`, never as a plausible-looking default. In particular:

* BMI appears only when height and weight both exist.
* Confidence appears only when a score was actually recorded — a 0 is treated as
  "never measured", not as a measured 0%.
* Timeline stages are `completed` only when a row or timestamp backs them.
* Drug interaction review is skipped, with a stated reason, when there is
  nothing to interact.

Authorisation is asserted separately: a doctor may only review a report for a
patient they have a case with.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from app.core.security import get_password_hash
from app.models.appointment import Appointment
from app.models.case import Case, Symptom
from app.models.doctor import Doctor
from app.models.intake import IntakeSessionRecord
from app.models.patient import Patient
from app.models.prescription import Medication, Prescription
from app.models.report import Report
from app.models.user import User
from app.services import clinical_review as review_module

pytestmark = pytest.mark.asyncio

PW = "password123"
DOC = "review.doctor@aronofy.com"
OTHER_DOC = "review.otherdoc@aronofy.com"
PAT = "review.patient@aronofy.com"
BARE_PAT = "review.bare@aronofy.com"

FAKE_SUGGESTIONS: dict[str, Any] = {
    "differential_diagnoses": ["Tension-type headache", "Migraine without aura"],
    "drug_interaction_warnings": ["Sumatriptan with SSRIs: serotonin syndrome risk"],
    "red_flag_symptoms": ["Sudden severe headache"],
    "suggested_lab_tests": ["Complete blood count"],
    "suggested_imaging": ["MRI brain if focal signs develop"],
    "clinical_guideline_summary": "Standard headache management applies.",
    "possible_contraindications": ["Avoid triptans with uncontrolled hypertension"],
    "relevant_medical_history": ["Prior migraine diagnosis"],
    "medication_alerts": ["Penicillin allergy on file"],
}


class _FakeGroq:
    def __init__(self, *, configured: bool = True, payload: dict | None = None):
        self.is_configured = configured
        self._payload = FAKE_SUGGESTIONS if payload is None else payload
        self.calls: list[str] = []

    async def complete_json(self, *, system_prompt, user_content, **_):
        self.calls.append(user_content)
        return self._payload


@pytest.fixture
def fake_groq(monkeypatch):
    client = _FakeGroq()
    monkeypatch.setattr(review_module, "get_groq_client", lambda: client)
    return client


@pytest.fixture
def offline_groq(monkeypatch):
    monkeypatch.setattr(
        review_module, "get_groq_client", lambda: _FakeGroq(configured=False)
    )


@pytest.fixture
async def workspace(db):
    """
    A fully populated case, plus a sparse patient for the absence assertions.

    The rich patient has height/weight/allergies/prescription/appointment; the
    bare patient has none of it, so "not recorded" paths are exercised for real
    rather than mocked.
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
        ("doctor", DOC, "doctor"),
        ("other_doctor", OTHER_DOC, "doctor"),
        ("patient", PAT, "patient"),
        ("bare_patient", BARE_PAT, "patient"),
    ):
        u = User(email=email, hashed_password=get_password_hash(PW), role=role, is_verified=True)
        db.add(u)
        await db.flush()
        ids[key] = u.id

    db.add(Doctor(
        verification_status="verified",
        id=ids["doctor"], first_name="Asha", last_name="Rao", phone="+910000000001",
        specialty="Neurology", hospital_name="MedBridge Central",
        license_number="LIC-REVIEW-1",
    ))
    db.add(Doctor(
        verification_status="verified",
        id=ids["other_doctor"], first_name="Vikram", last_name="Sen", phone="+910000000002",
        specialty="Cardiology", hospital_name="MedBridge East",
        license_number="LIC-REVIEW-2",
    ))
    db.add(Patient(
        id=ids["patient"], first_name="Meera", last_name="Iyer", phone="+910000000003",
        date_of_birth="1992-03-14", gender="female", blood_type="O+",
        height=165.0, weight=60.0,
        allergies=["Penicillin"], chronic_conditions=["Migraine"],
        medications=["Propranolol 40mg"],
    ))
    # No height/weight/blood group/allergies -> every optional field absent.
    db.add(Patient(
        id=ids["bare_patient"], first_name="Bare", last_name="Record",
        phone="+910000000004", date_of_birth="2000-01-01", gender="male",
        allergies=[], chronic_conditions=[], medications=[],
    ))
    await db.flush()

    def make_case(patient_id, doctor_id, summary, **kw):
        return Case(
            patient_id=patient_id, patient_name="Patient", patient_age=34,
            patient_gender="female", doctor_id=doctor_id, doctor_name="Dr. Asha Rao",
            specialty="Neurology", symptom_summary=summary, urgency_level="medium",
            status="routed", ai_extracted_symptoms=[], ai_confidence_score=0.0,
            attachments=[], notes="", **kw,
        )

    rich = make_case(
        ids["patient"], ids["doctor"],
        "Persistent headache for five days with photophobia.",
    )
    rich.ai_extracted_symptoms = ["headache", "photophobia"]
    rich.ai_confidence_score = 0.82
    rich.attachments = [{"name": "intake-photo.jpg", "type": "image", "url": "/uploads/x.jpg"}]
    db.add(rich)

    bare = make_case(ids["bare_patient"], ids["doctor"], "Sore throat.")
    db.add(bare)
    await db.flush()
    ids["case"], ids["bare_case"] = rich.id, bare.id

    db.add(Symptom(case_id=rich.id, name="headache", severity="moderate",
                   duration="5 days", body_part="head"))
    db.add(IntakeSessionRecord(
        session_key="review-sess-1", patient_user_id=ids["patient"], status="routed",
        language="hinglish", intent="symptom_report", followup_rounds=2,
        overall_confidence=0.82, red_flags=["Sudden severe headache"],
        transcript="Mujhe 5 din se sar dard hai aur light se problem hoti hai.",
        medical_case_snapshot={
            "chief_complaint": "Persistent headache for five days",
            "symptoms": ["headache", "photophobia"],
            "severity": "moderate", "onset": "gradual", "duration": "5 days",
            "differential_considerations": ["Tension-type headache", "Migraine"],
            "red_flags": ["Sudden severe headache"],
            "missing_information": ["blood pressure"],
            "recommended_specialty": "Neurology",
            "summary_for_doctor": "34F with a five-day headache and photophobia.",
            "overall_confidence": {"score": 0.82, "band": "high"},
        },
        routed_case_id=rich.id, routed_doctor_id=ids["doctor"],
    ))

    lab = Report(
        patient_id=ids["patient"], patient_name="Meera Iyer", type="lab_result",
        title="Complete Blood Count", summary="CBC normal.", content="Within range.",
        doctor_name="Dr. Asha Rao", date="2026-07-01", status="ready",
        file_url="/uploads/reports/cbc.pdf", ai_generated=False, tags=[], vitals={},
    )
    ai_report = Report(
        patient_id=ids["patient"], case_id=rich.id, patient_name="Meera Iyer",
        type="ai_report", title="AI Clinical Report", summary="Headache assessment.",
        content="Full clinical text.", doctor_name="Dr. Asha Rao", date="2026-07-20",
        status="ready", file_url="/uploads/reports/ai.pdf", ai_generated=True,
        ai_confidence_score=0.82, tags=[], vitals={},
    )
    bare_report = Report(
        patient_id=ids["bare_patient"], case_id=bare.id, patient_name="Bare Record",
        type="ai_report", title="Sparse Report", summary="", content="Minimal.",
        date="2026-07-21", status="ready", ai_generated=False, tags=[], vitals={},
    )
    db.add_all([lab, ai_report, bare_report])
    await db.flush()
    ids["report"], ids["lab_report"] = ai_report.id, lab.id
    ids["bare_report"] = bare_report.id

    rx = Prescription(
        case_id=rich.id, patient_id=ids["patient"], patient_name="Meera Iyer",
        doctor_id=ids["doctor"], doctor_name="Dr. Asha Rao",
        diagnosis="Migraine", notes="", status="active", follow_up_date="2026-08-05",
    )
    db.add(rx)
    await db.flush()
    db.add(Medication(
        prescription_id=rx.id, name="Sumatriptan", dosage="50mg",
        frequency="At onset", duration="7 days", status="active", scheduled_times=[],
        taken_doses=0, total_doses=7, start_date="2026-07-20", end_date="2026-07-27",
        side_effects=["Drowsiness"], interactions=["SSRIs"],
    ))
    db.add(Appointment(
        patient_id=ids["patient"], doctor_id=ids["doctor"], patient_name="Meera Iyer",
        doctor_name="Dr. Asha Rao", specialty="Neurology",
        hospital_name="MedBridge Central", date="2026-07-20", time="10:00",
        duration=30, type="in_person", status="completed", reason="Headache",
        notes="", case_id=rich.id,
    ))
    await db.commit()
    return ids


async def _login(client: AsyncClient, email: str) -> dict[str, str]:
    r = await client.post("/api/v1/auth/login", json={"email": email, "password": PW})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def _review(client, headers, report_id) -> dict:
    r = await client.get(
        f"/api/v1/doctor/reports/{report_id}/clinical-review", headers=headers
    )
    assert r.status_code == 200, r.text
    return r.json()


class TestPatientOverview:
    async def test_populates_from_profile(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        po = (await _review(client, headers, workspace["report"]))["patient_overview"]

        assert po["patient_name"] == "Meera Iyer"
        assert po["gender"] == "female"
        assert po["blood_group"] == "O+"
        assert po["height_cm"] == 165.0
        assert po["weight_kg"] == 60.0
        assert po["allergies"] == ["Penicillin"]
        assert po["chronic_conditions"] == ["Migraine"]
        assert po["current_medications"] == ["Propranolol 40mg"]
        assert po["assigned_doctor"] == "Dr. Asha Rao"
        assert po["assigned_doctor_specialty"] == "Neurology"
        assert po["appointment_date"] == "2026-07-20 10:00"
        assert po["previous_visits"] == 1

    async def test_bmi_computed_when_measurable(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        po = (await _review(client, headers, workspace["report"]))["patient_overview"]
        # 60kg / 1.65m^2 = 22.0
        assert po["bmi"] == pytest.approx(22.0, abs=0.1)
        assert po["bmi_category"] == "Normal"

    async def test_bmi_absent_without_measurements(self, client, workspace, fake_groq):
        """No height or weight must yield null BMI, never an estimate."""
        headers = await _login(client, DOC)
        body = await _review(client, headers, workspace["bare_report"])

        assert body["patient_overview"]["bmi"] is None
        assert body["patient_overview"]["bmi_category"] is None
        assert body["patient_overview"]["blood_group"] is None
        assert any("BMI cannot be computed" in g for g in body["data_gaps"])


class TestAIAnalysis:
    async def test_populates_from_intake_snapshot(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        ai = (await _review(client, headers, workspace["report"]))["ai_analysis"]

        assert ai["has_ai_intake"] is True
        assert "headache" in ai["chief_complaint"].lower()
        assert ai["ai_summary"].startswith("34F")
        assert "headache" in ai["extracted_symptoms"]
        assert ai["possible_causes"] == ["Tension-type headache", "Migraine"]
        assert ai["severity"] == "moderate"
        assert ai["onset"] == "gradual"
        assert ai["urgency_level"] == "medium"
        assert ai["recommended_specialist"] == "Neurology"
        assert ai["emergency_indicators"] == ["Sudden severe headache"]
        assert ai["language_detected"] == "hinglish"
        assert "sar dard" in ai["conversation_summary"]
        assert ai["missing_information"] == ["blood pressure"]

    async def test_symptom_timeline_from_symptom_rows(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        ai = (await _review(client, headers, workspace["report"]))["ai_analysis"]
        entry = ai["symptom_timeline"][0]
        assert entry["name"] == "headache"
        assert entry["severity"] == "moderate"
        assert entry["duration"] == "5 days"
        assert entry["body_part"] == "head"

    async def test_confidence_banded_when_recorded(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        conf = (await _review(client, headers, workspace["report"]))["ai_analysis"]["confidence"]
        assert conf["percentage"] == 82
        assert conf["level"] == "High"

    async def test_confidence_absent_when_never_scored(self, client, workspace, fake_groq):
        """A zero score means 'never measured' and must not render as 0%."""
        headers = await _login(client, DOC)
        body = await _review(client, headers, workspace["bare_report"])
        assert body["ai_analysis"]["confidence"] is None
        assert any("No AI confidence score" in g for g in body["data_gaps"])

    async def test_reports_absent_intake(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        body = await _review(client, headers, workspace["bare_report"])
        assert body["ai_analysis"]["has_ai_intake"] is False
        assert any("No AI intake session" in g for g in body["data_gaps"])


class TestMedicalEvidence:
    async def test_categorises_documents(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        ev = (await _review(client, headers, workspace["report"]))["medical_evidence"]

        assert [d["title"] for d in ev["lab_reports"]] == ["Complete Blood Count"]
        assert [d["title"] for d in ev["ai_report_analysis"]] == ["AI Clinical Report"]
        # Historical excludes the report currently open.
        assert str(workspace["report"]) not in {d["report_id"] for d in ev["historical_reports"]}
        assert all(d["downloadable"] for d in ev["uploaded_reports"])

    async def test_surfaces_prescriptions_and_interactions(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        ev = (await _review(client, headers, workspace["report"]))["medical_evidence"]

        rx = ev["previous_prescriptions"][0]
        assert rx["diagnosis"] == "Migraine"
        assert rx["follow_up_date"] == "2026-08-05"
        med = rx["medications"][0]
        assert med["name"] == "Sumatriptan"
        assert med["interactions"] == ["SSRIs"]

    async def test_includes_case_attachments(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        ev = (await _review(client, headers, workspace["report"]))["medical_evidence"]
        assert ev["case_attachments"][0]["name"] == "intake-photo.jpg"


class TestAISuggestions:
    async def test_labels_and_merges_record_items(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        sx = (await _review(client, headers, workspace["report"]))["ai_suggestions"]

        assert sx["generated"] is True
        assert sx["source"] == "groq"
        # Record-derived considerations survive alongside model output.
        assert "Tension-type headache" in sx["differential_diagnoses"]
        assert "Sudden severe headache" in sx["red_flag_symptoms"]
        # The stored per-medication interaction is present as recorded fact.
        assert any("Sumatriptan" in w for w in sx["drug_interaction_warnings"])
        assert sx["suggested_lab_tests"] == ["Complete blood count"]

    async def test_prompt_is_grounded_in_the_record(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        await _review(client, headers, workspace["report"])
        sent = fake_groq.calls[0]
        assert "Penicillin" in sent
        assert "Sumatriptan" in sent
        assert "headache" in sent

    async def test_prompt_lists_completed_investigations(
        self, client, workspace, fake_groq
    ):
        """
        Tests already performed must reach the model.

        Without them it re-suggests investigations the patient has already had,
        which wastes clinician attention and, if acted on, the patient's time.
        """
        headers = await _login(client, DOC)
        await _review(client, headers, workspace["report"])
        sent = fake_groq.calls[0]
        assert "completed_investigations" in sent
        assert "Complete Blood Count" in sent

    async def test_degrades_to_records_when_offline(self, client, workspace, offline_groq):
        headers = await _login(client, DOC)
        sx = (await _review(client, headers, workspace["report"]))["ai_suggestions"]

        assert sx["generated"] is False
        assert sx["source"] == "records"
        # Record-derived content still renders rather than an empty panel.
        assert "Tension-type headache" in sx["differential_diagnoses"]
        assert any("unavailable" in n for n in sx["notes"])

    async def test_states_why_interaction_review_is_skipped(
        self, client, workspace, fake_groq
    ):
        headers = await _login(client, DOC)
        sx = (await _review(client, headers, workspace["bare_report"]))["ai_suggestions"]
        assert any("fewer than two medications" in n for n in sx["notes"])


class TestTimeline:
    async def test_marks_only_backed_stages_complete(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        timeline = (await _review(client, headers, workspace["report"]))["timeline"]
        by_key = {e["key"]: e for e in timeline}

        assert by_key["case_created"]["status"] == "completed"
        assert by_key["ai_intake"]["status"] == "completed"
        assert by_key["ai_analysis"]["status"] == "completed"
        assert by_key["prescription"]["status"] == "completed"
        assert by_key["report_issued"]["status"] == "completed"
        # Never assigned and never completed -> must stay pending, no timestamp.
        assert by_key["doctor_opened"]["status"] == "pending"
        assert by_key["doctor_opened"]["timestamp"] is None
        assert by_key["completed"]["status"] == "pending"

    async def test_pending_stages_on_sparse_case(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        timeline = (await _review(client, headers, workspace["bare_report"]))["timeline"]
        by_key = {e["key"]: e for e in timeline}
        assert by_key["ai_intake"]["status"] == "pending"
        assert by_key["prescription"]["status"] == "pending"


class TestComparison:
    async def test_three_way_comparison(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        cmp = (await _review(client, headers, workspace["report"]))["comparison"]

        assert "sar dard" in cmp["patient_input"]
        assert "patient's own words" in cmp["patient_input_source"]
        assert "34F" in cmp["ai_interpretation"]
        assert cmp["doctor_decision"]


class TestAuthorization:
    async def test_other_doctor_denied(self, client, workspace, fake_groq):
        headers = await _login(client, OTHER_DOC)
        r = await client.get(
            f"/api/v1/doctor/reports/{workspace['report']}/clinical-review",
            headers=headers,
        )
        assert r.status_code == 403

    async def test_patient_denied(self, client, workspace, fake_groq):
        headers = await _login(client, PAT)
        r = await client.get(
            f"/api/v1/doctor/reports/{workspace['report']}/clinical-review",
            headers=headers,
        )
        assert r.status_code == 403

    async def test_unknown_report_404(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        r = await client.get(
            f"/api/v1/doctor/reports/{uuid.uuid4()}/clinical-review", headers=headers
        )
        assert r.status_code == 404


class TestClinicalDecisionWrites:
    async def test_save_consultation_persists_notes(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        r = await client.post(
            "/api/v1/doctor/cases/review/save",
            json={
                "case_id": str(workspace["case"]),
                "clinical_notes": "Reviewed AI analysis; consistent with migraine.",
                "diagnosis": "Migraine without aura",
                "complete_case": False,
            },
            headers=headers,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "in_consultation"
        assert "Migraine without aura" in body["notes"]
        assert "consistent with migraine" in body["notes"]

        # And it is visible on the next review load.
        review = await _review(client, headers, workspace["report"])
        assert "consistent with migraine" in review["medical_evidence"]["doctor_notes"]
        by_key = {e["key"]: e for e in review["timeline"]}
        assert by_key["doctor_notes"]["status"] == "completed"

    async def test_complete_case_closes_it(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        r = await client.post(
            "/api/v1/doctor/cases/review/save",
            json={
                "case_id": str(workspace["case"]),
                "clinical_notes": "Consultation complete.",
                "complete_case": True,
            },
            headers=headers,
        )
        assert r.status_code == 200
        assert r.json()["status"] == "completed"

        review = await _review(client, headers, workspace["report"])
        by_key = {e["key"]: e for e in review["timeline"]}
        assert by_key["completed"]["status"] == "completed"

    async def test_approve_ai_summary_is_attributed(self, client, workspace, fake_groq):
        headers = await _login(client, DOC)
        r = await client.post(
            "/api/v1/doctor/cases/review/approve-summary",
            json={
                "case_id": str(workspace["case"]),
                "summary": "34F with a five-day headache and photophobia.",
            },
            headers=headers,
        )
        assert r.status_code == 200, r.text

        review = await _review(client, headers, workspace["report"])
        notes = review["medical_evidence"]["doctor_notes"]
        assert "reviewed and approved" in notes
        assert "five-day headache" in notes

    async def test_cannot_write_to_another_doctors_case(self, client, workspace, fake_groq):
        headers = await _login(client, OTHER_DOC)
        r = await client.post(
            "/api/v1/doctor/cases/review/save",
            json={"case_id": str(workspace["case"]), "clinical_notes": "x"},
            headers=headers,
        )
        assert r.status_code == 403
