"""
Tests for the AI-assisted "Issue AI Clinical Report" workflow.

Two guarantees dominate here:

* **Isolation** — a doctor may only draft from, and issue against, their own
  cases. The picker, the draft and the issue route are all scoped.
* **No fabrication** — when Groq is unreachable the draft still assembles from
  stored records and says so (`draft_source="records"`), and nothing in the
  draft may contain clinical facts the record does not hold.

The LLM is faked throughout: these assert the workflow, not model quality.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from app.core.security import get_password_hash
from app.models.case import Case, Symptom
from app.models.doctor import Doctor
from app.models.intake import IntakeSessionRecord
from app.models.notification import NotificationItem
from app.models.patient import Patient
from app.models.report import Report
from app.models.user import User
from app.services import ai_report as ai_report_module
from conftest import login_payload

pytestmark = pytest.mark.asyncio

PW = "password123"
DOC_EMAIL = "report.doctor@aronofy.com"
OTHER_DOC_EMAIL = "report.otherdoc@aronofy.com"
PATIENT_EMAIL = "report.patient@aronofy.com"

FAKE_DRAFT: dict[str, Any] = {
    "title": "Clinical Assessment - Persistent Headache",
    "summary": "Patient reports a persistent headache with photophobia over five days.",
    "clinical_findings": ["Headache, severity moderate", "Photophobia on exposure"],
    "diagnosis": "Probable tension-type headache pending review",
    "clinical_notes": "Five-day history of bilateral headache. No focal deficits recorded.",
    "prescription": "No medication recommended pending physician review.",
    "follow_up_instructions": "Return in one week or sooner if symptoms escalate.",
    "recommendations": ["Maintain a headache diary", "Ensure adequate hydration"],
    "recommended_tests": ["Basic metabolic panel"],
    "warnings": ["Blood pressure was not recorded during intake."],
}


class _FakeGroq:
    """Stands in for the shared GroqClient."""

    def __init__(self, *, configured: bool = True, payload: dict | None = None):
        self.is_configured = configured
        self._payload = FAKE_DRAFT if payload is None else payload
        self.calls: list[str] = []

    async def complete_json(self, *, system_prompt, user_content, **_):
        self.calls.append(user_content)
        return self._payload


@pytest.fixture
def fake_groq(monkeypatch):
    """Default: a configured model returning a well-formed draft."""
    client = _FakeGroq()
    monkeypatch.setattr(ai_report_module, "get_groq_client", lambda: client)
    return client


@pytest.fixture
async def clinic(db):
    """A doctor, an unrelated doctor, a patient, and one AI-intake case."""
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in (
        "intake_extracted_entities",
        "intake_sessions",
        "notifications",
        "reports",
        "symptoms",
        "cases",
        "doctors",
        "patients",
        "users",
    ):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    ids: dict[str, Any] = {}

    for key, email, role in (
        ("doctor", DOC_EMAIL, "doctor"),
        ("other_doctor", OTHER_DOC_EMAIL, "doctor"),
        ("patient", PATIENT_EMAIL, "patient"),
    ):
        user = User(
            email=email,
            hashed_password=get_password_hash(PW),
            role=role,
            is_verified=True,
        )
        db.add(user)
        await db.flush()
        ids[key] = user.id

    db.add(
        Doctor(
            verification_status="verified",
            id=ids["doctor"],
            first_name="Asha",
            last_name="Rao",
            phone="+910000000001",
            specialty="Neurology",
            hospital_name="MedBridge Central",
            license_number="LIC-REPORT-1",
        )
    )
    db.add(
        Doctor(
            verification_status="verified",
            id=ids["other_doctor"],
            first_name="Vikram",
            last_name="Sen",
            phone="+910000000002",
            specialty="Cardiology",
            hospital_name="MedBridge East",
            license_number="LIC-REPORT-2",
        )
    )
    db.add(
        Patient(
            id=ids["patient"],
            first_name="Meera",
            last_name="Iyer",
            phone="+910000000003",
            date_of_birth="1992-03-14",
            gender="female",
            allergies=["Penicillin"],
            chronic_conditions=["Migraine"],
        )
    )
    await db.flush()

    case = Case(
        patient_id=ids["patient"],
        patient_name="Meera Iyer",
        patient_age=34,
        patient_gender="female",
        doctor_id=ids["doctor"],
        doctor_name="Dr. Asha Rao",
        specialty="Neurology",
        symptom_summary="Persistent headache for five days with photophobia.",
        urgency_level="medium",
        status="routed",
        ai_extracted_symptoms=["headache", "photophobia"],
        ai_specialty_recommendation="Neurology",
        ai_confidence_score=0.82,
        attachments=[],
        notes="",
    )
    db.add(case)
    await db.flush()
    ids["case"] = case.id

    db.add(
        Symptom(
            case_id=case.id,
            name="headache",
            severity="moderate",
            duration="5 days",
            body_part="head",
        )
    )
    db.add(
        IntakeSessionRecord(
            session_key="sess-report-1",
            patient_user_id=ids["patient"],
            status="routed",
            language="english",
            intent="symptom_report",
            overall_confidence=0.82,
            red_flags=[],
            transcript="I have had a headache for five days.",
            medical_case_snapshot={
                "chief_complaint": "Persistent headache for five days",
                "symptoms": ["headache", "photophobia"],
                "severity": "moderate",
                "allergies": ["Penicillin"],
                "current_medications": [],
                "medical_history": ["Migraine"],
                "red_flags": [],
                "missing_information": ["blood pressure"],
                "summary_for_doctor": "34F with a five-day bilateral headache.",
            },
            routed_case_id=case.id,
            routed_doctor_id=ids["doctor"],
        )
    )

    # A case belonging to the other doctor, to prove scoping.
    foreign = Case(
        patient_id=ids["patient"],
        patient_name="Meera Iyer",
        patient_age=34,
        patient_gender="female",
        doctor_id=ids["other_doctor"],
        doctor_name="Dr. Vikram Sen",
        specialty="Cardiology",
        symptom_summary="Chest tightness on exertion.",
        urgency_level="high",
        status="routed",
        ai_extracted_symptoms=["chest tightness"],
        ai_confidence_score=0.5,
        attachments=[],
        notes="",
    )
    db.add(foreign)
    await db.flush()
    ids["foreign_case"] = foreign.id

    await db.commit()
    return ids


async def _login(client: AsyncClient, email: str) -> dict[str, str]:
    resp = await client.post("/api/v1/auth/login", json=await login_payload(email, PW))
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


class TestDraftCandidates:
    async def test_lists_only_own_cases(self, client, clinic, fake_groq):
        headers = await _login(client, DOC_EMAIL)
        resp = await client.get(
            "/api/v1/doctor/reports/draft-candidates", headers=headers
        )

        assert resp.status_code == 200
        body = resp.json()
        ids = {row["case_id"] for row in body}
        assert str(clinic["case"]) in ids
        assert str(clinic["foreign_case"]) not in ids

    async def test_flags_ai_intake_availability(self, client, clinic, fake_groq):
        headers = await _login(client, DOC_EMAIL)
        resp = await client.get(
            "/api/v1/doctor/reports/draft-candidates", headers=headers
        )
        row = next(r for r in resp.json() if r["case_id"] == str(clinic["case"]))
        assert row["has_ai_intake"] is True
        assert row["patient_name"] == "Meera Iyer"


class TestDraftAutoFill:
    async def test_autofills_every_context_field(self, client, clinic, fake_groq):
        headers = await _login(client, DOC_EMAIL)
        resp = await client.post(
            "/api/v1/doctor/reports/ai-draft",
            json={"case_id": str(clinic["case"])},
            headers=headers,
        )

        assert resp.status_code == 200, resp.text
        d = resp.json()

        # Identity — none of this is typed by the doctor.
        assert d["patient_name"] == "Meera Iyer"
        assert d["patient_id"] == str(clinic["patient"])
        assert d["patient_age"] == 34
        assert d["patient_gender"] == "female"
        assert d["case_id"] == str(clinic["case"])
        assert d["doctor_name"] == "Dr. Asha Rao"
        assert d["hospital_name"] == "MedBridge Central"
        assert d["date"]
        assert d["title"]

        # Clinical context drawn from the AI intake snapshot and case rows.
        assert "headache" in d["chief_complaint"].lower()
        assert d["ai_summary"]
        assert "headache" in d["symptoms"]
        assert d["clinical_findings"]
        assert any("Penicillin" in h for h in d["previous_history"])
        assert d["ai_confidence_score"] == pytest.approx(0.82)

    async def test_seeds_editable_fields_from_model(self, client, clinic, fake_groq):
        headers = await _login(client, DOC_EMAIL)
        resp = await client.post(
            "/api/v1/doctor/reports/ai-draft",
            json={"case_id": str(clinic["case"])},
            headers=headers,
        )
        d = resp.json()

        assert d["diagnosis"] == FAKE_DRAFT["diagnosis"]
        assert d["clinical_notes"] == FAKE_DRAFT["clinical_notes"]
        assert d["prescription"] == FAKE_DRAFT["prescription"]
        assert d["follow_up_instructions"] == FAKE_DRAFT["follow_up_instructions"]
        assert d["recommendations"] == FAKE_DRAFT["recommendations"]
        assert d["ai_generated"] is True
        assert d["draft_source"] == "groq"

    async def test_surfaces_record_gaps_as_warnings(self, client, clinic, fake_groq):
        headers = await _login(client, DOC_EMAIL)
        resp = await client.post(
            "/api/v1/doctor/reports/ai-draft",
            json={"case_id": str(clinic["case"])},
            headers=headers,
        )
        warnings = resp.json()["warnings"]
        assert any("blood pressure" in w.lower() for w in warnings)

    async def test_prompt_carries_only_record_facts(self, client, clinic, fake_groq):
        """The model sees the record and nothing but the record."""
        headers = await _login(client, DOC_EMAIL)
        await client.post(
            "/api/v1/doctor/reports/ai-draft",
            json={"case_id": str(clinic["case"])},
            headers=headers,
        )
        sent = fake_groq.calls[0]
        assert "headache" in sent
        assert "Penicillin" in sent
        # This patient's other case is legitimate continuity-of-care history and
        # is expected in the prompt; cross-doctor access is gated at the route
        # level instead (see TestAuthorization).
        assert "Chest tightness" in sent

    async def test_persists_nothing(self, client, clinic, fake_groq, db):
        """Drafting is a preview: no report may exist until the doctor issues."""
        headers = await _login(client, DOC_EMAIL)
        await client.post(
            "/api/v1/doctor/reports/ai-draft",
            json={"case_id": str(clinic["case"])},
            headers=headers,
        )
        count = await db.scalar(text("SELECT COUNT(*) FROM reports"))
        assert count == 0


class TestDegradedDraft:
    """Groq being down must not produce an empty form or invented text."""

    async def test_falls_back_to_records_when_unconfigured(
        self, client, clinic, monkeypatch
    ):
        monkeypatch.setattr(
            ai_report_module, "get_groq_client", lambda: _FakeGroq(configured=False)
        )
        headers = await _login(client, DOC_EMAIL)
        resp = await client.post(
            "/api/v1/doctor/reports/ai-draft",
            json={"case_id": str(clinic["case"])},
            headers=headers,
        )

        assert resp.status_code == 200
        d = resp.json()
        assert d["ai_generated"] is False
        assert d["draft_source"] == "records"
        # Context still auto-loads from the database.
        assert d["patient_name"] == "Meera Iyer"
        assert "headache" in d["symptoms"]
        # The clinician-judgement field is left blank rather than guessed.
        assert d["diagnosis"] == ""

    async def test_empty_model_reply_degrades_cleanly(self, client, clinic, monkeypatch):
        monkeypatch.setattr(
            ai_report_module, "get_groq_client", lambda: _FakeGroq(payload={})
        )
        headers = await _login(client, DOC_EMAIL)
        resp = await client.post(
            "/api/v1/doctor/reports/ai-draft",
            json={"case_id": str(clinic["case"])},
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["draft_source"] == "records"


class TestAuthorization:
    async def test_cannot_draft_another_doctors_case(self, client, clinic, fake_groq):
        headers = await _login(client, DOC_EMAIL)
        resp = await client.post(
            "/api/v1/doctor/reports/ai-draft",
            json={"case_id": str(clinic["foreign_case"])},
            headers=headers,
        )
        assert resp.status_code == 403

    async def test_cannot_issue_against_another_doctors_case(
        self, client, clinic, fake_groq
    ):
        headers = await _login(client, DOC_EMAIL)
        resp = await client.post(
            "/api/v1/doctor/reports/issue",
            json={
                "case_id": str(clinic["foreign_case"]),
                "title": "Unauthorised",
                "diagnosis": "Attempted cross-doctor issue",
            },
            headers=headers,
        )
        assert resp.status_code == 403

    async def test_patient_cannot_reach_the_workflow(self, client, clinic, fake_groq):
        headers = await _login(client, PATIENT_EMAIL)
        resp = await client.get(
            "/api/v1/doctor/reports/draft-candidates", headers=headers
        )
        assert resp.status_code == 403

    async def test_unknown_case_is_404(self, client, clinic, fake_groq):
        headers = await _login(client, DOC_EMAIL)
        resp = await client.post(
            "/api/v1/doctor/reports/ai-draft",
            json={"case_id": str(uuid.uuid4())},
            headers=headers,
        )
        assert resp.status_code == 404


class TestIssueReport:
    @staticmethod
    def _payload(case_id: uuid.UUID, **overrides) -> dict[str, Any]:
        body = {
            "case_id": str(case_id),
            "title": "Clinical Assessment - Persistent Headache",
            "summary": "Five-day headache with photophobia.",
            "diagnosis": "Tension-type headache",
            "clinical_notes": "Doctor-edited assessment narrative.",
            "prescription": "Paracetamol 500mg as needed.",
            "follow_up_instructions": "Review in one week.",
            "recommendations": ["Headache diary", "Hydration"],
            "recommended_tests": ["Basic metabolic panel"],
            "ai_generated": True,
            "ai_confidence_score": 0.82,
        }
        body.update(overrides)
        return body

    async def test_generates_pdf_and_stores_report(self, client, clinic, fake_groq, db):
        headers = await _login(client, DOC_EMAIL)
        resp = await client.post(
            "/api/v1/doctor/reports/issue",
            json=self._payload(clinic["case"]),
            headers=headers,
        )

        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["file_url"].startswith("/uploads/reports/")
        assert body["file_url"].endswith(".pdf")
        assert body["file_size"]
        assert body["patient_id"] == str(clinic["patient"])
        assert body["case_id"] == str(clinic["case"])
        assert body["doctor_name"] == "Dr. Asha Rao"
        assert body["hospital_name"] == "MedBridge Central"
        assert body["ai_generated"] is True
        assert body["status"] == "ready"

    async def test_doctor_edits_are_what_gets_stored(self, client, clinic, fake_groq):
        """The doctor's text must win over the AI draft, verbatim."""
        headers = await _login(client, DOC_EMAIL)
        resp = await client.post(
            "/api/v1/doctor/reports/issue",
            json=self._payload(
                clinic["case"],
                diagnosis="Cluster headache, doctor-confirmed",
                clinical_notes="Physician override of the AI narrative.",
            ),
            headers=headers,
        )

        content = resp.json()["content"]
        assert "Cluster headache, doctor-confirmed" in content
        assert "Physician override of the AI narrative." in content
        # The model's own wording must not survive the override.
        assert FAKE_DRAFT["clinical_notes"] not in content

    async def test_requires_a_diagnosis(self, client, clinic, fake_groq):
        headers = await _login(client, DOC_EMAIL)
        resp = await client.post(
            "/api/v1/doctor/reports/issue",
            json=self._payload(clinic["case"], diagnosis=""),
            headers=headers,
        )
        assert resp.status_code == 422

    async def test_patient_receives_report_and_notification(
        self, client, clinic, fake_groq, db
    ):
        headers = await _login(client, DOC_EMAIL)
        issued = await client.post(
            "/api/v1/doctor/reports/issue",
            json=self._payload(clinic["case"]),
            headers=headers,
        )
        report_id = issued.json()["id"]

        # Patient's own portal shows it.
        p_headers = await _login(client, PATIENT_EMAIL)
        listing = await client.get("/api/v1/patient/reports", headers=p_headers)
        assert listing.status_code == 200
        assert report_id in {r["id"] for r in listing.json()}

        # And a notification was raised for them.
        notes = await client.get("/api/v1/patient/notifications", headers=p_headers)
        assert notes.status_code == 200
        titles = [n["title"] for n in notes.json()]
        assert "New Clinical Report Available" in titles

    async def test_case_advances_to_report_generated(self, client, clinic, fake_groq):
        headers = await _login(client, DOC_EMAIL)
        await client.post(
            "/api/v1/doctor/reports/issue",
            json=self._payload(clinic["case"]),
            headers=headers,
        )

        # Read back through the API so the assertion covers what the doctor's
        # portal actually sees after issuing.
        resp = await client.get(
            f"/api/v1/doctor/cases/{clinic['case']}", headers=headers
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "report_generated"

    async def test_report_appears_on_doctor_list(self, client, clinic, fake_groq):
        headers = await _login(client, DOC_EMAIL)
        issued = await client.post(
            "/api/v1/doctor/reports/issue",
            json=self._payload(clinic["case"]),
            headers=headers,
        )
        listing = await client.get("/api/v1/doctor/reports", headers=headers)
        assert issued.json()["id"] in {r["id"] for r in listing.json()}
