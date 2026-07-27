"""
Tests for the enriched doctor report cards.

Three things are asserted here beyond field mapping:

* **Superset compatibility** — the card must still carry every `ReportResponse`
  field, because existing callers depend on them.
* **Constant query count** — the whole point of bulk-loading. A regression to
  per-card lookups would still pass every content assertion, so the query count
  is asserted directly.
* **No fabrication** — a patient with no profile data yields nulls and empty
  lists, and a case with no recorded confidence yields no confidence badge
  rather than a "Low Confidence" one.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import event, text

from app.core.security import get_password_hash
from app.models.appointment import Appointment
from app.models.case import Case
from app.models.doctor import Doctor
from app.models.intake import IntakeSessionRecord
from app.models.patient import Patient
from app.models.prescription import Medication, Prescription
from app.models.report import Report
from app.models.user import User
from conftest import login_payload

pytestmark = pytest.mark.asyncio

PW = "password123"
DOC = "cards.doctor@aronofy.com"
OTHER_DOC = "cards.other@aronofy.com"
RICH = "cards.rich@aronofy.com"
BARE = "cards.bare@aronofy.com"

REPORT_RESPONSE_FIELDS = {
    "id", "patient_id", "case_id", "patient_name", "type", "title", "summary",
    "content", "doctor_name", "hospital_name", "date", "status", "file_url",
    "file_size", "ai_generated", "ai_confidence_score", "tags", "vitals",
}


@pytest.fixture
async def cards_fixture(db):
    """
    One richly populated patient and one with a bare profile.

    The bare patient is what makes the "no fabrication" assertions meaningful:
    every optional field genuinely has nothing behind it.
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
        ("doctor", DOC, "doctor"), ("other", OTHER_DOC, "doctor"),
        ("rich", RICH, "patient"), ("bare", BARE, "patient"),
    ):
        u = User(email=email, hashed_password=get_password_hash(PW), role=role, is_verified=True)
        db.add(u)
        await db.flush()
        ids[key] = u.id

    db.add(Doctor(verification_status="verified", id=ids["doctor"], first_name="Asha", last_name="Rao",
                  phone="+910000000001", specialty="Neurology",
                  hospital_name="MedBridge Central", license_number="LIC-CARD-1"))
    db.add(Doctor(verification_status="verified", id=ids["other"], first_name="Vikram", last_name="Sen",
                  phone="+910000000002", specialty="Cardiology",
                  hospital_name="MedBridge East", license_number="LIC-CARD-2"))
    db.add(Patient(id=ids["rich"], first_name="Meera", last_name="Iyer",
                   phone="+910000000003", date_of_birth="1992-03-14", gender="female",
                   blood_type="O+", height=165.0, weight=60.0,
                   allergies=["Penicillin"], chronic_conditions=["Migraine"],
                   medications=["Propranolol 40mg"]))
    db.add(Patient(id=ids["bare"], first_name="Bare", last_name="Record",
                   phone="+910000000004", date_of_birth="2000-01-01", gender="male",
                   allergies=[], chronic_conditions=[], medications=[]))
    await db.flush()

    rich_case = Case(
        patient_id=ids["rich"], patient_name="Meera Iyer", patient_age=34,
        patient_gender="female", doctor_id=ids["doctor"], doctor_name="Dr. Asha Rao",
        specialty="Neurology",
        symptom_summary="Persistent headache for five days with photophobia.",
        urgency_level="high", status="routed",
        ai_extracted_symptoms=["headache", "photophobia"],
        ai_confidence_score=0.82, attachments=[], notes="",
    )
    bare_case = Case(
        patient_id=ids["bare"], patient_name="Bare Record", patient_age=26,
        patient_gender="male", doctor_id=ids["doctor"], doctor_name="Dr. Asha Rao",
        specialty="General Medicine", symptom_summary="Sore throat.",
        urgency_level="low", status="routed", ai_extracted_symptoms=[],
        ai_confidence_score=0.0, attachments=[], notes="",
    )
    db.add_all([rich_case, bare_case])
    await db.flush()
    ids["rich_case"], ids["bare_case"] = rich_case.id, bare_case.id

    db.add(IntakeSessionRecord(
        session_key="cards-1", patient_user_id=ids["rich"], status="routed",
        language="hinglish", intent="symptom_report", overall_confidence=0.82,
        red_flags=["Sudden severe headache"], transcript="Sar dard hai.",
        medical_case_snapshot={
            "chief_complaint": "Persistent headache for five days",
            "symptoms": ["headache", "photophobia"],
            "red_flags": ["Sudden severe headache"],
            "summary_for_doctor": "34F with a five-day headache and photophobia.",
            "overall_confidence": {"score": 0.82, "band": "high"},
        },
        routed_case_id=rich_case.id, routed_doctor_id=ids["doctor"],
    ))

    db.add_all([
        Report(patient_id=ids["rich"], case_id=rich_case.id, patient_name="Meera Iyer",
               type="ai_report", title="AI Clinical Report", summary="Headache assessment.",
               content="Text.", doctor_name="Dr. Asha Rao", date="2026-07-20",
               status="pending_review", file_url="/uploads/reports/a.pdf",
               ai_generated=True, ai_confidence_score=0.82, tags=[], vitals={}),
        Report(patient_id=ids["rich"], patient_name="Meera Iyer", type="lab_result",
               title="Complete Blood Count", summary="CBC normal.", content="Range.",
               doctor_name="Dr. Asha Rao", date="2026-07-01", status="ready",
               file_url="/uploads/reports/cbc.pdf", ai_generated=False, tags=[], vitals={}),
        Report(patient_id=ids["bare"], case_id=bare_case.id, patient_name="Bare Record",
               type="ai_report", title="Sparse Report", summary="", content="Minimal.",
               date="2026-07-21", status="pending", ai_generated=False, tags=[], vitals={}),
    ])
    await db.flush()

    rx = Prescription(case_id=rich_case.id, patient_id=ids["rich"],
                      patient_name="Meera Iyer", doctor_id=ids["doctor"],
                      doctor_name="Dr. Asha Rao", diagnosis="Migraine", notes="",
                      status="active", follow_up_date="2026-08-05")
    db.add(rx)
    await db.flush()
    db.add(Medication(prescription_id=rx.id, name="Sumatriptan", dosage="50mg",
                      frequency="At onset", duration="7 days", status="active",
                      scheduled_times=[], taken_doses=0, total_doses=7,
                      start_date="2026-07-20", end_date="2026-07-27",
                      side_effects=[], interactions=[]))
    db.add(Appointment(patient_id=ids["rich"], doctor_id=ids["doctor"],
                       patient_name="Meera Iyer", doctor_name="Dr. Asha Rao",
                       specialty="Neurology", hospital_name="MedBridge Central",
                       date="2026-07-20", time="10:00", duration=30, type="in_person",
                       status="completed", reason="Headache", notes="", case_id=rich_case.id))
    await db.commit()
    return ids


async def _login(client: AsyncClient, email: str) -> dict[str, str]:
    r = await client.post("/api/v1/auth/login", json=await login_payload(email, PW))
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def _cards(client, headers) -> list[dict]:
    r = await client.get("/api/v1/doctor/reports", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def _by_title(cards: list[dict], title: str) -> dict:
    return next(c for c in cards if c["title"] == title)


class TestBackwardCompatibility:
    async def test_card_is_a_superset_of_report_response(self, client, cards_fixture):
        """Existing consumers read ReportResponse fields; they must all remain."""
        headers = await _login(client, DOC)
        card = _by_title(await _cards(client, headers), "AI Clinical Report")
        assert REPORT_RESPONSE_FIELDS <= set(card)

    async def test_scoping_unchanged(self, client, cards_fixture):
        """Another doctor must still see none of these patients' reports."""
        headers = await _login(client, OTHER_DOC)
        assert await _cards(client, headers) == []


class TestPatientInformation:
    async def test_populates_patient_block(self, client, cards_fixture):
        headers = await _login(client, DOC)
        card = _by_title(await _cards(client, headers), "AI Clinical Report")

        assert card["patient_name"] == "Meera Iyer"
        assert card["patient_age"] == 34
        assert card["patient_gender"] == "female"
        assert card["patient_short_id"] == str(cards_fixture["rich"]).split("-")[0]
        assert card["appointment_date"] == "2026-07-20"
        assert card["assigned_doctor"] == "Dr. Asha Rao"


class TestCaseInformation:
    async def test_populates_case_block(self, client, cards_fixture):
        headers = await _login(client, DOC)
        card = _by_title(await _cards(client, headers), "AI Clinical Report")

        assert "headache" in card["chief_complaint"].lower()
        assert card["extracted_symptoms"] == ["headache", "photophobia"]
        assert card["specialty"] == "Neurology"
        assert card["urgency_level"] == "high"
        assert card["case_status"] == "routed"
        assert card["language_detected"] == "hinglish"
        assert card["case_created_at"] is not None
        assert card["case_updated_at"] is not None

    async def test_confidence_banded_when_recorded(self, client, cards_fixture):
        headers = await _login(client, DOC)
        card = _by_title(await _cards(client, headers), "AI Clinical Report")
        assert card["ai_confidence"]["percentage"] == 82
        assert card["ai_confidence"]["level"] == "High"


class TestMedicalInformation:
    async def test_populates_medical_block(self, client, cards_fixture):
        headers = await _login(client, DOC)
        card = _by_title(await _cards(client, headers), "AI Clinical Report")

        assert card["allergies"] == ["Penicillin"]
        assert card["chronic_conditions"] == ["Migraine"]
        assert card["current_medications"] == ["Propranolol 40mg"]

    async def test_counts_are_real(self, client, cards_fixture):
        headers = await _login(client, DOC)
        card = _by_title(await _cards(client, headers), "AI Clinical Report")

        # Two of the rich patient's reports carry a file_url.
        assert card["uploaded_reports_count"] == 2
        assert card["previous_visits_count"] == 1
        assert card["previous_prescriptions_count"] == 1

    async def test_counts_are_zero_not_invented(self, client, cards_fixture):
        headers = await _login(client, DOC)
        card = _by_title(await _cards(client, headers), "Sparse Report")

        assert card["uploaded_reports_count"] == 0
        assert card["previous_visits_count"] == 0
        assert card["previous_prescriptions_count"] == 0


class TestAISummary:
    async def test_reuses_the_stored_intake_summary(self, client, cards_fixture):
        headers = await _login(client, DOC)
        card = _by_title(await _cards(client, headers), "AI Clinical Report")
        assert card["ai_summary"] == "34F with a five-day headache and photophobia."

    async def test_falls_back_to_report_summary(self, client, cards_fixture):
        """No intake session -> the report's own summary, not a generated one."""
        headers = await _login(client, DOC)
        card = _by_title(await _cards(client, headers), "Complete Blood Count")
        assert card["ai_summary"] == "CBC normal."

    async def test_unlinked_report_never_borrows_another_cases_narrative(
        self, client, cards_fixture
    ):
        """
        A report with no `case_id` must not inherit the patient's open case.

        The CBC below belongs to a patient whose other case is an active
        headache consultation. Attaching that case would print a headache chief
        complaint, urgency and AI summary onto a blood test — a document that
        reads as being about a condition it has nothing to do with.
        """
        headers = await _login(client, DOC)
        card = _by_title(await _cards(client, headers), "Complete Blood Count")

        assert card["case_id"] is None
        assert card["chief_complaint"] is None
        assert card["urgency_level"] is None
        assert card["case_status"] is None
        assert card["language_detected"] is None
        assert card["extracted_symptoms"] == []
        assert card["ai_confidence"] is None
        assert "headache" not in card["ai_summary"].lower()

        # Patient-level facts remain correct, because they are not case-derived.
        assert card["patient_age"] is not None
        assert card["allergies"] == ["Penicillin"]
        assert card["previous_visits_count"] == 1


class TestIndicators:
    async def test_derives_badges_from_stored_values(self, client, cards_fixture):
        headers = await _login(client, DOC)
        card = _by_title(await _cards(client, headers), "AI Clinical Report")
        labels = {i["label"] for i in card["indicators"]}

        assert "High Confidence" in labels
        assert "Emergency" in labels          # red flag on the intake record
        assert "Needs Review" in labels       # status pending_review
        assert "Follow-up Required" in labels  # prescription has a follow-up date

    async def test_no_confidence_badge_when_never_scored(self, client, cards_fixture):
        """A missing score must not surface as 'Low Confidence'."""
        headers = await _login(client, DOC)
        card = _by_title(await _cards(client, headers), "Sparse Report")
        labels = {i["label"] for i in card["indicators"]}

        assert card["ai_confidence"] is None
        assert not any("Confidence" in label for label in labels)
        assert "Emergency" not in labels

    async def test_awaiting_reports_when_no_documents(self, client, cards_fixture):
        headers = await _login(client, DOC)
        card = _by_title(await _cards(client, headers), "Sparse Report")
        assert "Awaiting Reports" in {i["label"] for i in card["indicators"]}

    async def test_badge_tones_are_valid_variants(self, client, cards_fixture):
        headers = await _login(client, DOC)
        valid = {"success", "warning", "error", "info", "neutral"}
        for card in await _cards(client, headers):
            for indicator in card["indicators"]:
                assert indicator["tone"] in valid


class TestNoFabrication:
    async def test_absent_profile_data_stays_absent(self, client, cards_fixture):
        headers = await _login(client, DOC)
        card = _by_title(await _cards(client, headers), "Sparse Report")

        assert card["allergies"] == []
        assert card["chronic_conditions"] == []
        assert card["current_medications"] == []
        assert card["appointment_date"] is None
        assert card["language_detected"] is None
        assert card["extracted_symptoms"] == []


class TestQueryEfficiency:
    async def test_query_count_does_not_grow_with_report_count(
        self, client, cards_fixture, db
    ):
        """
        The bulk-loading guarantee, asserted directly.

        Content assertions would all still pass if this regressed to a per-card
        lookup, so the query count is measured instead. Adding six more reports
        must not add six more round trips.
        """
        headers = await _login(client, DOC)

        from app.core.database import engine

        counter = {"n": 0}

        def _count(conn, cursor, statement, params, context, executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                counter["n"] += 1

        sync_engine = engine.sync_engine
        event.listen(sync_engine, "before_cursor_execute", _count)
        try:
            await client.get("/api/v1/doctor/reports", headers=headers)
            baseline = counter["n"]

            for i in range(6):
                db.add(Report(
                    patient_id=cards_fixture["rich"], patient_name="Meera Iyer",
                    type="ai_report", title=f"Extra {i}", summary="s", content="c",
                    date="2026-07-22", status="ready", ai_generated=False,
                    tags=[], vitals={},
                ))
            await db.commit()

            counter["n"] = 0
            cards = await _cards(client, headers)
            grown = counter["n"]
        finally:
            event.remove(sync_engine, "before_cursor_execute", _count)

        assert len(cards) == 9, "fixture should now hold nine reports"
        assert grown <= baseline, (
            f"query count grew from {baseline} to {grown} after adding 6 reports — "
            "the card projection is no longer bulk-loaded"
        )


class TestAuthorization:
    async def test_patient_cannot_list_doctor_reports(self, client, cards_fixture):
        headers = await _login(client, RICH)
        r = await client.get("/api/v1/doctor/reports", headers=headers)
        assert r.status_code == 403

    async def test_request_more_information_uses_existing_status(
        self, client, cards_fixture
    ):
        """The new quick action reuses needs_revision rather than a new concept."""
        headers = await _login(client, DOC)
        card = _by_title(await _cards(client, headers), "AI Clinical Report")

        r = await client.put(
            f"/api/v1/doctor/reports/{card['id']}/status",
            params={"status_str": "needs_revision"},
            headers=headers,
        )
        assert r.status_code == 200, r.text

        refreshed = _by_title(await _cards(client, headers), "AI Clinical Report")
        assert refreshed["status"] == "needs_revision"
        assert "More Information Requested" in {
            i["label"] for i in refreshed["indicators"]
        }
