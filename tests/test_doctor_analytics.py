"""
Tests for the doctor analytics dashboard.

What these assert, in order of consequence:

* **Isolation.** Every figure is scoped through `cases.doctor_id`. A doctor's
  dashboard must never include another clinician's patients — an analytics leak
  is quieter than a record leak and just as real.
* **No fabrication.** A metric with nothing behind it is `None`, not 0. A zero
  average consultation time claims instantaneous consultations; "not measured"
  claims nothing.
* **Diagnoses are clinician-written.** `common_diagnoses` comes from
  prescriptions. AI-suggested differentials must never appear there, or model
  output starts reading as clinical consensus.
* **Date ranges actually filter.** A metric that ignores the range is worse
  than no filter, because the number looks responsive and is not.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from app.core.security import get_password_hash
from app.models.appointment import Appointment
from app.models.case import Case, Symptom
from app.models.doctor import Doctor
from app.models.intake import IntakeSessionRecord
from app.models.notification import NotificationItem
from app.models.patient import Patient
from app.models.prescription import Medication, Prescription
from app.models.report import Report
from app.models.user import User
from app.services.doctor_analytics import resolve_range
from conftest import login_payload

pytestmark = pytest.mark.asyncio

PW = "password123"
DOC_A = "an.doca@aronofy.com"
DOC_B = "an.docb@aronofy.com"
PAT_A = "an.pata@aronofy.com"
PAT_B = "an.patb@aronofy.com"


@pytest.fixture
async def estate(db):
    """Doctor A with a populated caseload; doctor B with a separate one."""
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("audit_logs", "report_versions", "intake_extracted_entities",
                  "intake_sessions", "notifications", "medications", "prescriptions",
                  "appointments", "reports", "symptoms", "cases", "doctors",
                  "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")

    ids: dict[str, Any] = {}
    for k, e, r in (("doc_a", DOC_A, "doctor"), ("doc_b", DOC_B, "doctor"),
                    ("pat_a", PAT_A, "patient"), ("pat_b", PAT_B, "patient")):
        u = User(email=e, hashed_password=get_password_hash(PW), role=r, is_verified=True)
        db.add(u); await db.flush(); ids[k] = u.id

    db.add(Doctor(id=ids["doc_a"], first_name="Asha", last_name="Rao", phone="+911",
                  specialty="Neurology", hospital_name="Central",
                  license_number="LIC-AN-A", verification_status="verified"))
    db.add(Doctor(id=ids["doc_b"], first_name="Vikram", last_name="Sen", phone="+912",
                  specialty="Cardiology", hospital_name="East",
                  license_number="LIC-AN-B", verification_status="verified"))
    db.add(Patient(id=ids["pat_a"], first_name="Meera", last_name="Iyer", phone="+913",
                   date_of_birth="1992-03-14", gender="female",
                   allergies=[], chronic_conditions=[], medications=[]))
    db.add(Patient(id=ids["pat_b"], first_name="Rahul", last_name="Nair", phone="+914",
                   date_of_birth="1960-06-02", gender="male",
                   allergies=[], chronic_conditions=[], medications=[]))
    await db.flush()

    def mk(pid, did, dn, spec, urgency, **kw):
        return Case(patient_id=pid, patient_name="P", patient_age=34,
                    patient_gender="female", doctor_id=did, doctor_name=dn,
                    specialty=spec, symptom_summary="s", urgency_level=urgency,
                    status="routed", ai_extracted_symptoms=[],
                    ai_confidence_score=0.0, attachments=[], notes="", **kw)

    # Doctor A: one completed case with timings, one critical open case.
    completed = mk(ids["pat_a"], ids["doc_a"], "Dr. Asha Rao", "Neurology", "medium",
                   assigned_at=now - timedelta(hours=2), completed_at=now)
    completed.status = "completed"
    completed.ai_confidence_score = 0.9
    critical = mk(ids["pat_a"], ids["doc_a"], "Dr. Asha Rao", "Neurology", "critical")
    # Doctor B's case — must never appear in doctor A's analytics.
    foreign = mk(ids["pat_b"], ids["doc_b"], "Dr. Vikram Sen", "Cardiology", "high")
    db.add_all([completed, critical, foreign])
    await db.flush()
    ids.update(completed=completed.id, critical=critical.id, foreign=foreign.id)

    db.add(Symptom(case_id=completed.id, name="headache", severity="moderate",
                   duration="5 days", body_part="head"))
    db.add(Symptom(case_id=foreign.id, name="FOREIGN_SYMPTOM", severity="mild",
                   duration="1 day", body_part="chest"))

    db.add(IntakeSessionRecord(
        session_key="an-1", patient_user_id=ids["pat_a"], status="routed",
        language="english", intent="symptom_report", overall_confidence=0.9,
        red_flags=[], transcript="t", medical_case_snapshot={"chief_complaint": "c"},
        routed_case_id=completed.id, routed_doctor_id=ids["doc_a"]))

    db.add(Report(patient_id=ids["pat_a"], case_id=completed.id,
                  patient_name="Meera Iyer", type="ai_report", title="R1",
                  summary="s", content="c", date=today_str, status="pending_review",
                  ai_generated=True, tags=[], vitals={}))
    db.add(Report(patient_id=ids["pat_a"], case_id=completed.id,
                  patient_name="Meera Iyer", type="ai_report", title="R2",
                  summary="s", content="c", date=today_str, status="ready",
                  ai_generated=True, flagged_for_follow_up=True, tags=[], vitals={}))
    db.add(Report(patient_id=ids["pat_b"], case_id=foreign.id,
                  patient_name="Rahul Nair", type="ai_report", title="FOREIGN_REPORT",
                  summary="s", content="c", date=today_str, status="ready",
                  ai_generated=True, tags=[], vitals={}))

    rx = Prescription(case_id=completed.id, patient_id=ids["pat_a"],
                      patient_name="Meera Iyer", doctor_id=ids["doc_a"],
                      doctor_name="Dr. Asha Rao", diagnosis="Migraine", notes="",
                      status="active", follow_up_date="2026-09-01")
    foreign_rx = Prescription(case_id=foreign.id, patient_id=ids["pat_b"],
                              patient_name="Rahul Nair", doctor_id=ids["doc_b"],
                              doctor_name="Dr. Vikram Sen",
                              diagnosis="FOREIGN_DIAGNOSIS", notes="", status="active")
    db.add_all([rx, foreign_rx])
    await db.flush()
    db.add(Medication(prescription_id=rx.id, name="Sumatriptan", dosage="50mg",
                      frequency="daily", duration="5 days", status="active",
                      scheduled_times=[], taken_doses=0, total_doses=5,
                      start_date=today_str, end_date=today_str,
                      side_effects=[], interactions=[]))

    db.add(Appointment(patient_id=ids["pat_a"], doctor_id=ids["doc_a"],
                       patient_name="Meera Iyer", doctor_name="Dr. Asha Rao",
                       specialty="Neurology", hospital_name="Central",
                       date=today_str, time="10:00", duration=30, type="in_person",
                       status="completed", reason="Headache", notes="",
                       case_id=completed.id))
    db.add(Appointment(patient_id=ids["pat_a"], doctor_id=ids["doc_a"],
                       patient_name="Meera Iyer", doctor_name="Dr. Asha Rao",
                       specialty="Neurology", hospital_name="Central",
                       date=today_str, time="14:00", duration=30, type="in_person",
                       status="no_show", reason="Follow-up", notes=""))
    db.add(NotificationItem(user_id=ids["doc_a"], type="alert", title="T",
                            message="m", timestamp=now.isoformat(), read=False,
                            priority="low"))
    await db.commit()
    return ids


async def _login(client: AsyncClient, email: str) -> dict[str, str]:
    r = await client.post("/api/v1/auth/login", json=await login_payload(email, PW))
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def _analytics(client, headers, **params) -> dict:
    r = await client.get("/api/v1/doctor/analytics", params=params, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


class TestBackwardCompatibility:
    async def test_legacy_fields_are_preserved(self, client, estate):
        """Existing consumers read these; they must not disappear."""
        headers = await _login(client, DOC_A)
        body = await _analytics(client, headers)
        for key in ("age_distribution", "status_distribution", "adherence_rate",
                    "case_trend", "specialty_distribution"):
            assert key in body


class TestSummary:
    async def test_summary_counts_are_real(self, client, estate):
        headers = await _login(client, DOC_A)
        s = (await _analytics(client, headers))["summary"]

        assert s["todays_appointments"] == 2
        assert s["patients_seen_today"] == 1
        assert s["critical_cases"] == 1
        assert s["pending_reports"] == 1
        assert s["follow_up_cases"] == 1
        assert s["unread_notifications"] == 1


class TestWorkload:
    async def test_case_counts_and_groupings(self, client, estate):
        headers = await _login(client, DOC_A)
        w = (await _analytics(client, headers))["workload"]

        assert w["cases_opened"] == 2
        assert w["cases_completed"] == 1
        assert w["pending_cases"] == 1
        assert {g["name"] for g in w["cases_by_urgency"]} == {"medium", "critical"}
        assert [g["name"] for g in w["cases_by_specialty"]] == ["Neurology"]

    async def test_average_consultation_time_is_measured(self, client, estate):
        """Two hours between assignment and completion."""
        headers = await _login(client, DOC_A)
        w = (await _analytics(client, headers))["workload"]
        assert w["avg_consultation_minutes"] == pytest.approx(120, abs=2)

    async def test_unmeasurable_average_is_null_not_zero(self, client, estate):
        """Doctor B has no completed cases, so there is no average to report."""
        headers = await _login(client, DOC_B)
        w = (await _analytics(client, headers))["workload"]
        assert w["avg_consultation_minutes"] is None
        assert w["avg_review_minutes"] is None


class TestPatients:
    async def test_demographics_from_real_rows(self, client, estate):
        headers = await _login(client, DOC_A)
        p = (await _analytics(client, headers))["patients"]

        assert p["new_patients"] == 1
        assert [g["name"] for g in p["gender_distribution"]] == ["female"]
        assert sum(b["value"] for b in p["age_distribution"]) == 1
        assert "headache" in {s["name"] for s in p["common_symptoms"]}

    async def test_diagnoses_come_from_prescriptions_not_ai(self, client, estate):
        headers = await _login(client, DOC_A)
        p = (await _analytics(client, headers))["patients"]
        assert [d["name"] for d in p["common_diagnoses"]] == ["Migraine"]
        assert "prescription" in p["diagnoses_source"].lower()


class TestAI:
    async def test_ai_metrics_are_measured_values(self, client, estate):
        headers = await _login(client, DOC_A)
        ai = (await _analytics(client, headers))["ai"]

        assert ai["analyses_generated"] == 1
        assert ai["avg_confidence_percent"] == pytest.approx(90.0, abs=0.1)

    async def test_processing_time_is_null_because_it_is_not_recorded(
        self, client, estate
    ):
        headers = await _login(client, DOC_A)
        body = await _analytics(client, headers)
        assert body["ai"]["avg_processing_time_seconds"] is None
        metrics = {u["metric"] for u in body["unavailable_metrics"]}
        assert "avg_ai_processing_time_seconds" in metrics

    async def test_accuracy_is_never_reported(self, client, estate):
        """No validated ground truth exists, so accuracy must not appear."""
        headers = await _login(client, DOC_A)
        assert "accuracy" not in str(await _analytics(client, headers)).lower()

    async def test_confidence_is_null_when_nothing_was_scored(self, client, estate):
        headers = await _login(client, DOC_B)
        assert (await _analytics(client, headers))["ai"]["avg_confidence_percent"] is None


class TestReportsAndPrescriptions:
    async def test_report_status_breakdown(self, client, estate):
        headers = await _login(client, DOC_A)
        r = (await _analytics(client, headers))["reports"]
        assert r["generated"] == 2
        assert r["approved"] == 1
        assert r["pending"] == 1

    async def test_prescription_metrics(self, client, estate):
        headers = await _login(client, DOC_A)
        p = (await _analytics(client, headers))["prescriptions"]
        assert p["issued"] == 1
        assert p["follow_up_prescriptions"] == 1
        assert [m["name"] for m in p["top_medications"]] == ["Sumatriptan"]

    async def test_medication_categories_absent_with_a_stated_reason(
        self, client, estate
    ):
        headers = await _login(client, DOC_A)
        body = await _analytics(client, headers)
        assert body["prescriptions"]["medication_categories"] == []
        reasons = {u["metric"]: u["reason"] for u in body["unavailable_metrics"]}
        assert "classification" in reasons["medication_categories"].lower()


class TestAppointments:
    async def test_appointment_outcomes(self, client, estate):
        headers = await _login(client, DOC_A)
        a = (await _analytics(client, headers))["appointments"]
        assert a["today"] == 2
        assert a["completed"] == 1
        assert a["no_show"] == 1


class TestDateRange:
    async def test_range_metadata_is_returned(self, client, estate):
        headers = await _login(client, DOC_A)
        body = await _analytics(client, headers, range="7d")
        assert body["range"]["preset"] == "7d"
        assert body["range"]["date_from"] < body["range"]["date_to"]

    async def test_a_past_window_excludes_current_activity(self, client, estate):
        """Cases created today must not appear in a window that ended in 2020."""
        headers = await _login(client, DOC_A)
        body = await _analytics(client, headers, range="custom",
                                date_from="2020-01-01", date_to="2020-01-02")
        assert body["workload"]["cases_opened"] == 0
        assert body["reports"]["generated"] == 0
        assert body["prescriptions"]["issued"] == 0

    async def test_today_preset_includes_today(self, client, estate):
        headers = await _login(client, DOC_A)
        body = await _analytics(client, headers, range="today")
        assert body["workload"]["cases_opened"] == 2

    def test_resolve_range_swaps_inverted_dates(self):
        start, end, label = resolve_range("custom", "2026-05-10", "2026-05-01")
        assert start.date().isoformat() == "2026-05-01"
        assert end.date().isoformat() == "2026-05-10"
        assert label == "custom"

    def test_resolve_range_falls_back_on_bad_input(self):
        start, end, _ = resolve_range("custom", "not-a-date", None)
        assert (end.date() - start.date()).days == 29


class TestIsolation:
    async def test_doctor_a_sees_none_of_doctor_b_data(self, client, estate):
        headers = await _login(client, DOC_A)
        payload = str(await _analytics(client, headers))
        assert "FOREIGN_SYMPTOM" not in payload
        assert "FOREIGN_DIAGNOSIS" not in payload
        assert "FOREIGN_REPORT" not in payload
        assert "Cardiology" not in payload

    async def test_doctor_b_sees_only_their_own(self, client, estate):
        headers = await _login(client, DOC_B)
        body = await _analytics(client, headers)
        assert body["workload"]["cases_opened"] == 1
        assert [g["name"] for g in body["workload"]["cases_by_specialty"]] == ["Cardiology"]
        assert "Migraine" not in str(body["patients"]["common_diagnoses"])

    async def test_patient_cannot_read_doctor_analytics(self, client, estate):
        headers = await _login(client, PAT_A)
        r = await client.get("/api/v1/doctor/analytics", headers=headers)
        assert r.status_code == 403


class TestExport:
    async def test_csv_export_matches_the_dashboard(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.get("/api/v1/doctor/analytics/export",
                             params={"format": "csv"}, headers=headers)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        assert "Section,Metric,Value" in r.text
        assert "Critical Cases" in r.text
        # Absent metrics stay visibly absent in the export.
        assert "Not measured" in r.text

    async def test_csv_excludes_other_doctor_data(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.get("/api/v1/doctor/analytics/export",
                             params={"format": "csv"}, headers=headers)
        assert "FOREIGN_DIAGNOSIS" not in r.text

    async def test_pdf_export_renders(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.get("/api/v1/doctor/analytics/export",
                             params={"format": "pdf"}, headers=headers)
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"
        assert r.content[:5] == b"%PDF-"

    async def test_patient_cannot_export(self, client, estate):
        headers = await _login(client, PAT_A)
        r = await client.get("/api/v1/doctor/analytics/export",
                             params={"format": "csv"}, headers=headers)
        assert r.status_code == 403
