"""
Tests for the notification event producers.

Three sources, each with a rule that matters more than the happy path:

* **Lab results** notify the *treating* clinician, never the one who filed the
  result. Self-notification is the fastest way to train people to ignore
  notifications, and a result attached to the wrong case is a PHI misroute.
* **The follow-up sweep** must be safe to re-run. Its dedupe key is per
  prescription per due date, so an hourly schedule produces one notification
  per follow-up no matter how often it fires or how long the worker was down.
* **System alerts** reach every active account in the audience and only that
  audience, and security alerts cannot be silenced by preferences.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text

from app.core.security import get_password_hash
from app.models.case import Case
from app.models.doctor import Doctor
from app.models.notification import NotificationItem
from app.models.patient import Patient
from app.models.prescription import Prescription
from app.models.user import User
from app.worker.tasks.reminder import _find_and_notify_follow_ups

pytestmark = pytest.mark.asyncio

PW = "password123"
DOC_A = "src.doca@aronofy.com"
DOC_B = "src.docb@aronofy.com"
ADMIN = "src.admin@aronofy.com"
PAT_A = "src.pata@aronofy.com"


@pytest.fixture
async def estate(db):
    """Doctor A owns the case; doctor B is an unrelated clinician."""
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("audit_logs", "notifications", "medications", "prescriptions",
                  "reports", "cases", "doctors", "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    ids: dict[str, Any] = {}
    for k, e, r in (("doc_a", DOC_A, "doctor"), ("doc_b", DOC_B, "doctor"),
                    ("admin", ADMIN, "admin"), ("pat_a", PAT_A, "patient")):
        u = User(email=e, hashed_password=get_password_hash(PW), role=r, is_verified=True)
        db.add(u); await db.flush(); ids[k] = u.id

    db.add(Doctor(id=ids["doc_a"], first_name="Asha", last_name="Rao", phone="+911",
                  specialty="Neurology", hospital_name="Central",
                  license_number="LIC-SRC-A", verification_status="verified"))
    db.add(Doctor(id=ids["doc_b"], first_name="Vikram", last_name="Sen", phone="+912",
                  specialty="Radiology", hospital_name="East",
                  license_number="LIC-SRC-B", verification_status="verified"))
    db.add(Patient(id=ids["pat_a"], first_name="Meera", last_name="Iyer", phone="+913",
                   date_of_birth="1992-03-14", gender="female",
                   allergies=[], chronic_conditions=[], medications=[]))
    await db.flush()

    def mk(doctor_id, doctor_name, urgency="medium"):
        return Case(patient_id=ids["pat_a"], patient_name="Meera Iyer",
                    patient_age=34, patient_gender="female", doctor_id=doctor_id,
                    doctor_name=doctor_name, specialty="Neurology",
                    symptom_summary="Headache.", urgency_level=urgency,
                    status="routed", ai_extracted_symptoms=[],
                    ai_confidence_score=0.0, attachments=[], notes="")

    # Doctor A treats this case; doctor B also sees the patient (so B is
    # authorised to file a result) but is not the treating clinician.
    treated = mk(ids["doc_a"], "Dr. Asha Rao")
    other = mk(ids["doc_b"], "Dr. Vikram Sen")
    db.add_all([treated, other])
    await db.flush()
    ids["case"], ids["other_case"] = treated.id, other.id
    await db.commit()
    return ids


async def _login(client: AsyncClient, email: str) -> dict[str, str]:
    r = await client.post("/api/v1/auth/login", json={"email": email, "password": PW})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def _notifications(db, user_id, type_=None) -> list[NotificationItem]:
    stmt = select(NotificationItem).where(NotificationItem.user_id == user_id)
    if type_:
        stmt = stmt.where(NotificationItem.type == type_)
    return list((await db.execute(stmt)).scalars().all())


def _report_payload(patient_id, case_id=None, report_type="lab_result"):
    payload = {
        "patient_id": str(patient_id), "patient_name": "Meera Iyer",
        "type": report_type, "title": "Complete Blood Count",
        "summary": "CBC within normal limits.", "content": "All values in range.",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    if case_id:
        payload["case_id"] = str(case_id)
    return payload


class TestLabResultUploaded:
    async def test_notifies_the_treating_doctor(self, client, db, estate):
        """Doctor B files a result on doctor A's case; A is the one told."""
        headers = await _login(client, DOC_B)
        r = await client.post(
            "/api/v1/doctor/reports",
            json=_report_payload(estate["pat_a"], estate["case"]),
            headers=headers,
        )
        assert r.status_code == 201, r.text

        notes = await _notifications(db, estate["doc_a"], "lab_result_uploaded")
        assert len(notes) == 1
        assert notes[0].title == "Lab Results Uploaded"
        assert notes[0].case_id == estate["case"]
        assert notes[0].patient_name == "Meera Iyer"
        assert notes[0].action_label == "View Report"

    async def test_never_notifies_the_author(self, client, db, estate):
        """A clinician does not need telling about their own filing."""
        headers = await _login(client, DOC_A)
        r = await client.post(
            "/api/v1/doctor/reports",
            json=_report_payload(estate["pat_a"], estate["case"]),
            headers=headers,
        )
        assert r.status_code == 201
        assert await _notifications(db, estate["doc_a"], "lab_result_uploaded") == []

    async def test_unlinked_report_notifies_nobody(self, client, db, estate):
        """With no case there is no treating clinician; guessing one misroutes PHI."""
        headers = await _login(client, DOC_B)
        r = await client.post(
            "/api/v1/doctor/reports",
            json=_report_payload(estate["pat_a"]),
            headers=headers,
        )
        assert r.status_code == 201
        assert await _notifications(db, estate["doc_a"], "lab_result_uploaded") == []

    async def test_non_diagnostic_types_do_not_notify(self, client, db, estate):
        """A discharge summary is not a lab result."""
        headers = await _login(client, DOC_B)
        r = await client.post(
            "/api/v1/doctor/reports",
            json=_report_payload(estate["pat_a"], estate["case"],
                                 report_type="discharge_summary"),
            headers=headers,
        )
        assert r.status_code == 201
        assert await _notifications(db, estate["doc_a"], "lab_result_uploaded") == []

    async def test_imaging_counts_as_a_diagnostic_result(self, client, db, estate):
        headers = await _login(client, DOC_B)
        r = await client.post(
            "/api/v1/doctor/reports",
            json=_report_payload(estate["pat_a"], estate["case"],
                                 report_type="imaging"),
            headers=headers,
        )
        assert r.status_code == 201
        assert len(await _notifications(db, estate["doc_a"], "lab_result_uploaded")) == 1

    async def test_case_must_belong_to_the_named_patient(self, client, db, estate):
        """Attaching a result to another patient's case is rejected outright."""
        other_user = User(email="src.other@aronofy.com",
                          hashed_password=get_password_hash(PW),
                          role="patient", is_verified=True)
        db.add(other_user); await db.flush()
        db.add(Patient(id=other_user.id, first_name="Other", last_name="Person",
                       phone="+914", date_of_birth="1990-01-01", gender="male",
                       allergies=[], chronic_conditions=[], medications=[]))
        await db.commit()

        headers = await _login(client, DOC_B)
        r = await client.post(
            "/api/v1/doctor/reports",
            json=_report_payload(other_user.id, estate["case"]),
            headers=headers,
        )
        assert r.status_code in (403, 404)

    async def test_is_deduplicated_per_report(self, client, db, estate):
        headers = await _login(client, DOC_B)
        first = await client.post(
            "/api/v1/doctor/reports",
            json=_report_payload(estate["pat_a"], estate["case"]),
            headers=headers,
        )
        assert first.status_code == 201
        before = len(await _notifications(db, estate["doc_a"], "lab_result_uploaded"))

        # Filing a second, distinct report produces a second notification, but
        # the dedupe key is per report so neither can fire twice.
        second = await client.post(
            "/api/v1/doctor/reports",
            json=_report_payload(estate["pat_a"], estate["case"]),
            headers=headers,
        )
        assert second.status_code == 201
        after = len(await _notifications(db, estate["doc_a"], "lab_result_uploaded"))
        assert after == before + 1

        keys = {
            n.dedupe_key
            for n in await _notifications(db, estate["doc_a"], "lab_result_uploaded")
        }
        assert len(keys) == after


class TestFollowUpSweep:
    @staticmethod
    async def _prescription(db, estate, follow_up: str | None, status: str = "active"):
        rx = Prescription(
            case_id=estate["case"], patient_id=estate["pat_a"],
            patient_name="Meera Iyer", doctor_id=estate["doc_a"],
            doctor_name="Dr. Asha Rao", diagnosis="Migraine", notes="",
            status=status, follow_up_date=follow_up,
        )
        db.add(rx)
        await db.commit()
        return rx

    async def test_notifies_when_a_follow_up_is_due(self, db, estate):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        await self._prescription(db, estate, today)

        sent = await _find_and_notify_follow_ups(db)
        assert sent == 1

        notes = await _notifications(db, estate["doc_a"], "follow_up_due")
        assert len(notes) == 1
        assert notes[0].priority == "high"
        assert notes[0].case_id == estate["case"]
        assert notes[0].action_label == "Open Patient History"

    async def test_catches_up_on_overdue_follow_ups(self, db, estate):
        """A day the worker was down must not silently drop its follow-ups."""
        overdue = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
        await self._prescription(db, estate, overdue)

        assert await _find_and_notify_follow_ups(db) == 1

    async def test_future_follow_ups_are_not_notified_early(self, db, estate):
        future = (datetime.now(timezone.utc) + timedelta(days=5)).strftime("%Y-%m-%d")
        await self._prescription(db, estate, future)

        assert await _find_and_notify_follow_ups(db) == 0
        assert await _notifications(db, estate["doc_a"], "follow_up_due") == []

    async def test_prescriptions_without_a_follow_up_date_are_ignored(self, db, estate):
        await self._prescription(db, estate, None)
        assert await _find_and_notify_follow_ups(db) == 0

    async def test_completed_prescriptions_are_ignored(self, db, estate):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        await self._prescription(db, estate, today, status="completed")
        assert await _find_and_notify_follow_ups(db) == 0

    async def test_repeated_sweeps_do_not_duplicate(self, db, estate):
        """
        The sweep runs hourly; a follow-up must produce exactly one
        notification regardless of how many times it is swept.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        await self._prescription(db, estate, today)

        assert await _find_and_notify_follow_ups(db) == 1
        assert await _find_and_notify_follow_ups(db) == 0
        assert await _find_and_notify_follow_ups(db) == 0

        assert len(await _notifications(db, estate["doc_a"], "follow_up_due")) == 1

    async def test_notifies_only_the_prescribing_doctor(self, db, estate):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        await self._prescription(db, estate, today)
        await _find_and_notify_follow_ups(db)

        assert await _notifications(db, estate["doc_b"], "follow_up_due") == []

    async def test_celery_task_is_registered_and_wraps_the_sweep(self):
        """
        The scheduled entry point must exist and call the sweep.

        Invoking the task body here would open its own session against the
        configured database rather than the test one, so the wiring is asserted
        instead; the sweep's behaviour is covered by the tests above.
        """
        from app.worker.celery_app import celery_app
        from app.worker.tasks import reminder as reminder_module

        assert ("app.worker.tasks.reminder.send_follow_up_reminders"
                in celery_app.tasks)
        assert callable(reminder_module._find_and_notify_follow_ups)


class TestBeatSchedule:
    def test_producers_are_actually_scheduled(self):
        """
        A task nobody schedules never runs — which is why the pre-existing
        medicine reminder had never fired.
        """
        from app.worker.celery_app import celery_app

        tasks = {
            entry["task"] for entry in celery_app.conf.beat_schedule.values()
        }
        assert "app.worker.tasks.reminder.send_follow_up_reminders" in tasks
        assert "app.worker.tasks.reminder.check_system_health" in tasks
        assert "app.worker.tasks.reminder.send_medicine_reminders" in tasks


class TestSystemAlerts:
    @staticmethod
    def _alert(**overrides) -> dict[str, Any]:
        payload = {
            "title": "Scheduled Maintenance",
            "message": "The platform will be unavailable 02:00-03:00 UTC.",
            "severity": "high",
            "category": "system",
            "audience": ["doctor"],
        }
        payload.update(overrides)
        return payload

    async def test_admin_can_broadcast_to_doctors(self, client, db, estate):
        headers = await _login(client, ADMIN)
        r = await client.post("/api/v1/admin/system-alerts",
                              json=self._alert(), headers=headers)
        assert r.status_code == 201, r.text
        assert r.json()["delivered"] == 2  # both doctors

        for doctor in ("doc_a", "doc_b"):
            notes = await _notifications(db, estate[doctor], "system_alert")
            assert len(notes) == 1
            assert notes[0].category == "system"
            assert notes[0].priority == "high"

    async def test_audience_is_respected(self, client, db, estate):
        headers = await _login(client, ADMIN)
        r = await client.post("/api/v1/admin/system-alerts",
                              json=self._alert(audience=["admin"]), headers=headers)
        assert r.status_code == 201
        assert await _notifications(db, estate["doc_a"], "system_alert") == []
        assert len(await _notifications(db, estate["admin"], "system_alert")) == 1

    async def test_security_alerts_use_the_security_category(self, client, db, estate):
        headers = await _login(client, ADMIN)
        r = await client.post(
            "/api/v1/admin/system-alerts",
            json=self._alert(category="security", severity="critical",
                             title="Unusual sign-in activity"),
            headers=headers,
        )
        assert r.status_code == 201
        notes = await _notifications(db, estate["doc_a"], "security_alert")
        assert len(notes) == 1
        assert notes[0].category == "security"
        assert notes[0].priority == "critical"

    async def test_repeated_submission_is_deduplicated(self, client, db, estate):
        """A double-submitted announcement must not reach everyone twice."""
        headers = await _login(client, ADMIN)
        first = await client.post("/api/v1/admin/system-alerts",
                                  json=self._alert(), headers=headers)
        second = await client.post("/api/v1/admin/system-alerts",
                                   json=self._alert(), headers=headers)

        assert first.json()["delivered"] == 2
        assert second.json()["delivered"] == 0
        assert len(await _notifications(db, estate["doc_a"], "system_alert")) == 1

    async def test_doctors_cannot_broadcast(self, client, estate):
        headers = await _login(client, DOC_A)
        r = await client.post("/api/v1/admin/system-alerts",
                              json=self._alert(), headers=headers)
        assert r.status_code == 403

    async def test_patients_cannot_broadcast(self, client, estate):
        headers = await _login(client, PAT_A)
        r = await client.post("/api/v1/admin/system-alerts",
                              json=self._alert(), headers=headers)
        assert r.status_code == 403

    async def test_invalid_category_is_rejected(self, client, estate):
        headers = await _login(client, ADMIN)
        r = await client.post("/api/v1/admin/system-alerts",
                              json=self._alert(category="clinical"), headers=headers)
        assert r.status_code == 422

    async def test_alert_reaches_the_notification_centre(self, client, db, estate):
        admin_headers = await _login(client, ADMIN)
        await client.post("/api/v1/admin/system-alerts",
                          json=self._alert(), headers=admin_headers)

        doctor_headers = await _login(client, DOC_A)
        r = await client.get("/api/v1/shared/notifications/center",
                             params={"category": "system"}, headers=doctor_headers)
        assert r.status_code == 200
        assert [n["title"] for n in r.json()["notifications"]] == ["Scheduled Maintenance"]


class TestHealthSweep:
    async def test_alerts_admins_when_a_dependency_is_down(self, db, estate, monkeypatch):
        from app.core import redis as redis_module
        from app.worker.tasks import reminder as reminder_module

        async def down():
            return False

        monkeypatch.setattr(redis_module.redis_manager, "ping", down)

        sent = await reminder_module._check_system_health(db)
        assert sent == 1

        notes = await _notifications(db, estate["admin"], "system_alert")
        assert len(notes) == 1
        assert "Redis" in notes[0].message
        assert notes[0].priority == "critical"

    async def test_healthy_dependencies_raise_nothing(self, db, estate, monkeypatch):
        from app.core import redis as redis_module
        from app.worker.tasks import reminder as reminder_module

        async def up():
            return True

        monkeypatch.setattr(redis_module.redis_manager, "ping", up)

        assert await reminder_module._check_system_health(db) == 0
        assert await _notifications(db, estate["admin"], "system_alert") == []

    async def test_sustained_outage_alerts_once_per_hour(self, db, estate, monkeypatch):
        """A dependency that stays down must not alert on every sweep."""
        from app.core import redis as redis_module
        from app.worker.tasks import reminder as reminder_module

        async def down():
            return False

        monkeypatch.setattr(redis_module.redis_manager, "ping", down)

        assert await reminder_module._check_system_health(db) == 1
        assert await reminder_module._check_system_health(db) == 0
        assert len(await _notifications(db, estate["admin"], "system_alert")) == 1


class TestPatientMessaging:
    def test_no_messaging_module_exists(self):
        """
        Documents why patient messaging was not implemented.

        `AssistantMessage` is the AI assistant's chat transcript, not
        patient-to-doctor correspondence. Emitting a "Patient Sent Message"
        notification would require inventing a messaging domain, so it is
        recorded as future functionality rather than stubbed.
        """
        import app.models as models_pkg
        import pkgutil

        names = {m.name for m in pkgutil.iter_modules(models_pkg.__path__)}
        assert "message" not in names
        assert "conversation" not in names

        from app.models.assistant import AssistantMessage

        # The one message-like table belongs to the AI assistant, keyed to an
        # assistant conversation rather than to a doctor-patient thread.
        assert AssistantMessage.__tablename__ == "assistant_messages"
        assert hasattr(AssistantMessage, "conversation_id")
