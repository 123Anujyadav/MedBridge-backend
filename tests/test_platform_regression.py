"""
Regression guards for defects found during full-platform verification.

Each test here pins a bug that shipped and was fixed. They are grouped by the
class of mistake rather than by module, because the same mistake recurred in
several places:

* **Raw ORM objects returned without a `response_model`.** FastAPI cannot
  serialise a SQLAlchemy instance, so the endpoint 500s — but only once a row
  exists, which is why these survived until real data flowed through.
* **Notification rows built by hand instead of through the service**, losing
  category, deduplication, live delivery and audit.
* **Action links pointing at routes that do not exist**, landing the user on
  the 404 page instead of the workflow.
* **Pages calling an endpoint gated to a different role.**
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text

from app.core.security import get_password_hash
from app.models.appointment import Appointment
from app.models.case import Case
from app.models.doctor import Doctor
from app.models.notification import NotificationItem
from app.models.patient import Patient
from app.models.user import User

pytestmark = pytest.mark.asyncio

PW = "password123"
DOCTOR = "reg.doc@aronofy.com"
PATIENT = "reg.pat@aronofy.com"
ADMIN = "reg.admin@aronofy.com"

# Every route the SPA actually defines (Frontend/src/App.tsx). A notification
# whose action_url is not one of these sends the user to the 404 page.
FRONTEND_ROUTES = {
    "/", "/patient/dashboard", "/patient/ai-medical-assistant", "/patient/intake",
    "/patient/reports", "/patient/prescriptions", "/patient/reminders",
    "/patient/appointments", "/patient/records", "/patient/history",
    "/patient/emergency", "/patient/notifications", "/patient/settings",
    "/patient/profile", "/doctor/dashboard", "/doctor/cases", "/doctor/consultation",
    "/doctor/prescriptions", "/doctor/patients", "/doctor/schedule",
    "/doctor/ai-reports", "/doctor/notifications", "/doctor/settings",
    "/admin/dashboard", "/admin/doctors", "/admin/hospitals", "/admin/compliance",
    "/admin/verification", "/admin/system", "/admin/notifications",
    "/admin/settings", "/admin/cases",
}


@pytest.fixture
async def estate(db):
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("audit_logs", "notifications", "appointments", "reports",
                  "cases", "doctors", "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    ids: dict[str, Any] = {}
    for key, email, role in (("doctor", DOCTOR, "doctor"),
                             ("patient", PATIENT, "patient"),
                             ("admin", ADMIN, "admin")):
        u = User(email=email, hashed_password=get_password_hash(PW),
                 role=role, is_verified=True)
        db.add(u); await db.flush(); ids[key] = u.id

    db.add(Doctor(id=ids["doctor"], first_name="Asha", last_name="Rao",
                  phone="+911", specialty="Neurology", hospital_name="Central",
                  license_number="LIC-REG-1", verification_status="verified"))
    db.add(Patient(id=ids["patient"], first_name="Meera", last_name="Iyer",
                   phone="+913", date_of_birth="1992-03-14", gender="female",
                   allergies=[], chronic_conditions=[], medications=[]))
    await db.flush()

    case = Case(patient_id=ids["patient"], patient_name="Meera Iyer",
                patient_age=34, patient_gender="female", doctor_id=ids["doctor"],
                doctor_name="Dr. Asha Rao", specialty="Neurology",
                symptom_summary="Headache.", urgency_level="medium",
                status="routed", ai_extracted_symptoms=[], ai_confidence_score=0.0,
                attachments=[], notes="")
    db.add(case); await db.flush()
    ids["case"] = case.id

    db.add(Appointment(patient_id=ids["patient"], doctor_id=ids["doctor"],
                       patient_name="Meera Iyer", doctor_name="Dr. Asha Rao",
                       specialty="Neurology", hospital_name="Central",
                       date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                       time="10:00", duration=30, type="in_person",
                       status="scheduled", reason="Headache", notes="",
                       case_id=case.id))
    await db.commit()
    return ids


async def _login(client: AsyncClient, email: str) -> dict[str, str]:
    r = await client.post("/api/v1/auth/login", json={"email": email, "password": PW})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


class TestSerializationRegressions:
    """
    Endpoints returning ORM objects need a `response_model`.

    These only fail once a row exists, so an empty database hides them — which
    is exactly how they reached production.
    """

    async def test_admin_appointments_serialises(self, client, estate):
        headers = await _login(client, ADMIN)
        r = await client.get("/api/v1/admin/appointments", headers=headers)
        assert r.status_code == 200, r.text
        assert len(r.json()) == 1
        assert r.json()[0]["reason"] == "Headache"

    async def test_admin_appointment_status_update_serialises(self, client, estate, db):
        appt_id = await db.scalar(select(Appointment.id))
        headers = await _login(client, ADMIN)
        r = await client.put(f"/api/v1/admin/appointments/{appt_id}/status",
                             params={"status_str": "confirmed"}, headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "confirmed"

    async def test_patient_notifications_serialises(self, client, estate, db):
        """The list only broke once the patient actually had a notification."""
        db.add(NotificationItem(user_id=estate["patient"], type="report",
                                title="T", message="m",
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                read=False, priority="medium", category="report"))
        await db.commit()

        headers = await _login(client, PATIENT)
        r = await client.get("/api/v1/patient/notifications", headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()[0]["title"] == "T"

    async def test_every_orm_returning_endpoint_declares_a_response_model(self):
        """
        A route returning a database row without a response_model will 500.

        Asserted against the OpenAPI schema so a new one cannot be added
        silently: every GET/PUT/POST that is not a file download or a plain
        status envelope must publish a response schema.
        """
        from app.main import app

        allowed_without_schema = {
            # File responses and simple {"status": ...} envelopes.
            "/api/v1/shared/reports/{id}/download",
            "/api/v1/doctor/analytics/export",
            "/api/v1/doctor/reports/bulk/export",
            "/api/v1/shared/notifications/unread-count",
            "/api/v1/shared/notifications/read-all",
            "/api/v1/shared/notifications/read-selected",
            "/api/v1/shared/audit-logs",
            "/api/v1/shared/feedback",
            "/api/v1/admin/users/{id}",
            # Root banner; returns a literal dict, not a database row.
            "/api/v1/",
        }
        paths = app.openapi()["paths"]
        missing = []
        for path, methods in paths.items():
            if path in allowed_without_schema or not path.startswith("/api/v1"):
                continue
            for method, spec in methods.items():
                content = (spec.get("responses", {}).get("200")
                           or spec.get("responses", {}).get("201") or {}).get("content")
                if content is None:
                    continue
                schema = content.get("application/json", {}).get("schema", {})
                # An untyped `{}` schema is what a missing response_model looks like.
                if schema == {}:
                    missing.append(f"{method.upper()} {path}")
        assert not missing, f"endpoints without a response schema: {missing}"


class TestNotificationServiceRouting:
    """
    Every notification must go through `notification_service`.

    A hand-built row carries no category, is invisible to filters, gets no
    live push, is not deduplicated and leaves no audit entry.
    """

    async def test_issued_report_notification_is_categorised(self, client, estate):
        headers = await _login(client, DOCTOR)
        r = await client.post("/api/v1/doctor/reports/issue", headers=headers, json={
            "case_id": str(estate["case"]), "title": "Report",
            "summary": "s", "diagnosis": "Migraine", "clinical_notes": "n",
            "prescription": "", "follow_up_instructions": "",
            "recommendations": [], "recommended_tests": [],
            "ai_generated": True,
        })
        assert r.status_code == 201, r.text

        patient_headers = await _login(client, PATIENT)
        centre = await client.get("/api/v1/shared/notifications/center",
                                  params={"category": "report"},
                                  headers=patient_headers)
        assert centre.status_code == 200
        assert centre.json()["total"] >= 1, "issued-report notification not categorised"

    async def test_shared_report_notification_is_categorised(self, client, estate, db):
        headers = await _login(client, DOCTOR)
        issued = await client.post("/api/v1/doctor/reports/issue", headers=headers,
                                   json={"case_id": str(estate["case"]),
                                         "title": "Report", "summary": "s",
                                         "diagnosis": "Migraine",
                                         "clinical_notes": "n", "prescription": "",
                                         "follow_up_instructions": "",
                                         "recommendations": [],
                                         "recommended_tests": [],
                                         "ai_generated": True})
        report_id = issued.json()["id"]
        await client.put(f"/api/v1/doctor/reports/{report_id}/status",
                         params={"status_str": "shared"}, headers=headers)

        rows = (await db.execute(
            select(NotificationItem)
            .where(NotificationItem.user_id == estate["patient"])
        )).scalars().all()
        assert rows, "no patient notification created"
        assert all(n.category == "report" for n in rows), (
            "a patient notification is uncategorised — built outside the service"
        )
        assert all(n.dedupe_key for n in rows), "notification has no dedupe key"

    async def test_no_notification_is_built_outside_the_service(self):
        """
        Guards the pattern, not just the two known sites.

        `NotificationItem(...)` should appear only inside the notification
        service; anywhere else means a path that skips categorisation,
        deduplication, delivery and audit.
        """
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "app"
        offenders = []
        for file in root.rglob("*.py"):
            if file.name == "notifications.py" and file.parent.name == "services":
                continue
            if file.name == "notification.py" and file.parent.name == "models":
                continue
            text_body = file.read_text(encoding="utf-8", errors="ignore")
            if "NotificationItem(" in text_body:
                offenders.append(str(file.relative_to(root)))
        # jobs.py is the legacy Celery notification task, retained deliberately.
        offenders = [o for o in offenders if o.replace("\\", "/") != "worker/tasks/jobs.py"]
        assert not offenders, f"notifications built outside the service: {offenders}"


class TestNotificationLinksResolve:
    async def test_every_action_url_targets_a_real_route(self, client, estate, db):
        """
        A link to a route the SPA does not define lands on the 404 page.

        Two shipped that way: `/patient/reports/{id}` (the route takes no id)
        and `/admin/system-health` (the route is `/admin/system`).
        """
        headers = await _login(client, DOCTOR)
        issued = await client.post("/api/v1/doctor/reports/issue", headers=headers,
                                   json={"case_id": str(estate["case"]),
                                         "title": "Report", "summary": "s",
                                         "diagnosis": "Migraine",
                                         "clinical_notes": "n", "prescription": "",
                                         "follow_up_instructions": "",
                                         "recommendations": [],
                                         "recommended_tests": [],
                                         "ai_generated": True})
        report_id = issued.json()["id"]
        await client.put(f"/api/v1/doctor/reports/{report_id}/status",
                         params={"status_str": "shared"}, headers=headers)

        rows = (await db.execute(select(NotificationItem))).scalars().all()
        assert rows, "no notifications to check"

        broken = [
            n.action_url for n in rows
            if n.action_url
            and (n.action_url.split("?")[0].rstrip("/") or "/") not in FRONTEND_ROUTES
        ]
        assert not broken, f"notification links to nonexistent routes: {broken}"


class TestRoleScopedEndpoints:
    async def test_admin_has_its_own_case_list(self, client, estate):
        """
        The admin Cases page called the doctor-scoped list and got 403.

        Admins need their own endpoint; the doctor route stays doctor-only.
        """
        admin_headers = await _login(client, ADMIN)
        r = await client.get("/api/v1/admin/cases", headers=admin_headers)
        assert r.status_code == 200, r.text
        assert any(c["id"] == str(estate["case"]) for c in r.json())

        # And the doctor route remains closed to admins.
        assert (await client.get("/api/v1/doctor/cases",
                                 headers=admin_headers)).status_code == 403

    async def test_shared_routes_serve_every_role(self, client, estate):
        """
        The portals share notification and settings routes.

        These must stay role-agnostic: pointing a doctor or admin page at a
        `/patient/*` route would 403 for them.
        """
        for email in (PATIENT, DOCTOR, ADMIN):
            headers = await _login(client, email)
            for path in ("/api/v1/shared/notifications",
                         "/api/v1/shared/notifications/center",
                         "/api/v1/shared/settings"):
                r = await client.get(path, headers=headers)
                assert r.status_code == 200, f"{email} blocked from {path}: {r.text}"
