import pytest
import uuid
import io
from datetime import datetime, timezone
from httpx import AsyncClient
from sqlalchemy import select
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor
from app.models.hospital import Hospital
from app.models.appointment import Appointment
from app.models.case import Case
from app.models.notification import NotificationItem
from app.core.security import get_password_hash
from conftest import login_payload

@pytest.fixture
async def setup_shared_data(db):
    # Truncate tables to ensure test isolation
    from sqlalchemy import text
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    await db.execute(text("DELETE FROM appointments;"))
    await db.execute(text("DELETE FROM medications;"))
    await db.execute(text("DELETE FROM prescriptions;"))
    await db.execute(text("DELETE FROM reports;"))
    await db.execute(text("DELETE FROM cases;"))
    await db.execute(text("DELETE FROM doctors;"))
    await db.execute(text("DELETE FROM patients;"))
    await db.execute(text("DELETE FROM users;"))
    await db.execute(text("DELETE FROM hospitals;"))
    await db.execute(text("DELETE FROM audit_logs;"))
    await db.execute(text("DELETE FROM notifications;"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    # 1. Create Patient User & Profile
    patient_user = User(
        email="patient.shared@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="patient",
        is_verified=True
    )
    db.add(patient_user)
    await db.flush()

    patient_profile = Patient(
        id=patient_user.id,
        first_name="Frank",
        last_name="Simpson",
        phone="+1777777777",
        date_of_birth="1990-03-10",
        gender="male"
    )
    db.add(patient_profile)

    # 2. Create Doctor User & Profile
    doctor_user = User(
        email="doctor.shared@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="doctor",
        is_verified=True
    )
    db.add(doctor_user)
    await db.flush()

    doctor_profile = Doctor(
        id=doctor_user.id,
        first_name="Helen",
        last_name="Carter",
        phone="+1888888888",
        specialty="Cardiology",
        license_number="MD-998877",
        availability="available",
        verification_status="verified"
    )
    db.add(doctor_profile)

    # 3. Create Hospital
    hosp = Hospital(
        name="Aronofy Cardiology Center",
        address="300 Heart Way",
        city="Houston",
        state="TX",
        phone="+1713000000",
        email="houston@aronofy.com",
        coordinates={"lat": 29.760, "lng": -95.369},
        services=["Cardiology", "Emergency"],
        emergency_capacity="available",
        verification_status="verified",
        rating=4.8
    )
    db.add(hosp)
    await db.flush()

    # 4. Create Appointment
    appt = Appointment(
        patient_id=patient_user.id,
        doctor_id=doctor_user.id,
        patient_name="Frank Simpson",
        doctor_name="Helen Carter",
        specialty="Cardiology",
        hospital_name="Aronofy Cardiology Center",
        date="2026-09-15",
        time="10:00",
        duration=30,
        type="video",
        status="scheduled",
        reason="Heart Checkup"
    )
    db.add(appt)

    # 5. Create consultation Case
    case_id = uuid.uuid4()
    case_record = Case(
        id=case_id,
        patient_id=patient_user.id,
        patient_name="Frank Simpson",
        patient_age=36,
        patient_gender="male",
        doctor_id=doctor_user.id,
        doctor_name="Helen Carter",
        specialty="Cardiology",
        symptom_summary="Palpitations.",
        urgency_level="low",
        status="routed",
        ai_confidence_score=0.85
    )
    db.add(case_record)

    # 6. Create Notification
    notif = NotificationItem(
        user_id=patient_user.id,
        type="appointment",
        title="New Appointment Scheduled",
        message="Your appointment with Dr. Helen Carter is scheduled.",
        timestamp=datetime.now(timezone.utc).isoformat(),
        read=False,
        priority="medium"
    )
    db.add(notif)

    await db.flush()
    await db.commit()
    return {
        "patient_id": patient_user.id,
        "doctor_id": doctor_user.id,
        "case_id": case_id,
        "notification_id": notif.id
    }

@pytest.mark.asyncio
async def test_shared_uploads_and_notifications(client: AsyncClient, setup_shared_data, db):
    # Log in as Patient
    _login_body = await login_payload("patient.shared@aronofy.com", "password123")
    login_resp = await client.post("/api/v1/auth/login", json=_login_body)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. File Upload
    file_content = b"Mock lab report details."
    files = {"file": ("report.pdf", io.BytesIO(file_content), "application/pdf")}
    upload_resp = await client.post("/api/v1/shared/upload", files=files, headers=headers)
    assert upload_resp.status_code == 201
    assert upload_resp.json()["filename"] == "report.pdf"
    assert "report.pdf" not in upload_resp.json()["file_url"]  # Should have unique UUID prefix

    # 2. Get Notifications count & list
    count_resp = await client.get("/api/v1/shared/notifications/unread-count", headers=headers)
    assert count_resp.status_code == 200
    assert count_resp.json()["unread_count"] == 1

    list_resp = await client.get("/api/v1/shared/notifications", headers=headers)
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1
    assert list_resp.json()[0]["title"] == "New Appointment Scheduled"
    notif_id = list_resp.json()[0]["id"]

    # 3. Mark Notification as read
    read_resp = await client.put(f"/api/v1/shared/notifications/{notif_id}/read", headers=headers)
    assert read_resp.status_code == 200
    assert read_resp.json()["read"] is True


@pytest.mark.asyncio
async def test_shared_search_calendar_and_timeline(client: AsyncClient, setup_shared_data, db):
    case_id = setup_shared_data["case_id"]

    # Log in
    _login_body = await login_payload("patient.shared@aronofy.com", "password123")
    login_resp = await client.post("/api/v1/auth/login", json=_login_body)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Unified Search
    search_resp = await client.get("/api/v1/shared/search?q=Cardio", headers=headers)
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert len(search_data) >= 2  # Dr. Helen Carter (Cardiology) and Aronofy Cardiology Center
    assert any(item["type"] == "doctor" for item in search_data)
    assert any(item["type"] == "hospital" for item in search_data)

    # 2. Calendar Schedule
    cal_resp = await client.get("/api/v1/shared/calendar", headers=headers)
    assert cal_resp.status_code == 200
    assert len(cal_resp.json()) == 1
    assert "Heart Checkup" in cal_resp.json()[0]["title"]

    # 3. Case Timeline History
    #    The endpoint now returns a paginated envelope of typed events rather
    #    than a bare list, and events are newest-first — so case creation is the
    #    oldest entry rather than the first one.
    timeline_resp = await client.get(f"/api/v1/shared/timeline?case_id={case_id}", headers=headers)
    assert timeline_resp.status_code == 200
    body = timeline_resp.json()
    assert body["total"] >= 1
    assert body["events"]
    assert any(e["event_type"] == "case.created" for e in body["events"])


@pytest.mark.asyncio
async def test_shared_audit_settings_and_feedback(client: AsyncClient, setup_shared_data, db):
    # Log in
    _login_body = await login_payload("patient.shared@aronofy.com", "password123")
    login_resp = await client.post("/api/v1/auth/login", json=_login_body)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Post Client Audit Log
    audit_payload = {
        "action": "VIEW_DASHBOARD",
        "resource": "PatientDashboard",
        "resource_id": "dashboard-home",
        "details": "User accessed patient landing view",
        "status": "success"
    }
    audit_resp = await client.post("/api/v1/shared/audit-logs", json=audit_payload, headers=headers)
    assert audit_resp.status_code == 201

    # 2. Fetch & Update Settings
    settings_get = await client.get("/api/v1/shared/settings", headers=headers)
    assert settings_get.status_code == 200
    assert settings_get.json()["theme"] == "dark"

    settings_put_payload = {"theme": "light", "notifications_enabled": False}
    settings_put = await client.put("/api/v1/shared/settings", json=settings_put_payload, headers=headers)
    assert settings_put.status_code == 200
    assert settings_put.json()["theme"] == "light"
    assert settings_put.json()["notifications_enabled"] is False

    # 3. Submit Feedback
    feedback_payload = {
        "category": "usability",
        "subject": "Great Dashboard UI",
        "message": "The patient layout is extremely clean and modern.",
        "rating": 5
    }
    feedback_resp = await client.post("/api/v1/shared/feedback", json=feedback_payload, headers=headers)
    assert feedback_resp.status_code == 201
