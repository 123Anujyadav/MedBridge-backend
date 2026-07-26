import pytest
import uuid
from unittest.mock import MagicMock
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor
from app.models.case import Case
from app.models.prescription import Prescription, Medication
from app.models.notification import NotificationItem
from app.core.security import get_password_hash
from app.worker.tasks.jobs import send_email_task, send_notification_task, cleanup_expired_sessions
from app.worker.tasks.reminder import send_medicine_reminders

@pytest.fixture
async def setup_celery_test_data(db):
    # Truncate tables to ensure isolation
    from sqlalchemy import text
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    await db.execute(text("DELETE FROM appointments;"))
    await db.execute(text("DELETE FROM medications;"))
    await db.execute(text("DELETE FROM prescriptions;"))
    await db.execute(text("DELETE FROM cases;"))
    await db.execute(text("DELETE FROM doctors;"))
    await db.execute(text("DELETE FROM patients;"))
    await db.execute(text("DELETE FROM users;"))
    await db.execute(text("DELETE FROM notifications;"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    # 1. Create Patient User & Profile
    patient_user = User(
        email="patient.celery@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="patient",
        is_verified=True
    )
    db.add(patient_user)
    await db.flush()

    patient_profile = Patient(
        id=patient_user.id,
        first_name="Jane",
        last_name="Doe",
        phone="+1999999999",
        date_of_birth="1992-06-18",
        gender="female"
    )
    db.add(patient_profile)

    # 2. Create Doctor & Case (Required for Prescription foreign keys)
    doctor_uuid = uuid.uuid4()
    doc_profile = Doctor(
        id=doctor_uuid,
        first_name="Doctor",
        last_name="Strange",
        phone="+1333333333",
        specialty="General Medicine",
        license_number="MD-777777",
        verification_status="verified"
    )
    db.add(doc_profile)

    case_id = uuid.uuid4()
    case_record = Case(
        id=case_id,
        patient_id=patient_user.id,
        patient_name="Jane Doe",
        patient_age=34,
        patient_gender="female",
        doctor_id=doctor_uuid,
        doctor_name="Doctor Strange",
        specialty="General Medicine",
        symptom_summary="Headache",
        urgency_level="low",
        status="routed",
        ai_confidence_score=0.88
    )
    db.add(case_record)
    await db.flush()

    # 3. Create Prescription & Medication scheduled for today
    from datetime import datetime
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    rx = Prescription(
        case_id=case_id,
        patient_id=patient_user.id,
        doctor_id=doctor_uuid,
        patient_name="Jane Doe",
        doctor_name="Doctor Strange",
        diagnosis="Migraine",
        notes="Rest",
        status="active"
    )
    db.add(rx)
    await db.flush()

    med = Medication(
        prescription_id=rx.id,
        name="Aspirin",
        dosage="500mg",
        frequency="daily",
        duration="10 days",
        special_instructions="Take after breakfast",
        scheduled_times=["08:00"],
        taken_doses=0,
        total_doses=10,
        start_date=today_str,
        end_date=today_str,
        status="active"
    )
    db.add(med)

    await db.flush()
    await db.commit()
    return {
        "patient_id": patient_user.id,
        "medication_id": med.id
    }

@pytest.mark.asyncio
async def test_medicine_reminders_and_notifications(db, setup_celery_test_data):
    from unittest.mock import patch
    from conftest import TestSessionLocal
    
    # 1. Trigger medicine reminders task with TestSessionLocal patch
    with patch("app.core.database.AsyncSessionLocal", TestSessionLocal):
        success = send_medicine_reminders()
        assert success is True

    # 2. Verify medication reminder notification was written to database
    from sqlalchemy import select
    result = await db.execute(
        select(NotificationItem)
        .where(NotificationItem.user_id == setup_celery_test_data["patient_id"])
    )
    notifs = result.scalars().all()
    assert len(notifs) >= 1
    assert notifs[0].title == "Medicine Reminder"
    assert "Aspirin" in notifs[0].message



@pytest.mark.asyncio
async def test_email_queue_and_retry_mechanism():
    original_retry = send_email_task.retry
    original_retries = getattr(send_email_task.request, "retries", 0)

    try:
        # Mock retry method on the Celery task instance
        send_email_task.retry = MagicMock(side_effect=Exception("retry triggered exception"))
        
        # Adjust retries directly on request context instead of overwriting the request property
        send_email_task.request.retries = 0

        # 1. Test successful email dispatch
        success = send_email_task.run("test@aronofy.com", "Test Subject", "Body")
        assert success is True
        send_email_task.retry.assert_not_called()

        # 2. Test email dispatch failure & retry trigger
        send_email_task.request.retries = 0
        with pytest.raises(Exception, match="retry triggered exception"):
            send_email_task.run("fail@aronofy.com", "Retry Subject", "Body")
        
        send_email_task.retry.assert_called_once()
        
    finally:
        # Restore original retry and retries count
        send_email_task.retry = original_retry
        send_email_task.request.retries = original_retries

@pytest.mark.asyncio
async def test_cleanup_expired_sessions_task():
    # Test session cleanup logs and executes cleanly
    success = cleanup_expired_sessions()
    assert success is True
