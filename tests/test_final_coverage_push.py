import uuid
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.auth import auth_service
from app.services.admin import admin_service
from app.services.doctor import doctor_service
from app.services.consultation import consultation_service
from app.models.user import User
from app.models.doctor import Doctor
from app.models.hospital import Hospital
from app.models.patient import Patient
from app.models.case import Case
from app.models.prescription import Prescription
from app.core.exceptions import EntityNotFoundException, AuthorizationException

@pytest.mark.asyncio
async def test_auth_verify_and_password_reset_flow(db: AsyncSession, mock_redis):
    """
    Tests auth_service verify account token consumption and password reset workflows.
    """
    user = User(email="verify_test@aronofy.com", hashed_password="old_hashed_pass", role="patient", is_verified=False)
    db.add(user)
    await db.flush()

    # Store verification token in mock redis
    token = f"verif_token_{user.id}"
    await mock_redis.set(f"verify_token:{token}", str(user.id))

    # Consume token and verify account
    await auth_service.verify_account(db, token, mock_redis)
    await db.refresh(user)
    assert user.is_verified is True

    # Password reset flow
    reset_token = f"reset_token_{user.id}"
    await mock_redis.set(f"reset_token:{reset_token}", str(user.id))
    
    from app.schemas.auth import ResetPasswordRequest
    await auth_service.reset_password(
        db,
        ResetPasswordRequest(token=reset_token, new_password="NewPassword123!"),
        mock_redis
    )

@pytest.mark.asyncio
async def test_admin_and_doctor_verification_flows(db: AsyncSession):
    """
    Tests admin doctor verification, hospital verification, and user active toggles.
    """
    u_doc = User(email="verif_doc@aronofy.com", hashed_password="pw", role="doctor", is_active=True)
    db.add(u_doc)
    await db.flush()

    doc = Doctor(verification_status="verified", id=u_doc.id, first_name="Verif", last_name="Doc", phone="555-0000", license_number="LIC-V1", specialty="Surgery")
    hosp = Hospital(name="St Jude Hospital", address="200 Oak St", city="Salem", state="MA", phone="555-9999", email="info@stjude.org", emergency_capacity="available")
    db.add_all([doc, hosp])
    await db.flush()

    # Admin verifies doctor
    v_doc = await admin_service.verify_doctor(db, u_doc.id, "verified")
    assert v_doc.verification_status == "verified"

    # Admin verifies hospital
    v_hosp = await admin_service.verify_hospital(db, hosp.id, "verified")
    assert v_hosp.verification_status == "verified"

    # Admin updates user active status
    u_act = await admin_service.update_user_status(db, u_doc.id, False)
    assert u_act.is_active is False

@pytest.mark.asyncio
async def test_consultation_write_prescription_and_case_history(db: AsyncSession):
    """
    Tests consultation service case history generation and prescription dispatching.
    """
    p_user = User(email="rx_pat@aronofy.com", hashed_password="pw", role="patient", is_active=True)
    d_user = User(email="rx_doc@aronofy.com", hashed_password="pw", role="doctor", is_active=True)
    db.add_all([p_user, d_user])
    await db.flush()

    p_prof = Patient(id=p_user.id, first_name="Rx", last_name="Patient", phone="555-1111", date_of_birth="1988-08-08", gender="female")
    d_prof = Doctor(verification_status="verified", id=d_user.id, first_name="Rx", last_name="Doctor", phone="555-2222", license_number="LIC-RX", specialty="Neurology")
    db.add_all([p_prof, d_prof])
    await db.flush()

    case = Case(
        patient_id=p_user.id,
        patient_name="Rx Patient",
        patient_age=36,
        patient_gender="female",
        doctor_id=d_user.id,
        doctor_name="Rx Doctor",
        specialty="Neurology",
        symptom_summary="Migraine",
        urgency_level="medium",
        status="in_consultation",
        notes=""
    )
    db.add(case)
    await db.flush()

    from app.schemas.doctor_api import CreatePrescriptionRequest, CreateMedicationItem
    rx_req = CreatePrescriptionRequest(
        case_id=case.id,
        patient_id=p_user.id,
        diagnosis="Chronic Migraine",
        medications=[
            CreateMedicationItem(
                name="Topiramate",
                dosage="25mg",
                frequency="Daily",
                duration="30 days",
                scheduled_times=["09:00"],
                start_date="2026-07-18",
                end_date="2026-08-17"
            )
        ]
    )
    rx = await consultation_service.write_prescription(db, d_user.id, rx_req)
    assert rx is not None
    assert rx.diagnosis == "Chronic Migraine"
