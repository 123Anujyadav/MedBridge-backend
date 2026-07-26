import uuid
import pytest
from fastapi import UploadFile
from io import BytesIO
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.patient import patient_service
from app.services.doctor import doctor_service
from app.core.upload import validate_and_save_upload
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor
from app.schemas.patient import PatientUpdate, ConsentFlagsSchema
from app.schemas.doctor_api import UpdateAvailabilityRequest

@pytest.mark.asyncio
async def test_patient_service_extended(db: AsyncSession):
    """
    Tests patient profile update, consent update, and dashboard rendering.
    """
    # 1. Create Patient User
    patient_user = User(
        email="patient_ext@aronofy.com",
        hashed_password="hashed_secret_pass",
        role="patient",
        is_active=True
    )
    db.add(patient_user)
    await db.flush()

    patient_profile = Patient(
        id=patient_user.id,
        first_name="John",
        last_name="Doe",
        phone="555-000-1111",
        date_of_birth="1990-01-01",
        gender="male"
    )
    db.add(patient_profile)
    await db.flush()

    # 2. Get Profile
    prof = await patient_service.get_profile(db, patient_user.id)
    assert prof.first_name == "John"

    # 3. Update Profile
    updated_prof = await patient_service.update_profile(
        db,
        patient_user.id,
        PatientUpdate(first_name="Jonathan", last_name="Doe")
    )
    assert updated_prof.first_name == "Jonathan"

    # 4. Update Consent
    consent_in = ConsentFlagsSchema(
        dataSharing=True,
        researchParticipation=True,
        emergencyAccess=True,
        aiProcessing=False
    )
    consent_updated = await patient_service.update_consent(db, patient_user.id, consent_in)
    assert consent_updated.consent_flags["researchParticipation"] is True

    # 5. Get Dashboard
    dash = await patient_service.get_dashboard(db, patient_user.id)
    assert dash["patient_id"] == patient_user.id

@pytest.mark.asyncio
async def test_doctor_service_extended(db: AsyncSession):
    """
    Tests doctor profile retrieval, dashboard, analytics, and availability updates.
    """
    doctor_user = User(
        email="doctor_ext@aronofy.com",
        hashed_password="hashed_secret_pass",
        role="doctor",
        is_active=True
    )
    db.add(doctor_user)
    await db.flush()

    doctor_profile = Doctor(
        verification_status="verified",
        id=doctor_user.id,
        first_name="Sarah",
        last_name="Connor",
        phone="555-222-3333",
        license_number="LIC-999888",
        specialty="Cardiology",
        years_of_experience=10
    )
    db.add(doctor_profile)
    await db.flush()

    # Get Profile
    doc = await doctor_service.get_profile(db, doctor_user.id)
    assert doc.first_name == "Sarah"

    # Get Dashboard & Analytics
    dash = await doctor_service.get_dashboard(db, doctor_user.id)
    assert dash["doctor_id"] == doctor_user.id

    analytics = await doctor_service.get_analytics(db, doctor_user.id)
    assert "adherence_rate" in analytics

    # Update Availability
    avail_req = UpdateAvailabilityRequest(availability="available", next_available="Tomorrow 9:00 AM")
    avail_doc = await doctor_service.update_availability(db, doctor_user.id, avail_req)
    assert avail_doc.availability == "available"

@pytest.mark.asyncio
async def test_upload_security_execution():
    """
    Tests safe file upload validation and rejection of invalid files.
    """
    # Valid PDF upload
    valid_file = UploadFile(
        filename="medical_report.pdf",
        file=BytesIO(b"%PDF-1.4 sample content bytes"),
        headers={"content-type": "application/pdf"}
    )
    saved_path = await validate_and_save_upload(valid_file, target_dir="uploads_test")
    assert saved_path is not None
    import os
    if os.path.exists(saved_path):
        os.remove(saved_path)

    # Invalid MIME type file upload raises HTTP 400
    from fastapi import HTTPException
    invalid_file = UploadFile(
        filename="malicious.exe",
        file=BytesIO(b"binary payload"),
        headers={"content-type": "application/x-msdownload"}
    )
    with pytest.raises(HTTPException) as exc:
        await validate_and_save_upload(invalid_file, target_dir="uploads_test")
    assert exc.value.status_code == 400
