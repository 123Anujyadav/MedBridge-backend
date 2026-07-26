import uuid
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.consultation import consultation_service
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor
from app.models.case import Case
from app.schemas.doctor_api import (
    UpdateCaseNotesRequest,
    DiagnoseCaseRequest,
    CreateReportRequest
)
from app.core.exceptions import EntityNotFoundException, AuthorizationException

@pytest.mark.asyncio
async def test_consultation_service_methods(db: AsyncSession):
    """
    Tests consultation case notes, diagnosis, report creation, and authorization constraints.
    """
    # 1. Create Patient & Doctor Users
    doctor_user = User(email="doc_cons@aronofy.com", hashed_password="pw", role="doctor", is_active=True)
    other_doctor_user = User(email="other_doc@aronofy.com", hashed_password="pw", role="doctor", is_active=True)
    patient_user = User(email="pat_cons@aronofy.com", hashed_password="pw", role="patient", is_active=True)
    db.add_all([doctor_user, other_doctor_user, patient_user])
    await db.flush()

    doctor_profile = Doctor(verification_status="verified", id=doctor_user.id, first_name="Doc", last_name="Who", phone="555-1234", license_number="LIC-123", specialty="General")
    patient_profile = Patient(id=patient_user.id, first_name="Pat", last_name="Smith", phone="555-4321", date_of_birth="1985-01-01", gender="male")
    db.add_all([doctor_profile, patient_profile])
    await db.flush()

    # 2. Create Case
    case = Case(
        patient_id=patient_user.id,
        patient_name="Pat Smith",
        patient_age=39,
        patient_gender="male",
        doctor_id=doctor_user.id,
        doctor_name="Doc Who",
        specialty="General",
        symptom_summary="Persistent fever and cough",
        urgency_level="medium",
        status="in_consultation",
        notes=""
    )
    db.add(case)
    await db.flush()

    # 3. Update Case Notes
    updated_case = await consultation_service.update_case_notes(
        db, doctor_user.id, case.id, UpdateCaseNotesRequest(notes="Patient shows signs of flu.")
    )
    assert updated_case.notes == "Patient shows signs of flu."

    # 4. Unauthorized doctor editing notes raises AuthorizationException
    with pytest.raises(AuthorizationException):
        await consultation_service.update_case_notes(
            db, other_doctor_user.id, case.id, UpdateCaseNotesRequest(notes="Unauthorized edit")
        )

    # 5. Diagnose Case
    diagnosed = await consultation_service.diagnose_case(
        db, doctor_user.id, case.id, DiagnoseCaseRequest(diagnosis="Influenza A", notes="Prescribed rest and fluids.")
    )
    assert diagnosed.status == "completed"

    # 6. Write Report
    report_req = CreateReportRequest(
        patient_id=patient_user.id,
        patient_name="Pat Smith",
        type="Lab Result",
        title="Blood Panel Report",
        summary="Normal CBC panel",
        content="WBC count normal.",
        date="2026-07-18"
    )
    report = await consultation_service.write_report(db, doctor_user.id, report_req)
    assert report.id is not None
    assert report.status == "ready"
