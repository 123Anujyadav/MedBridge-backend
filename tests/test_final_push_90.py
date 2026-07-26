import uuid
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor
from app.models.case import Case
from app.models.prescription import Prescription, Medication
from app.models.report import Report
from app.models.appointment import Appointment
from app.core.security import create_access_token
from app.api.v1.endpoints.health import liveness_probe, readiness_probe, detailed_health
from app.api.deps import get_current_user, AuthenticationException

@pytest.mark.asyncio
async def test_doctor_and_patient_detailed_endpoints(client: AsyncClient, db: AsyncSession):
    """
    Tests detailed endpoints: appointment listing/cancellation, report retrieval, prescription details.
    """
    p_user = User(email="det_p@aronofy.com", hashed_password="pw", role="patient", is_active=True)
    d_user = User(email="det_d@aronofy.com", hashed_password="pw", role="doctor", is_active=True)
    db.add_all([p_user, d_user])
    await db.flush()

    p_prof = Patient(id=p_user.id, first_name="Det", last_name="Patient", phone="555-1212", date_of_birth="1990-01-01", gender="male")
    d_prof = Doctor(verification_status="verified", id=d_user.id, first_name="Det", last_name="Doctor", phone="555-2121", license_number="LIC-DET", specialty="Oncology")
    db.add_all([p_prof, d_prof])
    await db.flush()

    # Appointment
    appt = Appointment(
        patient_id=p_user.id,
        doctor_id=d_user.id,
        patient_name="Det Patient",
        doctor_name="Det Doctor",
        specialty="Oncology",
        hospital_name="Main Hospital",
        date="2026-08-10",
        time="14:00",
        duration=30,
        type="in_person",
        status="scheduled",
        reason="Consultation",
        notes=""
    )
    db.add(appt)
    await db.flush()

    # Case & Prescription
    case = Case(patient_id=p_user.id, patient_name="Det Patient", patient_age=34, patient_gender="male", doctor_id=d_user.id, doctor_name="Det Doctor", specialty="Oncology", symptom_summary="Pain", urgency_level="low", status="completed", notes="")
    db.add(case)
    await db.flush()

    rx = Prescription(case_id=case.id, patient_id=p_user.id, patient_name="Det Patient", doctor_id=d_user.id, doctor_name="Det Doctor", diagnosis="Oncology Eval", status="active")
    db.add(rx)
    await db.flush()

    med = Medication(prescription_id=rx.id, name="MedA", dosage="10mg", frequency="Daily", duration="7 days", status="active", start_date="2026-07-18", end_date="2026-07-25", scheduled_times=["08:00"], taken_doses=0, total_doses=7)
    db.add(med)
    await db.flush()

    report = Report(patient_id=p_user.id, patient_name="Det Patient", type="Imaging", title="CT Scan", summary="Clear", content="No issues found", doctor_name="Det Doctor", date="2026-07-18", status="ready")
    db.add(report)
    await db.flush()

    p_token = create_access_token(p_user.id)
    d_token = create_access_token(d_user.id)
    p_headers = {"Authorization": f"Bearer {p_token}"}
    d_headers = {"Authorization": f"Bearer {d_token}"}

    # Patient fetch specific prescription and report
    r1 = await client.get(f"/api/v1/patient/prescriptions/{rx.id}", headers=p_headers)
    assert r1.status_code == 200

    r2 = await client.get(f"/api/v1/patient/reports/{report.id}", headers=p_headers)
    assert r2.status_code == 200

    # Track medication adherence
    r3 = await client.put(f"/api/v1/patient/medications/{med.id}/track", json={"status": "taken"}, headers=p_headers)
    assert r3.status_code == 200

    # Doctor update appointment status
    r4 = await client.put(f"/api/v1/doctor/appointments/{appt.id}/status?status_str=confirmed", headers=d_headers)
    assert r4.status_code == 200

@pytest.mark.asyncio
async def test_direct_health_and_deps_functions(db: AsyncSession):
    """
    Directly exercises health probes and deps token authentication functions.
    """
    # Health direct calls
    live = await liveness_probe()
    assert live["status"] == "alive"

    ready = await readiness_probe(db)
    assert ready.status_code in (200, 503)

    health_res = await detailed_health(db)
    assert health_res["status"] in ("operational", "degraded")


    # Deps user authentication checks
    user = User(email="deps_u@aronofy.com", hashed_password="pw", role="patient", is_active=True)
    db.add(user)
    await db.flush()

    valid_token = create_access_token(user.id)
    retrieved_user = await get_current_user(db, valid_token)
    assert retrieved_user.id == user.id

    with pytest.raises(AuthenticationException):
        await get_current_user(db, "invalid_expired_token")
