import pytest
import uuid
from httpx import AsyncClient
from sqlalchemy import select
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor
from app.models.hospital import Hospital
from app.models.appointment import Appointment
from app.models.prescription import Prescription, Medication
from app.models.report import Report
from app.core.security import get_password_hash
from conftest import login_payload

@pytest.fixture
async def setup_patient_data(db):
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
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    # 1. Create Patient User & Profile
    patient_user = User(
        email="john.doe@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="patient",
        is_verified=True
    )
    db.add(patient_user)
    await db.flush()

    patient_profile = Patient(
        id=patient_user.id,
        first_name="John",
        last_name="Doe",
        phone="+1234567890",
        date_of_birth="1990-01-01",
        gender="male",
        consent_flags={"dataSharing": True, "researchParticipation": False, "emergencyAccess": True, "aiProcessing": True}
    )
    db.add(patient_profile)

    # 2. Create Doctor User & Profile
    doctor_user = User(
        email="doctor.smith@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="doctor",
        is_verified=True
    )
    db.add(doctor_user)
    await db.flush()

    doctor_profile = Doctor(
        id=doctor_user.id,
        first_name="Sarah",
        last_name="Smith",
        phone="+1999888777",
        specialty="Cardiology",
        license_number="MD-998877",
        availability="available",
        verification_status="verified"
    )
    db.add(doctor_profile)

    # 3. Create Hospital
    hospital = Hospital(
        name="Aronofy Boston Hospital",
        address="75 Binney St",
        city="Boston",
        state="MA",
        phone="+1617000000",
        email="boston@aronofy.com",
        coordinates={"lat": 42.338, "lng": -71.106},
        emergency_capacity="available",
        rating=4.8
    )
    db.add(hospital)
    await db.flush()

    await db.commit()
    return {
        "patient_id": patient_user.id,
        "doctor_id": doctor_user.id,
        "hospital_id": hospital.id

    }

@pytest.mark.asyncio
async def test_patient_profile_and_consent(client: AsyncClient, setup_patient_data, db):
    # Log in to get access token
    _login_body = await login_payload("john.doe@aronofy.com", "password123")
    login_resp = await client.post("/api/v1/auth/login", json=_login_body)
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Get Profile
    profile_resp = await client.get("/api/v1/patient/profile", headers=headers)
    assert profile_resp.status_code == 200
    assert profile_resp.json()["first_name"] == "John"

    # 2. Update Profile
    update_payload = {"first_name": "Johnny", "last_name": "Doe", "phone": "+1999999999", "date_of_birth": "1990-01-01", "gender": "male"}
    update_resp = await client.put("/api/v1/patient/profile", json=update_payload, headers=headers)
    assert update_resp.status_code == 200
    assert update_resp.json()["first_name"] == "Johnny"

    # 3. Update Consent
    consent_payload = {"dataSharing": False, "researchParticipation": True, "emergencyAccess": False, "aiProcessing": False}
    consent_resp = await client.put("/api/v1/patient/consent", json=consent_payload, headers=headers)
    assert consent_resp.status_code == 200
    assert consent_resp.json()["consent_flags"]["dataSharing"] is False


@pytest.mark.asyncio
async def test_patient_appointment_scheduling(client: AsyncClient, setup_patient_data, db):
    doctor_id = setup_patient_data["doctor_id"]

    # Log in
    _login_body = await login_payload("john.doe@aronofy.com", "password123")
    login_resp = await client.post("/api/v1/auth/login", json=_login_body)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Book Appointment
    booking_payload = {
        "doctor_id": str(doctor_id),
        "specialty": "Cardiology",
        "hospital_name": "Aronofy Boston Hospital",
        "date": "2026-08-10",
        "time": "10:30",
        "type": "video",
        "reason": "Routine heart checkup"
    }
    booking_resp = await client.post("/api/v1/patient/appointments", json=booking_payload, headers=headers)
    assert booking_resp.status_code == 201
    appt_id = booking_resp.json()["id"]

    # 2. Verify Conflict Booking (Double-Booking should fail)
    conflict_resp = await client.post("/api/v1/patient/appointments", json=booking_payload, headers=headers)
    assert conflict_resp.status_code == 422
    res_json = conflict_resp.json()
    assert "detail" in res_json or "message" in res_json or "details" in res_json


    # 3. List Appointments
    list_resp = await client.get("/api/v1/patient/appointments", headers=headers)
    assert list_resp.status_code == 200
    assert len(list_resp.json()) >= 1

    # 4. Cancel Appointment
    cancel_resp = await client.put(f"/api/v1/patient/appointments/{appt_id}/cancel", headers=headers)
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_patient_dashboard(client: AsyncClient, setup_patient_data, db):
    patient_id = setup_patient_data["patient_id"]
    doctor_id = setup_patient_data["doctor_id"]

    # Insert mock report
    report = Report(
        patient_id=patient_id,
        patient_name="John Doe",
        type="lab_result",
        title="Complete Blood Count",
        summary="Slightly low iron levels.",
        content="WBC: 6.5, RBC: 4.8, Hb: 13.2",
        date="2026-07-15",
        status="ready"
    )
    db.add(report)

    # Insert mock prescription and medication
    case_id = uuid.uuid4()
    # Create mock Case first because Prescription foreign key references Case.id
    from app.models.case import Case
    case_mock = Case(
        id=case_id,
        patient_id=patient_id,
        patient_name="John Doe",
        patient_age=36,
        patient_gender="male",
        specialty="Cardiology",
        symptom_summary="Chest pain",
        urgency_level="high",
        status="completed"
    )
    db.add(case_mock)
    await db.flush()

    rx = Prescription(
        case_id=case_id,
        patient_id=patient_id,
        patient_name="John Doe",
        doctor_id=doctor_id,
        doctor_name="Sarah Smith",
        diagnosis="Mild Hypertension",
        notes="Take daily.",
        status="active"
    )
    db.add(rx)
    await db.flush()

    med = Medication(
        prescription_id=rx.id,
        name="Lisinopril",
        dosage="10mg",
        frequency="once daily",
        duration="30 days",
        status="active",
        scheduled_times=["08:00"],
        start_date="2026-07-01",
        end_date="2026-07-31"
    )
    db.add(med)
    await db.flush()
    await db.commit()

    # Log in
    _login_body = await login_payload("john.doe@aronofy.com", "password123")
    login_resp = await client.post("/api/v1/auth/login", json=_login_body)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Fetch Dashboard
    dash_resp = await client.get("/api/v1/patient/dashboard", headers=headers)
    assert dash_resp.status_code == 200
    dash_data = dash_resp.json()
    assert dash_data["health_score"] == 100
    assert len(dash_data["today_medications"]) >= 1
    assert dash_data["today_medications"][0]["name"] == "Lisinopril"
    assert len(dash_data["recent_reports"]) >= 1
    assert dash_data["recent_reports"][0]["title"] == "Complete Blood Count"

    # 2. Track Medication Adherence
    med_id = dash_data["today_medications"][0]["id"]
    track_payload = {"status": "taken"}
    track_resp = await client.put(f"/api/v1/patient/medications/{med_id}/track", json=track_payload, headers=headers)
    assert track_resp.status_code == 200
    assert track_resp.json()["status"] == "taken"
    assert track_resp.json()["taken_doses"] == 1


@pytest.mark.asyncio
async def test_patient_emergency_panic(client: AsyncClient, setup_patient_data, db):
    # Log in
    _login_body = await login_payload("john.doe@aronofy.com", "password123")
    login_resp = await client.post("/api/v1/auth/login", json=_login_body)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Trigger Panic Button
    panic_payload = {
        "location": {
            "lat": 42.358,
            "lng": -71.060,
            "address": "State House, Boston, MA"
        }
    }
    panic_resp = await client.post("/api/v1/patient/emergency", json=panic_payload, headers=headers)
    assert panic_resp.status_code == 201
    panic_data = panic_resp.json()
    assert panic_data["ambulance_dispatched"] is True
    assert panic_data["hospital_name"] == "Aronofy Boston Hospital"  # Routed to our mock hospital
    req_id = panic_data["id"]

    # 2. Track Emergency Dispatch Status
    track_resp = await client.get(f"/api/v1/patient/emergency/{req_id}", headers=headers)
    assert track_resp.status_code == 200
    assert track_resp.json()["status"] == "dispatched"
