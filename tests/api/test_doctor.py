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
from app.models.case import Case
from app.core.security import get_password_hash

@pytest.fixture
async def setup_doctor_data(db):
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

    # 1. Create Doctor User & Profile
    doctor_user = User(
        email="doctor.jones@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="doctor",
        is_verified=True
    )
    db.add(doctor_user)
    await db.flush()

    doctor_profile = Doctor(
        id=doctor_user.id,
        first_name="David",
        last_name="Jones",
        phone="+1555555555",
        specialty="Pediatrics",
        license_number="MD-112233",
        availability="available",
        verification_status="verified",
        rating=4.9
    )
    db.add(doctor_profile)

    # 2. Create Patient User & Profile
    patient_user = User(
        email="kid.patient@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="patient",
        is_verified=True
    )
    db.add(patient_user)
    await db.flush()

    patient_profile = Patient(
        id=patient_user.id,
        first_name="Billy",
        last_name="Kid",
        phone="+1444444444",
        date_of_birth="2018-05-15",  # 8 years old
        gender="male",
        consent_flags={"dataSharing": True, "researchParticipation": False, "emergencyAccess": True, "aiProcessing": True}
    )
    db.add(patient_profile)
    await db.flush()

    # 3. Create consultation Case assigned to Doctor Jones
    case_id = uuid.uuid4()
    case_record = Case(
        id=case_id,
        patient_id=patient_user.id,
        patient_name="Billy Kid",
        patient_age=8,
        patient_gender="male",
        doctor_id=doctor_user.id,
        doctor_name="David Jones",
        specialty="Pediatrics",
        symptom_summary="Fever and cough for 2 days.",
        urgency_level="medium",
        status="routed",
        ai_confidence_score=0.92
    )
    db.add(case_record)
    await db.flush()

    # 4. Create Appointment
    appt = Appointment(
        patient_id=patient_user.id,
        doctor_id=doctor_user.id,
        patient_name="Billy Kid",
        doctor_name="David Jones",
        specialty="Pediatrics",
        hospital_name="Aronofy Boston Hospital",
        date="2026-08-12",
        time="14:00",
        duration=30,
        type="in_person",
        status="scheduled",
        reason="Fever consult"
    )
    db.add(appt)

    await db.flush()
    await db.commit()
    return {
        "doctor_id": doctor_user.id,
        "patient_id": patient_user.id,
        "case_id": case_id,
        "appointment_id": appt.id
    }

@pytest.mark.asyncio
async def test_doctor_availability_and_dashboard(client: AsyncClient, setup_doctor_data, db):
    # Log in as Doctor
    login_payload = {"email": "doctor.jones@aronofy.com", "password": "password123"}
    login_resp = await client.post("/api/v1/auth/login", json=login_payload)
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Update Availability
    avail_payload = {"availability": "busy", "next_available": "Tomorrow 9 AM"}
    avail_resp = await client.put("/api/v1/doctor/schedule/availability", json=avail_payload, headers=headers)
    assert avail_resp.status_code == 200
    assert avail_resp.json()["availability"] == "busy"
    assert avail_resp.json()["next_available"] == "Tomorrow 9 AM"

    # 2. Get Dashboard
    dash_resp = await client.get("/api/v1/doctor/dashboard", headers=headers)
    assert dash_resp.status_code == 200
    dash_data = dash_resp.json()
    assert dash_data["total_patients"] == 1
    assert len(dash_data["pending_cases"]) == 1
    assert dash_data["pending_cases"][0]["patient_name"] == "Billy Kid"


@pytest.mark.asyncio
async def test_doctor_clinical_operations(client: AsyncClient, setup_doctor_data, db):
    case_id = setup_doctor_data["case_id"]
    patient_id = setup_doctor_data["patient_id"]

    # Log in as Doctor
    login_payload = {"email": "doctor.jones@aronofy.com", "password": "password123"}
    login_resp = await client.post("/api/v1/auth/login", json=login_payload)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. List Patients and retrieve profile
    patients_resp = await client.get("/api/v1/doctor/patients", headers=headers)
    assert patients_resp.status_code == 200
    assert len(patients_resp.json()) == 1

    profile_resp = await client.get(f"/api/v1/doctor/patients/{patient_id}", headers=headers)
    assert profile_resp.status_code == 200
    assert profile_resp.json()["first_name"] == "Billy"

    # 2. Update Case Notes
    notes_payload = {"notes": "Patient shows signs of viral cold."}
    notes_resp = await client.put(f"/api/v1/doctor/cases/{case_id}/notes", json=notes_payload, headers=headers)
    assert notes_resp.status_code == 200
    assert notes_resp.json()["notes"] == "Patient shows signs of viral cold."

    # 3. Write Prescription
    rx_payload = {
        "case_id": str(case_id),
        "patient_id": str(patient_id),
        "diagnosis": "Viral Cough",
        "notes": "Rest and fluids.",
        "medications": [
            {
                "name": "Cough Syrup",
                "dosage": "5ml",
                "frequency": "three times daily",
                "duration": "5 days",
                "special_instructions": "After meals",
                "scheduled_times": ["08:00", "14:00", "20:00"],
                "start_date": "2026-07-20",
                "end_date": "2026-07-25"
            }
        ]
    }
    rx_resp = await client.post("/api/v1/doctor/prescriptions", json=rx_payload, headers=headers)
    assert rx_resp.status_code == 201
    assert rx_resp.json()["diagnosis"] == "Viral Cough"
    assert len(rx_resp.json()["medications"]) == 1

    # 4. Finalize Diagnosis
    diag_payload = {"diagnosis": "Acute Viral Nasopharyngitis", "notes": "Cough syrup prescribed."}
    diag_resp = await client.post(f"/api/v1/doctor/cases/{case_id}/diagnose", json=diag_payload, headers=headers)
    assert diag_resp.status_code == 200
    assert diag_resp.json()["status"] == "completed"

    # 5. Author Medical Report
    report_payload = {
        "patient_id": str(patient_id),
        "patient_name": "Billy Kid",
        "type": "discharge_summary",
        "title": "Pediatric Consultation Summary",
        "content": "Patient Billy Kid was diagnosed with Viral Cough. Discharged with Cough Syrup.",
        "date": "2026-07-20",
        "ai_generated": False
    }
    report_resp = await client.post("/api/v1/doctor/reports", json=report_payload, headers=headers)
    assert report_resp.status_code == 201
    assert report_resp.json()["title"] == "Pediatric Consultation Summary"


@pytest.mark.asyncio
async def test_doctor_analytics(client: AsyncClient, setup_doctor_data, db):
    # Log in as Doctor
    login_payload = {"email": "doctor.jones@aronofy.com", "password": "password123"}
    login_resp = await client.post("/api/v1/auth/login", json=login_payload)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Fetch Analytics
    analytics_resp = await client.get("/api/v1/doctor/analytics", headers=headers)
    assert analytics_resp.status_code == 200
    analytics_data = analytics_resp.json()
    assert "age_distribution" in analytics_data
    assert analytics_data["age_distribution"]["under_18"] == 1
    assert "status_distribution" in analytics_data
