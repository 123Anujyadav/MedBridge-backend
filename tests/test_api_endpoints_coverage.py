import uuid
import pytest
from httpx import AsyncClient
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor
from app.models.hospital import Hospital
from app.models.case import Case
from app.core.security import create_access_token
from sqlalchemy.ext.asyncio import AsyncSession

@pytest.mark.asyncio
async def test_all_portal_endpoints_coverage(client: AsyncClient, db: AsyncSession):
    """
    Executes authenticated requests across Patient, Doctor, Admin, and Shared API endpoints.
    """
    p_user = User(email="portal_p2@aronofy.com", hashed_password="pw", role="patient", is_active=True)
    d_user = User(email="portal_d2@aronofy.com", hashed_password="pw", role="doctor", is_active=True)
    a_user = User(email="portal_a2@aronofy.com", hashed_password="pw", role="admin", is_active=True)
    db.add_all([p_user, d_user, a_user])
    await db.flush()

    p_prof = Patient(id=p_user.id, first_name="Portal", last_name="Patient", phone="555-0001", date_of_birth="1990-01-01", gender="female")
    d_prof = Doctor(verification_status="verified", id=d_user.id, first_name="Portal", last_name="Doctor", phone="555-0002", license_number="LIC-PORTAL2", specialty="Neurology")
    hosp = Hospital(name="City Central Hospital", address="100 Main St", city="Boston", state="MA", phone="555-4444", email="info@cityhospital.com", emergency_capacity="available")
    db.add_all([p_prof, d_prof, hosp])
    await db.flush()

    case = Case(
        patient_id=p_user.id,
        patient_name="Portal Patient",
        patient_age=34,
        patient_gender="female",
        doctor_id=d_user.id,
        doctor_name="Portal Doctor",
        specialty="Neurology",
        symptom_summary="Headache",
        urgency_level="low",
        status="in_consultation",
        notes=""
    )
    db.add(case)
    await db.flush()

    p_token = create_access_token(p_user.id)
    d_token = create_access_token(d_user.id)
    a_token = create_access_token(a_user.id)

    p_headers = {"Authorization": f"Bearer {p_token}"}
    d_headers = {"Authorization": f"Bearer {d_token}"}
    a_headers = {"Authorization": f"Bearer {a_token}"}

    # Health Endpoints Coverage
    assert (await client.get("/health")).status_code == 200
    assert (await client.get("/live")).status_code == 200
    assert (await client.get("/ready")).status_code in (200, 503)

    # Auth Me Endpoint
    assert (await client.get("/api/v1/auth/me", headers=p_headers)).status_code == 200

    # Patient Portal Endpoints
    assert (await client.get("/api/v1/patient/profile", headers=p_headers)).status_code == 200
    assert (await client.get("/api/v1/patient/dashboard", headers=p_headers)).status_code == 200
    assert (await client.get("/api/v1/patient/appointments", headers=p_headers)).status_code == 200
    assert (await client.get("/api/v1/patient/reports", headers=p_headers)).status_code == 200
    assert (await client.get("/api/v1/patient/prescriptions", headers=p_headers)).status_code == 200
    assert (await client.get("/api/v1/patient/notifications", headers=p_headers)).status_code == 200

    # Patient Consent Update
    r_consent = await client.put(
        "/api/v1/patient/consent",
        json={"dataSharing": True, "researchParticipation": False, "emergencyAccess": True, "aiProcessing": True},
        headers=p_headers
    )
    assert r_consent.status_code == 200

    # Emergency Panic Trigger Endpoint
    r_emerg = await client.post(
        "/api/v1/patient/emergency",
        json={"location": {"lat": 40.7128, "lng": -74.0060, "address": "123 Fifth Ave"}},
        headers=p_headers
    )
    assert r_emerg.status_code == 201

    # Doctor Portal Endpoints
    assert (await client.get("/api/v1/doctor/profile", headers=d_headers)).status_code == 200
    assert (await client.get("/api/v1/doctor/dashboard", headers=d_headers)).status_code == 200
    assert (await client.get("/api/v1/doctor/analytics", headers=d_headers)).status_code == 200
    assert (await client.get("/api/v1/doctor/cases", headers=d_headers)).status_code == 200
    assert (await client.get("/api/v1/doctor/patients", headers=d_headers)).status_code == 200
    assert (await client.get("/api/v1/doctor/appointments", headers=d_headers)).status_code == 200
    assert (await client.get("/api/v1/doctor/prescriptions", headers=d_headers)).status_code == 200
    assert (await client.get(f"/api/v1/doctor/cases/{case.id}", headers=d_headers)).status_code == 200

    # Doctor Availability Update
    r_avail = await client.put(
        "/api/v1/doctor/schedule/availability",
        json={"availability": "available", "next_available": "Tomorrow 9 AM"},
        headers=d_headers
    )
    assert r_avail.status_code == 200

    # Admin Portal Endpoints
    assert (await client.get("/api/v1/admin/dashboard", headers=a_headers)).status_code == 200
    assert (await client.get("/api/v1/admin/analytics", headers=a_headers)).status_code == 200
    assert (await client.get("/api/v1/admin/users", headers=a_headers)).status_code == 200
    assert (await client.get("/api/v1/admin/doctors/pending", headers=a_headers)).status_code == 200
    assert (await client.get("/api/v1/admin/hospitals", headers=a_headers)).status_code == 200
    assert (await client.get("/api/v1/admin/audit-logs", headers=a_headers)).status_code == 200
    assert (await client.get("/api/v1/admin/monitor", headers=a_headers)).status_code == 200

    # Admin Doctor Verification
    r_vdoc = await client.put(f"/api/v1/admin/doctors/{d_user.id}/verify", json={"verification_status": "verified"}, headers=a_headers)
    assert r_vdoc.status_code == 200

    # Admin Hospital Verification
    r_vhosp = await client.put(f"/api/v1/admin/hospitals/{hosp.id}/verify", json={"verification_status": "verified"}, headers=a_headers)
    assert r_vhosp.status_code == 200

    # Admin User Status Deactivation
    r_uact = await client.put(f"/api/v1/admin/users/{p_user.id}/status", json={"is_active": True}, headers=a_headers)
    assert r_uact.status_code == 200

    # Shared Endpoints
    assert (await client.get("/api/v1/shared/search?q=Neurology", headers=p_headers)).status_code == 200
    assert (await client.get("/api/v1/shared/calendar", headers=p_headers)).status_code == 200
    assert (await client.get("/api/v1/shared/settings", headers=p_headers)).status_code == 200
