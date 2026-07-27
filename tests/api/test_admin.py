import pytest
import uuid
from httpx import AsyncClient
from sqlalchemy import select
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor
from app.models.hospital import Hospital
from app.models.audit import AuditLog
from app.core.security import get_password_hash
from conftest import login_payload

@pytest.fixture
async def setup_admin_data(db):
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
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    # 1. Create Admin User
    admin_user = User(
        email="admin.system@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="admin",
        is_verified=True
    )
    db.add(admin_user)

    # 2. Create Patient User & Profile
    patient_user = User(
        email="patient.user@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="patient",
        is_verified=True
    )
    db.add(patient_user)
    await db.flush()

    patient_profile = Patient(
        id=patient_user.id,
        first_name="Alice",
        last_name="Green",
        phone="+1222222222",
        date_of_birth="1985-11-22",
        gender="female"
    )
    db.add(patient_profile)

    # 3. Create Doctor User & Profile (Verification pending)
    doctor_user = User(
        email="doctor.pending@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="doctor",
        is_verified=True
    )
    db.add(doctor_user)
    await db.flush()

    doctor_profile = Doctor(
        id=doctor_user.id,
        first_name="Robert",
        last_name="Miller",
        phone="+1333333333",
        specialty="Dermatology",
        license_number="MD-776655",
        availability="available",
        verification_status="pending"
    )
    db.add(doctor_profile)

    # 4. Create Audit Log
    audit = AuditLog(
        user_id=admin_user.id,
        user_name="Admin System",
        user_role="admin",
        action="READ",
        resource="PATIENT_PHI",
        resource_id=str(patient_user.id),
        status="success",
        details="Accessed Patient Alice Green profile",
        ip_address="127.0.0.1"
    )
    db.add(audit)


    await db.flush()
    await db.commit()
    return {
        "admin_id": admin_user.id,
        "patient_id": patient_user.id,
        "doctor_id": doctor_user.id
    }

@pytest.mark.asyncio
async def test_admin_dashboard_and_monitor(client: AsyncClient, setup_admin_data, db):
    # Log in as Admin
    _login_body = await login_payload("admin.system@aronofy.com", "password123")
    login_resp = await client.post("/api/v1/auth/login", json=_login_body)
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Fetch Dashboard
    dash_resp = await client.get("/api/v1/admin/dashboard", headers=headers)
    assert dash_resp.status_code == 200
    dash_data = dash_resp.json()
    assert dash_data["total_users"] == 3
    assert dash_data["system_status"] == "operational"

    # 2. Fetch Monitor Checks
    monitor_resp = await client.get("/api/v1/admin/monitor", headers=headers)
    assert monitor_resp.status_code == 200
    monitor_data = monitor_resp.json()
    assert monitor_data["database"]["status"] == "online"
    assert monitor_data["redis"]["status"] == "online"


@pytest.mark.asyncio
async def test_admin_user_and_verification_ops(client: AsyncClient, setup_admin_data, db):
    patient_id = setup_admin_data["patient_id"]
    doctor_id = setup_doctor_data_id = setup_admin_data["doctor_id"]

    # Log in
    _login_body = await login_payload("admin.system@aronofy.com", "password123")
    login_resp = await client.post("/api/v1/auth/login", json=_login_body)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. List Users
    users_resp = await client.get("/api/v1/admin/users", headers=headers)
    assert users_resp.status_code == 200
    assert len(users_resp.json()) == 3

    # 2. Deactivate User Account
    deactivate_payload = {"is_active": False}
    deactivate_resp = await client.put(f"/api/v1/admin/users/{patient_id}/status", json=deactivate_payload, headers=headers)
    assert deactivate_resp.status_code == 200
    assert deactivate_resp.json()["is_active"] is False

    # 3. List Pending Doctors
    pending_resp = await client.get("/api/v1/admin/doctors/pending", headers=headers)
    assert pending_resp.status_code == 200
    assert len(pending_resp.json()) == 1
    assert pending_resp.json()[0]["first_name"] == "Robert"

    # 4. Verify Doctor Profile
    verify_payload = {"verification_status": "verified"}
    verify_resp = await client.put(f"/api/v1/admin/doctors/{doctor_id}/verify", json=verify_payload, headers=headers)
    assert verify_resp.status_code == 200
    assert verify_resp.json()["verification_status"] == "verified"


@pytest.mark.asyncio
async def test_admin_hospitals_and_audit_logs(client: AsyncClient, setup_admin_data, db):
    # Log in
    _login_body = await login_payload("admin.system@aronofy.com", "password123")
    login_resp = await client.post("/api/v1/auth/login", json=_login_body)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Onboard Hospital
    hospital_payload = {
        "name": "Aronofy Miami Hospital",
        "address": "100 Biscayne Blvd",
        "city": "Miami",
        "state": "FL",
        "phone": "+1305000000",
        "email": "miami@aronofy.com",
        "coordinates": {"lat": 25.761, "lng": -80.191},
        "services": ["Emergency", "Pediatrics"],
        "emergency_capacity": "available"
    }
    hosp_resp = await client.post("/api/v1/admin/hospitals", json=hospital_payload, headers=headers)
    assert hosp_resp.status_code == 201
    assert hosp_resp.json()["name"] == "Aronofy Miami Hospital"
    hosp_id = hosp_resp.json()["id"]

    # 2. Verify Hospital Status
    verify_hosp_payload = {"verification_status": "verified"}
    verify_hosp_resp = await client.put(f"/api/v1/admin/hospitals/{hosp_id}/verify", json=verify_hosp_payload, headers=headers)
    assert verify_hosp_resp.status_code == 200

    # 3. Retrieve Audit Log trace
    audit_resp = await client.get("/api/v1/admin/audit-logs", headers=headers)
    assert audit_resp.status_code == 200
    assert len(audit_resp.json()) >= 1
    assert audit_resp.json()[0]["action"] == "READ"
