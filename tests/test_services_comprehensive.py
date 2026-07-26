import uuid
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.exceptions import EntityNotFoundException, AuthenticationException
from app.services.admin import admin_service
from app.services.auth import auth_service
from app.services.emergency import emergency_service
from app.services.shared import shared_service
from app.models.user import User
from app.models.patient import Patient
from app.schemas.patient import PatientCreate
from app.schemas.auth import PatientSignup, LoginRequest

@pytest.mark.asyncio
async def test_admin_service_methods(db: AsyncSession, mock_redis):
    """
    Tests AdminService dashboard, analytics, system status, and user status toggle routines.
    """
    dashboard = await admin_service.get_dashboard(db)
    assert "total_users" in dashboard
    assert dashboard["system_status"] == "operational"

    analytics = await admin_service.get_analytics(db)
    assert "emergency_success_ratio" in analytics
    assert "users_by_role" in analytics

    status_data = await admin_service.get_system_status(db, mock_redis)
    assert status_data["database"]["status"] == "online"
    assert status_data["redis"]["status"] == "online"

    # Nonexistent user update status raises EntityNotFoundException
    with pytest.raises(EntityNotFoundException):
        await admin_service.update_user_status(db, uuid.uuid4(), False)

    # Nonexistent doctor verification raises EntityNotFoundException
    with pytest.raises(EntityNotFoundException):
        await admin_service.verify_doctor(db, uuid.uuid4(), "verified")

    # Nonexistent hospital verification raises EntityNotFoundException
    with pytest.raises(EntityNotFoundException):
        await admin_service.verify_hospital(db, uuid.uuid4(), "verified")

@pytest.mark.asyncio
async def test_auth_service_edge_cases(db: AsyncSession, mock_redis):
    """
    Tests Authentication service edge cases, password resets, and user verification.
    """
    signup_data = PatientSignup(
        email="edge_test_user@aronofy.com",
        password="Password123!",
        profile=PatientCreate(
            first_name="Edge",
            last_name="Tester",
            phone="555-987-6543",
            date_of_birth="1995-05-15",
            gender="male"
        )
    )
    user = await auth_service.signup_patient(db, signup_data, mock_redis)
    assert user is not None

    # Invalid login password
    with pytest.raises(AuthenticationException):
        await auth_service.login(
            db,
            LoginRequest(email="edge_test_user@aronofy.com", password="WrongPassword!"),
            mock_redis
        )

    # Nonexistent user login
    with pytest.raises(AuthenticationException):
        await auth_service.login(
            db,
            LoginRequest(email="nonexistent@aronofy.com", password="Password123!"),
            mock_redis
        )

    # Password reset request
    await auth_service.request_password_reset(db, "edge_test_user@aronofy.com", mock_redis)
    # Password reset for non-existent email fails silently for security
    await auth_service.request_password_reset(db, "ghost@aronofy.com", mock_redis)

@pytest.mark.asyncio
async def test_emergency_and_shared_services(db: AsyncSession, mock_redis):
    """
    Tests Emergency panic triggering and shared user settings preference routines.
    """
    # Create patient user first
    patient_user = User(
        email="panic_patient@aronofy.com",
        hashed_password="hashed_pass_123",
        role="patient",
        is_active=True
    )
    db.add(patient_user)
    await db.flush()

    patient_profile = Patient(
        id=patient_user.id,
        first_name="Panic",
        last_name="Patient",
        phone="555-111-2222",
        date_of_birth="1990-01-01",
        gender="female"
    )
    db.add(patient_profile)
    await db.flush()

    # Trigger panic button
    panic_req = await emergency_service.trigger_panic(
        db,
        patient_user.id,
        {"latitude": 40.7128, "longitude": -74.0060}
    )
    assert panic_req is not None
    assert panic_req.status in ["active", "dispatched"]

    # Retrieve panic request status
    tracked = await emergency_service.track_emergency(db, patient_user.id, panic_req.id)
    assert tracked.id == panic_req.id

    # Test user settings preferences
    settings = await shared_service.get_user_settings(mock_redis, patient_user.id)
    assert settings["theme"] == "dark"

    updated = await shared_service.update_user_settings(mock_redis, patient_user.id, {"theme": "light"})
    assert updated["theme"] == "light"
