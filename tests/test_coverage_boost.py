import uuid
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor
from app.services.appointment import appointment_service
from app.services.shared import shared_service
from app.services.auth import auth_service
from app.schemas.patient_api import AppointmentCreateRequest
from app.schemas.doctor import DoctorCreate
from app.schemas.auth import DoctorSignup
from app.api.deps import RoleChecker, get_current_active_user
from app.core.exceptions import EntityNotFoundException, AuthorizationException, AuthenticationException

@pytest.mark.asyncio
async def test_appointment_service_flow(db: AsyncSession):
    """
    Tests appointment booking, status fetching, and cancellation.
    """
    p_user = User(email="appt_p@aronofy.com", hashed_password="pw", role="patient", is_active=True)
    d_user = User(email="appt_d@aronofy.com", hashed_password="pw", role="doctor", is_active=True)
    db.add_all([p_user, d_user])
    await db.flush()

    p_prof = Patient(id=p_user.id, first_name="Appt", last_name="Patient", phone="555-9999", date_of_birth="1992-02-02", gender="male")
    d_prof = Doctor(verification_status="verified", id=d_user.id, first_name="Appt", last_name="Doctor", phone="555-8888", license_number="LIC-APP", specialty="Cardiology")
    db.add_all([p_prof, d_prof])
    await db.flush()

    req = AppointmentCreateRequest(
        doctor_id=d_user.id,
        specialty="Cardiology",
        hospital_name="Aronofy General Hospital",
        date="2026-08-01",
        time="10:00",
        type="video",
        reason="Routine Checkup"
    )

    # Book appointment
    appt = await appointment_service.book_appointment(db, p_user.id, req)
    assert appt.id is not None
    assert appt.status == "scheduled"

    # Cancel appointment
    cancelled = await appointment_service.cancel_appointment(db, p_user.id, appt.id)
    assert cancelled.status == "cancelled"

    # Cancel non-existent appointment
    with pytest.raises(EntityNotFoundException):
        await appointment_service.cancel_appointment(db, p_user.id, uuid.uuid4())

@pytest.mark.asyncio
async def test_shared_service_calendar_and_search(db: AsyncSession):
    """
    Tests shared service calendar aggregation and entity search routines.
    """
    user = User(email="cal_u@aronofy.com", hashed_password="pw", role="patient", is_active=True)
    db.add(user)
    await db.flush()

    events = await shared_service.get_calendar_events(db, user)
    assert isinstance(events, list)

    search_results = await shared_service.search_entities(db, "General")
    assert isinstance(search_results, list)

@pytest.mark.asyncio
async def test_auth_token_refresh_and_logout(db: AsyncSession, mock_redis):
    """
    Tests refresh token rotation, logout blacklisting, and doctor signup.
    """
    from app.core.security import create_refresh_token
    user = User(email="refresh_u@aronofy.com", hashed_password="pw", role="patient", is_active=True)
    db.add(user)
    await db.flush()

    ref_token = create_refresh_token(user.id)
    # Refresh access token
    new_tokens = await auth_service.refresh_token(db, ref_token, mock_redis)
    assert new_tokens.access_token is not None

    # Logout user
    await auth_service.logout(ref_token, mock_redis)

    # Doctor signup
    doc_signup = DoctorSignup(
        email="new_doc_signup@aronofy.com",
        password="Password123!",
        profile=DoctorCreate(
            first_name="Doc",
            last_name="Signup",
            phone="555-7777",
            license_number="LIC-777",
            specialty="Pediatrics"
        )
    )
    doc_user = await auth_service.signup_doctor(db, doc_signup, mock_redis)
    assert doc_user.id is not None

@pytest.mark.asyncio
async def test_rbac_role_checker():
    """
    Tests RBAC RoleChecker logic for active/inactive users and allowed roles.
    """
    active_patient = User(email="act_p@aronofy.com", hashed_password="pw", role="patient", is_active=True)
    inactive_patient = User(email="inact_p@aronofy.com", hashed_password="pw", role="patient", is_active=False)

    # get_current_active_user checks
    assert (await get_current_active_user(active_patient)) == active_patient
    with pytest.raises(AuthorizationException):
        await get_current_active_user(inactive_patient)

    # RoleChecker checks
    admin_checker = RoleChecker(["admin"])
    with pytest.raises(AuthorizationException):
        admin_checker(active_patient)
