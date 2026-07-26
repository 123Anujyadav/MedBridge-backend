import pytest
from httpx import AsyncClient
from sqlalchemy import select
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor
from tests.conftest import MockRedis

@pytest.mark.asyncio
async def test_patient_auth_flow(client: AsyncClient, mock_redis: MockRedis, db):
    """
    Validates complete Patient authentication and session management lifecycle.
    """
    # 1. Onboard / Sign Up Patient
    signup_payload = {
        "email": "patient@aronofy.com",
        "password": "securepassword123",
        "profile": {
            "first_name": "John",
            "last_name": "Doe",
            "phone": "+1234567890",
            "date_of_birth": "1990-01-01",
            "gender": "male",
            "blood_type": "O+",
            "height": 180.5,
            "weight": 75.2,
            "address": "123 Main St",
            "city": "Boston",
            "state": "MA",
            "emergency_contact": {
                "name": "Jane Doe",
                "phone": "+9876543210",
                "relationship": "Spouse"
            },
            "allergies": ["Peanuts"],
            "chronic_conditions": [],
            "medications": [],
            "insurance_provider": "BlueCross",
            "insurance_number": "BC1234567",
            "consent_flags": {
                "dataSharing": True,
                "researchParticipation": False,
                "emergencyAccess": True,
                "aiProcessing": True
            }
        }
    }
    
    response = await client.post("/api/v1/auth/signup/patient", json=signup_payload)
    assert response.status_code == 201
    assert "user_id" in response.json()
    user_id = response.json()["user_id"]

    # Verify Patient and User record created in Database
    stmt = select(User).where(User.email == "patient@aronofy.com")
    result = await db.execute(stmt)
    db_user = result.scalars().first()
    assert db_user is not None
    assert db_user.role == "patient"
    assert db_user.is_verified is False
    
    # 2. Account Verification
    # Extract verification token cached in mock Redis
    verify_token = None
    for key, val in mock_redis.store.items():
        if key.startswith("verify_token:") and val == user_id:
            verify_token = key.split(":")[1]
            break
            
    assert verify_token is not None
    
    # Verify account
    verify_response = await client.post(f"/api/v1/auth/verify-account?token={verify_token}")
    assert verify_response.status_code == 200
    
    # Verify User is marked as verified in DB
    await db.refresh(db_user)
    assert db_user.is_verified is True
    assert f"verify_token:{verify_token}" not in mock_redis.store

    # 3. Log In User
    login_payload = {
        "email": "patient@aronofy.com",
        "password": "securepassword123"
    }
    login_response = await client.post("/api/v1/auth/login", json=login_payload)
    assert login_response.status_code == 200
    tokens = login_response.json()
    assert "access_token" in tokens
    assert "refresh_token" in tokens
    access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]

    # 4. Fetch Current Active User (/me)
    headers = {"Authorization": f"Bearer {access_token}"}
    me_response = await client.get("/api/v1/auth/me", headers=headers)
    assert me_response.status_code == 200
    assert me_response.json()["email"] == "patient@aronofy.com"
    assert me_response.json()["role"] == "patient"

    # 5. Token Refresh and Rotation
    refresh_response = await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert refresh_response.status_code == 200
    new_tokens = refresh_response.json()
    assert "access_token" in new_tokens
    assert "refresh_token" in new_tokens
    new_access_token = new_tokens["access_token"]
    new_refresh_token = new_tokens["refresh_token"]

    # Verify old refresh token is blacklisted in mock Redis
    assert f"revoked_token:{refresh_token}" in mock_redis.store

    # Verify refresh failure on revoked token reuse (Blacklist check)
    failed_refresh = await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert failed_refresh.status_code == 401

    # 6. Logout
    logout_response = await client.post("/api/v1/auth/logout", json={"refresh_token": new_refresh_token})
    assert logout_response.status_code == 200
    assert f"revoked_token:{new_refresh_token}" in mock_redis.store


@pytest.mark.asyncio
async def test_doctor_auth_flow(client: AsyncClient, mock_redis: MockRedis, db):
    """
    Validates Doctor onboarding credentials mapping.
    """
    signup_payload = {
        "email": "doctor@aronofy.com",
        "password": "doctorsecurepass",
        "profile": {
            "first_name": "Sarah",
            "last_name": "Smith",
            "phone": "+1999888777",
            "specialty": "Cardiology",
            "sub_specialties": ["Heart Failure"],
            "license_number": "MD-998877",
            "years_of_experience": 12,
            "availability": "available",
            "consultation_fee": 150.0,
            "education": ["Harvard Medical School"],
            "certifications": ["American Board of Internal Medicine"],
            "languages": ["English", "Spanish"],
            "bio": "Cardiologist specializing in cardiovascular failure."
        }
    }
    
    response = await client.post("/api/v1/auth/signup/doctor", json=signup_payload)
    assert response.status_code == 201
    
    stmt = select(User).where(User.email == "doctor@aronofy.com")
    result = await db.execute(stmt)
    db_user = result.scalars().first()
    assert db_user is not None
    assert db_user.role == "doctor"


@pytest.mark.asyncio
async def test_password_recovery_flow(client: AsyncClient, mock_redis: MockRedis, db):
    """
    Validates Forgot and Reset Password routines.
    """
    # Create a user in DB
    user = User(email="recover@aronofy.com", hashed_password="oldhashedpassword", role="patient", is_verified=True)
    db.add(user)
    await db.flush()
    await db.commit()

    # 1. Request Password Reset Link
    forgot_payload = {"email": "recover@aronofy.com"}
    response = await client.post("/api/v1/auth/forgot-password", json=forgot_payload)
    assert response.status_code == 200

    # Extract reset token from mock Redis keys
    reset_token = None
    for key, val in mock_redis.store.items():
        if key.startswith("reset_token:") and val == str(user.id):
            reset_token = key.split(":")[1]
            break

    assert reset_token is not None

    # 2. Reset Password
    reset_payload = {
        "token": reset_token,
        "new_password": "brandnewpassword123"
    }
    reset_response = await client.post("/api/v1/auth/reset-password", json=reset_payload)
    assert reset_response.status_code == 200

    # Verify reset token consumed/deleted in mock Redis
    assert f"reset_token:{reset_token}" not in mock_redis.store

    # 3. Verify Log In works with new password
    login_payload = {
        "email": "recover@aronofy.com",
        "password": "brandnewpassword123"
    }
    login_response = await client.post("/api/v1/auth/login", json=login_payload)
    assert login_response.status_code == 200
    assert "access_token" in login_response.json()
