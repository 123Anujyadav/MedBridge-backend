import logging
from typing import Any
from fastapi import APIRouter, Depends, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db, get_redis, get_current_user
from app.core.exceptions import AuthenticationException
from app.models.user import User
from app.schemas.auth import (
    DoctorLoginRequest,
    DoctorSignup,
    ForgotPasswordRequest,
    LoginRequest,
    PatientSignup,
    RefreshTokenRequest,
    ResetPasswordRequest,
    Token,
)
from app.schemas.user import UserResponse
from app.services.auth import auth_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/signup/patient", status_code=status.HTTP_201_CREATED)
async def signup_patient(
    signup_data: PatientSignup,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
) -> Any:
    """
    Onboard a new patient along with their clinical profile.
    """
    user = await auth_service.signup_patient(db, signup_data, redis)
    return {"message": "Signup successful. Verification link sent.", "user_id": str(user.id)}

@router.post("/signup/doctor", status_code=status.HTTP_201_CREATED)
async def signup_doctor(
    signup_data: DoctorSignup,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
) -> Any:
    """
    Onboard a new doctor along with their clinical licensing profile.
    """
    user = await auth_service.signup_doctor(db, signup_data, redis)
    return {"message": "Signup successful. Verification link sent.", "user_id": str(user.id)}

@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
) -> Any:
    """
    Authenticate email credentials and issue access/refresh token pair.

    Patients and administrators present email and password. A clinician must
    additionally present their Doctor ID — the service refuses a doctor account
    without it, so this route cannot be used to sign a doctor in with two
    factors. Clinicians should use `/login/doctor`, which makes the requirement
    explicit in the schema.
    """
    return await auth_service.login(db, login_data, redis)

@router.post("/login/doctor", response_model=Token)
async def login_doctor(
    login_data: DoctorLoginRequest,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
) -> Any:
    """
    Clinician sign-in: Doctor ID, email and password.

    All three must belong to the same approved clinician. Every way the three
    can fail to line up returns the same message, so the endpoint cannot be used
    to discover which Doctor ID belongs to a known email address, or whether a
    given clinician has been approved yet.
    """
    return await auth_service.login_doctor(db, login_data, redis)

@router.post("/refresh", response_model=Token)
async def refresh(
    request_data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
) -> Any:
    """
    Rotate expired access tokens using a valid refresh token.

    The token is read from the request body, never the query string, so it does
    not end up in access logs or browser history.
    """
    return await auth_service.refresh_token(db, request_data.refresh_token, redis)

@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout(
    request_data: RefreshTokenRequest,
    redis: Redis = Depends(get_redis)
) -> Any:
    """
    Explicit session logout, blacklisting the active refresh token.
    """
    await auth_service.logout(request_data.refresh_token, redis)
    return {"message": "Logged out successfully."}

@router.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(
    request_data: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
) -> Any:
    """
    Triggers password recovery loop by generating a temporary reset token.
    """
    await auth_service.request_password_reset(db, request_data.email, redis)
    # Fail silently to avoid disclosing user registration status
    return {"message": "If the account exists, a password reset link has been dispatched."}

@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(
    reset_data: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
) -> Any:
    """
    Consumes password recovery token to update account credentials.
    """
    await auth_service.reset_password(db, reset_data, redis)
    return {"message": "Password updated successfully."}

@router.post("/verify-account", status_code=status.HTTP_200_OK)
async def verify_account(
    token: str,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
) -> Any:
    """
    Consumes registration verification token to activate/verify user accounts.
    """
    await auth_service.verify_account(db, token, redis)
    return {"message": "Account verified successfully."}

@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Fetch the currently authenticated user's credentials and role.
    """
    return current_user
