from typing import Optional
from pydantic import BaseModel, EmailStr, Field
from app.schemas.patient import PatientCreate
from app.schemas.doctor import DoctorCreate

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenPayload(BaseModel):
    sub: Optional[str] = None
    role: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class PatientSignup(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=100)
    profile: PatientCreate

class DoctorSignup(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=100)
    profile: DoctorCreate

class RefreshTokenRequest(BaseModel):
    """
    Refresh token carried in the request body.

    Deliberately not a query parameter: a refresh token is session-equivalent
    and lives for days, and query strings are recorded in server access logs,
    proxy logs and browser history.
    """

    refresh_token: str = Field(min_length=1)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=8, max_length=100)
