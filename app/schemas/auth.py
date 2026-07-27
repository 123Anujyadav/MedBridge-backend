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

    doctor_id: Optional[str] = Field(default=None, max_length=64)
    """
    The clinician's 8-character Doctor ID.

    Optional on this model so patients and administrators post exactly what they
    always did, and so the shared endpoint keeps one request shape. It is not
    optional in effect: `AuthService.login` refuses any account whose role is
    `doctor` unless the value is present and matches.

    Deliberately *not* validated against `[A-Z0-9]{8}` here. A pattern on the
    schema would answer a malformed guess with 422 and a well-formed wrong one
    with 401, which tells an attacker when their guess at least has the right
    shape. Every value that could plausibly be a credential is instead carried
    through to `doctor_codes_match`, which answers all of them identically.
    The 64-character ceiling is a request-size bound, not a format check — it
    is well clear of an 8-character code however much whitespace surrounds it.
    """


class DoctorLoginRequest(BaseModel):
    """
    The clinician sign-in form: all three factors, all required.

    Separate from `LoginRequest` so the doctor portal cannot omit the Doctor ID
    by accident — an absent field is a 422 from the schema rather than a
    decision made later in the service.

    An *empty* field is a different thing from an absent one: it is a supplied
    credential that happens to be wrong, so `min_length` is not set and it
    reaches the same uniform 401 as any other incorrect Doctor ID.
    """

    doctor_id: str = Field(max_length=64)
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
