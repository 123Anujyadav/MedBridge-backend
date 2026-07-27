"""
Whether a doctor account may practise on the platform.

One rule, in one place, used by both the login flow and the doctor API guard so
the two can never disagree about what "approved" means. A clinician reaching
patient records is the highest-consequence authorisation decision this system
makes, and it is deliberately a closed allowlist: a status this module has not
been taught about denies access rather than falling through to permit.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.doctor_code import doctor_codes_match
from app.core.exceptions import AuthenticationException, AuthorizationException
from app.models.doctor import Doctor
from app.models.user import User

logger = logging.getLogger(__name__)

APPROVED_STATUS = "verified"
"""The only verification status that grants access to the doctor portal."""

AWAITING_APPROVAL_MESSAGE = "Your account is awaiting administrator approval."
"""
Shown for `pending` and `under_review`.

Worded exactly as the authentication policy specifies, so the clinician is told
their account is progressing rather than that something failed.
"""

REJECTED_MESSAGE = (
    "Your clinician account was not approved. Please contact the administrator."
)
EXPIRED_MESSAGE = (
    "Your clinician credentials have expired. Please contact the administrator "
    "to re-verify your account."
)
MISSING_PROFILE_MESSAGE = (
    "Your clinician profile is incomplete. Please contact the administrator."
)

_MESSAGES = {
    "pending": AWAITING_APPROVAL_MESSAGE,
    "under_review": AWAITING_APPROVAL_MESSAGE,
    "rejected": REJECTED_MESSAGE,
    "expired": EXPIRED_MESSAGE,
}


async def assert_doctor_may_practise(db: AsyncSession, user: User) -> Doctor:
    """
    Raise unless this doctor has been approved by an administrator.

    Returns the doctor profile so callers that need it do not have to load it
    twice. Suspension is handled upstream by the `is_active` check, which is the
    same switch the admin "suspend" action flips.
    """
    doctor = await db.get(Doctor, user.id)

    if doctor is None:
        # A doctor user with no clinical profile cannot be evaluated, so it is
        # refused rather than allowed by default.
        logger.warning("[DOCTOR_ACCESS_DENIED] user=%s reason=no_profile", user.id)
        raise AuthorizationException(MISSING_PROFILE_MESSAGE)

    status = (doctor.verification_status or "").strip().lower()
    if status == APPROVED_STATUS:
        return doctor

    logger.warning(
        "[DOCTOR_ACCESS_DENIED] user=%s status=%s", user.id, status or "unset"
    )
    raise AuthorizationException(_MESSAGES.get(status, AWAITING_APPROVAL_MESSAGE))


BAD_DOCTOR_CREDENTIALS_MESSAGE = (
    "Doctor ID, email address or password is incorrect."
)
"""
One message for every way the three factors can fail to line up.

Deliberately undifferentiated: telling someone holding a stolen password that
the Doctor ID was the only wrong part turns a three-factor sign-in back into a
two-factor one plus a guessing oracle.
"""


async def assert_doctor_sign_in(
    db: AsyncSession, user: User, presented_code: str | None
) -> Doctor:
    """
    Complete the clinician sign-in checks after the password has been proven.

    Runs in the order the authentication policy specifies — the account is a
    doctor, a clinical profile exists, the Doctor ID matches, and only then is
    the approval status considered. Checking the Doctor ID before the status is
    what stops the endpoint reporting a clinician's approval state to someone
    who does not hold their Doctor ID.

    Returns the doctor profile so the caller does not load it a second time.
    """
    if user.role != "doctor":
        # The clinician endpoint is not a second door into a patient or
        # administrator account.
        logger.warning("[DOCTOR_LOGIN_DENIED] user=%s reason=not_a_doctor", user.id)
        raise AuthenticationException(BAD_DOCTOR_CREDENTIALS_MESSAGE)

    doctor = await db.get(Doctor, user.id)
    if doctor is None:
        logger.warning("[DOCTOR_LOGIN_DENIED] user=%s reason=no_profile", user.id)
        raise AuthenticationException(BAD_DOCTOR_CREDENTIALS_MESSAGE)

    if not doctor_codes_match(presented_code, doctor.doctor_code):
        # Covers both a wrong code and a profile that has never been issued one:
        # `doctor_codes_match` refuses a missing stored code, so omitting the
        # field can never authenticate anybody.
        logger.warning("[DOCTOR_LOGIN_DENIED] user=%s reason=doctor_id_mismatch",
                       user.id)
        raise AuthenticationException(BAD_DOCTOR_CREDENTIALS_MESSAGE)

    # Only now — the caller has proven all three factors, so it is safe to tell
    # them why an otherwise valid account cannot be used.
    await assert_doctor_may_practise(db, user)
    return doctor
