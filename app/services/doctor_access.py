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

from app.core.exceptions import AuthorizationException
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
