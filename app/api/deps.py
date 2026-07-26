from typing import List
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.database import get_db
from app.core.identity import InvalidTokenError, get_token_verifier
from app.core.redis import get_redis
from app.core.exceptions import AuthenticationException, AuthorizationException
from app.models.user import User


# OAuth2 Password flow setup
reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login"
)

from app.repositories.user import user_repository

async def get_current_user(
    db: AsyncSession = Depends(get_db),
    token: str = Depends(reusable_oauth2)
) -> User:
    """
    Resolve the caller from their token and the database.

    The token establishes only *which* account is calling. Everything an
    authorisation decision is made from — role, active status, clinical
    verification — is read from the `users` row here, so a token carrying a
    forged `role` claim gains nothing: the claim is never read.
    """
    try:
        identity = get_token_verifier().verify(token).require_access_token()
    except InvalidTokenError as exc:
        raise AuthenticationException(str(exc))
    except ValueError:
        raise AuthenticationException("Signature verification failed or token expired.")

    from app.services.identity_link import resolve_local_user

    user = await resolve_local_user(db, identity)
    if not user:
        raise AuthenticationException("User account does not exist.")

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Guards routes against deactivated user accounts.
    """
    if not current_user.is_active:
        raise AuthorizationException("This user account has been deactivated.")
    return current_user

class RoleChecker:
    """
    Enforces Role-Based Access Control (RBAC) on protected endpoints.

    The role compared here comes from the database row loaded by
    `get_current_user`, never from a token claim.
    """
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(
        self,
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        if current_user.role not in self.allowed_roles:
            raise AuthorizationException(
                "You do not possess the required role to execute this request."
            )
        return current_user


async def require_approved_doctor(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Gate every doctor API on an administrator's approval.

    Checked per request against the current `doctors` row rather than trusted
    from the session, because approval can be withdrawn while a doctor is
    signed in: a token minted before a rejection or suspension must stop
    working immediately, not at its next refresh.

    Login already refuses an unapproved doctor. This is the second line, and the
    one that actually protects the data.
    """
    if current_user.role != "doctor":
        raise AuthorizationException(
            "You do not possess the required role to execute this request."
        )

    from app.services.doctor_access import assert_doctor_may_practise

    await assert_doctor_may_practise(db, current_user)
    return current_user
