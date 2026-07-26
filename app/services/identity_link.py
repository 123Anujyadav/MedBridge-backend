"""
Resolving a verified identity to the local user record.

The identity provider says *who authenticated*. This module answers *which
account that is here* — and that account, not the token, carries the role,
the permissions and every clinical record.

Linking is by email, once, on first Supabase sign-in. An existing clinician who
has been using the platform for months keeps their user id, their cases, their
reports and their approval status; a second account is never created for
someone who already exists.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.identity import VerifiedIdentity
from app.models.user import User

logger = logging.getLogger(__name__)


async def resolve_local_user(
    db: AsyncSession, identity: VerifiedIdentity
) -> User | None:
    """
    Find the local account for a verified identity.

    Returns None when no account exists, which callers turn into a 401: an
    identity that authenticated with the provider but has no record here is not
    a user of this platform. Provisioning happens only through the signup
    endpoints, never implicitly from a token.
    """
    if identity.provider != "supabase":
        # The built-in provider issues tokens whose subject is the local user id.
        return await db.get(User, _as_uuid(identity.subject))

    # 1. Already linked — the common path, one indexed lookup.
    linked = await db.scalar(
        select(User).where(User.supabase_user_id == identity.subject)
    )
    if linked is not None:
        return linked

    # 2. First sign-in for a pre-existing account: match on the verified email.
    if not identity.email:
        logger.warning(
            "[IDENTITY_UNLINKED] supabase_user=%s has no email claim to link on",
            identity.subject,
        )
        return None

    existing = await db.scalar(
        select(User).where(User.email == identity.email.strip().lower())
    )
    if existing is None:
        logger.warning(
            "[IDENTITY_NO_LOCAL_ACCOUNT] email=%s authenticated but has no "
            "local record; refusing rather than provisioning one",
            identity.email,
        )
        return None

    if existing.supabase_user_id and existing.supabase_user_id != identity.subject:
        # Two identities claiming one account: refuse rather than silently
        # re-point an established account at a different credential.
        logger.error(
            "[IDENTITY_CONFLICT] local user=%s already linked to %s, token "
            "presented %s",
            existing.id, existing.supabase_user_id, identity.subject,
        )
        return None

    existing.supabase_user_id = identity.subject
    await db.flush()
    logger.info(
        "[IDENTITY_LINKED] local user=%s linked to supabase user=%s by email",
        existing.id, identity.subject,
    )
    return existing


def _as_uuid(value: str):
    """Local subjects are user ids; a malformed one simply matches nothing."""
    import uuid

    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None
