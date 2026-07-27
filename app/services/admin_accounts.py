"""
How many administrator accounts this platform may hold.

An administrator can read every patient record, approve clinicians and
deactivate accounts, so the number of them is a security control rather than a
capacity setting. The cap is two: enough that losing one person does not lock
the organisation out of its own system, few enough that the list can be
reasoned about.

**The database is the authority, not this module.** The
`users_admin_account_cap` trigger holds against a direct `psql` session, a
second application instance, and a background job — none of which run this
code. What is here is a secondary safeguard whose only real job is to turn the
refusal into a clean, explainable API error instead of a raw constraint
violation.

Both layers serialise on the *same* advisory lock. An audit reproduced the
alternative: with an unlocked `SELECT COUNT(*)`, two concurrent transactions
each saw one live administrator, each concluded there was room, and both
committed — three administrators. A count that is not taken under a lock is a
time-of-check/time-of-use read, and no amount of application validation fixes
that.

A deactivated or soft-deleted administrator holds no slot. That is what makes
the cap enforceable on a database that already contains more administrators
than the cap allows: retiring the extras frees their slots without deleting
history that audit logs still reference.
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BusinessRuleValidationException
from app.models.user import User

logger = logging.getLogger(__name__)

MAX_ADMIN_ACCOUNTS = 2
"""The maximum number of live administrator accounts, platform-wide."""

ADMIN_CAP_LOCK_KEY = 7749283016112233445
"""
The advisory lock guarding the administrator count.

Must stay identical to the constant compiled into the `users_admin_account_cap`
trigger by migration `c7d3ba61e904`. Two different keys would not exclude each
other, which is precisely the race both are there to close.
"""

CAP_REACHED_MESSAGE = (
    f"This platform allows a maximum of {MAX_ADMIN_ACCOUNTS} administrator "
    "accounts. Deactivate an existing administrator before creating another."
)


def _live_admin_filter():
    """
    What counts as an administrator holding a slot.

    Active, not soft-deleted, role `admin`. Kept in one place so the Python
    guard and the database trigger cannot drift apart in what they count.
    """
    return (
        User.role == "admin",
        User.is_active.is_(True),
        User.deleted_at.is_(None),
    )


async def count_admin_accounts(db: AsyncSession) -> int:
    """How many administrator accounts are currently live."""
    return int(
        await db.scalar(select(func.count(User.id)).where(*_live_admin_filter())) or 0
    )


async def _lock_admin_cap(db: AsyncSession) -> None:
    """
    Queue behind every other writer that could create an administrator.

    Transaction-scoped, so it is released by the commit or rollback that ends
    the request — there is no unlock to forget and none to leak if the caller
    raises. Skipped on SQLite, which has no advisory locks and, being a
    single-writer file, does not need them; the test harness runs there.
    """
    if db.bind is None or db.bind.dialect.name != "postgresql":
        return
    await db.execute(
        text("SELECT pg_advisory_xact_lock(:key)"), {"key": ADMIN_CAP_LOCK_KEY}
    )


async def assert_admin_slot_available(db: AsyncSession) -> int:
    """
    Raise unless another administrator may be created.

    Takes the advisory lock *before* counting. Without it this is the same
    time-of-check/time-of-use read the database trigger used to have: two
    requests could both observe a free slot and both proceed. The lock is held
    until the caller's transaction ends, so the write that follows this check
    is covered by the check.

    Returns the count observed, so a caller that wants to log or report it does
    not have to ask twice.
    """
    await _lock_admin_cap(db)

    current = await count_admin_accounts(db)
    if current >= MAX_ADMIN_ACCOUNTS:
        logger.warning(
            "[ADMIN_CAP_REACHED] refused a new administrator; %d of %d in use",
            current, MAX_ADMIN_ACCOUNTS,
        )
        raise BusinessRuleValidationException(CAP_REACHED_MESSAGE)
    return current


def is_admin_cap_violation(exc: BaseException) -> bool:
    """
    Whether a database error is the cap trigger refusing a write.

    The trigger raises `check_violation`, which SQLAlchemy surfaces as an
    `IntegrityError` carrying the raw PostgreSQL message. Callers use this to
    answer with the same wording the application guard uses, rather than
    leaking constraint text to the client.
    """
    return "administrator account limit reached" in str(exc).lower()
