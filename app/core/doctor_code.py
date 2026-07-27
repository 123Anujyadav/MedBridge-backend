"""
The clinician's 8-character Doctor ID.

A doctor signs in with three factors — Doctor ID, email and password — so the
Doctor ID is a *secret* as well as an identifier. That is the whole reason this
module exists rather than a one-line `random.choice` loop at the call site:

* It is generated with `secrets`, never `random`. A predictable identifier would
  reduce clinician sign-in to the two factors everybody else uses.
* It is unique, enforced by a unique index on `doctors.doctor_code` and by an
  allocation loop here that retries on collision rather than trusting one draw.
* It is compared in constant time, so a mismatched code cannot be recovered one
  character at a time from response timings.

The stored form is always upper case `A-Z0-9`; `normalise` is applied to
anything arriving from a user so a clinician typing lower case or pasting a code
with a stray space is not told their credentials are wrong.
"""

from __future__ import annotations

import logging
import re
import secrets
from hmac import compare_digest

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

DOCTOR_CODE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
"""Upper-case letters and digits, exactly as the authentication policy specifies."""

DOCTOR_CODE_LENGTH = 8

DOCTOR_CODE_PATTERN = re.compile(r"\A[A-Z0-9]{8}\Z")
"""
Anchored with `\\A`/`\\Z`, not `^`/`$`.

In Python `$` also matches immediately before a trailing newline, so
`^[A-Z0-9]{8}$` accepts `"ABCDEFGH\\n"` — a nine-character value that the
database's own `^[A-Z0-9]{8}$` check (POSIX, where `$` is end-of-string) would
reject. `\\Z` is a true end-of-string anchor and keeps the two in step.
"""

_MAX_ALLOCATION_ATTEMPTS = 12
"""
Draws before giving up.

The keyspace is 36**8 ≈ 2.8e12, so a collision is already vanishingly unlikely;
this bound exists so a misconfigured database cannot spin forever.
"""


def generate_doctor_code() -> str:
    """
    Draw one candidate Doctor ID.

    Uniqueness is *not* checked here — use `allocate_doctor_code` for anything
    that is about to be written to the database.
    """
    return "".join(
        secrets.choice(DOCTOR_CODE_ALPHABET) for _ in range(DOCTOR_CODE_LENGTH)
    )


def normalise_doctor_code(raw: str | None) -> str:
    """
    Put a user-supplied Doctor ID into its stored form.

    Upper-cased and stripped of surrounding and internal whitespace, because a
    code is read off a screen or an email and retyped, and neither `dr8a9xq2`
    nor `DR8A 9XQ2` is a wrong credential — just a differently typed one.
    """
    if not raw:
        return ""
    return "".join(raw.split()).upper()


def is_valid_doctor_code(value: str | None) -> bool:
    """Whether `value` is already in the canonical 8-character form."""
    return bool(value) and bool(DOCTOR_CODE_PATTERN.match(value))


def doctor_codes_match(presented: str | None, stored: str | None) -> bool:
    """
    Compare a presented Doctor ID against the stored one, in constant time.

    Both sides are validated into the canonical `[A-Z0-9]{8}` form *before* the
    comparison, and anything that is not already in that form is simply a
    non-match. Two reasons, and the first is not cosmetic:

    `hmac.compare_digest` raises `TypeError` on a string containing any
    non-ASCII character. An audit reproduced this: a Doctor ID of `ÀÀÀÀÀÀÀÀ`,
    an emoji, a Cyrillic letter or a Kelvin sign turned an unauthenticated
    sign-in attempt into an unhandled 500 with a stack trace in the log. A 500
    is also perfectly distinguishable from a 401, so it undid the uniform
    failure response the three-factor design depends on.

    Second, it keeps the comparison total: every input now yields True or
    False, so the caller has exactly one failure path to handle.

    A missing or malformed *stored* code never matches either. That is what
    stops a clinical profile which has never been issued an ID from being
    signed into by omitting the field.
    """
    if not presented or not stored:
        return False

    candidate = normalise_doctor_code(presented)
    if not is_valid_doctor_code(candidate) or not is_valid_doctor_code(stored):
        return False

    # Both operands are now known to be exactly 8 ASCII characters, so
    # `compare_digest` cannot raise and leaks nothing through its timing.
    return compare_digest(candidate, stored)


async def allocate_doctor_code(db: AsyncSession) -> str:
    """
    Reserve a Doctor ID that no other clinician holds.

    The database's unique index is the real guarantee; this pre-check just keeps
    the common path free of integrity errors. It is deliberately *not* the
    guarantee itself: two concurrent approvals can both pass the check and only
    one can win the index. Callers that write the result must therefore be able
    to retry — see `assign_doctor_code`, which is what the approval path uses.
    """
    from app.models.doctor import Doctor

    for _ in range(_MAX_ALLOCATION_ATTEMPTS):
        candidate = generate_doctor_code()
        taken = await db.scalar(
            select(Doctor.id).where(Doctor.doctor_code == candidate).limit(1)
        )
        if taken is None:
            return candidate
        logger.warning("[DOCTOR_CODE_COLLISION] retrying allocation")

    raise RuntimeError(
        "Could not allocate a unique Doctor ID after "
        f"{_MAX_ALLOCATION_ATTEMPTS} attempts."
    )


async def assign_doctor_code(db: AsyncSession, doctor) -> str:
    """
    Give this clinician a Doctor ID, surviving a concurrent allocation.

    `allocate_doctor_code` checks then writes, and between those two steps
    another approval can take the value. That loses a race roughly once in
    36**8, but when it does the unique index raises and the administrator sees
    a 500 on an approval that should simply have picked a different number.

    Each attempt writes inside a SAVEPOINT so a lost race rolls back only the
    failed assignment, leaving the surrounding transaction — and any other work
    the approval has already done — intact and usable.
    """
    from sqlalchemy.exc import IntegrityError

    for attempt in range(_MAX_ALLOCATION_ATTEMPTS):
        candidate = await allocate_doctor_code(db)
        try:
            async with db.begin_nested():
                doctor.doctor_code = candidate
                await db.flush()
            return candidate
        except IntegrityError:
            logger.warning(
                "[DOCTOR_CODE_RACE] lost a Doctor ID to a concurrent "
                "allocation, retrying (attempt %d)", attempt + 1,
            )

    raise RuntimeError(
        "Could not assign a unique Doctor ID after "
        f"{_MAX_ALLOCATION_ATTEMPTS} attempts."
    )
