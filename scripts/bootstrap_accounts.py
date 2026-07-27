"""
Install the platform's designated administrator and clinician accounts.

Idempotent: every step checks before it writes, so running this twice is the
same as running it once. That matters because it is meant to be run on a
database that is already in service, not only on an empty one.

    python -m scripts.bootstrap_accounts            # apply
    python -m scripts.bootstrap_accounts --dry-run  # report, change nothing

Three things happen, in this order:

1. **Surplus administrators are retired.** The platform allows two live
   administrator accounts. A database that predates that rule can hold more, and
   the cap cannot be enforced until it does not. Retiring means `is_active` off
   and `deleted_at` set — the row, its id and everything the audit log says
   about it are kept, because deleting an administrator would orphan the trail
   of what they approved. Reversible by clearing those two fields, subject to
   the cap.

2. **The designated administrator is created**, if absent.

3. **The designated clinician is created**, if absent — `pending`, never
   approved. A doctor account must not become usable without an administrator
   deciding so, and this script is not an administrator. Their Doctor ID is
   allocated at creation so it exists to be handed over once approved.

Under `AUTH_PROVIDER=supabase` the identity is created at the provider with its
address pre-confirmed, since a bootstrap account has no one to click a link in
its mailbox. The local password hash is written either way, so the accounts
still work if the provider is switched back to `local`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import select

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.doctor_code import allocate_doctor_code, is_valid_doctor_code
from app.core.security import get_password_hash
import app.db.base  # noqa: F401  — registers every model so relationships resolve
from app.models.doctor import Doctor
from app.models.user import User
from app.services.admin_accounts import MAX_ADMIN_ACCOUNTS, count_admin_accounts

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("bootstrap")


DEFAULT_ADMIN_EMAIL = "anujkum0989@gmail.com"
DEFAULT_ADMIN_PASSWORD = "admin123"

DEFAULT_DOCTOR_EMAIL = "anujkumaryadav0989@gmail.com"
DEFAULT_DOCTOR_PASSWORD = "doctor123"

ADMINS_TO_KEEP = (DEFAULT_ADMIN_EMAIL, "admin@aronofy.com")
"""
The two accounts that hold the platform's administrator slots.

Anything else with the admin role is retired by step 1. Listed by address
rather than discovered by age so the outcome of running this script is
predictable rather than dependent on what happens to be oldest.
"""


# ── identity provider ────────────────────────────────────────────────────

async def _ensure_identity(email: str, password: str, role: str) -> str | None:
    """
    Make sure Supabase holds a confirmed identity for this address.

    Returns its id, or None when the built-in provider is active. An address
    that already has an identity has its password reset to the designated one,
    so a half-finished earlier run cannot leave an account nobody can sign in to.
    """
    if settings.AUTH_PROVIDER != "supabase":
        return None

    from app.core.supabase import SupabaseAuthError, get_supabase_auth_client

    client = get_supabase_auth_client()
    existing = await client.admin_get_user_by_email(email)
    if existing and existing.get("id"):
        try:
            await client.admin_update_user(
                existing["id"], password=password, email_confirm=True
            )
            logger.info("  supabase identity already present, password reset")
        except SupabaseAuthError as exc:
            logger.warning("  could not update supabase identity: %s", exc.message)
        return existing["id"]

    created = await client.admin_create_user(
        email=email, password=password,
        email_confirm=True,  # no mailbox is watching a bootstrap account
        user_metadata={"role": role, "bootstrap": True},
    )
    logger.info("  supabase identity created")
    return created.get("id")


# ── step 1: bring the administrator count under the cap ──────────────────

async def retire_surplus_admins(db, dry_run: bool) -> int:
    """Deactivate and soft-delete every administrator outside the keep list."""
    from datetime import datetime, timezone

    result = await db.execute(
        select(User).where(
            User.role == "admin",
            User.is_active.is_(True),
            User.deleted_at.is_(None),
        )
    )
    live = list(result.scalars().all())
    surplus = [u for u in live if u.email.lower() not in
               {e.lower() for e in ADMINS_TO_KEEP}]

    logger.info("Step 1: %d live administrator account(s); %d to retire",
                len(live), len(surplus))

    for user in surplus:
        logger.info("  retiring %s", user.email)
        if not dry_run:
            user.is_active = False
            user.deleted_at = datetime.now(timezone.utc)

    if not dry_run and surplus:
        await db.flush()
    return len(surplus)


# ── step 2: the designated administrator ─────────────────────────────────

async def ensure_default_admin(db, dry_run: bool) -> User | None:
    """Create the designated administrator, unless the address already exists."""
    logger.info("Step 2: administrator %s", DEFAULT_ADMIN_EMAIL)

    existing = await db.scalar(
        select(User).where(User.email == DEFAULT_ADMIN_EMAIL)
    )
    if existing:
        logger.info("  already exists (role=%s, active=%s) — left untouched",
                    existing.role, existing.is_active)
        return existing

    in_use = await count_admin_accounts(db)
    if in_use >= MAX_ADMIN_ACCOUNTS:
        logger.error(
            "  refused: %d of %d administrator slots in use. Retire one first.",
            in_use, MAX_ADMIN_ACCOUNTS,
        )
        return None

    if dry_run:
        logger.info("  would create (slot %d of %d)", in_use + 1, MAX_ADMIN_ACCOUNTS)
        return None

    supabase_user_id = await _ensure_identity(
        DEFAULT_ADMIN_EMAIL, DEFAULT_ADMIN_PASSWORD, "admin"
    )
    user = User(
        email=DEFAULT_ADMIN_EMAIL,
        hashed_password=get_password_hash(DEFAULT_ADMIN_PASSWORD),
        role="admin",
        is_active=True,
        is_verified=True,
        supabase_user_id=supabase_user_id,
    )
    db.add(user)
    await db.flush()
    logger.info("  created (slot %d of %d)", in_use + 1, MAX_ADMIN_ACCOUNTS)
    return user


# ── step 3: the designated clinician ─────────────────────────────────────

async def ensure_default_doctor(db, dry_run: bool) -> Doctor | None:
    """
    Create the designated clinician — unapproved.

    Left `pending` on purpose. The whole point of the verification module is
    that no doctor becomes able to sign in without an administrator saying so,
    and a seed script that approved its own account would be the first bypass.
    """
    logger.info("Step 3: clinician %s", DEFAULT_DOCTOR_EMAIL)

    existing = await db.scalar(
        select(User).where(User.email == DEFAULT_DOCTOR_EMAIL)
    )
    if existing:
        doctor = await db.get(Doctor, existing.id)
        if doctor and not is_valid_doctor_code(doctor.doctor_code):
            if dry_run:
                logger.info("  exists without a Doctor ID — would allocate one")
            else:
                doctor.doctor_code = await allocate_doctor_code(db)
                await db.flush()
                logger.info("  allocated missing Doctor ID: %s", doctor.doctor_code)
        elif doctor:
            logger.info("  already exists — Doctor ID %s, status %s",
                        doctor.doctor_code, doctor.verification_status)
        return doctor

    if dry_run:
        logger.info("  would create (pending approval)")
        return None

    supabase_user_id = await _ensure_identity(
        DEFAULT_DOCTOR_EMAIL, DEFAULT_DOCTOR_PASSWORD, "doctor"
    )
    user = User(
        email=DEFAULT_DOCTOR_EMAIL,
        hashed_password=get_password_hash(DEFAULT_DOCTOR_PASSWORD),
        role="doctor",
        is_active=True,
        is_verified=True,
        supabase_user_id=supabase_user_id,
    )
    db.add(user)
    await db.flush()

    doctor = Doctor(
        id=user.id,
        doctor_code=await allocate_doctor_code(db),
        first_name="Anuj Kumar",
        last_name="Yadav",
        phone="+91 90000 00000",
        specialty="General Medicine",
        sub_specialties=["Internal Medicine"],
        hospital_name="MedBridge Central",
        license_number=f"MB-{user.id.hex[:10].upper()}",
        years_of_experience=5,
        education=["MBBS"],
        certifications=[],
        languages=["English", "Hindi"],
        availability="available",
        consultation_fee=0.0,
        verification_status="pending",  # an administrator decides, not this script
        bio="Clinician account awaiting administrator verification.",
    )
    db.add(doctor)
    await db.flush()

    logger.info("  created — Doctor ID %s, status pending", doctor.doctor_code)
    logger.info("  an administrator must approve this account before it can sign in")
    return doctor


# ── entry point ──────────────────────────────────────────────────────────

async def main(dry_run: bool) -> int:
    logger.info("Identity provider: %s", settings.AUTH_PROVIDER)
    if dry_run:
        logger.info("DRY RUN — no changes will be written\n")

    async with AsyncSessionLocal() as db:
        await retire_surplus_admins(db, dry_run)
        await ensure_default_admin(db, dry_run)
        doctor = await ensure_default_doctor(db, dry_run)

        if dry_run:
            await db.rollback()
        else:
            await db.commit()

        logger.info("\nAdministrator slots in use: %d of %d",
                    await count_admin_accounts(db), MAX_ADMIN_ACCOUNTS)
        if doctor and doctor.doctor_code:
            logger.info("Designated clinician Doctor ID: %s", doctor.doctor_code)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change, write nothing")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.dry_run)))
