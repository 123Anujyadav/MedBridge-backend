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
   approved and holding **no Doctor ID**. A doctor account must not become
   usable without an administrator deciding so, and this script is not an
   administrator. The Doctor ID is issued by the approval itself.

An address that already exists under another role is *promoted in place*
rather than deleted and recreated, so its id — and every audit-log row, case
and appointment referencing it — survives.

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
from app.core.security import get_password_hash
import app.db.base  # noqa: F401  — registers every model so relationships resolve
from app.models.doctor import Doctor
from app.models.user import User
from app.services.admin_accounts import (
    MAX_ADMIN_ACCOUNTS,
    assert_admin_slot_available,
    count_admin_accounts,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("bootstrap")


DEFAULT_ADMIN_EMAIL = os.getenv("BOOTSTRAP_ADMIN_EMAIL", "anujkum0989@gmail.com")
DEFAULT_ADMIN_PASSWORD = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "admin123")

DEFAULT_DOCTOR_EMAIL = os.getenv(
    "BOOTSTRAP_DOCTOR_EMAIL", "9473sandhyadevi@gmail.com"
)
DEFAULT_DOCTOR_PASSWORD = os.getenv("BOOTSTRAP_DOCTOR_PASSWORD", "doctor123")

ADMINS_TO_KEEP = (DEFAULT_ADMIN_EMAIL, "admin@aronofy.com")
"""
The two accounts that hold the platform's administrator slots.

Anything else with the admin role is retired by step 1. Listed by address
rather than discovered by age so the outcome of running this script is
predictable rather than dependent on what happens to be oldest.
"""


async def _sync_identity(db, user: User, password: str) -> None:
    """
    Make the account's Supabase identity real, confirmed, and correctly linked.

    An account can arrive here in three states, and all three occur in the
    production database this script is written for:

    * no `supabase_user_id` at all — never signed in through the provider;
    * an id that the provider does not recognise, left behind by a project
      migration or a restored database. The local row then points at nothing,
      and sign-in fails with a confusing "incorrect password";
    * already correct, in which case only the password is reset.

    In each case the address is looked up at the provider, created if absent,
    and the local link rewritten to whatever the provider actually holds. That
    is what "Supabase and PostgreSQL stay synchronized" has to mean in
    practice — not that they were synchronized once, but that a re-run repairs
    them if they have drifted.
    """
    if settings.AUTH_PROVIDER != "supabase":
        return

    from app.core.supabase import SupabaseAuthError, get_supabase_auth_client

    client = get_supabase_auth_client()
    existing = await client.admin_get_user_by_email(user.email)

    if existing and existing.get("id"):
        identity_id = existing["id"]
        try:
            await client.admin_update_user(
                identity_id, password=password, email_confirm=True
            )
        except SupabaseAuthError as exc:
            logger.warning("  could not reset provider password: %s", exc.message)
    else:
        created = await client.admin_create_user(
            email=user.email, password=password,
            email_confirm=True,  # no mailbox is watching a bootstrap account
            user_metadata={"role": user.role, "bootstrap": True},
        )
        identity_id = created.get("id")
        logger.info("  created the Supabase identity")

    if identity_id and user.supabase_user_id != identity_id:
        if user.supabase_user_id:
            logger.info(
                "  relinked a stale supabase_user_id (%s -> %s)",
                user.supabase_user_id, identity_id,
            )
        user.supabase_user_id = identity_id
        await db.flush()


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
    """
    Install the designated administrator.

    Creates the account if the address is new, and *promotes it in place* if the
    address already exists under another role. Promoting rather than deleting
    and recreating is what keeps the account's id stable, and every audit-log
    row, case and appointment that references it intact.
    """
    logger.info("Step 2: administrator %s", DEFAULT_ADMIN_EMAIL)

    user = await db.scalar(select(User).where(User.email == DEFAULT_ADMIN_EMAIL))

    if user is not None and user.role == "admin" and user.is_active:
        logger.info("  already an active administrator")
        if not dry_run:
            await _sync_identity(db, user, DEFAULT_ADMIN_PASSWORD)
        return user

    # Creating *or* promoting consumes a slot, so both are capped. The check
    # takes the same advisory lock the database trigger uses, so a concurrent
    # run cannot slip a third administrator past it.
    in_use = await count_admin_accounts(db)
    if in_use >= MAX_ADMIN_ACCOUNTS:
        logger.error(
            "  refused: %d of %d administrator slots in use. Retire one first.",
            in_use, MAX_ADMIN_ACCOUNTS,
        )
        return None

    if dry_run:
        action = "promote" if user is not None else "create"
        logger.info("  would %s (slot %d of %d)", action, in_use + 1,
                    MAX_ADMIN_ACCOUNTS)
        return None

    await assert_admin_slot_available(db)

    if user is None:
        user = User(
            email=DEFAULT_ADMIN_EMAIL,
            hashed_password=get_password_hash(DEFAULT_ADMIN_PASSWORD),
            role="admin", is_active=True, is_verified=True,
        )
        db.add(user)
        await db.flush()
        logger.info("  created (slot %d of %d)", in_use + 1, MAX_ADMIN_ACCOUNTS)
    else:
        previous = user.role
        user.role = "admin"
        user.is_active = True
        user.is_verified = True
        user.hashed_password = get_password_hash(DEFAULT_ADMIN_PASSWORD)
        await db.flush()
        logger.info("  promoted %s -> admin (slot %d of %d)",
                    previous, in_use + 1, MAX_ADMIN_ACCOUNTS)

    await _sync_identity(db, user, DEFAULT_ADMIN_PASSWORD)
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

    user = await db.scalar(select(User).where(User.email == DEFAULT_DOCTOR_EMAIL))
    doctor = await db.get(Doctor, user.id) if user else None

    if doctor is not None:
        logger.info("  clinical profile already present — status %s, Doctor ID %s",
                    doctor.verification_status, doctor.doctor_code or "not issued")
        if not dry_run:
            if user.role != "doctor":
                logger.info("  promoted %s -> doctor", user.role)
                user.role = "doctor"
            user.is_active = True
            user.hashed_password = get_password_hash(DEFAULT_DOCTOR_PASSWORD)
            await db.flush()
            await _sync_identity(db, user, DEFAULT_DOCTOR_PASSWORD)
        return doctor

    if dry_run:
        what = "promote and add a clinical profile" if user else "create"
        logger.info("  would %s (pending approval, no Doctor ID)", what)
        return None

    if user is None:
        user = User(
            email=DEFAULT_DOCTOR_EMAIL,
            hashed_password=get_password_hash(DEFAULT_DOCTOR_PASSWORD),
            role="doctor", is_active=True, is_verified=True,
        )
        db.add(user)
        await db.flush()
        logger.info("  created the account")
    else:
        previous = user.role
        user.role = "doctor"
        user.is_active = True
        user.is_verified = True
        user.hashed_password = get_password_hash(DEFAULT_DOCTOR_PASSWORD)
        await db.flush()
        logger.info("  promoted %s -> doctor", previous)

    doctor = Doctor(
        id=user.id,
        # No Doctor ID. It is issued by an administrator's approval and by
        # nothing else, so a clinician awaiting review has none to sign in with.
        first_name="Sandhya",
        last_name="Devi",
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
    await _sync_identity(db, user, DEFAULT_DOCTOR_PASSWORD)

    logger.info("  created — status pending, no Doctor ID until approved")
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
