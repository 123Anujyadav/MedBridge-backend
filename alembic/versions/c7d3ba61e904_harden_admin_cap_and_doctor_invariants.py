"""Close the administrator-cap race, and require a Doctor ID to be verified.

Three changes, all from a production audit that reproduced each defect.

**The administrator cap was not actually a constraint.** The trigger added by
`9f1c4a7be250` counted live administrators with a plain `SELECT COUNT(*)`. Under
READ COMMITTED two concurrent transactions each saw the pre-insert count, both
passed, and both committed — reproducibly producing a third administrator. The
same hole let two concurrent *reactivations* through. It is replaced here with
the same check behind `pg_advisory_xact_lock`, which serialises every writer
that could produce a live administrator and is released automatically on commit
or rollback. One fixed lock key, so there is no lock-ordering deadlock, and it
is taken only on the rare administrator path — ordinary user writes never touch
it.

**A doctor could be `verified` with no Doctor ID.** Nothing tied the two
together, so a direct write could produce a clinician who is approved but can
never sign in (the Doctor ID is a required sign-in factor). Added as a table
constraint, `NOT VALID` first and then validated, so the existing rows are
checked without holding an ACCESS EXCLUSIVE lock for the scan.

**The verification queue had no usable index.** It filters on
`verification_status` and orders by `created_at DESC`; with pagination now
issuing a COUNT as well, that is two sequential scans per page view.

Revision ID: c7d3ba61e904
Revises: 9f1c4a7be250
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c7d3ba61e904'
down_revision: Union[str, None] = '9f1c4a7be250'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


ADMIN_CAP_LOCK_KEY = 7749283016112233445
"""
The advisory lock every administrator-producing write serialises on.

An arbitrary but *fixed* 64-bit constant. It must never change: two releases
using different keys would not exclude each other, which is exactly the bug
this migration exists to close. The same value is mirrored in
`app.services.admin_accounts.ADMIN_CAP_LOCK_KEY` so the application guard
queues behind the same lock.
"""

ADMIN_CAP_FUNCTION_V2 = f"""
CREATE OR REPLACE FUNCTION users_admin_account_cap()
RETURNS TRIGGER AS $$
DECLARE
    live_admins INTEGER;
BEGIN
    -- Only rows that would become a live administrator are of interest. A
    -- deactivated or soft-deleted administrator holds no slot, which is what
    -- lets an over-capacity database be brought back under the cap by
    -- retiring accounts instead of deleting them.
    IF NEW.role <> 'admin' OR NEW.is_active IS NOT TRUE
       OR NEW.deleted_at IS NOT NULL THEN
        RETURN NEW;
    END IF;

    -- An update that leaves an already-live administrator live changes nothing
    -- about the count, and must not be refused.
    IF TG_OP = 'UPDATE' AND OLD.role = 'admin' AND OLD.is_active IS TRUE
       AND OLD.deleted_at IS NULL THEN
        RETURN NEW;
    END IF;

    -- Serialise every writer that could add a live administrator. Without this
    -- the COUNT below is a time-of-check/time-of-use read: concurrent
    -- transactions each see the old count and each conclude there is room.
    -- Transaction-scoped, so it is released on COMMIT or ROLLBACK with no
    -- explicit unlock and no leak if the statement raises.
    PERFORM pg_advisory_xact_lock({ADMIN_CAP_LOCK_KEY});

    -- Read *after* the lock. Under READ COMMITTED a volatile function takes a
    -- fresh snapshot per statement, so this sees whatever the transaction we
    -- just queued behind committed.
    SELECT COUNT(*) INTO live_admins
    FROM users
    WHERE role = 'admin' AND is_active IS TRUE AND deleted_at IS NULL
      AND id <> NEW.id;

    IF live_admins >= 2 THEN
        RAISE EXCEPTION
            'administrator account limit reached: this platform allows a '
            'maximum of 2 active administrator accounts'
            USING ERRCODE = 'check_violation';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""

ADMIN_CAP_FUNCTION_V1 = """
CREATE OR REPLACE FUNCTION users_admin_account_cap()
RETURNS TRIGGER AS $$
DECLARE
    live_admins INTEGER;
BEGIN
    IF NEW.role <> 'admin' OR NEW.is_active IS NOT TRUE
       OR NEW.deleted_at IS NOT NULL THEN
        RETURN NEW;
    END IF;

    IF TG_OP = 'UPDATE' AND OLD.role = 'admin' AND OLD.is_active IS TRUE
       AND OLD.deleted_at IS NULL THEN
        RETURN NEW;
    END IF;

    SELECT COUNT(*) INTO live_admins
    FROM users
    WHERE role = 'admin' AND is_active IS TRUE AND deleted_at IS NULL
      AND id <> NEW.id;

    IF live_admins >= 2 THEN
        RAISE EXCEPTION
            'administrator account limit reached: this platform allows a '
            'maximum of 2 active administrator accounts'
            USING ERRCODE = 'check_violation';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""

VERIFIED_REQUIRES_CODE = (
    "verification_status <> 'verified' OR doctor_code IS NOT NULL"
)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # The trigger and the partial index are PostgreSQL-specific; SQLite is
        # only used by the test harness, which builds its schema from the
        # models rather than from this file.
        return

    # ── C-1: make the cap a real constraint ──────────────────────────────
    # Replacing the function is enough — the trigger already points at it, so
    # no trigger is dropped or recreated and no table is locked.
    op.execute(ADMIN_CAP_FUNCTION_V2)

    # ── M-3: an approved clinician must hold a Doctor ID ─────────────────
    # NOT VALID first so adding the constraint does not scan the table under an
    # ACCESS EXCLUSIVE lock; VALIDATE then checks the existing rows under a
    # SHARE UPDATE EXCLUSIVE lock, which does not block reads or writes.
    op.execute(
        "ALTER TABLE doctors ADD CONSTRAINT doctor_verified_requires_code "
        f"CHECK ({VERIFIED_REQUIRES_CODE}) NOT VALID"
    )
    op.execute(
        "ALTER TABLE doctors VALIDATE CONSTRAINT doctor_verified_requires_code"
    )

    # ── M-2: the verification queue's access path ────────────────────────
    # Matches `WHERE verification_status = ? ORDER BY created_at DESC` and the
    # COUNT that paging now issues alongside it.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_doctors_verification_status_created_at "
        "ON doctors (verification_status, created_at DESC)"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("DROP INDEX IF EXISTS ix_doctors_verification_status_created_at")
    op.execute(
        "ALTER TABLE doctors DROP CONSTRAINT IF EXISTS "
        "doctor_verified_requires_code"
    )
    # Back to the unlocked count this revision replaced.
    op.execute(ADMIN_CAP_FUNCTION_V1)
