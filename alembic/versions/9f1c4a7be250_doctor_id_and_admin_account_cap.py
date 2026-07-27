"""Doctor ID as a sign-in factor, and a hard ceiling on administrator accounts.

Two independent controls, one migration, because both are the database half of
the same authentication policy.

**`doctors.doctor_code`** — the clinician's 8-character Doctor ID, the third
factor in doctor sign-in. Every existing clinician is backfilled with one, so
the login rule ("the ID must be present and must match") has no population it
cannot apply to; a doctor row without an ID would otherwise be a row that could
be signed into with two factors. Unique index, and a format check so nothing
outside `[A-Z0-9]{8}` can be written by any route into the database.

**`users_admin_account_cap`** — a trigger refusing any insert or update that
would produce a third live administrator. The application already refuses this
in `app.services.admin_accounts`; the trigger is what holds when the write does
not come through the application. "Live" means active and not soft-deleted, so
retiring an administrator frees their slot and the constraint is satisfiable on
a database that already holds more administrators than the cap allows.

Both halves are reversible.

Revision ID: 9f1c4a7be250
Revises: 202728407fe0
"""
from typing import Sequence, Union

import secrets

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9f1c4a7be250'
down_revision: Union[str, None] = '202728407fe0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


DOCTOR_CODE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
DOCTOR_CODE_LENGTH = 8


def _generate_code() -> str:
    """
    Kept local rather than imported from `app.core.doctor_code`.

    A migration has to keep producing the same result years from now; importing
    application code would let a later refactor of that module silently change
    what this historical revision does.
    """
    return "".join(
        secrets.choice(DOCTOR_CODE_ALPHABET) for _ in range(DOCTOR_CODE_LENGTH)
    )


ADMIN_CAP_FUNCTION = """
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

ADMIN_CAP_TRIGGER = """
CREATE TRIGGER users_admin_account_cap_trigger
BEFORE INSERT OR UPDATE OF role, is_active, deleted_at ON users
FOR EACH ROW EXECUTE FUNCTION users_admin_account_cap();
"""


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    # ── Doctor ID ────────────────────────────────────────────────────────
    op.add_column(
        "doctors",
        sa.Column("doctor_code", sa.String(length=8), nullable=True),
    )

    # Backfill before the unique index exists, drawing fresh until the value is
    # free. The keyspace is 36**8, so this converges immediately; the loop is
    # here because "immediately" is not "certainly".
    doctors = bind.execute(
        sa.text("SELECT id FROM doctors WHERE doctor_code IS NULL")
    ).fetchall()

    used: set[str] = {
        row[0]
        for row in bind.execute(
            sa.text("SELECT doctor_code FROM doctors WHERE doctor_code IS NOT NULL")
        ).fetchall()
    }

    for (doctor_id,) in doctors:
        code = _generate_code()
        while code in used:
            code = _generate_code()
        used.add(code)
        bind.execute(
            sa.text("UPDATE doctors SET doctor_code = :code WHERE id = :id"),
            {"code": code, "id": doctor_id},
        )

    op.create_index(
        "ix_doctors_doctor_code", "doctors", ["doctor_code"], unique=True
    )

    if is_postgres:
        # Format is enforced at the column so no code path — application,
        # script or psql session — can store an ID the login comparison would
        # never match.
        op.create_check_constraint(
            "doctor_code_format_check",
            "doctors",
            "doctor_code IS NULL OR doctor_code ~ '^[A-Z0-9]{8}$'",
        )

        # ── Administrator account cap ────────────────────────────────────
        op.execute(ADMIN_CAP_FUNCTION)
        op.execute(ADMIN_CAP_TRIGGER)


def downgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    if is_postgres:
        op.execute(
            "DROP TRIGGER IF EXISTS users_admin_account_cap_trigger ON users"
        )
        op.execute("DROP FUNCTION IF EXISTS users_admin_account_cap()")
        op.drop_constraint("doctor_code_format_check", "doctors", type_="check")

    op.drop_index("ix_doctors_doctor_code", table_name="doctors")
    op.drop_column("doctors", "doctor_code")
