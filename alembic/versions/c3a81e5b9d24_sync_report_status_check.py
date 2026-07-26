"""sync report_status_check with the model's status vocabulary

The `reports` table was originally created by `Base.metadata.create_all()` from
an older model, then reconciled with `alembic stamp` — which records a revision
without running its DDL. The stamp left the live CHECK constraint permitting
only ('pending', 'ready', 'reviewed', 'shared') while the model, the API and the
frontend had all moved on to eight statuses.

Alembic's autogenerate does not diff CHECK constraints, so `alembic check`
reported the schema as clean throughout.

The practical effect was that `PUT /doctor/reports/{id}/status` — whose own
regex accepts pending_review, approved, rejected and needs_revision — raised a
CheckViolationError for four of its six legal values. The Reject Report button
in the doctor portal was among them. Tests did not catch it because the SQLite
test database is built from the current model on every run.

Widening only: the new set is a superset of the old, so no existing row can
violate it and no data migration is needed.

Revision ID: c3a81e5b9d24
Revises: b7f2c9d41a58
Create Date: 2026-07-26 12:31:44.905118

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c3a81e5b9d24'
down_revision: Union[str, None] = 'b7f2c9d41a58'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CONSTRAINT = "report_status_check"

MODEL_STATUSES = (
    "pending", "ready", "reviewed", "shared",
    "pending_review", "approved", "rejected", "needs_revision",
)

ORIGINAL_STATUSES = ("pending", "ready", "reviewed", "shared")


def _condition(statuses: Sequence[str]) -> str:
    values = ", ".join(f"'{s}'" for s in statuses)
    return f"status IN ({values})"


def upgrade() -> None:
    # IF EXISTS: the constraint is absent on a database built from the
    # migrations alone, where the baseline already created the correct one.
    op.execute(f"ALTER TABLE reports DROP CONSTRAINT IF EXISTS {CONSTRAINT}")
    op.create_check_constraint(CONSTRAINT, "reports", _condition(MODEL_STATUSES))


def downgrade() -> None:
    # Narrowing back would fail on any row holding one of the four newer
    # statuses, so those are reset to the closest original value first.
    op.execute(
        "UPDATE reports SET status = 'reviewed' "
        "WHERE status IN ('pending_review', 'approved')"
    )
    op.execute(
        "UPDATE reports SET status = 'pending' "
        "WHERE status IN ('rejected', 'needs_revision')"
    )
    op.execute(f"ALTER TABLE reports DROP CONSTRAINT IF EXISTS {CONSTRAINT}")
    op.create_check_constraint(CONSTRAINT, "reports", _condition(ORIGINAL_STATUSES))
