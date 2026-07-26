"""add follow-up flag and 'archived' status for bulk report actions

Two additions backing the AI Reports bulk-action workflow:

* `reports.flagged_for_follow_up` — a clinician marker orthogonal to `status`.
  A report can be approved *and* require follow-up, so folding this into the
  status vocabulary would lose one of the two facts.
* `'archived'` added to `report_status_check`, so reports can be retired from
  the working list without being deleted.

Widening only for the constraint: the new set is a superset of the old, so no
existing row can violate it.

Revision ID: d5f27a1c8e60
Revises: c3a81e5b9d24
Create Date: 2026-07-26 13:02:55.417220

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5f27a1c8e60'
down_revision: Union[str, None] = 'c3a81e5b9d24'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CONSTRAINT = "report_status_check"

WITH_ARCHIVED = (
    "pending", "ready", "reviewed", "shared",
    "pending_review", "approved", "rejected", "needs_revision", "archived",
)

WITHOUT_ARCHIVED = (
    "pending", "ready", "reviewed", "shared",
    "pending_review", "approved", "rejected", "needs_revision",
)


def _condition(statuses: Sequence[str]) -> str:
    return f"status IN ({', '.join(chr(39) + s + chr(39) for s in statuses)})"


def upgrade() -> None:
    op.add_column(
        "reports",
        sa.Column(
            "flagged_for_follow_up",
            sa.Boolean(),
            nullable=False,
            # Existing rows are not flagged; the server_default is dropped
            # afterwards so the application remains the source of the default.
            server_default=sa.false(),
        ),
    )
    op.alter_column("reports", "flagged_for_follow_up", server_default=None)
    op.create_index(
        op.f("ix_reports_flagged_for_follow_up"),
        "reports",
        ["flagged_for_follow_up"],
        unique=False,
    )

    op.execute(f"ALTER TABLE reports DROP CONSTRAINT IF EXISTS {CONSTRAINT}")
    op.create_check_constraint(CONSTRAINT, "reports", _condition(WITH_ARCHIVED))


def downgrade() -> None:
    # Narrowing would fail on any archived row, so those are returned to the
    # closest surviving status first.
    op.execute("UPDATE reports SET status = 'shared' WHERE status = 'archived'")
    op.execute(f"ALTER TABLE reports DROP CONSTRAINT IF EXISTS {CONSTRAINT}")
    op.create_check_constraint(CONSTRAINT, "reports", _condition(WITHOUT_ARCHIVED))

    op.drop_index(op.f("ix_reports_flagged_for_follow_up"), table_name="reports")
    op.drop_column("reports", "flagged_for_follow_up")
