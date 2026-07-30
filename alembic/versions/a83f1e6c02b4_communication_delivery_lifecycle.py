"""Full delivery lifecycle for emergency communications.

Creating a Twilio message returns `queued` or `accepted` — an acknowledgement
that the provider took the request, not evidence that anybody received
anything. Everything after that arrives asynchronously as a status callback,
and until now there was nowhere to put it.

Two changes:

* The status vocabulary gains `accepted`, `undelivered` and `canceled`.
  `accepted` is kept distinct from `sent` on purpose: an emergency dashboard
  that showed a provider acknowledgement as a delivered message would tell a
  patient somebody had been reached when nobody had. `undelivered` (a carrier
  refused it) is likewise distinct from `failed` (we gave up), because they need
  different action.
* `provider_events` records every callback, in order, appended and never
  replaced. `status` says where an attempt got to; this says how.

Additive: existing rows keep their status, and `provider_events` defaults to an
empty list.

Revision ID: a83f1e6c02b4
Revises: f2c60a94db17
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a83f1e6c02b4'
down_revision: Union[str, None] = 'f2c60a94db17'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


OLD_STATUSES = ("queued", "sending", "sent", "delivered", "failed", "skipped")
NEW_STATUSES = OLD_STATUSES + ("accepted", "undelivered", "canceled")

OLD_SQL = ", ".join(f"'{s}'" for s in OLD_STATUSES)
NEW_SQL = ", ".join(f"'{s}'" for s in NEW_STATUSES)


def upgrade() -> None:
    op.add_column(
        "communication_logs",
        sa.Column("provider_events", sa.JSON(), nullable=False,
                  server_default=sa.text("'[]'::json")),
    )

    op.drop_constraint("communication_status_check", "communication_logs",
                       type_="check")
    op.create_check_constraint(
        "communication_status_check", "communication_logs",
        f"status IN ({NEW_SQL})",
    )


def downgrade() -> None:
    # Rows written under the new vocabulary cannot satisfy the old constraint,
    # so they are mapped onto the nearest older meaning before it is restored.
    # Without this the downgrade fails on any database that has taken a
    # callback.
    op.execute("UPDATE communication_logs SET status = 'sent' "
               "WHERE status = 'accepted'")
    op.execute("UPDATE communication_logs SET status = 'failed' "
               "WHERE status IN ('undelivered', 'canceled')")

    op.drop_constraint("communication_status_check", "communication_logs",
                       type_="check")
    op.create_check_constraint(
        "communication_status_check", "communication_logs",
        f"status IN ({OLD_SQL})",
    )

    op.drop_column("communication_logs", "provider_events")
