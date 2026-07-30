"""SOS emergency workflow.

Phase 2 turns `emergency_requests` from a row someone wrote once into a record
with a lifecycle: who is handling it, what the emergency contact was at the
moment it was raised, and how it moved from `pending` to `resolved`.

Deliberately an extension, not a new table. A second emergency table would have
produced two answers to "is this patient in trouble right now", and two places
for a dashboard to look. The rows already here keep their meaning: their old
statuses stay valid, and every new column is nullable, so nothing has to be
backfilled and nothing already written becomes invalid.

`emergency_status_events` is new — the timeline. The current status stays on
`emergency_requests` because that is what the authorisation guards read on
every request; this table is the history behind it.

Revision ID: e5b8d31c74af
Revises: d4a17c2fb8e3
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e5b8d31c74af'
down_revision: Union[str, None] = 'd4a17c2fb8e3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


SOS_STATUSES = (
    "pending", "accepted", "doctor_assigned", "ambulance_dispatched",
    "hospital_reached", "resolved", "cancelled",
)
LEGACY_STATUSES = ("active", "dispatched", "arrived", "completed")
ALL_STATUSES = SOS_STATUSES + LEGACY_STATUSES

STATUS_SQL = ", ".join(f"'{s}'" for s in ALL_STATUSES)
OLD_STATUS_SQL = ", ".join(f"'{s}'" for s in LEGACY_STATUSES + ("cancelled",))


def upgrade() -> None:
    # ── new columns on the existing table ────────────────────────────────
    # All nullable: the rows already present were written before any of this
    # existed and there is no honest value to invent for them.
    op.add_column("emergency_requests",
                  sa.Column("assigned_doctor_id", sa.UUID(), nullable=True))
    op.add_column("emergency_requests",
                  sa.Column("assigned_doctor_name", sa.String(length=200), nullable=True))
    op.add_column("emergency_requests",
                  sa.Column("contact_name", sa.String(length=120), nullable=True))
    op.add_column("emergency_requests",
                  sa.Column("contact_phone", sa.String(length=20), nullable=True))
    op.add_column("emergency_requests",
                  sa.Column("contact_relationship", sa.String(length=60), nullable=True))
    op.add_column("emergency_requests",
                  sa.Column("maps_url", sa.String(length=255), nullable=True))
    op.add_column("emergency_requests",
                  sa.Column("created_by", sa.String(length=50), nullable=True,
                            server_default="patient"))
    op.add_column("emergency_requests",
                  sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("emergency_requests",
                  sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("emergency_requests",
                  sa.Column("cancel_reason", sa.String(length=255), nullable=True))

    op.create_foreign_key(
        "emergency_requests_assigned_doctor_id_fkey", "emergency_requests",
        "doctors", ["assigned_doctor_id"], ["id"], ondelete="SET NULL",
    )
    # Every doctor-facing query filters on this column.
    op.create_index("ix_emergency_requests_assigned_doctor_id",
                    "emergency_requests", ["assigned_doctor_id"])
    # The dashboards' "active emergencies" count and the duplicate-SOS check
    # both scan status; the patient lookup adds patient_id.
    op.create_index("ix_emergency_requests_patient_status",
                    "emergency_requests", ["patient_id", "status"])

    # ── widen the status vocabulary ──────────────────────────────────────
    # Backfilled first: an existing row must satisfy the constraint at the
    # moment it is applied, and the legacy values remain permitted anyway.
    op.drop_constraint("emergency_status_check", "emergency_requests", type_="check")
    op.create_check_constraint(
        "emergency_status_check", "emergency_requests",
        f"status IN ({STATUS_SQL})",
    )

    # A terminal state must carry its timestamp. Legacy rows used `completed`
    # and `cancelled` without one, so the two new constraints are written to
    # bite only on the new vocabulary — `completed` is untouched, and the
    # existing `cancelled` rows are backfilled from `updated_at` so the
    # constraint can be validated rather than left permanently NOT VALID.
    op.execute(
        "UPDATE emergency_requests SET cancelled_at = COALESCE(updated_at, created_at) "
        "WHERE status = 'cancelled' AND cancelled_at IS NULL"
    )
    op.create_check_constraint(
        "emergency_resolved_at_check", "emergency_requests",
        "status <> 'resolved' OR resolved_at IS NOT NULL",
    )
    op.create_check_constraint(
        "emergency_cancelled_at_check", "emergency_requests",
        "status <> 'cancelled' OR cancelled_at IS NOT NULL",
    )

    # ── the timeline ─────────────────────────────────────────────────────
    op.create_table(
        "emergency_status_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("emergency_id", sa.UUID(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("note", sa.String(length=500), nullable=True),
        sa.Column("actor_user_id", sa.UUID(), nullable=True),
        sa.Column("actor_role", sa.String(length=50), nullable=True),
        sa.Column("actor_name", sa.String(length=200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["emergency_id"], ["emergency_requests.id"],
                                ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["actor_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(f"status IN ({STATUS_SQL})",
                           name="emergency_event_status_check"),
    )
    op.create_index(op.f("ix_emergency_status_events_id"),
                    "emergency_status_events", ["id"])
    op.create_index("ix_emergency_status_events_emergency_id",
                    "emergency_status_events", ["emergency_id"])


def downgrade() -> None:
    op.drop_index("ix_emergency_status_events_emergency_id",
                  table_name="emergency_status_events")
    op.drop_index(op.f("ix_emergency_status_events_id"),
                  table_name="emergency_status_events")
    op.drop_table("emergency_status_events")

    op.drop_constraint("emergency_cancelled_at_check", "emergency_requests",
                       type_="check")
    op.drop_constraint("emergency_resolved_at_check", "emergency_requests",
                       type_="check")

    # Rows written under the SOS vocabulary cannot satisfy the old constraint,
    # so they are mapped back onto the nearest legacy meaning before it is
    # restored. Without this the downgrade fails on any real database.
    op.execute("UPDATE emergency_requests SET status = 'active' "
               "WHERE status IN ('pending', 'accepted', 'doctor_assigned')")
    op.execute("UPDATE emergency_requests SET status = 'dispatched' "
               "WHERE status = 'ambulance_dispatched'")
    op.execute("UPDATE emergency_requests SET status = 'arrived' "
               "WHERE status = 'hospital_reached'")
    op.execute("UPDATE emergency_requests SET status = 'completed' "
               "WHERE status = 'resolved'")

    op.drop_constraint("emergency_status_check", "emergency_requests", type_="check")
    op.create_check_constraint(
        "emergency_status_check", "emergency_requests",
        f"status IN ({OLD_STATUS_SQL})",
    )

    op.drop_index("ix_emergency_requests_patient_status",
                  table_name="emergency_requests")
    op.drop_index("ix_emergency_requests_assigned_doctor_id",
                  table_name="emergency_requests")
    op.drop_constraint("emergency_requests_assigned_doctor_id_fkey",
                       "emergency_requests", type_="foreignkey")

    for column in ("cancel_reason", "cancelled_at", "resolved_at", "created_by",
                   "maps_url", "contact_relationship", "contact_phone",
                   "contact_name", "assigned_doctor_name", "assigned_doctor_id"):
        op.drop_column("emergency_requests", column)
