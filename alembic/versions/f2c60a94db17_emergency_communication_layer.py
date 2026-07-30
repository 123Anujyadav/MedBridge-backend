"""Emergency communication layer.

Adds the durable record behind Phase 3: `communication_logs`, one row per
person per channel per emergency, written *before* the provider is called so a
crash or an outage cannot lose the fact that somebody was owed a call.

Also adds the fields the Maps integration fills in when a key is configured —
the reverse-geocoded address and the nearest hospital's coordinates and
distance. They are nullable and stay null while `GOOGLE_MAPS_API_KEY` is
absent, which is the deployed state today: the platform runs on the patient's
own registered address and the key-free map link until the key is added.

Purely additive. No existing column is altered, no constraint is relaxed, and
nothing has to be backfilled.

Revision ID: f2c60a94db17
Revises: e5b8d31c74af
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f2c60a94db17'
down_revision: Union[str, None] = 'e5b8d31c74af'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


CHANNELS = ("voice", "sms", "whatsapp")
STATUSES = ("queued", "sending", "sent", "delivered", "failed", "skipped")
ROLES = ("emergency_contact", "doctor", "admin")

CHANNEL_SQL = ", ".join(f"'{c}'" for c in CHANNELS)
STATUS_SQL = ", ".join(f"'{s}'" for s in STATUSES)
ROLE_SQL = ", ".join(f"'{r}'" for r in ROLES)


def upgrade() -> None:
    # ── Maps-backed fields on the emergency ──────────────────────────────
    op.add_column("emergency_requests",
                  sa.Column("resolved_address", sa.String(length=400), nullable=True))
    op.add_column("emergency_requests",
                  sa.Column("hospital_distance_km", sa.Float(), nullable=True))
    op.add_column("emergency_requests",
                  sa.Column("hospital_latitude", sa.Float(), nullable=True))
    op.add_column("emergency_requests",
                  sa.Column("hospital_longitude", sa.Float(), nullable=True))

    # ── the communication log ────────────────────────────────────────────
    op.create_table(
        "communication_logs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("emergency_id", sa.UUID(), nullable=False),

        sa.Column("channel", sa.String(length=20), nullable=False),
        sa.Column("recipient_role", sa.String(length=30), nullable=False),
        sa.Column("recipient_name", sa.String(length=200), nullable=True),
        sa.Column("recipient_phone", sa.String(length=32), nullable=True),

        sa.Column("status", sa.String(length=20), nullable=False,
                  server_default="queued"),
        sa.Column("provider", sa.String(length=30), nullable=True,
                  server_default="twilio"),
        sa.Column("provider_sid", sa.String(length=64), nullable=True),
        sa.Column("provider_status", sa.String(length=40), nullable=True),
        sa.Column("error_code", sa.String(length=40), nullable=True),
        sa.Column("error_message", sa.String(length=500), nullable=True),

        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),

        sa.Column("template_key", sa.String(length=60), nullable=True),
        sa.Column("body_preview", sa.Text(), nullable=True),

        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),

        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),

        sa.ForeignKeyConstraint(["emergency_id"], ["emergency_requests.id"],
                                ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(f"channel IN ({CHANNEL_SQL})",
                           name="communication_channel_check"),
        sa.CheckConstraint(f"status IN ({STATUS_SQL})",
                           name="communication_status_check"),
        sa.CheckConstraint(f"recipient_role IN ({ROLE_SQL})",
                           name="communication_recipient_role_check"),
        sa.CheckConstraint("attempts >= 0", name="communication_attempts_check"),
    )

    op.create_index(op.f("ix_communication_logs_id"), "communication_logs", ["id"])
    op.create_index("ix_communication_logs_emergency_id",
                    "communication_logs", ["emergency_id"])
    op.create_index("ix_communication_logs_provider_sid",
                    "communication_logs", ["provider_sid"])
    # The retry sweep's only query is "queued rows whose time has come", and it
    # runs every thirty seconds — this is the index it lives on.
    op.create_index("ix_communication_logs_due",
                    "communication_logs", ["status", "next_attempt_at"])


def downgrade() -> None:
    op.drop_index("ix_communication_logs_due", table_name="communication_logs")
    op.drop_index("ix_communication_logs_provider_sid",
                  table_name="communication_logs")
    op.drop_index("ix_communication_logs_emergency_id",
                  table_name="communication_logs")
    op.drop_index(op.f("ix_communication_logs_id"), table_name="communication_logs")
    op.drop_table("communication_logs")

    for column in ("hospital_longitude", "hospital_latitude",
                   "hospital_distance_km", "resolved_address"):
        op.drop_column("emergency_requests", column)
