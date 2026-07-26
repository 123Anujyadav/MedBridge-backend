"""extend notifications for the real-time notification centre

Adds the clinical context a notification card needs to link into a workflow
(case, patient), the fields the centre filters and groups by (category,
group_key, archived), and `dedupe_key`, which is what prevents one event
producing the same card twice when a request is retried.

`priority` gains 'critical' alongside the existing 'urgent'. Widening only, so
no existing row can violate it.

Revision ID: a1d63f92c704
Revises: f4c19d8b30ae
Create Date: 2026-07-26 16:41:20.553108

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1d63f92c704'
down_revision: Union[str, None] = 'f4c19d8b30ae'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PRIORITY = "notification_priority_check"
CATEGORY = "notification_category_check"

WITH_CRITICAL = "priority IN ('low', 'medium', 'high', 'urgent', 'critical')"
WITHOUT_CRITICAL = "priority IN ('low', 'medium', 'high', 'urgent')"

CATEGORIES = (
    "category IN ('case', 'ai', 'appointment', 'report', 'prescription', "
    "'patient', 'system', 'security', 'general')"
)


def upgrade() -> None:
    op.add_column("notifications", sa.Column("case_id", sa.UUID(), nullable=True))
    op.add_column("notifications", sa.Column("patient_id", sa.UUID(), nullable=True))
    op.add_column("notifications", sa.Column("patient_name", sa.String(length=200), nullable=True))
    op.add_column("notifications", sa.Column("group_key", sa.String(length=80), nullable=True))
    op.add_column("notifications", sa.Column("dedupe_key", sa.String(length=160), nullable=True))
    op.add_column("notifications", sa.Column("delivered_at", sa.String(length=50), nullable=True))
    op.add_column("notifications", sa.Column("read_at", sa.String(length=50), nullable=True))
    op.add_column(
        "notifications",
        sa.Column("category", sa.String(length=20), nullable=False,
                  server_default="general"),
    )
    op.alter_column("notifications", "category", server_default=None)
    op.add_column(
        "notifications",
        sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.alter_column("notifications", "archived", server_default=None)

    op.create_index(op.f("ix_notifications_case_id"), "notifications", ["case_id"])
    op.create_index(op.f("ix_notifications_patient_id"), "notifications", ["patient_id"])
    op.create_index(op.f("ix_notifications_category"), "notifications", ["category"])
    op.create_index(op.f("ix_notifications_archived"), "notifications", ["archived"])
    op.create_index(op.f("ix_notifications_group_key"), "notifications", ["group_key"])
    op.create_index(op.f("ix_notifications_dedupe_key"), "notifications", ["dedupe_key"])
    op.create_index("idx_notification_user_read", "notifications",
                    ["user_id", "read", "created_at"])

    op.create_foreign_key("fk_notifications_case_id_cases", "notifications", "cases",
                          ["case_id"], ["id"], ondelete="SET NULL")
    op.create_foreign_key("fk_notifications_patient_id_patients", "notifications",
                          "patients", ["patient_id"], ["id"], ondelete="SET NULL")

    op.execute(f"ALTER TABLE notifications DROP CONSTRAINT IF EXISTS {PRIORITY}")
    op.create_check_constraint(PRIORITY, "notifications", WITH_CRITICAL)
    op.create_check_constraint(CATEGORY, "notifications", CATEGORIES)


def downgrade() -> None:
    # Narrowing would fail on any critical row, so those return to 'urgent'.
    op.execute("UPDATE notifications SET priority = 'urgent' WHERE priority = 'critical'")
    op.drop_constraint(CATEGORY, "notifications", type_="check")
    op.execute(f"ALTER TABLE notifications DROP CONSTRAINT IF EXISTS {PRIORITY}")
    op.create_check_constraint(PRIORITY, "notifications", WITHOUT_CRITICAL)

    op.drop_constraint("fk_notifications_patient_id_patients", "notifications",
                       type_="foreignkey")
    op.drop_constraint("fk_notifications_case_id_cases", "notifications",
                       type_="foreignkey")

    op.drop_index("idx_notification_user_read", table_name="notifications")
    for name in ("dedupe_key", "group_key", "archived", "category",
                 "patient_id", "case_id"):
        op.drop_index(op.f(f"ix_notifications_{name}"), table_name="notifications")

    for column in ("archived", "category", "read_at", "delivered_at", "dedupe_key",
                   "group_key", "patient_name", "patient_id", "case_id"):
        op.drop_column("notifications", column)
