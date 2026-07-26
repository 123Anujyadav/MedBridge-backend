"""extend audit_logs into the case timeline event store, and make it append-only

Two changes:

1. Clinical dimensions on `audit_logs` — case, patient, actor type, semantic
   event type, and the field/previous/new triple a change history needs. These
   live here rather than in a second history table so there is one record of
   what happened, not two that can disagree.

2. Append-only enforcement at the database. A trigger rejects UPDATE and DELETE
   outright, so an audit entry cannot be rewritten or removed even by code
   holding a live session. Application-level discipline is not sufficient for a
   compliance record: the guarantee has to survive a bug or a console.

The trigger is PostgreSQL-only and is created defensively so a non-Postgres
target (the SQLite test database is built from models, not migrations) is
unaffected.

Revision ID: e8b3f0c7a92d
Revises: d5f27a1c8e60
Create Date: 2026-07-26 14:08:31.664512

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e8b3f0c7a92d'
down_revision: Union[str, None] = 'd5f27a1c8e60'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The guard blocks DELETE outright and blocks every content change, but must
# still permit the one UPDATE PostgreSQL performs itself: `ON DELETE SET NULL`
# detaching a reference when a user, case or patient row is removed. A blanket
# rejection made those rows undeletable — an audit trail that quietly prevents
# closing an account is a bug, not a compliance control.
GUARD_FUNCTION = """
CREATE OR REPLACE FUNCTION audit_logs_append_only() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'audit_logs is append-only: DELETE is not permitted'
            USING ERRCODE = 'check_violation';
    END IF;

    -- Every recorded fact must survive byte-identical. deleted_at is included
    -- so soft-deletion cannot be used to hide an entry from the trail.
    IF NEW.id             IS DISTINCT FROM OLD.id
    OR NEW.created_at     IS DISTINCT FROM OLD.created_at
    OR NEW.deleted_at     IS DISTINCT FROM OLD.deleted_at
    OR NEW.user_name      IS DISTINCT FROM OLD.user_name
    OR NEW.user_role      IS DISTINCT FROM OLD.user_role
    OR NEW.action         IS DISTINCT FROM OLD.action
    OR NEW.resource       IS DISTINCT FROM OLD.resource
    OR NEW.resource_id    IS DISTINCT FROM OLD.resource_id
    OR NEW.ip_address     IS DISTINCT FROM OLD.ip_address
    OR NEW.status         IS DISTINCT FROM OLD.status
    OR NEW.details        IS DISTINCT FROM OLD.details
    OR NEW.actor_type     IS DISTINCT FROM OLD.actor_type
    OR NEW.event_type     IS DISTINCT FROM OLD.event_type
    OR NEW.field_changed  IS DISTINCT FROM OLD.field_changed
    OR NEW.previous_value IS DISTINCT FROM OLD.previous_value
    OR NEW.new_value      IS DISTINCT FROM OLD.new_value
    OR NEW.reason         IS DISTINCT FROM OLD.reason
    THEN
        RAISE EXCEPTION 'audit_logs is append-only: recorded content cannot be modified'
            USING ERRCODE = 'check_violation';
    END IF;

    -- References may only move toward NULL. Re-pointing an entry at a
    -- different case or user would rewrite who and what it was about.
    IF (NEW.user_id    IS NOT NULL AND NEW.user_id    IS DISTINCT FROM OLD.user_id)
    OR (NEW.case_id    IS NOT NULL AND NEW.case_id    IS DISTINCT FROM OLD.case_id)
    OR (NEW.patient_id IS NOT NULL AND NEW.patient_id IS DISTINCT FROM OLD.patient_id)
    THEN
        RAISE EXCEPTION 'audit_logs is append-only: references may only be detached'
            USING ERRCODE = 'check_violation';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""

GUARD_TRIGGER = """
CREATE TRIGGER audit_logs_no_mutation
BEFORE UPDATE OR DELETE ON audit_logs
FOR EACH ROW EXECUTE FUNCTION audit_logs_append_only();
"""


def upgrade() -> None:
    op.add_column("audit_logs", sa.Column("case_id", sa.UUID(), nullable=True))
    op.add_column("audit_logs", sa.Column("patient_id", sa.UUID(), nullable=True))
    op.add_column(
        "audit_logs",
        sa.Column(
            "actor_type", sa.String(length=20), nullable=False,
            server_default="system",
        ),
    )
    op.alter_column("audit_logs", "actor_type", server_default=None)
    op.add_column("audit_logs", sa.Column("event_type", sa.String(length=60), nullable=True))
    op.add_column("audit_logs", sa.Column("field_changed", sa.String(length=80), nullable=True))
    op.add_column("audit_logs", sa.Column("previous_value", sa.Text(), nullable=True))
    op.add_column("audit_logs", sa.Column("new_value", sa.Text(), nullable=True))
    op.add_column("audit_logs", sa.Column("reason", sa.Text(), nullable=True))

    op.create_index(op.f("ix_audit_logs_case_id"), "audit_logs", ["case_id"])
    op.create_index(op.f("ix_audit_logs_patient_id"), "audit_logs", ["patient_id"])
    op.create_index(op.f("ix_audit_logs_actor_type"), "audit_logs", ["actor_type"])
    op.create_index(op.f("ix_audit_logs_event_type"), "audit_logs", ["event_type"])
    op.create_index("idx_audit_case_created", "audit_logs", ["case_id", "created_at"])

    op.create_foreign_key(
        "fk_audit_logs_case_id_cases", "audit_logs", "cases",
        ["case_id"], ["id"], ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_audit_logs_patient_id_patients", "audit_logs", "patients",
        ["patient_id"], ["id"], ondelete="SET NULL",
    )
    op.create_check_constraint(
        "audit_log_actor_type_check", "audit_logs",
        "actor_type IN ('patient', 'doctor', 'ai', 'admin', 'system')",
    )

    # Append-only guard. Installed last so the column work above is unaffected.
    if op.get_bind().dialect.name == "postgresql":
        op.execute(GUARD_FUNCTION)
        op.execute("DROP TRIGGER IF EXISTS audit_logs_no_mutation ON audit_logs")
        op.execute(GUARD_TRIGGER)


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS audit_logs_no_mutation ON audit_logs")
        op.execute("DROP FUNCTION IF EXISTS audit_logs_append_only()")

    op.drop_constraint("audit_log_actor_type_check", "audit_logs", type_="check")
    op.drop_constraint("fk_audit_logs_patient_id_patients", "audit_logs", type_="foreignkey")
    op.drop_constraint("fk_audit_logs_case_id_cases", "audit_logs", type_="foreignkey")

    op.drop_index("idx_audit_case_created", table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_event_type"), table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_actor_type"), table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_patient_id"), table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_case_id"), table_name="audit_logs")

    for column in ("reason", "new_value", "previous_value", "field_changed",
                   "event_type", "actor_type", "patient_id", "case_id"):
        op.drop_column("audit_logs", column)
