"""add immutable report version history

Introduces `report_versions`, a frozen snapshot of a clinical document at each
revision, plus `reports.current_version`.

Immutability is enforced by a trigger rather than by convention:

* DELETE is rejected.
* Content columns are frozen once written.
* Lifecycle columns (status, approval note, rejection reason, and the rendered
  file for the version) may change only on a report's newest version. Older
  versions are entirely read-only.

Snapshots are stored whole rather than as diffs: a diff chain is only as
recoverable as every link in it, and losing one link would lose the clinical
documentation on both sides of it.

Revision ID: f4c19d8b30ae
Revises: e8b3f0c7a92d
Create Date: 2026-07-26 15:14:07.882301

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f4c19d8b30ae'
down_revision: Union[str, None] = 'e8b3f0c7a92d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

GUARD_FUNCTION = """
CREATE OR REPLACE FUNCTION report_versions_immutable() RETURNS trigger AS $$
DECLARE
    latest INTEGER;
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'report_versions is immutable: DELETE is not permitted'
            USING ERRCODE = 'check_violation';
    END IF;

    -- Identity and the clinical snapshot are frozen for every version,
    -- newest included. An approved document that can be edited in place is
    -- not a record of what was approved.
    IF NEW.id                     IS DISTINCT FROM OLD.id
    OR NEW.report_id              IS DISTINCT FROM OLD.report_id
    OR NEW.version_number         IS DISTINCT FROM OLD.version_number
    OR NEW.created_at             IS DISTINCT FROM OLD.created_at
    OR NEW.deleted_at             IS DISTINCT FROM OLD.deleted_at
    OR NEW.author_name            IS DISTINCT FROM OLD.author_name
    OR NEW.author_type            IS DISTINCT FROM OLD.author_type
    OR NEW.title                  IS DISTINCT FROM OLD.title
    OR NEW.chief_complaint        IS DISTINCT FROM OLD.chief_complaint
    OR NEW.summary                IS DISTINCT FROM OLD.summary
    OR NEW.content                IS DISTINCT FROM OLD.content
    OR NEW.diagnosis              IS DISTINCT FROM OLD.diagnosis
    OR NEW.clinical_notes         IS DISTINCT FROM OLD.clinical_notes
    OR NEW.prescription           IS DISTINCT FROM OLD.prescription
    OR NEW.follow_up_instructions IS DISTINCT FROM OLD.follow_up_instructions
    OR NEW.ai_findings            IS DISTINCT FROM OLD.ai_findings
    OR NEW.symptoms::text         IS DISTINCT FROM OLD.symptoms::text
    OR NEW.recommended_tests::text IS DISTINCT FROM OLD.recommended_tests::text
    OR NEW.recommendations::text  IS DISTINCT FROM OLD.recommendations::text
    OR NEW.content_hash           IS DISTINCT FROM OLD.content_hash
    OR NEW.restored_from_version  IS DISTINCT FROM OLD.restored_from_version
    THEN
        RAISE EXCEPTION 'report_versions is immutable: version content cannot be modified'
            USING ERRCODE = 'check_violation';
    END IF;

    -- Lifecycle may only advance on the newest version of the report.
    SELECT MAX(version_number) INTO latest
    FROM report_versions WHERE report_id = OLD.report_id;

    IF OLD.version_number < latest THEN
        RAISE EXCEPTION
            'report_versions is immutable: version % is historical and read-only',
            OLD.version_number
            USING ERRCODE = 'check_violation';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""

GUARD_TRIGGER = """
CREATE TRIGGER report_versions_no_mutation
BEFORE UPDATE OR DELETE ON report_versions
FOR EACH ROW EXECUTE FUNCTION report_versions_immutable();
"""


def upgrade() -> None:
    op.create_table(
        "report_versions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("report_id", sa.UUID(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("author_id", sa.UUID(), nullable=True),
        sa.Column("author_name", sa.String(length=200), nullable=False),
        sa.Column("author_type", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("chief_complaint", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("diagnosis", sa.Text(), nullable=False),
        sa.Column("clinical_notes", sa.Text(), nullable=False),
        sa.Column("prescription", sa.Text(), nullable=False),
        sa.Column("follow_up_instructions", sa.Text(), nullable=False),
        sa.Column("ai_findings", sa.Text(), nullable=False),
        sa.Column("symptoms", sa.JSON(), nullable=False),
        sa.Column("recommended_tests", sa.JSON(), nullable=False),
        sa.Column("recommendations", sa.JSON(), nullable=False),
        sa.Column("ai_confidence_score", sa.Float(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("file_url", sa.String(length=255), nullable=True),
        sa.Column("file_size", sa.String(length=50), nullable=True),
        sa.Column("approval_note", sa.Text(), nullable=True),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.Column("approved_by_name", sa.String(length=200), nullable=True),
        sa.Column("approved_at", sa.String(length=50), nullable=True),
        sa.Column("restored_from_version", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["report_id"], ["reports.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["author_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("report_id", "version_number", name="uq_report_version_number"),
        sa.CheckConstraint("author_type IN ('doctor', 'ai', 'system')",
                           name="report_version_author_type_check"),
        sa.CheckConstraint(
            "status IN ('draft', 'ai_draft', 'under_review', 'approved', "
            "'rejected', 'shared', 'archived')",
            name="report_version_status_check"),
        sa.CheckConstraint("version_number >= 1", name="report_version_number_check"),
    )
    # The shared Base indexes `id`; every table in this schema carries it.
    op.create_index(op.f("ix_report_versions_id"), "report_versions", ["id"])
    op.create_index(op.f("ix_report_versions_report_id"), "report_versions", ["report_id"])
    op.create_index(op.f("ix_report_versions_author_id"), "report_versions", ["author_id"])
    op.create_index(op.f("ix_report_versions_author_type"), "report_versions", ["author_type"])
    op.create_index(op.f("ix_report_versions_status"), "report_versions", ["status"])
    op.create_index(op.f("ix_report_versions_content_hash"), "report_versions", ["content_hash"])
    op.create_index("idx_report_version_lookup", "report_versions",
                    ["report_id", "version_number"])

    op.add_column(
        "reports",
        sa.Column("current_version", sa.Integer(), nullable=False, server_default="0"),
    )
    op.alter_column("reports", "current_version", server_default=None)

    if op.get_bind().dialect.name == "postgresql":
        op.execute(GUARD_FUNCTION)
        op.execute("DROP TRIGGER IF EXISTS report_versions_no_mutation ON report_versions")
        op.execute(GUARD_TRIGGER)


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS report_versions_no_mutation ON report_versions")
        op.execute("DROP FUNCTION IF EXISTS report_versions_immutable()")
    op.drop_column("reports", "current_version")
    op.drop_table("report_versions")
