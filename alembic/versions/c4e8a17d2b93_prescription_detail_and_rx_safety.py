"""prescription detail fields and AI safety verification

Phase 1 of the prescription-to-pharmacy workflow.

Three groups of change, all additive:

* `medications` gains the dispensing detail an order needs — strength, brand,
  food timing, route, quantity — plus `rxcui`, the RxNorm key that free-text
  drug names cannot provide. Inventory lookups and safety checks both join on
  it, so it is indexed.

* `prescriptions` gains a prescriber snapshot. A prescription is a legal record
  of who ordered what on whose authority; reading the clinician's hospital and
  registration live would let a later profile edit rewrite prescriptions signed
  years earlier. The columns are copied at issue time instead.

* `prescription_verifications` and `verification_findings` hold AI-assisted
  safety reviews. They sit beside a prescription and never inside it — nothing
  in this feature may alter what a doctor prescribed.

Every column is nullable and every table is new, so existing rows stay valid and
the downgrade is a clean drop.

Revision ID: c4e8a17d2b93
Revises: b91d47e3ca08
Create Date: 2026-08-03 14:20:11.402887

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "c4e8a17d2b93"
down_revision: Union[str, None] = "b91d47e3ca08"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _uuid():
    """UUID type that also works on SQLite, which the test suite uses."""
    return postgresql.UUID(as_uuid=True).with_variant(sa.String(36), "sqlite")


def _json():
    return sa.JSON().with_variant(postgresql.JSONB, "postgresql")


def upgrade() -> None:
    # ── medications: dispensing detail ──────────────────────────────────
    op.add_column("medications", sa.Column("brand_name", sa.String(150), nullable=True))
    op.add_column("medications", sa.Column("strength", sa.String(100), nullable=True))
    op.add_column("medications", sa.Column("food_instruction", sa.String(50), nullable=True))
    op.add_column("medications", sa.Column("route", sa.String(50), nullable=True))
    op.add_column("medications", sa.Column("quantity", sa.Integer(), nullable=True))
    op.add_column("medications", sa.Column("rxcui", sa.String(20), nullable=True))
    op.create_index("ix_medications_rxcui", "medications", ["rxcui"])

    # ── prescriptions: prescriber snapshot ──────────────────────────────
    op.add_column("prescriptions", sa.Column("doctor_specialty", sa.String(100), nullable=True))
    op.add_column("prescriptions", sa.Column("doctor_qualification", sa.String(255), nullable=True))
    op.add_column("prescriptions", sa.Column("doctor_hospital", sa.String(150), nullable=True))
    op.add_column("prescriptions", sa.Column("doctor_registration_number", sa.String(100), nullable=True))
    op.add_column("prescriptions", sa.Column("doctor_experience_years", sa.Integer(), nullable=True))
    op.add_column("prescriptions", sa.Column("doctor_avatar_url", sa.String(255), nullable=True))
    op.add_column("prescriptions", sa.Column("consultation_date", sa.DateTime(timezone=True), nullable=True))
    op.add_column("prescriptions", sa.Column("signed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("prescriptions", sa.Column("doctor_signature_url", sa.String(255), nullable=True))
    op.add_column("prescriptions", sa.Column("prescription_image_url", sa.String(255), nullable=True))
    op.add_column("prescriptions", sa.Column("pdf_url", sa.String(255), nullable=True))

    # ── safety reviews ──────────────────────────────────────────────────
    op.create_table(
        "prescription_verifications",
        sa.Column("id", _uuid(), primary_key=True, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "prescription_id",
            _uuid(),
            sa.ForeignKey("prescriptions.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("status", sa.String(30), nullable=False, server_default="pending"),
        sa.Column("verdict", sa.String(20), nullable=False, server_default="unknown"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("checked_medication_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("unchecked_medications", _json(), nullable=False, server_default="[]"),
        sa.Column("sources_used", _json(), nullable=False, server_default="[]"),
        sa.Column("engine_version", sa.String(50), nullable=False, server_default=""),
        sa.Column("model_used", sa.String(100), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "verdict IN ('safe', 'warning', 'critical', 'unknown')",
            name="rx_verification_verdict_check",
        ),
        sa.CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="rx_verification_confidence_check",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'completed', 'failed', 'degraded')",
            name="rx_verification_status_check",
        ),
    )

    op.create_table(
        "verification_findings",
        sa.Column("id", _uuid(), primary_key=True, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "verification_id",
            _uuid(),
            sa.ForeignKey("prescription_verifications.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("category", sa.String(40), nullable=False, index=True),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False, server_default=""),
        sa.Column("recommendation", sa.Text(), nullable=False, server_default=""),
        sa.Column("medications_involved", _json(), nullable=False, server_default="[]"),
        sa.Column("source", sa.String(30), nullable=False, server_default=""),
        sa.Column("evidence", _json(), nullable=False, server_default="[]"),
        sa.CheckConstraint(
            "severity IN ('safe', 'warning', 'critical', 'unknown')",
            name="rx_finding_severity_check",
        ),
        sa.CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="rx_finding_confidence_check",
        ),
    )


def downgrade() -> None:
    op.drop_table("verification_findings")
    op.drop_table("prescription_verifications")

    for column in (
        "pdf_url",
        "prescription_image_url",
        "doctor_signature_url",
        "signed_at",
        "consultation_date",
        "doctor_avatar_url",
        "doctor_experience_years",
        "doctor_registration_number",
        "doctor_hospital",
        "doctor_qualification",
        "doctor_specialty",
    ):
        op.drop_column("prescriptions", column)

    op.drop_index("ix_medications_rxcui", table_name="medications")
    for column in ("rxcui", "quantity", "route", "food_instruction", "strength", "brand_name"):
        op.drop_column("medications", column)
