"""pharmacy administration module

Adds the business, compliance and catalogue detail an administrator needs to
onboard and monitor partner pharmacies.

Three groups of change, all additive:

* `pharmacies` gains business identity, branding, fulfilment options,
  settlement details and a verification lifecycle.
* `pharmacy_inventory` gains catalogue, batch, shelf-life and replenishment
  columns.
* `pharmacy_documents` and `pharmacy_verification_events` are new.

Every existing partner is backfilled to `approved`. They were onboarded before
verification existed, and defaulting them to `pending` would have removed them
from patient-facing search the moment this migration ran — a silent outage of
the Phase 2 dispensing path. New rows default to `pending` and must be taken
through the workflow.

Revision ID: e8c15b473a06
Revises: d7b93e5a14c2
Create Date: 2026-08-03 18:12:36.740915

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "e8c15b473a06"
down_revision: Union[str, None] = "d7b93e5a14c2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _uuid():
    return postgresql.UUID(as_uuid=True).with_variant(sa.String(36), "sqlite")


def _json():
    return sa.JSON().with_variant(postgresql.JSONB, "postgresql")


def _audit_columns() -> list[sa.Column]:
    return [
        sa.Column("id", _uuid(), primary_key=True, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    ]


PHARMACY_COLUMNS = [
    ("owner_name", sa.String(200)),
    ("business_name", sa.String(250)),
    ("gst_number", sa.String(20)),
    ("drug_license_number", sa.String(100)),
    ("email", sa.String(255)),
    ("whatsapp", sa.String(50)),
    ("emergency_phone", sa.String(50)),
    ("logo_url", sa.String(500)),
    ("banner_url", sa.String(500)),
    ("express_delivery_radius_km", sa.Float()),
    ("upi_id", sa.String(120)),
    ("bank_account_name", sa.String(200)),
    ("bank_account_number", sa.String(60)),
    ("bank_ifsc", sa.String(20)),
    ("verification_notes", sa.Text()),
    ("rejection_reason", sa.String(500)),
    ("suspension_reason", sa.String(500)),
]

INVENTORY_COLUMNS = [
    ("composition", sa.String(500)),
    ("drug_schedule", sa.String(20)),
    ("category", sa.String(120)),
    ("barcode", sa.String(64)),
    ("storage_instructions", sa.String(300)),
    ("batch_number", sa.String(60)),
    ("min_stock", sa.Integer()),
    ("max_stock", sa.Integer()),
    ("reorder_level", sa.Integer()),
]


def upgrade() -> None:
    # ── pharmacies ───────────────────────────────────────────────────────
    for name, column_type in PHARMACY_COLUMNS:
        op.add_column("pharmacies", sa.Column(name, column_type, nullable=True))

    for name in ("drug_license_expiry", "verified_at", "suspended_at"):
        op.add_column("pharmacies", sa.Column(name, sa.DateTime(timezone=True), nullable=True))

    op.add_column("pharmacies", sa.Column("verified_by", _uuid(), nullable=True))
    op.add_column(
        "pharmacies",
        sa.Column("store_images", _json(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "pharmacies",
        sa.Column("holiday_dates", _json(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "pharmacies",
        sa.Column("express_delivery", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "pharmacies",
        sa.Column("pickup_available", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "pharmacies",
        sa.Column(
            "platform_commission_percent", sa.Float(), nullable=False, server_default="0"
        ),
    )
    op.add_column(
        "pharmacies",
        sa.Column(
            "verification_status", sa.String(30), nullable=False, server_default="pending"
        ),
    )

    # Grandfather everyone already dispensing. Without this, the Phase 2
    # patient search would return zero partner pharmacies immediately after
    # deployment.
    op.execute(
        "UPDATE pharmacies SET verification_status = 'approved' WHERE is_partner = true"
    )

    op.create_index("ix_pharmacies_gst_number", "pharmacies", ["gst_number"])
    op.create_index(
        "ix_pharmacies_drug_license_number", "pharmacies", ["drug_license_number"]
    )
    op.create_index(
        "ix_pharmacies_verification_status", "pharmacies", ["verification_status"]
    )

    # ── pharmacy_inventory ───────────────────────────────────────────────
    for name, column_type in INVENTORY_COLUMNS:
        op.add_column("pharmacy_inventory", sa.Column(name, column_type, nullable=True))

    for name in ("manufacturing_date", "expiry_date"):
        op.add_column(
            "pharmacy_inventory", sa.Column(name, sa.DateTime(timezone=True), nullable=True)
        )

    op.add_column(
        "pharmacy_inventory",
        sa.Column("gst_percent", sa.Float(), nullable=False, server_default="0"),
    )

    op.create_index("ix_pharmacy_inventory_category", "pharmacy_inventory", ["category"])
    op.create_index("ix_pharmacy_inventory_barcode", "pharmacy_inventory", ["barcode"])
    # Expiry sweeps scan by date across every pharmacy, so this is indexed on
    # its own rather than only in the per-pharmacy composite.
    op.create_index("ix_pharmacy_inventory_expiry", "pharmacy_inventory", ["expiry_date"])

    # ── documents ────────────────────────────────────────────────────────
    op.create_table(
        "pharmacy_documents",
        *_audit_columns(),
        sa.Column(
            "pharmacy_id", _uuid(),
            sa.ForeignKey("pharmacies.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("doc_type", sa.String(50), nullable=False, index=True),
        sa.Column("file_url", sa.String(500), nullable=False),
        sa.Column("file_name", sa.String(255), nullable=False, server_default=""),
        sa.Column("document_number", sa.String(120), nullable=True),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(30), nullable=False, server_default="uploaded"),
        sa.Column("review_notes", sa.String(500), nullable=True),
        sa.Column("reviewed_by", _uuid(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('uploaded', 'under_review', 'approved', 'rejected', 'expired')",
            name="pharmacy_document_status_check",
        ),
    )
    op.create_index("ix_pharmacy_documents_expiry", "pharmacy_documents", ["expires_at"])

    # ── verification trail ───────────────────────────────────────────────
    op.create_table(
        "pharmacy_verification_events",
        *_audit_columns(),
        sa.Column(
            "pharmacy_id", _uuid(),
            sa.ForeignKey("pharmacies.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("from_status", sa.String(30), nullable=True),
        sa.Column("to_status", sa.String(30), nullable=False),
        sa.Column("note", sa.String(1000), nullable=False, server_default=""),
        sa.Column("actor_id", _uuid(), nullable=True),
        sa.Column("actor_name", sa.String(200), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_table("pharmacy_verification_events")
    op.drop_index("ix_pharmacy_documents_expiry", table_name="pharmacy_documents")
    op.drop_table("pharmacy_documents")

    op.drop_index("ix_pharmacy_inventory_expiry", table_name="pharmacy_inventory")
    op.drop_index("ix_pharmacy_inventory_barcode", table_name="pharmacy_inventory")
    op.drop_index("ix_pharmacy_inventory_category", table_name="pharmacy_inventory")
    op.drop_column("pharmacy_inventory", "gst_percent")
    for name in ("expiry_date", "manufacturing_date"):
        op.drop_column("pharmacy_inventory", name)
    for name, _ in reversed(INVENTORY_COLUMNS):
        op.drop_column("pharmacy_inventory", name)

    op.drop_index("ix_pharmacies_verification_status", table_name="pharmacies")
    op.drop_index("ix_pharmacies_drug_license_number", table_name="pharmacies")
    op.drop_index("ix_pharmacies_gst_number", table_name="pharmacies")
    for name in (
        "verification_status",
        "platform_commission_percent",
        "pickup_available",
        "express_delivery",
        "holiday_dates",
        "store_images",
        "verified_by",
        "suspended_at",
        "verified_at",
        "drug_license_expiry",
    ):
        op.drop_column("pharmacies", name)
    for name, _ in reversed(PHARMACY_COLUMNS):
        op.drop_column("pharmacies", name)
