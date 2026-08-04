"""pharmacy network, inventory, ordering and delivery tracking

Phase 2 of the prescription-to-pharmacy workflow. Five new tables, no changes
to any existing one, so the migration is purely additive and the downgrade is a
clean drop.

`pharmacies` holds two kinds of row. Onboarded partners (`is_partner=True`)
have inventory and can take orders; places discovered through Google Places
(`is_partner=False`) are visible and routable but hold no stock, because Places
knows where a chemist is and nothing about its shelves.

`pharmacy_inventory` joins to prescriptions on `rxcui`, not on drug name —
"Crocin", "Paracetamol" and "Acetaminophen" are one ingredient under three
labels, and name matching silently mis-fills prescriptions.

`medicine_orders` references its prescription with RESTRICT rather than
CASCADE: a dispensing record must outlive edits to the prescription behind it.

Revision ID: d7b93e5a14c2
Revises: c4e8a17d2b93
Create Date: 2026-08-03 16:05:44.118203

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "d7b93e5a14c2"
down_revision: Union[str, None] = "c4e8a17d2b93"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _uuid():
    """UUID type that also works on SQLite, which the test suite uses."""
    return postgresql.UUID(as_uuid=True).with_variant(sa.String(36), "sqlite")


def _audit_columns() -> list[sa.Column]:
    """The Base mixin columns every table in this schema carries."""
    return [
        sa.Column("id", _uuid(), primary_key=True, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    ]


def upgrade() -> None:
    op.create_table(
        "pharmacies",
        *_audit_columns(),
        sa.Column("name", sa.String(200), nullable=False, index=True),
        sa.Column("address", sa.String(500), nullable=False, server_default=""),
        sa.Column("city", sa.String(120), nullable=True),
        sa.Column("postal_code", sa.String(20), nullable=True),
        sa.Column("phone", sa.String(50), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("google_place_id", sa.String(255), nullable=True, unique=True, index=True),
        sa.Column("licence_number", sa.String(100), nullable=True),
        sa.Column("rating", sa.Float(), nullable=False, server_default="0"),
        sa.Column("total_ratings", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_partner", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_24x7", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("opens_at", sa.String(5), nullable=True),
        sa.Column("closes_at", sa.String(5), nullable=True),
        sa.Column("delivers", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("delivery_radius_km", sa.Float(), nullable=False, server_default="8"),
        sa.Column("delivery_fee", sa.Float(), nullable=False, server_default="0"),
        sa.Column("free_delivery_above", sa.Float(), nullable=True),
        sa.Column("min_order_value", sa.Float(), nullable=False, server_default="0"),
        sa.Column("avg_prep_minutes", sa.Integer(), nullable=False, server_default="15"),
        sa.CheckConstraint("rating >= 0.0 AND rating <= 5.0", name="pharmacy_rating_check"),
        sa.CheckConstraint(
            "latitude >= -90.0 AND latitude <= 90.0", name="pharmacy_latitude_check"
        ),
        sa.CheckConstraint(
            "longitude >= -180.0 AND longitude <= 180.0", name="pharmacy_longitude_check"
        ),
    )
    # Bounding-box scans hit both columns together; the composite index is what
    # keeps a nearby search from degrading into a full table scan as the network
    # grows.
    op.create_index("ix_pharmacies_geo", "pharmacies", ["latitude", "longitude"])

    op.create_table(
        "pharmacy_inventory",
        *_audit_columns(),
        sa.Column(
            "pharmacy_id", _uuid(),
            sa.ForeignKey("pharmacies.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("sku", sa.String(64), nullable=False),
        sa.Column("rxcui", sa.String(20), nullable=True),
        sa.Column("medicine_name", sa.String(200), nullable=False, index=True),
        sa.Column("generic_name", sa.String(200), nullable=True, index=True),
        sa.Column("brand_name", sa.String(200), nullable=True),
        sa.Column("manufacturer", sa.String(200), nullable=True),
        sa.Column("strength", sa.String(100), nullable=True),
        sa.Column("form", sa.String(50), nullable=True),
        sa.Column("pack_size", sa.String(50), nullable=True),
        sa.Column("is_generic", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "requires_prescription", sa.Boolean(), nullable=False, server_default=sa.true()
        ),
        sa.Column("mrp", sa.Float(), nullable=False, server_default="0"),
        sa.Column("selling_price", sa.Float(), nullable=False, server_default="0"),
        sa.Column("discount_percent", sa.Float(), nullable=False, server_default="0"),
        sa.Column("stock_quantity", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("low_stock_threshold", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("restock_expected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stock_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("pharmacy_id", "sku", name="uq_pharmacy_inventory_pharmacy_sku"),
        sa.CheckConstraint("stock_quantity >= 0", name="inventory_stock_non_negative"),
        sa.CheckConstraint(
            "mrp >= 0 AND selling_price >= 0", name="inventory_price_non_negative"
        ),
        sa.CheckConstraint(
            "discount_percent >= 0 AND discount_percent <= 100",
            name="inventory_discount_range",
        ),
    )
    op.create_index("ix_pharmacy_inventory_rxcui", "pharmacy_inventory", ["rxcui"])
    op.create_index(
        "ix_pharmacy_inventory_lookup", "pharmacy_inventory", ["pharmacy_id", "rxcui"]
    )

    op.create_table(
        "medicine_orders",
        *_audit_columns(),
        sa.Column("order_number", sa.String(24), nullable=False, unique=True, index=True),
        sa.Column(
            "patient_id", _uuid(),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column(
            "prescription_id", _uuid(),
            sa.ForeignKey("prescriptions.id", ondelete="RESTRICT"),
            nullable=False, index=True,
        ),
        sa.Column(
            "pharmacy_id", _uuid(),
            sa.ForeignKey("pharmacies.id", ondelete="RESTRICT"),
            nullable=False, index=True,
        ),
        sa.Column("pharmacy_name", sa.String(200), nullable=False),
        sa.Column("status", sa.String(30), nullable=False, server_default="received", index=True),
        sa.Column("subtotal", sa.Float(), nullable=False, server_default="0"),
        sa.Column("discount_total", sa.Float(), nullable=False, server_default="0"),
        sa.Column("delivery_fee", sa.Float(), nullable=False, server_default="0"),
        sa.Column("total", sa.Float(), nullable=False, server_default="0"),
        sa.Column("currency", sa.String(3), nullable=False, server_default="INR"),
        sa.Column("delivery_address", sa.String(500), nullable=False, server_default=""),
        sa.Column("delivery_latitude", sa.Float(), nullable=True),
        sa.Column("delivery_longitude", sa.Float(), nullable=True),
        sa.Column("delivery_notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("distance_km", sa.Float(), nullable=True),
        sa.Column("eta_minutes", sa.Integer(), nullable=True),
        sa.Column("estimated_delivery_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivery_partner_name", sa.String(150), nullable=True),
        sa.Column("delivery_partner_phone", sa.String(50), nullable=True),
        sa.Column("placed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancellation_reason", sa.String(500), nullable=True),
        sa.Column(
            "fulfilment_provider", sa.String(40), nullable=False, server_default="local_db"
        ),
        sa.CheckConstraint(
            "status IN ('received', 'preparing', 'packed', 'out_for_delivery', "
            "'delivered', 'cancelled')",
            name="medicine_order_status_check",
        ),
        sa.CheckConstraint("total >= 0", name="medicine_order_total_non_negative"),
    )
    op.create_index(
        "ix_medicine_orders_patient_status", "medicine_orders", ["patient_id", "status"]
    )

    op.create_table(
        "medicine_order_items",
        *_audit_columns(),
        sa.Column(
            "order_id", _uuid(),
            sa.ForeignKey("medicine_orders.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column(
            "inventory_id", _uuid(),
            sa.ForeignKey("pharmacy_inventory.id", ondelete="SET NULL"), nullable=True,
        ),
        sa.Column(
            "medication_id", _uuid(),
            sa.ForeignKey("medications.id", ondelete="SET NULL"), nullable=True,
        ),
        sa.Column("medicine_name", sa.String(200), nullable=False),
        sa.Column("generic_name", sa.String(200), nullable=True),
        sa.Column("brand_name", sa.String(200), nullable=True),
        sa.Column("strength", sa.String(100), nullable=True),
        sa.Column("rxcui", sa.String(20), nullable=True),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("unit_price", sa.Float(), nullable=False, server_default="0"),
        sa.Column("mrp", sa.Float(), nullable=False, server_default="0"),
        sa.Column("discount_percent", sa.Float(), nullable=False, server_default="0"),
        sa.Column("line_total", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "is_generic_substitute", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("substituted_for", sa.String(200), nullable=True),
        sa.CheckConstraint("quantity > 0", name="order_item_quantity_positive"),
        sa.CheckConstraint("line_total >= 0", name="order_item_total_non_negative"),
    )

    op.create_table(
        "order_status_events",
        *_audit_columns(),
        sa.Column(
            "order_id", _uuid(),
            sa.ForeignKey("medicine_orders.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("note", sa.String(500), nullable=False, server_default=""),
        sa.Column("actor_type", sa.String(30), nullable=False, server_default="system"),
        sa.Column("actor_id", _uuid(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("order_status_events")
    op.drop_table("medicine_order_items")
    op.drop_index("ix_medicine_orders_patient_status", table_name="medicine_orders")
    op.drop_table("medicine_orders")
    op.drop_index("ix_pharmacy_inventory_lookup", table_name="pharmacy_inventory")
    op.drop_index("ix_pharmacy_inventory_rxcui", table_name="pharmacy_inventory")
    op.drop_table("pharmacy_inventory")
    op.drop_index("ix_pharmacies_geo", table_name="pharmacies")
    op.drop_table("pharmacies")
