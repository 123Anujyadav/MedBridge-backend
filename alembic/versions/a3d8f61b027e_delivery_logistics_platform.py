"""delivery and logistics platform

Adds the last actor in the chain: the rider who carries medicine from a
verified pharmacy to the patient.

Two groups of change, both additive:

* `users.role` accepts `'delivery_partner'`. This extends the existing RBAC in
  exactly the way `'pharmacy'` did in f2a71c94e6d8 — same login, same token,
  same `RoleChecker`. No second identity system.

* `delivery_partners`, `delivery_assignments` and `delivery_events` are new.
  Nothing in `medicine_orders` changes: the order keeps the coarse status
  Phase 2 defined, and the assignment tracks the finer legs beside it.

Rewriting a CHECK constraint is not additive on SQLite, which cannot ALTER one
in place, so `batch_alter_table` rebuilds the table there while Postgres does a
plain DROP/ADD. Every existing row already holds one of the four permitted
roles, so all of them satisfy the widened constraint and no data moves.

Revision ID: a3d8f61b027e
Revises: f2a71c94e6d8
Create Date: 2026-08-03 23:18:44.905612

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "a3d8f61b027e"
down_revision: Union[str, None] = "f2a71c94e6d8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_ROLES = "role IN ('patient', 'doctor', 'admin', 'pharmacy')"
NEW_ROLES = "role IN ('patient', 'doctor', 'admin', 'pharmacy', 'delivery_partner')"


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


def upgrade() -> None:
    bind = op.get_bind()

    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("users", schema=None) as batch:
            batch.drop_constraint("user_role_check", type_="check")
            batch.create_check_constraint("user_role_check", NEW_ROLES)
    else:
        op.drop_constraint("user_role_check", "users", type_="check")
        op.create_check_constraint("user_role_check", "users", NEW_ROLES)

    op.create_table(
        "delivery_partners",
        *_audit_columns(),
        sa.Column(
            "user_id", _uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False, unique=True, index=True,
        ),
        sa.Column("full_name", sa.String(200), nullable=False),
        sa.Column("phone", sa.String(50), nullable=False, index=True),
        sa.Column("photo_url", sa.String(500), nullable=True),
        sa.Column("date_of_birth", sa.String(50), nullable=True),
        sa.Column("address", sa.String(500), nullable=True),
        sa.Column("city", sa.String(120), nullable=True, index=True),
        sa.Column("vehicle_type", sa.String(30), nullable=True),
        sa.Column("vehicle_number", sa.String(30), nullable=True, index=True),
        sa.Column("vehicle_model", sa.String(120), nullable=True),
        sa.Column("driving_licence_number", sa.String(60), nullable=True),
        sa.Column("driving_licence_expiry", sa.DateTime(timezone=True), nullable=True),
        sa.Column("documents", _json(), nullable=False, server_default="[]"),
        sa.Column(
            "verification_status", sa.String(30), nullable=False, server_default="pending"
        ),
        sa.Column("verification_notes", sa.Text(), nullable=True),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("verified_by", _uuid(), nullable=True),
        sa.Column("suspension_reason", sa.String(500), nullable=True),
        sa.Column("is_online", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("current_latitude", sa.Float(), nullable=True),
        sa.Column("current_longitude", sa.Float(), nullable=True),
        sa.Column("location_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("experience_years", sa.Integer(), nullable=True),
        sa.Column("rating", sa.Float(), nullable=False, server_default="0"),
        sa.Column("total_ratings", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_deliveries", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_deliveries", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_distance_km", sa.Float(), nullable=False, server_default="0"),
        sa.Column("total_earnings", sa.Float(), nullable=False, server_default="0"),
        sa.CheckConstraint(
            "verification_status IN ('pending', 'document_review', 'approved', "
            "'rejected', 'suspended')",
            name="delivery_partner_verification_check",
        ),
        sa.CheckConstraint(
            "rating >= 0.0 AND rating <= 5.0", name="delivery_partner_rating_check"
        ),
    )
    # Dispatch searches for approved riders who are clocked on; both columns are
    # in every such query, so they are indexed together.
    op.create_index(
        "ix_delivery_partners_availability",
        "delivery_partners",
        ["is_online", "verification_status"],
    )

    op.create_table(
        "delivery_assignments",
        *_audit_columns(),
        sa.Column(
            "order_id", _uuid(),
            sa.ForeignKey("medicine_orders.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column(
            "partner_id", _uuid(),
            sa.ForeignKey("delivery_partners.id", ondelete="RESTRICT"), nullable=False,
        ),
        sa.Column(
            "pharmacy_id", _uuid(),
            sa.ForeignKey("pharmacies.id", ondelete="RESTRICT"), nullable=False,
        ),
        sa.Column("status", sa.String(30), nullable=False, server_default="offered", index=True),
        sa.Column("pickup_address", sa.String(500), nullable=False, server_default=""),
        sa.Column("pickup_latitude", sa.Float(), nullable=True),
        sa.Column("pickup_longitude", sa.Float(), nullable=True),
        sa.Column("drop_address", sa.String(500), nullable=False, server_default=""),
        sa.Column("drop_latitude", sa.Float(), nullable=True),
        sa.Column("drop_longitude", sa.Float(), nullable=True),
        sa.Column("distance_km", sa.Float(), nullable=True),
        sa.Column("eta_minutes", sa.Integer(), nullable=True),
        sa.Column("estimated_arrival_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivery_fee", sa.Float(), nullable=False, server_default="0"),
        sa.Column("partner_earning", sa.Float(), nullable=False, server_default="0"),
        sa.Column("otp_hash", sa.String(255), nullable=True),
        sa.Column("otp_issued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("otp_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("otp_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("proof_photo_url", sa.String(500), nullable=True),
        sa.Column("proof_signature_url", sa.String(500), nullable=True),
        sa.Column("delivery_notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("proof_latitude", sa.Float(), nullable=True),
        sa.Column("proof_longitude", sa.Float(), nullable=True),
        sa.Column("proof_captured_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("offered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("picked_up_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_reason", sa.String(500), nullable=True),
        sa.Column("assigned_by", _uuid(), nullable=True),
        sa.CheckConstraint(
            "status IN ('offered', 'accepted', 'en_route_pickup', 'at_pharmacy', "
            "'picked_up', 'out_for_delivery', 'at_patient', 'delivered', "
            "'cancelled', 'failed')",
            name="delivery_assignment_status_check",
        ),
        sa.CheckConstraint("otp_attempts >= 0", name="delivery_otp_attempts_check"),
    )
    op.create_index(
        "ix_delivery_assignments_partner_status",
        "delivery_assignments", ["partner_id", "status"],
    )
    op.create_index("ix_delivery_assignments_order", "delivery_assignments", ["order_id"])

    op.create_table(
        "delivery_events",
        *_audit_columns(),
        sa.Column(
            "assignment_id", _uuid(),
            sa.ForeignKey("delivery_assignments.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("note", sa.String(500), nullable=False, server_default=""),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("actor_type", sa.String(30), nullable=False, server_default="partner"),
        sa.Column("actor_id", _uuid(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("delivery_events")
    op.drop_index("ix_delivery_assignments_order", table_name="delivery_assignments")
    op.drop_index(
        "ix_delivery_assignments_partner_status", table_name="delivery_assignments"
    )
    op.drop_table("delivery_assignments")
    op.drop_index("ix_delivery_partners_availability", table_name="delivery_partners")
    op.drop_table("delivery_partners")

    # Demote riders first, or the narrowed constraint rejects rows that are
    # currently valid.
    op.execute("UPDATE users SET is_active = false WHERE role = 'delivery_partner'")
    op.execute("UPDATE users SET role = 'patient' WHERE role = 'delivery_partner'")

    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("users", schema=None) as batch:
            batch.drop_constraint("user_role_check", type_="check")
            batch.create_check_constraint("user_role_check", OLD_ROLES)
    else:
        op.drop_constraint("user_role_check", "users", type_="check")
        op.create_check_constraint("user_role_check", "users", OLD_ROLES)
