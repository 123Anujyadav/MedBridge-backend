"""pharmacy owner role and store link

Adds the identity half of the Pharmacy Owner Portal. Two changes, both additive:

* `users.role` accepts `'pharmacy'`. This extends the existing RBAC rather than
  introducing a second identity system — the same login, the same token, the
  same `RoleChecker`.

* `users.pharmacy_id` links an owner account to the store it operates. It is
  the reason no portal endpoint takes a store id: the pharmacy an owner may act
  on is read from their own user row, so there is no parameter to tamper with
  to reach another store's inventory or orders.

Rewriting a CHECK constraint is not additive on SQLite, which cannot ALTER a
constraint in place — `batch_alter_table` rebuilds the table there while
Postgres does a plain DROP/ADD. Existing rows all hold one of the three
original roles, so every row satisfies the widened constraint and no data
moves.

Revision ID: f2a71c94e6d8
Revises: e8c15b473a06
Create Date: 2026-08-03 20:41:52.336104

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "f2a71c94e6d8"
down_revision: Union[str, None] = "e8c15b473a06"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_ROLES = "role IN ('patient', 'doctor', 'admin')"
NEW_ROLES = "role IN ('patient', 'doctor', 'admin', 'pharmacy')"


def _uuid():
    return postgresql.UUID(as_uuid=True).with_variant(sa.String(36), "sqlite")


def upgrade() -> None:
    bind = op.get_bind()

    if bind.dialect.name == "sqlite":
        # SQLite has no ALTER CONSTRAINT; batch mode copies the table.
        with op.batch_alter_table("users", schema=None) as batch:
            batch.add_column(sa.Column("pharmacy_id", _uuid(), nullable=True))
            batch.drop_constraint("user_role_check", type_="check")
            batch.create_check_constraint("user_role_check", NEW_ROLES)
    else:
        op.add_column("users", sa.Column("pharmacy_id", _uuid(), nullable=True))
        op.drop_constraint("user_role_check", "users", type_="check")
        op.create_check_constraint("user_role_check", "users", NEW_ROLES)
        op.create_foreign_key(
            "fk_users_pharmacy_id",
            "users",
            "pharmacies",
            ["pharmacy_id"],
            ["id"],
            ondelete="SET NULL",
        )

    op.create_index("ix_users_pharmacy_id", "users", ["pharmacy_id"])


def downgrade() -> None:
    op.drop_index("ix_users_pharmacy_id", table_name="users")
    bind = op.get_bind()

    # Any pharmacy account must be demoted first, or the narrowed constraint
    # would reject rows that are currently valid.
    op.execute("UPDATE users SET is_active = false WHERE role = 'pharmacy'")
    op.execute("UPDATE users SET role = 'patient' WHERE role = 'pharmacy'")

    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("users", schema=None) as batch:
            batch.drop_constraint("user_role_check", type_="check")
            batch.create_check_constraint("user_role_check", OLD_ROLES)
            batch.drop_column("pharmacy_id")
    else:
        op.drop_constraint("fk_users_pharmacy_id", "users", type_="foreignkey")
        op.drop_constraint("user_role_check", "users", type_="check")
        op.create_check_constraint("user_role_check", "users", OLD_ROLES)
        op.drop_column("users", "pharmacy_id")
