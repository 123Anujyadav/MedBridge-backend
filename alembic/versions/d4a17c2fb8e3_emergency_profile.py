"""Patient Emergency Profile.

The standing emergency record for a patient: who to call, the registered
address broken into parts an ambulance can be given, and the last known
coordinates.

Purely additive. One new table, no existing table altered, nothing dropped —
`patients.emergency_contact` is left exactly as it is so every screen and query
already reading it keeps working. The new table is the structured record the
SOS system will read later; this revision installs no dispatch, messaging or
notification behaviour of any kind.

Shape notes:

* The primary key *is* `patients.id`, the same one-to-one pattern `patients`
  and `doctors` already use against `users`. A second profile for one patient
  is therefore unrepresentable rather than merely discouraged, and no separate
  unique index is needed.
* `ON DELETE CASCADE`, so removing a patient removes their emergency record —
  including the third-party contact details it holds.
* Latitude and longitude are constrained to real ranges and to being written or
  cleared as a pair; half a coordinate points nowhere.

Revision ID: d4a17c2fb8e3
Revises: c7d3ba61e904
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd4a17c2fb8e3'
down_revision: Union[str, None] = 'c7d3ba61e904'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "emergency_profiles",
        sa.Column("id", sa.UUID(), nullable=False),

        # ── emergency contact ────────────────────────────────────────────
        sa.Column("contact_name", sa.String(length=120), nullable=False),
        sa.Column("contact_phone", sa.String(length=20), nullable=False),
        sa.Column("contact_relationship", sa.String(length=60), nullable=False),
        sa.Column("alternate_phone", sa.String(length=20), nullable=True),

        # ── registered address ───────────────────────────────────────────
        sa.Column("house_number", sa.String(length=60), nullable=False),
        sa.Column("street", sa.String(length=150), nullable=False),
        sa.Column("landmark", sa.String(length=150), nullable=True),
        sa.Column("locality", sa.String(length=120), nullable=False),
        sa.Column("city", sa.String(length=100), nullable=False),
        sa.Column("district", sa.String(length=100), nullable=False),
        sa.Column("state", sa.String(length=100), nullable=False),
        sa.Column("country", sa.String(length=100), nullable=False,
                  server_default="India"),
        sa.Column("pincode", sa.String(length=12), nullable=False),

        # ── last known coordinates ───────────────────────────────────────
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("maps_url", sa.String(length=255), nullable=True),
        sa.Column("location_updated_at", sa.DateTime(timezone=True), nullable=True),

        # ── audit columns inherited from the declarative base ────────────
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),

        sa.ForeignKeyConstraint(["id"], ["patients.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "latitude IS NULL OR (latitude >= -90 AND latitude <= 90)",
            name="emergency_profile_latitude_check",
        ),
        sa.CheckConstraint(
            "longitude IS NULL OR (longitude >= -180 AND longitude <= 180)",
            name="emergency_profile_longitude_check",
        ),
        sa.CheckConstraint(
            "(latitude IS NULL) = (longitude IS NULL)",
            name="emergency_profile_coordinate_pair_check",
        ),
    )

    # The base class indexes `id` on every model; kept consistent here even
    # though the primary key already provides one, so a later autogenerate
    # does not see a difference and try to "fix" it.
    op.create_index(
        op.f("ix_emergency_profiles_id"), "emergency_profiles", ["id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_emergency_profiles_id"), table_name="emergency_profiles")
    op.drop_table("emergency_profiles")
