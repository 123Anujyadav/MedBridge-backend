"""link reports to the consultation case they were issued from

Adds a nullable `reports.case_id`. The AI-assisted clinical report workflow
issues a report directly from a case, and without this column there was no way
to trace an issued report back to the case (and therefore back to the AI intake
session) that produced it.

Nullable by design: lab results and imaging reports are ingested without a
case. `ON DELETE SET NULL` keeps the report — part of the patient's permanent
record — alive if the case row is ever removed.

Revision ID: b7f2c9d41a58
Revises: 6351e60135f1
Create Date: 2026-07-26 12:04:18.220117

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7f2c9d41a58'
down_revision: Union[str, None] = '6351e60135f1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'reports',
        sa.Column('case_id', sa.UUID(), nullable=True),
    )
    op.create_index(
        op.f('ix_reports_case_id'), 'reports', ['case_id'], unique=False
    )
    op.create_foreign_key(
        'fk_reports_case_id_cases',
        'reports',
        'cases',
        ['case_id'],
        ['id'],
        ondelete='SET NULL',
    )


def downgrade() -> None:
    op.drop_constraint('fk_reports_case_id_cases', 'reports', type_='foreignkey')
    op.drop_index(op.f('ix_reports_case_id'), table_name='reports')
    op.drop_column('reports', 'case_id')
