"""Add the missing cases.diagnosis column.

`ConsultationService.complete_consultation` has always executed
`case.diagnosis = diagnosis`, but `Case` declared no such column. SQLAlchemy
treats an assignment to an unmapped attribute as a plain Python attribute, so
the statement succeeded, the request returned 200, and the value was discarded
at flush — the diagnosis survived only on the generated report and on any
prescription written for the case.

Additive and nullable: existing rows keep a NULL diagnosis, which is the honest
representation of "this consultation was completed before the column existed".

Revision ID: b91d47e3ca08
Revises: a83f1e6c02b4
Create Date: 2026-08-01

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b91d47e3ca08"
down_revision: Union[str, None] = "a83f1e6c02b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("cases", sa.Column("diagnosis", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("cases", "diagnosis")
