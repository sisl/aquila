"""Load-phase detail reported by the client while a deployment is loading.

Revision ID: 0003_failure_detail
Revises: 0002_lifecycle
"""

import sqlalchemy as sa
from alembic import op

revision = "0003_failure_detail"
down_revision = "0002_lifecycle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("deployments", sa.Column("detail", sa.String(64), nullable=True))


def downgrade() -> None:
    op.drop_column("deployments", "detail")
