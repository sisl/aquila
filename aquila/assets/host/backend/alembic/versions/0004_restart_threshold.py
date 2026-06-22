"""Per-deployment crash-loop restart threshold.

Revision ID: 0004_restart_threshold
Revises: 0003_failure_detail
"""

import sqlalchemy as sa
from alembic import op

revision = "0004_restart_threshold"
down_revision = "0003_failure_detail"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "deployments", sa.Column("max_failed_restarts", sa.Integer, nullable=True)
    )


def downgrade() -> None:
    op.drop_column("deployments", "max_failed_restarts")
