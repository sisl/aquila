"""Node maintenance (cordon) flag.

Revision ID: 0006_node_maintenance
Revises: 0005_node_metrics
"""

import sqlalchemy as sa
from alembic import op

revision = "0006_node_maintenance"
down_revision = "0005_node_metrics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "nodes",
        sa.Column(
            "maintenance",
            sa.Boolean,
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("nodes", "maintenance")
