"""Node disk usage as reported by the client agent.

Revision ID: 0007_node_disk_usage
Revises: 0006_node_maintenance
"""

import sqlalchemy as sa
from alembic import op

revision = "0007_node_disk_usage"
down_revision = "0006_node_maintenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("nodes", sa.Column("disk_usage", sa.JSON, nullable=True))


def downgrade() -> None:
    op.drop_column("nodes", "disk_usage")
