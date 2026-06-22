"""Structured vLLM engine arguments.

Revision ID: 0008_engine_args
Revises: 0007_node_disk_usage
"""

import sqlalchemy as sa
from alembic import op

revision = "0008_engine_args"
down_revision = "0007_node_disk_usage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("deployments", sa.Column("engine_args", sa.JSON, nullable=True))


def downgrade() -> None:
    op.drop_column("deployments", "engine_args")
