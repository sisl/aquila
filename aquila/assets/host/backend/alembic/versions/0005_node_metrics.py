"""Historical node metric samples for the dashboard charts.

Revision ID: 0005_node_metrics
Revises: 0004_restart_threshold
"""

import sqlalchemy as sa
from alembic import op

revision = "0005_node_metrics"
down_revision = "0004_restart_threshold"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "node_metrics",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "node_id",
            sa.Integer,
            sa.ForeignKey("nodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.Column("gpus", sa.JSON, nullable=True),
        sa.Column("cpu_percent", sa.Float, nullable=True),
        sa.Column("memory_percent", sa.Float, nullable=True),
    )
    op.create_index(
        "ix_node_metrics_node_recorded",
        "node_metrics",
        ["node_id", "recorded_at"],
    )


def downgrade() -> None:
    op.drop_table("node_metrics")
