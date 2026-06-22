"""Per-node container runtime selection (Docker/Podman).

Revision ID: 0014_container_runtimes
Revises: 0013_app_settings
"""

import sqlalchemy as sa
from alembic import op

revision = "0014_container_runtimes"
down_revision = "0013_app_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Per-node override (null = auto: preferred runtime, else whichever exists).
    op.add_column(
        "nodes", sa.Column("container_runtime", sa.String(16), nullable=True)
    )
    # Detected runtimes, synced from the agent's metrics.
    op.add_column(
        "nodes",
        sa.Column(
            "available_runtimes",
            sa.JSON,
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )
    # Which runtime a deployment was launched with (provenance + restarts).
    op.add_column(
        "deployments", sa.Column("container_runtime", sa.String(16), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("deployments", "container_runtime")
    op.drop_column("nodes", "available_runtimes")
    op.drop_column("nodes", "container_runtime")
