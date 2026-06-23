"""Warm-cache (pause/resume) columns: per-node policy + per-deployment pin.

Revision ID: 0016_warm_cache
Revises: 0015_persist_token_speeds
"""

import sqlalchemy as sa
from alembic import op

revision = "0016_warm_cache"
down_revision = "0015_persist_token_speeds"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Per-node warm-offload policy: opt-in toggle + RAM-cache budget.
    op.add_column(
        "nodes",
        sa.Column(
            "warm_offload_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "nodes", sa.Column("ram_cache_limit_mb", sa.Integer(), nullable=True)
    )
    # Per-deployment pin: protect a model from automatic eviction.
    op.add_column(
        "deployments",
        sa.Column(
            "pinned",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    op.drop_column("deployments", "pinned")
    op.drop_column("nodes", "ram_cache_limit_mb")
    op.drop_column("nodes", "warm_offload_enabled")
