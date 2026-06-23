"""Persist last-known read/generation token speeds on deployments.

Revision ID: 0015_persist_token_speeds
Revises: 0014_container_runtimes
"""

import sqlalchemy as sa
from alembic import op

revision = "0015_persist_token_speeds"
down_revision = "0014_container_runtimes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Last-known per-request averages from the client scrape, so the usage
    # stats survive the deployment stopping (live values exist only while
    # the agent reports the deployment as running).
    op.add_column("deployments", sa.Column("prompt_tps", sa.Float(), nullable=True))
    op.add_column(
        "deployments", sa.Column("generation_tps", sa.Float(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("deployments", "generation_tps")
    op.drop_column("deployments", "prompt_tps")
