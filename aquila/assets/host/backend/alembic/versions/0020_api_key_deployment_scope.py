"""Add per-deployment scoping to API keys.

Revision ID: 0020_api_key_deployment_scope
Revises: 0019_per_gpu_maintenance
"""

import sqlalchemy as sa
from alembic import op

revision = "0020_api_key_deployment_scope"
down_revision = "0019_per_gpu_maintenance"


def upgrade() -> None:
    op.add_column(
        "api_keys",
        sa.Column("allowed_deployment_ids", sa.JSON, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("api_keys", "allowed_deployment_ids")
