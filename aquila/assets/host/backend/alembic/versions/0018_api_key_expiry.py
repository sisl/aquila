"""Add expires_at to api_keys for temporary keys.

Revision ID: 0018_api_key_expiry
Revises: 0017_api_keys
"""

import sqlalchemy as sa
from alembic import op

revision = "0018_api_key_expiry"
down_revision = "0017_api_keys"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "api_keys",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("api_keys", "expires_at")
