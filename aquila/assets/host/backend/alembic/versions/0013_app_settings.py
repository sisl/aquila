"""Global settings overrides, editable from the dashboard.

Revision ID: 0013_app_settings
Revises: 0012_drop_request_logs
"""

import sqlalchemy as sa
from alembic import op

revision = "0013_app_settings"
down_revision = "0012_drop_request_logs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "app_settings",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("value", sa.JSON, nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
    )


def downgrade() -> None:
    op.drop_table("app_settings")
