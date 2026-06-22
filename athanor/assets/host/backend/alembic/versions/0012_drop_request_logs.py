"""Drop per-request gateway logs; per-deployment scrape metrics suffice.

Revision ID: 0012_drop_request_logs
Revises: 0011_lora_modules
"""

import sqlalchemy as sa
from alembic import op

revision = "0012_drop_request_logs"
down_revision = "0011_lora_modules"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_request_logs_model", table_name="request_logs")
    op.drop_index("ix_request_logs_created_at", table_name="request_logs")
    op.drop_table("request_logs")


def downgrade() -> None:
    op.create_table(
        "request_logs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.Column("deployment_id", sa.Integer, nullable=True),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("owner", sa.String(255), nullable=True),
        sa.Column("endpoint", sa.String(32), nullable=False),
        sa.Column("status_code", sa.Integer, nullable=False),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("prompt_tokens", sa.Integer, nullable=True),
        sa.Column("completion_tokens", sa.Integer, nullable=True),
        sa.Column("streamed", sa.Boolean, server_default=sa.text("false")),
    )
    op.create_index("ix_request_logs_created_at", "request_logs", ["created_at"])
    op.create_index("ix_request_logs_model", "request_logs", ["model"])
