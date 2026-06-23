"""Usage accounting: gateway request logs + per-deployment token counters.

Revision ID: 0010_usage_accounting
Revises: 0009_provenance
"""

import sqlalchemy as sa
from alembic import op

revision = "0010_usage_accounting"
down_revision = "0009_provenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "request_logs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        # No FK: deployments are deletable, logs outlive them.
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

    op.add_column(
        "deployments",
        sa.Column(
            "total_prompt_tokens", sa.BigInteger, nullable=False, server_default="0"
        ),
    )
    op.add_column(
        "deployments",
        sa.Column(
            "total_completion_tokens", sa.BigInteger, nullable=False, server_default="0"
        ),
    )
    op.add_column(
        "deployments",
        sa.Column("total_requests", sa.BigInteger, nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("deployments", "total_requests")
    op.drop_column("deployments", "total_completion_tokens")
    op.drop_column("deployments", "total_prompt_tokens")
    op.drop_index("ix_request_logs_model", table_name="request_logs")
    op.drop_index("ix_request_logs_created_at", table_name="request_logs")
    op.drop_table("request_logs")
