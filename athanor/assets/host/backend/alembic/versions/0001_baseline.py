"""Baseline schema.

Replaces the old create_all + ensure_*_column() startup path. Handles both
fresh databases (creates the full pre-existing schema) and databases created
by earlier releases (adds any column the old ensure_* helpers would have
added, idempotently).

Revision ID: 0001_baseline
Revises:
Create Date: 2026-06-09
"""

import sqlalchemy as sa
from alembic import op

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def _existing_columns(inspector, table: str) -> set[str]:
    return {col["name"] for col in inspector.get_columns(table)}


def _add_missing(inspector, table: str, columns: list[sa.Column]) -> None:
    present = _existing_columns(inspector, table)
    for column in columns:
        if column.name not in present:
            op.add_column(table, column)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "nodes" not in tables:
        op.create_table(
            "nodes",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("hostname", sa.String(255), nullable=False),
            sa.Column("ip_address", sa.String(64), nullable=False),
            sa.Column("port", sa.Integer, nullable=True),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("gpu_usage", sa.JSON, nullable=True),
            sa.Column("default_pip_packages", sa.JSON, nullable=True),
            sa.Column("installed_packages", sa.JSON, nullable=True),
            sa.Column("default_vllm_version", sa.String(255), nullable=True),
            sa.Column(
                "last_heartbeat_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
            ),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
            ),
        )
        op.create_index("ix_nodes_hostname", "nodes", ["hostname"], unique=True)
    else:
        _add_missing(
            inspector,
            "nodes",
            [
                sa.Column("port", sa.Integer, nullable=True),
                sa.Column("gpu_usage", sa.JSON, nullable=True),
                sa.Column("default_pip_packages", sa.JSON, nullable=True),
                sa.Column("installed_packages", sa.JSON, nullable=True),
                sa.Column("default_vllm_version", sa.String(255), nullable=True),
            ],
        )

    if "deployments" not in tables:
        op.create_table(
            "deployments",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("node_id", sa.Integer, sa.ForeignKey("nodes.id"), nullable=False),
            sa.Column("model_name", sa.String(255), nullable=False),
            sa.Column("port", sa.Integer, nullable=False),
            sa.Column("gpu_memory_fraction", sa.Float, nullable=False),
            sa.Column("gpu_ids", sa.JSON, nullable=True),
            sa.Column("tensor_parallel_size", sa.Integer, nullable=True),
            sa.Column("extra_args", sa.JSON, nullable=True),
            sa.Column("env_vars", sa.JSON, nullable=True),
            sa.Column("pip_packages", sa.JSON, nullable=True),
            sa.Column("vllm_version", sa.String(255), nullable=True),
            sa.Column("extra_packages", sa.JSON, nullable=True),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("owner", sa.String(255), nullable=True),
            sa.Column("duration_seconds", sa.Integer, nullable=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
            ),
        )
        op.create_index(
            "ix_deployments_model_name", "deployments", ["model_name"], unique=False
        )
    else:
        _add_missing(
            inspector,
            "deployments",
            [
                sa.Column("gpu_ids", sa.JSON, nullable=True),
                sa.Column("tensor_parallel_size", sa.Integer, nullable=True),
                sa.Column("extra_args", sa.JSON, nullable=True),
                sa.Column("env_vars", sa.JSON, nullable=True),
                sa.Column("pip_packages", sa.JSON, nullable=True),
                sa.Column("vllm_version", sa.String(255), nullable=True),
                sa.Column("extra_packages", sa.JSON, nullable=True),
                sa.Column("owner", sa.String(255), nullable=True),
                sa.Column("duration_seconds", sa.Integer, nullable=True),
                sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            ],
        )

    if "deployment_configs" not in tables:
        op.create_table(
            "deployment_configs",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("payload", sa.JSON, nullable=False),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
            ),
        )
        op.create_index(
            "ix_deployment_configs_name", "deployment_configs", ["name"], unique=True
        )


def downgrade() -> None:
    op.drop_table("deployment_configs")
    op.drop_table("deployments")
    op.drop_table("nodes")
