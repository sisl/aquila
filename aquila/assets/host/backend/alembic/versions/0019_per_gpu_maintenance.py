"""Replace per-node maintenance bool with per-GPU maintenance_gpus list.

Revision ID: 0019_per_gpu_maintenance
Revises: 0018_api_key_expiry
"""

import json

import sqlalchemy as sa
from alembic import op

revision = "0019_per_gpu_maintenance"
down_revision = "0018_api_key_expiry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "nodes",
        sa.Column("maintenance_gpus", sa.JSON, nullable=False, server_default="[]"),
    )

    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, gpu_usage FROM nodes WHERE maintenance = true")
    )
    for row in rows:
        gpu_usage = row.gpu_usage
        if isinstance(gpu_usage, str):
            gpu_usage = json.loads(gpu_usage)
        gpu_usage = gpu_usage or []
        indices = sorted(g.get("index", i) for i, g in enumerate(gpu_usage))
        conn.execute(
            sa.text("UPDATE nodes SET maintenance_gpus = :gpus WHERE id = :id"),
            {"gpus": json.dumps(indices), "id": row.id},
        )

    op.drop_column("nodes", "maintenance")


def downgrade() -> None:
    op.add_column(
        "nodes",
        sa.Column("maintenance", sa.Boolean, nullable=False, server_default="false"),
    )

    conn = op.get_bind()
    conn.execute(
        sa.text(
            "UPDATE nodes SET maintenance = true "
            "WHERE maintenance_gpus != '[]' AND maintenance_gpus IS NOT NULL"
        )
    )

    op.drop_column("nodes", "maintenance_gpus")
