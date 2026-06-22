"""Deployment lifecycle: last_error, status_changed_at, atomic port reservation.

Revision ID: 0002_lifecycle
Revises: 0001_baseline
"""

import sqlalchemy as sa
from alembic import op

revision = "0002_lifecycle"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None

_ACTIVE = "('starting', 'loading', 'running', 'stopping')"


def upgrade() -> None:
    op.add_column("deployments", sa.Column("last_error", sa.Text, nullable=True))
    op.add_column(
        "deployments",
        sa.Column("status_changed_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Demote all-but-newest of any duplicate active (node_id, port) rows so
    # the unique reservation index can be created.
    op.execute(
        sa.text(
            "UPDATE deployments SET status = 'error', "
            "last_error = 'Demoted by migration 0002: duplicate active port reservation' "
            "WHERE id IN ("
            "  SELECT id FROM ("
            "    SELECT id, ROW_NUMBER() OVER ("
            "      PARTITION BY node_id, port ORDER BY id DESC"
            "    ) AS rn FROM deployments "
            f"   WHERE status IN {_ACTIVE}"
            "  ) ranked WHERE rn > 1"
            ")"
        )
    )
    op.create_index(
        "uq_active_node_port",
        "deployments",
        ["node_id", "port"],
        unique=True,
        postgresql_where=sa.text(f"status IN {_ACTIVE}"),
        sqlite_where=sa.text(f"status IN {_ACTIVE}"),
    )


def downgrade() -> None:
    op.drop_index("uq_active_node_port", table_name="deployments")
    op.drop_column("deployments", "status_changed_at")
    op.drop_column("deployments", "last_error")
