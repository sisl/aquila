"""Provenance: exact image digest per deployment.

Revision ID: 0009_provenance
Revises: 0008_engine_args
"""

import sqlalchemy as sa
from alembic import op

revision = "0009_provenance"
down_revision = "0008_engine_args"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("deployments", sa.Column("image_digest", sa.String(512), nullable=True))


def downgrade() -> None:
    op.drop_column("deployments", "image_digest")
