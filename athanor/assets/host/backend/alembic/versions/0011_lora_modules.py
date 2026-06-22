"""LoRA adapters served alongside the base model.

Revision ID: 0011_lora_modules
Revises: 0010_usage_accounting
"""

import sqlalchemy as sa
from alembic import op

revision = "0011_lora_modules"
down_revision = "0010_usage_accounting"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("deployments", sa.Column("lora_modules", sa.JSON, nullable=True))


def downgrade() -> None:
    op.drop_column("deployments", "lora_modules")
