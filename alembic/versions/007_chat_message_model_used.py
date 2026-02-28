"""Add model_used column to chat_messages.

Revision ID: 007
Revises: 006
Create Date: 2026-02-28
"""

import sqlalchemy as sa
from alembic import op

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        # SQLite: add the column (nullable, no default needed)
        op.add_column("chat_messages", sa.Column("model_used", sa.String(), nullable=True))
    else:
        op.add_column("chat_messages", sa.Column("model_used", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_messages", "model_used")
