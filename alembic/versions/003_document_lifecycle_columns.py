"""Add lifecycle columns to documents table.

Revision ID: 003
Revises: 002
Create Date: 2026-02-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("documents") as batch_op:
        batch_op.add_column(sa.Column("deleted_at", sa.DateTime, nullable=True))
        batch_op.add_column(sa.Column("valid_from", sa.DateTime, nullable=True))
        batch_op.add_column(sa.Column("valid_to", sa.DateTime, nullable=True))
        batch_op.add_column(sa.Column("idempotency_key", sa.String(255), nullable=True))

    op.create_index("ix_documents_deleted_at", "documents", ["deleted_at"])
    op.create_index("ix_documents_idempotency_key", "documents", ["idempotency_key"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_documents_idempotency_key", table_name="documents")
    op.drop_index("ix_documents_deleted_at", table_name="documents")
    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_column("idempotency_key")
        batch_op.drop_column("valid_to")
        batch_op.drop_column("valid_from")
        batch_op.drop_column("deleted_at")
