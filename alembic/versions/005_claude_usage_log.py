"""Create claude_usage_log table for cost tracking.

Revision ID: 005
Revises: 004
Create Date: 2026-02-28

Creates:
- claude_usage_log  — per-call Claude API usage log for cost cap enforcement
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: str | None = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "claude_usage_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("operation", sa.String(), nullable=False),
        sa.Column("document_id", sa.String(), nullable=True),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("tenant_id", sa.String(), nullable=False, server_default="default"),
        sa.Column(
            "recorded_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_claude_usage_log_tenant_recorded",
        "claude_usage_log",
        ["tenant_id", "recorded_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_claude_usage_log_tenant_recorded", table_name="claude_usage_log")
    op.drop_table("claude_usage_log")
