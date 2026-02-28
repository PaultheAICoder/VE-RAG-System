"""Add excel_table_registry table

Revision ID: 008
Revises: 007
Create Date: 2026-02-28
"""

import sqlalchemy as sa
from alembic import op

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.create_table(
        "excel_table_registry",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("table_name", sa.String(), nullable=False),
        sa.Column(
            "schema_name",
            sa.String(),
            nullable=False,
            server_default="excel_tables",
        ),
        sa.Column("columns", sa.Text(), nullable=False),  # JSON list of column names
        sa.Column("column_types", sa.Text(), nullable=True),  # JSON dict of col -> dtype
        sa.Column("document_name", sa.String(), nullable=True),
        sa.Column("document_id", sa.String(), nullable=True),
        sa.Column("tenant_id", sa.String(), nullable=False, server_default="default"),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("table_metadata", sa.Text(), nullable=True),  # JSON extras
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_excel_table_registry_table_name",
        "excel_table_registry",
        ["table_name"],
    )
    op.create_index(
        "ix_excel_table_registry_tenant_id",
        "excel_table_registry",
        ["tenant_id"],
    )
    op.create_index(
        "ix_excel_table_registry_schema_table",
        "excel_table_registry",
        ["schema_name", "table_name"],
        unique=True,
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.drop_index("ix_excel_table_registry_schema_table", "excel_table_registry")
    op.drop_index("ix_excel_table_registry_tenant_id", "excel_table_registry")
    op.drop_index("ix_excel_table_registry_table_name", "excel_table_registry")
    op.drop_table("excel_table_registry")
