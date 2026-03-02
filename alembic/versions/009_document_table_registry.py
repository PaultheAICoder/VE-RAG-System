"""Add document_table_registry table

Revision ID: 009
Revises: 008
Create Date: 2026-03-01
"""

import sqlalchemy as sa
from alembic import op

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    # Skip if table already exists (e.g. created by create_all on a fresh DB)
    inspector = sa.inspect(bind)
    if "document_table_registry" in inspector.get_table_names():
        return

    op.create_table(
        "document_table_registry",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False, server_default="default"),
        sa.Column("table_name", sa.String(), nullable=False),
        sa.Column(
            "schema_name",
            sa.String(),
            nullable=False,
            server_default="document_tables",
        ),
        sa.Column("source_format", sa.String(), nullable=False),
        sa.Column("source_page", sa.Integer(), nullable=True),
        sa.Column("table_index", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("columns", sa.Text(), nullable=False),
        sa.Column("column_types", sa.Text(), nullable=True),
        sa.Column("row_value_samples", sa.Text(), nullable=True),
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("document_name", sa.String(), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("table_metadata", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_dtr_tenant_id",
        "document_table_registry",
        ["tenant_id"],
    )
    op.create_index(
        "ix_dtr_document_id",
        "document_table_registry",
        ["document_id"],
    )
    op.create_index(
        "ix_dtr_source_format",
        "document_table_registry",
        ["source_format"],
    )
    op.create_index(
        "ix_dtr_tenant_schema_table",
        "document_table_registry",
        ["tenant_id", "schema_name", "table_name"],
        unique=True,
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.drop_index("ix_dtr_tenant_schema_table", "document_table_registry")
    op.drop_index("ix_dtr_source_format", "document_table_registry")
    op.drop_index("ix_dtr_document_id", "document_table_registry")
    op.drop_index("ix_dtr_tenant_id", "document_table_registry")
    op.drop_table("document_table_registry")
