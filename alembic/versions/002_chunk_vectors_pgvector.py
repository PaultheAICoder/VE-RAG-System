"""Add chunk_vectors table with pgvector for embeddings storage.

Revision ID: 002
Revises: 001
Create Date: 2026-02-27

Creates chunk_vectors table with pgvector embedding column.
Note: pgvector extension must be installed on PostgreSQL before running this migration.
For SQLite deployments this migration is a no-op.
"""

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Detect if we're on SQLite (skip pgvector for SQLite)
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect == "sqlite":
        return  # pgvector not supported on SQLite

    # Ensure pgvector extension is loaded
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Skip if table already exists (e.g. created by create_all on a fresh DB)
    inspector = sa.inspect(bind)
    if "chunk_vectors" in inspector.get_table_names():
        return

    # chunk_vectors — stores embeddings for all document chunks
    op.create_table(
        "chunk_vectors",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column(
            "document_id",
            sa.String,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("chunk_text", sa.Text, nullable=True),
        sa.Column("enriched_text", sa.Text, nullable=True),  # Synopsis-enriched version
        sa.Column(
            "embedding", sa.Text, nullable=True
        ),  # Stored as JSON text (pgvector type added below)
        sa.Column("metadata_", sa.Text, nullable=True),  # JSON metadata (tags, page, section)
        sa.Column("tenant_id", sa.String, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_chunk_vectors_document_id", "chunk_vectors", ["document_id"])
    op.create_index("ix_chunk_vectors_tenant_id", "chunk_vectors", ["tenant_id"])

    # Add pgvector-specific vector column (requires vector extension)
    try:
        op.execute("ALTER TABLE chunk_vectors ADD COLUMN vector_embedding vector(768)")
        # Create IVFFlat index for cosine similarity search
        op.execute(
            "CREATE INDEX ix_chunk_vectors_ivfflat ON chunk_vectors "
            "USING ivfflat (vector_embedding vector_cosine_ops) WITH (lists = 100)"
        )
    except Exception:
        pass  # Vector column may already exist or extension unavailable


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    op.drop_table("chunk_vectors")
