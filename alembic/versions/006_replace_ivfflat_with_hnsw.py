"""Replace IVFFlat index with HNSW for small-dataset compatibility.

Revision ID: 006
Revises: 005
Create Date: 2026-02-28

IVFFlat with lists=100 returns 0 results when fewer than ~100 rows exist
because it probes only 1 cluster (the default) and the index is poorly
initialized. HNSW works correctly for any dataset size and has better
recall in practice.

SQLite: no-op (no pgvector).
"""

from alembic import op

revision: str = "006"
down_revision: str | None = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return

    # Drop the broken IVFFlat index
    op.execute("DROP INDEX IF EXISTS ix_chunk_vectors_ivfflat")

    # Create HNSW index — works for any number of rows, better recall
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunk_vectors_hnsw ON chunk_vectors "
        "USING hnsw (vector_embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return

    op.execute("DROP INDEX IF EXISTS ix_chunk_vectors_hnsw")
    op.execute(
        "CREATE INDEX ix_chunk_vectors_ivfflat ON chunk_vectors "
        "USING ivfflat (vector_embedding vector_cosine_ops) WITH (lists = 100)"
    )
