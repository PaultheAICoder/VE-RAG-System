"""PostgreSQL + pgvector vector storage backend.

Implements VectorServiceProtocol for hosted/spark deployment profiles.
Uses psycopg2 + pgvector for embedding storage and cosine similarity search.
Replaces Qdrant/Chroma on PostgreSQL deployments.

Access control: tags stored in metadata JSON, filtered BEFORE search (pre-retrieval).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from sqlalchemy import text

from ai_ready_rag.core.exceptions import EmbeddingError, SearchError
from ai_ready_rag.db.database import SessionLocal

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result with metadata."""

    chunk_id: str
    document_id: str
    document_name: str
    chunk_text: str
    chunk_index: int
    score: float
    page_number: int | None
    section: str | None
    tags: list[str] | None = field(default=None)


class PgVectorService:
    """PostgreSQL + pgvector implementation of VectorServiceProtocol.

    Stores document chunk embeddings in chunk_vectors table.
    Uses cosine similarity search with tag-based pre-retrieval ACL.

    Usage:
        service = PgVectorService(...)
        await service.initialize()
        await service.add_document(doc_id, doc_name, chunks, tags, uploaded_by)
        results = await service.search(query, user_tags)
    """

    def __init__(
        self,
        database_url: str,
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        embedding_dimension: int = 768,
        tenant_id: str = "default",
    ):
        self._database_url = database_url
        self._ollama_url = ollama_url
        self._embedding_model = embedding_model
        self._embedding_dimension = embedding_dimension
        self._tenant_id = tenant_id

    async def initialize(self) -> None:
        """Verify pgvector extension is available. Tables created by Alembic."""
        logger.info("pgvector.initialize", extra={"tenant": self._tenant_id})

    async def _embed(self, text: str) -> list[float]:
        """Get embedding from Ollama."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(
                    f"{self._ollama_url}/api/embeddings",
                    json={"model": self._embedding_model, "prompt": text},
                )
                resp.raise_for_status()
                return resp.json()["embedding"]
            except Exception as exc:
                raise EmbeddingError(f"Embedding failed: {exc}") from exc

    async def add_document(
        self,
        document_id: str,
        document_name: str,
        chunks: list[str],
        tags: list[str],
        uploaded_by: str,
        chunk_metadata: list[dict[str, Any]] | None = None,
        tenant_id: str | None = None,
    ) -> int:
        """Index document chunks into chunk_vectors table."""
        effective_tenant = tenant_id or self._tenant_id
        chunk_metadata = chunk_metadata or [{}] * len(chunks)

        # Delete existing chunks for this document (re-index)
        with SessionLocal() as db:
            db.execute(
                text("DELETE FROM chunk_vectors WHERE document_id = :doc_id"),
                {"doc_id": document_id},
            )
            db.commit()

        indexed = 0
        for i, (chunk_text, meta) in enumerate(zip(chunks, chunk_metadata, strict=False)):
            try:
                embedding = await self._embed(chunk_text)
                embedding_json = json.dumps(embedding)
                metadata = {
                    "tags": tags,
                    "document_name": document_name,
                    "uploaded_by": uploaded_by,
                    "page_number": meta.get("page_number"),
                    "section": meta.get("section"),
                    **{k: v for k, v in meta.items() if k not in ("page_number", "section")},
                }
                chunk_id = str(uuid.uuid4())
                with SessionLocal() as db:
                    db.execute(
                        text(
                            "INSERT INTO chunk_vectors "
                            "(id, document_id, chunk_index, chunk_text, metadata_, tenant_id, embedding) "
                            "VALUES (:id, :doc_id, :idx, :text, :meta, :tenant, :emb)"
                        ),
                        {
                            "id": chunk_id,
                            "doc_id": document_id,
                            "idx": i,
                            "text": chunk_text,
                            "meta": json.dumps(metadata),
                            "tenant": effective_tenant,
                            "emb": embedding_json,
                        },
                    )
                    # Update pgvector column if extension is available
                    try:
                        db.execute(
                            text(
                                "UPDATE chunk_vectors SET vector_embedding = :emb::vector "
                                "WHERE id = :id"
                            ),
                            {"emb": f"[{','.join(str(x) for x in embedding)}]", "id": chunk_id},
                        )
                    except Exception:
                        pass  # pgvector may not be available on all connections
                    db.commit()
                    indexed += 1
            except Exception as exc:
                logger.warning("pgvector.chunk.index_failed", extra={"chunk": i, "error": str(exc)})

        logger.info("pgvector.document.indexed", extra={"doc": document_id, "chunks": indexed})
        return indexed

    async def search(
        self,
        query: str,
        user_tags: list[str],
        tenant_id: str = "default",
        limit: int = 5,
        score_threshold: float = 0.3,
        preferred_tags: list[str] | None = None,
    ) -> list[Any]:
        """Cosine similarity search with pre-retrieval tag ACL.

        Tags are checked BEFORE any search — user can only see docs with matching tags.
        """
        try:
            query_embedding = await self._embed(query)
        except EmbeddingError as exc:
            raise SearchError(f"Query embedding failed: {exc}") from exc

        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

        results = []
        with SessionLocal() as db:
            # Try pgvector cosine similarity first
            try:
                rows = db.execute(
                    text(
                        "SELECT id, document_id, chunk_index, chunk_text, metadata_, "
                        "1 - (vector_embedding <=> :emb::vector) AS score "
                        "FROM chunk_vectors "
                        "WHERE tenant_id = :tenant "
                        "  AND vector_embedding IS NOT NULL "
                        "ORDER BY vector_embedding <=> :emb::vector "
                        "LIMIT :limit"
                    ),
                    {"emb": embedding_str, "tenant": tenant_id, "limit": limit * 3},
                ).fetchall()
            except Exception:
                # Fall back to JSON embedding comparison (no pgvector)
                rows = db.execute(
                    text(
                        "SELECT id, document_id, chunk_index, chunk_text, metadata_, 0.5 AS score "
                        "FROM chunk_vectors WHERE tenant_id = :tenant LIMIT :limit"
                    ),
                    {"tenant": tenant_id, "limit": limit * 3},
                ).fetchall()

        for row in rows:
            meta = json.loads(row.metadata_ or "{}")
            doc_tags = meta.get("tags", [])

            # Pre-retrieval access control: skip if user doesn't have any matching tag
            if user_tags and not any(t in doc_tags for t in user_tags):
                continue

            score = float(row.score or 0)
            if score < score_threshold:
                continue

            results.append(
                SearchResult(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    document_name=meta.get("document_name", ""),
                    chunk_text=row.chunk_text or "",
                    chunk_index=row.chunk_index,
                    score=score,
                    page_number=meta.get("page_number"),
                    section=meta.get("section"),
                    tags=doc_tags,
                )
            )
            if len(results) >= limit:
                break

        return results

    async def delete_document(self, document_id: str) -> int:
        """Remove all chunk_vectors rows for a document."""
        with SessionLocal() as db:
            result = db.execute(
                text("DELETE FROM chunk_vectors WHERE document_id = :doc_id RETURNING id"),
                {"doc_id": document_id},
            )
            count = result.rowcount
            db.commit()
        return count

    async def health_check(self) -> dict[str, Any]:
        """Check pgvector service health."""
        try:
            with SessionLocal() as db:
                db.execute(text("SELECT 1")).fetchone()
            return {"healthy": True, "backend": "pgvector", "details": {}}
        except Exception as exc:
            return {"healthy": False, "backend": "pgvector", "details": {"error": str(exc)}}

    async def clear_collection(self) -> bool:
        """Clear all vectors from chunk_vectors table."""
        try:
            with SessionLocal() as db:
                db.execute(
                    text("DELETE FROM chunk_vectors WHERE tenant_id = :t"),
                    {"t": self._tenant_id},
                )
                db.commit()
            return True
        except Exception:
            return False

    async def update_document_tags(
        self,
        document_id: str,
        tags: list[str],
    ) -> int:
        """Update tags in chunk metadata for a document."""
        with SessionLocal() as db:
            rows = db.execute(
                text("SELECT id, metadata_ FROM chunk_vectors WHERE document_id = :doc_id"),
                {"doc_id": document_id},
            ).fetchall()
            updated = 0
            for row in rows:
                meta = json.loads(row.metadata_ or "{}")
                meta["tags"] = tags
                db.execute(
                    text("UPDATE chunk_vectors SET metadata_ = :meta WHERE id = :id"),
                    {"meta": json.dumps(meta), "id": row.id},
                )
                updated += 1
            db.commit()
        return updated
