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
from typing import Any

import httpx
from sqlalchemy import text

from ai_ready_rag.core.exceptions import EmbeddingError, SearchError
from ai_ready_rag.db.database import SessionLocal
from ai_ready_rag.services.vector_types import SearchResult

logger = logging.getLogger(__name__)


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
                vector_str = f"[{','.join(str(x) for x in embedding)}]"
                with SessionLocal() as db:
                    # Insert text embedding and vector in one statement to avoid
                    # partial-transaction issues: if the CAST fails the whole INSERT
                    # fails atomically rather than leaving an aborted transaction that
                    # rolls back the text INSERT too.
                    try:
                        db.execute(
                            text(
                                "INSERT INTO chunk_vectors "
                                "(id, document_id, chunk_index, chunk_text, metadata_, tenant_id, embedding, vector_embedding) "
                                "VALUES (:id, :doc_id, :idx, :text, :meta, :tenant, :emb, CAST(:vec AS vector))"
                            ),
                            {
                                "id": chunk_id,
                                "doc_id": document_id,
                                "idx": i,
                                "text": chunk_text,
                                "meta": json.dumps(metadata),
                                "tenant": effective_tenant,
                                "emb": embedding_json,
                                "vec": vector_str,
                            },
                        )
                    except Exception:
                        # Fallback: insert without vector_embedding (no pgvector extension)
                        db.rollback()
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
                    db.commit()
                    indexed += 1
            except Exception as exc:
                logger.warning("pgvector.chunk.index_failed", extra={"chunk": i, "error": str(exc)})

        logger.info("pgvector.document.indexed", extra={"doc": document_id, "chunks": indexed})
        return indexed

    async def add_synopsis_chunk(
        self,
        document_id: str,
        document_name: str,
        synopsis_text: str,
        tags: list[str],
        uploaded_by: str,
        tenant_id: str | None = None,
    ) -> None:
        """Insert a synthetic synopsis chunk into chunk_vectors.

        The synopsis is produced by Claude enrichment and contains coverage limits,
        entity names, and key dates that are often absent from raw Docling chunks.
        Indexing it as a separate chunk (chunk_index=9999) makes this information
        retrievable via vector search without disturbing the original chunk set.
        """
        effective_tenant = tenant_id or self._tenant_id
        embedding = await self._embed(synopsis_text)
        embedding_json = json.dumps(embedding)
        vector_str = f"[{','.join(str(x) for x in embedding)}]"
        metadata = {
            "tags": tags,
            "document_name": document_name,
            "uploaded_by": uploaded_by,
            "chunk_type": "synopsis",
        }
        chunk_id = str(uuid.uuid4())
        with SessionLocal() as db:
            # Remove any previous synopsis chunk for this document
            db.execute(
                text(
                    "DELETE FROM chunk_vectors WHERE document_id = :doc_id AND chunk_index = 9999"
                ),
                {"doc_id": document_id},
            )
            try:
                db.execute(
                    text(
                        "INSERT INTO chunk_vectors "
                        "(id, document_id, chunk_index, chunk_text, metadata_, tenant_id, embedding, vector_embedding) "
                        "VALUES (:id, :doc_id, 9999, :text, :meta, :tenant, :emb, CAST(:vec AS vector))"
                    ),
                    {
                        "id": chunk_id,
                        "doc_id": document_id,
                        "text": synopsis_text,
                        "meta": json.dumps(metadata),
                        "tenant": effective_tenant,
                        "emb": embedding_json,
                        "vec": vector_str,
                    },
                )
            except Exception:
                db.rollback()
                db.execute(
                    text(
                        "INSERT INTO chunk_vectors "
                        "(id, document_id, chunk_index, chunk_text, metadata_, tenant_id, embedding) "
                        "VALUES (:id, :doc_id, 9999, :text, :meta, :tenant, :emb)"
                    ),
                    {
                        "id": chunk_id,
                        "doc_id": document_id,
                        "text": synopsis_text,
                        "meta": json.dumps(metadata),
                        "tenant": effective_tenant,
                        "emb": embedding_json,
                    },
                )
            db.commit()
        logger.info("pgvector.synopsis_chunk.indexed", extra={"doc": document_id})

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
            # Note: use CAST(:emb AS vector) instead of :emb::vector to avoid
            # SQLAlchemy treating ::vector as part of the parameter name.
            try:
                rows = db.execute(
                    text(
                        "SELECT id, document_id, chunk_index, chunk_text, metadata_, "
                        "1 - (vector_embedding <=> CAST(:emb AS vector)) AS score "
                        "FROM chunk_vectors "
                        "WHERE tenant_id = :tenant "
                        "  AND vector_embedding IS NOT NULL "
                        "ORDER BY vector_embedding <=> CAST(:emb AS vector) "
                        "LIMIT :limit"
                    ),
                    {"emb": embedding_str, "tenant": tenant_id, "limit": limit * 12},
                ).fetchall()
            except Exception:
                # Fall back to JSON embedding comparison (no pgvector)
                rows = db.execute(
                    text(
                        "SELECT id, document_id, chunk_index, chunk_text, metadata_, 0.5 AS score "
                        "FROM chunk_vectors WHERE tenant_id = :tenant LIMIT :limit"
                    ),
                    {"tenant": tenant_id, "limit": limit * 6},
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

    async def embed(self, text: str) -> list[float]:
        """Public wrapper around _embed() for cache seeding and external callers."""
        return await self._embed(text)

    async def get_extended_stats(self) -> dict[str, Any]:
        """Return per-document chunk counts and totals for admin endpoints."""
        with SessionLocal() as db:
            try:
                # metadata_ is stored as TEXT; cast to JSON to extract fields
                rows = db.execute(
                    text(
                        "SELECT document_id, "
                        "metadata_::json->>'document_name' AS filename, "
                        "COUNT(*) AS chunk_count "
                        "FROM chunk_vectors "
                        "WHERE tenant_id = :tenant "
                        "GROUP BY document_id, metadata_::json->>'document_name'"
                    ),
                    {"tenant": self._tenant_id},
                ).fetchall()
            except Exception:
                # SQLite fallback (used in tests): json_extract syntax
                db.rollback()
                rows = db.execute(
                    text(
                        "SELECT document_id, "
                        "json_extract(metadata_, '$.document_name') AS filename, "
                        "COUNT(*) AS chunk_count "
                        "FROM chunk_vectors "
                        "WHERE tenant_id = :tenant "
                        "GROUP BY document_id, json_extract(metadata_, '$.document_name')"
                    ),
                    {"tenant": self._tenant_id},
                ).fetchall()

        files = [{"document_id": row[0], "filename": row[1], "chunk_count": row[2]} for row in rows]
        return {
            "total_chunks": sum(r["chunk_count"] for r in files),
            "unique_files": len(files),
            "collection_name": "chunk_vectors",
            "collection_size_bytes": None,
            "files": files,
        }

    async def refresh_capabilities(self) -> dict[str, Any]:
        """Return backend capability descriptor (no-op for pgvector)."""
        return {
            "backend": "pgvector",
            "capabilities": ["vector_search", "cosine_similarity", "tag_filtering"],
        }
