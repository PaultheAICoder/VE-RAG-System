"""Adapter classes that bridge ingestkit-excel backends to VE-RAG's infrastructure.

Each adapter satisfies an ingestkit Protocol interface while injecting VE-RAG's
access-control metadata (tags, document_id, tenant_id) and using VE-RAG's
configured services (Qdrant collection, Ollama models).

The four adapters:
- VERagVectorStoreAdapter: Writes to VE-RAG's Qdrant collection with matching payload schema
- VERagEmbeddingAdapter: Delegates to ingestkit's OllamaEmbedding (same Ollama server)
- VERagLLMAdapter: Delegates to ingestkit's OllamaLLM (same Ollama server)
- ExcelStructuredDB: Delegates to ingestkit's SQLiteStructuredDB (separate DB file)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

if TYPE_CHECKING:
    from ingestkit_core.models import ChunkPayload

logger = logging.getLogger(__name__)


class VERagVectorStoreAdapter:
    """Adapts ingestkit's VectorStoreBackend protocol to VE-RAG's Qdrant collection.

    Merges VE-RAG access-control fields (tags, document_id, tenant_id, etc.) into
    every Qdrant point payload so that existing search filters and lifecycle
    operations (delete_document, update_document_tags) work on ingestkit-written points.

    Uses sync QdrantClient because ingestkit's pipeline is synchronous.
    """

    def __init__(
        self,
        *,
        qdrant_url: str,
        collection_name: str,
        embedding_dimension: int,
        document_id: str,
        document_name: str,
        tags: list[str],
        uploaded_by: str,
        tenant_id: str = "default",
    ) -> None:
        self._client = QdrantClient(url=qdrant_url, timeout=30.0)
        self._collection_name = collection_name
        self._embedding_dimension = embedding_dimension
        self._document_id = document_id
        self._document_name = document_name
        self._tags = tags
        self._uploaded_by = uploaded_by
        self._tenant_id = tenant_id
        self._uploaded_at = datetime.now(UTC).isoformat()

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        """Create the collection if it does not already exist."""
        name = self._resolve_collection(collection)
        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s' (dim=%d)", name, vector_size)

    def upsert_chunks(self, collection: str, chunks: list[ChunkPayload]) -> int:
        """Upsert chunks with VE-RAG's payload schema merged in.

        Each point payload contains:
        - VE-RAG fields: chunk_id, document_id, document_name, chunk_index,
          chunk_text, tags, tenant_id, uploaded_by, uploaded_at, page_number, section
        - ingestkit provenance fields (prefixed with ingestkit_*): source_format,
          ingestion_method, parser_version, ingest_key, sheet_name, etc.
        """
        if not chunks:
            return 0

        name = self._resolve_collection(collection)
        points = []

        for chunk in chunks:
            meta = chunk.metadata
            chunk_index = meta.chunk_index

            payload = {
                # VE-RAG standard fields (must match vector_service.py schema)
                "chunk_id": chunk.id,
                "document_id": self._document_id,
                "document_name": self._document_name,
                "chunk_index": chunk_index,
                "chunk_text": chunk.text,
                "tags": self._tags,
                "tenant_id": self._tenant_id,
                "uploaded_by": self._uploaded_by,
                "uploaded_at": self._uploaded_at,
                "page_number": None,  # Excel files don't have page numbers
                "section": meta.section_title,
                # ingestkit provenance fields
                "ingestkit_source_format": meta.source_format,
                "ingestkit_ingestion_method": meta.ingestion_method,
                "ingestkit_parser_version": meta.parser_version,
                "ingestkit_ingest_key": meta.ingest_key,
                "ingestkit_chunk_hash": meta.chunk_hash,
                "ingestkit_source_uri": meta.source_uri,
                "ingestkit_ingest_run_id": meta.ingest_run_id,
            }

            # Add Excel-specific fields if present
            if meta.table_name:
                payload["ingestkit_table_name"] = meta.table_name
            if meta.row_count is not None:
                payload["ingestkit_row_count"] = meta.row_count
            if meta.columns:
                payload["ingestkit_columns"] = meta.columns

            # Add sheet_name if the metadata has it (ChunkMetadata extends BaseChunkMetadata)
            if hasattr(meta, "sheet_name"):
                payload["ingestkit_sheet_name"] = meta.sheet_name

            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=chunk.vector,
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=name, points=points)
        logger.info(
            "Upserted %d ingestkit chunks for document %s to '%s'",
            len(points),
            self._document_id,
            name,
        )
        return len(points)

    def create_payload_index(self, collection: str, field: str, field_type: str) -> None:
        """Create a payload index on the specified field."""
        type_map = {
            "keyword": PayloadSchemaType.KEYWORD,
            "integer": PayloadSchemaType.INTEGER,
        }
        schema_type = type_map.get(field_type)
        if schema_type is None:
            raise ValueError(f"Unsupported field_type '{field_type}'")

        name = self._resolve_collection(collection)
        self._client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=schema_type,
        )

    def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        """Delete points by their IDs."""
        if not ids:
            return 0

        name = self._resolve_collection(collection)
        self._client.delete(
            collection_name=name,
            points_selector=PointIdsList(points=ids),
        )
        return len(ids)

    def _resolve_collection(self, collection: str) -> str:
        """Use VE-RAG's collection name, ignoring ingestkit's default_collection."""
        return self._collection_name


def create_embedding_adapter(
    *,
    ollama_url: str,
    embedding_model: str,
    embedding_dimension: int,
    backend_timeout: float = 30.0,
):
    """Create an ingestkit OllamaEmbedding that uses VE-RAG's Ollama settings.

    Returns an OllamaEmbedding instance (satisfies EmbeddingBackend protocol).
    """
    from ingestkit_excel.backends.ollama import OllamaEmbedding
    from ingestkit_excel.config import ExcelProcessorConfig

    config = ExcelProcessorConfig(backend_timeout_seconds=backend_timeout)
    return OllamaEmbedding(
        base_url=ollama_url,
        model=embedding_model,
        embedding_dimension=embedding_dimension,
        config=config,
    )


def create_llm_adapter(
    *,
    ollama_url: str,
    backend_timeout: float = 30.0,
):
    """Create an ingestkit OllamaLLM that uses VE-RAG's Ollama settings.

    Returns an OllamaLLM instance (satisfies LLMBackend protocol).
    """
    from ingestkit_excel.backends.ollama import OllamaLLM
    from ingestkit_excel.config import ExcelProcessorConfig

    config = ExcelProcessorConfig(backend_timeout_seconds=backend_timeout)
    return OllamaLLM(base_url=ollama_url, config=config)


def create_structured_db(*, db_path: str):
    """Create an ingestkit SQLiteStructuredDB for Excel table storage.

    Returns an SQLiteStructuredDB instance (satisfies StructuredDBBackend protocol).
    Uses a separate SQLite file from VE-RAG's app database.
    """
    from ingestkit_excel.backends.sqlite import SQLiteStructuredDB

    return SQLiteStructuredDB(db_path=db_path)
