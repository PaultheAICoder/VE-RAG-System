"""Vector Service for semantic search and document indexing.

Provides an abstraction over Qdrant (vector database) and Ollama (embeddings).
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from ai_ready_rag.core.exceptions import EmbeddingError, IndexingError, SearchError
from ai_ready_rag.services.vector_utils import generate_chunk_id

logger = logging.getLogger(__name__)

# Retry configuration
RETRY_CONFIG = {
    "max_attempts": 3,
    "base_delay_seconds": 1.0,
    "exponential_base": 2.0,
}

# Timeouts
QDRANT_TIMEOUT_SECONDS = 5
OLLAMA_TIMEOUT_SECONDS = 60


@dataclass
class HealthStatus:
    """Health check response with detailed status."""

    qdrant_healthy: bool
    qdrant_latency_ms: float | None
    ollama_healthy: bool
    ollama_latency_ms: float | None
    collection_exists: bool
    collection_vector_count: int | None

    @property
    def healthy(self) -> bool:
        """Overall health - all components must be healthy."""
        return self.qdrant_healthy and self.ollama_healthy and self.collection_exists


@dataclass
class IndexResult:
    """Result of document indexing operation."""

    document_id: str
    chunks_indexed: int
    replaced_existing: bool
    embedding_time_ms: float
    indexing_time_ms: float


@dataclass
class SearchResult:
    """Single search result with metadata."""

    chunk_id: str
    document_id: str
    document_name: str
    chunk_text: str
    chunk_index: int
    score: float  # 0.0 to 1.0 (cosine similarity)
    page_number: int | None
    section: str | None


@dataclass
class CollectionStats:
    """Collection statistics for monitoring."""

    total_chunks: int
    total_documents: int
    collection_size_bytes: int
    tenant_id: str


class VectorService:
    """Handles vector storage and semantic search operations.

    Thread-safe and async-compatible. Abstracts Qdrant for vector storage
    and Ollama for embedding generation.

    Example:
        >>> service = VectorService()
        >>> await service.initialize()
        >>> health = await service.health_check()
        >>> if health.healthy:
        ...     embedding = await service.embed("Hello world")
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        ollama_url: str = "http://localhost:11434",
        collection_name: str = "documents",
        embedding_model: str = "nomic-embed-text",
        embedding_dimension: int = 768,
        max_tokens: int = 8192,
        tenant_id: str = "default",
    ):
        """Initialize vector service with connection parameters.

        Args:
            qdrant_url: Qdrant server URL.
            ollama_url: Ollama server URL.
            collection_name: Name of the Qdrant collection.
            embedding_model: Ollama model for embeddings.
            embedding_dimension: Vector dimension (must match model).
            max_tokens: Maximum tokens before truncation.
            tenant_id: Default tenant identifier.
        """
        self.qdrant_url = qdrant_url
        self.ollama_url = ollama_url
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.max_tokens = max_tokens
        self.tenant_id = tenant_id

        # Initialize async Qdrant client
        # Note: check_compatibility=False to allow minor version mismatches
        self._qdrant = AsyncQdrantClient(url=qdrant_url, check_compatibility=False)

    async def initialize(self) -> None:
        """Create collection if not exists.

        Called on application startup. Idempotent - safe to call multiple times.
        Creates payload indexes for tags, document_id, and tenant_id.
        """
        # Check if collection exists
        collections = await self._qdrant.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            await self._qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
            )

            # Create payload indexes for efficient filtering
            await self._qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="tags",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await self._qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await self._qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="tenant_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Collection {self.collection_name} created with indexes")
        else:
            logger.debug(f"Collection {self.collection_name} already exists")

    async def health_check(self) -> HealthStatus:
        """Check connectivity to Qdrant and Ollama.

        Returns:
            HealthStatus with component status and latencies.

        Note:
            This method never raises exceptions - failures are reported
            in the HealthStatus object.
        """
        # Check Qdrant
        qdrant_healthy = False
        qdrant_latency_ms = None
        collection_exists = False
        collection_vector_count = None

        try:
            start = time.perf_counter()
            collections = await asyncio.wait_for(
                self._qdrant.get_collections(),
                timeout=QDRANT_TIMEOUT_SECONDS,
            )
            qdrant_latency_ms = (time.perf_counter() - start) * 1000
            qdrant_healthy = True

            # Check collection
            collection_names = [c.name for c in collections.collections]
            collection_exists = self.collection_name in collection_names

            if collection_exists:
                info = await self._qdrant.get_collection(self.collection_name)
                collection_vector_count = info.points_count

        except TimeoutError:
            logger.warning(f"Qdrant health check timed out after {QDRANT_TIMEOUT_SECONDS}s")
        except (ResponseHandlingException, Exception) as e:
            logger.warning(f"Qdrant health check failed: {e}")

        # Check Ollama
        ollama_healthy = False
        ollama_latency_ms = None

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                response.raise_for_status()
            ollama_latency_ms = (time.perf_counter() - start) * 1000
            ollama_healthy = True

        except httpx.TimeoutException:
            logger.warning(f"Ollama health check timed out after {OLLAMA_TIMEOUT_SECONDS}s")
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")

        return HealthStatus(
            qdrant_healthy=qdrant_healthy,
            qdrant_latency_ms=qdrant_latency_ms,
            ollama_healthy=ollama_healthy,
            ollama_latency_ms=ollama_latency_ms,
            collection_exists=collection_exists,
            collection_vector_count=collection_vector_count,
        )

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text.

        Args:
            text: Input text to embed.

        Returns:
            768-dimensional float vector (normalized, suitable for cosine similarity).

        Raises:
            EmbeddingError: If Ollama is unavailable or model not found.

        Note:
            If input exceeds max_tokens, text is truncated from the END
            and a warning is logged.
        """
        # Simple character-based truncation approximation
        # Average ~4 chars per token for English text
        max_chars = self.max_tokens * 4

        if len(text) > max_chars:
            original_len = len(text)
            text = text[:max_chars]
            logger.warning(
                f"Text truncated from {original_len} to {len(text)} chars "
                f"(max_tokens: {self.max_tokens})"
            )

        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding")

                if not embedding:
                    raise EmbeddingError(f"No embedding in response: {data}")

                return embedding

        except httpx.TimeoutException as e:
            raise EmbeddingError(f"Ollama request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(f"Ollama HTTP error: {e}") from e
        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(f"Embedding generation failed: {e}") from e

    async def embed_batch(
        self,
        texts: list[str],
        max_batch_size: int = 100,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts.
            max_batch_size: Maximum texts per batch (default: 100).

        Returns:
            List of 768-dimensional vectors (same order as input).

        Raises:
            EmbeddingError: If embedding fails after retries.

        Note:
            - Texts are processed in batches
            - Failed batches are retried with exponential backoff
            - If a batch fails after retries, raises EmbeddingError
        """
        if not texts:
            return []

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i : i + max_batch_size]
            batch_embeddings = await self._embed_batch_with_retry(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    async def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts with retry logic.

        Args:
            texts: Batch of texts to embed.

        Returns:
            List of embeddings.

        Raises:
            EmbeddingError: If all retries fail.
        """
        last_error = None

        for attempt in range(RETRY_CONFIG["max_attempts"]):
            try:
                # Embed each text in the batch
                # Note: Ollama doesn't have a native batch API, so we parallelize
                tasks = [self.embed(text) for text in texts]
                return await asyncio.gather(*tasks)

            except EmbeddingError as e:
                last_error = e
                if attempt < RETRY_CONFIG["max_attempts"] - 1:
                    delay = RETRY_CONFIG["base_delay_seconds"] * (
                        RETRY_CONFIG["exponential_base"] ** attempt
                    )
                    logger.warning(
                        f"Batch embedding failed (attempt {attempt + 1}), retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)

        raise EmbeddingError(
            f"Batch embedding failed after {RETRY_CONFIG['max_attempts']} attempts: {last_error}"
        )

    async def add_document(
        self,
        document_id: str,
        document_name: str,
        chunks: list[str],
        tags: list[str],
        uploaded_by: str,
        chunk_metadata: list[dict] | None = None,
        tenant_id: str | None = None,
    ) -> IndexResult:
        """Index a document's chunks to the vector store.

        Args:
            document_id: Unique document identifier (UUID).
            document_name: Original filename for display.
            chunks: List of text chunks (pre-chunked by caller).
            tags: Access control tags (must be pre-validated).
            uploaded_by: User ID of uploader.
            chunk_metadata: Optional per-chunk metadata (page_number, section).
            tenant_id: Tenant identifier (defaults to service's tenant_id).

        Returns:
            IndexResult with chunk count and timing info.

        Raises:
            ValueError: If tags list is empty.
            IndexingError: If operation fails after retries.
        """
        if not tags:
            raise ValueError("At least one tag is required")

        if not chunks:
            raise ValueError("At least one chunk is required")

        tenant = tenant_id or self.tenant_id
        chunk_metadata = chunk_metadata or []
        uploaded_at = datetime.now(UTC).isoformat()

        # Check if document already exists
        existing_count = await self._count_document_chunks(document_id)
        replaced_existing = existing_count > 0

        # Delete existing chunks if re-indexing
        if replaced_existing:
            logger.info(f"Replacing {existing_count} existing chunks for document {document_id}")
            await self._delete_document_chunks(document_id)

        # Generate embeddings
        embed_start = time.perf_counter()
        try:
            embeddings = await self.embed_batch(chunks)
        except EmbeddingError as e:
            raise IndexingError(f"Failed to generate embeddings: {e}") from e
        embedding_time_ms = (time.perf_counter() - embed_start) * 1000

        # Build points for Qdrant
        points = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            chunk_id = generate_chunk_id(document_id, i)

            # Get metadata for this chunk
            meta = chunk_metadata[i] if i < len(chunk_metadata) else {}

            payload = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "document_name": document_name,
                "chunk_index": i,
                "chunk_text": chunk_text,
                "tags": tags,
                "tenant_id": tenant,
                "uploaded_by": uploaded_by,
                "uploaded_at": uploaded_at,
                "page_number": meta.get("page_number"),
                "section": meta.get("section"),
            }

            points.append(
                models.PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert to Qdrant
        index_start = time.perf_counter()
        try:
            await self._qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        except Exception as e:
            # Rollback: delete any chunks that might have been inserted
            logger.error(f"Indexing failed, attempting rollback: {e}")
            await self._delete_document_chunks(document_id)
            raise IndexingError(f"Failed to index document: {e}") from e

        indexing_time_ms = (time.perf_counter() - index_start) * 1000

        logger.info(
            f"Indexed {len(chunks)} chunks for document {document_id} "
            f"(embed: {embedding_time_ms:.1f}ms, index: {indexing_time_ms:.1f}ms)"
        )

        return IndexResult(
            document_id=document_id,
            chunks_indexed=len(chunks),
            replaced_existing=replaced_existing,
            embedding_time_ms=embedding_time_ms,
            indexing_time_ms=indexing_time_ms,
        )

    async def delete_document(self, document_id: str) -> bool:
        """Remove all chunks for a document.

        Args:
            document_id: Document to delete.

        Returns:
            True if deletion was successful (regardless of whether document existed).
            False if deletion failed due to Qdrant error.
        """
        try:
            await self._delete_document_chunks(document_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def update_document_tags(self, document_id: str, tag_names: list[str]) -> int:
        """Update tags payload for all vectors of a document.

        Uses Qdrant's set_payload API for efficient payload-only updates
        without re-embedding.

        Args:
            document_id: Document whose vectors to update.
            tag_names: New list of tag names.

        Returns:
            Number of points updated (0 if document not found).

        Raises:
            Exception: If Qdrant operation fails.
        """
        try:
            # Use set_payload with filter to update all points for this document
            await self._qdrant.set_payload(
                collection_name=self.collection_name,
                payload={"tags": tag_names},
                points=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
            logger.info(f"Updated tags for document {document_id}: {tag_names}")
            # set_payload doesn't return count, so we count separately
            count = await self._count_document_chunks(document_id)
            return count
        except Exception as e:
            logger.error(f"Failed to update tags for document {document_id}: {e}")
            raise

    async def search(
        self,
        query: str,
        user_tags: list[str] | None,
        limit: int = 5,
        score_threshold: float = 0.0,
        tenant_id: str | None = None,
    ) -> list[SearchResult]:
        """Semantic search with access control filtering.

        Args:
            query: Natural language query.
            user_tags: Tags the user has access to. None = no tag filtering
                (system admin bypass), empty list = public only.
            limit: Maximum results to return (1-100, default: 5).
            score_threshold: Minimum similarity score (0.0-1.0, default: 0.0).
            tenant_id: Tenant to search within (defaults to service's tenant_id).

        Returns:
            List of SearchResult, ordered by relevance (highest score first).

        Raises:
            SearchError: If search fails.

        Note:
            Access control filter is applied BEFORE vector search (pre-retrieval).
            - user_tags=None: No tag filtering (system admin cache warming)
            - user_tags=[]: Only documents with "public" tag
            - user_tags=["hr"]: Public + hr documents
        """
        tenant = tenant_id or self.tenant_id

        # Generate query embedding
        try:
            query_embedding = await self.embed(query)
        except EmbeddingError as e:
            raise SearchError(f"Failed to embed query: {e}") from e

        # Build access control filter
        if user_tags is None:
            # System admin: no tag filtering, tenant only
            access_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=tenant),
                    ),
                ],
            )
        else:
            access_filter = self._build_access_filter(user_tags, tenant)

        # Search Qdrant using query_points (newer API)
        try:
            response = await self._qdrant.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=access_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as e:
            raise SearchError(f"Qdrant search failed: {e}") from e

        # Convert to SearchResult objects
        search_results = []
        for point in response.points:
            payload = point.payload or {}
            search_results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", str(point.id)),
                    document_id=payload.get("document_id", ""),
                    document_name=payload.get("document_name", ""),
                    chunk_text=payload.get("chunk_text", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    score=point.score,
                    page_number=payload.get("page_number"),
                    section=payload.get("section"),
                )
            )

        return search_results

    def _build_access_filter(self, user_tags: list[str], tenant_id: str) -> models.Filter:
        """Build Qdrant filter for access-controlled search.

        Logic:
            (tenant_id matches) AND (
                "public" in tags
                OR any(user_tags) in tags
            )
        """
        # Tag conditions: public OR any user tag
        tag_conditions = [
            models.FieldCondition(
                key="tags",
                match=models.MatchValue(value="public"),
            )
        ]

        if user_tags:
            tag_conditions.append(
                models.FieldCondition(
                    key="tags",
                    match=models.MatchAny(any=user_tags),
                )
            )

        # Combine tenant filter with tag filter using nested must
        # Structure: must[tenant_id] AND must[should[public OR user_tags]]
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                ),
                # Nested filter: at least one tag condition must match
                models.Filter(should=tag_conditions),
            ],
        )

    async def _count_document_chunks(self, document_id: str) -> int:
        """Count chunks for a document."""
        try:
            result = await self._qdrant.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
            )
            return result.count
        except Exception:
            return 0

    async def _delete_document_chunks(self, document_id: str) -> None:
        """Delete all chunks for a document."""
        await self._qdrant.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    async def get_stats(self) -> CollectionStats:
        """Get collection statistics.

        Returns:
            CollectionStats with counts and size information.

        Note:
            Returns zeros on error rather than raising exceptions.
            Counts unique documents by scrolling collection (may be slow on large collections).
        """
        try:
            # Get collection info
            collection_info = await self._qdrant.get_collection(self.collection_name)

            # Total chunks = points_count
            total_chunks = collection_info.points_count or 0

            # Collection size from storage info
            collection_size_bytes = 0
            if (
                hasattr(collection_info, "payload_storage_size")
                and collection_info.payload_storage_size
            ):
                collection_size_bytes += collection_info.payload_storage_size
            if (
                hasattr(collection_info, "vectors_storage_size")
                and collection_info.vectors_storage_size
            ):
                collection_size_bytes += collection_info.vectors_storage_size

            # Count unique document_ids by scrolling
            unique_doc_ids: set[str] = set()
            offset = None

            while True:
                results, offset = await self._qdrant.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=["document_id"],
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="tenant_id",
                                match=models.MatchValue(value=self.tenant_id),
                            )
                        ]
                    ),
                )

                if not results:
                    break

                for point in results:
                    if point.payload and "document_id" in point.payload:
                        unique_doc_ids.add(point.payload["document_id"])

                if offset is None:
                    break

            return CollectionStats(
                total_chunks=total_chunks,
                total_documents=len(unique_doc_ids),
                collection_size_bytes=collection_size_bytes,
                tenant_id=self.tenant_id,
            )

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return CollectionStats(
                total_chunks=0,
                total_documents=0,
                collection_size_bytes=0,
                tenant_id=self.tenant_id,
            )

    async def get_extended_stats(self) -> dict:
        """Get extended collection statistics including file details.

        Returns:
            dict with:
            - total_chunks: int
            - unique_files: int
            - collection_name: str
            - collection_size_bytes: int
            - files: list[dict] with document_id, filename, chunk_count
        """
        try:
            # Get collection info
            collection_info = await self._qdrant.get_collection(self.collection_name)

            total_chunks = collection_info.points_count or 0

            # Collection size from storage info
            collection_size_bytes = 0
            if (
                hasattr(collection_info, "payload_storage_size")
                and collection_info.payload_storage_size
            ):
                collection_size_bytes += collection_info.payload_storage_size
            if (
                hasattr(collection_info, "vectors_storage_size")
                and collection_info.vectors_storage_size
            ):
                collection_size_bytes += collection_info.vectors_storage_size

            # Scroll collection to gather file details
            # Aggregate by document_id: count chunks, capture filename
            file_stats: dict[str, dict] = {}
            offset = None

            while True:
                results, offset = await self._qdrant.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=["document_id", "document_name"],
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="tenant_id",
                                match=models.MatchValue(value=self.tenant_id),
                            )
                        ]
                    ),
                )

                if not results:
                    break

                for point in results:
                    if point.payload:
                        doc_id = point.payload.get("document_id", "")
                        doc_name = point.payload.get("document_name", "")
                        if doc_id:
                            if doc_id not in file_stats:
                                file_stats[doc_id] = {
                                    "document_id": doc_id,
                                    "filename": doc_name,
                                    "chunk_count": 0,
                                }
                            file_stats[doc_id]["chunk_count"] += 1

                if offset is None:
                    break

            files_list = list(file_stats.values())

            return {
                "total_chunks": total_chunks,
                "unique_files": len(files_list),
                "collection_name": self.collection_name,
                "collection_size_bytes": collection_size_bytes,
                "files": files_list,
            }

        except Exception as e:
            logger.error(f"Failed to get extended collection stats: {e}")
            return {
                "total_chunks": 0,
                "unique_files": 0,
                "collection_name": self.collection_name,
                "collection_size_bytes": 0,
                "files": [],
            }

    async def clear_collection(self) -> bool:
        """Delete all vectors in collection for this tenant.

        Returns:
            True if successful, False if failed.

        Warning:
            This is a destructive operation. Primarily intended for testing.
            Does NOT delete the collection itself, only its contents.
        """
        logger.warning(
            f"DESTRUCTIVE OPERATION: Clearing all vectors from collection "
            f"'{self.collection_name}' (tenant: {self.tenant_id})"
        )

        try:
            # Delete all points matching this tenant
            await self._qdrant.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="tenant_id",
                                match=models.MatchValue(value=self.tenant_id),
                            )
                        ]
                    )
                ),
            )
            logger.info(f"Collection cleared successfully for tenant: {self.tenant_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
