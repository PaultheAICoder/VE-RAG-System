"""Vector Service for semantic search and document indexing.

Provides an abstraction over Qdrant (vector database) and Ollama (embeddings).
"""

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

try:
    from fastembed import SparseTextEmbedding

    FASTEMBED_AVAILABLE = True
except ImportError:
    SparseTextEmbedding = None
    FASTEMBED_AVAILABLE = False
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import Fusion, FusionQuery, Prefetch

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
    sparse_indexed: bool = True  # Default True for backward compat


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

        # Sparse embedding state (thread-safe lazy loading)
        self._sparse_model: SparseTextEmbedding | None = None
        self._sparse_available: bool = FASTEMBED_AVAILABLE
        self._sparse_lock = threading.Lock()

        # Hybrid search capability detection
        self._collection_has_sparse: bool = False
        self._capabilities_checked_at: datetime | None = None

    async def _detect_collection_capabilities(self) -> None:
        """Check collection for sparse vector support (named vector 'sparse')."""
        try:
            info = await self._qdrant.get_collection(self.collection_name)
            sparse_vectors = getattr(info.config.params, "sparse_vectors", None)
            self._collection_has_sparse = sparse_vectors is not None and "sparse" in sparse_vectors
            self._capabilities_checked_at = datetime.now(UTC)
            logger.info(f"Collection capabilities detected: sparse={self._collection_has_sparse}")
        except Exception as e:
            logger.warning(f"Failed to detect collection capabilities: {e}")
            self._collection_has_sparse = False

    @property
    def hybrid_enabled(self) -> bool:
        """Whether hybrid search is enabled via admin settings."""
        from ai_ready_rag.services.settings_service import get_rag_setting

        return bool(get_rag_setting("retrieval_hybrid_enabled", False))

    @property
    def min_similarity_score(self) -> float:
        """Get the appropriate score threshold based on active search mode."""
        from ai_ready_rag.services.settings_service import get_rag_setting

        if self.hybrid_enabled and self._collection_has_sparse:
            return float(get_rag_setting("retrieval_min_score_hybrid", 0.05))
        return float(get_rag_setting("retrieval_min_score_dense", 0.3))

    @property
    def prefetch_multiplier(self) -> int:
        """Multiplier for prefetch limit in hybrid search."""
        from ai_ready_rag.services.settings_service import get_rag_setting

        return int(get_rag_setting("retrieval_prefetch_multiplier", 3))

    def _get_sparse_model(self) -> SparseTextEmbedding | None:
        """Thread-safe lazy-load of sparse embedding model.

        Uses double-checked locking pattern for thread safety.
        Returns None if model fails to load (graceful degradation).
        """
        if not self._sparse_available:
            return None
        if self._sparse_model is not None:
            return self._sparse_model
        with self._sparse_lock:
            if self._sparse_model is not None:
                return self._sparse_model
            try:
                self._sparse_model = SparseTextEmbedding(
                    model_name="Qdrant/bm25",
                    cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"),
                )
            except Exception as e:
                logger.error(f"Failed to load sparse embedding model: {e}")
                self._sparse_available = False
                return None
        return self._sparse_model

    def sparse_embed(self, text: str) -> models.SparseVector | None:
        """Generate sparse BM25 vector for text.

        Args:
            text: Input text to embed.

        Returns:
            SparseVector with indices and values, or None if model unavailable.
        """
        model = self._get_sparse_model()
        if model is None:
            return None
        result = list(model.embed([text]))[0]
        return models.SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist(),
        )

    def sparse_embed_batch(self, texts: list[str]) -> list[models.SparseVector | None]:
        """Generate sparse BM25 vectors for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of SparseVector objects (same order as input),
            or empty list if model unavailable.
        """
        if not texts:
            return []
        model = self._get_sparse_model()
        if model is None:
            return [None] * len(texts)
        results = list(model.embed(texts))
        return [
            models.SparseVector(
                indices=r.indices.tolist(),
                values=r.values.tolist(),
            )
            for r in results
        ]

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
            if self.hybrid_enabled:
                # Named vectors: dense + sparse
                await self._qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self.embedding_dimension,
                            distance=models.Distance.COSINE,
                        ),
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            modifier=models.Modifier.IDF,
                        ),
                    },
                )
            else:
                # Legacy: unnamed dense vector only
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
            await self._qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="sparse_indexed",
                field_schema=models.PayloadSchemaType.BOOL,
            )
            logger.info(f"Collection {self.collection_name} created with indexes")
        else:
            logger.debug(f"Collection {self.collection_name} already exists")

        # Detect collection capabilities (sparse vector support)
        await self._detect_collection_capabilities()

    async def refresh_capabilities(self) -> dict:
        """Re-detect collection capabilities. Called on config change or migration."""
        await self._detect_collection_capabilities()
        return {
            "collection_has_sparse": self._collection_has_sparse,
            "capabilities_checked_at": (
                self._capabilities_checked_at.isoformat() if self._capabilities_checked_at else None
            ),
        }

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

        # Generate sparse vectors (hybrid mode)
        sparse_vectors: list[models.SparseVector | None] = []
        sparse_indexed = False
        if self.hybrid_enabled and self._collection_has_sparse:
            try:
                sparse_vectors = self.sparse_embed_batch(chunks)
                sparse_indexed = all(sv is not None for sv in sparse_vectors)
                if not sparse_indexed:
                    logger.warning(
                        f"Some sparse vectors failed for {document_id}, "
                        f"indexing dense-only for affected chunks"
                    )
            except Exception as e:
                logger.error(f"Sparse embedding failed for {document_id}, indexing dense-only: {e}")
                sparse_vectors = [None] * len(chunks)
                sparse_indexed = False

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
                "sparse_indexed": (
                    sparse_indexed
                    if (self.hybrid_enabled and self._collection_has_sparse)
                    else True  # Legacy collections: not applicable, default True
                ),
            }

            # Pass through extra metadata keys (e.g., is_summary, document_type)
            for key, value in meta.items():
                if key not in payload and value is not None:
                    payload[key] = value

            # Build vector: named vectors if collection supports it, unnamed otherwise
            if self._collection_has_sparse:
                vector: dict | list = {"dense": embedding}
                if sparse_vectors and i < len(sparse_vectors) and sparse_vectors[i] is not None:
                    vector["sparse"] = sparse_vectors[i]
                points.append(
                    models.PointStruct(
                        id=chunk_id,
                        vector=vector,
                        payload=payload,
                    )
                )
            else:
                # Legacy unnamed vector (pre-migration collection)
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
            sparse_indexed=sparse_indexed
            if (self.hybrid_enabled and self._collection_has_sparse)
            else True,
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

        Supports three execution paths:
        - Hybrid: Prefetch dense + sparse, fuse with RRF, normalize scores
        - Degraded: Sparse embed fails, fall back to dense-only with named vector
        - Dense-only: Hybrid disabled or collection lacks sparse vectors

        Args:
            query: Natural language query.
            user_tags: Tags the user has access to. None = no tag filtering
                (admin bypass or tag_access_enabled=False), empty list = public only.
            limit: Maximum results to return (1-100, default: 5).
            score_threshold: Minimum similarity score (0.0-1.0, default: 0.0).
            tenant_id: Tenant to search within (defaults to service's tenant_id).

        Returns:
            List of SearchResult, ordered by relevance (highest score first).

        Raises:
            SearchError: If search fails.

        Note:
            Access control filter is applied BEFORE vector search (pre-retrieval).
            In hybrid mode, the SAME filter is applied to BOTH prefetch queries.
            - user_tags=None: No tag filtering (admin or tag_access_enabled=False users)
            - user_tags=[]: Only documents with "public" tag
            - user_tags=["hr"]: Public + hr documents
        """
        tenant = tenant_id or self.tenant_id

        # 1. Dense embedding (always needed)
        try:
            query_embedding = await self.embed(query)
        except EmbeddingError as e:
            raise SearchError(f"Failed to embed query: {e}") from e

        # 2. Access filter (unchanged)
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

        # 3. Execute search
        degraded = False
        use_hybrid = self.hybrid_enabled and self._collection_has_sparse

        try:
            if use_hybrid:
                # Try sparse embedding
                query_sparse = None
                try:
                    query_sparse = self.sparse_embed(query)
                except Exception as e:
                    logger.warning(f"Sparse embed failed, falling back to dense-only: {e}")
                    degraded = True

                if query_sparse is None and not degraded:
                    # sparse_embed returned None (model unavailable)
                    logger.warning("Sparse model unavailable, falling back to dense-only")
                    degraded = True

                if query_sparse is not None:
                    # HYBRID PATH (Path A)
                    prefetch_limit = max(20, min(100, limit * self.prefetch_multiplier))
                    response = await self._qdrant.query_points(
                        collection_name=self.collection_name,
                        prefetch=[
                            Prefetch(
                                query=query_embedding,
                                using="dense",
                                limit=prefetch_limit,
                                filter=access_filter,
                            ),
                            Prefetch(
                                query=query_sparse,
                                using="sparse",
                                limit=prefetch_limit,
                                filter=access_filter,
                            ),
                        ],
                        query=FusionQuery(fusion=Fusion.RRF),
                        limit=limit,
                        with_payload=True,
                        # NOTE: No score_threshold - RRF scores are raw, apply post-normalization
                        # NOTE: No query_filter - filters are on Prefetch, not top-level
                    )
                    points = self._normalize_scores(response.points)
                    if score_threshold > 0:
                        points = [p for p in points if p.score >= score_threshold]
                else:
                    # DEGRADED PATH (Path B): dense-only with named vector
                    response = await self._qdrant.query_points(
                        collection_name=self.collection_name,
                        query=query_embedding,
                        using="dense",
                        query_filter=access_filter,
                        limit=limit,
                        score_threshold=score_threshold if score_threshold > 0 else None,
                        with_payload=True,
                    )
                    points = response.points
            else:
                # DENSE-ONLY PATH (Path C): hybrid disabled or no sparse in collection
                query_kwargs: dict = {
                    "collection_name": self.collection_name,
                    "query": query_embedding,
                    "query_filter": access_filter,
                    "limit": limit,
                    "score_threshold": score_threshold if score_threshold > 0 else None,
                    "with_payload": True,
                }
                if self._collection_has_sparse:
                    # Collection has named vectors but hybrid is disabled
                    query_kwargs["using"] = "dense"
                response = await self._qdrant.query_points(**query_kwargs)
                points = response.points
        except Exception as e:
            if isinstance(e, SearchError):
                raise
            raise SearchError(f"Qdrant search failed: {e}") from e

        # 4. Convert to SearchResult (unchanged)
        search_results = []
        for point in points:
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

        if degraded:
            logger.info(
                f"Search completed in degraded mode (dense-only): {len(search_results)} results"
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

    def _normalize_scores(self, points: list) -> list:
        """Min-max normalize RRF fusion scores to 0.0-1.0 range.

        RRF scores are rank-based (typically 0.008-0.033) and not cosine-calibrated.
        Normalization preserves relative ordering while producing values compatible
        with existing confidence scoring and score_threshold filtering.

        Applied only in hybrid mode. Dense-only returns native cosine scores.
        """
        if not points:
            return points

        scores = [p.score for p in points]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All same rank - assign 1.0 (best possible)
            for p in points:
                p.score = 1.0
            return points

        for p in points:
            p.score = (p.score - min_score) / (max_score - min_score)

        return points

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

    async def backfill_sparse_vectors(self, batch_size: int = 100) -> int:
        """Scan for points with sparse_indexed=false and add sparse vectors.

        Called manually via CLI or scheduled job. Idempotent.

        Args:
            batch_size: Number of points to process per batch.

        Returns:
            Count of points backfilled.
        """
        if not self._collection_has_sparse:
            logger.warning("Collection does not support sparse vectors, skipping backfill")
            return 0

        total_backfilled = 0
        offset = None

        while True:
            # Scroll points where sparse_indexed == false
            results, offset = await self._qdrant.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=["chunk_text", "sparse_indexed"],
                with_vectors=False,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="sparse_indexed",
                            match=models.MatchValue(value=False),
                        ),
                    ]
                ),
            )

            if not results:
                break

            # Extract texts for batch embedding
            texts = []
            point_ids = []
            for point in results:
                chunk_text = point.payload.get("chunk_text", "") if point.payload else ""
                if chunk_text:
                    texts.append(chunk_text)
                    point_ids.append(point.id)

            if not texts:
                if offset is None:
                    break
                continue

            # Generate sparse vectors
            try:
                sparse_vectors = self.sparse_embed_batch(texts)
            except Exception as e:
                logger.error(f"Sparse embedding failed during backfill: {e}")
                break

            # Update points with sparse vectors
            for pid, sv in zip(point_ids, sparse_vectors, strict=True):
                if sv is None:
                    continue
                try:
                    await self._qdrant.update_vectors(
                        collection_name=self.collection_name,
                        points=[
                            models.PointVectors(
                                id=pid,
                                vector={"sparse": sv},
                            )
                        ],
                    )
                    await self._qdrant.set_payload(
                        collection_name=self.collection_name,
                        payload={"sparse_indexed": True},
                        points=[pid],
                    )
                    total_backfilled += 1
                except Exception as e:
                    logger.error(f"Failed to backfill point {pid}: {e}")

            logger.info(f"Backfilled {total_backfilled} points so far")

            if offset is None:
                break

        logger.info(f"Backfill complete: {total_backfilled} points updated")
        return total_backfilled
