# Vector Service Specification

**Version:** 1.1
**Date:** January 28, 2026
**Status:** Draft
**Depends On:** Qdrant 1.13.x, Ollama (nomic-embed-text)

---

## Overview

The Vector Service provides semantic search capabilities for the RAG system. It abstracts vector database operations (Qdrant) and embedding generation (Ollama) behind a clean interface.

**Key Principles:**
- Access control filtering happens BEFORE retrieval
- Service is independent of FastAPI (can be used in CLI, tests, etc.)
- Embedding model is configurable
- All operations are async
- Atomic operations - partial failures roll back

---

## File Location

```
ai_ready_rag/
├── services/
│   ├── __init__.py
│   └── vector_service.py    # This specification
```

---

## Dependencies

| Dependency | Purpose | Version |
|------------|---------|---------|
| qdrant-client | Vector database client | >=1.13.0 |
| httpx | Async HTTP for Ollama API | >=0.27.0 |

---

## Configuration

Environment variables (loaded via config.py):

| Variable | Default | Description |
|----------|---------|-------------|
| QDRANT_URL | http://localhost:6333 | Qdrant server URL |
| QDRANT_COLLECTION | documents | Collection name |
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama server URL |
| EMBEDDING_MODEL | nomic-embed-text | Model for embeddings |
| EMBEDDING_DIMENSION | 768 | Vector dimension (must match model) |
| EMBEDDING_MAX_TOKENS | 8192 | Maximum tokens before truncation |
| DEFAULT_TENANT_ID | default | Default tenant for single-tenant deployments |

---

## Chunk ID Strategy

**Critical:** Every chunk requires a stable, deterministic ID for updates, deletes, and citation integrity.

### ID Generation

```python
import uuid

def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """
    Generate deterministic chunk ID using UUIDv5.

    Same document_id + chunk_index always produces same chunk_id.
    This enables idempotent upserts and reliable citations.
    """
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # UUID namespace
    name = f"{document_id}:{chunk_index}"
    return str(uuid.uuid5(namespace, name))
```

### Lifecycle

| Operation | Behavior |
|-----------|----------|
| **Insert** | Add new chunks with generated IDs |
| **Update** | Delete all chunks for document_id, then insert new chunks (atomic) |
| **Delete** | Remove all chunks matching document_id |

**Rationale:** Upsert-by-document ensures chunk count changes (re-chunking) don't leave orphans.

---

## Tag System

### Reserved Tags

| Tag | Purpose | Assignable by Users |
|-----|---------|---------------------|
| `public` | Documents visible to all authenticated users | No (admin only) |
| `system` | System-internal documents | No (system only) |

### Tag Validation Rules

```python
import re

TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9\-]{0,62}[a-z0-9]$|^[a-z0-9]$")
RESERVED_TAGS = {"public", "system"}
MAX_TAG_LENGTH = 64

def validate_tag(tag: str) -> str:
    """
    Validate and normalize a tag.

    Rules:
    - Lowercase only (normalized)
    - Alphanumeric and hyphens only
    - Must start and end with alphanumeric
    - 1-64 characters
    - No consecutive hyphens

    Returns:
        Normalized tag (lowercase)

    Raises:
        ValueError: If tag is invalid
    """
    normalized = tag.lower().strip()

    if not normalized:
        raise ValueError("Tag cannot be empty")

    if len(normalized) > MAX_TAG_LENGTH:
        raise ValueError(f"Tag exceeds {MAX_TAG_LENGTH} characters")

    if "--" in normalized:
        raise ValueError("Tag cannot contain consecutive hyphens")

    if not TAG_PATTERN.match(normalized):
        raise ValueError(
            "Tag must be alphanumeric with hyphens, "
            "starting and ending with alphanumeric"
        )

    return normalized

def validate_tags_for_ingestion(tags: list[str], is_admin: bool = False) -> list[str]:
    """
    Validate tags for document ingestion.

    Args:
        tags: List of tags to validate
        is_admin: Whether the user has admin privileges

    Returns:
        List of normalized, validated tags

    Raises:
        ValueError: If any tag is invalid or reserved (for non-admins)
    """
    if not tags:
        raise ValueError("At least one tag is required")

    validated = []
    for tag in tags:
        normalized = validate_tag(tag)

        if normalized in RESERVED_TAGS and not is_admin:
            raise ValueError(f"Tag '{normalized}' is reserved and requires admin privileges")

        validated.append(normalized)

    return list(set(validated))  # Deduplicate
```

### Enforcement Points

| Point | Validation |
|-------|------------|
| Document upload API | `validate_tags_for_ingestion()` with user's admin status |
| Vector service `add_document()` | `validate_tag()` for each tag (assumes pre-validated) |
| User tag assignment | Admin-only endpoint, validates against reserved tags |

---

## Qdrant Collection Schema

Collection: `documents` (configurable)

**Vector Configuration:**
- Size: 768 (nomic-embed-text)
- Distance: Cosine
- On-disk: false (for dev), true (for production with >100k vectors)

**Payload Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| chunk_id | string | Yes | UUIDv5(document_id:chunk_index) - also used as point ID |
| document_id | string | Yes | Parent document UUID |
| document_name | string | Yes | Original filename |
| chunk_index | integer | Yes | Position in document (0-based) |
| chunk_text | string | Yes | The actual text content |
| tags | string[] | Yes | Access control tags (normalized, validated) |
| tenant_id | string | Yes | Tenant identifier (default: "default") |
| page_number | integer | No | Page number if applicable |
| section | string | No | Section heading if detected |
| uploaded_by | string | Yes | User ID who uploaded |
| uploaded_at | string | Yes | ISO 8601 timestamp |

**Payload Indexes:**

| Field | Index Type | Purpose |
|-------|------------|---------|
| tags | keyword | Access control filtering |
| document_id | keyword | Document deletion |
| tenant_id | keyword | Multi-tenant filtering |

---

## Class Interface

```python
class VectorService:
    """
    Handles vector storage and semantic search operations.

    Thread-safe and async-compatible.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        ollama_url: str = "http://localhost:11434",
        collection_name: str = "documents",
        embedding_model: str = "nomic-embed-text",
        embedding_dimension: int = 768,
        max_tokens: int = 8192,
        tenant_id: str = "default"
    ):
        """Initialize vector service with connection parameters."""

    async def initialize(self) -> None:
        """
        Create collection if not exists.
        Called on application startup.
        Idempotent - safe to call multiple times.

        Creates payload indexes for tags, document_id, tenant_id.
        """

    async def health_check(self) -> HealthStatus:
        """
        Check connectivity to Qdrant and Ollama.

        Returns:
            HealthStatus with component status and latencies

        Timeouts:
            - Qdrant: 5 seconds
            - Ollama: 10 seconds (model loading may be slow)
        """
```

### Health Check Response

```python
@dataclass
class HealthStatus:
    """Health check response with detailed status."""
    qdrant_healthy: bool
    qdrant_latency_ms: float | None  # None if unhealthy
    ollama_healthy: bool
    ollama_latency_ms: float | None
    collection_exists: bool
    collection_vector_count: int | None

    @property
    def healthy(self) -> bool:
        """Overall health - all components must be healthy."""
        return self.qdrant_healthy and self.ollama_healthy and self.collection_exists
```

---

## Core Methods

### Embedding Generation

```python
async def embed(self, text: str) -> list[float]:
    """
    Generate embedding vector for text.

    Args:
        text: Input text

    Truncation Behavior:
        - Model: nomic-embed-text (BERT-based tokenizer)
        - Max tokens: 8192 (configurable)
        - If input exceeds max_tokens, text is truncated from the END
        - Truncation logs a warning with original vs truncated length
        - Tokenization uses model's native tokenizer via Ollama

    Returns:
        768-dimensional float vector (normalized, suitable for cosine similarity)

    Raises:
        EmbeddingError: If Ollama is unavailable or model not found
    """

async def embed_batch(
    self,
    texts: list[str],
    max_batch_size: int = 100
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts.

    Args:
        texts: List of input texts
        max_batch_size: Maximum texts per batch (default: 100)

    Batch Behavior:
        - Texts are processed in batches of max_batch_size
        - Each batch is a single Ollama request
        - If any embedding in a batch fails, entire batch is retried
        - Retry: 3 attempts with exponential backoff (1s, 2s, 4s)
        - If batch fails after retries, raises EmbeddingError

    Returns:
        List of 768-dimensional vectors (same order as input)

    Raises:
        EmbeddingError: If embedding fails after retries
    """
```

### Document Indexing

```python
async def add_document(
    self,
    document_id: str,
    document_name: str,
    chunks: list[str],
    tags: list[str],
    uploaded_by: str,
    chunk_metadata: list[dict] | None = None,
    tenant_id: str | None = None
) -> IndexResult:
    """
    Index a document's chunks to the vector store.

    Args:
        document_id: Unique document identifier (UUID)
        document_name: Original filename for display
        chunks: List of text chunks (pre-chunked by caller)
        tags: Access control tags (must be pre-validated)
        uploaded_by: User ID of uploader
        chunk_metadata: Optional per-chunk metadata, must align 1:1 with chunks
                       Each dict may contain: page_number (int), section (str)
        tenant_id: Tenant identifier (defaults to service's tenant_id)

    Chunk Metadata:
        If provided, chunk_metadata[i] applies to chunks[i].
        If chunk_metadata is shorter than chunks, remaining chunks get empty metadata.

        Example:
            chunks = ["chunk1", "chunk2", "chunk3"]
            chunk_metadata = [
                {"page_number": 1, "section": "Introduction"},
                {"page_number": 1},
                {"page_number": 2, "section": "Methods"}
            ]

    Atomicity:
        - If document_id already exists, ALL existing chunks are deleted first
        - Then all new chunks are inserted
        - If any step fails, operation is rolled back (no partial state)

    Batch Behavior:
        - Chunks are embedded in batches of 100
        - Qdrant upsert in batches of 100 points
        - Retry: 3 attempts with exponential backoff per batch

    Returns:
        IndexResult with chunk count and status

    Raises:
        ValueError: If tags list is empty or chunk_metadata length mismatches
        IndexingError: If operation fails after retries
    """

@dataclass
class IndexResult:
    """Result of document indexing operation."""
    document_id: str
    chunks_indexed: int
    replaced_existing: bool  # True if document was re-indexed
    embedding_time_ms: float
    indexing_time_ms: float

async def delete_document(self, document_id: str) -> bool:
    """
    Remove all chunks for a document.

    Args:
        document_id: Document to delete

    Returns:
        True if deletion was successful (regardless of whether document existed)
        False if deletion failed due to Qdrant error

    Notes:
        - Does not pre-count chunks (avoids extra round-trip)
        - Deletion is idempotent - deleting non-existent document returns True
    """
```

### Search

```python
@dataclass
class SearchResult:
    """Single search result with metadata."""
    chunk_id: str           # UUIDv5 identifier (can be used for citations)
    document_id: str
    document_name: str
    chunk_text: str
    chunk_index: int
    score: float            # 0.0 to 1.0 (see Score Semantics below)
    page_number: int | None
    section: str | None

async def search(
    self,
    query: str,
    user_tags: list[str],
    limit: int = 5,
    score_threshold: float = 0.0,
    tenant_id: str | None = None
) -> list[SearchResult]:
    """
    Semantic search with access control filtering.

    Args:
        query: Natural language query
        user_tags: Tags the user has access to (empty = public only)
        limit: Maximum results to return (1-100, default: 5)
        score_threshold: Minimum similarity score (0.0-1.0, default: 0.0)
        tenant_id: Tenant to search within (defaults to service's tenant_id)

    Returns:
        List of SearchResult, ordered by relevance (highest score first)

    Access Control:
        - Filter is applied BEFORE vector search (pre-retrieval)
        - Empty user_tags returns only documents with "public" tag
        - User with tags sees "public" + documents matching ANY of their tags
        - Results NEVER include documents user cannot access
    """
```

### Score Semantics

**Important:** Understanding score interpretation is critical for threshold tuning.

| Metric | Value Range | Interpretation |
|--------|-------------|----------------|
| Qdrant Cosine Similarity | 0.0 to 1.0 | Higher = more similar |
| 0.0 | Orthogonal vectors | No semantic relationship |
| 0.5 | Moderate similarity | Loosely related topics |
| 0.7 | Good similarity | Related content, good match |
| 0.85+ | High similarity | Very relevant, strong match |
| 1.0 | Identical vectors | Same or near-identical text |

**Recommended Thresholds:**

| Use Case | Threshold | Rationale |
|----------|-----------|-----------|
| Broad recall (exploration) | 0.3 | Include loosely related content |
| Balanced (default) | 0.5 | Good precision/recall tradeoff |
| High precision (citations) | 0.7 | Only confident matches |

**Note:** Scores are normalized cosine similarity from Qdrant with `distance: Cosine`. Vectors are L2-normalized before storage, ensuring scores are always 0.0-1.0.

### Collection Management

```python
async def get_stats(self) -> CollectionStats:
    """
    Get collection statistics.

    Returns:
        CollectionStats with counts and size information
    """

@dataclass
class CollectionStats:
    """Collection statistics."""
    total_chunks: int
    total_documents: int      # Count of unique document_ids
    collection_size_bytes: int
    tenant_id: str

async def clear_collection(self) -> bool:
    """
    Delete all vectors in collection. Use with caution.

    Returns:
        True if successful, False if failed

    Notes:
        - Primarily for testing
        - In production, prefer delete_document() for targeted removal
        - Does not delete the collection itself, only its contents
    """
```

---

## Error Handling

Custom exceptions in `core/exceptions.py`:

```python
class VectorServiceError(Exception):
    """Base exception for vector service errors."""
    pass

class EmbeddingError(VectorServiceError):
    """
    Failed to generate embeddings.

    Causes:
        - Ollama unavailable
        - Model not found/loaded
        - Request timeout (>10s)
        - Batch retry exhausted
    """
    pass

class IndexingError(VectorServiceError):
    """
    Failed to index documents to Qdrant.

    Causes:
        - Qdrant unavailable
        - Collection doesn't exist
        - Batch retry exhausted
        - Payload validation failed
    """
    pass

class SearchError(VectorServiceError):
    """
    Failed to execute search query.

    Causes:
        - Qdrant unavailable
        - Invalid filter construction
        - Query embedding failed
    """
    pass
```

---

## Access Control Filter Logic

The search filter ensures users only see authorized documents:

```python
def build_access_filter(user_tags: list[str], tenant_id: str) -> models.Filter:
    """
    Build Qdrant filter for access-controlled search.

    Logic:
        (tenant_id matches) AND (
            "public" in tags
            OR any(user_tags) in tags
        )
    """
    tag_conditions = [
        models.FieldCondition(
            key="tags",
            match=models.MatchValue(value="public")
        )
    ]

    if user_tags:
        tag_conditions.append(
            models.FieldCondition(
                key="tags",
                match=models.MatchAny(any=user_tags)
            )
        )

    return models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id)
            )
        ],
        should=tag_conditions,
        min_should_match=1  # At least one tag condition must match
    )
```

**Access Matrix:**

| User Tags | Document Tags | Access |
|-----------|---------------|--------|
| [] | ["public"] | ✅ Yes |
| [] | ["hr"] | ❌ No |
| ["hr"] | ["public"] | ✅ Yes |
| ["hr"] | ["hr", "finance"] | ✅ Yes (ANY match) |
| ["hr"] | ["finance"] | ❌ No |
| ["hr", "finance"] | ["legal"] | ❌ No |

---

## Retry and Failure Behavior

### Retry Configuration

```python
RETRY_CONFIG = {
    "max_attempts": 3,
    "base_delay_seconds": 1.0,
    "max_delay_seconds": 10.0,
    "exponential_base": 2.0,
}

# Delay sequence: 1s, 2s, 4s (capped at max_delay)
```

### Failure Modes

| Operation | Failure Behavior |
|-----------|------------------|
| `embed()` | Retry 3x, then raise EmbeddingError |
| `embed_batch()` | Retry failed batch 3x, then raise EmbeddingError |
| `add_document()` | Atomic - if any batch fails after retries, roll back all changes |
| `search()` | Retry 3x, then raise SearchError |
| `delete_document()` | Retry 3x, return False on final failure |

### Partial Failure Handling

For `add_document()`:
1. Delete existing chunks (if any) for document_id
2. Embed all chunks in batches
3. If embedding fails: operation aborts, no chunks indexed
4. Upsert to Qdrant in batches
5. If upsert fails: delete any chunks that were inserted, raise IndexingError

**Rationale:** Users should never see partially indexed documents.

---

## Usage Examples

### Initialization (app startup)

```python
from ai_ready_rag.services.vector_service import VectorService
from ai_ready_rag.config import get_settings

settings = get_settings()
vector_service = VectorService(
    qdrant_url=settings.qdrant_url,
    ollama_url=settings.ollama_base_url,
    tenant_id=settings.default_tenant_id
)
await vector_service.initialize()

# Health check
health = await vector_service.health_check()
if not health.healthy:
    logger.error(f"Vector service unhealthy: {health}")
```

### Indexing a Document

```python
chunks = ["First paragraph...", "Second paragraph...", "Third paragraph..."]
chunk_metadata = [
    {"page_number": 1, "section": "Introduction"},
    {"page_number": 1, "section": "Introduction"},
    {"page_number": 2, "section": "Background"},
]

result = await vector_service.add_document(
    document_id="doc_123",
    document_name="Employee Handbook.pdf",
    chunks=chunks,
    tags=["hr", "policy"],  # Pre-validated by API layer
    uploaded_by="user_456",
    chunk_metadata=chunk_metadata
)

print(f"Indexed {result.chunks_indexed} chunks in {result.indexing_time_ms}ms")
```

### Searching

```python
# User has tags ["hr", "engineering"]
results = await vector_service.search(
    query="What is the vacation policy?",
    user_tags=["hr", "engineering"],
    limit=5,
    score_threshold=0.5  # Only good matches
)

for r in results:
    print(f"{r.document_name} (score: {r.score:.2f})")
    print(f"  Chunk ID: {r.chunk_id}")  # For citation tracking
    print(f"  {r.chunk_text[:100]}...")
```

---

## Testing Strategy

### Unit Tests (mocked)
- Embedding generation with mocked Ollama
- Filter construction logic
- Tag validation rules
- Chunk ID generation determinism
- Error handling paths
- Retry logic

### Integration Tests (requires services)
- End-to-end index and search
- Access control filtering verification
- Atomic rollback on failure
- Collection management
- Multi-tenant isolation

### Test Fixtures
- Sample documents with various tag combinations
- Test users with different tag assignments
- Edge cases: empty tags, reserved tags, long text

---

## Production Configuration (Appendix)

For deployments with >100k vectors, consider these Qdrant optimizations:

### HNSW Parameters

```python
# Collection creation with optimized HNSW
await qdrant_client.create_collection(
    collection_name="documents",
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
        on_disk=True,  # Enable for large collections
    ),
    hnsw_config=models.HnswConfigDiff(
        m=16,              # Connections per node (default: 16)
        ef_construct=100,  # Build-time search width (default: 100)
    ),
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=20000,  # Points before indexing
    ),
)
```

### Quantization (Memory Optimization)

```python
# Scalar quantization for 4x memory reduction
await qdrant_client.update_collection(
    collection_name="documents",
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=True,  # Keep quantized vectors in RAM
        ),
    ),
)
```

### Recommended Settings by Scale

| Vector Count | on_disk | Quantization | ef_search |
|--------------|---------|--------------|-----------|
| <50k | false | none | 64 |
| 50k-500k | true | INT8 | 128 |
| >500k | true | INT8 | 256 |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-28 | Initial specification |
| 1.1 | 2026-01-28 | Address engineering review: chunk ID lifecycle, tag validation, truncation behavior, metadata alignment, score semantics, batch/retry handling, health check timeouts, tenant_id, production config appendix |
