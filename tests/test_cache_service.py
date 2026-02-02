"""Tests for CacheService."""

import concurrent.futures
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from ai_ready_rag.db.models import Document, EmbeddingCache, ResponseCache, Tag, document_tags
from ai_ready_rag.services.cache_service import (
    CacheEntry,
    CacheService,
    MemoryCache,
    UploadBatchContext,
    cosine_similarity,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_response():
    """Mock RAGResponse for cache testing."""
    response = MagicMock()
    response.answer = "This is a cached answer."
    response.confidence = MagicMock()
    response.confidence.overall = 75
    response.confidence.retrieval_score = 0.85  # Match ConfidenceScore dataclass
    response.confidence.coverage_score = 0.80  # Match ConfidenceScore dataclass
    response.confidence.llm_score = 70  # Match ConfidenceScore dataclass
    response.generation_time_ms = 34000.0
    response.model_used = "llama3.2"
    response.citations = []
    return response


@pytest.fixture
def sample_entry():
    """Sample CacheEntry for testing."""
    return CacheEntry(
        query_hash="abc123",
        query_text="What is the vacation policy?",
        query_embedding=[0.1] * 768,
        answer="Employees get 20 days.",
        sources=[],
        confidence_overall=75,
        confidence_retrieval=0.85,
        confidence_coverage=0.80,
        confidence_llm=70,
        generation_time_ms=34000.0,
        model_used="llama3.2",
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        access_count=1,
        document_ids=["doc-1"],
    )


@pytest.fixture
def sample_documents(db, admin_user, sample_tag):
    """Create test documents with tags for access verification."""
    # Create a document with the hr tag
    doc = Document(
        id="doc-1",
        filename="test.pdf",
        original_filename="test.pdf",
        file_path="/tmp/test.pdf",
        file_type="pdf",
        file_size=1000,
        status="ready",
        uploaded_by=admin_user.id,
    )
    db.add(doc)
    db.flush()

    # Associate document with tag
    stmt = document_tags.insert().values(document_id=doc.id, tag_id=sample_tag.id)
    db.execute(stmt)
    db.flush()

    return [doc]


# =============================================================================
# Cosine Similarity Tests
# =============================================================================


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        """Identical vectors have cosine similarity 1.0."""
        vec = [0.5, 0.5, 0.5]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        """Orthogonal vectors have cosine similarity 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_similar_vectors_high_similarity(self):
        """Slightly different vectors have high similarity."""
        a = [1.0, 0.0, 0.0]
        b = [0.9, 0.1, 0.0]
        sim = cosine_similarity(a, b)
        assert sim > 0.9  # High similarity

    def test_opposite_vectors_negative_similarity(self):
        """Opposite vectors have similarity -1.0."""
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_first_returns_zero(self):
        """Zero first vector returns 0.0 (no division error)."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 1.0, 1.0]
        assert cosine_similarity(a, b) == 0.0

    def test_zero_vector_second_returns_zero(self):
        """Zero second vector returns 0.0 (no division error)."""
        a = [1.0, 1.0, 1.0]
        b = [0.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors_returns_zero(self):
        """Both zero vectors return 0.0."""
        a = [0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_negative_values_computed_correctly(self):
        """Negative vector components handled correctly."""
        a = [-0.5, 0.5]
        b = [0.5, 0.5]
        # Dot product: -0.25 + 0.25 = 0
        # norm_a = sqrt(0.25 + 0.25) = sqrt(0.5)
        # norm_b = sqrt(0.25 + 0.25) = sqrt(0.5)
        # similarity = 0 / (sqrt(0.5) * sqrt(0.5)) = 0
        assert cosine_similarity(a, b) == pytest.approx(0.0)


# =============================================================================
# MemoryCache Tests
# =============================================================================


class TestMemoryCache:
    def test_put_and_get_exact(self, sample_entry):
        """Put entry, get by exact hash returns entry."""
        cache = MemoryCache(max_size=10)
        cache.put(sample_entry)
        result = cache.get(sample_entry.query_hash)
        assert result is not None
        assert result.answer == sample_entry.answer

    def test_get_miss_returns_none(self):
        """Get nonexistent hash returns None."""
        cache = MemoryCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_lru_eviction_order(self, sample_entry):
        """Adding max_size+1 entries evicts oldest."""
        cache = MemoryCache(max_size=2)
        entry1 = sample_entry
        entry2 = CacheEntry(
            query_hash="def456",
            query_text=sample_entry.query_text,
            query_embedding=sample_entry.query_embedding,
            answer=sample_entry.answer,
            sources=sample_entry.sources,
            confidence_overall=sample_entry.confidence_overall,
            confidence_retrieval=sample_entry.confidence_retrieval,
            confidence_coverage=sample_entry.confidence_coverage,
            confidence_llm=sample_entry.confidence_llm,
            generation_time_ms=sample_entry.generation_time_ms,
            model_used=sample_entry.model_used,
            created_at=sample_entry.created_at,
            last_accessed_at=sample_entry.last_accessed_at,
            access_count=sample_entry.access_count,
            document_ids=sample_entry.document_ids,
        )
        entry3 = CacheEntry(
            query_hash="ghi789",
            query_text=sample_entry.query_text,
            query_embedding=sample_entry.query_embedding,
            answer=sample_entry.answer,
            sources=sample_entry.sources,
            confidence_overall=sample_entry.confidence_overall,
            confidence_retrieval=sample_entry.confidence_retrieval,
            confidence_coverage=sample_entry.confidence_coverage,
            confidence_llm=sample_entry.confidence_llm,
            generation_time_ms=sample_entry.generation_time_ms,
            model_used=sample_entry.model_used,
            created_at=sample_entry.created_at,
            last_accessed_at=sample_entry.last_accessed_at,
            access_count=sample_entry.access_count,
            document_ids=sample_entry.document_ids,
        )

        cache.put(entry1)
        cache.put(entry2)
        cache.put(entry3)  # Should evict entry1

        assert cache.get("abc123") is None
        assert cache.get("def456") is not None
        assert cache.get("ghi789") is not None

    def test_get_moves_to_mru(self, sample_entry):
        """Accessing entry moves it to MRU position."""
        cache = MemoryCache(max_size=2)

        entry_a = sample_entry
        entry_b = CacheEntry(
            query_hash="bbb",
            query_text=sample_entry.query_text,
            query_embedding=sample_entry.query_embedding,
            answer=sample_entry.answer,
            sources=sample_entry.sources,
            confidence_overall=sample_entry.confidence_overall,
            confidence_retrieval=sample_entry.confidence_retrieval,
            confidence_coverage=sample_entry.confidence_coverage,
            confidence_llm=sample_entry.confidence_llm,
            generation_time_ms=sample_entry.generation_time_ms,
            model_used=sample_entry.model_used,
            created_at=sample_entry.created_at,
            last_accessed_at=sample_entry.last_accessed_at,
            access_count=sample_entry.access_count,
            document_ids=sample_entry.document_ids,
        )
        entry_c = CacheEntry(
            query_hash="ccc",
            query_text=sample_entry.query_text,
            query_embedding=sample_entry.query_embedding,
            answer=sample_entry.answer,
            sources=sample_entry.sources,
            confidence_overall=sample_entry.confidence_overall,
            confidence_retrieval=sample_entry.confidence_retrieval,
            confidence_coverage=sample_entry.confidence_coverage,
            confidence_llm=sample_entry.confidence_llm,
            generation_time_ms=sample_entry.generation_time_ms,
            model_used=sample_entry.model_used,
            created_at=sample_entry.created_at,
            last_accessed_at=sample_entry.last_accessed_at,
            access_count=sample_entry.access_count,
            document_ids=sample_entry.document_ids,
        )

        cache.put(entry_a)  # A in cache
        cache.put(entry_b)  # A, B in cache
        cache.get("abc123")  # Access A -> B, A in order
        cache.put(entry_c)  # B evicted -> A, C

        assert cache.get("bbb") is None  # B evicted
        assert cache.get("abc123") is not None  # A kept (was accessed)
        assert cache.get("ccc") is not None  # C kept

    def test_clear_returns_count(self, sample_entry):
        """Clear returns count of removed entries."""
        cache = MemoryCache()
        cache.put(sample_entry)
        count = cache.clear()
        assert count == 1
        assert len(cache) == 0

    def test_len_reflects_entries(self, sample_entry):
        """`len()` returns accurate count."""
        cache = MemoryCache()
        assert len(cache) == 0
        cache.put(sample_entry)
        assert len(cache) == 1

    def test_access_count_incremented(self, sample_entry):
        """Get increments access_count."""
        cache = MemoryCache()
        cache.put(sample_entry)
        initial_count = sample_entry.access_count

        cache.get(sample_entry.query_hash)
        cache.get(sample_entry.query_hash)

        result = cache.get(sample_entry.query_hash)
        assert result.access_count > initial_count


# =============================================================================
# Semantic Cache Tests
# =============================================================================


class TestSemanticCache:
    def test_get_semantic_exact_embedding_match(self, sample_entry):
        """Exact embedding match returns entry."""
        cache = MemoryCache()
        cache.put(sample_entry)

        # Same embedding should match
        result = cache.get_semantic(sample_entry.query_embedding, threshold=0.95)
        assert result is not None
        assert result.query_hash == sample_entry.query_hash

    def test_get_semantic_similar_above_threshold(self, sample_entry):
        """Similar embedding above threshold returns entry."""
        cache = MemoryCache()
        cache.put(sample_entry)

        # Slightly different embedding (should still exceed 0.95 threshold)
        similar_embedding = [v + 0.001 for v in sample_entry.query_embedding]
        result = cache.get_semantic(similar_embedding, threshold=0.95)
        assert result is not None

    def test_get_semantic_below_threshold_returns_none(self, sample_entry):
        """Embedding below threshold returns None."""
        cache = MemoryCache()
        cache.put(sample_entry)

        # Completely different embedding (alternating pattern)
        different = [0.9 if i % 2 == 0 else 0.1 for i in range(768)]
        result = cache.get_semantic(different, threshold=0.95)
        assert result is None

    def test_get_semantic_empty_cache_returns_none(self):
        """Empty cache returns None."""
        cache = MemoryCache()
        result = cache.get_semantic([0.5] * 768, threshold=0.95)
        assert result is None

    def test_get_semantic_multiple_entries_returns_best(self, sample_entry):
        """With multiple entries, returns highest similarity match."""
        cache = MemoryCache(max_size=10)

        # Entry 1: low similarity (orthogonal-ish)
        entry_low = CacheEntry(
            query_hash="low",
            query_text="low similarity",
            query_embedding=[0.9 if i % 2 == 0 else -0.9 for i in range(768)],
            answer="low",
            sources=[],
            confidence_overall=75,
            confidence_retrieval=0.85,
            confidence_coverage=0.80,
            confidence_llm=70,
            generation_time_ms=34000.0,
            model_used="llama3.2",
            created_at=datetime.utcnow(),
            last_accessed_at=datetime.utcnow(),
            access_count=1,
            document_ids=[],
        )

        # Entry 2: high similarity (close to query)
        query_embedding = [0.1] * 768
        entry_high = CacheEntry(
            query_hash="high",
            query_text="high similarity",
            query_embedding=[v + 0.0001 for v in query_embedding],  # Very close
            answer="high",
            sources=[],
            confidence_overall=75,
            confidence_retrieval=0.85,
            confidence_coverage=0.80,
            confidence_llm=70,
            generation_time_ms=34000.0,
            model_used="llama3.2",
            created_at=datetime.utcnow(),
            last_accessed_at=datetime.utcnow(),
            access_count=1,
            document_ids=[],
        )

        cache.put(entry_low)
        cache.put(entry_high)

        result = cache.get_semantic(query_embedding, threshold=0.95)
        assert result is not None
        assert result.query_hash == "high"  # Best match

    def test_get_semantic_updates_access_count(self, sample_entry):
        """Semantic hit increments access_count."""
        cache = MemoryCache()
        cache.put(sample_entry)
        initial_count = sample_entry.access_count

        result = cache.get_semantic(sample_entry.query_embedding, threshold=0.95)
        assert result.access_count > initial_count

    def test_get_semantic_updates_last_accessed_at(self, sample_entry):
        """Semantic hit updates last_accessed_at."""
        cache = MemoryCache()
        # Set old time
        old_time = datetime.utcnow() - timedelta(hours=1)
        sample_entry.last_accessed_at = old_time
        cache.put(sample_entry)

        result = cache.get_semantic(sample_entry.query_embedding, threshold=0.95)
        assert result.last_accessed_at > old_time

    def test_get_semantic_moves_to_mru(self, sample_entry):
        """Semantic hit moves entry to MRU position."""
        cache = MemoryCache(max_size=2)

        # Create entry_b with orthogonal embedding (won't match sample_entry semantically)
        entry_b = CacheEntry(
            query_hash="bbb",
            query_text="other query",
            query_embedding=[1.0 if i % 2 == 0 else -1.0 for i in range(768)],  # Orthogonal
            answer="other answer",
            sources=[],
            confidence_overall=75,
            confidence_retrieval=0.85,
            confidence_coverage=0.80,
            confidence_llm=70,
            generation_time_ms=34000.0,
            model_used="llama3.2",
            created_at=datetime.utcnow(),
            last_accessed_at=datetime.utcnow(),
            access_count=1,
            document_ids=[],
        )

        cache.put(sample_entry)  # sample_entry is LRU
        cache.put(entry_b)  # entry_b is MRU

        # Access sample_entry via semantic search -> moves to MRU
        cache.get_semantic(sample_entry.query_embedding, threshold=0.95)

        # Now add third entry with orthogonal embedding -> should evict entry_b (now LRU)
        entry_c = CacheEntry(
            query_hash="ccc",
            query_text="third query",
            query_embedding=[1.0 if i % 3 == 0 else -1.0 for i in range(768)],  # Orthogonal
            answer="third answer",
            sources=[],
            confidence_overall=75,
            confidence_retrieval=0.85,
            confidence_coverage=0.80,
            confidence_llm=70,
            generation_time_ms=34000.0,
            model_used="llama3.2",
            created_at=datetime.utcnow(),
            last_accessed_at=datetime.utcnow(),
            access_count=1,
            document_ids=[],
        )
        cache.put(entry_c)

        # sample_entry should still be in cache (was accessed, now MRU)
        assert cache.get(sample_entry.query_hash) is not None
        # entry_b should be evicted
        assert cache.get("bbb") is None


# =============================================================================
# CacheService Core Tests
# =============================================================================


class TestCacheServiceCore:
    @pytest.mark.asyncio
    async def test_cache_disabled_returns_none(self, db):
        """When cache_enabled=False, get() returns None."""
        with patch("ai_ready_rag.services.settings_service.get_cache_setting", return_value=False):
            service = CacheService(db)
            result = await service.get("test query", ["hr"])
            assert result is None

    @pytest.mark.asyncio
    async def test_cache_miss_logs_access(self, db):
        """Cache miss increments miss count."""
        service = CacheService(db)
        result = await service.get("unknown query", ["hr"])
        assert result is None
        assert service._miss_count == 1

    def test_hash_query_deterministic(self):
        """Same input always produces same hash."""
        hash1 = CacheService._hash_query("test query")
        hash2 = CacheService._hash_query("test query")
        assert hash1 == hash2

    def test_hash_query_normalization(self):
        """Queries differing in case/whitespace produce same hash."""
        hash1 = CacheService._hash_query("Test Query")
        hash2 = CacheService._hash_query("test query")
        hash3 = CacheService._hash_query("  TEST QUERY  ")
        assert hash1 == hash2 == hash3

    def test_semantic_threshold_default(self, db):
        """Default semantic threshold is 0.95."""
        with patch(
            "ai_ready_rag.services.settings_service.get_cache_setting",
            side_effect=lambda key, default: default,
        ):
            service = CacheService(db)
            assert service.semantic_threshold == 0.95

    @pytest.mark.asyncio
    async def test_semantic_fallback_on_layer1_miss(self, db, mock_response):
        """CacheService tries semantic match when exact hash misses."""
        service = CacheService(db)
        embedding = [0.1] * 768

        # Store response with original query
        await service.put("original query", embedding, mock_response)

        # Different query text but similar embedding should hit semantically
        similar_embedding = [v + 0.001 for v in embedding]
        result = await service.get("different query text", [], query_embedding=similar_embedding)

        assert result is not None
        assert result.answer == mock_response.answer

    @pytest.mark.asyncio
    async def test_semantic_skipped_without_embedding(self, db, mock_response):
        """CacheService skips Layer 2 when query_embedding not provided."""
        service = CacheService(db)
        embedding = [0.1] * 768

        # Store response
        await service.put("original query", embedding, mock_response)

        # Different query text without embedding -> Layer 2 skipped -> miss
        result = await service.get("different query text", [])
        assert result is None

    @pytest.mark.asyncio
    async def test_semantic_hit_verifies_access(
        self, db, mock_response, sample_documents, sample_tag, admin_user
    ):
        """Semantic hit still respects access verification."""
        from types import SimpleNamespace

        # Mock response with citation to doc-1 (requires hr tag)
        # Use SimpleNamespace for JSON-serializable attributes
        citation = SimpleNamespace(
            document_id="doc-1",
            document_name="test.pdf",
            page_number=1,
            section="Section 1",
            snippet="Test excerpt",  # Cache uses 'snippet' not 'excerpt'
        )
        mock_response.citations = [citation]

        service = CacheService(db)
        embedding = [0.1] * 768

        # Store response
        await service.put("original query", embedding, mock_response)

        # Try to access with wrong tag -> access denied
        similar_embedding = [v + 0.001 for v in embedding]
        result = await service.get(
            "different query text", ["finance"], query_embedding=similar_embedding
        )
        assert result is None  # Access denied

        # Try with correct tag -> access granted
        result = await service.get(
            "different query text", ["hr"], query_embedding=similar_embedding
        )
        assert result is not None


# =============================================================================
# TTL Expiration Tests
# =============================================================================


class TestTTLExpiration:
    @pytest.mark.asyncio
    async def test_expired_entry_not_returned(self, db):
        """Entry older than TTL is treated as miss."""
        import json

        # Insert expired entry directly into SQLite
        old_time = datetime.utcnow() - timedelta(hours=25)
        expired_row = ResponseCache(
            query_hash=CacheService._hash_query("expired query"),
            query_text="expired query",
            query_embedding=json.dumps([0.1] * 768),
            answer="Old answer",
            sources=json.dumps([]),
            confidence_overall=75,
            confidence_retrieval=0.85,
            confidence_coverage=0.80,
            confidence_llm=70,
            generation_time_ms=34000.0,
            model_used="llama3.2",
            document_ids=json.dumps([]),
            created_at=old_time,
        )
        db.add(expired_row)
        db.commit()

        # Mock settings to ensure TTL is 24 hours
        with patch(
            "ai_ready_rag.services.settings_service.get_cache_setting",
            side_effect=lambda key, default: default,
        ):
            service = CacheService(db)
            result = await service.get("expired query", [])
            assert result is None

    def test_evict_expired_removes_old(self, db):
        """evict_expired() removes entries older than TTL."""
        import json

        # Insert 2 expired, 1 fresh
        old_time = datetime.utcnow() - timedelta(hours=25)
        for i in range(2):
            row = ResponseCache(
                query_hash=f"expired_{i}",
                query_text=f"query {i}",
                query_embedding=json.dumps([0.1] * 768),
                answer="answer",
                sources=json.dumps([]),
                confidence_overall=75,
                confidence_retrieval=0.85,
                confidence_coverage=0.80,
                confidence_llm=70,
                generation_time_ms=34000.0,
                model_used="llama3.2",
                document_ids=json.dumps([]),
                created_at=old_time,
            )
            db.add(row)

        fresh_row = ResponseCache(
            query_hash="fresh",
            query_text="fresh query",
            query_embedding=json.dumps([0.1] * 768),
            answer="fresh answer",
            sources=json.dumps([]),
            confidence_overall=75,
            confidence_retrieval=0.85,
            confidence_coverage=0.80,
            confidence_llm=70,
            generation_time_ms=34000.0,
            model_used="llama3.2",
            document_ids=json.dumps([]),
        )
        db.add(fresh_row)
        db.commit()

        # Mock settings to ensure TTL is 24 hours
        with patch(
            "ai_ready_rag.services.settings_service.get_cache_setting",
            side_effect=lambda key, default: default,
        ):
            service = CacheService(db)
            removed = service.evict_expired()
            assert removed == 2

            # Fresh entry should remain
            remaining = db.query(ResponseCache).count()
            assert remaining == 1


# =============================================================================
# Low-Confidence Exclusion Tests
# =============================================================================


class TestLowConfidenceExclusion:
    @pytest.mark.asyncio
    async def test_low_confidence_not_cached(self, db, mock_response):
        """Response with confidence < threshold is not stored."""
        mock_response.confidence.overall = 30  # Below default threshold of 40
        service = CacheService(db)
        await service.put("test query", [0.1] * 768, mock_response)

        # Verify nothing in SQLite
        count = db.query(ResponseCache).count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_high_confidence_cached(self, db, mock_response):
        """Response with confidence >= threshold is stored."""
        mock_response.confidence.overall = 75
        service = CacheService(db)
        await service.put("test query", [0.1] * 768, mock_response)

        # Verify stored in SQLite
        count = db.query(ResponseCache).count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_boundary_confidence_cached(self, db, mock_response):
        """Response with confidence == threshold is stored."""
        mock_response.confidence.overall = 40  # Exact threshold
        service = CacheService(db)
        await service.put("boundary query", [0.1] * 768, mock_response)

        count = db.query(ResponseCache).count()
        assert count == 1


# =============================================================================
# Access Verification Tests (CRITICAL)
# =============================================================================


class TestAccessVerification:
    @pytest.mark.asyncio
    async def test_verify_access_no_documents(self, db, sample_entry):
        """Entry with no document_ids passes verification."""
        sample_entry.document_ids = []
        service = CacheService(db)
        result = await service.verify_access(sample_entry, ["any_tag"])
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_access_user_has_tag(self, db, sample_entry, sample_documents, sample_tag):
        """User with matching tag can access cached response."""
        service = CacheService(db)
        result = await service.verify_access(sample_entry, ["hr"])
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_access_user_lacks_tag(
        self, db, sample_entry, sample_documents, admin_user
    ):
        """User without matching tag cannot access cached response."""
        # Create a different tag
        other_tag = Tag(
            name="finance",
            display_name="Finance",
            created_by=admin_user.id,
        )
        db.add(other_tag)
        db.flush()

        service = CacheService(db)
        result = await service.verify_access(sample_entry, ["finance"])
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_access_user_no_tags(self, db, sample_entry, sample_documents):
        """User with no matching tags cannot access."""
        service = CacheService(db)
        result = await service.verify_access(sample_entry, [])
        assert result is False


# =============================================================================
# Embedding Cache Tests
# =============================================================================


class TestEmbeddingCache:
    @pytest.mark.asyncio
    async def test_get_embedding_miss(self, db):
        """Query not in embedding cache returns None."""
        service = CacheService(db)
        result = await service.get_embedding("unknown query")
        assert result is None

    @pytest.mark.asyncio
    async def test_put_and_get_embedding(self, db):
        """Stored embedding is retrievable."""
        service = CacheService(db)
        embedding = [0.1] * 768

        await service.put_embedding("test query", embedding)
        result = await service.get_embedding("test query")

        assert result == embedding

    @pytest.mark.asyncio
    async def test_embedding_not_duplicated(self, db):
        """Putting same query twice doesn't create duplicate."""
        service = CacheService(db)
        embedding = [0.1] * 768

        await service.put_embedding("test query", embedding)
        await service.put_embedding("test query", embedding)

        count = db.query(EmbeddingCache).count()
        assert count == 1


# =============================================================================
# Statistics Tests
# =============================================================================


class TestCacheStats:
    def test_hit_rate_zero_total(self, db):
        """Hit rate is 0.0 when no operations (no division error)."""
        service = CacheService(db)
        stats = service.get_stats()
        assert stats.hit_rate == 0.0

    def test_stats_enabled_property(self, db):
        """Stats reflect cache enabled state."""
        service = CacheService(db)
        stats = service.get_stats()
        assert stats.enabled is True  # Default is True


# =============================================================================
# Batch Upload Context Tests
# =============================================================================


class TestUploadBatchContext:
    def test_pauses_invalidation(self, db):
        """Invalidation deferred during batch context."""
        service = CacheService(db)

        # Add an entry to cache memory
        with UploadBatchContext(service):
            # Try to invalidate during batch
            result = service.invalidate_all(reason="test")
            assert result == 0  # Nothing cleared
            assert service._invalidation_paused is True

    def test_resumes_on_exit(self, db):
        """Invalidation resumed after context exit."""
        service = CacheService(db)

        with UploadBatchContext(service):
            pass

        assert service._invalidation_paused is False

    def test_clears_on_exit_with_pending(self, db, mock_response):
        """Cache cleared on context exit when pending=True."""
        service = CacheService(db)

        # Pre-populate cache
        import asyncio

        asyncio.get_event_loop().run_until_complete(
            service.put("test query", [0.1] * 768, mock_response)
        )

        with UploadBatchContext(service) as batch:
            batch.pending_invalidation = True

        # Cache should be cleared after exit
        count = db.query(ResponseCache).count()
        assert count == 0

    def test_exception_safe(self, db):
        """Context resumes invalidation even on exception."""
        service = CacheService(db)

        try:
            with UploadBatchContext(service):
                raise ValueError("test error")
        except ValueError:
            pass

        assert service._invalidation_paused is False


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    def test_concurrent_puts(self, sample_entry):
        """Multiple concurrent puts don't corrupt cache."""
        cache = MemoryCache(max_size=100)

        def put_entry(index):
            entry = CacheEntry(
                query_hash=f"hash_{index}",
                query_text=sample_entry.query_text,
                query_embedding=sample_entry.query_embedding,
                answer=sample_entry.answer,
                sources=sample_entry.sources,
                confidence_overall=sample_entry.confidence_overall,
                confidence_retrieval=sample_entry.confidence_retrieval,
                confidence_coverage=sample_entry.confidence_coverage,
                confidence_llm=sample_entry.confidence_llm,
                generation_time_ms=sample_entry.generation_time_ms,
                model_used=sample_entry.model_used,
                created_at=sample_entry.created_at,
                last_accessed_at=sample_entry.last_accessed_at,
                access_count=sample_entry.access_count,
                document_ids=sample_entry.document_ids,
            )
            cache.put(entry)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(put_entry, range(50)))

        # Should have 50 entries, no corruption
        assert len(cache) == 50

    def test_concurrent_gets(self, sample_entry):
        """Multiple concurrent gets return consistent results."""
        cache = MemoryCache()
        cache.put(sample_entry)

        results = []

        def get_entry():
            result = cache.get(sample_entry.query_hash)
            results.append(result)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(lambda _: get_entry(), range(50)))

        # All results should be non-None
        assert all(r is not None for r in results)
        assert len(results) == 50

    def test_concurrent_semantic_gets(self, sample_entry):
        """Multiple concurrent semantic gets don't corrupt cache."""
        cache = MemoryCache()
        cache.put(sample_entry)

        results = []
        exceptions = []

        def get_semantic():
            try:
                result = cache.get_semantic(sample_entry.query_embedding, threshold=0.95)
                results.append(result)
            except Exception as e:
                exceptions.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(lambda _: get_semantic(), range(50)))

        # No exceptions
        assert len(exceptions) == 0
        # All results should be non-None
        assert all(r is not None for r in results)
        assert len(results) == 50


# =============================================================================
# Database Table Tests
# =============================================================================


class TestDatabaseTables:
    def test_response_cache_table_exists(self, db):
        """ResponseCache table exists and is queryable."""
        count = db.query(ResponseCache).count()
        assert count >= 0  # No exception means table exists

    def test_embedding_cache_table_exists(self, db):
        """EmbeddingCache table exists and is queryable."""
        count = db.query(EmbeddingCache).count()
        assert count >= 0

    def test_cache_access_log_table_exists(self, db):
        """CacheAccessLog table exists and is queryable."""
        from ai_ready_rag.db.models import CacheAccessLog

        count = db.query(CacheAccessLog).count()
        assert count >= 0


# =============================================================================
# Cache Warming Tests
# =============================================================================


class TestCacheWarming:
    def test_get_top_queries_empty(self, db):
        """Empty access log returns empty list."""
        service = CacheService(db)
        result = service.get_top_queries(limit=10)
        assert result == []

    def test_get_top_queries_returns_sorted(self, db):
        """Queries returned in descending order by access count."""
        from ai_ready_rag.db.models import CacheAccessLog

        # Insert access logs with different counts
        # 5 accesses for "popular query"
        for _ in range(5):
            log = CacheAccessLog(
                query_hash="hash_popular",
                query_text="popular query",
                was_hit=True,
            )
            db.add(log)

        # 2 accesses for "less popular query"
        for _ in range(2):
            log = CacheAccessLog(
                query_hash="hash_less",
                query_text="less popular query",
                was_hit=True,
            )
            db.add(log)

        db.commit()

        service = CacheService(db)
        result = service.get_top_queries(limit=10)

        assert len(result) == 2
        assert result[0]["query_text"] == "popular query"
        assert result[0]["access_count"] == 5
        assert result[1]["query_text"] == "less popular query"
        assert result[1]["access_count"] == 2

    def test_get_top_queries_respects_limit(self, db):
        """Limit parameter restricts result count."""
        from ai_ready_rag.db.models import CacheAccessLog

        # Insert 5 different queries
        for i in range(5):
            log = CacheAccessLog(
                query_hash=f"hash_{i}",
                query_text=f"query {i}",
                was_hit=True,
            )
            db.add(log)
        db.commit()

        service = CacheService(db)
        result = service.get_top_queries(limit=3)

        assert len(result) == 3

    def test_get_top_queries_limit_exceeds_data(self, db):
        """Limit > available queries returns all available."""
        from ai_ready_rag.db.models import CacheAccessLog

        # Insert 2 different queries
        for i in range(2):
            log = CacheAccessLog(
                query_hash=f"hash_{i}",
                query_text=f"query {i}",
                was_hit=True,
            )
            db.add(log)
        db.commit()

        service = CacheService(db)
        result = service.get_top_queries(limit=10)

        assert len(result) == 2

    def test_get_top_queries_includes_last_accessed(self, db):
        """Result includes last_accessed timestamp."""
        from ai_ready_rag.db.models import CacheAccessLog

        log = CacheAccessLog(
            query_hash="hash_test",
            query_text="test query",
            was_hit=True,
        )
        db.add(log)
        db.commit()

        service = CacheService(db)
        result = service.get_top_queries(limit=10)

        assert len(result) == 1
        assert result[0]["last_accessed"] is not None
        assert isinstance(result[0]["last_accessed"], datetime)

    def test_get_top_queries_counts_hits_and_misses(self, db):
        """Both hits and misses contribute to access_count."""
        from ai_ready_rag.db.models import CacheAccessLog

        # Insert 3 hits + 2 misses for same query_hash
        for _ in range(3):
            log = CacheAccessLog(
                query_hash="hash_mixed",
                query_text="mixed query",
                was_hit=True,
            )
            db.add(log)

        for _ in range(2):
            log = CacheAccessLog(
                query_hash="hash_mixed",
                query_text="mixed query",
                was_hit=False,
            )
            db.add(log)

        db.commit()

        service = CacheService(db)
        result = service.get_top_queries(limit=10)

        assert len(result) == 1
        assert result[0]["access_count"] == 5  # 3 hits + 2 misses

    def test_auto_warm_enabled_property_default(self, db):
        """auto_warm_enabled returns True by default."""
        with patch(
            "ai_ready_rag.services.settings_service.get_cache_setting",
            side_effect=lambda key, default: default,
        ):
            service = CacheService(db)
            assert service.auto_warm_enabled is True

    def test_auto_warm_count_property_default(self, db):
        """auto_warm_count returns 20 by default."""
        with patch(
            "ai_ready_rag.services.settings_service.get_cache_setting",
            side_effect=lambda key, default: default,
        ):
            service = CacheService(db)
            assert service.auto_warm_count == 20

    def test_auto_warm_enabled_property_overridden(self, db):
        """auto_warm_enabled respects settings override."""
        with patch(
            "ai_ready_rag.services.settings_service.get_cache_setting",
            return_value=False,
        ):
            service = CacheService(db)
            assert service.auto_warm_enabled is False

    def test_auto_warm_count_property_overridden(self, db):
        """auto_warm_count respects settings override."""
        with patch(
            "ai_ready_rag.services.settings_service.get_cache_setting",
            return_value=50,
        ):
            service = CacheService(db)
            assert service.auto_warm_count == 50
