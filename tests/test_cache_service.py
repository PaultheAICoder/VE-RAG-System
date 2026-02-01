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
    response.confidence.retrieval = 0.85
    response.confidence.coverage = 0.80
    response.confidence.llm = 70
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
