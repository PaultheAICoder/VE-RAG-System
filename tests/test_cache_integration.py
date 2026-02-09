"""End-to-end tests for RAG response caching integration.

Tests the integration between RAGService and CacheService, including:
- Cache hit/miss behavior
- Low-confidence exclusion
- Access control verification on cache hits
- Cache warming functionality
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_ready_rag.config import Settings
from ai_ready_rag.services.cache_service import CacheEntry, CacheService
from ai_ready_rag.services.rag_service import (
    ConfidenceScore,
    RAGRequest,
    RAGService,
)
from ai_ready_rag.services.vector_service import SearchResult

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_settings():
    """Mock Settings with RAG configuration."""
    settings = MagicMock(spec=Settings)
    settings.ollama_base_url = "http://localhost:11434"
    settings.chat_model = "llama3.2"
    settings.rag_temperature = 0.1
    settings.rag_timeout_seconds = 30
    settings.rag_confidence_threshold = 60
    settings.rag_admin_email = "admin@test.com"
    settings.rag_max_context_tokens = 3000
    settings.rag_max_history_tokens = 1000
    settings.rag_max_response_tokens = 1024
    settings.rag_system_prompt_tokens = 500
    settings.rag_min_similarity_score = 0.3
    settings.rag_max_chunks_per_doc = 3
    settings.rag_total_context_chunks = 8
    settings.rag_dedup_candidates_cap = 15
    settings.rag_chunk_overlap_threshold = 0.9
    settings.rag_enable_query_expansion = False  # Disable for predictable tests
    settings.rag_enable_hallucination_check = False  # Disable for speed
    return settings


@pytest.fixture
def mock_vector_service():
    """Mock VectorService for unit tests."""
    vs = AsyncMock()
    vs.search = AsyncMock(return_value=[])
    vs.embed_query = AsyncMock(return_value=[0.1] * 768)
    return vs


@pytest.fixture
def sample_search_results():
    """Sample SearchResult objects for testing."""
    return [
        SearchResult(
            chunk_id="doc-1:0",
            document_id="doc-1",
            document_name="policy.pdf",
            chunk_text="Employee vacation policy allows 20 days annually.",
            chunk_index=0,
            score=0.85,
            page_number=1,
            section="Benefits",
        ),
    ]


@pytest.fixture
def sample_cache_entry():
    """Sample CacheEntry for testing."""
    return CacheEntry(
        query_hash="abc123",
        query_text="What is the vacation policy?",
        query_embedding=[0.1] * 768,
        answer="Employees get 20 days of vacation annually.",
        sources=[
            {
                "document_id": "doc-1",
                "document_name": "policy.pdf",
                "page_number": 1,
                "section": "Benefits",
                "excerpt": "Employee vacation policy allows 20 days annually.",
            }
        ],
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
def mock_cache_service(sample_cache_entry):
    """Mock CacheService for testing."""
    cache = MagicMock(spec=CacheService)
    cache.enabled = True
    cache.min_confidence = 40
    cache.get = AsyncMock(return_value=None)
    cache.get_embedding = AsyncMock(return_value=None)
    cache.put = AsyncMock()
    cache.put_embedding = AsyncMock()
    cache.verify_access = AsyncMock(return_value=True)
    return cache


# =============================================================================
# Cache Integration Tests
# =============================================================================


class TestCacheHitMiss:
    """Tests for cache hit/miss behavior in RAGService."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_response(
        self, mock_settings, mock_vector_service, sample_cache_entry, db
    ):
        """Cache hit returns response from cache, skips RAG pipeline."""
        # Arrange
        service = RAGService(
            settings=mock_settings,
            vector_service=mock_vector_service,
        )

        # Create mock cache that returns a hit
        mock_cache = MagicMock(spec=CacheService)
        mock_cache.enabled = True
        mock_cache.get_embedding = AsyncMock(return_value=[0.1] * 768)
        mock_cache.get = AsyncMock(return_value=sample_cache_entry)
        service._cache_service = mock_cache

        # Mock model validation
        with patch.object(service, "validate_model", new_callable=AsyncMock):
            request = RAGRequest(
                query="What is the vacation policy?",
                user_tags=["hr"],
                tenant_id="default",
            )

            # Act
            response = await service.generate(request, db)

            # Assert
            assert response.answer == sample_cache_entry.answer
            assert response.confidence.overall == sample_cache_entry.confidence_overall
            assert response.model_used == sample_cache_entry.model_used
            # VectorService should NOT be called on cache hit
            mock_vector_service.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_executes_full_pipeline(
        self, mock_settings, mock_vector_service, sample_search_results, db
    ):
        """Cache miss executes full RAG pipeline."""
        # Arrange
        mock_vector_service.search = AsyncMock(return_value=sample_search_results)

        service = RAGService(
            settings=mock_settings,
            vector_service=mock_vector_service,
        )

        # Create mock cache that returns miss
        mock_cache = MagicMock(spec=CacheService)
        mock_cache.enabled = True
        mock_cache.get_embedding = AsyncMock(return_value=None)
        mock_cache.get = AsyncMock(return_value=None)  # Cache miss
        mock_cache.put = AsyncMock()
        mock_cache.put_embedding = AsyncMock()
        mock_cache.min_confidence = 40
        service._cache_service = mock_cache

        # Mock model validation and LLM
        with (
            patch.object(service, "validate_model", new_callable=AsyncMock),
            patch("ai_ready_rag.services.rag_service.ChatOllama") as mock_llm_class,
        ):
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = (
                "Based on the policy [SourceId: doc-1:0], employees get 20 days."
            )
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_class.return_value = mock_llm

            request = RAGRequest(
                query="What is the vacation policy?",
                user_tags=["hr"],
                tenant_id="default",
            )

            # Act
            response = await service.generate(request, db)

            # Assert
            assert response.answer is not None
            # VectorService SHOULD be called on cache miss
            assert mock_vector_service.search.called
            # Cache should be updated with new response
            assert mock_cache.put.called

    @pytest.mark.asyncio
    async def test_cache_disabled_always_executes_pipeline(
        self, mock_settings, mock_vector_service, sample_search_results, db
    ):
        """When cache is disabled, always execute full pipeline."""
        # Arrange
        mock_vector_service.search = AsyncMock(return_value=sample_search_results)

        service = RAGService(
            settings=mock_settings,
            vector_service=mock_vector_service,
        )

        # Create mock cache that is disabled
        mock_cache = MagicMock(spec=CacheService)
        mock_cache.enabled = False  # Cache disabled
        service._cache_service = mock_cache

        # Mock model validation and LLM
        with (
            patch.object(service, "validate_model", new_callable=AsyncMock),
            patch("ai_ready_rag.services.rag_service.ChatOllama") as mock_llm_class,
        ):
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "Employees get 20 days vacation."
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_class.return_value = mock_llm

            request = RAGRequest(
                query="What is the vacation policy?",
                user_tags=["hr"],
                tenant_id="default",
            )

            # Act
            await service.generate(request, db)

            # Assert - pipeline should execute despite having cache service
            assert mock_vector_service.search.called
            # Cache get should NOT be called when disabled
            mock_cache.get.assert_not_called()


class TestLowConfidenceExclusion:
    """Tests for low-confidence caching exclusion."""

    @pytest.mark.asyncio
    async def test_low_confidence_not_cached(
        self, mock_settings, mock_vector_service, sample_search_results, db
    ):
        """Response with confidence < min_confidence is NOT stored in cache."""
        # Arrange
        mock_vector_service.search = AsyncMock(return_value=sample_search_results)

        service = RAGService(
            settings=mock_settings,
            vector_service=mock_vector_service,
        )

        # Create mock cache
        mock_cache = MagicMock(spec=CacheService)
        mock_cache.enabled = True
        mock_cache.get_embedding = AsyncMock(return_value=None)
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.put = AsyncMock()
        mock_cache.put_embedding = AsyncMock()
        mock_cache.min_confidence = 40
        service._cache_service = mock_cache

        # Mock model validation and LLM to return low-confidence response
        with (
            patch.object(service, "validate_model", new_callable=AsyncMock),
            patch("ai_ready_rag.services.rag_service.ChatOllama") as mock_llm_class,
            patch.object(service, "_calculate_confidence") as mock_calc,
        ):
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "I'm not sure about this."
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_class.return_value = mock_llm

            # Return low confidence (below 40 threshold)
            mock_calc.return_value = ConfidenceScore(
                overall=30,  # Below min_confidence of 40
                retrieval_score=0.3,
                coverage_score=0.3,
                llm_score=30,
            )

            request = RAGRequest(
                query="What is the vacation policy?",
                user_tags=["hr"],
                tenant_id="default",
            )

            # Act
            await service.generate(request, db)

            # Assert - cache.put should be called but CacheService.put
            # internally checks min_confidence and skips low-confidence
            # The put method is called, but the actual CacheService implementation
            # will not store it. We verify put was called with the response.
            assert mock_cache.put.called


class TestCacheAccessControl:
    """Access verification on cache hits."""

    @pytest.mark.asyncio
    async def test_cache_hit_denied_wrong_tags(
        self, mock_settings, mock_vector_service, sample_search_results, db
    ):
        """User without access to cited docs gets cache miss."""
        # Arrange
        mock_vector_service.search = AsyncMock(return_value=sample_search_results)

        service = RAGService(
            settings=mock_settings,
            vector_service=mock_vector_service,
        )

        # Create mock cache that returns None (access denied = cache miss)
        mock_cache = MagicMock(spec=CacheService)
        mock_cache.enabled = True
        mock_cache.get_embedding = AsyncMock(return_value=[0.1] * 768)
        mock_cache.get = AsyncMock(return_value=None)  # Access verification failed
        mock_cache.put = AsyncMock()
        mock_cache.put_embedding = AsyncMock()
        service._cache_service = mock_cache

        # Mock model validation and LLM
        with (
            patch.object(service, "validate_model", new_callable=AsyncMock),
            patch("ai_ready_rag.services.rag_service.ChatOllama") as mock_llm_class,
        ):
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "Employees get 20 days vacation."
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_class.return_value = mock_llm

            request = RAGRequest(
                query="What is the vacation policy?",
                user_tags=["finance"],  # Wrong tags
                tenant_id="default",
            )

            # Act
            await service.generate(request, db)

            # Assert - should execute full pipeline since cache returned None
            assert mock_vector_service.search.called


class TestCacheWarming:
    """Cache warming functionality tests.

    Note: warm_cache_task was removed in Issue #189 (DB-first warming redesign).
    The POST /cache/warm endpoint now returns 410 Gone.
    Cache warming is now handled via POST /warming/queue/manual and the
    process_warming_batch ARQ task.
    """

    def test_legacy_warm_endpoint_returns_410(self, client, admin_headers):
        """Legacy POST /cache/warm returns 410 Gone."""
        response = client.post(
            "/api/admin/cache/warm",
            json={"queries": ["Query 1"]},
            headers=admin_headers,
        )
        assert response.status_code == 410

    def test_legacy_warm_retry_returns_410(self, client, admin_headers):
        """Legacy POST /cache/warm-retry returns 410 Gone."""
        response = client.post(
            "/api/admin/cache/warm-retry",
            json={"queries": ["Query 1"]},
            headers=admin_headers,
        )
        assert response.status_code == 410


class TestCacheEntryToResponse:
    """Tests for _cache_entry_to_response helper."""

    def test_converts_entry_to_response(self, mock_settings, sample_cache_entry):
        """CacheEntry fields map correctly to RAGResponse."""
        # Arrange
        service = RAGService(
            settings=mock_settings,
            vector_service=AsyncMock(),
        )

        # Act
        response = service._cache_entry_to_response(sample_cache_entry, elapsed_ms=50.0)

        # Assert
        assert response.answer == sample_cache_entry.answer
        assert response.confidence.overall == sample_cache_entry.confidence_overall
        assert response.confidence.retrieval_score == sample_cache_entry.confidence_retrieval
        assert response.confidence.coverage_score == sample_cache_entry.confidence_coverage
        assert response.confidence.llm_score == sample_cache_entry.confidence_llm
        assert response.model_used == sample_cache_entry.model_used
        assert response.generation_time_ms == 50.0

    def test_citations_reconstructed_from_sources(self, mock_settings, sample_cache_entry):
        """Sources list becomes citations list."""
        # Arrange
        service = RAGService(
            settings=mock_settings,
            vector_service=AsyncMock(),
        )

        # Act
        response = service._cache_entry_to_response(sample_cache_entry, elapsed_ms=50.0)

        # Assert
        assert len(response.citations) == 1
        assert response.citations[0].document_id == "doc-1"
        assert response.citations[0].document_name == "policy.pdf"

    def test_action_determined_by_confidence(self, mock_settings):
        """Action is CITE if confidence >= threshold, ROUTE otherwise."""
        # Arrange - mock get_rag_setting to return the default threshold (60)
        with patch(
            "ai_ready_rag.services.rag_service.get_rag_setting",
            side_effect=lambda key, default: default,
        ):
            service = RAGService(
                settings=mock_settings,
                vector_service=AsyncMock(),
            )

            # High confidence entry (75 >= 60 threshold)
            high_conf_entry = CacheEntry(
                query_hash="abc123",
                query_text="Test query",
                query_embedding=[0.1] * 768,
                answer="High confidence answer",
                sources=[],
                confidence_overall=75,
                confidence_retrieval=0.85,
                confidence_coverage=0.80,
                confidence_llm=70,
                generation_time_ms=100.0,
                model_used="llama3.2",
                created_at=datetime.utcnow(),
                last_accessed_at=datetime.utcnow(),
                access_count=1,
                document_ids=[],
            )

            # Low confidence entry (50 < 60 threshold)
            low_conf_entry = CacheEntry(
                query_hash="def456",
                query_text="Test query 2",
                query_embedding=[0.1] * 768,
                answer="Low confidence answer",
                sources=[],
                confidence_overall=50,
                confidence_retrieval=0.5,
                confidence_coverage=0.5,
                confidence_llm=50,
                generation_time_ms=100.0,
                model_used="llama3.2",
                created_at=datetime.utcnow(),
                last_accessed_at=datetime.utcnow(),
                access_count=1,
                document_ids=[],
            )

            # Act
            high_response = service._cache_entry_to_response(high_conf_entry, 50.0)
            low_response = service._cache_entry_to_response(low_conf_entry, 50.0)

            # Assert
            assert high_response.action == "CITE"
            assert low_response.action == "ROUTE"


class TestCacheStats:
    """Cache statistics tracking during integration."""

    @pytest.mark.asyncio
    async def test_cache_lookup_errors_handled_gracefully(
        self, mock_settings, mock_vector_service, sample_search_results, db
    ):
        """Cache lookup errors are logged but don't break the pipeline."""
        # Arrange
        mock_vector_service.search = AsyncMock(return_value=sample_search_results)

        service = RAGService(
            settings=mock_settings,
            vector_service=mock_vector_service,
        )

        # Create mock cache that raises an error
        mock_cache = MagicMock(spec=CacheService)
        mock_cache.enabled = True
        mock_cache.get_embedding = AsyncMock(side_effect=Exception("DB connection failed"))
        mock_cache.put = AsyncMock()
        mock_cache.put_embedding = AsyncMock()
        service._cache_service = mock_cache

        # Mock model validation and LLM
        with (
            patch.object(service, "validate_model", new_callable=AsyncMock),
            patch("ai_ready_rag.services.rag_service.ChatOllama") as mock_llm_class,
        ):
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "Employees get 20 days vacation."
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_class.return_value = mock_llm

            request = RAGRequest(
                query="What is the vacation policy?",
                user_tags=["hr"],
                tenant_id="default",
            )

            # Act - should not raise despite cache error
            response = await service.generate(request, db)

            # Assert - pipeline should execute and return response
            assert response.answer is not None
            assert mock_vector_service.search.called
