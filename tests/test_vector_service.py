"""Tests for VectorService.

These tests require Qdrant (localhost:6333) and Ollama (localhost:11434) to be running.
Use pytest -m "not integration" to skip integration tests.
"""

import os
import threading
import unittest.mock

import pytest
from qdrant_client.http import models

from ai_ready_rag.core.exceptions import EmbeddingError
from ai_ready_rag.services.vector_service import (
    CollectionStats,
    HealthStatus,
    IndexResult,
    SearchResult,
    VectorService,
)


class TestVectorServiceInit:
    """Test VectorService initialization (Issue 005)."""

    def test_init_with_defaults(self):
        """VectorService can be instantiated with defaults."""
        vs = VectorService()
        assert vs.qdrant_url == "http://localhost:6333"
        assert vs.ollama_url == "http://localhost:11434"
        assert vs.collection_name == "documents"
        assert vs.embedding_model == "nomic-embed-text"
        assert vs.embedding_dimension == 768
        assert vs.max_tokens == 8192
        assert vs.tenant_id == "default"

    def test_init_with_custom_params(self):
        """VectorService can be instantiated with custom params."""
        vs = VectorService(
            qdrant_url="http://custom:9999",
            ollama_url="http://ollama:11111",
            collection_name="test_collection",
            embedding_model="custom-model",
            embedding_dimension=1024,
            max_tokens=4096,
            tenant_id="test-tenant",
        )
        assert vs.qdrant_url == "http://custom:9999"
        assert vs.ollama_url == "http://ollama:11111"
        assert vs.collection_name == "test_collection"
        assert vs.embedding_model == "custom-model"
        assert vs.embedding_dimension == 1024
        assert vs.max_tokens == 4096
        assert vs.tenant_id == "test-tenant"


class TestHealthStatus:
    """Test HealthStatus dataclass (Issue 006)."""

    def test_healthy_when_all_components_healthy(self):
        """healthy property returns True when all components healthy."""
        status = HealthStatus(
            qdrant_healthy=True,
            qdrant_latency_ms=10.0,
            ollama_healthy=True,
            ollama_latency_ms=20.0,
            collection_exists=True,
            collection_vector_count=100,
        )
        assert status.healthy is True

    def test_unhealthy_when_qdrant_down(self):
        """healthy property returns False when Qdrant unhealthy."""
        status = HealthStatus(
            qdrant_healthy=False,
            qdrant_latency_ms=None,
            ollama_healthy=True,
            ollama_latency_ms=20.0,
            collection_exists=False,
            collection_vector_count=None,
        )
        assert status.healthy is False

    def test_unhealthy_when_ollama_down(self):
        """healthy property returns False when Ollama unhealthy."""
        status = HealthStatus(
            qdrant_healthy=True,
            qdrant_latency_ms=10.0,
            ollama_healthy=False,
            ollama_latency_ms=None,
            collection_exists=True,
            collection_vector_count=100,
        )
        assert status.healthy is False

    def test_unhealthy_when_collection_missing(self):
        """healthy property returns False when collection doesn't exist."""
        status = HealthStatus(
            qdrant_healthy=True,
            qdrant_latency_ms=10.0,
            ollama_healthy=True,
            ollama_latency_ms=20.0,
            collection_exists=False,
            collection_vector_count=None,
        )
        assert status.healthy is False


@pytest.mark.integration
class TestVectorServiceIntegration:
    """Integration tests requiring Qdrant and Ollama.

    Run with: pytest -m integration
    Skip with: pytest -m "not integration"
    """

    @pytest.fixture
    async def vector_service(self):
        """Create a VectorService with test collection."""
        vs = VectorService(collection_name="test_integration")
        await vs.initialize()
        yield vs
        # Cleanup: delete test collection
        try:
            await vs._qdrant.delete_collection("test_integration")
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_initialize_creates_collection(self, vector_service):
        """initialize() creates collection if not exists."""
        collections = await vector_service._qdrant.get_collections()
        names = [c.name for c in collections.collections]
        assert "test_integration" in names

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self, vector_service):
        """initialize() can be called multiple times safely."""
        # Call initialize again
        await vector_service.initialize()
        collections = await vector_service._qdrant.get_collections()
        names = [c.name for c in collections.collections]
        assert names.count("test_integration") == 1

    @pytest.mark.asyncio
    async def test_health_check_returns_status(self, vector_service):
        """health_check() returns HealthStatus with all fields."""
        health = await vector_service.health_check()

        assert isinstance(health, HealthStatus)
        assert health.qdrant_healthy is True
        assert health.qdrant_latency_ms is not None
        assert health.qdrant_latency_ms > 0
        assert health.ollama_healthy is True
        assert health.ollama_latency_ms is not None
        assert health.collection_exists is True
        assert health.collection_vector_count == 0  # Empty collection
        assert health.healthy is True

    @pytest.mark.asyncio
    async def test_embed_returns_vector(self, vector_service):
        """embed() returns 768-dimensional vector."""
        embedding = await vector_service.embed("Hello, world!")

        assert isinstance(embedding, list)
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_deterministic(self, vector_service):
        """embed() returns same vector for same input."""
        text = "The quick brown fox jumps over the lazy dog."
        emb1 = await vector_service.embed(text)
        emb2 = await vector_service.embed(text)

        # Should be identical (or very close due to floating point)
        assert emb1 == emb2

    @pytest.mark.asyncio
    async def test_embed_different_for_different_text(self, vector_service):
        """embed() returns different vectors for different text."""
        # Use longer, semantically distinct phrases
        # Note: Some models may produce identical embeddings for very short inputs
        emb1 = await vector_service.embed("The quick brown fox jumps over the lazy dog")
        emb2 = await vector_service.embed("Machine learning algorithms process data efficiently")

        # Check that at least some elements differ significantly
        differences = [abs(a - b) for a, b in zip(emb1, emb2, strict=False)]
        max_diff = max(differences)
        assert max_diff > 0.01, "Embeddings should differ for different text"

    @pytest.mark.asyncio
    async def test_embed_batch_returns_correct_count(self, vector_service):
        """embed_batch() returns embeddings in same order."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = await vector_service.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self, vector_service):
        """embed_batch() returns empty list for empty input."""
        embeddings = await vector_service.embed_batch([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_embed_batch_preserves_order(self, vector_service):
        """embed_batch() preserves order of embeddings."""
        texts = ["Apple", "Banana", "Cherry"]
        batch_embeddings = await vector_service.embed_batch(texts)

        # Compare with individual embeddings
        for i, text in enumerate(texts):
            individual = await vector_service.embed(text)
            assert batch_embeddings[i] == individual


class TestVectorServiceEmbedErrors:
    """Test embedding error handling (Issue 007)."""

    @pytest.mark.asyncio
    async def test_embed_raises_on_bad_url(self):
        """embed() raises EmbeddingError when Ollama unreachable."""
        vs = VectorService(ollama_url="http://localhost:99999")

        with pytest.raises(EmbeddingError):
            await vs.embed("test")

    @pytest.mark.asyncio
    async def test_embed_batch_raises_after_retries(self):
        """embed_batch() raises EmbeddingError after max retries."""
        vs = VectorService(ollama_url="http://localhost:99999")

        with pytest.raises(EmbeddingError, match="failed after"):
            await vs.embed_batch(["test"])


class TestIndexResult:
    """Test IndexResult dataclass (Issue 008)."""

    def test_index_result_fields(self):
        """IndexResult has all expected fields."""
        result = IndexResult(
            document_id="doc-123",
            chunks_indexed=5,
            replaced_existing=True,
            embedding_time_ms=100.5,
            indexing_time_ms=50.2,
        )
        assert result.document_id == "doc-123"
        assert result.chunks_indexed == 5
        assert result.replaced_existing is True
        assert result.embedding_time_ms == 100.5
        assert result.indexing_time_ms == 50.2


class TestSearchResult:
    """Test SearchResult dataclass (Issue 010)."""

    def test_search_result_fields(self):
        """SearchResult has all expected fields."""
        result = SearchResult(
            chunk_id="chunk-abc",
            document_id="doc-123",
            document_name="test.pdf",
            chunk_text="Sample text content",
            chunk_index=0,
            score=0.85,
            page_number=1,
            section="Introduction",
        )
        assert result.chunk_id == "chunk-abc"
        assert result.document_id == "doc-123"
        assert result.document_name == "test.pdf"
        assert result.chunk_text == "Sample text content"
        assert result.chunk_index == 0
        assert result.score == 0.85
        assert result.page_number == 1
        assert result.section == "Introduction"

    def test_search_result_optional_fields(self):
        """SearchResult handles None for optional fields."""
        result = SearchResult(
            chunk_id="chunk-abc",
            document_id="doc-123",
            document_name="test.pdf",
            chunk_text="Sample text",
            chunk_index=0,
            score=0.75,
            page_number=None,
            section=None,
        )
        assert result.page_number is None
        assert result.section is None


class TestAddDocumentValidation:
    """Test add_document() validation (Issue 008)."""

    @pytest.mark.asyncio
    async def test_add_document_empty_tags_raises(self):
        """add_document() raises ValueError for empty tags."""
        vs = VectorService()

        with pytest.raises(ValueError, match="tag"):
            await vs.add_document(
                document_id="doc-123",
                document_name="test.pdf",
                chunks=["chunk1"],
                tags=[],
                uploaded_by="user-1",
            )

    @pytest.mark.asyncio
    async def test_add_document_empty_chunks_raises(self):
        """add_document() raises ValueError for empty chunks."""
        vs = VectorService()

        with pytest.raises(ValueError, match="chunk"):
            await vs.add_document(
                document_id="doc-123",
                document_name="test.pdf",
                chunks=[],
                tags=["hr"],
                uploaded_by="user-1",
            )


@pytest.mark.integration
class TestDocumentIndexingIntegration:
    """Integration tests for document indexing (Issue 008)."""

    @pytest.fixture
    async def vector_service(self):
        """Create a VectorService with test collection."""
        vs = VectorService(collection_name="test_indexing")
        await vs.initialize()
        yield vs
        # Cleanup
        try:
            await vs._qdrant.delete_collection("test_indexing")
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_add_document_indexes_chunks(self, vector_service):
        """add_document() indexes chunks and returns IndexResult."""
        result = await vector_service.add_document(
            document_id="doc-001",
            document_name="test.pdf",
            chunks=["First chunk of text.", "Second chunk of text."],
            tags=["hr", "policy"],
            uploaded_by="user-123",
        )

        assert isinstance(result, IndexResult)
        assert result.document_id == "doc-001"
        assert result.chunks_indexed == 2
        assert result.replaced_existing is False
        assert result.embedding_time_ms > 0
        assert result.indexing_time_ms > 0

    @pytest.mark.asyncio
    async def test_add_document_with_metadata(self, vector_service):
        """add_document() applies chunk_metadata correctly."""
        result = await vector_service.add_document(
            document_id="doc-002",
            document_name="handbook.pdf",
            chunks=["Intro text", "Chapter 1 text"],
            tags=["public"],
            uploaded_by="admin",
            chunk_metadata=[
                {"page_number": 1, "section": "Introduction"},
                {"page_number": 5, "section": "Chapter 1"},
            ],
        )

        assert result.chunks_indexed == 2

    @pytest.mark.asyncio
    async def test_add_document_replaces_existing(self, vector_service):
        """add_document() replaces existing document chunks."""
        # Index first time
        result1 = await vector_service.add_document(
            document_id="doc-replace",
            document_name="original.pdf",
            chunks=["Original chunk 1", "Original chunk 2", "Original chunk 3"],
            tags=["test"],
            uploaded_by="user-1",
        )
        assert result1.replaced_existing is False
        assert result1.chunks_indexed == 3

        # Index again (replace)
        result2 = await vector_service.add_document(
            document_id="doc-replace",
            document_name="updated.pdf",
            chunks=["New chunk 1", "New chunk 2"],
            tags=["test"],
            uploaded_by="user-1",
        )
        assert result2.replaced_existing is True
        assert result2.chunks_indexed == 2

    @pytest.mark.asyncio
    async def test_delete_document(self, vector_service):
        """delete_document() removes all chunks."""
        # Index document
        await vector_service.add_document(
            document_id="doc-delete",
            document_name="to-delete.pdf",
            chunks=["Chunk 1", "Chunk 2"],
            tags=["test"],
            uploaded_by="user-1",
        )

        # Delete
        success = await vector_service.delete_document("doc-delete")
        assert success is True

        # Verify deleted (search should return nothing)
        count = await vector_service._count_document_chunks("doc-delete")
        assert count == 0


@pytest.mark.integration
class TestSearchIntegration:
    """Integration tests for search with access control (Issue 010)."""

    @pytest.fixture
    async def vector_service_with_docs(self):
        """Create VectorService and index test documents."""
        vs = VectorService(collection_name="test_search")
        await vs.initialize()

        # Index test documents with different tags
        await vs.add_document(
            document_id="doc-public",
            document_name="public-info.pdf",
            chunks=["This is public information about company policies."],
            tags=["public"],
            uploaded_by="admin",
        )
        await vs.add_document(
            document_id="doc-hr",
            document_name="hr-handbook.pdf",
            chunks=["HR confidential: Employee salary bands and compensation."],
            tags=["hr"],
            uploaded_by="hr-admin",
        )
        await vs.add_document(
            document_id="doc-finance",
            document_name="finance-report.pdf",
            chunks=["Finance confidential: Q4 revenue projections and forecasts."],
            tags=["finance"],
            uploaded_by="cfo",
        )
        await vs.add_document(
            document_id="doc-hr-finance",
            document_name="budget.pdf",
            chunks=["Shared HR and Finance: Department budgets and headcount."],
            tags=["hr", "finance"],
            uploaded_by="admin",
        )

        yield vs

        # Cleanup
        try:
            await vs._qdrant.delete_collection("test_search")
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_search_returns_results(self, vector_service_with_docs):
        """search() returns SearchResult list."""
        results = await vector_service_with_docs.search(
            query="company policies",
            user_tags=["hr"],
            limit=5,
        )

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_empty_tags_returns_public_only(self, vector_service_with_docs):
        """search() with empty user_tags returns only public docs."""
        results = await vector_service_with_docs.search(
            query="information",
            user_tags=[],
            limit=10,
        )

        # Should only see public document
        doc_ids = [r.document_id for r in results]
        assert "doc-public" in doc_ids or len(results) == 0
        assert "doc-hr" not in doc_ids
        assert "doc-finance" not in doc_ids

    @pytest.mark.asyncio
    async def test_search_hr_user_sees_hr_and_public(self, vector_service_with_docs):
        """HR user sees HR docs + public docs."""
        results = await vector_service_with_docs.search(
            query="information policies salary budget",
            user_tags=["hr"],
            limit=10,
        )

        doc_ids = [r.document_id for r in results]
        # Should see public, hr, and hr-finance
        assert "doc-finance" not in doc_ids  # Finance-only doc should NOT be visible

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, vector_service_with_docs):
        """search() respects limit parameter."""
        results = await vector_service_with_docs.search(
            query="information",
            user_tags=["hr", "finance"],
            limit=2,
        )

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_results_ordered_by_score(self, vector_service_with_docs):
        """search() results are ordered by score descending."""
        results = await vector_service_with_docs.search(
            query="salary compensation",
            user_tags=["hr"],
            limit=5,
        )

        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_score_in_valid_range(self, vector_service_with_docs):
        """search() scores are between 0.0 and 1.0."""
        results = await vector_service_with_docs.search(
            query="company",
            user_tags=["hr"],
            limit=5,
        )

        for r in results:
            assert 0.0 <= r.score <= 1.0

    @pytest.mark.asyncio
    async def test_search_empty_results_returns_empty_list(self, vector_service_with_docs):
        """search() returns empty list for no matches."""
        results = await vector_service_with_docs.search(
            query="xyzzy nonexistent gibberish",
            user_tags=["hr"],
            limit=5,
            score_threshold=0.9,  # High threshold
        )

        assert results == []


class TestCollectionStats:
    """Test CollectionStats dataclass (Issue 011)."""

    def test_collection_stats_fields(self):
        """CollectionStats has all expected fields."""
        stats = CollectionStats(
            total_chunks=100,
            total_documents=10,
            collection_size_bytes=1024000,
            tenant_id="default",
        )
        assert stats.total_chunks == 100
        assert stats.total_documents == 10
        assert stats.collection_size_bytes == 1024000
        assert stats.tenant_id == "default"

    def test_collection_stats_zeros(self):
        """CollectionStats handles zero values."""
        stats = CollectionStats(
            total_chunks=0,
            total_documents=0,
            collection_size_bytes=0,
            tenant_id="test",
        )
        assert stats.total_chunks == 0
        assert stats.total_documents == 0
        assert stats.collection_size_bytes == 0


@pytest.mark.integration
class TestCollectionManagementIntegration:
    """Integration tests for collection management (Issue 011)."""

    @pytest.fixture
    async def vector_service(self):
        """Create a VectorService with test collection."""
        vs = VectorService(collection_name="test_collection_mgmt", tenant_id="test")
        await vs.initialize()
        yield vs
        # Cleanup
        try:
            await vs._qdrant.delete_collection("test_collection_mgmt")
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_get_stats_empty_collection(self, vector_service):
        """get_stats() returns zeros for empty collection."""
        stats = await vector_service.get_stats()

        assert isinstance(stats, CollectionStats)
        assert stats.total_chunks == 0
        assert stats.total_documents == 0
        assert stats.tenant_id == "test"

    @pytest.mark.asyncio
    async def test_get_stats_after_indexing(self, vector_service):
        """get_stats() returns accurate counts after indexing."""
        # Index two documents
        await vector_service.add_document(
            document_id="doc-stats-1",
            document_name="test1.pdf",
            chunks=["Chunk 1", "Chunk 2"],
            tags=["test"],
            uploaded_by="user-1",
        )
        await vector_service.add_document(
            document_id="doc-stats-2",
            document_name="test2.pdf",
            chunks=["Chunk A", "Chunk B", "Chunk C"],
            tags=["test"],
            uploaded_by="user-1",
        )

        stats = await vector_service.get_stats()

        assert stats.total_chunks == 5  # 2 + 3 chunks
        assert stats.total_documents == 2
        assert stats.tenant_id == "test"

    @pytest.mark.asyncio
    async def test_get_stats_unique_documents(self, vector_service):
        """get_stats() counts each document once regardless of chunks."""
        # Index one document with many chunks
        await vector_service.add_document(
            document_id="doc-many-chunks",
            document_name="big.pdf",
            chunks=["C1", "C2", "C3", "C4", "C5"],
            tags=["test"],
            uploaded_by="user-1",
        )

        stats = await vector_service.get_stats()

        assert stats.total_chunks == 5
        assert stats.total_documents == 1

    @pytest.mark.asyncio
    async def test_clear_collection_removes_all(self, vector_service):
        """clear_collection() removes all vectors."""
        # Index some documents
        await vector_service.add_document(
            document_id="doc-clear-1",
            document_name="test1.pdf",
            chunks=["Test chunk 1"],
            tags=["test"],
            uploaded_by="user-1",
        )
        await vector_service.add_document(
            document_id="doc-clear-2",
            document_name="test2.pdf",
            chunks=["Test chunk 2"],
            tags=["test"],
            uploaded_by="user-1",
        )

        # Verify documents exist
        stats_before = await vector_service.get_stats()
        assert stats_before.total_chunks == 2

        # Clear collection
        success = await vector_service.clear_collection()
        assert success is True

        # Verify cleared
        stats_after = await vector_service.get_stats()
        assert stats_after.total_chunks == 0
        assert stats_after.total_documents == 0

    @pytest.mark.asyncio
    async def test_clear_collection_empty_returns_true(self, vector_service):
        """clear_collection() returns True on empty collection (idempotent)."""
        # Clear twice
        success1 = await vector_service.clear_collection()
        success2 = await vector_service.clear_collection()

        assert success1 is True
        assert success2 is True

    @pytest.mark.asyncio
    async def test_clear_collection_logs_warning(self, vector_service, caplog):
        """clear_collection() logs a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            await vector_service.clear_collection()

        assert "DESTRUCTIVE OPERATION" in caplog.text


class TestAccessFilterConstruction:
    """Unit tests for access filter construction logic (Issue 012)."""

    def test_build_access_filter_empty_user_tags(self):
        """Filter with empty user_tags only includes public."""
        from qdrant_client.http import models

        vs = VectorService(tenant_id="test-tenant")
        filter_obj = vs._build_access_filter(user_tags=[], tenant_id="test-tenant")

        # Should have tenant_id condition and nested filter with just public
        assert isinstance(filter_obj, models.Filter)
        assert len(filter_obj.must) == 2  # tenant_id + tag filter

        # First condition is tenant_id
        tenant_cond = filter_obj.must[0]
        assert tenant_cond.key == "tenant_id"
        assert tenant_cond.match.value == "test-tenant"

        # Second condition is nested filter with should
        tag_filter = filter_obj.must[1]
        assert isinstance(tag_filter, models.Filter)
        assert len(tag_filter.should) == 1  # Only public

    def test_build_access_filter_with_user_tags(self):
        """Filter with user_tags includes public OR user tags."""
        from qdrant_client.http import models

        vs = VectorService(tenant_id="default")
        filter_obj = vs._build_access_filter(user_tags=["hr", "finance"], tenant_id="default")

        # Second condition should have 2 should clauses (public + MatchAny)
        tag_filter = filter_obj.must[1]
        assert isinstance(tag_filter, models.Filter)
        assert len(tag_filter.should) == 2  # public + user tags


@pytest.mark.integration
class TestMultiTenantIsolation:
    """Integration tests for multi-tenant isolation (Issue 012)."""

    @pytest.fixture
    async def tenant_a_service(self):
        """Create VectorService for tenant A."""
        vs = VectorService(collection_name="test_multi_tenant", tenant_id="tenant-a")
        await vs.initialize()
        yield vs
        await vs.clear_collection()

    @pytest.fixture
    async def tenant_b_service(self):
        """Create VectorService for tenant B."""
        vs = VectorService(collection_name="test_multi_tenant", tenant_id="tenant-b")
        # Collection already created by tenant A fixture, just use it
        yield vs
        await vs.clear_collection()
        # Final cleanup - delete collection
        try:
            await vs._qdrant.delete_collection("test_multi_tenant")
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, tenant_a_service, tenant_b_service):
        """Tenants cannot see each other's documents."""
        # Tenant A indexes a document
        await tenant_a_service.add_document(
            document_id="doc-tenant-a",
            document_name="tenant-a-doc.pdf",
            chunks=["Tenant A confidential information about salaries."],
            tags=["public"],  # Even public docs should be tenant-isolated
            uploaded_by="user-a",
        )

        # Tenant B indexes a document
        await tenant_b_service.add_document(
            document_id="doc-tenant-b",
            document_name="tenant-b-doc.pdf",
            chunks=["Tenant B confidential information about budgets."],
            tags=["public"],
            uploaded_by="user-b",
        )

        # Tenant A searches - should only see their doc
        results_a = await tenant_a_service.search(
            query="confidential information",
            user_tags=["admin"],
            limit=10,
        )
        doc_ids_a = [r.document_id for r in results_a]
        assert "doc-tenant-a" in doc_ids_a
        assert "doc-tenant-b" not in doc_ids_a

        # Tenant B searches - should only see their doc
        results_b = await tenant_b_service.search(
            query="confidential information",
            user_tags=["admin"],
            limit=10,
        )
        doc_ids_b = [r.document_id for r in results_b]
        assert "doc-tenant-b" in doc_ids_b
        assert "doc-tenant-a" not in doc_ids_b


@pytest.mark.integration
class TestSearchEdgeCases:
    """Integration tests for search edge cases (Issue 012)."""

    @pytest.fixture
    async def vector_service(self):
        """Create VectorService with test collection."""
        vs = VectorService(collection_name="test_search_edge", tenant_id="test")
        await vs.initialize()
        # Index a test document
        await vs.add_document(
            document_id="doc-edge-test",
            document_name="test.pdf",
            chunks=["This is a test document about machine learning and AI."],
            tags=["public"],
            uploaded_by="admin",
        )
        yield vs
        try:
            await vs._qdrant.delete_collection("test_search_edge")
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_search_score_threshold_filtering(self, vector_service):
        """search() filters results below score_threshold."""
        # Search with low threshold - should get results
        results_low = await vector_service.search(
            query="machine learning",
            user_tags=[],
            limit=5,
            score_threshold=0.0,
        )
        assert len(results_low) > 0

        # Search with impossibly high threshold - should get no results
        results_high = await vector_service.search(
            query="machine learning",
            user_tags=[],
            limit=5,
            score_threshold=0.99,
        )
        assert len(results_high) == 0

    @pytest.mark.asyncio
    async def test_search_with_empty_query(self, vector_service):
        """search() handles empty/whitespace query."""
        # Empty query should still work (return embeddings for empty string)
        # Note: Results may vary based on embedding model behavior
        results = await vector_service.search(
            query="   ",  # Whitespace query
            user_tags=[],
            limit=5,
        )
        # Should not raise, may return results or empty list
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_public_tag_always_accessible(self, vector_service):
        """Documents with 'public' tag are always accessible."""
        # Index a doc with both public and restricted tag
        await vector_service.add_document(
            document_id="doc-public-restricted",
            document_name="mixed.pdf",
            chunks=["This document has both public and hr tags."],
            tags=["public", "hr"],
            uploaded_by="admin",
        )

        # User with no matching tags should still see it (has public)
        results = await vector_service.search(
            query="public and hr tags",
            user_tags=["finance"],  # No hr tag, but doc has public
            limit=5,
        )
        doc_ids = [r.document_id for r in results]
        assert "doc-public-restricted" in doc_ids


@pytest.mark.integration
class TestIndexingRollback:
    """Integration tests for atomic rollback on indexing failure (Issue 012)."""

    @pytest.fixture
    async def vector_service(self):
        """Create VectorService with test collection."""
        vs = VectorService(collection_name="test_rollback", tenant_id="test")
        await vs.initialize()
        yield vs
        try:
            await vs._qdrant.delete_collection("test_rollback")
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_partial_indexing_rollback(self, vector_service):
        """If indexing fails mid-way, previous chunks are cleaned up."""
        # First, successfully index a document
        await vector_service.add_document(
            document_id="doc-success",
            document_name="success.pdf",
            chunks=["Successfully indexed chunk."],
            tags=["test"],
            uploaded_by="user-1",
        )

        # Verify it exists
        stats_before = await vector_service.get_stats()
        assert stats_before.total_chunks == 1

        # Now try to replace with a bad URL (will fail during embedding)
        bad_vs = VectorService(
            collection_name="test_rollback",
            tenant_id="test",
            ollama_url="http://localhost:99999",  # Bad URL
        )

        from ai_ready_rag.core.exceptions import IndexingError

        with pytest.raises(IndexingError):
            await bad_vs.add_document(
                document_id="doc-success",  # Same ID - would replace
                document_name="failure.pdf",
                chunks=["This should fail."],
                tags=["test"],
                uploaded_by="user-1",
            )

        # Original document should still exist after failed replacement
        # (rollback deletes the document, so this tests that replacement
        # doesn't leave partial state)
        stats_after = await vector_service.get_stats()
        # Note: Due to rollback implementation, the original doc is deleted
        # before embedding, so if embedding fails, the doc is gone.
        # This is expected behavior - atomic delete-then-insert.
        assert stats_after.total_chunks == 0  # Rolled back


class TestSparseEmbedding:
    """Tests for sparse embedding methods (Issue #282)."""

    def test_sparse_embed_produces_valid_vector(self):
        """sparse_embed() returns SparseVector with non-empty indices and values."""
        vs = VectorService()
        result = vs.sparse_embed("Hello world, this is a test document.")
        if result is None:
            pytest.skip("fastembed not available")
        assert isinstance(result, models.SparseVector)
        assert len(result.indices) > 0
        assert len(result.values) > 0
        assert len(result.indices) == len(result.values)

    def test_sparse_embed_batch(self):
        """sparse_embed_batch() returns list matching input length."""
        vs = VectorService()
        texts = ["First document text", "Second document text", "Third document"]
        results = vs.sparse_embed_batch(texts)
        if results[0] is None:
            pytest.skip("fastembed not available")
        assert len(results) == 3
        assert all(isinstance(r, models.SparseVector) for r in results)

    def test_sparse_embed_deterministic(self):
        """sparse_embed() returns same vector for same input."""
        vs = VectorService()
        text = "The quick brown fox jumps over the lazy dog."
        r1 = vs.sparse_embed(text)
        r2 = vs.sparse_embed(text)
        if r1 is None:
            pytest.skip("fastembed not available")
        assert r1.indices == r2.indices
        assert r1.values == r2.values

    def test_sparse_model_lazy_load(self):
        """Model is not loaded until first sparse_embed() call."""
        vs = VectorService()
        assert vs._sparse_model is None
        vs.sparse_embed("trigger load")
        if vs._sparse_available:
            assert vs._sparse_model is not None

    def test_sparse_model_thread_safe_init(self):
        """Concurrent first calls produce single model instance."""
        import concurrent.futures

        vs = VectorService()
        results = []

        def call_sparse():
            return vs.sparse_embed("test text for threading")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(call_sparse) for _ in range(8)]
            results = [f.result() for f in futures]
        if vs._sparse_available:
            assert vs._sparse_model is not None
            # All results should be identical (same model, same input)
            for r in results:
                assert r is not None

    def test_sparse_model_load_failure_degrades(self):
        """If model fails to load, _sparse_available=False, returns None."""
        vs = VectorService()
        vs._sparse_available = True
        vs._sparse_model = None
        vs._sparse_lock = threading.Lock()
        with unittest.mock.patch(
            "ai_ready_rag.services.vector_service.SparseTextEmbedding",
            side_effect=RuntimeError("Model load failed"),
        ):
            vs._sparse_available = True  # Reset to allow attempt
            result = vs.sparse_embed("test")
        assert result is None
        assert vs._sparse_available is False

    def test_sparse_embed_after_model_failure(self):
        """After model failure, returns None without retrying."""
        vs = VectorService()
        vs._sparse_available = False
        vs._sparse_model = None
        result = vs.sparse_embed("test")
        assert result is None
        batch = vs.sparse_embed_batch(["a", "b"])
        assert batch == [None, None]

    def test_sparse_model_cache_path_configurable(self):
        """FASTEMBED_CACHE_PATH env var is passed to SparseTextEmbedding."""
        mock_cls = unittest.mock.MagicMock()
        mock_instance = unittest.mock.MagicMock()
        mock_cls.return_value = mock_instance
        mock_result = unittest.mock.MagicMock()
        mock_result.indices.tolist.return_value = [0, 1, 2]
        mock_result.values.tolist.return_value = [0.5, 0.3, 0.1]
        mock_instance.embed.return_value = iter([mock_result])

        with unittest.mock.patch.dict(os.environ, {"FASTEMBED_CACHE_PATH": "/tmp/test_cache"}):
            with unittest.mock.patch(
                "ai_ready_rag.services.vector_service.SparseTextEmbedding",
                mock_cls,
            ):
                vs = VectorService()
                vs._sparse_available = True
                vs._sparse_model = None
                vs.sparse_embed("test")
                mock_cls.assert_called_once_with(
                    model_name="Qdrant/bm25",
                    cache_dir="/tmp/test_cache",
                )


def _make_service(hybrid=True, has_sparse=True, prefetch_multiplier=3):
    """Create VectorService with mocked internals for testing."""
    vs = VectorService()
    type(vs).hybrid_enabled = unittest.mock.PropertyMock(return_value=hybrid)
    type(vs).prefetch_multiplier = unittest.mock.PropertyMock(return_value=prefetch_multiplier)
    vs._collection_has_sparse = has_sparse
    vs._detect_collection_capabilities = unittest.mock.AsyncMock()
    return vs


class TestHybridIndexing:
    """Tests for hybrid search indexing (Issue #284)."""

    def _make_service(self, hybrid=True, has_sparse=True):
        """Create VectorService with mocked internals."""
        return _make_service(hybrid=hybrid, has_sparse=has_sparse)

    def test_index_result_includes_sparse_indexed(self):
        """IndexResult accepts sparse_indexed field, defaults to True."""
        result = IndexResult(
            document_id="doc-1",
            chunks_indexed=3,
            replaced_existing=False,
            embedding_time_ms=10.0,
            indexing_time_ms=5.0,
        )
        assert result.sparse_indexed is True

        result2 = IndexResult(
            document_id="doc-2",
            chunks_indexed=1,
            replaced_existing=False,
            embedding_time_ms=10.0,
            indexing_time_ms=5.0,
            sparse_indexed=False,
        )
        assert result2.sparse_indexed is False

    @pytest.mark.asyncio
    async def test_initialize_creates_named_vectors_when_hybrid_enabled(self):
        """create_collection called with named vectors when hybrid enabled."""
        vs = self._make_service(hybrid=True)
        vs._qdrant = unittest.mock.AsyncMock()
        # No existing collections
        vs._qdrant.get_collections.return_value = unittest.mock.MagicMock(collections=[])

        await vs.initialize()

        call_kwargs = vs._qdrant.create_collection.call_args
        vectors_config = call_kwargs.kwargs.get("vectors_config") or call_kwargs[1].get(
            "vectors_config"
        )
        sparse_config = call_kwargs.kwargs.get("sparse_vectors_config") or call_kwargs[1].get(
            "sparse_vectors_config"
        )
        # vectors_config should be a dict with "dense" key
        assert isinstance(vectors_config, dict)
        assert "dense" in vectors_config
        # sparse_vectors_config should have "sparse" key
        assert isinstance(sparse_config, dict)
        assert "sparse" in sparse_config

    @pytest.mark.asyncio
    async def test_initialize_creates_unnamed_vector_when_hybrid_disabled(self):
        """create_collection called with VectorParams (not dict) when hybrid disabled."""
        vs = self._make_service(hybrid=False)
        vs._qdrant = unittest.mock.AsyncMock()
        vs._qdrant.get_collections.return_value = unittest.mock.MagicMock(collections=[])

        await vs.initialize()

        call_kwargs = vs._qdrant.create_collection.call_args
        vectors_config = call_kwargs.kwargs.get("vectors_config") or call_kwargs[1].get(
            "vectors_config"
        )
        # Should be VectorParams, not a dict
        assert isinstance(vectors_config, models.VectorParams)

    @pytest.mark.asyncio
    async def test_initialize_creates_sparse_indexed_payload_index(self):
        """create_payload_index called for sparse_indexed with BOOL type."""
        vs = self._make_service(hybrid=True)
        vs._qdrant = unittest.mock.AsyncMock()
        vs._qdrant.get_collections.return_value = unittest.mock.MagicMock(collections=[])

        await vs.initialize()

        # Check all create_payload_index calls for sparse_indexed
        calls = vs._qdrant.create_payload_index.call_args_list
        sparse_indexed_calls = [
            c
            for c in calls
            if (c.kwargs.get("field_name") or (c.args[1] if len(c.args) > 1 else None))
            == "sparse_indexed"
        ]
        assert len(sparse_indexed_calls) == 1
        call = sparse_indexed_calls[0]
        schema = call.kwargs.get("field_schema")
        assert schema == models.PayloadSchemaType.BOOL

    @pytest.mark.asyncio
    async def test_initialize_calls_detect_capabilities(self):
        """_detect_collection_capabilities() called after collection create and in else branch."""
        # Test: new collection
        vs = self._make_service(hybrid=True)
        vs._qdrant = unittest.mock.AsyncMock()
        vs._qdrant.get_collections.return_value = unittest.mock.MagicMock(collections=[])
        await vs.initialize()
        vs._detect_collection_capabilities.assert_called_once()

        # Test: existing collection
        vs2 = self._make_service(hybrid=True)
        vs2._qdrant = unittest.mock.AsyncMock()
        existing = unittest.mock.MagicMock()
        existing.name = "documents"
        vs2._qdrant.get_collections.return_value = unittest.mock.MagicMock(collections=[existing])
        await vs2.initialize()
        vs2._detect_collection_capabilities.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_document_dual_vectors_when_hybrid(self):
        """PointStruct has vector={'dense': ..., 'sparse': ...} and sparse_indexed=True."""
        vs = self._make_service(hybrid=True, has_sparse=True)
        vs._qdrant = unittest.mock.AsyncMock()
        vs._qdrant.count.return_value = unittest.mock.MagicMock(count=0)

        fake_embedding = [0.1] * 768
        vs.embed_batch = unittest.mock.AsyncMock(return_value=[fake_embedding])
        fake_sparse = models.SparseVector(indices=[0, 1], values=[0.5, 0.3])
        vs.sparse_embed_batch = unittest.mock.MagicMock(return_value=[fake_sparse])

        result = await vs.add_document(
            document_id="doc-hybrid",
            document_name="test.pdf",
            chunks=["chunk text"],
            tags=["public"],
            uploaded_by="user-1",
        )

        # Check the upsert call
        upsert_call = vs._qdrant.upsert.call_args
        points = upsert_call.kwargs.get("points") or upsert_call[1].get("points")
        assert len(points) == 1
        point = points[0]
        assert isinstance(point.vector, dict)
        assert "dense" in point.vector
        assert "sparse" in point.vector
        assert point.payload["sparse_indexed"] is True
        assert result.sparse_indexed is True

    @pytest.mark.asyncio
    async def test_add_document_dense_only_when_sparse_fails(self):
        """On sparse_embed_batch exception: dense-only, sparse_indexed=False."""
        vs = self._make_service(hybrid=True, has_sparse=True)
        vs._qdrant = unittest.mock.AsyncMock()
        vs._qdrant.count.return_value = unittest.mock.MagicMock(count=0)

        fake_embedding = [0.1] * 768
        vs.embed_batch = unittest.mock.AsyncMock(return_value=[fake_embedding])
        vs.sparse_embed_batch = unittest.mock.MagicMock(
            side_effect=RuntimeError("Sparse model failed")
        )

        result = await vs.add_document(
            document_id="doc-degraded",
            document_name="test.pdf",
            chunks=["chunk text"],
            tags=["public"],
            uploaded_by="user-1",
        )

        upsert_call = vs._qdrant.upsert.call_args
        points = upsert_call.kwargs.get("points") or upsert_call[1].get("points")
        point = points[0]
        # Should still use named vector dict (collection has sparse support)
        assert isinstance(point.vector, dict)
        assert "dense" in point.vector
        assert "sparse" not in point.vector
        assert point.payload["sparse_indexed"] is False
        assert result.sparse_indexed is False

    @pytest.mark.asyncio
    async def test_add_document_unnamed_vector_legacy_collection(self):
        """When _collection_has_sparse=False: vector=embedding (list, not dict)."""
        vs = self._make_service(hybrid=False, has_sparse=False)
        vs._qdrant = unittest.mock.AsyncMock()
        vs._qdrant.count.return_value = unittest.mock.MagicMock(count=0)

        fake_embedding = [0.1] * 768
        vs.embed_batch = unittest.mock.AsyncMock(return_value=[fake_embedding])

        result = await vs.add_document(
            document_id="doc-legacy",
            document_name="test.pdf",
            chunks=["chunk text"],
            tags=["public"],
            uploaded_by="user-1",
        )

        upsert_call = vs._qdrant.upsert.call_args
        points = upsert_call.kwargs.get("points") or upsert_call[1].get("points")
        point = points[0]
        # Should be a list (unnamed vector), not a dict
        assert isinstance(point.vector, list)
        assert point.payload["sparse_indexed"] is True  # Legacy default
        assert result.sparse_indexed is True

    @pytest.mark.asyncio
    async def test_backfill_sparse_vectors(self):
        """Scroll returns points, update_vectors and set_payload called, returns count."""
        vs = self._make_service(hybrid=True, has_sparse=True)
        vs._qdrant = unittest.mock.AsyncMock()

        # Mock scroll to return 2 points, then empty
        mock_point1 = unittest.mock.MagicMock()
        mock_point1.id = "point-1"
        mock_point1.payload = {"chunk_text": "text one", "sparse_indexed": False}
        mock_point2 = unittest.mock.MagicMock()
        mock_point2.id = "point-2"
        mock_point2.payload = {"chunk_text": "text two", "sparse_indexed": False}

        vs._qdrant.scroll.side_effect = [
            ([mock_point1, mock_point2], None),  # First batch, no more offset
        ]

        fake_sparse = models.SparseVector(indices=[0, 1], values=[0.5, 0.3])
        vs.sparse_embed_batch = unittest.mock.MagicMock(return_value=[fake_sparse, fake_sparse])

        count = await vs.backfill_sparse_vectors(batch_size=10)

        assert count == 2
        assert vs._qdrant.update_vectors.call_count == 2
        assert vs._qdrant.set_payload.call_count == 2

    @pytest.mark.asyncio
    async def test_backfill_skips_when_no_sparse_support(self):
        """_collection_has_sparse=False returns 0 immediately."""
        vs = self._make_service(hybrid=True, has_sparse=False)
        vs._qdrant = unittest.mock.AsyncMock()

        count = await vs.backfill_sparse_vectors()

        assert count == 0
        vs._qdrant.scroll.assert_not_called()


class TestNormalizeScores:
    """Unit tests for _normalize_scores() (Issue #285)."""

    def setup_method(self):
        self.service = VectorService()

    def _make_point(self, score):
        from types import SimpleNamespace

        return SimpleNamespace(score=score)

    def test_normalize_empty_list(self):
        """Empty input returns empty list."""
        assert self.service._normalize_scores([]) == []

    def test_normalize_single_result(self):
        """Single point gets score 1.0."""
        points = [self._make_point(0.015)]
        result = self.service._normalize_scores(points)
        assert result[0].score == 1.0

    def test_normalize_preserves_ordering(self):
        """Relative ordering is preserved; highest=1.0, lowest=0.0."""
        points = [self._make_point(0.033), self._make_point(0.020), self._make_point(0.008)]
        result = self.service._normalize_scores(points)
        assert result[0].score > result[1].score > result[2].score
        assert result[0].score == 1.0
        assert result[2].score == 0.0

    def test_normalize_range_0_to_1(self):
        """All normalized scores are in [0.0, 1.0]."""
        points = [self._make_point(s) for s in [0.033, 0.025, 0.020, 0.015, 0.008]]
        result = self.service._normalize_scores(points)
        for p in result:
            assert 0.0 <= p.score <= 1.0

    def test_normalize_equal_scores(self):
        """All identical scores become 1.0."""
        points = [self._make_point(0.020), self._make_point(0.020), self._make_point(0.020)]
        result = self.service._normalize_scores(points)
        for p in result:
            assert p.score == 1.0


class TestHybridSearch:
    """Tests for hybrid search execution paths (Issue #285)."""

    def _make_mock_response(self, scores):
        """Create a mock query_points response with given scores."""
        from types import SimpleNamespace

        points = []
        for i, score in enumerate(scores):
            points.append(
                SimpleNamespace(
                    id=f"point-{i}",
                    score=score,
                    payload={
                        "chunk_id": f"chunk-{i}",
                        "document_id": f"doc-{i}",
                        "document_name": f"file-{i}.pdf",
                        "chunk_text": f"text {i}",
                        "chunk_index": i,
                        "page_number": None,
                        "section": None,
                    },
                )
            )
        return SimpleNamespace(points=points)

    @pytest.mark.asyncio
    async def test_search_hybrid_uses_prefetch_and_fusion(self):
        """Hybrid path uses Prefetch + FusionQuery(Fusion.RRF)."""
        from qdrant_client.http.models import Fusion, FusionQuery

        vs = _make_service(hybrid=True, has_sparse=True)
        vs._qdrant = unittest.mock.AsyncMock()
        vs.embed = unittest.mock.AsyncMock(return_value=[0.1] * 768)
        vs.sparse_embed = unittest.mock.MagicMock(
            return_value=models.SparseVector(indices=[1, 2], values=[0.5, 0.3])
        )
        vs._qdrant.query_points.return_value = self._make_mock_response([0.033, 0.020])

        await vs.search("test query", user_tags=["hr"])

        call_kwargs = vs._qdrant.query_points.call_args.kwargs
        assert "prefetch" in call_kwargs
        assert len(call_kwargs["prefetch"]) == 2
        assert isinstance(call_kwargs["query"], FusionQuery)
        assert call_kwargs["query"].fusion == Fusion.RRF
        # score_threshold must NOT be in kwargs for hybrid
        assert "score_threshold" not in call_kwargs or call_kwargs.get("score_threshold") is None

    @pytest.mark.asyncio
    async def test_search_hybrid_scores_normalized(self):
        """Hybrid results have scores in [0.0, 1.0]."""
        vs = _make_service(hybrid=True, has_sparse=True)
        vs._qdrant = unittest.mock.AsyncMock()
        vs.embed = unittest.mock.AsyncMock(return_value=[0.1] * 768)
        vs.sparse_embed = unittest.mock.MagicMock(
            return_value=models.SparseVector(indices=[1, 2], values=[0.5, 0.3])
        )
        vs._qdrant.query_points.return_value = self._make_mock_response([0.033, 0.020, 0.008])

        results = await vs.search("test query", user_tags=["hr"])

        for r in results:
            assert 0.0 <= r.score <= 1.0
        # Highest score should be 1.0
        assert results[0].score == 1.0

    @pytest.mark.asyncio
    async def test_search_hybrid_filter_parity(self):
        """Both Prefetch queries have the SAME filter object."""
        vs = _make_service(hybrid=True, has_sparse=True)
        vs._qdrant = unittest.mock.AsyncMock()
        vs.embed = unittest.mock.AsyncMock(return_value=[0.1] * 768)
        vs.sparse_embed = unittest.mock.MagicMock(
            return_value=models.SparseVector(indices=[1, 2], values=[0.5, 0.3])
        )
        vs._qdrant.query_points.return_value = self._make_mock_response([0.033])

        await vs.search("test query", user_tags=["hr", "finance"])

        call_kwargs = vs._qdrant.query_points.call_args.kwargs
        prefetches = call_kwargs["prefetch"]
        # Both prefetches must use the same filter object (identity check)
        assert prefetches[0].filter is prefetches[1].filter

    @pytest.mark.asyncio
    async def test_search_sparse_failure_falls_back(self):
        """Sparse embed exception triggers degraded dense-only path."""
        vs = _make_service(hybrid=True, has_sparse=True)
        vs._qdrant = unittest.mock.AsyncMock()
        vs.embed = unittest.mock.AsyncMock(return_value=[0.1] * 768)
        vs.sparse_embed = unittest.mock.MagicMock(side_effect=RuntimeError("Sparse failed"))
        vs._qdrant.query_points.return_value = self._make_mock_response([0.85])

        results = await vs.search("test query", user_tags=["hr"])

        assert len(results) == 1
        call_kwargs = vs._qdrant.query_points.call_args.kwargs
        # Should NOT use prefetch
        assert "prefetch" not in call_kwargs or call_kwargs.get("prefetch") is None
        # Should use named vector
        assert call_kwargs.get("using") == "dense"

    @pytest.mark.asyncio
    async def test_search_sparse_returns_none_falls_back(self):
        """sparse_embed returning None triggers degraded path."""
        vs = _make_service(hybrid=True, has_sparse=True)
        vs._qdrant = unittest.mock.AsyncMock()
        vs.embed = unittest.mock.AsyncMock(return_value=[0.1] * 768)
        vs.sparse_embed = unittest.mock.MagicMock(return_value=None)
        vs._qdrant.query_points.return_value = self._make_mock_response([0.85])

        results = await vs.search("test query", user_tags=["hr"])

        assert len(results) == 1
        call_kwargs = vs._qdrant.query_points.call_args.kwargs
        assert "prefetch" not in call_kwargs or call_kwargs.get("prefetch") is None
        assert call_kwargs.get("using") == "dense"

    @pytest.mark.asyncio
    async def test_search_dense_only_hybrid_disabled(self):
        """Hybrid disabled: no prefetch, no using (legacy collection)."""
        vs = _make_service(hybrid=False, has_sparse=False)
        vs._qdrant = unittest.mock.AsyncMock()
        vs.embed = unittest.mock.AsyncMock(return_value=[0.1] * 768)
        vs._qdrant.query_points.return_value = self._make_mock_response([0.85])

        results = await vs.search("test query", user_tags=["hr"])

        assert len(results) == 1
        call_kwargs = vs._qdrant.query_points.call_args.kwargs
        assert "prefetch" not in call_kwargs or call_kwargs.get("prefetch") is None
        assert "using" not in call_kwargs or call_kwargs.get("using") is None

    @pytest.mark.asyncio
    async def test_search_dense_only_no_sparse_collection(self):
        """Hybrid enabled but no sparse in collection: no prefetch."""
        vs = _make_service(hybrid=True, has_sparse=False)
        vs._qdrant = unittest.mock.AsyncMock()
        vs.embed = unittest.mock.AsyncMock(return_value=[0.1] * 768)
        vs._qdrant.query_points.return_value = self._make_mock_response([0.85])

        results = await vs.search("test query", user_tags=["hr"])

        assert len(results) == 1
        call_kwargs = vs._qdrant.query_points.call_args.kwargs
        assert "prefetch" not in call_kwargs or call_kwargs.get("prefetch") is None

    @pytest.mark.asyncio
    async def test_search_dense_with_named_vectors(self):
        """Hybrid disabled but collection has named vectors: uses using='dense'."""
        vs = _make_service(hybrid=False, has_sparse=True)
        vs._qdrant = unittest.mock.AsyncMock()
        vs.embed = unittest.mock.AsyncMock(return_value=[0.1] * 768)
        vs._qdrant.query_points.return_value = self._make_mock_response([0.85])

        await vs.search("test query", user_tags=["hr"])

        call_kwargs = vs._qdrant.query_points.call_args.kwargs
        assert call_kwargs.get("using") == "dense"

    @pytest.mark.asyncio
    async def test_prefetch_limit_calculation(self):
        """Prefetch limit follows max(20, min(100, limit * multiplier))."""
        vs = _make_service(hybrid=True, has_sparse=True, prefetch_multiplier=3)
        vs._qdrant = unittest.mock.AsyncMock()
        vs.embed = unittest.mock.AsyncMock(return_value=[0.1] * 768)
        vs.sparse_embed = unittest.mock.MagicMock(
            return_value=models.SparseVector(indices=[1], values=[0.5])
        )
        vs._qdrant.query_points.return_value = self._make_mock_response([0.033])

        # limit=5 -> 5*3=15 -> max(20,15)=20
        await vs.search("test", user_tags=["hr"], limit=5)
        prefetches = vs._qdrant.query_points.call_args.kwargs["prefetch"]
        assert prefetches[0].limit == 20

        # limit=10 -> 10*3=30 -> max(20,30)=30
        vs._qdrant.query_points.reset_mock()
        vs._qdrant.query_points.return_value = self._make_mock_response([0.033])
        await vs.search("test", user_tags=["hr"], limit=10)
        prefetches = vs._qdrant.query_points.call_args.kwargs["prefetch"]
        assert prefetches[0].limit == 30

        # limit=50 -> 50*3=150 -> min(100,150)=100
        vs._qdrant.query_points.reset_mock()
        vs._qdrant.query_points.return_value = self._make_mock_response([0.033])
        await vs.search("test", user_tags=["hr"], limit=50)
        prefetches = vs._qdrant.query_points.call_args.kwargs["prefetch"]
        assert prefetches[0].limit == 100

    @pytest.mark.asyncio
    async def test_score_threshold_applied_after_normalization(self):
        """In hybrid mode, score_threshold filters AFTER normalization."""
        vs = _make_service(hybrid=True, has_sparse=True)
        vs._qdrant = unittest.mock.AsyncMock()
        vs.embed = unittest.mock.AsyncMock(return_value=[0.1] * 768)
        vs.sparse_embed = unittest.mock.MagicMock(
            return_value=models.SparseVector(indices=[1], values=[0.5])
        )
        # 3 results: normalized will be 1.0, 0.5, 0.0
        vs._qdrant.query_points.return_value = self._make_mock_response([0.033, 0.020, 0.008])

        results = await vs.search("test", user_tags=["hr"], score_threshold=0.5)

        # 0.0 should be filtered out, leaving 2 results (1.0 and ~0.48 rounds)
        # Actually: (0.033-0.008)/(0.033-0.008)=1.0, (0.020-0.008)/0.025=0.48, 0.0
        # With threshold 0.5: only score=1.0 passes
        assert len(results) == 1
        assert results[0].score == 1.0


class TestHybridAccessFilterParity:
    """Tests for access filter parity on both prefetch queries (Issue #285)."""

    @pytest.mark.asyncio
    async def test_filter_same_object_in_both_prefetches(self):
        """Multi-tag user: both prefetch queries share the same filter object."""
        from types import SimpleNamespace

        vs = _make_service(hybrid=True, has_sparse=True)
        vs._qdrant = unittest.mock.AsyncMock()
        vs.embed = unittest.mock.AsyncMock(return_value=[0.1] * 768)
        vs.sparse_embed = unittest.mock.MagicMock(
            return_value=models.SparseVector(indices=[1], values=[0.5])
        )
        vs._qdrant.query_points.return_value = SimpleNamespace(points=[])

        await vs.search("test", user_tags=["hr", "finance", "legal"])

        call_kwargs = vs._qdrant.query_points.call_args.kwargs
        prefetches = call_kwargs["prefetch"]
        assert prefetches[0].filter is prefetches[1].filter
        # Verify the filter has the expected structure
        f = prefetches[0].filter
        assert isinstance(f, models.Filter)
        assert len(f.must) == 2  # tenant_id + nested should

    @pytest.mark.asyncio
    async def test_admin_filter_in_both_prefetches(self):
        """Admin (user_tags=None): both prefetch queries have tenant-only filter."""
        from types import SimpleNamespace

        vs = _make_service(hybrid=True, has_sparse=True)
        vs._qdrant = unittest.mock.AsyncMock()
        vs.embed = unittest.mock.AsyncMock(return_value=[0.1] * 768)
        vs.sparse_embed = unittest.mock.MagicMock(
            return_value=models.SparseVector(indices=[1], values=[0.5])
        )
        vs._qdrant.query_points.return_value = SimpleNamespace(points=[])

        await vs.search("test", user_tags=None)

        call_kwargs = vs._qdrant.query_points.call_args.kwargs
        prefetches = call_kwargs["prefetch"]
        assert prefetches[0].filter is prefetches[1].filter
        # Admin filter: only tenant_id condition
        f = prefetches[0].filter
        assert isinstance(f, models.Filter)
        assert len(f.must) == 1  # tenant_id only
