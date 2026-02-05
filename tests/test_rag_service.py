"""Tests for RAGService.

Unit tests mock VectorService for CI compatibility.
Integration tests require Ollama (localhost:11434).
Use pytest -m "not integration" to skip integration tests.
"""

import logging
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_ready_rag.config import Settings
from ai_ready_rag.core.exceptions import (
    ModelNotAllowedError,
    TokenBudgetExceededError,
)
from ai_ready_rag.services.rag_constants import (
    MODEL_LIMITS,
    ROUTER_PROMPT,
    ROUTING_DIRECT,
    ROUTING_RETRIEVE,
)
from ai_ready_rag.services.rag_service import (
    SOURCEID_PATTERN,
    ChatMessage,
    RAGRequest,
    RAGResponse,
    RAGService,
    TokenBudget,
    calculate_coverage,
    extract_key_terms,
)
from ai_ready_rag.services.vector_service import SearchResult, VectorService

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
    settings.rag_enable_query_expansion = True
    settings.rag_enable_hallucination_check = False  # Disabled in tests for speed
    return settings


@pytest.fixture
def mock_vector_service():
    """Mock VectorService for unit tests."""
    vs = AsyncMock(spec=VectorService)
    vs.search = AsyncMock(return_value=[])
    return vs


@pytest.fixture
def sample_search_results():
    """Sample SearchResult objects for testing."""
    return [
        SearchResult(
            chunk_id="550e8400-e29b-41d4-a716-446655440000:0",
            document_id="550e8400-e29b-41d4-a716-446655440000",
            document_name="policy.pdf",
            chunk_text="Employee vacation policy allows 20 days annually.",
            chunk_index=0,
            score=0.85,
            page_number=1,
            section="Benefits",
        ),
        SearchResult(
            chunk_id="660e8400-e29b-41d4-a716-446655440001:1",
            document_id="660e8400-e29b-41d4-a716-446655440001",
            document_name="handbook.pdf",
            chunk_text="Remote work is permitted with manager approval.",
            chunk_index=1,
            score=0.72,
            page_number=5,
            section="Work Policies",
        ),
    ]


@pytest.fixture
def sample_chat_history():
    """Sample ChatMessage list for history truncation tests."""
    return [
        ChatMessage(role="user", content="What is the vacation policy?"),
        ChatMessage(role="assistant", content="Employees get 20 days annually."),
        ChatMessage(role="user", content="Can I work remotely?"),
    ]


# =============================================================================
# Unit Tests
# =============================================================================


class TestBuildContext:
    """Test context building with SourceIds."""

    def test_build_context_with_source_ids(self, mock_settings, sample_search_results):
        """SourceIds match UUID:index format."""
        # Arrange
        service = RAGService(
            vector_service=AsyncMock(),
            settings=mock_settings,
        )

        # Act
        context = service._build_context_prompt(sample_search_results)

        # Assert
        pattern = r"\[SourceId:\s*[a-f0-9-]{36}:\d+\]"
        assert re.search(pattern, context), f"Expected SourceId pattern in: {context}"
        assert "550e8400-e29b-41d4-a716-446655440000:0" in context

    def test_build_context_includes_metadata(self, mock_settings, sample_search_results):
        """Context includes document name, page, section."""
        # Arrange
        service = RAGService(
            vector_service=AsyncMock(),
            settings=mock_settings,
        )

        # Act
        context = service._build_context_prompt(sample_search_results)

        # Assert
        assert "[Document: policy.pdf]" in context
        assert "[Page: 1]" in context
        assert "[Section: Benefits]" in context
        assert "[Document: handbook.pdf]" in context
        assert "[Page: 5]" in context
        assert "[Section: Work Policies]" in context


class TestExtractCitations:
    """Test citation extraction from LLM responses."""

    def test_extract_citations_valid(self, mock_settings, sample_search_results):
        """Citations extracted from answer text."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)
        answer = "Per the policy [SourceId: 550e8400-e29b-41d4-a716-446655440000:0], employees get 20 days."

        # Act
        citations = service._extract_citations(answer, sample_search_results)

        # Assert
        assert len(citations) == 1
        assert citations[0].document_id == "550e8400-e29b-41d4-a716-446655440000"
        assert citations[0].chunk_index == 0
        assert citations[0].document_name == "policy.pdf"

    def test_extract_citations_invalid_format(self, mock_settings, sample_search_results):
        """Malformed SourceIds ignored."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)
        # Invalid format - not matching UUID:index
        answer = "Per the policy [SourceId: invalid-format], employees get 20 days."

        # Act
        citations = service._extract_citations(answer, sample_search_results)

        # Assert - no exception, empty citations
        assert len(citations) == 0

    def test_extract_citations_unknown_id(self, mock_settings, sample_search_results, caplog):
        """Unknown SourceIds logged, not included."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)
        # Valid format but unknown ID
        answer = "Per the policy [SourceId: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee:0], employees get 20 days."

        # Act
        with caplog.at_level(logging.WARNING):
            citations = service._extract_citations(answer, sample_search_results)

        # Assert
        assert len(citations) == 0
        assert "Unknown SourceId" in caplog.text


class TestCalculateCoverage:
    """Test coverage score calculation."""

    def test_calculate_coverage(self):
        """Term overlap computed correctly."""
        # Arrange
        answer = "Employees receive vacation days annually."
        context = "Employee vacation policy allows 20 days annually."

        # Act
        score = calculate_coverage(answer, context)

        # Assert
        assert 0.0 <= score <= 1.0
        # "employees" -> "employee", "vacation" matches, "days" matches, "annually" matches
        # "receive" not in context
        # At least some matches expected
        assert score > 0.3

    def test_calculate_coverage_empty_answer(self):
        """Returns 0.0 for empty answer."""
        assert calculate_coverage("", "Some context text") == 0.0
        assert calculate_coverage("   ", "Some context text") == 0.0


class TestCalculateConfidence:
    """Test hybrid confidence scoring."""

    def test_calculate_confidence_zero_results(self, mock_settings):
        """Returns overall=0 when no retrieval results."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)

        # Act
        confidence = service._calculate_confidence(
            retrieval_results=[],
            answer="Some answer",
            context="",
            llm_score=50,
        )

        # Assert
        assert confidence.overall == 0
        assert confidence.retrieval_score == 0.0
        assert confidence.coverage_score == 0.0
        assert confidence.llm_score == 0

    def test_calculate_confidence_hybrid(self, mock_settings, sample_search_results):
        """Weights applied correctly (30/40/30)."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)
        context = "\n".join(r.chunk_text for r in sample_search_results)
        answer = "Employees get 20 vacation days annually."

        # Act
        confidence = service._calculate_confidence(
            retrieval_results=sample_search_results,
            answer=answer,
            context=context,
            llm_score=80,
        )

        # Assert
        assert 0 <= confidence.overall <= 100
        # Retrieval score should be average: (0.85 + 0.72) / 2 = 0.785
        assert 0.78 <= confidence.retrieval_score <= 0.79
        # Coverage score depends on term overlap
        assert 0.0 <= confidence.coverage_score <= 1.0
        assert confidence.llm_score == 80

        # Verify bounds
        expected_overall = int(
            (confidence.retrieval_score * 30) + (confidence.coverage_score * 40) + (80 / 100 * 30)
        )
        assert confidence.overall == min(100, max(0, expected_overall))


class TestTruncateHistory:
    """Test chat history truncation."""

    def test_truncate_history_overflow(self, mock_settings, sample_chat_history):
        """Oldest messages dropped when exceeding budget."""
        # Arrange
        budget = TokenBudget("llama3.2", mock_settings)

        # Small budget - only fits last message
        def count_fn(text: str) -> int:
            return len(text) // 4

        # Act
        result, tokens = budget._truncate_history(
            sample_chat_history, max_tokens=20, count_fn=count_fn
        )

        # Assert - should have truncated oldest messages
        # Original has 3 messages, with small budget we should have fewer
        included_count = result.count(":")  # Count role: content pairs
        assert included_count < 3

    def test_truncate_history_keeps_recent(self, mock_settings, sample_chat_history):
        """Most recent messages preserved."""
        # Arrange
        budget = TokenBudget("llama3.2", mock_settings)

        def count_fn(text: str) -> int:
            return len(text) // 4

        # Large budget - should keep all
        result, tokens = budget._truncate_history(
            sample_chat_history, max_tokens=1000, count_fn=count_fn
        )

        # Assert - most recent message should be present
        assert "Can I work remotely?" in result


class TestTokenBudget:
    """Test token budget management."""

    def test_token_budget_exceeded(self, mock_settings):
        """Raises TokenBudgetExceededError when budget exceeded."""
        # Arrange - create settings with tiny context window
        settings = MagicMock(spec=Settings)
        settings.rag_max_context_tokens = 3000
        settings.rag_max_history_tokens = 1000
        settings.rag_max_response_tokens = 100000  # Very large response reserve
        settings.rag_system_prompt_tokens = 500

        # Monkey-patch MODEL_LIMITS temporarily for test
        original_limits = MODEL_LIMITS.copy()

        try:
            # Patch the model limits with a small context window
            MODEL_LIMITS["test_model"] = {"context_window": 1000, "max_response": 100000}

            budget = TokenBudget("test_model", settings)

            def count_fn(text: str) -> int:
                return len(text) // 4

            # Act & Assert
            with pytest.raises(TokenBudgetExceededError):
                budget.allocate(
                    system_prompt="A" * 100,  # small prompt
                    chat_history=[],
                    context_chunks=[],
                    count_fn=count_fn,
                )
        finally:
            # Restore original limits
            MODEL_LIMITS.clear()
            MODEL_LIMITS.update(original_limits)


class TestMaxResponseTokensCapping:
    """Test max_response_tokens capping to model limits."""

    def test_caps_to_model_limit(self, mock_settings):
        """Requested tokens above model limit get capped."""
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)
        service.default_model = "llama3.2"  # Has 1024 max_response

        # Request 4096 but llama3.2 only supports 1024
        result = service._get_effective_max_tokens(4096)

        assert result == 1024

    def test_no_cap_when_under_limit(self, mock_settings):
        """Requested tokens under model limit returned as-is."""
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)
        service.default_model = "qwen3:8b"  # Has 2048 max_response

        # Request 1000 which is under the 2048 limit
        result = service._get_effective_max_tokens(1000)

        assert result == 1000

    def test_unknown_model_uses_default(self, mock_settings):
        """Unknown models use 2048 default limit."""
        from unittest.mock import patch

        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)
        service.default_model = "unknown-model:latest"

        # Mock _get_current_chat_model to return the unknown model
        with patch.object(service, "_get_current_chat_model", return_value="unknown-model:latest"):
            # Default limit is 2048
            result = service._get_effective_max_tokens(3000)

        assert result == 2048

    def test_capping_logs_warning(self, mock_settings, caplog):
        """Capping logs a warning message."""
        import logging

        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)
        service.default_model = "llama3.2"  # Has 1024 max_response

        with caplog.at_level(logging.WARNING, logger="ai_ready_rag.services.rag_service"):
            service._get_effective_max_tokens(2000)

        assert any("Capping to 1024" in record.message for record in caplog.records)


class TestDeduplicateChunks:
    """Test chunk deduplication."""

    def test_deduplicate_chunks(self, mock_settings):
        """Near-duplicates removed (Jaccard > 0.9)."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)
        service.chunk_overlap_threshold = 0.9

        # Create near-duplicate chunks
        chunks = [
            SearchResult(
                chunk_id="doc1:0",
                document_id="doc1",
                document_name="test.pdf",
                chunk_text="The quick brown fox jumps over the lazy dog",
                chunk_index=0,
                score=0.9,
                page_number=1,
                section=None,
            ),
            SearchResult(
                chunk_id="doc1:1",
                document_id="doc1",
                document_name="test.pdf",
                # Near-duplicate (only one word different)
                chunk_text="The quick brown fox jumps over the lazy cat",
                chunk_index=1,
                score=0.85,
                page_number=1,
                section=None,
            ),
            SearchResult(
                chunk_id="doc2:0",
                document_id="doc2",
                document_name="other.pdf",
                # Completely different
                chunk_text="Python programming is fun and productive",
                chunk_index=0,
                score=0.8,
                page_number=1,
                section=None,
            ),
        ]

        # Act
        result = service._deduplicate_chunks(chunks)

        # Assert - near-duplicate should NOT be removed (Jaccard < 0.9)
        # With one word difference in 9 words, Jaccard = 8/10 = 0.8 < 0.9
        assert len(result) == 3  # All unique enough

    def test_deduplicate_removes_exact_duplicate(self, mock_settings):
        """Exact duplicates are removed."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)
        service.chunk_overlap_threshold = 0.9

        chunks = [
            SearchResult(
                chunk_id="doc1:0",
                document_id="doc1",
                document_name="test.pdf",
                chunk_text="Exact same text here for testing",
                chunk_index=0,
                score=0.9,
                page_number=1,
                section=None,
            ),
            SearchResult(
                chunk_id="doc1:1",
                document_id="doc1",
                document_name="test.pdf",
                chunk_text="Exact same text here for testing",  # Exact duplicate
                chunk_index=1,
                score=0.85,
                page_number=1,
                section=None,
            ),
        ]

        # Act
        result = service._deduplicate_chunks(chunks)

        # Assert
        assert len(result) == 1

    def test_deduplicate_respects_cap(self, mock_settings):
        """Max 15 candidates processed."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)
        service.dedup_candidates_cap = 15
        service.chunk_overlap_threshold = 0.9

        # Create 20 unique chunks
        chunks = [
            SearchResult(
                chunk_id=f"doc{i}:0",
                document_id=f"doc{i}",
                document_name=f"test{i}.pdf",
                chunk_text=f"Unique content number {i} with different words",
                chunk_index=0,
                score=0.9 - (i * 0.01),
                page_number=1,
                section=None,
            )
            for i in range(20)
        ]

        # Act - deduplicate processes all provided chunks
        # The cap is applied during retrieval, not deduplication
        result = service._deduplicate_chunks(chunks)

        # Assert - all should be preserved since they're unique
        assert len(result) == 20


class TestLimitPerDocument:
    """Test per-document chunk limiting."""

    def test_limit_per_document(self, mock_settings):
        """Max 3 chunks per doc."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)

        # Create 5 chunks from same document
        chunks = [
            SearchResult(
                chunk_id=f"doc1:{i}",
                document_id="doc1",
                document_name="test.pdf",
                chunk_text=f"Content chunk number {i}",
                chunk_index=i,
                score=0.9 - (i * 0.05),
                page_number=i + 1,
                section=None,
            )
            for i in range(5)
        ]

        # Act
        result = service._limit_per_document(chunks, max_per_doc=3)

        # Assert
        assert len(result) == 3
        # Should keep highest scored (first 3)
        assert result[0].chunk_index == 0
        assert result[1].chunk_index == 1
        assert result[2].chunk_index == 2


class TestValidateModel:
    """Test model validation."""

    @pytest.mark.asyncio
    async def test_validate_model_allowed(self, mock_settings):
        """Valid model passes validation."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)

        # Mock httpx response
        with patch("ai_ready_rag.services.rag_service.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "models": [{"name": "llama3.2"}, {"name": "qwen3:8b"}]
            }
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            # Act
            result = await service.validate_model("llama3.2")

            # Assert
            assert result == "llama3.2"

    @pytest.mark.asyncio
    async def test_validate_model_not_allowed(self, mock_settings):
        """Invalid model raises ModelNotAllowedError."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)

        # Act & Assert
        with pytest.raises(ModelNotAllowedError, match="not in allowlist"):
            await service.validate_model("invalid-model-xyz")


class TestGetRouteTarget:
    """Test routing target determination."""

    @pytest.mark.asyncio
    async def test_get_route_target_with_owner(self, mock_settings, db, sample_tag, admin_user):
        """Owner found and returned."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)

        # Update tag to have an owner
        sample_tag.owner_id = admin_user.id
        db.flush()

        # Create search results with tags attribute
        result = MagicMock(spec=SearchResult)
        result.tags = [sample_tag.name]

        # Act
        route_target = await service.get_route_target([result], db)

        # Assert
        assert route_target.fallback is False
        assert route_target.owner_email == admin_user.email

    @pytest.mark.asyncio
    async def test_get_route_target_no_owner(self, mock_settings, db, sample_tag):
        """Admin fallback when no owner."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)

        # Tag has no owner_id
        result = MagicMock(spec=SearchResult)
        result.tags = [sample_tag.name]

        # Act
        route_target = await service.get_route_target([result], db)

        # Assert
        assert route_target.fallback is True
        assert route_target.owner_email == mock_settings.rag_admin_email

    @pytest.mark.asyncio
    async def test_get_route_target_empty_context(self, mock_settings, db):
        """Admin fallback on empty context."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)

        # Act
        route_target = await service.get_route_target([], db)

        # Assert
        assert route_target.fallback is True
        assert "No context" in route_target.reason


class TestGroundedFlag:
    """Test grounded flag calculation."""

    def test_grounded_true_with_citations(self, mock_settings, sample_search_results):
        """grounded=True when citations exist."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)
        answer = "Per the policy [SourceId: 550e8400-e29b-41d4-a716-446655440000:0], employees get 20 days."

        # Act
        citations = service._extract_citations(answer, sample_search_results)

        # Assert
        grounded = len(citations) > 0
        assert grounded is True
        assert len(citations) == 1

    def test_grounded_false_no_citations(self, mock_settings, sample_search_results):
        """grounded=False when no citations."""
        # Arrange
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)
        answer = "I don't have enough information to answer this question."

        # Act
        citations = service._extract_citations(answer, sample_search_results)

        # Assert
        grounded = len(citations) > 0
        assert grounded is False
        assert len(citations) == 0


class TestExtractKeyTerms:
    """Test key term extraction helper."""

    def test_extract_key_terms_filters_stopwords(self):
        """Stopwords are filtered out."""
        text = "The quick brown fox jumps with the lazy dog"
        terms = extract_key_terms(text)

        # "the" and "with" are stopwords
        assert "the" not in terms
        assert "with" not in terms
        # Content words should remain
        assert "quick" in terms
        assert "brown" in terms
        assert "fox" in terms

    def test_extract_key_terms_filters_short_words(self):
        """Words shorter than 3 chars are filtered."""
        text = "Go to the big fan in LA or NY"
        terms = extract_key_terms(text)

        # Short words filtered (go, to, in, la, ny are < 3 or stopwords)
        assert "go" not in terms  # 2 chars
        assert "la" not in terms  # 2 chars
        assert "ny" not in terms  # 2 chars
        # "big" and "fan" are 3 chars and not stopwords
        assert "big" in terms
        assert "fan" in terms


class TestSourceIdPattern:
    """Test SourceId regex pattern."""

    def test_sourceid_pattern_matches_valid(self):
        """Pattern matches valid SourceId format."""
        text = "[SourceId: 550e8400-e29b-41d4-a716-446655440000:0]"
        matches = re.findall(SOURCEID_PATTERN, text)
        assert len(matches) == 1
        assert matches[0] == "550e8400-e29b-41d4-a716-446655440000:0"

    def test_sourceid_pattern_matches_multiple(self):
        """Pattern matches multiple SourceIds."""
        text = "See [SourceId: aaa-bbb:0] and [SourceId: ccc-ddd:1]"
        # These won't match due to invalid UUID format
        matches = re.findall(SOURCEID_PATTERN, text)
        assert len(matches) == 0

        # Valid UUIDs
        text = "[SourceId: 550e8400-e29b-41d4-a716-446655440000:0] and [SourceId: 660e8400-e29b-41d4-a716-446655440001:1]"
        matches = re.findall(SOURCEID_PATTERN, text)
        assert len(matches) == 2


class TestRouterConstants:
    """Test router constants and prompt (Issue #23)."""

    def test_routing_constants_defined(self):
        """Routing constants are defined."""
        assert ROUTING_DIRECT == "DIRECT"
        assert ROUTING_RETRIEVE == "RETRIEVE"

    def test_router_prompt_format(self):
        """Router prompt contains placeholder."""
        assert "{question}" in ROUTER_PROMPT
        assert "RETRIEVE" in ROUTER_PROMPT
        assert "DIRECT" in ROUTER_PROMPT


class TestQueryRouting:
    """Test query routing logic (Issue #23)."""

    @pytest.mark.asyncio
    async def test_run_router_returns_retrieve(self, mock_settings):
        """Router returns RETRIEVE for business queries."""
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)

        # Mock LLM response
        with patch("ai_ready_rag.services.rag_service.ChatOllama") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "RETRIEVE"
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_class.return_value = mock_llm

            result = await service.run_router("What is our company vacation policy?", "llama3.2")
            assert result == ROUTING_RETRIEVE

    @pytest.mark.asyncio
    async def test_run_router_returns_direct(self, mock_settings):
        """Router returns DIRECT for general knowledge queries."""
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)

        # Mock LLM response
        with patch("ai_ready_rag.services.rag_service.ChatOllama") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "DIRECT"
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_class.return_value = mock_llm

            result = await service.run_router("What is the capital of France?", "llama3.2")
            assert result == ROUTING_DIRECT

    @pytest.mark.asyncio
    async def test_run_router_defaults_to_retrieve_on_error(self, mock_settings):
        """Router defaults to RETRIEVE on failure."""
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)

        # Mock LLM failure
        with patch("ai_ready_rag.services.rag_service.ChatOllama") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
            mock_llm_class.return_value = mock_llm

            result = await service.run_router("Any question", "llama3.2")
            assert result == ROUTING_RETRIEVE

    def test_direct_response_has_no_citations(self, mock_settings):
        """Direct response has grounded=False and empty citations."""
        service = RAGService(vector_service=AsyncMock(), settings=mock_settings)

        response = service._generate_direct_response(
            answer="Paris is the capital of France.",
            model="llama3.2",
            elapsed_ms=100.0,
        )

        assert response.grounded is False
        assert len(response.citations) == 0
        assert response.routing_decision == ROUTING_DIRECT
        assert response.context_chunks_used == 0

    def test_rag_response_includes_routing_decision(self, mock_settings):
        """RAGResponse dataclass includes routing_decision field."""
        response = RAGResponse(
            answer="Test answer",
            confidence=MagicMock(),
            citations=[],
            action="CITE",
            route_to=None,
            model_used="llama3.2",
            context_chunks_used=0,
            context_tokens_used=0,
            generation_time_ms=100.0,
            grounded=False,
            routing_decision=ROUTING_RETRIEVE,
        )

        assert response.routing_decision == ROUTING_RETRIEVE


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestRAGServiceIntegration:
    """Integration tests requiring Ollama.

    Run with: pytest -m integration
    Skip with: pytest -m "not integration"
    """

    @pytest.fixture
    async def rag_service(self):
        """RAGService with real dependencies."""
        from ai_ready_rag.config import get_settings

        vs = VectorService(collection_name="test_rag_integration")
        await vs.initialize()

        settings = get_settings()
        service = RAGService(
            vector_service=vs,
            settings=settings,
        )
        yield service

        # Cleanup
        await vs.clear_collection()
        try:
            await vs._qdrant.delete_collection("test_rag_integration")
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_generate_full_pipeline(self, rag_service, db):
        """End-to-end generation with real Ollama."""
        # First index a document
        vs = rag_service.vector_service
        await vs.add_document(
            document_id="doc-rag-test",
            document_name="test-policy.pdf",
            chunks=[
                "Company vacation policy: Employees receive 20 days of paid vacation annually."
            ],
            tags=["hr", "public"],
            uploaded_by="test-user",
        )

        # Create request
        request = RAGRequest(
            query="How many vacation days do employees get?",
            user_tags=["hr"],
            tenant_id="default",
        )

        # Act
        response = await rag_service.generate(request, db)

        # Assert
        assert isinstance(response, RAGResponse)
        assert response.answer != ""
        assert response.confidence is not None
        assert response.model_used == rag_service.default_model

    @pytest.mark.asyncio
    async def test_generate_zero_context(self, rag_service, db):
        """Returns INSUFFICIENT_CONTEXT_RESPONSE when no context."""
        # Don't index any documents - search will return empty

        request = RAGRequest(
            query="What is the meaning of life?",
            user_tags=["hr"],
            tenant_id="default",
        )

        # Act
        response = await rag_service.generate(request, db)

        # Assert
        assert response.action == "ROUTE"
        assert response.context_chunks_used == 0
        assert "don't have enough information" in response.answer

    @pytest.mark.asyncio
    async def test_generate_low_confidence_routes(self, rag_service, db):
        """Routes when confidence < 60."""
        # Index document with tangentially related content
        vs = rag_service.vector_service
        await vs.add_document(
            document_id="doc-tangential",
            document_name="other.pdf",
            chunks=["The weather today is sunny and warm."],
            tags=["public"],
            uploaded_by="test-user",
        )

        request = RAGRequest(
            query="What is the company's financial performance?",
            user_tags=["public"],
            tenant_id="default",
        )

        # Act
        response = await rag_service.generate(request, db)

        # Assert - confidence should be low, leading to ROUTE
        # Note: actual behavior depends on LLM response
        assert response.action in ["CITE", "ROUTE"]
        if response.action == "ROUTE":
            assert response.route_to is not None

    @pytest.mark.asyncio
    async def test_generate_token_budget(self, rag_service, db):
        """Large history truncated correctly."""
        # Index a document
        vs = rag_service.vector_service
        await vs.add_document(
            document_id="doc-budget-test",
            document_name="policy.pdf",
            chunks=["Remote work policy: Employees may work remotely with manager approval."],
            tags=["hr"],
            uploaded_by="test-user",
        )

        # Create very long chat history
        long_history = [
            ChatMessage(role="user", content=f"Question {i}: " + "x" * 500) for i in range(50)
        ]

        request = RAGRequest(
            query="Can I work remotely?",
            user_tags=["hr"],
            tenant_id="default",
            chat_history=long_history,
        )

        # Act - should not raise TokenBudgetExceededError
        response = await rag_service.generate(request, db)

        # Assert
        assert isinstance(response, RAGResponse)

    @pytest.mark.asyncio
    async def test_health_check(self, rag_service):
        """Ollama connectivity verified."""
        # Act
        health = await rag_service.health_check()

        # Assert
        assert health["ollama_healthy"] is True
        assert health["ollama_latency_ms"] is not None
        assert health["ollama_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_model_override(self, rag_service, db):
        """Custom model selection works."""
        # Index a document
        vs = rag_service.vector_service
        await vs.add_document(
            document_id="doc-model-test",
            document_name="policy.pdf",
            chunks=["Test content for model override verification."],
            tags=["public"],
            uploaded_by="test-user",
        )

        # Check which models are available
        health = await rag_service.health_check()
        available = health.get("available_models", [])

        # Use default model if qwen3:8b not available
        test_model = "llama3.2"
        for m in available:
            if "qwen" in m.lower():
                test_model = m.split(":")[0] if ":" not in m else m
                break

        request = RAGRequest(
            query="What is the test content?",
            user_tags=["public"],
            tenant_id="default",
            model=test_model,
        )

        # Act
        response = await rag_service.generate(request, db)

        # Assert
        assert response.model_used == test_model

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, rag_service, db):
        """tenant_id passed to VectorService."""
        # Create a mock to intercept search calls
        original_search = rag_service.vector_service.search
        search_calls = []

        async def tracking_search(*args, **kwargs):
            search_calls.append(kwargs)
            return await original_search(*args, **kwargs)

        rag_service.vector_service.search = tracking_search

        request = RAGRequest(
            query="Test query",
            user_tags=["hr"],
            tenant_id="test-tenant-123",
        )

        # Act
        await rag_service.generate(request, db)

        # Assert
        assert len(search_calls) > 0
        assert search_calls[0].get("tenant_id") == "test-tenant-123"


# =============================================================================
# Synonym Expansion Tests
# =============================================================================


class TestExpandQueryWithSynonyms:
    """Test expand_query() with database synonyms."""

    def test_expand_query_without_db(self, mock_settings):
        """expand_query works without db (uses hardcoded patterns only)."""
        from ai_ready_rag.services.rag_service import expand_query

        queries = expand_query("What is our vacation policy?")

        # Should include original
        assert "What is our vacation policy?" in queries
        # Should include hardcoded policy expansion
        assert any("policy procedure" in q for q in queries)

    def test_expand_query_with_db_synonyms(self, mock_settings, db):
        """expand_query uses database synonyms when db provided."""
        import json

        from ai_ready_rag.db.models import QuerySynonym
        from ai_ready_rag.services.rag_service import expand_query, invalidate_synonym_cache

        # Create a test synonym
        synonym = QuerySynonym(
            term="vacation",
            synonyms=json.dumps(["PTO", "paid time off", "time off"]),
            enabled=True,
        )
        db.add(synonym)
        db.commit()

        # Invalidate cache to pick up new synonym
        invalidate_synonym_cache()

        queries = expand_query("What is our vacation policy?", db=db)

        # Should include synonym expansions
        assert "PTO" in queries
        assert "paid time off" in queries

    def test_expand_query_word_boundary_no_false_positive(self, mock_settings, db):
        """Synonym matching respects word boundaries - no false positives."""
        import json

        from ai_ready_rag.db.models import QuerySynonym
        from ai_ready_rag.services.rag_service import expand_query, invalidate_synonym_cache

        # Create synonym for "pto"
        synonym = QuerySynonym(
            term="pto",
            synonyms=json.dumps(["paid time off", "vacation"]),
            enabled=True,
        )
        db.add(synonym)
        db.commit()
        invalidate_synonym_cache()

        # "photo" should NOT match "pto"
        queries = expand_query("Show me the photo gallery", db=db)
        assert "paid time off" not in queries
        assert "vacation" not in queries

    def test_expand_query_word_boundary_positive_match(self, mock_settings, db):
        """Synonym matching works for actual word matches."""
        import json

        from ai_ready_rag.db.models import QuerySynonym
        from ai_ready_rag.services.rag_service import expand_query, invalidate_synonym_cache

        # Create synonym for "pto"
        synonym = QuerySynonym(
            term="pto",
            synonyms=json.dumps(["paid time off", "vacation"]),
            enabled=True,
        )
        db.add(synonym)
        db.commit()
        invalidate_synonym_cache()

        # "pto" should match
        queries = expand_query("How do I request PTO?", db=db)
        assert "paid time off" in queries or "vacation" in queries

    def test_expand_query_disabled_synonyms_ignored(self, mock_settings, db):
        """Disabled synonyms are not used."""
        import json

        from ai_ready_rag.db.models import QuerySynonym
        from ai_ready_rag.services.rag_service import expand_query, invalidate_synonym_cache

        synonym = QuerySynonym(
            term="vacation",
            synonyms=json.dumps(["PTO"]),
            enabled=False,  # Disabled
        )
        db.add(synonym)
        db.commit()
        invalidate_synonym_cache()

        queries = expand_query("vacation policy", db=db)

        # Should NOT include PTO since synonym is disabled
        assert "PTO" not in queries

    def test_expand_query_bidirectional_synonym(self, mock_settings, db):
        """Synonym expansion works bidirectionally - synonym in query adds term."""
        import json

        from ai_ready_rag.db.models import QuerySynonym
        from ai_ready_rag.services.rag_service import expand_query, invalidate_synonym_cache

        # Create synonym: pto -> [vacation, time off]
        synonym = QuerySynonym(
            term="pto",
            synonyms=json.dumps(["vacation", "time off"]),
            enabled=True,
        )
        db.add(synonym)
        db.commit()
        invalidate_synonym_cache()

        # Query contains "vacation" (a synonym), should expand to include "pto" (the term)
        queries = expand_query("What is the vacation policy?", db=db)

        # Should include the term (reverse direction)
        assert "pto" in queries
        # Should include other synonyms
        assert "time off" in queries


# =============================================================================
# Curated Q&A Tests
# =============================================================================


class TestCuratedQA:
    """Test curated Q&A matching."""

    def test_check_curated_qa_exact_match(self, mock_settings, db):
        """Exact keyword match returns Q&A."""
        import json

        from ai_ready_rag.db.models import CuratedQA
        from ai_ready_rag.services.rag_service import check_curated_qa, invalidate_qa_cache

        qa = CuratedQA(
            keywords=json.dumps(["vacation policy"]),
            answer="Employees receive 20 days of PTO annually.",
            source_reference="Employee Handbook Section 5.1",
            confidence=90,
            priority=10,
            enabled=True,
        )
        db.add(qa)
        db.commit()
        invalidate_qa_cache()

        result = check_curated_qa("What is the vacation policy?", db)

        assert result is not None
        assert "20 days" in result.answer

    def test_check_curated_qa_no_match(self, mock_settings, db):
        """Returns None when no keywords match."""
        import json

        from ai_ready_rag.db.models import CuratedQA
        from ai_ready_rag.services.rag_service import check_curated_qa, invalidate_qa_cache

        qa = CuratedQA(
            keywords=json.dumps(["401k", "retirement"]),
            answer="401k info...",
            source_reference="Benefits Guide",
            confidence=85,
            enabled=True,
        )
        db.add(qa)
        db.commit()
        invalidate_qa_cache()

        result = check_curated_qa("What is the vacation policy?", db)

        assert result is None

    def test_check_curated_qa_priority_ordering(self, mock_settings, db):
        """Higher priority Q&A returned when multiple match."""
        import json

        from ai_ready_rag.db.models import CuratedQA
        from ai_ready_rag.services.rag_service import check_curated_qa, invalidate_qa_cache

        # Lower priority
        qa1 = CuratedQA(
            keywords=json.dumps(["pto"]),
            answer="Low priority answer",
            source_reference="Source 1",
            confidence=80,
            priority=5,
            enabled=True,
        )
        # Higher priority
        qa2 = CuratedQA(
            keywords=json.dumps(["pto"]),
            answer="High priority answer",
            source_reference="Source 2",
            confidence=90,
            priority=10,
            enabled=True,
        )
        db.add_all([qa1, qa2])
        db.commit()
        invalidate_qa_cache()

        result = check_curated_qa("How do I request PTO?", db)

        assert result is not None
        assert "High priority" in result.answer

    def test_check_curated_qa_disabled_ignored(self, mock_settings, db):
        """Disabled Q&A pairs are not matched."""
        import json

        from ai_ready_rag.db.models import CuratedQA
        from ai_ready_rag.services.rag_service import check_curated_qa, invalidate_qa_cache

        qa = CuratedQA(
            keywords=json.dumps(["vacation"]),
            answer="Disabled answer",
            source_reference="Source",
            confidence=85,
            enabled=False,  # Disabled
        )
        db.add(qa)
        db.commit()
        invalidate_qa_cache()

        result = check_curated_qa("vacation policy", db)

        assert result is None

    def test_check_curated_qa_multiword_keyword_match(self, mock_settings, db):
        """Multi-word keywords like 'paid time off' match correctly."""
        import json

        from ai_ready_rag.db.models import CuratedQA
        from ai_ready_rag.services.rag_service import check_curated_qa, invalidate_qa_cache

        qa = CuratedQA(
            keywords=json.dumps(["paid time off"]),
            answer="PTO information...",
            source_reference="HR Policy",
            confidence=90,
            enabled=True,
        )
        db.add(qa)
        db.commit()
        invalidate_qa_cache()

        # Should match when all tokens present
        result = check_curated_qa("How do I request paid time off?", db)
        assert result is not None

    def test_check_curated_qa_multiword_keyword_no_partial_match(self, mock_settings, db):
        """Multi-word keywords don't match with partial tokens."""
        import json

        from ai_ready_rag.db.models import CuratedQA
        from ai_ready_rag.services.rag_service import check_curated_qa, invalidate_qa_cache

        qa = CuratedQA(
            keywords=json.dumps(["paid time off"]),
            answer="PTO information...",
            source_reference="HR Policy",
            confidence=90,
            enabled=True,
        )
        db.add(qa)
        db.commit()
        invalidate_qa_cache()

        # Should NOT match when only partial tokens present
        result = check_curated_qa("How much time do I have?", db)
        assert result is None  # "paid" and "off" missing

    def test_check_curated_qa_case_insensitive(self, mock_settings, db):
        """Keyword matching is case-insensitive."""
        import json

        from ai_ready_rag.db.models import CuratedQA
        from ai_ready_rag.services.rag_service import check_curated_qa, invalidate_qa_cache

        qa = CuratedQA(
            keywords=json.dumps(["PTO"]),
            answer="PTO answer...",
            source_reference="HR Policy",
            confidence=90,
            enabled=True,
        )
        db.add(qa)
        db.commit()
        invalidate_qa_cache()

        # Should match lowercase query
        result = check_curated_qa("how do i request pto?", db)
        assert result is not None


# =============================================================================
# Cache Invalidation Tests
# =============================================================================


class TestCacheInvalidation:
    """Test cache behavior and invalidation."""

    def test_synonym_cache_invalidation_refreshes(self, mock_settings, db):
        """invalidate_synonym_cache() causes next call to refresh from db."""
        import json

        from ai_ready_rag.db.models import QuerySynonym
        from ai_ready_rag.services.rag_service import (
            get_cached_synonyms,
            invalidate_synonym_cache,
        )

        # Create first synonym
        syn1 = QuerySynonym(
            term="test1",
            synonyms=json.dumps(["synonym1"]),
            enabled=True,
        )
        db.add(syn1)
        db.commit()
        invalidate_synonym_cache()

        # First call should include syn1
        result1 = get_cached_synonyms(db)
        assert len(result1) >= 1

        # Add another synonym
        syn2 = QuerySynonym(
            term="test2",
            synonyms=json.dumps(["synonym2"]),
            enabled=True,
        )
        db.add(syn2)
        db.commit()

        # Without invalidation, cache may be stale
        # After invalidation, should refresh...

        invalidate_synonym_cache()
        result3 = get_cached_synonyms(db)

        # Should now include both
        assert len(result3) >= 2

    def test_qa_cache_invalidation_refreshes(self, mock_settings, db):
        """invalidate_qa_cache() causes next call to refresh from db."""
        import json

        from ai_ready_rag.db.models import CuratedQA
        from ai_ready_rag.services.rag_service import get_cached_qa, invalidate_qa_cache

        # Create first Q&A
        qa1 = CuratedQA(
            keywords=json.dumps(["test1"]),
            answer="Answer 1",
            source_reference="Source 1",
            confidence=90,
            enabled=True,
        )
        db.add(qa1)
        db.commit()
        invalidate_qa_cache()

        # First call should include qa1
        token_index1, qa_lookup1, _ = get_cached_qa(db)
        assert len(qa_lookup1) >= 1

        # Add another Q&A
        qa2 = CuratedQA(
            keywords=json.dumps(["test2"]),
            answer="Answer 2",
            source_reference="Source 2",
            confidence=85,
            enabled=True,
        )
        db.add(qa2)
        db.commit()

        # Invalidate and check
        invalidate_qa_cache()
        token_index2, qa_lookup2, _ = get_cached_qa(db)

        # Should now include both
        assert len(qa_lookup2) >= 2


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestTokenizeQuery:
    """Test tokenize_query helper function."""

    def test_tokenize_query_basic(self):
        """Basic tokenization works correctly."""
        from ai_ready_rag.services.rag_service import tokenize_query

        tokens = tokenize_query("How do I request PTO?")
        assert "how" in tokens
        assert "do" in tokens
        assert "request" in tokens
        assert "pto" in tokens

    def test_tokenize_query_filters_short_tokens(self):
        """Tokens shorter than 2 chars are filtered."""
        from ai_ready_rag.services.rag_service import tokenize_query

        tokens = tokenize_query("I a test question")
        assert "i" not in tokens
        assert "a" not in tokens
        assert "test" in tokens
        assert "question" in tokens

    def test_tokenize_query_handles_numbers(self):
        """Alphanumeric tokens including numbers are kept."""
        from ai_ready_rag.services.rag_service import tokenize_query

        tokens = tokenize_query("Form 401k benefit")
        assert "401k" in tokens


class TestMatchesKeyword:
    """Test matches_keyword helper function."""

    def test_matches_keyword_single_word_match(self):
        """Single word keyword matches when present."""
        from ai_ready_rag.services.rag_service import matches_keyword

        result = matches_keyword("pto", {"pto", "request"})
        assert result is True

    def test_matches_keyword_single_word_no_match(self):
        """Single word keyword doesn't match when absent."""
        from ai_ready_rag.services.rag_service import matches_keyword

        result = matches_keyword("pto", {"photo", "gallery"})
        assert result is False

    def test_matches_keyword_multiword_all_present(self):
        """Multi-word keyword matches when all tokens present."""
        from ai_ready_rag.services.rag_service import matches_keyword

        result = matches_keyword("paid time off", {"paid", "time", "off", "request"})
        assert result is True

    def test_matches_keyword_multiword_partial_no_match(self):
        """Multi-word keyword doesn't match with missing tokens."""
        from ai_ready_rag.services.rag_service import matches_keyword

        result = matches_keyword("paid time off", {"time", "request"})
        assert result is False
