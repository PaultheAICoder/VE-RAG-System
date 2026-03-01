"""Tests for QueryRouter integration with RAGService.

Tests two acceptance criteria from issue #424:
1. Unit test: mock registry with one template, assert SQL route taken for matching query
2. Integration test: assert non-matching query still reaches _run_rag_pipeline
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_ready_rag.config import Settings
from ai_ready_rag.modules.registry import ModuleRegistry, SQLTemplate
from ai_ready_rag.services.cache_service import CacheService
from ai_ready_rag.services.query_router import QueryRouter
from ai_ready_rag.services.rag_service import RAGRequest, RAGService
from ai_ready_rag.services.vector_service import VectorService

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry singleton between tests."""
    ModuleRegistry.reset()
    yield
    ModuleRegistry.reset()


@pytest.fixture
def mock_settings():
    """Mock Settings with structured_query_enabled=True."""
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
    settings.rag_max_chunks_per_doc = 5
    settings.rag_total_context_chunks = 8
    settings.rag_dedup_candidates_cap = 15
    settings.rag_chunk_overlap_threshold = 0.9
    settings.rag_enable_query_expansion = False
    settings.rag_enable_hallucination_check = False
    settings.forms_db_path = None
    # Key: structured queries enabled
    settings.structured_query_enabled = True
    settings.structured_query_row_cap = 100
    return settings


@pytest.fixture
def mock_settings_disabled(mock_settings):
    """Mock Settings with structured_query_enabled=False."""
    mock_settings.structured_query_enabled = False
    return mock_settings


@pytest.fixture
def disabled_cache():
    """A CacheService stub with enabled=False so the cache path is skipped."""
    cache = MagicMock(spec=CacheService)
    cache.enabled = False
    return cache


@pytest.fixture
def router_with_coverage_template():
    """QueryRouter with a single coverage lookup SQL template registered."""
    registry = ModuleRegistry.get_instance()
    registry.register_sql_templates(
        "test_module",
        {
            "coverage_lookup": SQLTemplate(
                name="coverage_lookup",
                sql="SELECT name, limit FROM coverages LIMIT :row_cap",
                trigger_phrases=["coverage", "insurance limit", "what is the coverage"],
                description="Look up coverage limits",
            ),
        },
    )
    return QueryRouter(sql_confidence_threshold=0.3)


@pytest.fixture
def mock_db():
    """Mock database session."""
    return MagicMock()


@pytest.fixture
def mock_vector_service():
    """Mock VectorService."""
    vs = AsyncMock(spec=VectorService)
    vs.search = AsyncMock(return_value=[])
    return vs


# =============================================================================
# Tests: SQL route taken on matching query
# =============================================================================


class TestSQLRouteOnMatch:
    """Unit test: mock registry with one template, assert SQL route taken for matching query."""

    @pytest.mark.asyncio
    async def test_sql_route_taken_for_matching_query(
        self,
        mock_settings,
        router_with_coverage_template,
        mock_db,
        mock_vector_service,
        disabled_cache,
    ):
        """A query matching a registered SQL template's trigger_phrases returns SQL-derived answer."""
        # Arrange: mock the DB execution to return sample rows
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("General Liability", 1000000)]
        mock_result.keys.return_value = ["name", "limit"]
        mock_db.execute.return_value = mock_result

        rag_service = RAGService(
            mock_settings,
            vector_service=mock_vector_service,
            cache_service=disabled_cache,
            query_router=router_with_coverage_template,
        )

        with (
            patch(
                "ai_ready_rag.services.rag_service.check_curated_qa",
                return_value=None,
            ),
            patch(
                "ai_ready_rag.services.settings_service.SettingsService",
                autospec=True,
            ) as MockSettingsService,
            patch(
                "ai_ready_rag.services.rag_service.RAGService.validate_model",
                new_callable=AsyncMock,
            ),
            patch(
                "ai_ready_rag.modules.registry.get_registry",
                return_value=ModuleRegistry.get_instance(),
            ),
        ):
            MockSettingsService.return_value.get.return_value = "retrieve_only"

            request = RAGRequest(
                query="what is the coverage limit for property?",
                user_tags=["hr"],
            )
            response = await rag_service.generate(request, mock_db)

        # Assert SQL-route taken
        assert response.routing_decision == "SQL"
        assert response.model_used == "claude-cli"
        assert response.action == "CITE"
        assert response.grounded is True
        # Answer should contain column headers or row data
        assert "name" in response.answer or "General Liability" in response.answer

    @pytest.mark.asyncio
    async def test_sql_route_metadata_records_sql_routing(
        self,
        mock_settings,
        router_with_coverage_template,
        mock_db,
        mock_vector_service,
        disabled_cache,
    ):
        """routing_decision is 'SQL' when SQL route is taken."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("GL", 500000)]
        mock_result.keys.return_value = ["type", "limit"]
        mock_db.execute.return_value = mock_result

        rag_service = RAGService(
            mock_settings,
            vector_service=mock_vector_service,
            cache_service=disabled_cache,
            query_router=router_with_coverage_template,
        )

        with (
            patch("ai_ready_rag.services.rag_service.check_curated_qa", return_value=None),
            patch(
                "ai_ready_rag.services.settings_service.SettingsService", autospec=True
            ) as MockSvc,
            patch(
                "ai_ready_rag.services.rag_service.RAGService.validate_model",
                new_callable=AsyncMock,
            ),
            patch(
                "ai_ready_rag.modules.registry.get_registry",
                return_value=ModuleRegistry.get_instance(),
            ),
        ):
            MockSvc.return_value.get.return_value = "retrieve_only"

            request = RAGRequest(
                query="what is the coverage line?",
                user_tags=None,
            )
            response = await rag_service.generate(request, mock_db)

        assert response.routing_decision == "SQL"

    @pytest.mark.asyncio
    async def test_row_cap_respected(
        self,
        mock_settings,
        router_with_coverage_template,
        mock_db,
        mock_vector_service,
        disabled_cache,
    ):
        """Row cap (structured_query_row_cap) is passed as :row_cap param to SQL execution."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = ["name", "limit"]
        mock_db.execute.return_value = mock_result

        mock_settings.structured_query_row_cap = 42

        rag_service = RAGService(
            mock_settings,
            vector_service=mock_vector_service,
            cache_service=disabled_cache,
            query_router=router_with_coverage_template,
        )

        with (
            patch("ai_ready_rag.services.rag_service.check_curated_qa", return_value=None),
            patch(
                "ai_ready_rag.services.settings_service.SettingsService", autospec=True
            ) as MockSvc,
            patch(
                "ai_ready_rag.services.rag_service.RAGService.validate_model",
                new_callable=AsyncMock,
            ),
            patch(
                "ai_ready_rag.modules.registry.get_registry",
                return_value=ModuleRegistry.get_instance(),
            ),
        ):
            MockSvc.return_value.get.return_value = "retrieve_only"

            request = RAGRequest(
                query="what is the coverage limit?",
                user_tags=[],
            )
            await rag_service.generate(request, mock_db)

        # Verify db.execute was called with row_cap=42 in params
        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args
        # Second positional argument is the params dict
        params = call_args[0][1]
        assert params.get("row_cap") == 42


# =============================================================================
# Tests: Non-matching query still reaches _run_rag_pipeline
# =============================================================================


class TestRAGFallbackOnNonMatch:
    """Integration test: non-matching query still reaches _run_rag_pipeline."""

    @pytest.mark.asyncio
    async def test_non_matching_query_goes_to_rag(
        self,
        mock_settings,
        router_with_coverage_template,
        mock_db,
        mock_vector_service,
        disabled_cache,
    ):
        """A query not matching any SQL template continues through _run_rag_pipeline."""
        rag_service = RAGService(
            mock_settings,
            vector_service=mock_vector_service,
            cache_service=disabled_cache,
            query_router=router_with_coverage_template,
        )

        with (
            patch("ai_ready_rag.services.rag_service.check_curated_qa", return_value=None),
            patch(
                "ai_ready_rag.services.settings_service.SettingsService", autospec=True
            ) as MockSvc,
            patch(
                "ai_ready_rag.services.rag_service.RAGService.validate_model",
                new_callable=AsyncMock,
            ),
            patch.object(
                rag_service,
                "_run_rag_pipeline",
                new_callable=AsyncMock,
            ) as mock_pipeline,
        ):
            MockSvc.return_value.get.return_value = "retrieve_only"

            # Mock _run_rag_pipeline to return a minimal valid response
            from ai_ready_rag.services.rag_service import ConfidenceScore, RAGResponse

            mock_pipeline.return_value = (
                RAGResponse(
                    answer="Board meeting notes...",
                    confidence=ConfidenceScore(
                        overall=80, retrieval_score=0.8, coverage_score=0.7, llm_score=80
                    ),
                    citations=[],
                    action="CITE",
                    route_to=None,
                    model_used="llama3.2",
                    context_chunks_used=2,
                    context_tokens_used=300,
                    generation_time_ms=100.0,
                    grounded=False,
                    routing_decision="RETRIEVE",
                ),
                [],
            )

            request = RAGRequest(
                query="summarize the board meeting minutes from December",
                user_tags=["hr"],
            )
            response = await rag_service.generate(request, mock_db)

        # Assert _run_rag_pipeline was called (SQL route NOT taken)
        mock_pipeline.assert_called_once()
        assert response.routing_decision == "RETRIEVE"

    @pytest.mark.asyncio
    async def test_disabled_flag_bypasses_sql_router(
        self,
        mock_settings_disabled,
        router_with_coverage_template,
        mock_db,
        mock_vector_service,
        disabled_cache,
    ):
        """When structured_query_enabled=False, all queries go to _run_rag_pipeline."""
        rag_service = RAGService(
            mock_settings_disabled,
            vector_service=mock_vector_service,
            cache_service=disabled_cache,
            query_router=router_with_coverage_template,
        )

        with (
            patch("ai_ready_rag.services.rag_service.check_curated_qa", return_value=None),
            patch(
                "ai_ready_rag.services.settings_service.SettingsService", autospec=True
            ) as MockSvc,
            patch(
                "ai_ready_rag.services.rag_service.RAGService.validate_model",
                new_callable=AsyncMock,
            ),
            patch.object(
                rag_service,
                "_run_rag_pipeline",
                new_callable=AsyncMock,
            ) as mock_pipeline,
        ):
            MockSvc.return_value.get.return_value = "retrieve_only"

            from ai_ready_rag.services.rag_service import ConfidenceScore, RAGResponse

            mock_pipeline.return_value = (
                RAGResponse(
                    answer="Coverage is $1M.",
                    confidence=ConfidenceScore(
                        overall=80, retrieval_score=0.8, coverage_score=0.7, llm_score=80
                    ),
                    citations=[],
                    action="CITE",
                    route_to=None,
                    model_used="llama3.2",
                    context_chunks_used=1,
                    context_tokens_used=100,
                    generation_time_ms=50.0,
                    grounded=False,
                    routing_decision="RETRIEVE",
                ),
                [],
            )

            request = RAGRequest(
                query="what is the coverage limit?",
                user_tags=["insurance"],
            )
            await rag_service.generate(request, mock_db)

        # _run_rag_pipeline should be called even though query matches trigger phrases
        mock_pipeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_query_router_goes_to_rag(
        self,
        mock_settings,
        mock_db,
        mock_vector_service,
        disabled_cache,
    ):
        """When query_router=None, all queries go to _run_rag_pipeline."""
        # RAGService without a query_router
        rag_service = RAGService(
            mock_settings,
            vector_service=mock_vector_service,
            cache_service=disabled_cache,
            query_router=None,  # Explicit None
        )

        with (
            patch("ai_ready_rag.services.rag_service.check_curated_qa", return_value=None),
            patch(
                "ai_ready_rag.services.settings_service.SettingsService", autospec=True
            ) as MockSvc,
            patch(
                "ai_ready_rag.services.rag_service.RAGService.validate_model",
                new_callable=AsyncMock,
            ),
            patch.object(
                rag_service,
                "_run_rag_pipeline",
                new_callable=AsyncMock,
            ) as mock_pipeline,
        ):
            MockSvc.return_value.get.return_value = "retrieve_only"

            from ai_ready_rag.services.rag_service import ConfidenceScore, RAGResponse

            mock_pipeline.return_value = (
                RAGResponse(
                    answer="No router, goes to RAG.",
                    confidence=ConfidenceScore(
                        overall=70, retrieval_score=0.7, coverage_score=0.6, llm_score=70
                    ),
                    citations=[],
                    action="CITE",
                    route_to=None,
                    model_used="llama3.2",
                    context_chunks_used=1,
                    context_tokens_used=100,
                    generation_time_ms=80.0,
                    grounded=False,
                    routing_decision="RETRIEVE",
                ),
                [],
            )

            request = RAGRequest(
                query="what is the coverage limit?",
                user_tags=[],
            )
            await rag_service.generate(request, mock_db)

        mock_pipeline.assert_called_once()
