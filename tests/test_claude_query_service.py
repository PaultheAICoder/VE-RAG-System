"""Tests for ClaudeQueryService and ClaudeModelRouter."""

from unittest.mock import MagicMock

import pytest

from ai_ready_rag.services.claude_query_service import (
    COMPLEX_QUERY_SIGNALS,
    ClaudeModelRouter,
    ClaudeQueryService,
    QueryResponse,
)


class TestClaudeModelRouter:
    def test_simple_query_uses_haiku(self):
        router = ClaudeModelRouter()
        model, is_complex = router.select_model("What is the coverage limit?")
        assert "haiku" in model
        assert is_complex is False

    def test_complex_query_uses_sonnet(self):
        router = ClaudeModelRouter()
        model, is_complex = router.select_model(
            "Compare the coverage across all policies and analyze the gaps"
        )
        assert "sonnet" in model
        assert is_complex is True

    def test_compliance_signal_triggers_complex(self):
        router = ClaudeModelRouter()
        _, is_complex = router.select_model("Check compliance with Fannie Mae requirements")
        assert is_complex is True

    def test_from_settings(self):
        settings = MagicMock()
        settings.claude_query_model_simple = "claude-haiku-4-5-20251001"
        settings.claude_query_model_complex = "claude-sonnet-4-6"
        router = ClaudeModelRouter.from_settings(settings)
        model, _ = router.select_model("simple question")
        assert model == "claude-haiku-4-5-20251001"

    def test_complex_signals_list_not_empty(self):
        assert len(COMPLEX_QUERY_SIGNALS) >= 5

    def test_from_tenant_config_uses_tenant_models(self):
        """Tenant config overrides global settings models."""
        settings = MagicMock()
        settings.claude_query_model_simple = "claude-haiku-4-5-20251001"
        settings.claude_query_model_complex = "claude-sonnet-4-6"

        tenant_config = MagicMock()
        tenant_config.ai_models.query_model_simple = "claude-haiku-4-5-20251001"
        tenant_config.ai_models.query_model_complex = "claude-sonnet-4-6"

        router = ClaudeModelRouter.from_tenant_config(tenant_config, settings)
        model, _ = router.select_model("simple question")
        assert "haiku" in model

    def test_from_tenant_config_falls_back_to_settings_on_error(self):
        """If tenant config raises, falls back to settings gracefully."""
        settings = MagicMock()
        settings.claude_query_model_simple = "claude-haiku-4-5-20251001"
        settings.claude_query_model_complex = "claude-sonnet-4-6"

        broken_tenant = MagicMock(spec=[])  # no ai_models attribute

        router = ClaudeModelRouter.from_tenant_config(broken_tenant, settings)
        # Should not raise and should use settings values
        model, _ = router.select_model("simple question")
        assert model == "claude-haiku-4-5-20251001"

    def test_custom_signals(self):
        """Custom complex_signals override defaults."""
        router = ClaudeModelRouter(complex_signals=["custom_trigger"])
        _, is_complex = router.select_model("query with custom_trigger inside")
        assert is_complex is True

        _, is_complex_2 = router.select_model("simple question without trigger")
        assert is_complex_2 is False

    def test_case_insensitive_signal_matching(self):
        """Signal matching is case-insensitive."""
        router = ClaudeModelRouter()
        _, is_complex = router.select_model("COMPARE the two documents")
        assert is_complex is True


class TestClaudeQueryServiceDisabled:
    def test_disabled_on_sqlite(self):
        settings = MagicMock()
        settings.claude_query_enabled = True
        settings.claude_api_key = "sk-test"
        settings.database_backend = "sqlite"
        svc = ClaudeQueryService(settings)
        assert svc._is_enabled() is False

    def test_disabled_when_no_key(self):
        settings = MagicMock()
        settings.claude_query_enabled = True
        settings.claude_api_key = None
        settings.database_backend = "postgresql"
        svc = ClaudeQueryService(settings)
        assert svc._is_enabled() is False

    def test_disabled_when_flag_off(self):
        settings = MagicMock()
        settings.claude_query_enabled = False
        settings.claude_api_key = "sk-test"
        settings.database_backend = "postgresql"
        svc = ClaudeQueryService(settings)
        assert svc._is_enabled() is False

    def test_enabled_on_postgresql_with_key_and_flag(self):
        settings = MagicMock()
        settings.claude_query_enabled = True
        settings.claude_api_key = "sk-test"
        settings.database_backend = "postgresql"
        svc = ClaudeQueryService(settings)
        assert svc._is_enabled() is True

    @pytest.mark.asyncio
    async def test_answer_returns_none_when_disabled(self):
        settings = MagicMock()
        settings.claude_query_enabled = False
        settings.database_backend = "sqlite"
        svc = ClaudeQueryService(settings)
        result = await svc.answer("test query", [])
        assert result is None

    def test_query_response_dataclass(self):
        resp = QueryResponse(
            answer="test",
            model_used="claude-haiku-4-5-20251001",
            is_complex=False,
            token_cost=100,
            cost_usd=0.00008,
            route_type="claude_simple",
        )
        assert resp.route_type == "claude_simple"

    def test_disabled_when_database_backend_missing(self):
        """Falls back to sqlite behaviour when database_backend attr is absent."""
        settings = MagicMock(spec=[])  # no attributes at all
        svc = ClaudeQueryService(settings)
        assert svc._is_enabled() is False

    def test_service_initialises_with_tenant_config(self):
        """Service accepts tenant_config and builds router from it."""
        settings = MagicMock()
        settings.claude_query_enabled = True
        settings.claude_api_key = "sk-test"
        settings.database_backend = "postgresql"
        settings.claude_query_model_simple = "claude-haiku-4-5-20251001"
        settings.claude_query_model_complex = "claude-sonnet-4-6"

        tenant_config = MagicMock()
        tenant_config.ai_models.query_model_simple = "claude-haiku-4-5-20251001"
        tenant_config.ai_models.query_model_complex = "claude-sonnet-4-6"

        svc = ClaudeQueryService(settings, tenant_config=tenant_config)
        assert svc._is_enabled() is True


class TestClaudeQueryServiceGetClient:
    def test_get_client_raises_when_anthropic_missing(self):
        """Raises RuntimeError with helpful message when anthropic package absent."""
        import sys

        settings = MagicMock()
        settings.claude_api_key = "sk-test"
        svc = ClaudeQueryService(settings)

        # Temporarily hide the anthropic module
        original = sys.modules.get("anthropic")
        sys.modules["anthropic"] = None  # type: ignore[assignment]

        try:
            with pytest.raises((RuntimeError, ImportError)):
                svc._get_client()
        finally:
            if original is None:
                sys.modules.pop("anthropic", None)
            else:
                sys.modules["anthropic"] = original


class TestQueryResponseDataclass:
    def test_route_type_claude_simple(self):
        resp = QueryResponse(
            answer="The coverage limit is $500,000.",
            model_used="claude-haiku-4-5-20251001",
            is_complex=False,
            token_cost=250,
            cost_usd=0.0002,
            route_type="claude_simple",
        )
        assert resp.route_type == "claude_simple"
        assert resp.is_complex is False
        assert "haiku" in resp.model_used

    def test_route_type_claude_complex(self):
        resp = QueryResponse(
            answer="The analysis shows...",
            model_used="claude-sonnet-4-6",
            is_complex=True,
            token_cost=1500,
            cost_usd=0.027,
            route_type="claude_complex",
        )
        assert resp.route_type == "claude_complex"
        assert resp.is_complex is True
        assert "sonnet" in resp.model_used
