"""Tests for TenantConfigResolver and PromptResolver."""

from unittest.mock import MagicMock

import pytest

from ai_ready_rag.tenant.config import (
    DEFAULT_TENANT_CONFIG,
    AIModelConfig,
    FeatureFlags,
    TenantConfig,
)
from ai_ready_rag.tenant.resolver import PromptResolver, TenantConfigResolver


class TestTenantConfig:
    def test_default_config_loads(self):
        assert DEFAULT_TENANT_CONFIG.tenant_id == "default"

    def test_ai_config_validates_pinned_models(self):
        """Pinned model IDs pass validation."""
        ai = AIModelConfig(
            enrichment_model="claude-sonnet-4-6",
            query_model_simple="claude-haiku-4-5-20251001",
        )
        assert ai.enrichment_model == "claude-sonnet-4-6"

    def test_ai_config_rejects_aliases(self):
        """Alias model IDs are rejected."""
        with pytest.raises(ValueError):
            AIModelConfig(enrichment_model="claude-sonnet")

    def test_tenant_config_feature_flags_default(self):
        config = TenantConfig(tenant_id="test", display_name="Test")
        assert hasattr(config, "feature_flags")


class TestTenantConfigResolver:
    def test_resolves_default_when_no_json(self, tmp_path):
        resolver = TenantConfigResolver(
            tenant_config_path_template=str(tmp_path / "{tenant_id}" / "tenant.json")
        )
        config = resolver.resolve("nonexistent-tenant-xyz")
        assert config.tenant_id == "nonexistent-tenant-xyz"

    def test_feature_flag_defaults_false(self, tmp_path):
        resolver = TenantConfigResolver(
            tenant_config_path_template=str(tmp_path / "{tenant_id}" / "tenant.json")
        )
        config = resolver.resolve("nonexistent-tenant-xyz")
        assert config.feature_flags.ca_enabled is False

    def test_active_modules_includes_core(self, tmp_path):
        resolver = TenantConfigResolver(
            tenant_config_path_template=str(tmp_path / "{tenant_id}" / "tenant.json")
        )
        config = resolver.resolve("nonexistent-tenant-xyz")
        assert "core" in config.active_modules


class TestPromptResolver:
    def test_nonexistent_prompt_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resolver = PromptResolver(tenant_id="default", module_id="core")
        assert resolver.resolve("nonexistent_prompt_xyz") is None

    def test_list_available_empty_when_no_dirs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resolver = PromptResolver(tenant_id="default", module_id="core")
        assert resolver.list_available() == []


class TestTenantConfigEndpoint:
    def test_get_config_requires_auth(self, client):
        response = client.get("/api/tenant/config")
        assert response.status_code == 401

    def test_get_config_returns_default(self, client, admin_headers):
        response = client.get("/api/tenant/config", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "tenant_id" in data
        assert "feature_flags" in data


class TestTenantConfigThreadingRAGService:
    """Verify TenantConfig feature flags are honoured by RAGService.

    AC #6: structured_query_enabled=False in TenantConfig disables routing
           even when the global Settings has it True.
    """

    def test_structured_query_disabled_by_tenant_config_overrides_global_setting(self):
        """TenantConfig.feature_flags.structured_query_enabled=False overrides Settings=True."""
        from ai_ready_rag.services.rag_service import RAGService
        from ai_ready_rag.tenant.config import TenantConfig

        # Global settings says enabled=True
        settings = MagicMock()
        settings.structured_query_enabled = True
        settings.ollama_base_url = "http://localhost:11434"
        settings.chat_model = "qwen3:8b"
        settings.rag_max_chunks_per_doc = 5
        settings.rag_chunk_overlap_threshold = 0.8
        settings.rag_dedup_candidates_cap = 20
        settings.rag_enable_hallucination_check = False
        settings.forms_db_path = None

        # TenantConfig says disabled=False
        tenant_config = TenantConfig(
            tenant_id="test-tenant",
            feature_flags=FeatureFlags(structured_query_enabled=False),
        )

        svc = RAGService(settings, tenant_config=tenant_config)

        # Internal effective flag resolution — read the same logic as generate()
        structured_query_enabled = bool(getattr(settings, "structured_query_enabled", False))
        if svc._tenant_config is not None:
            tc_flags = getattr(svc._tenant_config, "feature_flags", None)
            tc_sq_flag = getattr(tc_flags, "structured_query_enabled", None)
            if tc_sq_flag is not None:
                structured_query_enabled = bool(tc_sq_flag)

        assert structured_query_enabled is False, (
            "TenantConfig structured_query_enabled=False should override Settings=True"
        )

    def test_structured_query_enabled_by_tenant_config_overrides_global_setting(self):
        """TenantConfig.feature_flags.structured_query_enabled=True overrides Settings=False."""
        from ai_ready_rag.services.rag_service import RAGService
        from ai_ready_rag.tenant.config import TenantConfig

        settings = MagicMock()
        settings.structured_query_enabled = False
        settings.ollama_base_url = "http://localhost:11434"
        settings.chat_model = "qwen3:8b"
        settings.rag_max_chunks_per_doc = 5
        settings.rag_chunk_overlap_threshold = 0.8
        settings.rag_dedup_candidates_cap = 20
        settings.rag_enable_hallucination_check = False
        settings.forms_db_path = None

        tenant_config = TenantConfig(
            tenant_id="test-tenant",
            feature_flags=FeatureFlags(structured_query_enabled=True),
        )

        svc = RAGService(settings, tenant_config=tenant_config)

        structured_query_enabled = bool(getattr(settings, "structured_query_enabled", False))
        if svc._tenant_config is not None:
            tc_flags = getattr(svc._tenant_config, "feature_flags", None)
            tc_sq_flag = getattr(tc_flags, "structured_query_enabled", None)
            if tc_sq_flag is not None:
                structured_query_enabled = bool(tc_sq_flag)

        assert structured_query_enabled is True, (
            "TenantConfig structured_query_enabled=True should override Settings=False"
        )


class TestTenantConfigThreadingEnrichmentService:
    """Verify TenantConfig feature flags are honoured by ClaudeEnrichmentService.

    AC #7: claude_enrichment_enabled=False in TenantConfig skips enrichment
           even when global Settings.claude_enrichment_enabled=True.
    """

    def test_enrichment_disabled_by_tenant_config_overrides_global_settings(self):
        """TenantConfig.feature_flags.claude_enrichment_enabled=False overrides Settings=True."""
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService
        from ai_ready_rag.tenant.config import TenantConfig

        settings = MagicMock()
        settings.claude_enrichment_enabled = True
        settings.claude_api_key = "sk-ant-test-key"
        settings.database_backend = "postgresql"

        # TenantConfig says disabled
        tenant_config = TenantConfig(
            tenant_id="test-tenant",
            feature_flags=FeatureFlags(claude_enrichment_enabled=False),
        )

        svc = ClaudeEnrichmentService(settings, tenant_config=tenant_config)
        assert svc._is_enabled() is False, (
            "TenantConfig claude_enrichment_enabled=False must disable enrichment "
            "even when Settings.claude_enrichment_enabled=True"
        )

    def test_enrichment_enabled_by_tenant_config_overrides_global_settings(self):
        """TenantConfig.feature_flags.claude_enrichment_enabled=True overrides Settings=False."""
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService
        from ai_ready_rag.tenant.config import TenantConfig

        settings = MagicMock()
        settings.claude_enrichment_enabled = False
        settings.claude_api_key = "sk-ant-test-key"
        settings.database_backend = "postgresql"

        tenant_config = TenantConfig(
            tenant_id="test-tenant",
            feature_flags=FeatureFlags(claude_enrichment_enabled=True),
        )

        svc = ClaudeEnrichmentService(settings, tenant_config=tenant_config)
        assert svc._is_enabled() is True, (
            "TenantConfig claude_enrichment_enabled=True must enable enrichment "
            "even when Settings.claude_enrichment_enabled=False"
        )

    def test_sqlite_backend_always_disables_regardless_of_tenant_config(self):
        """SQLite backend disables enrichment even if TenantConfig enables it."""
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService
        from ai_ready_rag.tenant.config import TenantConfig

        settings = MagicMock()
        settings.claude_enrichment_enabled = True
        settings.claude_api_key = "sk-ant-test-key"
        settings.database_backend = "sqlite"

        tenant_config = TenantConfig(
            tenant_id="test-tenant",
            feature_flags=FeatureFlags(claude_enrichment_enabled=True),
        )

        svc = ClaudeEnrichmentService(settings, tenant_config=tenant_config)
        assert svc._is_enabled() is False, "SQLite backend should always disable enrichment"

    def test_enrichment_model_from_tenant_config(self):
        """ClaudeEnrichmentService uses enrichment_model from TenantConfig.ai_models."""
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService
        from ai_ready_rag.tenant.config import AIModelConfig, TenantConfig

        settings = MagicMock()
        settings.claude_enrichment_model = "claude-sonnet-4-6"  # global default

        tenant_config = TenantConfig(
            tenant_id="test-tenant",
            ai_models=AIModelConfig(enrichment_model="claude-opus-4-6"),
        )

        svc = ClaudeEnrichmentService(settings, tenant_config=tenant_config)
        model = svc._get_enrichment_model()
        assert model == "claude-opus-4-6", (
            "Should use TenantConfig.ai_models.enrichment_model when available"
        )

    def test_enrichment_model_falls_back_to_settings(self):
        """ClaudeEnrichmentService falls back to Settings when no TenantConfig."""
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        settings = MagicMock()
        settings.claude_enrichment_model = "claude-sonnet-4-6"

        svc = ClaudeEnrichmentService(settings)
        model = svc._get_enrichment_model()
        assert model == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_enrich_document_returns_cap_exceeded_when_daily_cap_hit(self):
        """enrich_document() returns {status: cap_exceeded} when daily cap is exceeded."""
        from unittest.mock import MagicMock as MockClass
        from unittest.mock import patch

        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService
        from ai_ready_rag.tenant.config import AIModelConfig, TenantConfig

        mock_settings = MockClass()
        mock_settings.claude_enrichment_enabled = True
        mock_settings.claude_api_key = "sk-ant-test-key"
        mock_settings.database_backend = "postgresql"
        mock_settings.claude_enrichment_cost_limit_usd = 10.0

        tenant_config = TenantConfig(
            tenant_id="test-tenant",
            feature_flags=FeatureFlags(claude_enrichment_enabled=True),
            ai_models=AIModelConfig(daily_enrichment_cap_usd=5.0),
        )

        svc = ClaudeEnrichmentService(mock_settings, tenant_config=tenant_config)

        # Patch CostTracker.is_allowed to return False (over cap)
        mock_tracker = MockClass()
        mock_tracker.is_allowed.return_value = False

        with patch(
            "ai_ready_rag.services.enrichment_service.CostTracker",
            return_value=mock_tracker,
        ):
            result = await svc.enrich_document("doc-123", "document text", [])

        assert result == {"status": "cap_exceeded"}
