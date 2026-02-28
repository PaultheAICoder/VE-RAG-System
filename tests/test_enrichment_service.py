"""Tests for ClaudeEnrichmentService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestClaudeEnrichmentServiceUnit:
    def test_import(self):
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        assert ClaudeEnrichmentService is not None

    def test_disabled_on_sqlite(self):
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        settings = MagicMock()
        settings.claude_enrichment_enabled = True
        settings.claude_api_key = "sk-ant-test"
        settings.database_backend = "sqlite"
        svc = ClaudeEnrichmentService(settings)
        assert svc._is_enabled() is False

    def test_disabled_when_no_api_key(self):
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        settings = MagicMock()
        settings.claude_enrichment_enabled = True
        settings.claude_api_key = None
        settings.database_backend = "postgresql"
        svc = ClaudeEnrichmentService(settings)
        assert svc._is_enabled() is False

    def test_disabled_when_flag_off(self):
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        settings = MagicMock()
        settings.claude_enrichment_enabled = False
        settings.claude_api_key = "sk-ant-test"
        settings.database_backend = "postgresql"
        svc = ClaudeEnrichmentService(settings)
        assert svc._is_enabled() is False

    def test_enabled_on_postgresql_with_key(self):
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        settings = MagicMock()
        settings.claude_enrichment_enabled = True
        settings.claude_api_key = "sk-ant-test"
        settings.database_backend = "postgresql"
        svc = ClaudeEnrichmentService(settings)
        assert svc._is_enabled() is True

    @pytest.mark.asyncio
    async def test_enrich_noop_when_disabled(self):
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        settings = MagicMock()
        settings.claude_enrichment_enabled = False
        settings.database_backend = "sqlite"
        svc = ClaudeEnrichmentService(settings)
        result = await svc.enrich_document("doc1", "text", [])
        assert result == {}

    def test_synopsis_result_dataclass(self):
        from ai_ready_rag.services.enrichment_service import SynopsisResult

        r = SynopsisResult(
            synopsis_text="test",
            model_id="model",
            token_cost=100,
            cost_usd=0.001,
            raw_json={},
        )
        assert r.synopsis_text == "test"

    def test_entity_result_dataclass(self):
        from ai_ready_rag.services.enrichment_service import EntityResult

        e = EntityResult(
            entity_type="carrier",
            value="State Farm",
            canonical_value=None,
            confidence=0.9,
            source_chunk_index=0,
        )
        assert e.entity_type == "carrier"

    def test_get_client_raises_when_anthropic_not_installed(self):
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        settings = MagicMock()
        settings.claude_api_key = "sk-ant-test"
        svc = ClaudeEnrichmentService(settings)

        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(RuntimeError, match="anthropic package not installed"):
                svc._get_client()

    @pytest.mark.asyncio
    async def test_enrich_noop_returns_empty_dict_on_sqlite_even_with_key(self):
        """SQLite backend always returns empty dict regardless of other settings."""
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        settings = MagicMock()
        settings.claude_enrichment_enabled = True
        settings.claude_api_key = "sk-ant-real-key"
        settings.database_backend = "sqlite"
        svc = ClaudeEnrichmentService(settings)
        result = await svc.enrich_document("doc-id", "some text", [{"text": "chunk"}])
        assert result == {}

    @pytest.mark.asyncio
    async def test_enrich_document_calls_synopsis_and_entities_when_enabled(self):
        """Full pipeline is called when enabled (mocked API)."""
        from ai_ready_rag.services.enrichment_service import (
            ClaudeEnrichmentService,
            EntityResult,
            SynopsisResult,
        )

        settings = MagicMock()
        settings.claude_enrichment_enabled = True
        settings.claude_api_key = "sk-ant-test"
        settings.database_backend = "postgresql"

        svc = ClaudeEnrichmentService(settings)

        mock_synopsis = SynopsisResult(
            synopsis_text='{"document_type": "policy"}',
            model_id="claude-sonnet-4-6",
            token_cost=500,
            cost_usd=0.005,
            raw_json={"document_type": "policy"},
        )
        mock_entities = [
            EntityResult(
                entity_type="insurance_carrier",
                value="Acme Insurance",
                canonical_value=None,
                confidence=0.95,
                source_chunk_index=0,
            )
        ]

        svc._call_synopsis = AsyncMock(return_value=mock_synopsis)
        svc._call_entity_extraction = AsyncMock(return_value=mock_entities)

        result = await svc.enrich_document("doc-123", "document text", [])

        assert result["document_id"] == "doc-123"
        assert result["enrichment_model"] == "claude-sonnet-4-6"
        assert result["token_cost"] == 500
        assert result["cost_usd"] == 0.005
        assert result["synopsis"] is mock_synopsis
        assert result["entities"] is mock_entities

        svc._call_synopsis.assert_called_once_with("doc-123", "document text")
        svc._call_entity_extraction.assert_called_once_with("doc-123", mock_synopsis, [])

    @pytest.mark.asyncio
    async def test_enrich_document_persists_when_db_provided(self):
        """Persist is called when db_session is provided."""
        from ai_ready_rag.services.enrichment_service import (
            ClaudeEnrichmentService,
            EntityResult,
            SynopsisResult,
        )

        settings = MagicMock()
        settings.claude_enrichment_enabled = True
        settings.claude_api_key = "sk-ant-test"
        settings.database_backend = "postgresql"

        mock_db = MagicMock()
        svc = ClaudeEnrichmentService(settings, db_session=mock_db)

        mock_synopsis = SynopsisResult(
            synopsis_text="{}",
            model_id="claude-sonnet-4-6",
            token_cost=100,
            cost_usd=0.001,
            raw_json={},
        )
        mock_entities: list[EntityResult] = []

        svc._call_synopsis = AsyncMock(return_value=mock_synopsis)
        svc._call_entity_extraction = AsyncMock(return_value=mock_entities)
        svc._persist = AsyncMock()

        await svc.enrich_document("doc-456", "text", [])

        svc._persist.assert_called_once_with("doc-456", mock_synopsis, mock_entities)

    @pytest.mark.asyncio
    async def test_enrich_document_propagates_exceptions(self):
        """Exceptions from synopsis call bubble up to caller."""
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        settings = MagicMock()
        settings.claude_enrichment_enabled = True
        settings.claude_api_key = "sk-ant-test"
        settings.database_backend = "postgresql"

        svc = ClaudeEnrichmentService(settings)
        svc._call_synopsis = AsyncMock(side_effect=RuntimeError("API timeout"))

        with pytest.raises(RuntimeError, match="API timeout"):
            await svc.enrich_document("doc-789", "text", [])

    def test_entity_result_fields(self):
        from ai_ready_rag.services.enrichment_service import EntityResult

        e = EntityResult(
            entity_type="policy_number",
            value="POL-12345",
            canonical_value="POL-12345",
            confidence=1.0,
            source_chunk_index=3,
        )
        assert e.value == "POL-12345"
        assert e.canonical_value == "POL-12345"
        assert e.confidence == 1.0
        assert e.source_chunk_index == 3

    def test_synopsis_result_fields(self):
        from ai_ready_rag.services.enrichment_service import SynopsisResult

        r = SynopsisResult(
            synopsis_text='{"summary": "test doc"}',
            model_id="claude-sonnet-4-6",
            token_cost=1234,
            cost_usd=0.012,
            raw_json={"summary": "test doc"},
        )
        assert r.model_id == "claude-sonnet-4-6"
        assert r.token_cost == 1234
        assert r.raw_json == {"summary": "test doc"}


class TestPromptResolverIntegration:
    """Tests that ClaudeEnrichmentService correctly uses PromptResolver."""

    def test_loads_synopsis_prompt_from_core_dir(self, tmp_path, monkeypatch):
        """Service picks up enrichment_synopsis.txt from core prompts directory."""
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        monkeypatch.chdir(tmp_path)
        core_dir = tmp_path / "ai_ready_rag" / "prompts"
        core_dir.mkdir(parents=True)
        (core_dir / "enrichment_synopsis.txt").write_text("Custom synopsis prompt")
        (core_dir / "enrichment_entities.txt").write_text("Custom entity prompt")

        settings = MagicMock()
        settings.claude_enrichment_enabled = True
        settings.claude_api_key = "sk-ant-test"
        settings.database_backend = "postgresql"

        svc = ClaudeEnrichmentService(settings)
        assert svc._synopsis_prompt == "Custom synopsis prompt"
        assert svc._entity_prompt == "Custom entity prompt"

    def test_falls_back_to_constant_when_no_prompt_file(self, tmp_path, monkeypatch):
        """Service uses module-level constant when no prompt file exists."""
        from ai_ready_rag.services.enrichment_service import (
            ENTITY_EXTRACTION_PROMPT,
            SYNOPSIS_SYSTEM_PROMPT,
            ClaudeEnrichmentService,
        )

        monkeypatch.chdir(tmp_path)

        settings = MagicMock()
        settings.claude_enrichment_enabled = False
        settings.database_backend = "sqlite"

        svc = ClaudeEnrichmentService(settings)
        assert svc._synopsis_prompt == SYNOPSIS_SYSTEM_PROMPT
        assert svc._entity_prompt == ENTITY_EXTRACTION_PROMPT

    def test_tenant_override_takes_precedence(self, tmp_path, monkeypatch):
        """Tenant-level prompt overrides the core prompt for enrichment_synopsis."""
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        monkeypatch.chdir(tmp_path)

        # Core prompt
        core_dir = tmp_path / "ai_ready_rag" / "prompts"
        core_dir.mkdir(parents=True)
        (core_dir / "enrichment_synopsis.txt").write_text("Core synopsis prompt")
        (core_dir / "enrichment_entities.txt").write_text("Core entity prompt")

        # Tenant override
        tenant_dir = tmp_path / "tenant-instances" / "default" / "prompts"
        tenant_dir.mkdir(parents=True)
        (tenant_dir / "enrichment_synopsis.txt").write_text("Tenant synopsis override")

        settings = MagicMock()
        settings.claude_enrichment_enabled = False
        settings.database_backend = "sqlite"

        svc = ClaudeEnrichmentService(settings, tenant_id="default")
        # Tenant override takes precedence for synopsis
        assert svc._synopsis_prompt == "Tenant synopsis override"
        # No tenant override for entities — falls through to core
        assert svc._entity_prompt == "Core entity prompt"

    def test_list_available_returns_enrichment_prompts(self, tmp_path, monkeypatch):
        """PromptResolver.list_available() includes both enrichment prompt names."""
        from ai_ready_rag.tenant.resolver import PromptResolver

        monkeypatch.chdir(tmp_path)
        core_dir = tmp_path / "ai_ready_rag" / "prompts"
        core_dir.mkdir(parents=True)
        (core_dir / "enrichment_synopsis.txt").write_text("synopsis")
        (core_dir / "enrichment_entities.txt").write_text("entities")

        resolver = PromptResolver(tenant_id="default", module_id="core")
        available = resolver.list_available()
        assert "enrichment_synopsis" in available
        assert "enrichment_entities" in available
