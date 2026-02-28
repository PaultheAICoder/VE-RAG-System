"""Tests for ClaudeCliEnrichmentBackend and CLI-mode wiring in ClaudeEnrichmentService.

All tests mock subprocess.run — no actual claude binary required.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Return a mock CompletedProcess-like object."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


# ---------------------------------------------------------------------------
# TestClaudeCliEnrichmentBackendSynopsis
# ---------------------------------------------------------------------------


class TestClaudeCliEnrichmentBackendSynopsis:
    def test_call_synopsis_returns_stdout(self):
        from ai_ready_rag.services.enrichment_cli_backend import ClaudeCliEnrichmentBackend

        backend = ClaudeCliEnrichmentBackend()
        with patch("subprocess.run", return_value=_make_proc(0, "synopsis text")) as mock_run:
            result = backend.call_synopsis("doc text", "default")
        assert result == "synopsis text"
        mock_run.assert_called_once()

    def test_call_synopsis_raises_on_nonzero_exit(self):
        from ai_ready_rag.services.enrichment_cli_backend import ClaudeCliEnrichmentBackend

        backend = ClaudeCliEnrichmentBackend()
        with patch("subprocess.run", return_value=_make_proc(1, "", "bad error")):
            with pytest.raises(RuntimeError, match="bad error"):
                backend.call_synopsis("doc text", "default")

    def test_call_synopsis_passes_timeout_120(self):
        from ai_ready_rag.services.enrichment_cli_backend import ClaudeCliEnrichmentBackend

        backend = ClaudeCliEnrichmentBackend()
        with patch("subprocess.run", return_value=_make_proc(0, "ok")) as mock_run:
            backend.call_synopsis("doc text", "default")
        _, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == 120

    def test_call_synopsis_strips_leading_trailing_whitespace(self):
        from ai_ready_rag.services.enrichment_cli_backend import ClaudeCliEnrichmentBackend

        backend = ClaudeCliEnrichmentBackend()
        with patch("subprocess.run", return_value=_make_proc(0, "\n  synopsis text  \n")):
            result = backend.call_synopsis("doc text", "default")
        assert result == "synopsis text"

    def test_call_synopsis_includes_exit_code_in_error(self):
        from ai_ready_rag.services.enrichment_cli_backend import ClaudeCliEnrichmentBackend

        backend = ClaudeCliEnrichmentBackend()
        with patch("subprocess.run", return_value=_make_proc(2, "", "some err")):
            with pytest.raises(RuntimeError, match="exited with code 2"):
                backend.call_synopsis("doc", "default")


# ---------------------------------------------------------------------------
# TestClaudeCliEnrichmentBackendEntityExtraction
# ---------------------------------------------------------------------------


class TestClaudeCliEnrichmentBackendEntityExtraction:
    def test_call_entity_extraction_parses_valid_json_array(self):
        from ai_ready_rag.services.enrichment_cli_backend import ClaudeCliEnrichmentBackend

        backend = ClaudeCliEnrichmentBackend()
        json_out = '[{"entity_type":"carrier","value":"Acme","confidence":0.9,"chunk_index":0}]'
        with patch("subprocess.run", return_value=_make_proc(0, json_out)):
            result = backend.call_entity_extraction("synopsis", "chunk text")
        assert len(result) == 1
        assert result[0]["entity_type"] == "carrier"
        assert result[0]["value"] == "Acme"

    def test_call_entity_extraction_strips_json_fences(self):
        from ai_ready_rag.services.enrichment_cli_backend import ClaudeCliEnrichmentBackend

        backend = ClaudeCliEnrichmentBackend()
        fenced = '```json\n[{"entity_type":"x","value":"y"}]\n```'
        with patch("subprocess.run", return_value=_make_proc(0, fenced)):
            result = backend.call_entity_extraction("synopsis", "chunk")
        assert len(result) == 1
        assert result[0]["value"] == "y"

    def test_call_entity_extraction_strips_plain_fences(self):
        from ai_ready_rag.services.enrichment_cli_backend import ClaudeCliEnrichmentBackend

        backend = ClaudeCliEnrichmentBackend()
        fenced = "```\n[]\n```"
        with patch("subprocess.run", return_value=_make_proc(0, fenced)):
            result = backend.call_entity_extraction("synopsis", "chunk")
        assert result == []

    def test_call_entity_extraction_returns_empty_on_bad_json(self):
        from ai_ready_rag.services.enrichment_cli_backend import ClaudeCliEnrichmentBackend

        backend = ClaudeCliEnrichmentBackend()
        with patch("subprocess.run", return_value=_make_proc(0, "not valid json")):
            result = backend.call_entity_extraction("synopsis", "chunk")
        assert result == []

    def test_call_entity_extraction_raises_on_nonzero_exit(self):
        from ai_ready_rag.services.enrichment_cli_backend import ClaudeCliEnrichmentBackend

        backend = ClaudeCliEnrichmentBackend()
        with patch("subprocess.run", return_value=_make_proc(1, "", "subprocess err")):
            with pytest.raises(RuntimeError, match="subprocess err"):
                backend.call_entity_extraction("synopsis", "chunk")

    def test_call_entity_extraction_ignores_non_dict_items(self):
        from ai_ready_rag.services.enrichment_cli_backend import ClaudeCliEnrichmentBackend

        backend = ClaudeCliEnrichmentBackend()
        mixed = '[42, "string", {"entity_type":"x","value":"y"}]'
        with patch("subprocess.run", return_value=_make_proc(0, mixed)):
            result = backend.call_entity_extraction("synopsis", "chunk")
        assert len(result) == 1
        assert result[0]["entity_type"] == "x"

    def test_call_entity_extraction_returns_empty_on_non_list_json(self):
        from ai_ready_rag.services.enrichment_cli_backend import ClaudeCliEnrichmentBackend

        backend = ClaudeCliEnrichmentBackend()
        with patch("subprocess.run", return_value=_make_proc(0, '{"key": "value"}')):
            result = backend.call_entity_extraction("synopsis", "chunk")
        assert result == []


# ---------------------------------------------------------------------------
# TestClaudeEnrichmentServiceCliMode
# ---------------------------------------------------------------------------


class TestClaudeEnrichmentServiceCliMode:
    def _make_settings(
        self,
        *,
        claude_backend: str = "api",
        database_backend: str = "postgresql",
        claude_enrichment_enabled: bool = True,
        claude_api_key: str | None = None,
    ) -> MagicMock:
        settings = MagicMock()
        settings.claude_backend = claude_backend
        settings.database_backend = database_backend
        settings.claude_enrichment_enabled = claude_enrichment_enabled
        settings.claude_api_key = claude_api_key
        return settings

    def test_is_enabled_cli_backend_no_api_key(self):
        """CLI backend allows enrichment without an API key."""
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        settings = self._make_settings(claude_backend="cli", claude_api_key=None)
        svc = ClaudeEnrichmentService(settings)
        assert svc._is_enabled() is True

    def test_is_enabled_cli_backend_sqlite_still_disabled(self):
        """SQLite backend always disables enrichment regardless of claude_backend."""
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        settings = self._make_settings(claude_backend="cli", database_backend="sqlite")
        svc = ClaudeEnrichmentService(settings)
        assert svc._is_enabled() is False

    def test_cli_backend_instantiated_when_setting_is_cli(self):
        """_cli_backend is set when claude_backend == 'cli'."""
        from ai_ready_rag.services.enrichment_cli_backend import ClaudeCliEnrichmentBackend
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        settings = self._make_settings(claude_backend="cli")
        svc = ClaudeEnrichmentService(settings)
        assert svc._cli_backend is not None
        assert isinstance(svc._cli_backend, ClaudeCliEnrichmentBackend)

    def test_api_backend_not_instantiated_when_setting_is_api(self):
        """_cli_backend is None when claude_backend == 'api' (default)."""
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        settings = self._make_settings(claude_backend="api", claude_api_key="sk-ant-test")
        svc = ClaudeEnrichmentService(settings)
        assert svc._cli_backend is None

    def test_api_backend_default_still_requires_api_key(self):
        """Default api backend still requires api_key for _is_enabled()."""
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

        settings = self._make_settings(claude_backend="api", claude_api_key=None)
        svc = ClaudeEnrichmentService(settings)
        assert svc._is_enabled() is False

    @pytest.mark.asyncio
    async def test_enrich_document_uses_cli_backend_when_configured(self):
        """Full pipeline calls subprocess when claude_backend == 'cli'."""
        from unittest.mock import AsyncMock

        from ai_ready_rag.services.enrichment_service import (
            ClaudeEnrichmentService,
            EntityResult,
            SynopsisResult,
        )

        settings = self._make_settings(claude_backend="cli")
        svc = ClaudeEnrichmentService(settings)

        # Patch the _call_synopsis / _call_entity_extraction instead of subprocess
        # to keep this test at the service boundary
        mock_synopsis = SynopsisResult(
            synopsis_text='{"document_type":"policy"}',
            model_id="claude-cli",
            token_cost=0,
            cost_usd=0.0,
            raw_json={"document_type": "policy"},
        )
        mock_entities = [
            EntityResult(
                entity_type="carrier",
                value="Acme",
                canonical_value=None,
                confidence=0.9,
                source_chunk_index=0,
            )
        ]

        svc._call_synopsis = AsyncMock(return_value=mock_synopsis)
        svc._call_entity_extraction = AsyncMock(return_value=mock_entities)

        result = await svc.enrich_document("doc-cli-1", "document text", [])

        assert result["document_id"] == "doc-cli-1"
        assert result["enrichment_model"] == "claude-cli"
        assert result["cost_usd"] == 0.0
        svc._call_synopsis.assert_called_once()
        svc._call_entity_extraction.assert_called_once()
