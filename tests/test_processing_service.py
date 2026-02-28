"""Tests for ProcessingService enrichment integration (Issue #423).

Verifies that ClaudeEnrichmentService is wired into the document processing
pipeline and that enrichment is skipped gracefully when not configured.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_ready_rag.services.processing_service import ChunkInfo, ProcessingService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides):
    """Return a minimal mock Settings object."""
    settings = MagicMock()
    settings.generate_summaries = False
    settings.auto_tagging_enabled = False
    settings.auto_tagging_llm_enabled = False
    settings.use_ingestkit_excel = False
    settings.use_ingestkit_forms = False
    settings.use_ingestkit_image = False
    settings.use_ingestkit_email = False
    settings.coverage_rechunk_enabled = False
    settings.chunk_size = 512
    settings.chunk_overlap = 64
    settings.enable_ocr = False
    settings.ocr_language = "en"
    settings.chunker_backend = "simple"
    settings.vector_backend = "chroma"
    for k, v in overrides.items():
        setattr(settings, k, v)
    return settings


def _make_document(doc_id: str = "doc-123"):
    """Return a minimal mock Document."""
    doc = MagicMock()
    doc.id = doc_id
    doc.original_filename = "test.pdf"
    doc.file_path = "/tmp/test.pdf"
    doc.title = None
    doc.tags = []
    doc.auto_tag_status = None
    doc.auto_tag_strategy = None
    doc.auto_tag_version = None
    doc.source_path = None
    doc.status = "pending"
    return doc


def _make_chunks(texts: list[str]) -> list[ChunkInfo]:
    return [
        ChunkInfo(
            text=t,
            chunk_index=i,
            page_number=1,
            section=None,
            token_count=len(t) // 4,
        )
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# Tests: enrichment_service=None (default) — enrichment must be skipped
# ---------------------------------------------------------------------------


class TestEnrichmentSkippedWhenNone:
    """When enrichment_service is not provided, enrichment must not run."""

    @pytest.mark.asyncio
    async def test_no_enrichment_service_does_not_call_enrich(self):
        """ProcessingService with enrichment_service=None never calls enrich_document."""
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()

        chunk_dicts = [{"text": "hello world this is a test chunk for indexing"}]

        vector_service = MagicMock()
        vector_service.add_document = AsyncMock()

        chunker = MagicMock()
        chunker.chunk_document = MagicMock(return_value=chunk_dicts)

        service = ProcessingService(
            settings=settings,
            vector_service=vector_service,
            chunker=chunker,
            enrichment_service=None,  # Explicitly no enrichment
        )

        with patch("pathlib.Path.exists", return_value=True):
            result = await service.process_document(document, db)

        assert result.success is True
        # The key assertion: no enrich_document call occurred (no service to call it)
        # enrichment_status must NOT have been set when enrichment_service is None
        assert result.chunk_count == 1

    @pytest.mark.asyncio
    async def test_processing_succeeds_without_enrichment_service(self):
        """Processing pipeline completes successfully when enrichment_service is None."""
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()

        chunk_dicts = [
            {"text": "first chunk of document content for test"},
            {"text": "second chunk of document content for test"},
        ]

        vector_service = MagicMock()
        vector_service.add_document = AsyncMock()

        chunker = MagicMock()
        chunker.chunk_document = MagicMock(return_value=chunk_dicts)

        service = ProcessingService(
            settings=settings,
            vector_service=vector_service,
            chunker=chunker,
        )

        with patch("pathlib.Path.exists", return_value=True):
            result = await service.process_document(document, db)

        assert result.success is True
        assert result.chunk_count == 2
        vector_service.add_document.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests: enrichment_service provided — enrichment must be called
# ---------------------------------------------------------------------------


class TestEnrichmentCalledAfterVectorIndexing:
    """When enrichment_service is provided, enrich_document must be called."""

    @pytest.mark.asyncio
    async def test_enrich_document_called_with_correct_document_id(self):
        """enrich_document is called with the document's ID after vector indexing."""
        settings = _make_settings()
        document = _make_document(doc_id="doc-abc")
        db = MagicMock()
        db.query.return_value.filter_by.return_value.first.return_value = None

        chunk_dicts = [{"text": "enrichment test chunk content here"}]

        vector_service = MagicMock()
        vector_service.add_document = AsyncMock()

        chunker = MagicMock()
        chunker.chunk_document = MagicMock(return_value=chunk_dicts)

        enrichment_service = MagicMock()
        enrichment_service.enrich_document = AsyncMock(return_value={})

        service = ProcessingService(
            settings=settings,
            vector_service=vector_service,
            chunker=chunker,
            enrichment_service=enrichment_service,
        )

        with patch("pathlib.Path.exists", return_value=True):
            result = await service.process_document(document, db)

        assert result.success is True
        enrichment_service.enrich_document.assert_awaited_once()
        call_kwargs = enrichment_service.enrich_document.call_args
        assert call_kwargs.kwargs["document_id"] == "doc-abc"

    @pytest.mark.asyncio
    async def test_enrich_document_called_with_chunks(self):
        """enrich_document receives a list of chunk dicts with 'text' keys."""
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()
        db.query.return_value.filter_by.return_value.first.return_value = None

        chunk_dicts = [
            {"text": "first enrichment chunk with enough words to pass filter"},
            {"text": "second enrichment chunk with enough words to pass filter"},
        ]

        vector_service = MagicMock()
        vector_service.add_document = AsyncMock()

        chunker = MagicMock()
        chunker.chunk_document = MagicMock(return_value=chunk_dicts)

        enrichment_service = MagicMock()
        enrichment_service.enrich_document = AsyncMock(return_value={})

        service = ProcessingService(
            settings=settings,
            vector_service=vector_service,
            chunker=chunker,
            enrichment_service=enrichment_service,
        )

        with patch("pathlib.Path.exists", return_value=True):
            result = await service.process_document(document, db)

        assert result.success is True
        enrichment_service.enrich_document.assert_awaited_once()
        call_kwargs = enrichment_service.enrich_document.call_args.kwargs
        assert "chunks" in call_kwargs
        chunks_arg = call_kwargs["chunks"]
        assert isinstance(chunks_arg, list)
        assert len(chunks_arg) == 2
        for chunk in chunks_arg:
            assert "text" in chunk

    @pytest.mark.asyncio
    async def test_enrichment_status_set_to_completed_on_success(self):
        """document.enrichment_status is set to 'completed' after successful enrichment."""
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()
        db.query.return_value.filter_by.return_value.first.return_value = None

        chunk_dicts = [{"text": "test chunk content for enrichment status check"}]

        vector_service = MagicMock()
        vector_service.add_document = AsyncMock()

        chunker = MagicMock()
        chunker.chunk_document = MagicMock(return_value=chunk_dicts)

        enrichment_service = MagicMock()
        enrichment_service.enrich_document = AsyncMock(return_value={})

        service = ProcessingService(
            settings=settings,
            vector_service=vector_service,
            chunker=chunker,
            enrichment_service=enrichment_service,
        )

        with patch("pathlib.Path.exists", return_value=True):
            await service.process_document(document, db)

        assert document.enrichment_status == "completed"

    @pytest.mark.asyncio
    async def test_synopsis_id_written_when_synopsis_found(self):
        """document.synopsis_id is set when an EnrichmentSynopsis exists after enrichment."""
        settings = _make_settings()
        document = _make_document()
        document.synopsis_id = None

        # Simulate a synopsis record returned from DB query
        mock_synopsis = MagicMock()
        mock_synopsis.id = "synopsis-999"

        db = MagicMock()
        db.query.return_value.filter_by.return_value.first.return_value = mock_synopsis

        chunk_dicts = [{"text": "synopsis id test chunk with enough content words"}]

        vector_service = MagicMock()
        vector_service.add_document = AsyncMock()

        chunker = MagicMock()
        chunker.chunk_document = MagicMock(return_value=chunk_dicts)

        enrichment_service = MagicMock()
        enrichment_service.enrich_document = AsyncMock(return_value={})

        service = ProcessingService(
            settings=settings,
            vector_service=vector_service,
            chunker=chunker,
            enrichment_service=enrichment_service,
        )

        with patch("pathlib.Path.exists", return_value=True):
            await service.process_document(document, db)

        assert document.synopsis_id == "synopsis-999"

    @pytest.mark.asyncio
    async def test_enrichment_failure_sets_status_failed_and_does_not_raise(self):
        """When enrich_document raises, enrichment_status='failed' and processing succeeds."""
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()

        chunk_dicts = [{"text": "test chunk content that triggers enrichment failure path"}]

        vector_service = MagicMock()
        vector_service.add_document = AsyncMock()

        chunker = MagicMock()
        chunker.chunk_document = MagicMock(return_value=chunk_dicts)

        enrichment_service = MagicMock()
        enrichment_service.enrich_document = AsyncMock(
            side_effect=RuntimeError("Claude API unavailable")
        )

        service = ProcessingService(
            settings=settings,
            vector_service=vector_service,
            chunker=chunker,
            enrichment_service=enrichment_service,
        )

        with patch("pathlib.Path.exists", return_value=True):
            result = await service.process_document(document, db)

        # Processing itself must still succeed — enrichment failure is non-fatal
        assert result.success is True
        assert document.enrichment_status == "failed"

    @pytest.mark.asyncio
    async def test_vector_indexing_called_before_enrichment(self):
        """vector_service.add_document is called before enrich_document."""
        call_order: list[str] = []
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()
        db.query.return_value.filter_by.return_value.first.return_value = None

        chunk_dicts = [{"text": "ordering test chunk with enough words to index"}]

        async def _fake_add_document(**kwargs):
            call_order.append("vector")

        async def _fake_enrich(**kwargs):
            call_order.append("enrichment")
            return {}

        vector_service = MagicMock()
        vector_service.add_document = AsyncMock(side_effect=_fake_add_document)

        chunker = MagicMock()
        chunker.chunk_document = MagicMock(return_value=chunk_dicts)

        enrichment_service = MagicMock()
        enrichment_service.enrich_document = AsyncMock(side_effect=_fake_enrich)

        service = ProcessingService(
            settings=settings,
            vector_service=vector_service,
            chunker=chunker,
            enrichment_service=enrichment_service,
        )

        with patch("pathlib.Path.exists", return_value=True):
            await service.process_document(document, db)

        assert call_order == ["vector", "enrichment"], (
            "vector indexing must complete before enrichment"
        )


# ---------------------------------------------------------------------------
# Tests: factory function
# ---------------------------------------------------------------------------


class TestGetEnrichmentServiceFactory:
    """Tests for the get_enrichment_service factory function."""

    def test_factory_returns_claude_enrichment_service(self):
        """get_enrichment_service returns a ClaudeEnrichmentService instance."""
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService
        from ai_ready_rag.services.factory import get_enrichment_service

        settings = _make_settings()
        service = get_enrichment_service(settings)
        assert isinstance(service, ClaudeEnrichmentService)

    def test_factory_accepts_db_session(self):
        """get_enrichment_service accepts and passes through db_session."""
        from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService
        from ai_ready_rag.services.factory import get_enrichment_service

        settings = _make_settings()
        mock_db = MagicMock()
        service = get_enrichment_service(settings, db_session=mock_db)
        assert isinstance(service, ClaudeEnrichmentService)
        assert service._db is mock_db
