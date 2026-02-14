"""Integration tests for Excel processing via ingestkit-excel pipeline.

Tests the ExcelProcessingService orchestration, error mapping, fallback
to SimpleChunker, and document lifecycle (cleanup on delete).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from ai_ready_rag.config import Settings
from ai_ready_rag.services.excel_processing_service import ExcelProcessingService
from ai_ready_rag.services.processing_service import ProcessingService


def _make_settings(**overrides) -> Settings:
    """Create test settings with ingestkit enabled."""
    defaults = {
        "use_ingestkit_excel": True,
        "qdrant_url": "http://localhost:6333",
        "qdrant_collection": "documents",
        "ollama_base_url": "http://localhost:11434",
        "embedding_model": "nomic-embed-text",
        "embedding_dimension": 768,
        "excel_tables_db_path": ":memory:",
        "default_tenant_id": "default",
        "env_profile": "laptop",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_document(doc_id="doc-123", file_path="/tmp/test.xlsx", tags=None):
    """Create a mock Document with required attributes."""
    doc = MagicMock()
    doc.id = doc_id
    doc.file_path = file_path
    doc.original_filename = "test.xlsx"
    doc.uploaded_by = "user-456"
    doc.status = "pending"
    doc.error_message = None
    doc.chunk_count = None
    doc.page_count = None
    doc.word_count = None
    doc.processing_time_ms = None
    doc.title = None
    doc.excel_file_type = None
    doc.excel_classification_tier = None
    doc.excel_ingest_key = None
    doc.excel_tables_created = None
    doc.excel_db_table_names = None

    if tags is None:
        tag = MagicMock()
        tag.name = "hr"
        doc.tags = [tag]
    else:
        doc.tags = tags

    return doc


def _make_ingestkit_result(
    *,
    chunks_created=5,
    tables_created=1,
    tables=None,
    errors=None,
    warnings=None,
    processing_time=1.5,
    file_type="tabular_data",
    tier_used="rule_based",
    confidence=0.95,
    ingest_key="abc123",
):
    """Create a mock ingestkit ProcessingResult."""
    result = MagicMock()
    result.chunks_created = chunks_created
    result.tables_created = tables_created
    result.tables = tables or ["sheet1_employees"]
    result.errors = errors or []
    result.warnings = warnings or []
    result.processing_time_seconds = processing_time
    result.ingest_key = ingest_key

    cls_result = MagicMock()
    cls_result.file_type.value = file_type
    cls_result.tier_used.value = tier_used
    cls_result.confidence = confidence
    result.classification_result = cls_result

    return result


def _patch_excel_service_deps():
    """Patch all ingestkit imports used inside ExcelProcessingService.process_excel().

    Since these are imported inside the method body via `from ... import ...`,
    we must patch at the source modules.
    """
    return [
        patch("ingestkit_excel.router.ExcelRouter"),
        patch(
            "ai_ready_rag.services.ingestkit_adapters.VERagVectorStoreAdapter",
            return_value=MagicMock(),
        ),
        patch(
            "ai_ready_rag.services.ingestkit_adapters.create_embedding_adapter",
            return_value=MagicMock(),
        ),
        patch(
            "ai_ready_rag.services.ingestkit_adapters.create_llm_adapter",
            return_value=MagicMock(),
        ),
        patch(
            "ai_ready_rag.services.ingestkit_adapters.create_structured_db",
            return_value=MagicMock(),
        ),
    ]


class TestExcelProcessingService:
    """Tests for ExcelProcessingService orchestration."""

    @pytest.mark.asyncio
    async def test_successful_processing(self):
        """Successful ingestkit processing returns ProcessingResult with success=True."""
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()

        ingestkit_result = _make_ingestkit_result()

        patches = _patch_excel_service_deps()
        with patches[0] as mock_router_cls, patches[1], patches[2], patches[3], patches[4]:
            mock_router = MagicMock()
            mock_router.process.return_value = ingestkit_result
            mock_router_cls.return_value = mock_router

            service = ExcelProcessingService(settings)
            result, should_fallback = await service.process_excel(document, db)

        assert should_fallback is False
        assert result is not None
        assert result.success is True
        assert result.chunk_count == 5

        # Document metadata should be updated
        assert document.excel_file_type == "tabular_data"
        assert document.excel_classification_tier == "rule_based"
        assert document.excel_ingest_key == "abc123"
        assert document.excel_tables_created == 1
        assert json.loads(document.excel_db_table_names) == ["sheet1_employees"]

    @pytest.mark.asyncio
    async def test_inconclusive_classification_triggers_fallback(self):
        """E_CLASSIFY_INCONCLUSIVE should trigger fallback to SimpleChunker."""
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()

        ingestkit_result = _make_ingestkit_result(
            errors=["E_CLASSIFY_INCONCLUSIVE"],
            chunks_created=0,
        )

        patches = _patch_excel_service_deps()
        with patches[0] as mock_router_cls, patches[1], patches[2], patches[3], patches[4]:
            mock_router = MagicMock()
            mock_router.process.return_value = ingestkit_result
            mock_router_cls.return_value = mock_router

            service = ExcelProcessingService(settings)
            result, should_fallback = await service.process_excel(document, db)

        assert should_fallback is True
        assert result is None

    @pytest.mark.asyncio
    async def test_hard_error_returns_failure(self):
        """Backend errors should return failed ProcessingResult."""
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()

        ingestkit_result = _make_ingestkit_result(
            errors=["E_PARSE_CORRUPT"],
            chunks_created=0,
        )

        patches = _patch_excel_service_deps()
        with patches[0] as mock_router_cls, patches[1], patches[2], patches[3], patches[4]:
            mock_router = MagicMock()
            mock_router.process.return_value = ingestkit_result
            mock_router_cls.return_value = mock_router

            service = ExcelProcessingService(settings)
            result, should_fallback = await service.process_excel(document, db)

        assert should_fallback is False
        assert result is not None
        assert result.success is False
        assert "E_PARSE_CORRUPT" in result.error_message

    @pytest.mark.asyncio
    async def test_retryable_error_marked_in_message(self):
        """Retryable backend errors should be flagged in error_message."""
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()

        ingestkit_result = _make_ingestkit_result(
            errors=["E_BACKEND_EMBED_TIMEOUT"],
            chunks_created=0,
        )

        patches = _patch_excel_service_deps()
        with patches[0] as mock_router_cls, patches[1], patches[2], patches[3], patches[4]:
            mock_router = MagicMock()
            mock_router.process.return_value = ingestkit_result
            mock_router_cls.return_value = mock_router

            service = ExcelProcessingService(settings)
            result, should_fallback = await service.process_excel(document, db)

        assert result.success is False
        assert "(retryable)" in result.error_message

    @pytest.mark.asyncio
    async def test_exception_triggers_fallback(self):
        """Unhandled exceptions should trigger fallback to SimpleChunker."""
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()

        patches = _patch_excel_service_deps()
        with patches[0] as mock_router_cls, patches[1], patches[2], patches[3], patches[4]:
            mock_router = MagicMock()
            mock_router.process.side_effect = RuntimeError("Ollama down")
            mock_router_cls.return_value = mock_router

            service = ExcelProcessingService(settings)
            result, should_fallback = await service.process_excel(document, db)

        assert should_fallback is True
        assert result is None

    @pytest.mark.asyncio
    async def test_warnings_only_still_succeeds(self):
        """Warnings without errors should still result in success."""
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()

        ingestkit_result = _make_ingestkit_result(
            warnings=["W_SHEET_SKIPPED_CHART", "W_ROWS_TRUNCATED"],
        )

        patches = _patch_excel_service_deps()
        with patches[0] as mock_router_cls, patches[1], patches[2], patches[3], patches[4]:
            mock_router = MagicMock()
            mock_router.process.return_value = ingestkit_result
            mock_router_cls.return_value = mock_router

            service = ExcelProcessingService(settings)
            result, should_fallback = await service.process_excel(document, db)

        assert should_fallback is False
        assert result.success is True


class TestProcessingServiceRouting:
    """Tests for Excel routing in ProcessingService."""

    def test_should_use_ingestkit_disabled(self):
        """Feature flag off should return False."""
        settings = _make_settings(use_ingestkit_excel=False)
        service = ProcessingService(settings)
        assert service._should_use_ingestkit() is False

    def test_should_use_ingestkit_enabled_and_importable(self):
        """Feature flag on + package available should return True."""
        settings = _make_settings(use_ingestkit_excel=True)
        service = ProcessingService(settings)

        with patch.dict("sys.modules", {"ingestkit_excel": MagicMock()}):
            assert service._should_use_ingestkit() is True

    def test_should_use_ingestkit_enabled_but_not_importable(self):
        """Feature flag on but package missing should return False."""
        settings = _make_settings(use_ingestkit_excel=True)
        service = ProcessingService(settings)

        with patch("builtins.__import__", side_effect=ImportError):
            assert service._should_use_ingestkit() is False


class TestDocumentLifecycle:
    """Tests for Excel table cleanup on document delete."""

    def test_cleanup_excel_tables_on_delete(self):
        """Deleting a document with excel_db_table_names should drop those tables."""
        from ai_ready_rag.services.document_service import DocumentService

        settings = _make_settings()

        # Create mock document with Excel table names
        document = MagicMock()
        document.id = "doc-123"
        document.excel_db_table_names = json.dumps(["employees", "departments"])

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = document

        with (
            patch.object(DocumentService, "_cleanup_excel_tables") as mock_cleanup,
            patch("shutil.rmtree"),
        ):
            service = DocumentService(db, settings)
            service.storage_path = MagicMock()
            service.storage_path.__truediv__ = MagicMock(
                return_value=MagicMock(exists=MagicMock(return_value=False))
            )

            import asyncio

            asyncio.run(service.delete_document("doc-123"))

            mock_cleanup.assert_called_once_with(json.dumps(["employees", "departments"]))

    def test_cleanup_excel_tables_skips_when_no_tables(self):
        """Documents without excel_db_table_names should skip cleanup."""
        from ai_ready_rag.services.document_service import DocumentService

        settings = _make_settings()

        document = MagicMock()
        document.id = "doc-123"
        document.excel_db_table_names = None

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = document

        with (
            patch.object(DocumentService, "_cleanup_excel_tables") as mock_cleanup,
            patch("shutil.rmtree"),
        ):
            service = DocumentService(db, settings)
            service.storage_path = MagicMock()
            service.storage_path.__truediv__ = MagicMock(
                return_value=MagicMock(exists=MagicMock(return_value=False))
            )

            import asyncio

            asyncio.run(service.delete_document("doc-123"))

            mock_cleanup.assert_not_called()

    def test_cleanup_excel_tables_handles_invalid_json(self):
        """Invalid JSON in excel_db_table_names should not raise."""
        from ai_ready_rag.services.document_service import DocumentService

        settings = _make_settings()
        service = DocumentService(MagicMock(), settings)

        # Should not raise
        service._cleanup_excel_tables("not valid json")

    def test_cleanup_excel_tables_drops_tables(self):
        """Should call drop_table for each table name."""
        from ai_ready_rag.services.document_service import DocumentService

        settings = _make_settings(excel_tables_db_path=":memory:")
        service = DocumentService(MagicMock(), settings)

        mock_structured_db = MagicMock()
        with (
            patch("ai_ready_rag.services.document_service.Path") as mock_path,
            patch(
                "ai_ready_rag.services.ingestkit_adapters.create_structured_db",
                return_value=mock_structured_db,
            ),
        ):
            mock_path.return_value.exists.return_value = True

            service._cleanup_excel_tables(json.dumps(["employees", "departments"]))

            assert mock_structured_db.drop_table.call_count == 2
            mock_structured_db.drop_table.assert_any_call("employees")
            mock_structured_db.drop_table.assert_any_call("departments")
