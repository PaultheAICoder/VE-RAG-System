"""Integration tests for Forms processing via ingestkit-forms pipeline.

Tests the FormsProcessingService orchestration, error mapping, fallback
to SimpleChunker, routing logic, and compensation mechanisms.
"""

from unittest.mock import MagicMock, patch

import pytest

from ai_ready_rag.config import Settings


def _make_settings(**overrides) -> Settings:
    """Create test settings with forms enabled."""
    defaults = {
        "use_ingestkit_forms": True,
        "forms_match_confidence_threshold": 0.6,
        "forms_ocr_engine": "tesseract",
        "forms_vlm_enabled": False,
        "forms_vlm_model": "qwen2.5-vl:7b",
        "forms_redact_high_risk_fields": False,
        "forms_template_storage_path": "/tmp/forms/templates",
        "forms_db_path": ":memory:",
        "forms_template_require_approval": True,
        "qdrant_url": "http://localhost:6333",
        "qdrant_collection": "documents",
        "ollama_base_url": "http://localhost:11434",
        "embedding_model": "nomic-embed-text",
        "embedding_dimension": 768,
        "default_tenant_id": "default",
        "env_profile": "laptop",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_document(doc_id="doc-123", file_path="/tmp/test.pdf", tags=None):
    """Create a mock Document with forms-relevant attributes."""
    doc = MagicMock()
    doc.id = doc_id
    doc.file_path = file_path
    doc.original_filename = "test.pdf"
    doc.uploaded_by = "user-456"
    doc.status = "pending"
    doc.error_message = None
    doc.chunk_count = None
    doc.page_count = None
    doc.word_count = None
    doc.processing_time_ms = None
    doc.title = None

    # Forms-specific fields
    doc.forms_template_id = None
    doc.forms_template_name = None
    doc.forms_template_version = None
    doc.forms_overall_confidence = None
    doc.forms_extraction_method = None
    doc.forms_match_method = None
    doc.forms_ingest_key = None
    doc.forms_db_table_names = None

    if tags is None:
        tag = MagicMock()
        tag.name = "hr"
        doc.tags = [tag]
    else:
        doc.tags = tags

    return doc


def _make_match_result(template_id="tpl-w2", confidence=0.85):
    """Create a mock TemplateMatch."""
    match = MagicMock()
    match.template_id = template_id
    match.template_name = "W-2 Wage and Tax Statement"
    match.confidence = confidence
    return match


def _make_extraction_result(
    template_id="tpl-w2",
    extraction_method="ocr",
    overall_confidence=0.82,
):
    """Create a mock FormExtractionResult."""
    result = MagicMock()
    result.template_id = template_id
    result.template_name = "W-2 Wage and Tax Statement"
    result.template_version = 1
    result.overall_confidence = overall_confidence
    result.extraction_method = extraction_method
    result.match_method = "auto_detect"
    result.pages_processed = 1
    return result


def _make_form_result(
    *,
    chunks_created=3,
    tables=None,
    errors=None,
    warnings=None,
    processing_time=2.5,
    ingest_key="form-abc123",
):
    """Create a mock FormProcessingResult."""
    result = MagicMock()
    result.chunks_created = chunks_created
    result.tables = tables or ["form_w2_2024"]
    result.errors = errors or []
    result.error_details = []
    if errors:
        for err_code in errors:
            err = MagicMock()
            err.code.value = err_code
            result.error_details.append(err)
    result.warnings = warnings or []
    result.processing_time_seconds = processing_time
    result.ingest_key = ingest_key
    result.extraction_result = _make_extraction_result()
    result.written = MagicMock()
    result.written.vector_point_ids = [f"vec-{i}" for i in range(chunks_created)]
    return result


def _patch_forms_deps():
    """Patch all ingestkit-forms imports for FormsProcessingService tests."""
    return [
        patch("ingestkit_forms.create_default_router"),
        patch("ingestkit_forms.FileSystemTemplateStore"),
        patch("ingestkit_forms.config.FormProcessorConfig"),
        patch("ai_ready_rag.services.ingestkit_adapters.VERagVectorStoreAdapter"),
        patch("ai_ready_rag.services.ingestkit_adapters.create_embedding_adapter"),
        patch("ai_ready_rag.services.ingestkit_adapters.VERagFormDBAdapter"),
        patch("ai_ready_rag.services.ingestkit_adapters.VERagLayoutFingerprinter"),
        patch("ai_ready_rag.services.ingestkit_adapters.VERagOCRAdapter"),
        patch("ai_ready_rag.services.ingestkit_adapters.VERagVLMAdapter"),
    ]


class TestFormsProcessingService:
    """Tests for FormsProcessingService orchestration."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_forms_no_match_falls_back(self):
        """No matching template should trigger fallback to standard chunker."""
        settings = _make_settings(use_ingestkit_forms=True)
        document = _make_document()
        db = MagicMock()

        patches = _patch_forms_deps()
        with (
            patches[0] as mock_router_cls,
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
        ):
            mock_router = MagicMock()
            mock_router.try_match.return_value = None  # No match
            mock_router_cls.return_value = mock_router

            from ai_ready_rag.services.forms_processing_service import (
                FormsProcessingService,
            )

            service = FormsProcessingService(settings)
            result, should_fallback = await service.process_form(document, db)

        assert should_fallback is True
        assert result is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_forms_successful_extraction(self):
        """Successful extraction should update document fields and return success."""
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()

        match = _make_match_result(template_id="tpl-w2", confidence=0.85)
        form_result = _make_form_result(
            chunks_created=3,
            tables=["form_w2_2024"],
            ingest_key="form-abc123",
        )

        patches = _patch_forms_deps()
        with (
            patches[0] as mock_router_cls,
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
        ):
            mock_router = MagicMock()
            mock_router.try_match.return_value = match
            mock_router.extract_form.return_value = form_result
            mock_router_cls.return_value = mock_router

            from ai_ready_rag.services.forms_processing_service import (
                FormsProcessingService,
            )

            service = FormsProcessingService(settings)
            result, should_fallback = await service.process_form(document, db)

        assert should_fallback is False
        assert result is not None
        assert result.success is True
        assert result.chunk_count == 3

        # Verify document fields updated
        assert document.forms_template_id == "tpl-w2"
        assert document.forms_template_name == "W-2 Wage and Tax Statement"
        assert document.forms_template_version == 1
        assert document.forms_overall_confidence == 0.82
        assert document.forms_extraction_method == "ocr"
        assert document.forms_match_method == "auto_detect"
        assert document.forms_ingest_key == "form-abc123"

        import json

        assert json.loads(document.forms_db_table_names) == ["form_w2_2024"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_forms_fallback_error_code_triggers_fallback(self):
        """E_FORM_NO_MATCH error should trigger fallback to standard chunker."""
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()

        match = _make_match_result()
        form_result = _make_form_result(
            errors=["E_FORM_NO_MATCH"],
            chunks_created=2,  # Partial writes exist, need compensation
        )

        patches = _patch_forms_deps()
        with (
            patches[0] as mock_router_cls,
            patches[1],
            patches[2],
            patches[3] as mock_vector_adapter,
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
        ):
            mock_router = MagicMock()
            mock_router.try_match.return_value = match
            mock_router.extract_form.return_value = form_result
            mock_router_cls.return_value = mock_router

            mock_vector = MagicMock()
            mock_vector_adapter.return_value = mock_vector

            from ai_ready_rag.services.forms_processing_service import (
                FormsProcessingService,
            )

            service = FormsProcessingService(settings)
            result, should_fallback = await service.process_form(document, db)

        assert should_fallback is True
        assert result is None
        # Compensation should be called (vectors deleted)
        mock_vector.delete_by_filter.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_forms_hard_error_returns_failure(self):
        """Hard errors (non-fallback) should return failed ProcessingResult."""
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()

        match = _make_match_result()
        form_result = _make_form_result(
            errors=["E_FORM_EXTRACTION_FAILED"],
            chunks_created=0,
        )

        patches = _patch_forms_deps()
        with (
            patches[0] as mock_router_cls,
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
        ):
            mock_router = MagicMock()
            mock_router.try_match.return_value = match
            mock_router.extract_form.return_value = form_result
            mock_router_cls.return_value = mock_router

            from ai_ready_rag.services.forms_processing_service import (
                FormsProcessingService,
            )

            service = FormsProcessingService(settings)
            result, should_fallback = await service.process_form(document, db)

        assert should_fallback is False
        assert result is not None
        assert result.success is False
        assert "E_FORM_EXTRACTION_FAILED" in result.error_message

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_forms_partial_write_compensation(self):
        """Partial vector writes should be compensated on fallback."""
        settings = _make_settings()
        document = _make_document()
        db = MagicMock()

        match = _make_match_result()
        form_result = _make_form_result(
            errors=["E_FORM_UNSUPPORTED_FORMAT"],
            chunks_created=2,  # Partial write happened
        )

        patches = _patch_forms_deps()
        with (
            patches[0] as mock_router_cls,
            patches[1],
            patches[2],
            patches[3] as mock_vector_adapter,
            patches[4],
            patches[5] as mock_form_db_adapter,
            patches[6],
            patches[7],
            patches[8],
        ):
            mock_router = MagicMock()
            mock_router.try_match.return_value = match
            mock_router.extract_form.return_value = form_result
            mock_router_cls.return_value = mock_router

            mock_vector = MagicMock()
            mock_vector_adapter.return_value = mock_vector

            mock_form_db = MagicMock()
            mock_form_db_adapter.return_value = mock_form_db

            from ai_ready_rag.services.forms_processing_service import (
                FormsProcessingService,
            )

            service = FormsProcessingService(settings)
            result, should_fallback = await service.process_form(document, db)

        assert should_fallback is True
        assert result is None

        # Compensation should clean up vectors and tables
        mock_vector.delete_by_filter.assert_called_once_with(
            settings.qdrant_collection,
            "ingestkit_ingest_key",
            "form-abc123",
        )
        mock_form_db.execute_sql.assert_called()  # DROP TABLE called

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_forms_redaction_applied(self):
        """High-risk field redaction should be applied when flag enabled."""
        settings = _make_settings(forms_redact_high_risk_fields=True)
        document = _make_document()
        db = MagicMock()

        match = _make_match_result()
        form_result = _make_form_result()

        patches = _patch_forms_deps()
        with (
            patches[0] as mock_router_cls,
            patches[1],
            patches[2] as mock_config_cls,
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
        ):
            mock_router = MagicMock()
            mock_router.try_match.return_value = match
            mock_router.extract_form.return_value = form_result
            mock_router_cls.return_value = mock_router

            from ai_ready_rag.services.forms_processing_service import (
                FormsProcessingService,
            )

            service = FormsProcessingService(settings)
            await service.process_form(document, db)

            # Verify config was created with redact_patterns
            config_call = mock_config_cls.call_args
            assert "redact_patterns" in config_call.kwargs
            patterns = config_call.kwargs["redact_patterns"]
            assert len(patterns) > 0
            # Check for SSN pattern
            assert any(r"\d{3}-\d{2}-\d{4}" in p for p in patterns)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_forms_structured_log_events(self, caplog):
        """Forms processing should emit structured log events."""
        import logging

        caplog.set_level(logging.INFO)

        settings = _make_settings()
        document = _make_document()
        db = MagicMock()

        match = _make_match_result()
        form_result = _make_form_result()

        patches = _patch_forms_deps()
        with (
            patches[0] as mock_router_cls,
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
        ):
            mock_router = MagicMock()
            mock_router.try_match.return_value = match
            mock_router.extract_form.return_value = form_result
            mock_router_cls.return_value = mock_router

            from ai_ready_rag.services.forms_processing_service import (
                FormsProcessingService,
            )

            service = FormsProcessingService(settings)
            await service.process_form(document, db)

        # Check for structured log events
        log_messages = [record.message for record in caplog.records]
        assert any("forms.match.success" in msg for msg in log_messages)
        assert any("forms.extract.success" in msg for msg in log_messages)

    @pytest.mark.unit
    def test_forms_disabled_flag(self):
        """Feature flag off should return False from _should_use_ingestkit_forms."""
        settings = _make_settings(use_ingestkit_forms=False)

        from ai_ready_rag.services.processing_service import ProcessingService

        service = ProcessingService(settings)

        assert service._should_use_ingestkit_forms() is False

    @pytest.mark.unit
    def test_forms_package_missing(self):
        """Missing ingestkit-forms package should return False."""
        settings = _make_settings(use_ingestkit_forms=True)

        from ai_ready_rag.services.processing_service import ProcessingService

        service = ProcessingService(settings)

        with patch("builtins.__import__", side_effect=ImportError):
            assert service._should_use_ingestkit_forms() is False

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_forms_exception_triggers_fallback(self, caplog):
        """Unhandled exceptions should trigger fallback to standard chunker."""
        import logging

        caplog.set_level(logging.WARNING)

        settings = _make_settings()
        document = _make_document()
        db = MagicMock()

        patches = _patch_forms_deps()
        with (
            patches[0] as mock_router_cls,
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
        ):
            mock_router = MagicMock()
            mock_router.try_match.side_effect = RuntimeError("Fingerprint service down")
            mock_router_cls.return_value = mock_router

            from ai_ready_rag.services.forms_processing_service import (
                FormsProcessingService,
            )

            service = FormsProcessingService(settings)
            result, should_fallback = await service.process_form(document, db)

        assert should_fallback is True
        assert result is None

        # Check for warning log
        assert any("forms.match.error" in record.message for record in caplog.records)
