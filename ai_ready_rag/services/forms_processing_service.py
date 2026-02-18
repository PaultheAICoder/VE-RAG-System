"""Forms processing service using ingestkit-forms pipeline.

Orchestrates ingestkit's FormRouter with VE-RAG adapter backends, handling
the sync-to-async bridge, error mapping, ordered writes, compensation on
failure, and fallback to the standard chunker pipeline when no template matches.

Mirrors the ExcelProcessingService pattern.
"""

from __future__ import annotations

import asyncio
import json
import logging

from sqlalchemy.orm import Session

from ai_ready_rag.config import Settings
from ai_ready_rag.db.models import Document
from ai_ready_rag.services.forms_metrics import metrics as forms_metrics
from ai_ready_rag.services.processing_service import ProcessingResult

logger = logging.getLogger(__name__)

# Error codes that trigger fallback to standard chunker
_FALLBACK_ERROR_CODES = {"E_FORM_NO_MATCH", "E_FORM_UNSUPPORTED_FORMAT"}

# High-risk PII patterns for redaction (SSN, EIN, account numbers)
_HIGH_RISK_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
    r"\b\d{2}-\d{7}\b",  # EIN
    r"\b\d{9,18}\b",  # Account numbers
]


def _pymupdf_renderer(file_path: str, dpi: int) -> list:
    """Render PDF pages to PIL Images using PyMuPDF.

    Required by compute_layout_fingerprint_from_file for non-image formats.
    """
    import io

    import fitz
    from PIL import Image

    doc = fitz.open(file_path)
    images = []
    for page in doc:
        # Clear form field values before rendering so filled forms produce
        # the same fingerprint as blank templates (in-memory only, not saved).
        for widget in page.widgets():
            try:
                if widget.field_value:
                    widget.field_value = ""
                    widget.update()
            except (ValueError, RuntimeError):
                pass  # Skip widgets with invalid rects
        pix = page.get_pixmap(dpi=dpi)
        images.append(Image.open(io.BytesIO(pix.tobytes("png"))))
    doc.close()
    return images


class FormsProcessingService:
    """Processes PDF/XLSX forms through ingestkit-forms extraction pipeline.

    Creates VE-RAG adapter backends, configures a FormRouter, and maps
    ingestkit's FormProcessingResult back to VE-RAG's ProcessingResult.

    Owns the full forms lifecycle: match -> extract -> write -> update document.
    ProcessingService only calls process_form(); never calls try_match directly.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def process_form(
        self,
        document: Document,
        db: Session,
    ) -> tuple[ProcessingResult | None, bool]:
        """Process a form document using ingestkit-forms.

        Returns:
            Tuple of (ProcessingResult or None, should_fallback).
            If should_fallback is True, caller should fall back to standard chunker.
            If ProcessingResult is not None, processing completed (success or failure).
        """
        from ingestkit_forms import FileSystemTemplateStore, create_default_router
        from ingestkit_forms.config import FormProcessorConfig
        from ingestkit_forms.models import FormIngestRequest

        from ai_ready_rag.services.ingestkit_adapters import (
            VERagFormDBAdapter,
            VERagLayoutFingerprinter,
            VERagOCRAdapter,
            VERagPDFWidgetAdapter,
            VERagVectorStoreAdapter,
            VERagVLMAdapter,
            create_embedding_adapter,
        )

        settings = self.settings
        tag_names = [tag.name for tag in document.tags]
        file_path = document.file_path

        # 1. Build config FIRST (config before adapters)
        config_kwargs: dict = {
            "form_match_confidence_threshold": settings.forms_match_confidence_threshold,
            "form_ocr_engine": settings.forms_ocr_engine,
            "form_vlm_enabled": settings.forms_vlm_enabled,
            "embedding_model": settings.embedding_model or "nomic-embed-text",
            "embedding_dimension": settings.embedding_dimension,
            "default_collection": settings.qdrant_collection,
            "tenant_id": settings.default_tenant_id,
            "form_template_storage_path": settings.forms_template_storage_path,
            "redact_patterns": _HIGH_RISK_PATTERNS
            if settings.forms_redact_high_risk_fields
            else [],
            "redact_target": "chunks_only" if settings.forms_redact_high_risk_fields else "both",
        }
        # Only pass form_vlm_model if set (field has a non-None default)
        if settings.forms_vlm_model is not None:
            config_kwargs["form_vlm_model"] = settings.forms_vlm_model
        config = FormProcessorConfig(**config_kwargs)

        # 2. Create adapters
        vector_store = VERagVectorStoreAdapter(
            qdrant_url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            embedding_dimension=settings.embedding_dimension,
            document_id=document.id,
            document_name=document.original_filename,
            tags=tag_names,
            uploaded_by=document.uploaded_by or "system",
            tenant_id=settings.default_tenant_id,
        )
        embedder = create_embedding_adapter(
            ollama_url=settings.ollama_base_url,
            embedding_model=settings.embedding_model or "nomic-embed-text",
            embedding_dimension=settings.embedding_dimension,
        )
        form_db = VERagFormDBAdapter(db_path=settings.forms_db_path)
        template_store = FileSystemTemplateStore(
            base_path=settings.forms_template_storage_path,
        )
        fingerprinter = VERagLayoutFingerprinter(
            config,
            renderer=_pymupdf_renderer,
        )

        # OCR backend (required for scanned/flattened PDFs)
        ocr_backend = VERagOCRAdapter(engine=settings.forms_ocr_engine)

        # VLM backend (optional, for low-confidence field fallback)
        vlm_backend = None
        if settings.forms_vlm_enabled:
            vlm_backend = VERagVLMAdapter(
                ollama_url=settings.ollama_base_url,
                model=settings.forms_vlm_model,
            )

        # 3. Create router (with OCR + PDF widget + optional VLM backends)
        pdf_widget_backend = VERagPDFWidgetAdapter()
        router = create_default_router(
            template_store=template_store,
            form_db=form_db,
            vector_store=vector_store,
            embedder=embedder,
            fingerprinter=fingerprinter,
            ocr_backend=ocr_backend,
            pdf_widget_backend=pdf_widget_backend,
            vlm_backend=vlm_backend,
            config=config,
        )

        # 4. Match (FormsProcessingService owns this decision)
        try:
            match = await asyncio.to_thread(
                router.try_match,
                str(file_path),
                tenant_id=settings.default_tenant_id,
            )
        except Exception as e:
            logger.warning("forms.match.error", extra={"document_id": document.id, "error": str(e)})
            forms_metrics.inc_fallback("error")
            return (None, True)  # Fallback on match failure

        if match is None:
            return (None, True)  # No match, fallback to standard chunker

        logger.info(
            "forms.match.success",
            extra={
                "document_id": document.id,
                "template_id": match.template_id,
                "confidence": match.confidence,
            },
        )
        forms_metrics.inc_match_confidence(match.template_id, match.confidence)

        # 5. Extract + dual-write
        # IMPORTANT: Do NOT pass template_id from the auto-match result here.
        # extract_form does its own auto-matching internally. Passing template_id
        # would trigger the manual override path, incorrectly recording
        # match_method="manual_override" instead of "auto_detect".
        request = FormIngestRequest(
            file_path=str(file_path),
            tenant_id=settings.default_tenant_id,
        )
        try:
            result = await asyncio.to_thread(router.extract_form, request)
        except Exception as e:
            logger.error("forms.extract.error", extra={"document_id": document.id, "error": str(e)})
            forms_metrics.inc_documents_processed("failed")
            return (
                ProcessingResult(
                    success=False,
                    chunk_count=0,
                    page_count=None,
                    word_count=0,
                    processing_time_ms=0,
                    error_message=f"Forms extraction failed: {e}",
                ),
                False,
            )

        if result is None:
            return (None, True)

        # 6. Check for fallback error codes
        for err in result.error_details:
            if err.code.value in _FALLBACK_ERROR_CODES:
                # Compensate: clean up any partial writes
                await self._compensate(result, form_db, vector_store)
                forms_metrics.inc_documents_processed("fallback")
                forms_metrics.inc_fallback("unsupported")
                return (None, True)

        # 7. Check for hard errors (non-fallback)
        hard_errors = [e for e in result.errors if e not in _FALLBACK_ERROR_CODES]
        if hard_errors:
            error_msg = f"ingestkit-forms errors: {', '.join(hard_errors)}"
            logger.error("Forms processing failed for %s: %s", document.id, error_msg)
            return (
                ProcessingResult(
                    success=False,
                    chunk_count=0,
                    page_count=None,
                    word_count=0,
                    processing_time_ms=int(result.processing_time_seconds * 1000),
                    error_message=error_msg,
                ),
                False,
            )

        # 8. Success — log and record metrics
        er = result.extraction_result
        processing_time_ms = int(result.processing_time_seconds * 1000)

        logger.info(
            "forms.extract.success",
            extra={
                "document_id": document.id,
                "template_id": er.template_id,
                "extraction_method": er.extraction_method,
                "processing_time_ms": processing_time_ms,
            },
        )
        forms_metrics.inc_documents_processed("success")
        forms_metrics.observe_extraction_duration(
            er.extraction_method, result.processing_time_seconds
        )

        if result.warnings:
            logger.info(
                "ingestkit-forms completed with warnings for %s: %s",
                document.id,
                result.warnings,
            )

        # 9. Update document columns (last step — after all writes succeeded)
        document.forms_template_id = er.template_id
        document.forms_template_name = er.template_name
        document.forms_template_version = er.template_version
        document.forms_overall_confidence = er.overall_confidence
        document.forms_extraction_method = er.extraction_method
        document.forms_match_method = er.match_method
        document.forms_ingest_key = result.ingest_key
        if result.tables:
            document.forms_db_table_names = json.dumps(result.tables)

        return (
            ProcessingResult(
                success=True,
                chunk_count=result.chunks_created,
                page_count=er.pages_processed if hasattr(er, "pages_processed") else None,
                word_count=0,  # ingestkit-forms doesn't track word count
                processing_time_ms=processing_time_ms,
            ),
            False,
        )

    async def _compensate(
        self,
        result,
        form_db,
        vector_store,
    ) -> None:
        """Best-effort compensation: remove partial writes on fallback/failure."""
        # Delete vectors by ingest_key
        if result.written and result.written.vector_point_ids:
            try:
                vector_store.delete_by_filter(
                    self.settings.qdrant_collection,
                    "ingestkit_ingest_key",
                    result.ingest_key,
                )
            except Exception as e:
                logger.warning("forms.compensate.vectors_failed", extra={"error": str(e)})
                forms_metrics.inc_compensation("vectors", "failed")

        # Drop created tables
        for table_name in result.tables:
            try:
                form_db.check_table_name(table_name)
                form_db.execute_sql(f"DROP TABLE IF EXISTS [{table_name}]")
            except Exception as e:
                logger.warning(
                    "forms.compensate.table_failed",
                    extra={"table": table_name, "error": str(e)},
                )
                forms_metrics.inc_compensation("tables", "failed")
