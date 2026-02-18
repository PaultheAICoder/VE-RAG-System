"""Forms processing service using ingestkit-forms pipeline.

Orchestrates ingestkit's FormRouter with VE-RAG adapter backends, handling
the sync-to-async bridge, error mapping, ordered writes, compensation on
failure, and fallback to the standard chunker pipeline when no template matches.

Mirrors the ExcelProcessingService pattern.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import re
import sqlite3

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

# Field name prefix mapping for ACORD 25 rechunking.
# Keys are group names; values are fnmatch patterns matched against field names.
ACORD_25_GROUPS: dict[str, list[str]] = {
    "Producer Info": ["Producer_*", "Producer*"],
    "Named Insured": ["NamedInsured_*", "NamedInsured*", "Insured*"],
    "Insurers": ["Insurer_*", "Insurer*"],
    "General Liability": ["GeneralLiability_*", "GeneralLiability*", "Policy_GeneralLiability_*"],
    "Auto Liability": [
        "Vehicle_*",
        "Vehicle*",
        "Policy_AutomobileLiability_*",
        "AutomobileLiability*",
    ],
    "Umbrella/Excess": [
        "ExcessUmbrella_*",
        "ExcessUmbrella*",
        "Policy_ExcessLiability_*",
        "Umbrella*",
    ],
    "Workers Comp": ["WorkersCompensation*", "Policy_WorkersCompensation*"],
    "Other": [
        "OtherPolicy_*",
        "CertificateHolder_*",
        "CertificateOf*",
        "Description*",
        "Remarks*",
    ],
}


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

        # Reject matches below our confidence threshold (prevents false positives
        # like loss runs being misidentified as ACORD forms)
        if match.confidence < settings.forms_match_confidence_threshold:
            logger.info(
                "forms.match.below_threshold",
                extra={
                    "document_id": document.id,
                    "template_id": match.template_id,
                    "confidence": match.confidence,
                    "threshold": settings.forms_match_confidence_threshold,
                },
            )
            forms_metrics.inc_fallback("low_confidence")
            return (None, True)  # Below threshold, fallback to standard chunker

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

        # 8.5 Rechunk form mega-chunk into field groups (if enabled)
        rechunk_count = 0
        if self.settings.forms_rechunk_enabled:
            try:
                rechunk_count = await self._rechunk_form(
                    document=document,
                    result=result,
                    settings=settings,
                    form_db=form_db,
                    vector_store=vector_store,
                    embedder=embedder,
                )
                if rechunk_count > 0:
                    logger.info(
                        "forms.rechunk.success",
                        extra={
                            "document_id": document.id,
                            "original_chunks": result.chunks_created,
                            "rechunked_chunks": rechunk_count,
                        },
                    )
            except Exception as e:
                # Rechunking is best-effort — original chunks still exist if this fails
                logger.warning(
                    "forms.rechunk.failed",
                    extra={"document_id": document.id, "error": str(e)},
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
                chunk_count=rechunk_count if rechunk_count > 0 else result.chunks_created,
                page_count=er.pages_processed if hasattr(er, "pages_processed") else None,
                word_count=0,  # ingestkit-forms doesn't track word count
                processing_time_ms=processing_time_ms,
            ),
            False,
        )

    async def _rechunk_form(
        self,
        document: Document,
        result,
        settings: Settings,
        form_db,
        vector_store,
        embedder,
    ) -> int:
        """Split form mega-chunk into logical field-group chunks.

        Reads extracted fields from the forms SQLite DB, groups them by
        ACORD section, deletes the original mega-chunk from Qdrant,
        formats each group as natural text, embeds, and writes back.

        Returns the number of new chunks created (0 if rechunking was skipped).
        """
        # 1. Find the forms DB table(s) for this document
        table_names = result.tables
        if not table_names:
            logger.debug("forms.rechunk.skip: no tables for %s", document.id)
            return 0

        # 2. Read fields from the first table (ACORD forms typically have one)
        table_name = table_names[0]
        form_db.check_table_name(table_name)
        db_path = form_db.get_connection_uri().replace("sqlite:///", "")
        conn = sqlite3.connect(db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"SELECT * FROM [{table_name}] LIMIT 1")
            row = cursor.fetchone()
            if row is None:
                return 0
            columns = row.keys()
            fields: dict[str, str] = {}
            for col in columns:
                val = row[col]
                if val is not None and str(val).strip():
                    fields[col] = str(val).strip()
        finally:
            conn.close()

        if not fields:
            return 0

        # 3. Group fields by ACORD section
        groups: dict[str, dict[str, str]] = {}
        ungrouped: dict[str, str] = {}

        for field_name, field_value in fields.items():
            # Skip metadata columns
            if field_name.lower() in (
                "id",
                "ingest_key",
                "document_id",
                "tenant_id",
                "created_at",
                "updated_at",
            ):
                continue

            matched = False
            for group_name, patterns in ACORD_25_GROUPS.items():
                for pattern in patterns:
                    if fnmatch.fnmatch(field_name, pattern):
                        if group_name not in groups:
                            groups[group_name] = {}
                        groups[group_name][field_name] = field_value
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                ungrouped[field_name] = field_value

        # Add ungrouped fields to "Other"
        if ungrouped:
            if "Other" not in groups:
                groups["Other"] = {}
            groups["Other"].update(ungrouped)

        if not groups:
            return 0

        # 4. Build document header from key fields
        template_name = document.forms_template_name or "Form"
        insured_name = fields.get("NamedInsured_Name", fields.get("Insured_Name", ""))
        header = f"{template_name}\n"
        if insured_name:
            header += f"Insured: {insured_name}\n"

        # 5. Format each group as natural text
        chunk_texts: list[str] = []
        chunk_sections: list[str] = []
        for group_name, group_fields in groups.items():
            if not group_fields:
                continue

            lines = [f"{header}{group_name}\n"]
            for fname, fval in group_fields.items():
                # Clean up field name: CamelCase_SubField -> readable label
                label = self._field_name_to_label(fname)
                lines.append(f"{label}: {fval}")

            chunk_text = "\n".join(lines)
            chunk_texts.append(chunk_text)
            chunk_sections.append(group_name)

        if not chunk_texts:
            return 0

        # 6. Delete original mega-chunk from Qdrant by ingest_key
        if result.ingest_key:
            try:
                vector_store.delete_by_filter(
                    settings.qdrant_collection,
                    "ingestkit_ingest_key",
                    result.ingest_key,
                )
            except Exception as e:
                logger.warning(
                    "forms.rechunk.delete_original_failed",
                    extra={"document_id": document.id, "error": str(e)},
                )
                # Continue anyway — we'll have duplicates rather than missing data

        # 7. Embed each chunk via the ingestkit embedder
        from ingestkit_core.models import ChunkMetadata, ChunkPayload

        chunks_to_upsert: list[ChunkPayload] = []

        for i, (text, section) in enumerate(zip(chunk_texts, chunk_sections, strict=True)):
            embedding = embedder.embed(text)
            chunk_id = f"{document.id}_form_{i}"
            metadata = ChunkMetadata(
                chunk_index=i,
                section_title=section,
                source_format="form_rechunk",
                ingestion_method="rechunk",
                parser_version="ve-rag-rechunk-1.0",
                ingest_key=result.ingest_key or "",
                chunk_hash="",
                source_uri=str(document.file_path or ""),
                ingest_run_id="",
            )
            chunks_to_upsert.append(
                ChunkPayload(
                    id=chunk_id,
                    text=text,
                    vector=embedding,
                    metadata=metadata,
                )
            )

        # 8. Write new chunks to Qdrant via the adapter
        count = vector_store.upsert_chunks(settings.qdrant_collection, chunks_to_upsert)
        return count

    @staticmethod
    def _field_name_to_label(field_name: str) -> str:
        """Convert a form field name to a human-readable label.

        Examples:
            GeneralLiability_EachOccurrenceLimit -> Each Occurrence Limit
            Producer_ContactName -> Contact Name
        """
        # Remove group prefix (everything before first underscore)
        parts = field_name.split("_", 1)
        name = parts[-1] if len(parts) > 1 else parts[0]
        # Split CamelCase
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        # Replace underscores with spaces
        name = name.replace("_", " ")
        return name.strip()

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
