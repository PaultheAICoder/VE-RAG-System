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

from sqlalchemy.orm import Session

from ai_ready_rag.config import Settings
from ai_ready_rag.db.models import Document
from ai_ready_rag.services.forms_metrics import metrics as forms_metrics
from ai_ready_rag.services.processing_service import ProcessingResult

logger = logging.getLogger(__name__)

# Error codes that trigger fallback to standard chunker
_FALLBACK_ERROR_CODES = {"E_FORM_NO_MATCH", "E_FORM_UNSUPPORTED_FORMAT"}

# Checkbox/binary field values that carry no semantic meaning and dilute embeddings
_BINARY_FIELD_VALUES = {"0", "1", "\\"}

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
    # OtherPolicy rows capture D&O, Crime, and similar non-standard coverages.
    # Keep them separate from CertificateHolder address noise.
    "Other Coverages": [
        "OtherPolicy_*",
    ],
    "Certificate Holder": [
        "CertificateHolder_*",
        "CertificateOf*",
        "Description*",
        "Remarks*",
        "Form_CompletionDate*",
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
            # Use tenant_id=None so "default" global templates are visible to all tenants.
            # Tenant-specific templates (if added later) will also match because
            # list_templates with tenant_id=None returns all templates.
            "tenant_id": None,
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
            database_url=settings.database_url,
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
        form_db = VERagFormDBAdapter(database_url=settings.database_url)
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
        # Pass tenant_id=None so "default" tenant templates are visible to all tenants.
        # tenant-specific templates (if any) will also match because list_templates
        # with tenant_id=None returns all templates regardless of their tenant_id.
        try:
            match = await asyncio.to_thread(
                router.try_match,
                str(file_path),
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
                # Rechunking is best-effort — original chunks are preserved on failure
                # (deletion now happens AFTER successful upsert of new chunks)
                logger.warning(
                    "forms.rechunk.failed: %s — %s",
                    type(e).__name__,
                    str(e),
                    exc_info=True,
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
        ACORD section, formats each group as natural text, embeds, and
        writes back. The original mega-chunk is deleted only AFTER the
        new chunks are successfully upserted (atomic swap to prevent
        data loss).

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
        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(form_db.get_connection_uri())
        conn.autocommit = True
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(f'SELECT * FROM "{form_db._SCHEMA}"."{table_name}" LIMIT 1')
            row = cursor.fetchone()
            if row is None:
                return 0
            fields: dict[str, str] = {}
            for col, val in row.items():
                v = str(val).strip() if val is not None else ""
                # Skip empty and binary checkbox values (0, 1, \) — they carry
                # no semantic meaning and dilute chunk embeddings.
                if v and v not in _BINARY_FIELD_VALUES:
                    fields[col] = v
        finally:
            conn.close()

        if not fields:
            return 0

        # 2.5 Claude structured extraction — reads raw PDF text and fills in fields
        #     that the OCR overlay missed (insurer names, OtherPolicy rows, etc.).
        #     Best-effort: failures are logged and skipped; OCR values are preserved.
        if getattr(settings, "forms_claude_extraction_enabled", False):
            _tmpl = document.forms_template_name or "Form"
            _pdf_path = str(document.file_path or "")
            _pdf_text = self._extract_pdf_text_for_forms(_pdf_path)
            claude_fields = self._extract_structured_fields_via_claude(
                fields, _pdf_text, _tmpl, settings
            )
            if claude_fields:
                fields.update(claude_fields)
                _form_id = fields.get("_form_id", "")
                if _form_id:
                    try:
                        self._write_claude_extractions_to_db(
                            form_db, table_name, _form_id, claude_fields
                        )
                    except Exception as _exc:
                        logger.warning(
                            "forms.claude_extract.db_write_failed",
                            extra={"document_id": document.id, "error": str(_exc)},
                        )

        # 3. Group fields by ACORD section
        groups: dict[str, dict[str, str]] = {}
        ungrouped: dict[str, str] = {}

        for field_name, field_value in fields.items():
            # Skip metadata columns (postgres schema uses _ prefix)
            if field_name.startswith("_"):
                continue

            # Strip XFA widget prefix (e.g. "F[0].P1[0].") so patterns like
            # "GeneralLiability_*" match the semantic part of the field name.
            import re as _re

            base_name = _re.sub(r"^F\[.*?\]\.[^.]+\.", "", field_name)

            matched = False
            for group_name, patterns in ACORD_25_GROUPS.items():
                for pattern in patterns:
                    if fnmatch.fnmatch(base_name, pattern):
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

        # 5. Format each group as natural text, splitting on character boundaries
        #    so no single chunk exceeds the embedding model's context window.
        #    nomic-embed-text accepts ~4 096 chars; use 3 800 as a safe margin.
        _MAX_CHUNK_CHARS = 3800

        chunk_texts: list[str] = []
        chunk_sections: list[str] = []

        for group_name, group_fields in groups.items():
            if not group_fields:
                continue

            group_header = f"{header}{group_name}\n"
            field_lines: list[str] = []
            for fname, fval in group_fields.items():
                # Strip XFA prefix before generating human-readable label
                import re as _re

                clean_name = _re.sub(r"^F\[.*?\]\.[^.]+\.", "", fname)
                label = self._field_name_to_label(clean_name)
                field_lines.append(f"{label}: {fval}")

            # Accumulate field lines into segments that stay under _MAX_CHUNK_CHARS
            segments: list[list[str]] = []
            current_lines: list[str] = []
            current_len = len(group_header)

            for line in field_lines:
                line_len = len(line) + 1  # +1 for \n
                if current_len + line_len > _MAX_CHUNK_CHARS and current_lines:
                    segments.append(current_lines)
                    current_lines = [line]
                    current_len = len(group_header) + line_len
                else:
                    current_lines.append(line)
                    current_len += line_len

            if current_lines:
                segments.append(current_lines)

            total = len(segments)
            for seg_idx, seg_lines in enumerate(segments):
                section_label = (
                    group_name if total == 1 else f"{group_name} ({seg_idx + 1}/{total})"
                )
                chunk_text = group_header + "\n".join(seg_lines)
                chunk_texts.append(chunk_text)
                chunk_sections.append(section_label)

        if not chunk_texts:
            return 0

        # 6. Embed each chunk via the ingestkit embedder
        from ingestkit_core.models import BaseChunkMetadata, ChunkPayload

        chunks_to_upsert: list[ChunkPayload] = []

        for i, (text, section) in enumerate(zip(chunk_texts, chunk_sections, strict=True)):
            embedding = embedder.embed([text])[0]
            chunk_id = f"{document.id}_form_{i}"
            metadata = BaseChunkMetadata(
                chunk_index=i,
                section_title=section,
                source_format="form_rechunk",
                ingestion_method="rechunk",
                parser_version="ve-rag-rechunk-1.0",
                # Use empty ingest_key so the post-upsert delete_by_filter
                # (which targets result.ingest_key) only removes the original
                # mega-chunk and never the rechunked replacements.
                ingest_key="",
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

        # 7. Generate synopsis chunk BEFORE upsert so all chunks go in one call.
        #    upsert_chunks() deletes existing document chunks first — a second call
        #    would wipe the 8 rechunked chunks written by the first batch.
        synopsis = self._generate_forms_synopsis(fields, template_name, settings)
        if synopsis:
            try:
                synopsis_embedding = embedder.embed([synopsis])[0]
                synopsis_id = f"{document.id}_form_synopsis"
                synopsis_meta = BaseChunkMetadata(
                    chunk_index=9999,
                    section_title="Coverage Synopsis",
                    source_format="form_synopsis",
                    ingestion_method="rechunk",
                    parser_version="ve-rag-rechunk-1.0",
                    ingest_key="",
                    chunk_hash="",
                    source_uri=str(document.file_path or ""),
                    ingest_run_id="",
                )
                chunks_to_upsert.append(
                    ChunkPayload(
                        id=synopsis_id,
                        text=synopsis,
                        vector=synopsis_embedding,
                        metadata=synopsis_meta,
                    )
                )
                logger.info(
                    "forms.synopsis.generated",
                    extra={"document_id": document.id, "length": len(synopsis)},
                )
            except Exception as exc:
                logger.warning(
                    "forms.synopsis.embed_failed",
                    extra={"document_id": document.id, "error": str(exc)},
                )

        # Inject insured entity name into adapter so all chunks carry insured_name metadata.
        if hasattr(vector_store, "set_entity_name"):
            vector_store.set_entity_name(insured_name or None)

        # Write all chunks (rechunked + synopsis) in a single upsert call.
        count = vector_store.upsert_chunks("", chunks_to_upsert)

        # 8. Delete original mega-chunk AFTER successful upsert of new chunks.
        #    This ordering prevents data loss: if embedding or upsert fails above,
        #    the exception propagates and original chunks are preserved.
        if result.ingest_key:
            try:
                vector_store.delete_by_filter(
                    "",
                    "ingestkit_ingest_key",
                    result.ingest_key,
                )
            except Exception as e:
                logger.warning(
                    "forms.rechunk.delete_original_failed",
                    extra={"document_id": document.id, "error": str(e)},
                )
                # Continue anyway — we'll have duplicates rather than missing data

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

    @staticmethod
    def _generate_forms_synopsis(
        fields: dict[str, str],
        template_name: str,
        settings: Settings,
    ) -> str | None:
        """Generate a natural-language synopsis of form fields via Claude CLI.

        Returns synopsis text, or None if Claude CLI is unavailable/disabled.
        Only runs when CLAUDE_BACKEND=cli.
        """
        import os
        import subprocess

        claude_backend = getattr(settings, "claude_backend", "api")
        if claude_backend != "cli":
            return None

        # Build condensed field text (skip metadata/internal columns)
        lines = [f"{k}: {v}" for k, v in fields.items() if not k.startswith("_")]
        field_text = "\n".join(lines[:200])  # cap at 200 fields

        prompt = (
            f"You are reviewing an ACORD 25 Certificate of Liability Insurance (COI) form: {template_name}.\n\n"
            "FIELD NAMING CONVENTION:\n"
            "- 'Insurer_FullName_A[0]' through '_F[0]' = insurer company names by letter.\n"
            "- Standard coverage rows (GL, Auto, etc.) use the INSR LTR suffix to identify the insurer.\n"
            "- OtherPolicy fields use a ROW INDEX suffix — NOT an insurer letter:\n"
            "    OtherPolicy_*_A[0]  = first Other coverage row\n"
            "    OtherPolicy_*_B[0]  = second Other coverage row\n"
            "  All fields sharing the same row suffix belong to the SAME row:\n"
            "    OtherPolicyDescription_A[0]   = coverage type for row A\n"
            "    PolicyNumberIdentifier_A[0]   = policy number for row A\n"
            "    OtherPolicy_InsurerLetterCode_A[0] = which insurer (A/B/C) covers row A\n"
            "    CoverageLimitAmount_A[0]       = dollar limit for row A\n"
            "  Apply the same logic for _B[0], _C[0] etc.\n\n"
            "RULES:\n"
            "1. For each OtherPolicy row, use OtherPolicyDescription_*[0] as the coverage name.\n"
            "2. Match CoverageLimitAmount_*[0] to the row with the SAME suffix letter.\n"
            "   Do NOT mix limits across rows (e.g. _B[0] limit belongs only to _B[0] row).\n"
            "3. Ignore CoverageCode_*[0] fields — they contain OCR noise, not reliable data.\n"
            "4. If a limit field is missing or contains only noise (e.g. 'overage Limit'), omit the limit rather than guessing.\n\n"
            "Write a plain-text retrieval-optimized COI coverage summary (no markdown, no bullets, no bold).\n"
            "Format:\n"
            "1. First line: 'INSURED: [insured name] — COI coverage limits, [form type], dated [date]:'\n"
            "2. List each insurer: 'Insurer A: [company from Insurer_FullName_A[0], or Unknown if missing]'\n"
            "3. For each coverage row, one sentence:\n"
            "   Standard rows: 'Insurer [INSR LTR] ([company]), Policy [number], [eff] to [exp]: [type] — [limits].'\n"
            "   OtherPolicy rows: 'Insurer [InsurerLetterCode] ([company]), Policy [PolicyNumberIdentifier], "
            "[eff] to [exp]: [OtherPolicyDescription] — [CoverageLimitAmount] limit.' "
            "(use the matching row-index suffix for every field in that sentence)\n"
            "4. End with a one-sentence summary of total policies and insurers.\n\n"
            "Include the words 'COI', 'coverage limits', and the insured name in the text.\n\n"
            f"Form fields:\n{field_text}"
        )

        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        try:
            proc = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
            if proc.returncode != 0:
                logger.warning(
                    "forms.synopsis.cli_failed",
                    extra={
                        "returncode": proc.returncode,
                        "stderr": proc.stderr.strip(),
                    },
                )
                return None
            synopsis = proc.stdout.strip()
            return synopsis if synopsis else None
        except Exception as exc:
            logger.warning("forms.synopsis.error", extra={"error": str(exc)})
            return None

    @staticmethod
    def _extract_pdf_text_for_forms(file_path: str, max_chars: int = 8000) -> str:
        """Extract raw text from PDF using PyMuPDF for Claude enrichment context.

        Used by _extract_structured_fields_via_claude to give Claude the full
        document text rather than relying solely on OCR overlay field values.
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)
            parts = [page.get_text() for page in doc]
            doc.close()
            return "\n".join(parts)[:max_chars]
        except Exception as exc:
            logger.debug("forms.pdf_text_extract.failed", extra={"error": str(exc)})
            return ""

    @staticmethod
    def _extract_structured_fields_via_claude(
        fields: dict[str, str],
        pdf_text: str,
        template_name: str,
        settings: Settings,
    ) -> dict[str, str] | None:
        """Extract/correct ACORD 25 form fields via Claude CLI using raw PDF text.

        The OCR overlay pipeline frequently misses insurer names and OtherPolicy
        coverage rows due to bounding box alignment issues. This method gives
        Claude the full raw PDF text and asks it to extract the specific fields
        that matter most for COI retrieval quality.

        Returns a dict of {xfa_field_name: value} using the same key format as
        the existing fields dict, so values can be merged directly and written
        back to the forms SQL table. Returns None if disabled or on failure.

        Only runs when CLAUDE_BACKEND=cli and forms_claude_extraction_enabled=True.
        """
        import os
        import subprocess

        claude_backend = getattr(settings, "claude_backend", "api")
        if claude_backend != "cli":
            return None

        # Build condensed OCR field text for context (skip metadata columns)
        ocr_lines = [f"{k}: {v}" for k, v in fields.items() if not k.startswith("_")]
        ocr_text = "\n".join(ocr_lines[:150])

        prompt = (
            f"You are extracting structured data from an ACORD 25 Certificate of Liability Insurance (COI) form: {template_name}\n\n"
            "Return ONLY a JSON object — no explanation, no markdown, no preamble.\n\n"
            "Extract values for the fields listed below from the raw document text.\n"
            "Use the EXACT JSON key strings shown. Omit any key where you cannot find a reliable value.\n\n"
            "TARGET FIELDS:\n"
            '  "F[0].P1[0].Insurer_FullName_A[0]": full legal company name of Insurer A\n'
            '  "F[0].P1[0].Insurer_FullName_B[0]": full legal company name of Insurer B\n'
            '  "F[0].P1[0].Insurer_FullName_C[0]": full legal company name of Insurer C (if present)\n'
            '  "F[0].P1[0].OtherPolicy_OtherPolicyDescription_A[0]": coverage type of first Other row (e.g. "Directors & Officers Liability")\n'
            '  "F[0].P1[0].OtherPolicy_PolicyNumberIdentifier_A[0]": policy number for first Other row\n'
            '  "F[0].P1[0].OtherPolicy_InsurerLetterCode_A[0]": INSR LTR letter (A/B/C) for first Other row\n'
            '  "F[0].P1[0].OtherPolicy_PolicyEffectiveDate_A[0]": effective date for first Other row (MM/DD/YYYY)\n'
            '  "F[0].P1[0].OtherPolicy_PolicyExpirationDate_A[0]": expiration date for first Other row (MM/DD/YYYY)\n'
            '  "F[0].P1[0].OtherPolicy_CoverageLimitAmount_A[0]": coverage limit for first Other row (digits and commas, e.g. "1,000,000")\n'
            '  "F[0].P1[0].OtherPolicy_OtherPolicyDescription_B[0]": coverage type of second Other row\n'
            '  "F[0].P1[0].OtherPolicy_PolicyNumberIdentifier_B[0]": policy number for second Other row\n'
            '  "F[0].P1[0].OtherPolicy_InsurerLetterCode_B[0]": INSR LTR letter for second Other row\n'
            '  "F[0].P1[0].OtherPolicy_PolicyEffectiveDate_B[0]": effective date for second Other row (MM/DD/YYYY)\n'
            '  "F[0].P1[0].OtherPolicy_PolicyExpirationDate_B[0]": expiration date for second Other row (MM/DD/YYYY)\n'
            '  "F[0].P1[0].OtherPolicy_CoverageLimitAmount_B[0]": coverage limit for second Other row (digits and commas)\n\n'
            "INSTRUCTIONS:\n"
            "- Insurer names (Insurer A, B, C...) appear in the INSURERS section listing companies by letter.\n"
            "- The INSR LTR column on each coverage row links it to an insurer letter (A, B, C...).\n"
            "- Policy numbers are alphanumeric, e.g. 'HOA1000040673-01' or 'SFD00001993 01'.\n"
            "- For limits, use digits and commas only (e.g. '250,000' not '$250,000').\n"
            "- Return {} (empty JSON) if you cannot find any of these values.\n\n"
            f"Existing OCR-extracted fields (may be incomplete or garbled):\n{ocr_text}\n\n"
            f"Raw document text:\n{pdf_text}"
        )

        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        try:
            proc = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
            if proc.returncode != 0:
                logger.warning(
                    "forms.claude_extract.cli_failed",
                    extra={"returncode": proc.returncode, "stderr": proc.stderr.strip()[:200]},
                )
                return None

            raw = proc.stdout.strip()
            # Strip accidental markdown fencing
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)

            try:
                extracted = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "forms.claude_extract.invalid_json",
                    extra={"raw_preview": raw[:300]},
                )
                return None

            if not isinstance(extracted, dict) or not extracted:
                return None

            # Keep only non-empty string values
            result = {k: str(v).strip() for k, v in extracted.items() if v and str(v).strip()}
            logger.info(
                "forms.claude_extract.success",
                extra={"fields_extracted": len(result)},
            )
            return result or None

        except Exception as exc:
            logger.warning("forms.claude_extract.error", extra={"error": str(exc)})
            return None

    @staticmethod
    def _write_claude_extractions_to_db(
        form_db,
        table_name: str,
        form_id: str,
        extracted: dict[str, str],
    ) -> None:
        """UPDATE forms SQL table row with Claude-extracted field values.

        Uses a parameterised UPDATE targeting _form_id. Column names are
        truncated to 63 chars to match the sanitiser applied by ingestkit-forms
        when the table was created (PostgreSQL max identifier length).
        """
        if not extracted or not form_id:
            return

        def _col(name: str) -> str:
            return name[:63]

        set_parts = [f'"{_col(col)}" = %s' for col in extracted]
        schema = getattr(form_db, "_SCHEMA", "forms_data")
        sql = f'UPDATE "{schema}"."{table_name}" SET {", ".join(set_parts)} WHERE "_form_id" = %s'
        params = tuple(extracted.values()) + (form_id,)
        form_db.execute_sql(sql, params)
        logger.info(
            "forms.claude_extract.db_updated",
            extra={"form_id": form_id, "fields_updated": len(extracted)},
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
                    "",
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
