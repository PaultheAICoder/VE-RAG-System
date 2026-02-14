"""Excel processing service using ingestkit-excel pipeline.

Orchestrates ingestkit's ExcelRouter with VE-RAG adapter backends, handling
the sync-to-async bridge, error mapping, and fallback to SimpleChunker when
classification is inconclusive.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from ai_ready_rag.config import Settings
from ai_ready_rag.db.models import Document
from ai_ready_rag.services.processing_service import ProcessingResult

logger = logging.getLogger(__name__)

# Error codes that trigger fallback to SimpleChunker
_FALLBACK_ERROR_CODES = {"E_CLASSIFY_INCONCLUSIVE"}

# Error codes that are retryable (transient backend failures)
_RETRYABLE_ERROR_CODES = {
    "E_BACKEND_EMBED_TIMEOUT",
    "E_BACKEND_VECTOR_TIMEOUT",
    "E_BACKEND_EMBED_CONNECT",
    "E_BACKEND_VECTOR_CONNECT",
}


class ExcelProcessingService:
    """Processes Excel files through ingestkit's 3-tier classification pipeline.

    Creates VE-RAG adapter backends, configures an ExcelRouter, and maps
    ingestkit's ProcessingResult back to VE-RAG's ProcessingResult.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def process_excel(
        self,
        document: Document,
        db: Session,
    ) -> tuple[ProcessingResult | None, bool]:
        """Process an Excel file using ingestkit-excel.

        Returns:
            Tuple of (ProcessingResult or None, should_fallback).
            If should_fallback is True, the caller should fall back to SimpleChunker.
            If ProcessingResult is not None, processing completed (success or failure).
        """
        from ingestkit_excel.config import ExcelProcessorConfig
        from ingestkit_excel.router import ExcelRouter

        from ai_ready_rag.services.ingestkit_adapters import (
            VERagVectorStoreAdapter,
            create_embedding_adapter,
            create_llm_adapter,
            create_structured_db,
        )

        settings = self.settings
        tag_names = [tag.name for tag in document.tags]
        file_path = document.file_path

        # Ensure excel tables DB directory exists
        db_path = Path(settings.excel_tables_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create adapter backends
        vector_store = VERagVectorStoreAdapter(
            qdrant_url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            embedding_dimension=settings.embedding_dimension,
            document_id=document.id,
            document_name=document.original_filename,
            tags=tag_names,
            uploaded_by=document.uploaded_by,
            tenant_id=settings.default_tenant_id,
        )

        embedder = create_embedding_adapter(
            ollama_url=settings.ollama_base_url,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
        )

        llm = create_llm_adapter(ollama_url=settings.ollama_base_url)

        structured_db = create_structured_db(db_path=str(db_path))

        # Build ingestkit config from VE-RAG settings
        config = ExcelProcessorConfig(
            classification_model=settings.excel_classification_model,
            reasoning_model=settings.excel_reasoning_model,
            tier2_confidence_threshold=settings.excel_tier2_confidence_threshold,
            enable_tier3=settings.excel_enable_tier3,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
            default_collection=settings.qdrant_collection,
            tenant_id=settings.default_tenant_id,
        )

        # Create router and process (sync pipeline, run in thread)
        router = ExcelRouter(
            vector_store=vector_store,
            structured_db=structured_db,
            llm=llm,
            embedder=embedder,
            config=config,
        )

        try:
            result = await asyncio.to_thread(router.process, file_path)
        except Exception as e:
            logger.error("ingestkit-excel processing failed for %s: %s", document.id, e)
            return None, True  # Fallback to SimpleChunker

        # Check for fallback-triggering errors
        if any(code in _FALLBACK_ERROR_CODES for code in result.errors):
            logger.warning(
                "ingestkit classification inconclusive for %s, falling back to SimpleChunker",
                document.id,
            )
            return None, True

        # Check for hard failures
        hard_errors = [code for code in result.errors if code not in _FALLBACK_ERROR_CODES]

        if hard_errors:
            error_msg = f"ingestkit-excel errors: {', '.join(hard_errors)}"
            logger.error("Excel processing failed for %s: %s", document.id, error_msg)

            is_retryable = any(code in _RETRYABLE_ERROR_CODES for code in hard_errors)
            if is_retryable:
                error_msg += " (retryable)"

            return ProcessingResult(
                success=False,
                chunk_count=0,
                page_count=None,
                word_count=0,
                processing_time_ms=int(result.processing_time_seconds * 1000),
                error_message=error_msg,
            ), False

        # Success (or warnings-only)
        if result.warnings:
            logger.info(
                "ingestkit-excel completed with warnings for %s: %s",
                document.id,
                result.warnings,
            )

        # Update document with ingestkit metadata
        cls_result = result.classification_result
        document.excel_file_type = cls_result.file_type.value
        document.excel_classification_tier = cls_result.tier_used.value
        document.excel_ingest_key = result.ingest_key
        document.excel_tables_created = result.tables_created

        if result.tables:
            document.excel_db_table_names = json.dumps(result.tables)

        processing_time_ms = int(result.processing_time_seconds * 1000)

        return ProcessingResult(
            success=True,
            chunk_count=result.chunks_created,
            page_count=None,
            word_count=0,  # ingestkit doesn't track word count
            processing_time_ms=processing_time_ms,
        ), False
