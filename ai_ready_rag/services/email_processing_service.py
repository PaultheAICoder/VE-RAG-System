"""Email processing service using ingestkit-email pipeline.

Orchestrates ingestkit's EmailRouter with VE-RAG adapter backends, handling
the sync-to-async bridge and error mapping.

Emails have NO fallback to the standard Docling chunker. should_fallback is
always False. Errors are hard failures.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy.orm import Session

from ai_ready_rag.config import Settings
from ai_ready_rag.db.models import Document
from ai_ready_rag.services.processing_service import ProcessingResult

logger = logging.getLogger(__name__)


class EmailProcessingService:
    """Processes email files (.eml, .msg) through ingestkit-email pipeline.

    Creates VE-RAG adapter backends, configures an EmailRouter, and maps
    ingestkit's result back to VE-RAG's ProcessingResult.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def process_email(
        self,
        document: Document,
        db: Session,
    ) -> tuple[ProcessingResult | None, bool]:
        """Process an email file using ingestkit-email.

        Returns:
            Tuple of (ProcessingResult or None, should_fallback).
            should_fallback is always False for emails (no fallback path).
        """
        from ingestkit_email import EmailRouter
        from ingestkit_email.config import EmailProcessorConfig

        from ai_ready_rag.services.ingestkit_adapters import (
            VERagVectorStoreAdapter,
            create_embedding_adapter,
        )

        settings = self.settings
        tag_names = [tag.name for tag in document.tags]
        file_path = document.file_path

        # Build config
        config = EmailProcessorConfig(
            include_headers=settings.email_include_headers,
            embedding_model=settings.embedding_model or "nomic-embed-text",
            embedding_dimension=settings.embedding_dimension,
            default_collection=settings.qdrant_collection,
            tenant_id=settings.default_tenant_id,
        )

        # Create adapters
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

        # Create router
        router = EmailRouter(
            vector_store=vector_store,
            embedder=embedder,
            config=config,
        )

        try:
            result = await asyncio.to_thread(router.process, file_path)
        except Exception as e:
            logger.error("ingestkit-email processing failed for %s: %s", document.id, e)
            return (
                ProcessingResult(
                    success=False,
                    chunk_count=0,
                    page_count=None,
                    word_count=0,
                    processing_time_ms=0,
                    error_message=f"Email processing failed: {e}",
                ),
                False,
            )

        # Check for hard errors
        if result.errors:
            error_msg = f"ingestkit-email errors: {', '.join(result.errors)}"
            logger.error("Email processing failed for %s: %s", document.id, error_msg)
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

        # Success
        if result.warnings:
            logger.info(
                "ingestkit-email completed with warnings for %s: %s",
                document.id,
                result.warnings,
            )

        processing_time_ms = int(result.processing_time_seconds * 1000)

        return (
            ProcessingResult(
                success=True,
                chunk_count=result.chunks_created,
                page_count=None,
                word_count=0,
                processing_time_ms=processing_time_ms,
            ),
            False,
        )
