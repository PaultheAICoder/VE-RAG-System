"""Image processing service using ingestkit-image pipeline.

Orchestrates ingestkit's ImageRouter with VE-RAG adapter backends, handling
the sync-to-async bridge and error mapping.

Images have NO fallback to the standard Docling chunker. should_fallback is
always False. Images with 0 extractable text (e.g. property photos) are still
marked "ready" with 0 chunks.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy.orm import Session

from ai_ready_rag.config import Settings
from ai_ready_rag.db.models import Document
from ai_ready_rag.services.processing_service import ProcessingResult

logger = logging.getLogger(__name__)


class ImageProcessingService:
    """Processes image files through ingestkit-image OCR pipeline.

    Creates VE-RAG adapter backends, configures an ImageRouter, and maps
    ingestkit's result back to VE-RAG's ProcessingResult.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def process_image(
        self,
        document: Document,
        db: Session,
    ) -> tuple[ProcessingResult | None, bool]:
        """Process an image file using ingestkit-image.

        Returns:
            Tuple of (ProcessingResult or None, should_fallback).
            should_fallback is always False for images (no fallback path).
        """
        from ingestkit_image import ImageRouter
        from ingestkit_image.config import ImageProcessorConfig

        from ai_ready_rag.services.ingestkit_adapters import (
            VERagImageOCRAdapter,
            VERagVectorStoreAdapter,
            create_embedding_adapter,
        )

        settings = self.settings
        tag_names = [tag.name for tag in document.tags]
        file_path = document.file_path

        # Build config
        config = ImageProcessorConfig(
            enable_ocr=True,
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
        ocr_backend = VERagImageOCRAdapter(
            language=settings.image_ocr_language,
            config=settings.image_ocr_config,
        )

        # Create router (OCR-only mode)
        router = ImageRouter(
            ocr=ocr_backend,
            vector_store=vector_store,
            embedder=embedder,
            config=config,
        )

        try:
            result = await asyncio.to_thread(router.process, file_path)
        except Exception as e:
            logger.error("ingestkit-image processing failed for %s: %s", document.id, e)
            return (
                ProcessingResult(
                    success=False,
                    chunk_count=0,
                    page_count=None,
                    word_count=0,
                    processing_time_ms=0,
                    error_message=f"Image processing failed: {e}",
                ),
                False,
            )

        # Check for hard errors
        if result.errors:
            error_msg = f"ingestkit-image errors: {', '.join(result.errors)}"
            logger.error("Image processing failed for %s: %s", document.id, error_msg)
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

        # Success: 0 chunks is OK for property photos (no extractable text)
        if result.warnings:
            logger.info(
                "ingestkit-image completed with warnings for %s: %s",
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
