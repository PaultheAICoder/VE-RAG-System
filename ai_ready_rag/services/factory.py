"""Service factories for profile-based backend selection.

Uses lazy imports to avoid loading unused dependencies:
- Laptop profile doesn't load Qdrant/Docling
- Spark profile doesn't load Chroma
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ai_ready_rag.config import Settings
from ai_ready_rag.services.settings_service import get_model_setting

if TYPE_CHECKING:
    from ai_ready_rag.services.processing_service import ProcessingOptions
    from ai_ready_rag.services.protocols import ChunkerProtocol, VectorServiceProtocol
    from ai_ready_rag.services.query_router import QueryRouter

logger = logging.getLogger(__name__)


def get_vector_service(settings: Settings) -> VectorServiceProtocol:
    """Factory that returns the appropriate vector backend.

    Args:
        settings: Application settings with vector_backend configured

    Returns:
        VectorServiceProtocol implementation (PgVector)

    Raises:
        ValueError: If vector_backend is not configured
    """
    backend = settings.vector_backend
    if not backend:
        raise ValueError("vector_backend not configured in settings")

    # Get embedding model from database first, fall back to config
    embedding_model = get_model_setting("embedding_model", settings.embedding_model)
    logger.info(f"Using embedding model: {embedding_model}")

    if backend == "pgvector":
        from ai_ready_rag.services.pgvector_service import PgVectorService

        logger.info(f"Creating PgVectorService: {settings.database_url[:30]}...")
        return PgVectorService(
            database_url=settings.database_url,
            ollama_url=settings.ollama_base_url,
            embedding_model=embedding_model,
            embedding_dimension=settings.embedding_dimension,
            tenant_id=settings.default_tenant_id,
        )
    else:
        raise ValueError(f"Unknown vector_backend: {backend!r}. Valid option: pgvector")


def get_chunker(
    settings: Settings,
    processing_options: ProcessingOptions | None = None,
    chunk_size_override: int | None = None,
) -> ChunkerProtocol:
    """Factory that returns the appropriate chunker.

    Args:
        settings: Application settings with chunker_backend configured
        processing_options: Optional per-upload processing options to override
            global settings for this specific document.

    Returns:
        ChunkerProtocol implementation (Simple or Docling)

    Raises:
        ValueError: If chunker_backend is not configured
    """
    backend = settings.chunker_backend
    if not backend:
        raise ValueError("chunker_backend not configured in settings")

    if backend == "docling":
        from ai_ready_rag.services.chunker_docling import DoclingChunker

        # Use per-upload options when provided, otherwise fall back to settings
        enable_ocr = (
            processing_options.enable_ocr
            if processing_options and processing_options.enable_ocr is not None
            else settings.enable_ocr or False
        )
        force_full_page_ocr = (
            processing_options.force_full_page_ocr
            if processing_options and processing_options.force_full_page_ocr is not None
            else settings.force_full_page_ocr
        )
        ocr_language = (
            processing_options.ocr_language
            if processing_options and processing_options.ocr_language is not None
            else settings.ocr_language
        )
        table_extraction_mode = (
            processing_options.table_extraction_mode
            if processing_options and processing_options.table_extraction_mode is not None
            else settings.table_extraction_mode
        )
        include_image_descriptions = (
            processing_options.include_image_descriptions
            if processing_options and processing_options.include_image_descriptions is not None
            else settings.include_image_descriptions
        )

        effective_chunk_size = chunk_size_override or settings.chunk_size
        logger.info(
            "Creating DoclingChunker with OCR=%s chunk_size=%d", enable_ocr, effective_chunk_size
        )
        return DoclingChunker(
            enable_ocr=enable_ocr,
            ocr_language=ocr_language,
            max_tokens=effective_chunk_size,
            force_full_page_ocr=force_full_page_ocr,
            table_extraction_mode=table_extraction_mode,
            include_image_descriptions=include_image_descriptions,
        )
    elif backend == "simple":
        from ai_ready_rag.services.chunker_simple import SimpleChunker

        enable_ocr = (
            processing_options.enable_ocr
            if processing_options and processing_options.enable_ocr is not None
            else settings.enable_ocr or False
        )
        ocr_language = (
            processing_options.ocr_language
            if processing_options and processing_options.ocr_language is not None
            else settings.ocr_language
        )

        effective_chunk_size = chunk_size_override or settings.chunk_size
        logger.info(
            "Creating SimpleChunker with OCR=%s chunk_size=%d", enable_ocr, effective_chunk_size
        )
        return SimpleChunker(
            chunk_size=effective_chunk_size * 4,  # Characters, not tokens
            chunk_overlap=settings.chunk_overlap * 4,
            enable_ocr=enable_ocr,
            ocr_language=ocr_language,
        )
    else:
        raise ValueError(f"Unknown chunker_backend: {backend}")


def get_enrichment_service(
    settings: Settings,
    db_session: object = None,
    tenant_config: object = None,
) -> object:
    """Factory that returns a ClaudeEnrichmentService instance.

    Args:
        settings: Application settings (used to detect enabled/disabled state).
        db_session: Optional SQLAlchemy session for persisting enrichment results.
        tenant_config: Optional TenantConfig — when provided, its feature flags
            (e.g. claude_enrichment_enabled) take precedence over global Settings.

    Returns:
        ClaudeEnrichmentService — self-disables when claude_enrichment_enabled is
        False, claude_api_key is None, or database_backend == 'sqlite'.
    """
    from ai_ready_rag.services.enrichment_service import ClaudeEnrichmentService

    return ClaudeEnrichmentService(
        settings=settings,
        db_session=db_session,
        tenant_config=tenant_config,
    )


def get_query_router() -> QueryRouter:
    """Factory that returns a QueryRouter configured from application settings.

    Returns:
        QueryRouter instance with sql_confidence_threshold from settings
        (defaults to 0.6 if setting is not present).
    """
    from ai_ready_rag.config import get_settings
    from ai_ready_rag.services.query_router import QueryRouter

    settings = get_settings()
    sql_threshold = getattr(settings, "structured_query_confidence_threshold", 0.6)
    return QueryRouter(sql_confidence_threshold=sql_threshold)
