"""Service factories for profile-based backend selection.

Uses lazy imports to avoid loading unused dependencies:
- Laptop profile doesn't load Qdrant/Docling
- Spark profile doesn't load Chroma
"""

import logging
from typing import TYPE_CHECKING

from ai_ready_rag.config import Settings

if TYPE_CHECKING:
    from ai_ready_rag.services.protocols import ChunkerProtocol, VectorServiceProtocol

logger = logging.getLogger(__name__)


def get_vector_service(settings: Settings) -> "VectorServiceProtocol":
    """Factory that returns the appropriate vector backend.

    Args:
        settings: Application settings with vector_backend configured

    Returns:
        VectorServiceProtocol implementation (Qdrant or Chroma)

    Raises:
        ValueError: If vector_backend is not configured
    """
    backend = settings.vector_backend
    if not backend:
        raise ValueError("vector_backend not configured in settings")

    if backend == "qdrant":
        from ai_ready_rag.services.vector_service import VectorService

        logger.info(f"Creating QdrantVectorService: {settings.qdrant_url}")
        return VectorService(
            qdrant_url=settings.qdrant_url,
            ollama_url=settings.ollama_base_url,
            collection_name=settings.qdrant_collection,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
        )
    elif backend == "chroma":
        from ai_ready_rag.services.vector_chroma import ChromaVectorService

        logger.info(f"Creating ChromaVectorService: {settings.chroma_persist_dir}")
        return ChromaVectorService(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.qdrant_collection,  # Reuse collection name setting
            embedding_model=settings.embedding_model,
            ollama_url=settings.ollama_base_url,
            embedding_dimension=settings.embedding_dimension,
        )
    else:
        raise ValueError(f"Unknown vector_backend: {backend}")


def get_chunker(settings: Settings) -> "ChunkerProtocol":
    """Factory that returns the appropriate chunker.

    Args:
        settings: Application settings with chunker_backend configured

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

        logger.info("Creating DoclingChunker with OCR=%s", settings.enable_ocr)
        return DoclingChunker(
            enable_ocr=settings.enable_ocr or False,
            ocr_language=settings.ocr_language,
            max_tokens=settings.chunk_size,
            force_full_page_ocr=settings.force_full_page_ocr,
            table_extraction_mode=settings.table_extraction_mode,
            include_image_descriptions=settings.include_image_descriptions,
        )
    elif backend == "simple":
        from ai_ready_rag.services.chunker_simple import SimpleChunker

        logger.info("Creating SimpleChunker")
        return SimpleChunker(
            chunk_size=settings.chunk_size * 4,  # Characters, not tokens
            chunk_overlap=settings.chunk_overlap * 4,
        )
    else:
        raise ValueError(f"Unknown chunker_backend: {backend}")
