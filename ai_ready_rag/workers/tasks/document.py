"""Document processing ARQ task.

Handles document parsing, chunking, and vector indexing as a
persistent background job via ARQ + Redis.
"""

import asyncio
import logging

from ai_ready_rag.config import get_settings

logger = logging.getLogger(__name__)

# Module-level semaphore for concurrency control
_processing_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the processing semaphore."""
    global _processing_semaphore
    if _processing_semaphore is None:
        settings = get_settings()
        _processing_semaphore = asyncio.Semaphore(settings.max_concurrent_processing)
    return _processing_semaphore


async def process_document(
    ctx: dict,
    document_id: str,
    processing_options_dict: dict | None = None,
    delete_existing: bool = False,
) -> dict:
    """ARQ task: process a document (parse, chunk, index vectors).

    Args:
        ctx: ARQ context dict (contains settings, vector_service from on_startup)
        document_id: Document ID to process
        processing_options_dict: Optional per-upload processing overrides
        delete_existing: If True, delete existing vectors first (reprocess)

    Returns:
        Dict with processing result (success, chunk_count, error)
    """
    import os

    from ai_ready_rag.db.database import SessionLocal
    from ai_ready_rag.db.models import Document
    from ai_ready_rag.services.factory import get_vector_service
    from ai_ready_rag.services.processing_service import ProcessingOptions, ProcessingService

    logger.info(f"[ARQ] Starting processing for document {document_id}")
    logger.info(f"[ARQ] TESSDATA_PREFIX={os.environ.get('TESSDATA_PREFIX', 'NOT SET')}")

    semaphore = _get_semaphore()
    async with semaphore:
        settings = ctx.get("settings") or get_settings()
        db = SessionLocal()

        # Reconstruct ProcessingOptions if provided
        processing_options = None
        if processing_options_dict:
            processing_options = ProcessingOptions(**processing_options_dict)

        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                logger.error(f"Document {document_id} not found for processing")
                return {"success": False, "error": "Document not found"}

            # Idempotency guard: skip if another worker already processed this document.
            # Prevents race conditions when duplicate jobs exist in the queue (e.g. a
            # stale standalone ARQ CLI worker competing with the embedded worker).
            if document.status == "ready":
                logger.info(f"[ARQ] Document {document_id} already ready, skipping")
                return {"success": True, "chunk_count": document.chunk_count, "skipped": True}

            # Use worker's vector service if available, otherwise create one
            vector_service = ctx.get("vector_service") or get_vector_service(settings)
            if not ctx.get("vector_service"):
                await vector_service.initialize()

            # Delete existing vectors if reprocessing
            if delete_existing:
                try:
                    await vector_service.delete_document(document_id)
                    logger.info(f"Deleted existing vectors for document {document_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete vectors for {document_id}: {e}")

            processing_service = ProcessingService(
                vector_service=vector_service,
                settings=settings,
            )

            result = await processing_service.process_document(
                document, db, processing_options=processing_options
            )

            if result.success:
                logger.info(
                    f"[ARQ] Document {document_id} processed: "
                    f"{result.chunk_count} chunks in {result.processing_time_ms}ms"
                )
                return {
                    "success": True,
                    "chunk_count": result.chunk_count,
                    "processing_time_ms": result.processing_time_ms,
                }
            else:
                logger.warning(f"[ARQ] Document {document_id} failed: {result.error_message}")
                return {"success": False, "error": result.error_message}

        except Exception as e:
            logger.exception(f"[ARQ] Unexpected error processing document {document_id}: {e}")
            try:
                document = db.query(Document).filter(Document.id == document_id).first()
                if document:
                    document.status = "failed"
                    document.error_message = f"Unexpected error: {e}"
                    db.commit()
            except Exception:
                logger.exception("Failed to update document status after error")
            return {"success": False, "error": str(e)}
        finally:
            db.close()
