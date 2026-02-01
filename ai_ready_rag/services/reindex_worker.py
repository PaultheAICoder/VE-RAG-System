"""Background Reindex Worker.

Processes all documents for a reindex job, respecting pause/abort signals.
"""

import asyncio
import logging

from ai_ready_rag.config import get_settings
from ai_ready_rag.db.database import SessionLocal
from ai_ready_rag.db.models import Document
from ai_ready_rag.services.reindex_service import ReindexService

logger = logging.getLogger(__name__)


async def run_reindex_job(job_id: str) -> None:
    """Background task to process all documents for a reindex job.

    Creates its own db session to avoid session lifecycle issues.
    Checks job status on each iteration to detect pause/abort signals.

    Args:
        job_id: The reindex job ID to process.
    """
    from ai_ready_rag.services.factory import get_vector_service
    from ai_ready_rag.services.processing_service import ProcessingService

    logger.info(f"[REINDEX WORKER] Starting reindex job {job_id}")
    print(f"[REINDEX WORKER] Starting reindex job {job_id}", flush=True)

    settings = get_settings()
    db = SessionLocal()

    try:
        # Get the job
        reindex_service = ReindexService(db)
        job = reindex_service.get_job(job_id)
        if not job:
            logger.error(f"Reindex job {job_id} not found")
            return

        # Update status to running
        reindex_service.update_job_status(job_id, "running")

        # Get all ready documents
        documents = (
            db.query(Document)
            .filter(Document.status == "ready")
            .order_by(Document.uploaded_at)
            .all()
        )

        total_docs = len(documents)
        processed = 0
        failed = 0

        logger.info(f"[REINDEX] Processing {total_docs} documents")
        print(f"[REINDEX WORKER] Processing {total_docs} documents", flush=True)

        # Initialize vector service
        vector_service = get_vector_service(settings)
        await vector_service.initialize()

        processing_service = ProcessingService(
            vector_service=vector_service,
            settings=settings,
        )

        for doc in documents:
            # Check job status from database (detect pause/abort)
            db.refresh(job)
            current_status = job.status

            if current_status == "aborted":
                logger.info(f"[REINDEX] Job {job_id} aborted by user")
                print("[REINDEX WORKER] Job aborted", flush=True)
                break

            if current_status == "paused":
                logger.info(f"[REINDEX] Job {job_id} paused, waiting for resume...")
                print("[REINDEX WORKER] Job paused, waiting...", flush=True)
                # Wait for resume signal
                while True:
                    await asyncio.sleep(2)
                    db.refresh(job)
                    if job.status == "running":
                        logger.info(f"[REINDEX] Job {job_id} resumed")
                        print("[REINDEX WORKER] Job resumed", flush=True)
                        break
                    if job.status == "aborted":
                        logger.info(f"[REINDEX] Job {job_id} aborted while paused")
                        print("[REINDEX WORKER] Job aborted while paused", flush=True)
                        return

            # Update current document being processed
            reindex_service.update_job_progress(job_id, processed, failed, doc.id)

            try:
                # Delete existing vectors for this document
                try:
                    await vector_service.delete_document(doc.id)
                except Exception as e:
                    logger.warning(f"Failed to delete vectors for {doc.id}: {e}")
                    # Continue anyway - vectors may not exist

                # Process the document
                result = await processing_service.process_document(doc, db)

                if result.success:
                    processed += 1
                    logger.info(
                        f"[REINDEX] {processed}/{total_docs} - {doc.original_filename}: "
                        f"{result.chunk_count} chunks"
                    )
                    print(
                        f"[REINDEX WORKER] {processed}/{total_docs} - {doc.original_filename}: OK",
                        flush=True,
                    )
                else:
                    # Record failure
                    reindex_service.record_failure(
                        job_id, doc.id, result.error_message or "Unknown error"
                    )
                    db.refresh(job)

                    # Check if we should continue (auto-skip) or pause
                    if job.status == "paused":
                        failed += 1
                        logger.warning(
                            f"[REINDEX] Paused due to failure: {doc.original_filename} - {result.error_message}"
                        )
                        print(
                            f"[REINDEX WORKER] PAUSED - {doc.original_filename}: {result.error_message}",
                            flush=True,
                        )
                        # Wait for resume
                        while True:
                            await asyncio.sleep(2)
                            db.refresh(job)
                            if job.status == "running":
                                # Check if we should retry or skip
                                if job.current_document_id == doc.id and job.retry_count > 0:
                                    # Retry was requested - don't increment processed, try again
                                    logger.info(f"[REINDEX] Retrying {doc.original_filename}")
                                    print(
                                        f"[REINDEX WORKER] Retrying {doc.original_filename}",
                                        flush=True,
                                    )
                                    # Re-attempt this document (loop will handle it)
                                    continue
                                else:
                                    # Skip was requested
                                    processed += 1  # Count as processed (skipped)
                                    logger.info(f"[REINDEX] Skipping {doc.original_filename}")
                                    print(
                                        f"[REINDEX WORKER] Skipped {doc.original_filename}",
                                        flush=True,
                                    )
                                break
                            if job.status == "aborted":
                                logger.info(f"[REINDEX] Job {job_id} aborted while paused")
                                return
                    else:
                        # Auto-skip mode - continue
                        processed += 1
                        failed += 1
                        logger.warning(
                            f"[REINDEX] {processed}/{total_docs} - {doc.original_filename}: "
                            f"FAILED (auto-skip) - {result.error_message}"
                        )
                        print(
                            f"[REINDEX WORKER] {processed}/{total_docs} - {doc.original_filename}: FAILED (skipped)",
                            flush=True,
                        )

            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                logger.exception(f"[REINDEX] Error processing {doc.original_filename}: {e}")

                # Record failure
                reindex_service.record_failure(job_id, doc.id, error_msg)
                db.refresh(job)

                if job.status == "paused":
                    failed += 1
                    print(
                        f"[REINDEX WORKER] PAUSED - {doc.original_filename}: {error_msg}",
                        flush=True,
                    )
                    # Wait for resume
                    while True:
                        await asyncio.sleep(2)
                        db.refresh(job)
                        if job.status == "running":
                            processed += 1  # Skip and continue
                            break
                        if job.status == "aborted":
                            return
                else:
                    # Auto-skip
                    processed += 1
                    failed += 1

            # Update progress
            reindex_service.update_job_progress(job_id, processed, failed, None)

        # Job completed
        db.refresh(job)
        if job.status == "running":
            reindex_service.update_job_status(job_id, "completed")
            logger.info(f"[REINDEX] Job {job_id} completed: {processed} processed, {failed} failed")
            print(
                f"[REINDEX WORKER] COMPLETED: {processed} processed, {failed} failed",
                flush=True,
            )

    except Exception as e:
        logger.exception(f"[REINDEX] Fatal error in reindex job {job_id}: {e}")
        print(f"[REINDEX WORKER] FATAL ERROR: {e}", flush=True)
        try:
            reindex_service = ReindexService(db)
            reindex_service.update_job_status(job_id, "failed", str(e))
        except Exception:
            logger.exception("Failed to update job status after fatal error")
    finally:
        db.close()
