"""Knowledge base reindex ARQ task.

Delegates to the existing run_reindex_job() worker function.
Falls back to BackgroundTasks if Redis is unavailable.
"""

import logging

logger = logging.getLogger(__name__)


async def reindex_knowledge_base(
    ctx: dict,
    job_id: str,
) -> dict:
    """ARQ task: reindex all documents in the knowledge base.

    Thin wrapper around the existing run_reindex_job() function which
    handles all the complexity (pause/abort detection, failure recording,
    progress tracking via ReindexService).

    Args:
        ctx: ARQ context dict (unused — run_reindex_job manages its own resources)
        job_id: The reindex job ID to process

    Returns:
        Dict with reindex result (success, job_id)
    """
    from ai_ready_rag.services.reindex_worker import run_reindex_job

    logger.info(f"[ARQ] Starting reindex job {job_id}")

    try:
        await run_reindex_job(job_id)
        logger.info(f"[ARQ] Reindex job {job_id} completed")
        return {"success": True, "job_id": job_id}
    except Exception as e:
        logger.exception(f"[ARQ] Reindex job {job_id} failed: {e}")
        return {"success": False, "job_id": job_id, "error": str(e)}
