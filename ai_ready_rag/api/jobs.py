"""Unified Jobs API for status polling, SSE streaming, and cancellation.

Provides a single interface across job types (cache_warming, reindex).
"""

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ai_ready_rag.core.dependencies import get_optional_current_user, require_system_admin
from ai_ready_rag.db.database import SessionLocal, get_db
from ai_ready_rag.db.models import ReindexJob, User
from ai_ready_rag.db.models.warming import WarmingBatch, WarmingQuery
from ai_ready_rag.schemas.jobs import JobCancelResponse, JobProgress, JobStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _find_job(db: Session, job_id: str) -> tuple[str, WarmingBatch | ReindexJob | None]:
    """Find a job by ID across WarmingBatch and ReindexJob tables.

    Returns (type, job) tuple. Type is "cache_warming" or "reindex".
    """
    warming_batch = db.query(WarmingBatch).filter(WarmingBatch.id == job_id).first()
    if warming_batch:
        return "cache_warming", warming_batch

    reindex_job = db.query(ReindexJob).filter(ReindexJob.id == job_id).first()
    if reindex_job:
        return "reindex", reindex_job

    return "", None


def _warming_to_status(db: Session, batch: WarmingBatch) -> JobStatusResponse:
    """Convert WarmingBatch to unified JobStatusResponse.

    Computes processed/failed counts from WarmingQuery aggregation
    since WarmingBatch does not store these as direct columns.
    """
    processed = (
        db.query(WarmingQuery)
        .filter(WarmingQuery.batch_id == batch.id, WarmingQuery.status == "completed")
        .count()
    )
    failed = (
        db.query(WarmingQuery)
        .filter(WarmingQuery.batch_id == batch.id, WarmingQuery.status == "failed")
        .count()
    )

    return JobStatusResponse(
        job_id=batch.id,
        type="cache_warming",
        status=batch.status,
        progress=JobProgress(
            total=batch.total_queries or 0,
            processed=processed,
            failed=failed,
        ),
        result_summary=(
            f"{processed}/{batch.total_queries} queries processed"
            if batch.status in ("completed", "completed_with_errors", "failed", "cancelled")
            else None
        ),
        created_at=batch.created_at,
        updated_at=batch.updated_at if hasattr(batch, "updated_at") else None,
    )


def _reindex_to_status(job: ReindexJob) -> JobStatusResponse:
    """Convert ReindexJob to unified JobStatusResponse."""
    return JobStatusResponse(
        job_id=job.id,
        type="reindex",
        status=job.status,
        progress=JobProgress(
            total=job.total_documents or 0,
            processed=job.processed_documents or 0,
            failed=job.failed_documents or 0,
        ),
        result_summary=(
            f"{job.processed_documents}/{job.total_documents} documents processed"
            if job.status in ("completed", "failed", "aborted")
            else None
        ),
        created_at=job.created_at,
        updated_at=job.updated_at if hasattr(job, "updated_at") else None,
    )


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get unified status for any job type.

    Queries WarmingBatch and ReindexJob tables by ID.
    Returns 404 if job not found in either table.

    Admin only.
    """
    job_type, job = _find_job(db, job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    if job_type == "cache_warming":
        return _warming_to_status(db, job)
    return _reindex_to_status(job)


@router.get("/{job_id}/stream")
async def stream_job_progress(
    job_id: str,
    token: str | None = None,
    current_user: User | None = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """Stream real-time SSE updates for a job.

    Polls job status from database every 2 seconds and emits progress events.
    Terminates when job reaches a terminal state.

    Token can be passed as query parameter for EventSource compatibility.

    Admin only.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )

    job_type, job = _find_job(db, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    async def event_generator():
        terminal_statuses = {
            "completed",
            "completed_with_errors",
            "failed",
            "cancelled",
            "aborted",
        }

        # Send connected event
        event_id = str(uuid.uuid4())
        data = {"job_id": job_id, "type": job_type, "message": "Connected"}
        yield f"id: {event_id}\nevent: connected\ndata: {json.dumps(data)}\n\n"

        while True:
            poll_db = SessionLocal()
            try:
                if job_type == "cache_warming":
                    current = poll_db.query(WarmingBatch).filter(WarmingBatch.id == job_id).first()
                    if current:
                        response = _warming_to_status(poll_db, current)
                else:
                    current = poll_db.query(ReindexJob).filter(ReindexJob.id == job_id).first()
                    if current:
                        response = _reindex_to_status(current)

                if current is None:
                    event_id = str(uuid.uuid4())
                    yield f"id: {event_id}\nevent: error\ndata: {json.dumps({'error': 'Job not found'})}\n\n"
                    return

                event_id = str(uuid.uuid4())
                payload = response.model_dump(mode="json")
                yield f"id: {event_id}\nevent: progress\ndata: {json.dumps(payload)}\n\n"

                if current.status in terminal_statuses:
                    event_id = str(uuid.uuid4())
                    yield f"id: {event_id}\nevent: complete\ndata: {json.dumps(payload)}\n\n"
                    return

            finally:
                poll_db.close()

            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{job_id}/cancel", response_model=JobCancelResponse)
async def cancel_job(
    job_id: str,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Cancel a running job.

    For warming batches: sets is_cancel_requested flag.
    For reindex jobs: sets status to 'aborted'.

    Admin only.
    """
    job_type, job = _find_job(db, job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    terminal_statuses = {"completed", "completed_with_errors", "failed", "cancelled", "aborted"}
    if job.status in terminal_statuses:
        return JobCancelResponse(
            job_id=job_id,
            cancelled=False,
            message=f"Job already in terminal state: {job.status}",
        )

    if job_type == "cache_warming":
        job.is_cancel_requested = True
        db.commit()
        logger.info(f"Admin {current_user.email} cancelled warming batch {job_id}")
        return JobCancelResponse(
            job_id=job_id,
            cancelled=True,
            message="Cancel requested for warming batch.",
        )
    else:
        job.status = "aborted"
        db.commit()
        logger.info(f"Admin {current_user.email} aborted reindex job {job_id}")
        return JobCancelResponse(
            job_id=job_id,
            cancelled=True,
            message="Reindex job aborted.",
        )
