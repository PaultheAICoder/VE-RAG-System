"""Batch warming ARQ task.

Processes a WarmingBatch by iterating through its WarmingQuery rows
with idempotent claiming, lease management, retry logic, and
pause/cancel support.

Public helper functions (acquire_batch_lease, claim_next_query, etc.)
are shared by both the ARQ task and the background WarmingWorker.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta

from sqlalchemy import update
from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.exceptions import (
    ConnectionTimeoutError,
    RateLimitExceededError,
    ServiceUnavailableError,
)
from ai_ready_rag.db.models.warming import WarmingBatch, WarmingQuery

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (
    ConnectionTimeoutError,
    ServiceUnavailableError,
    RateLimitExceededError,
    asyncio.TimeoutError,
)

__all__ = [
    "RETRYABLE_EXCEPTIONS",
    "acquire_batch_lease",
    "claim_next_query",
    "warm_query_with_retry",
    "cancel_batch",
    "wait_for_resume_or_cancel",
    "finalize_batch",
    "process_warming_batch",
]


def acquire_batch_lease(db: Session, batch_id: str, worker_id: str, settings: object) -> bool:
    """Attempt to acquire or re-acquire a lease on the batch.

    Succeeds when:
    1. Batch is pending (first acquisition), OR
    2. Batch is running with the same worker_id (ARQ retry), OR
    3. Batch is running but the lease has expired (stale worker recovery).

    Returns True if lease was acquired, False otherwise.
    """
    now = datetime.utcnow()
    lease_expires = now + timedelta(minutes=settings.warming_lease_duration_minutes)

    result = db.execute(
        update(WarmingBatch)
        .where(
            WarmingBatch.id == batch_id,
            (
                (WarmingBatch.status == "pending")
                | ((WarmingBatch.status == "running") & (WarmingBatch.worker_id == worker_id))
                | (
                    (WarmingBatch.status == "running")
                    & (WarmingBatch.worker_lease_expires_at < now)
                )
            ),
        )
        .values(
            status="running",
            worker_id=worker_id,
            worker_lease_expires_at=lease_expires,
            started_at=db.query(WarmingBatch.started_at)
            .filter(WarmingBatch.id == batch_id)
            .scalar()
            or now,
            updated_at=now,
        )
    )
    db.commit()
    return result.rowcount > 0


def claim_next_query(db: Session, batch_id: str) -> WarmingQuery | None:
    """Idempotently claim the next pending query in sort order.

    Uses SELECT then UPDATE WHERE status='pending' to handle
    concurrent workers safely -- if rowcount is 0, the query
    was already claimed by another worker.

    Returns the claimed WarmingQuery row or None.
    """
    query_row = (
        db.query(WarmingQuery)
        .filter(WarmingQuery.batch_id == batch_id, WarmingQuery.status == "pending")
        .order_by(WarmingQuery.sort_order.asc())
        .first()
    )
    if query_row is None:
        return None

    now = datetime.utcnow()
    result = db.execute(
        update(WarmingQuery)
        .where(WarmingQuery.id == query_row.id, WarmingQuery.status == "pending")
        .values(status="processing", updated_at=now)
    )
    db.commit()

    if result.rowcount == 0:
        return None

    db.refresh(query_row)
    return query_row


async def warm_query_with_retry(
    rag_service: object,
    db: Session,
    query_row: WarmingQuery,
    settings: object,
) -> bool:
    """Execute a single query through RAG pipeline with retry on transient errors.

    Returns True on success, False on failure (max retries exhausted or
    non-retryable error).
    """
    from ai_ready_rag.services.rag_service import RAGRequest

    retry_delays = [int(d) for d in settings.warming_retry_delays.split(",")]
    max_attempts = settings.warming_max_retries + 1
    now = datetime.utcnow

    for attempt in range(max_attempts):
        try:
            request = RAGRequest(
                query=query_row.query_text,
                user_tags=[],  # Admin context -- cached without tag restriction
                tenant_id="default",
                is_warming=True,
            )
            await rag_service.generate(request, db)

            # Success
            db.execute(
                update(WarmingQuery)
                .where(WarmingQuery.id == query_row.id)
                .values(
                    status="completed",
                    retry_count=attempt,
                    processed_at=now(),
                    updated_at=now(),
                )
            )
            db.commit()
            return True

        except RETRYABLE_EXCEPTIONS as exc:
            if attempt < max_attempts - 1:
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                logger.warning(
                    f"Retryable error on query {query_row.id} "
                    f"(attempt {attempt + 1}/{max_attempts}): {exc}. "
                    f"Retrying in {delay}s"
                )
                await asyncio.sleep(delay)
            else:
                error_msg = str(exc)[:500]
                db.execute(
                    update(WarmingQuery)
                    .where(WarmingQuery.id == query_row.id)
                    .values(
                        status="failed",
                        error_message=error_msg,
                        error_type=type(exc).__name__,
                        retry_count=attempt,
                        processed_at=now(),
                        updated_at=now(),
                    )
                )
                db.commit()
                return False

        except Exception as exc:
            error_msg = str(exc)[:500]
            db.execute(
                update(WarmingQuery)
                .where(WarmingQuery.id == query_row.id)
                .values(
                    status="failed",
                    error_message=error_msg,
                    error_type=type(exc).__name__,
                    retry_count=attempt,
                    processed_at=now(),
                    updated_at=now(),
                )
            )
            db.commit()
            return False

    return False  # pragma: no cover


def cancel_batch(db: Session, batch_id: str) -> None:
    """Cancel a batch: skip remaining pending queries, mark batch cancelled."""
    now = datetime.utcnow()

    db.execute(
        update(WarmingQuery)
        .where(WarmingQuery.batch_id == batch_id, WarmingQuery.status == "pending")
        .values(status="skipped", updated_at=now)
    )
    db.execute(
        update(WarmingBatch)
        .where(WarmingBatch.id == batch_id)
        .values(status="cancelled", completed_at=now, updated_at=now)
    )
    db.commit()


async def wait_for_resume_or_cancel(db: Session, batch_id: str, settings: object) -> str:
    """Block while batch is paused, polling DB for state changes.

    Sets batch status to 'paused' on entry. Polls every 2 seconds.
    Returns 'cancel' if is_cancel_requested becomes True,
    or 'resume' if is_paused becomes False.
    """
    now = datetime.utcnow()
    db.execute(
        update(WarmingBatch)
        .where(WarmingBatch.id == batch_id)
        .values(status="paused", updated_at=now)
    )
    db.commit()

    while True:
        await asyncio.sleep(2)

        # Re-read fresh state
        db.expire_all()
        batch = db.query(WarmingBatch).filter(WarmingBatch.id == batch_id).first()
        if batch is None:
            return "cancel"

        if batch.is_cancel_requested:
            return "cancel"

        if not batch.is_paused:
            now = datetime.utcnow()
            db.execute(
                update(WarmingBatch)
                .where(WarmingBatch.id == batch_id)
                .values(status="running", updated_at=now)
            )
            db.commit()
            return "resume"


def finalize_batch(db: Session, batch_id: str) -> None:
    """Set terminal batch status based on query outcomes.

    - All completed (+ skipped): "completed"
    - Any failed: "completed_with_errors"
    - Guard: do NOT finalize if any queries are still pending/processing.
    """
    now = datetime.utcnow()

    # Guard: don't finalize if any queries are still in non-terminal state
    non_terminal_count = (
        db.query(WarmingQuery)
        .filter(
            WarmingQuery.batch_id == batch_id,
            WarmingQuery.status.in_(["pending", "processing"]),
        )
        .count()
    )
    if non_terminal_count > 0:
        logger.warning(
            f"Cannot finalize batch {batch_id}: "
            f"{non_terminal_count} queries still in non-terminal state"
        )
        return

    failed_count = (
        db.query(WarmingQuery)
        .filter(WarmingQuery.batch_id == batch_id, WarmingQuery.status == "failed")
        .count()
    )

    status = "completed_with_errors" if failed_count > 0 else "completed"
    db.execute(
        update(WarmingBatch)
        .where(WarmingBatch.id == batch_id)
        .values(status=status, completed_at=now, updated_at=now)
    )
    db.commit()


async def process_warming_batch(ctx: dict, batch_id: str) -> dict:
    """ARQ task: process all queries in a warming batch.

    Args:
        ctx: ARQ context dict (contains settings, vector_service from on_startup)
        batch_id: WarmingBatch ID to process

    Returns:
        Dict with processing result (success, processed, failed, error)
    """
    from ai_ready_rag.db.database import SessionLocal
    from ai_ready_rag.services.factory import get_vector_service
    from ai_ready_rag.services.rag_service import RAGService

    settings = ctx.get("settings") or get_settings()
    db = SessionLocal()
    worker_id = str(uuid.uuid4())

    logger.info(f"[ARQ] Starting batch warming for batch {batch_id} (worker {worker_id})")

    try:
        if not acquire_batch_lease(db, batch_id, worker_id, settings):
            logger.warning(f"[ARQ] Batch {batch_id} not available for lease acquisition")
            return {"success": False, "error": "Batch not available for processing"}

        vector_service = ctx.get("vector_service") or get_vector_service(settings)
        if not ctx.get("vector_service"):
            await vector_service.initialize()

        rag_service = RAGService(settings, vector_service=vector_service)
        processed = 0
        failed = 0
        cancelled = False

        while True:
            # Re-read batch for pause/cancel flags
            db.expire_all()
            batch = db.query(WarmingBatch).filter(WarmingBatch.id == batch_id).first()
            if batch is None:
                logger.error(f"[ARQ] Batch {batch_id} disappeared during processing")
                return {"success": False, "error": "Batch not found"}

            if batch.is_cancel_requested:
                cancel_batch(db, batch_id)
                cancelled = True
                break

            if batch.is_paused:
                result = await wait_for_resume_or_cancel(db, batch_id, settings)
                if result == "cancel":
                    cancel_batch(db, batch_id)
                    cancelled = True
                    break
                continue  # Resume -- re-check state at top of loop

            query_row = claim_next_query(db, batch_id)
            if query_row is None:
                break  # All queries processed

            success = await warm_query_with_retry(rag_service, db, query_row, settings)
            if success:
                processed += 1
            else:
                failed += 1

            # Throttle between queries
            if settings.warming_delay_seconds > 0:
                await asyncio.sleep(settings.warming_delay_seconds)

        if not cancelled:
            finalize_batch(db, batch_id)

        logger.info(
            f"[ARQ] Batch {batch_id} {'cancelled' if cancelled else 'completed'}: "
            f"{processed} processed, {failed} failed"
        )
        return {"success": True, "processed": processed, "failed": failed}

    except Exception as e:
        logger.exception(f"[ARQ] Unexpected error processing batch {batch_id}: {e}")
        try:
            now = datetime.utcnow()
            db.execute(
                update(WarmingBatch)
                .where(WarmingBatch.id == batch_id)
                .values(
                    status="completed_with_errors",
                    error_message=str(e)[:500],
                    completed_at=now,
                    updated_at=now,
                )
            )
            db.commit()
        except Exception:
            logger.exception("Failed to update batch status after error")
        return {"success": False, "error": str(e)}
    finally:
        db.close()
