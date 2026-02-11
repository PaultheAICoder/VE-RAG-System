"""Async background worker for DB-first cache warming.

Polls the warming_batches table for pending batches and processes them
using shared helpers from warming_batch.py. All file I/O logic has been
removed in favor of the WarmingBatch/WarmingQuery DB-first architecture.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import update

from ai_ready_rag.config import Settings, get_settings
from ai_ready_rag.db.database import SessionLocal
from ai_ready_rag.db.models.warming import WarmingBatch, WarmingQuery
from ai_ready_rag.workers.tasks.warming_batch import (
    acquire_batch_lease,
    cancel_batch,
    claim_next_query,
    finalize_batch,
    wait_for_resume_or_cancel,
    warm_query_with_retry,
)

logger = logging.getLogger(__name__)


class WarmingWorker:
    """Async background worker that processes warming batches from DB.

    Discovers pending WarmingBatch rows, acquires leases, and delegates
    per-query processing to shared helpers from warming_batch.py.
    """

    def __init__(self, rag_service, settings: Settings | None = None):
        """Initialize worker.

        Args:
            rag_service: RAGService instance for cache warming.
            settings: Optional settings override (uses get_settings() if None).
        """
        self.rag_service = rag_service
        self.settings = settings or get_settings()
        self.worker_id = f"worker-{uuid4().hex[:8]}"
        self._task: asyncio.Task | None = None
        self._lease_task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._current_batch_id: str | None = None

    async def start(self) -> None:
        """Start worker as asyncio task."""
        self._shutdown.clear()
        self._task = asyncio.create_task(self._run_loop())
        self._lease_task = asyncio.create_task(self._lease_renewal_loop())
        logger.info(f"WarmingWorker {self.worker_id} started")

    async def stop(self) -> None:
        """Graceful shutdown."""
        logger.info(f"WarmingWorker {self.worker_id} stopping...")
        self._shutdown.set()

        for task in [self._task, self._lease_task]:
            if task:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=10)
                except (TimeoutError, asyncio.CancelledError):
                    pass

        logger.info(f"WarmingWorker {self.worker_id} stopped")

    async def _run_loop(self) -> None:
        """Main worker loop: discover pending batches, acquire lease, process."""
        while not self._shutdown.is_set():
            try:
                db = SessionLocal()
                try:
                    batch_id = self._find_pending_batch(db)
                finally:
                    db.close()

                if batch_id:
                    # Try to acquire lease
                    db = SessionLocal()
                    try:
                        acquired = acquire_batch_lease(db, batch_id, self.worker_id, self.settings)
                    finally:
                        db.close()

                    if acquired:
                        self._current_batch_id = batch_id
                        logger.info(
                            f"[WARM] Acquired batch {batch_id[:8]}... (worker={self.worker_id})"
                        )
                        try:
                            await self._process_batch(batch_id)
                        except Exception as e:
                            logger.exception(f"[WARM] Batch {batch_id} failed: {e}")
                            db = SessionLocal()
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
                            finally:
                                db.close()
                        finally:
                            self._current_batch_id = None
                    else:
                        logger.debug(f"[WARM] Could not acquire lease on batch {batch_id[:8]}...")
                else:
                    logger.debug(
                        f"[WARM] No batches available, sleeping "
                        f"{self.settings.warming_scan_interval_seconds}s"
                    )
                    await asyncio.sleep(self.settings.warming_scan_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[WARM] Worker loop error: {e}")
                await asyncio.sleep(5)

    def _find_pending_batch(self, db) -> str | None:
        """Find oldest pending or stale-lease batch.

        Returns batch ID or None.
        """
        now = datetime.utcnow()

        batch = (
            db.query(WarmingBatch)
            .filter(
                (WarmingBatch.status == "pending")
                | (
                    (WarmingBatch.status == "running")
                    & (WarmingBatch.worker_lease_expires_at < now)
                )
            )
            .order_by(WarmingBatch.created_at.asc())
            .first()
        )

        return batch.id if batch else None

    async def _process_batch(self, batch_id: str) -> None:
        """Process all queries in a batch using shared helpers.

        Mirrors the process_warming_batch ARQ task logic but runs
        as part of the background worker loop.
        """
        processed = 0
        failed = 0
        cancelled = False

        while not self._shutdown.is_set():
            db = SessionLocal()
            try:
                # Re-read batch for pause/cancel flags
                db.expire_all()
                batch = db.query(WarmingBatch).filter(WarmingBatch.id == batch_id).first()
                if batch is None:
                    logger.error(f"[WARM] Batch {batch_id} disappeared during processing")
                    return

                if batch.is_cancel_requested:
                    cancel_batch(db, batch_id)
                    cancelled = True
                    logger.info(f"[WARM] Batch {batch_id} cancelled")
                    return

                if batch.is_paused:
                    logger.info(f"[WARM] Batch {batch_id} paused, waiting for resume...")
                    result = await wait_for_resume_or_cancel(db, batch_id, self.settings)
                    if result == "cancel":
                        cancel_batch(db, batch_id)
                        cancelled = True
                        logger.info(f"[WARM] Batch {batch_id} cancelled during pause")
                        return
                    continue  # Resume -- re-check state at top of loop

                query_row = claim_next_query(db, batch_id)
                if query_row is None:
                    break  # All queries processed

                logger.info(
                    f"[WARM] Processing query {query_row.sort_order}: "
                    f"{query_row.query_text[:50]}..."
                )
                success = await warm_query_with_retry(
                    self.rag_service, db, query_row, self.settings
                )

                # Check cancel after LLM call — discard result if cancelled
                db.expire_all()
                batch_check = db.query(WarmingBatch).filter(WarmingBatch.id == batch_id).first()
                if batch_check and batch_check.is_cancel_requested:
                    now = datetime.utcnow()
                    db.execute(
                        update(WarmingQuery)
                        .where(
                            WarmingQuery.id == query_row.id,
                            WarmingQuery.status.notin_(["skipped"]),
                        )
                        .values(status="skipped", updated_at=now)
                    )
                    db.commit()
                    cancel_batch(db, batch_id)
                    cancelled = True
                    logger.info(f"[WARM] Batch {batch_id} cancelled after LLM call")
                    return

                if success:
                    processed += 1
                    logger.debug(f"[WARM] Query {query_row.sort_order} completed")
                else:
                    failed += 1
                    logger.warning(f"[WARM] Query {query_row.sort_order} failed")

            finally:
                db.close()

            # Throttle between queries
            if self.settings.warming_delay_seconds > 0:
                await asyncio.sleep(self.settings.warming_delay_seconds)

        if not cancelled:
            db = SessionLocal()
            try:
                finalize_batch(db, batch_id)
            finally:
                db.close()

        logger.info(
            f"[WARM] Batch {batch_id} "
            f"{'cancelled' if cancelled else 'completed'}: "
            f"{processed} processed, {failed} failed"
        )

    async def _lease_renewal_loop(self) -> None:
        """Independent loop that renews lease for current batch.

        Runs every warming_lease_renewal_seconds and extends lease
        if we have an active batch.
        """
        renewal_interval = self.settings.warming_lease_renewal_seconds
        while not self._shutdown.is_set():
            try:
                await asyncio.sleep(renewal_interval)

                if self._current_batch_id:
                    db = SessionLocal()
                    try:
                        renewed = self._renew_batch_lease(db, self._current_batch_id)
                        if renewed:
                            logger.debug(f"[WARM] Lease renewed for batch {self._current_batch_id}")
                        else:
                            logger.warning(
                                f"[WARM] Failed to renew lease for batch {self._current_batch_id}"
                            )
                    finally:
                        db.close()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[WARM] Lease renewal error: {e}")

    def _renew_batch_lease(self, db, batch_id: str) -> bool:
        """Extend lease expiry for active batch.

        Returns True if lease was renewed, False otherwise.
        """
        now = datetime.utcnow()
        new_expiry = now + timedelta(minutes=self.settings.warming_lease_duration_minutes)

        updated = (
            db.query(WarmingBatch)
            .filter(
                WarmingBatch.id == batch_id,
                WarmingBatch.worker_id == self.worker_id,
                WarmingBatch.status == "running",
            )
            .update({"worker_lease_expires_at": new_expiry})
        )
        db.commit()
        return updated > 0


async def recover_stale_batches() -> int:
    """Reset batches that were running when server crashed (expired leases).

    This should be called on server startup.

    Batches with is_cancel_requested=True are cancelled (not re-queued).

    Returns:
        Number of batches recovered.
    """
    db = SessionLocal()
    try:
        now = datetime.utcnow()

        # Cancel stale batches that had cancellation requested
        cancelled_count = (
            db.query(WarmingBatch)
            .filter(
                WarmingBatch.status.in_(["running", "paused"]),
                WarmingBatch.worker_lease_expires_at < now,
                WarmingBatch.is_cancel_requested.is_(True),
            )
            .update(
                {
                    "status": "cancelled",
                    "completed_at": now,
                    "worker_id": None,
                    "worker_lease_expires_at": None,
                }
            )
        )
        if cancelled_count:
            logger.info(f"Cancelled {cancelled_count} stale batches with pending cancel request")

        # Re-queue remaining stale batches (no cancel requested)
        count = (
            db.query(WarmingBatch)
            .filter(
                WarmingBatch.status == "running",
                WarmingBatch.worker_lease_expires_at < now,
            )
            .update(
                {
                    "status": "pending",
                    "worker_id": None,
                    "worker_lease_expires_at": None,
                }
            )
        )

        # Reset orphaned "processing" queries back to "pending" for recovered batches
        if count:
            query_reset_count = (
                db.query(WarmingQuery)
                .filter(
                    WarmingQuery.status == "processing",
                    WarmingQuery.batch_id.in_(
                        db.query(WarmingBatch.id).filter(WarmingBatch.status == "pending")
                    ),
                )
                .update({"status": "pending"}, synchronize_session=False)
            )
            if query_reset_count:
                logger.info(f"Reset {query_reset_count} orphaned processing queries to pending")

        db.commit()
        if count:
            logger.info(f"Recovered {count} warming batches with expired leases")
        return count + cancelled_count
    finally:
        db.close()
