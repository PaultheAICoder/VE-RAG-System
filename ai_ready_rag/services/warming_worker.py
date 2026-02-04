"""Async background worker for cache warming queue.

Processes WarmingQueue jobs from database with:
- Job lease acquisition and renewal
- aiofiles streaming for large query files
- Batch checkpoints for crash recovery
- Pause/cancel detection from DB flags
- Retry logic with configurable delays
- EMA-based progress estimation
"""

import asyncio
import logging
import os
from collections import deque
from datetime import datetime, timedelta
from uuid import uuid4

import aiofiles

from ai_ready_rag.config import Settings, get_settings
from ai_ready_rag.core.exceptions import (
    ConnectionTimeoutError,
    RateLimitExceededError,
    ServiceUnavailableError,
    WarmingCancelledException,
)
from ai_ready_rag.db.database import SessionLocal
from ai_ready_rag.db.models import WarmingFailedQuery, WarmingQueue
from ai_ready_rag.services.sse_buffer_service import store_sse_event

logger = logging.getLogger(__name__)

# Retryable exceptions (class references for isinstance checks)
RETRYABLE_EXCEPTIONS = (
    ConnectionTimeoutError,
    ServiceUnavailableError,
    RateLimitExceededError,
    asyncio.TimeoutError,  # Added: Timeouts are retryable
)


class WarmingWorker:
    """Async background worker that processes warming queue."""

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
        self._current_job_id: str | None = None

        # Progress estimation (EMA)
        self._query_durations: deque[float] = deque(maxlen=20)

        # Parse retry delays from settings
        self._retry_delays = [int(d.strip()) for d in self.settings.warming_retry_delays.split(",")]

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

    async def _lease_renewal_loop(self) -> None:
        """Independent loop that renews lease for current job.

        Runs every warming_lease_renewal_seconds and extends lease
        if we have an active job.
        """
        renewal_interval = self.settings.warming_lease_renewal_seconds
        while not self._shutdown.is_set():
            try:
                await asyncio.sleep(renewal_interval)

                if self._current_job_id:
                    db = SessionLocal()
                    try:
                        renewed = self._renew_lease(db, self._current_job_id)
                        if renewed:
                            logger.debug(f"[WARM] Lease renewed for job {self._current_job_id}")
                        else:
                            logger.warning(
                                f"[WARM] Failed to renew lease for job {self._current_job_id}"
                            )
                    finally:
                        db.close()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[WARM] Lease renewal error: {e}")

    async def _run_loop(self) -> None:
        """Main worker loop: acquire job, process, release."""
        while not self._shutdown.is_set():
            try:
                # Try to acquire a job
                db = SessionLocal()
                try:
                    job = self._acquire_job_lease(db)
                finally:
                    db.close()

                if job:
                    self._current_job_id = job.id
                    logger.info(f"[WARM] Acquired job {job.id} ({job.total_queries} queries)")
                    print(f"[WARM] Acquired job {job.id} ({job.total_queries} queries)", flush=True)

                    try:
                        await self._process_job(job)
                    except WarmingCancelledException:
                        logger.info(f"[WARM] Job {job.id} cancelled")
                        db = SessionLocal()
                        try:
                            self._mark_job_cancelled(db, job.id)
                            # Delete query file after graceful shutdown
                            if job.file_path and os.path.exists(job.file_path):
                                os.remove(job.file_path)
                                logger.info(f"[WARM] Deleted query file for cancelled job {job.id}")
                        finally:
                            db.close()
                    except Exception as e:
                        logger.exception(f"[WARM] Job {job.id} failed: {e}")
                        db = SessionLocal()
                        try:
                            self._mark_job_failed(db, job.id, str(e))
                        finally:
                            db.close()
                    finally:
                        self._current_job_id = None
                        # Release lease
                        db = SessionLocal()
                        try:
                            self._release_lease(db, job.id)
                        finally:
                            db.close()
                else:
                    # No job available, wait before checking again
                    await asyncio.sleep(self.settings.warming_scan_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[WARM] Worker loop error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    def _acquire_job_lease(self, db) -> WarmingQueue | None:
        """Acquire lease on the oldest pending job.

        Uses SELECT ... FOR UPDATE SKIP LOCKED pattern (simulated for SQLite).
        SQLite doesn't support FOR UPDATE, so we use immediate UPDATE + check.

        Args:
            db: Database session.

        Returns:
            WarmingQueue job if acquired, None otherwise.
        """
        now = datetime.utcnow()
        lease_expires = now + timedelta(minutes=self.settings.warming_lease_duration_minutes)

        # Find oldest pending job or job with expired lease
        job = (
            db.query(WarmingQueue)
            .filter(
                (WarmingQueue.status == "pending")
                | (
                    (WarmingQueue.status == "running")
                    & (WarmingQueue.worker_lease_expires_at < now)
                )
            )
            .order_by(WarmingQueue.created_at)
            .first()
        )

        if not job:
            return None

        # Try to acquire the lease atomically
        updated = (
            db.query(WarmingQueue)
            .filter(
                WarmingQueue.id == job.id,
                (WarmingQueue.worker_id.is_(None)) | (WarmingQueue.worker_lease_expires_at < now),
            )
            .update(
                {
                    "worker_id": self.worker_id,
                    "worker_lease_expires_at": lease_expires,
                    "status": "running",
                    "started_at": now if job.started_at is None else job.started_at,
                }
            )
        )
        db.commit()

        if updated:
            db.refresh(job)
            return job
        return None

    def _renew_lease(self, db, job_id: str) -> bool:
        """Extend lease expiry for active job.

        Args:
            db: Database session.
            job_id: Job ID to renew.

        Returns:
            True if lease was renewed, False otherwise.
        """
        now = datetime.utcnow()
        new_expiry = now + timedelta(minutes=self.settings.warming_lease_duration_minutes)

        updated = (
            db.query(WarmingQueue)
            .filter(
                WarmingQueue.id == job_id,
                WarmingQueue.worker_id == self.worker_id,
                WarmingQueue.status == "running",
            )
            .update({"worker_lease_expires_at": new_expiry})
        )
        db.commit()
        return updated > 0

    def _release_lease(self, db, job_id: str) -> None:
        """Clear lease on job completion/failure.

        Args:
            db: Database session.
            job_id: Job ID to release.
        """
        (
            db.query(WarmingQueue)
            .filter(WarmingQueue.id == job_id, WarmingQueue.worker_id == self.worker_id)
            .update({"worker_id": None, "worker_lease_expires_at": None})
        )
        db.commit()

    def _mark_job_completed(self, db, job_id: str) -> None:
        """Mark job as completed."""
        (
            db.query(WarmingQueue)
            .filter(WarmingQueue.id == job_id)
            .update({"status": "completed", "completed_at": datetime.utcnow()})
        )
        db.commit()

    def _mark_job_failed(self, db, job_id: str, error: str) -> None:
        """Mark job as failed with error message."""
        (
            db.query(WarmingQueue)
            .filter(WarmingQueue.id == job_id)
            .update(
                {
                    "status": "failed",
                    "error_message": error[:1000],
                    "completed_at": datetime.utcnow(),
                }
            )
        )
        db.commit()

        # Emit job_failed SSE event
        store_sse_event(
            db,
            "job_failed",
            job_id,
            {
                "job_id": job_id,
                "error": error[:500],
            },
        )

    def _mark_job_cancelled(self, db, job_id: str) -> None:
        """Mark job as cancelled."""
        (
            db.query(WarmingQueue)
            .filter(WarmingQueue.id == job_id)
            .update({"status": "cancelled", "completed_at": datetime.utcnow()})
        )
        db.commit()

    async def _process_job(self, job: WarmingQueue) -> None:
        """Process a warming job by streaming queries from file.

        Args:
            job: WarmingQueue job to process.

        Raises:
            WarmingCancelledException: If job was cancelled.
        """
        processed = job.processed_queries
        failed = job.failed_queries
        skipped = 0  # Queries processed but not cached (low confidence)
        byte_offset = job.byte_offset
        line_number = 0
        checkpoint_count = 0
        last_checkpoint_time = asyncio.get_event_loop().time()

        logger.info(
            f"[WARM] Processing job {job.id}: {job.total_queries} queries, "
            f"resuming from offset {byte_offset}"
        )

        # Emit job_started event
        db = SessionLocal()
        try:
            store_sse_event(
                db,
                "job_started",
                job.id,
                {
                    "job_id": job.id,
                    "file_path": job.file_path,
                    "total_queries": job.total_queries,
                    "worker_id": self.worker_id,
                },
            )
        finally:
            db.close()

        async with aiofiles.open(job.file_path, encoding="utf-8") as f:
            # Seek to byte offset for resume
            if byte_offset > 0:
                await f.seek(byte_offset)

            async for line in f:
                # Track byte position
                current_offset = await f.tell()
                line_number += 1

                # Skip empty lines and comments
                query = line.strip()
                if not query or query.startswith("#"):
                    continue

                # Check for pause/cancel signals
                db = SessionLocal()
                try:
                    should_stop, reason = self._should_stop(db, job.id)
                    if should_stop:
                        # Save checkpoint before stopping
                        self._checkpoint(db, job.id, processed, failed, current_offset)
                        if reason == "cancelled":
                            # Emit job_cancelled event
                            store_sse_event(
                                db,
                                "job_cancelled",
                                job.id,
                                {
                                    "job_id": job.id,
                                    "processed": processed,
                                    "total": job.total_queries,
                                    "file_deleted": False,
                                },
                            )
                            raise WarmingCancelledException()
                        elif reason == "paused":
                            # Emit job_paused event
                            store_sse_event(
                                db,
                                "job_paused",
                                job.id,
                                {
                                    "job_id": job.id,
                                    "processed": processed,
                                    "total": job.total_queries,
                                },
                            )
                            logger.info(f"[WARM] Job {job.id} paused, waiting for resume...")
                            await self._wait_for_resume(db, job.id)
                finally:
                    db.close()

                # Warm the query with cancel/pause support
                print(f"[WARM] Processing query {line_number}: {query[:50]}...", flush=True)
                start_time = asyncio.get_event_loop().time()
                success, was_cached = await self._warm_query_with_cancel_check(
                    query, job.id, line_number
                )
                duration = asyncio.get_event_loop().time() - start_time
                self._query_durations.append(duration)

                if success:
                    processed += 1
                    if not was_cached:
                        skipped += 1  # Processed but not cached (low confidence)
                        print(
                            f"[WARM] Query {line_number} SKIPPED (low confidence) "
                            f"in {duration:.2f}s",
                            flush=True,
                        )
                    else:
                        print(
                            f"[WARM] Query {line_number} CACHED in {duration:.2f}s",
                            flush=True,
                        )
                else:
                    failed += 1
                    print(f"[WARM] Query {line_number} FAILED in {duration:.2f}s", flush=True)

                checkpoint_count += 1

                # Checkpoint every N queries or T seconds
                now = asyncio.get_event_loop().time()
                should_checkpoint = (
                    checkpoint_count >= self.settings.warming_checkpoint_interval
                    or (now - last_checkpoint_time) >= self.settings.warming_checkpoint_time_seconds
                )

                if should_checkpoint:
                    db = SessionLocal()
                    try:
                        self._checkpoint(db, job.id, processed, failed, current_offset)
                        # Emit progress event at checkpoints
                        percent = (
                            int((processed + failed) / job.total_queries * 100)
                            if job.total_queries > 0
                            else 0
                        )
                        store_sse_event(
                            db,
                            "progress",
                            job.id,
                            {
                                "job_id": job.id,
                                "processed": processed,
                                "failed": failed,
                                "skipped": skipped,  # Processed but not cached (low confidence)
                                "total": job.total_queries,
                                "percent": percent,
                                "estimated_remaining_seconds": self._estimate_remaining(job),
                                "queries_per_second": round(self._calculate_qps(), 2),
                            },
                        )
                    finally:
                        db.close()
                    checkpoint_count = 0
                    last_checkpoint_time = now

                # Delay between queries to reduce contention
                await asyncio.sleep(self.settings.warming_delay_seconds)

        # Final checkpoint and mark completed
        db = SessionLocal()
        try:
            self._checkpoint(db, job.id, processed, failed, 0)  # Reset offset on completion
            self._mark_job_completed(db, job.id)

            # Emit job_completed event
            # Calculate final duration and QPS
            job_refreshed = db.query(WarmingQueue).filter(WarmingQueue.id == job.id).first()
            duration_seconds = 0
            if job_refreshed and job_refreshed.started_at and job_refreshed.completed_at:
                duration_seconds = int(
                    (job_refreshed.completed_at - job_refreshed.started_at).total_seconds()
                )
            final_qps = processed / duration_seconds if duration_seconds > 0 else 0.0

            store_sse_event(
                db,
                "job_completed",
                job.id,
                {
                    "job_id": job.id,
                    "processed": processed,
                    "failed": failed,
                    "skipped": skipped,  # Processed but not cached (low confidence)
                    "total": job.total_queries,
                    "duration_seconds": duration_seconds,
                    "queries_per_second": round(final_qps, 2),
                },
            )
        finally:
            db.close()

        logger.info(
            f"[WARM] Job {job.id} completed: {processed} processed "
            f"({skipped} skipped due to low confidence), {failed} failed"
        )
        print(
            f"[WARM] Job {job.id} COMPLETED: {processed} processed "
            f"({skipped} skipped due to low confidence), {failed} failed",
            flush=True,
        )

    def _should_stop(self, db, job_id: str) -> tuple[bool, str | None]:
        """Check if job should stop due to pause/cancel flags.

        Also validates this worker still owns the job lease.

        Args:
            db: Database session.
            job_id: Job ID to check.

        Returns:
            Tuple of (should_stop, reason) where reason is 'paused', 'cancelled', or None.
        """
        # Verify this worker still owns the job lease
        job = (
            db.query(WarmingQueue)
            .filter(
                WarmingQueue.id == job_id,
                WarmingQueue.worker_id == self.worker_id,  # Verify ownership
            )
            .first()
        )

        if not job:
            # Job deleted or lease lost to another worker
            logger.warning(f"[WARM] Job {job_id} lease lost or deleted")
            return True, "cancelled"

        if job.is_cancel_requested:
            return True, "cancelled"
        if job.is_paused:
            return True, "paused"
        return False, None

    async def _wait_for_resume(self, db, job_id: str) -> None:
        """Wait for job to be resumed or cancelled.

        Args:
            db: Database session (caller must close).
            job_id: Job ID to monitor.

        Raises:
            WarmingCancelledException: If job was cancelled while paused.
        """
        while True:
            await asyncio.sleep(2)

            # Need fresh session for each check
            check_db = SessionLocal()
            try:
                job = check_db.query(WarmingQueue).filter(WarmingQueue.id == job_id).first()
                if not job:
                    raise WarmingCancelledException()
                if job.is_cancel_requested:
                    raise WarmingCancelledException()
                if not job.is_paused:
                    logger.info(f"[WARM] Job {job_id} resumed")
                    return
            finally:
                check_db.close()

    async def _interruptible_sleep(self, seconds: float, job_id: str) -> None:
        """Sleep with periodic cancel checks (every 2 seconds).

        Args:
            seconds: Total time to sleep.
            job_id: Job ID for cancel checks.

        Raises:
            WarmingCancelledException: If cancel requested during sleep.
        """
        check_interval = 2.0
        remaining = seconds

        while remaining > 0:
            sleep_time = min(check_interval, remaining)
            await asyncio.sleep(sleep_time)
            remaining -= sleep_time

            db = SessionLocal()
            try:
                should_stop, reason = self._should_stop(db, job_id)
                if should_stop and reason == "cancelled":
                    logger.info(f"[WARM] Cancel detected during retry sleep for job {job_id}")
                    raise WarmingCancelledException()
                # Note: Don't raise for pause - let main loop handle
            finally:
                db.close()

    async def _warm_query_with_cancel_check(
        self, query: str, job_id: str, line_number: int
    ) -> tuple[bool, bool]:
        """Wrap warm_query_with_retry with cancel timeout support.

        Cancel: 5 second timeout, abandon current query
        Pause: No timeout, wait for completion (graceful)

        Args:
            query: Query string to warm.
            job_id: Job ID for tracking.
            line_number: Line number in source file.

        Returns:
            Tuple of (success, was_cached):
            - (True, True) = query warmed and cached
            - (True, False) = query processed but not cached (low confidence)
            - (False, False) = query failed

        Raises:
            WarmingCancelledException: If cancelled and timeout exceeded.
        """
        cancel_timeout = self.settings.warming_cancel_timeout_seconds
        start_time = asyncio.get_event_loop().time()

        # Start the warming task
        task = asyncio.create_task(self._warm_query_with_retry(query, job_id, line_number))

        try:
            # Check for cancel flag periodically while task runs
            while not task.done():
                db = SessionLocal()
                try:
                    should_stop, reason = self._should_stop(db, job_id)
                    if should_stop and reason == "cancelled":
                        # Cancel requested - wait up to cancel_timeout then abandon
                        cancel_requested_time = asyncio.get_event_loop().time()
                        logger.info(
                            f"[WARM] Cancel detected for query {line_number}, "
                            f"waiting up to {cancel_timeout}s for completion"
                        )
                        try:
                            result = await asyncio.wait_for(
                                asyncio.shield(task),
                                timeout=cancel_timeout,
                            )
                            # Query completed within timeout - record cancel latency
                            cancel_latency = asyncio.get_event_loop().time() - cancel_requested_time
                            duration = asyncio.get_event_loop().time() - start_time
                            logger.info(
                                f"[WARM] Query {line_number} completed despite cancel "
                                f"(latency {cancel_latency:.2f}s, duration {duration:.2f}s)"
                            )
                            return result
                        except TimeoutError:
                            task.cancel()
                            cancel_latency = asyncio.get_event_loop().time() - cancel_requested_time
                            logger.info(
                                f"[WARM] Query {line_number} abandoned after cancel timeout "
                                f"({cancel_timeout}s, latency {cancel_latency:.2f}s)"
                            )
                            print(
                                f"[WARM] Query {line_number} CANCELLED (timeout)",
                                flush=True,
                            )
                            self._record_failed_query(
                                job_id,
                                query,
                                line_number,
                                "Cancelled by user",
                                "CancelledError",
                                0,
                            )
                            raise WarmingCancelledException() from None
                    elif should_stop and reason == "paused":
                        # Pause requested - wait for completion (graceful)
                        logger.info(
                            f"[WARM] Pause requested, waiting for query {line_number} to complete"
                        )
                        result = await task
                        duration = asyncio.get_event_loop().time() - start_time
                        logger.info(
                            f"[WARM] Query {line_number} completed before pause in {duration:.2f}s"
                        )
                        return result
                finally:
                    db.close()

                # Poll every 1 second
                try:
                    result = await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
                    duration = asyncio.get_event_loop().time() - start_time
                    logger.info(f"[WARM] Query {line_number} completed in {duration:.2f}s")
                    return result
                except TimeoutError:
                    continue  # Task still running, check flags again

            # Task completed normally
            result = await task
            duration = asyncio.get_event_loop().time() - start_time
            logger.info(f"[WARM] Query {line_number} completed in {duration:.2f}s")
            return result
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def _warm_query_with_retry(
        self, query: str, job_id: str, line_number: int
    ) -> tuple[bool, bool]:
        """Warm a single query with retry logic.

        Args:
            query: Query string to warm.
            job_id: Job ID for error recording.
            line_number: Line number in source file.

        Returns:
            Tuple of (success, was_cached):
            - (True, True) = query warmed and cached
            - (True, False) = query processed but not cached (low confidence)
            - (False, False) = query failed
        """
        max_retries = self.settings.warming_max_retries

        for attempt in range(max_retries + 1):
            try:
                was_cached = await self.rag_service.warm_cache(query)
                return (True, was_cached)
            except RETRYABLE_EXCEPTIONS as e:
                if attempt < max_retries:
                    delay = self._retry_delays[min(attempt, len(self._retry_delays) - 1)]
                    logger.warning(
                        f"[WARM] Retry {attempt + 1}/{max_retries} for query "
                        f"(line {line_number}): {e}, waiting {delay}s"
                    )
                    # Use interruptible sleep to allow cancel during retry delay
                    await self._interruptible_sleep(delay, job_id)
                else:
                    # Max retries exhausted
                    self._record_failed_query(
                        job_id, query, line_number, str(e), type(e).__name__, attempt + 1
                    )
                    return (False, False)
            except Exception as e:
                # Non-retryable error - fail immediately
                self._record_failed_query(
                    job_id, query, line_number, str(e), type(e).__name__, attempt + 1
                )
                return (False, False)

        return (False, False)

    def _record_failed_query(
        self,
        job_id: str,
        query: str,
        line_number: int,
        error_message: str,
        error_type: str,
        retry_count: int,
    ) -> None:
        """Record a failed query in the database.

        Args:
            job_id: Parent job ID.
            query: The failed query.
            line_number: Line number in source file.
            error_message: Error message.
            error_type: Exception class name.
            retry_count: Number of retries attempted.
        """
        db = SessionLocal()
        try:
            failed_query = WarmingFailedQuery(
                job_id=job_id,
                query=query[:1000],  # Truncate long queries
                line_number=line_number,
                error_message=error_message[:500],
                error_type=error_type,
                retry_count=retry_count,
            )
            db.add(failed_query)
            db.commit()

            # Emit query_failed SSE event
            store_sse_event(
                db,
                "query_failed",
                job_id,
                {
                    "job_id": job_id,
                    "query": query[:200],  # Truncate for SSE event
                    "line_number": line_number,
                    "error": error_message[:200],
                    "error_type": error_type,
                },
            )
        except Exception as e:
            logger.error(f"[WARM] Failed to record failed query: {e}")
        finally:
            db.close()

    def _checkpoint(self, db, job_id: str, processed: int, failed: int, byte_offset: int) -> None:
        """Save progress checkpoint to database.

        Args:
            db: Database session.
            job_id: Job ID to update.
            processed: Number of queries processed.
            failed: Number of queries failed.
            byte_offset: Current file position.
        """
        (
            db.query(WarmingQueue)
            .filter(WarmingQueue.id == job_id)
            .update(
                {
                    "processed_queries": processed,
                    "failed_queries": failed,
                    "byte_offset": byte_offset,
                }
            )
        )
        db.commit()

    def _estimate_remaining(self, job: WarmingQueue) -> int:
        """Estimate remaining time in seconds using EMA.

        Args:
            job: Current job with progress info.

        Returns:
            Estimated seconds remaining, or -1 if cannot estimate.
        """
        if not self._query_durations:
            return -1

        remaining_queries = job.total_queries - job.processed_queries - job.failed_queries
        if remaining_queries <= 0:
            return 0

        # EMA: weight recent queries more heavily
        alpha = 0.3
        avg_duration = self._query_durations[0]
        for duration in list(self._query_durations)[1:]:
            avg_duration = alpha * duration + (1 - alpha) * avg_duration

        # Add delay between queries
        avg_duration += self.settings.warming_delay_seconds

        return int(remaining_queries * avg_duration)

    def _calculate_qps(self) -> float:
        """Calculate queries per second from recent durations.

        Returns:
            Queries per second, or 0.0 if no data.
        """
        if not self._query_durations:
            return 0.0

        total_time = sum(self._query_durations)
        if total_time <= 0:
            return 0.0

        return len(self._query_durations) / total_time


async def recover_stale_jobs() -> int:
    """Reset jobs that were running when server crashed (expired leases).

    This should be called on server startup.

    Returns:
        Number of jobs recovered.
    """
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        count = (
            db.query(WarmingQueue)
            .filter(
                WarmingQueue.status == "running",
                WarmingQueue.worker_lease_expires_at < now,
            )
            .update(
                {
                    "status": "pending",
                    "worker_id": None,
                    "worker_lease_expires_at": None,
                }
            )
        )
        db.commit()
        if count:
            logger.info(f"Recovered {count} warming jobs with expired leases")
        return count
    finally:
        db.close()
