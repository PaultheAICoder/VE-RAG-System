"""Async background worker for evaluation run processing.

Mirrors WarmingWorker pattern: lease-based claiming, heartbeat renewal,
cancellation support, retry policy, and max duration enforcement.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from uuid import uuid4

from ai_ready_rag.config import Settings, get_settings
from ai_ready_rag.db.database import SessionLocal
from ai_ready_rag.db.models.evaluation import EvaluationRun, EvaluationSample
from ai_ready_rag.db.repositories.evaluation import (
    EvaluationRunRepository,
    EvaluationSampleRepository,
)

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, asyncio.TimeoutError, OSError)


class EvaluationWorker:
    """Async background worker that processes evaluation runs.

    Discovers pending EvaluationRun rows, acquires leases, and processes
    samples sequentially through RAG + RAGAS metrics pipeline.
    """

    def __init__(self, eval_service, settings: Settings | None = None):
        self.eval_service = eval_service
        self.settings = settings or get_settings()
        self.worker_id = f"eval-worker-{uuid4().hex[:8]}"
        self._task: asyncio.Task | None = None
        self._lease_task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._current_run_id: str | None = None

    async def start(self) -> None:
        """Start worker as asyncio task."""
        self._shutdown.clear()
        self._task = asyncio.create_task(self._run_loop())
        self._lease_task = asyncio.create_task(self._lease_renewal_loop())
        logger.info(f"EvaluationWorker {self.worker_id} started")

    async def stop(self) -> None:
        """Graceful shutdown."""
        logger.info(f"EvaluationWorker {self.worker_id} stopping...")
        self._shutdown.set()

        for task in [self._task, self._lease_task]:
            if task:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=10)
                except (TimeoutError, asyncio.CancelledError):
                    pass

        logger.info(f"EvaluationWorker {self.worker_id} stopped")

    async def _run_loop(self) -> None:
        """Main worker loop: discover pending runs, acquire lease, process."""
        while not self._shutdown.is_set():
            try:
                # Find claimable run
                db = SessionLocal()
                try:
                    run_repo = EvaluationRunRepository(db)
                    run = run_repo.get_next_claimable()
                    run_id = run.id if run else None
                finally:
                    db.close()

                if run_id:
                    # Try to acquire lease
                    db = SessionLocal()
                    try:
                        run_repo = EvaluationRunRepository(db)
                        acquired = run_repo.claim_run(
                            run_id,
                            self.worker_id,
                            self.settings.eval_lease_duration_minutes,
                        )
                    finally:
                        db.close()

                    if acquired:
                        self._current_run_id = run_id
                        logger.info(
                            "eval.run.started",
                            extra={"run_id": run_id, "worker_id": self.worker_id},
                        )
                        try:
                            await self._process_run(run_id)
                        except Exception as e:
                            logger.exception(f"Evaluation run {run_id} failed: {e}")
                            self._mark_run_failed(run_id, str(e))
                        finally:
                            self._current_run_id = None
                else:
                    await asyncio.sleep(self.settings.eval_scan_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Evaluation worker loop error: {e}")
                await asyncio.sleep(5)

    async def _process_run(self, run_id: str) -> None:
        """Process all samples in an evaluation run."""
        run_start = time.monotonic()
        completed = 0
        failed = 0
        cancelled = False

        while not self._shutdown.is_set():
            db = SessionLocal()
            try:
                db.expire_all()
                run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
                if run is None:
                    logger.error(f"Evaluation run {run_id} disappeared during processing")
                    return

                # Check cancellation
                if run.is_cancel_requested:
                    await self._cancel_run(db, run_id)
                    cancelled = True
                    return

                # Check max duration
                max_hours = run.max_duration_hours or self.settings.eval_max_run_duration_hours
                elapsed_hours = (time.monotonic() - run_start) / 3600
                if elapsed_hours > max_hours:
                    logger.warning(
                        "eval.run.max_duration_exceeded",
                        extra={"run_id": run_id, "elapsed_hours": round(elapsed_hours, 2)},
                    )
                    run.is_cancel_requested = True
                    db.commit()
                    await self._cancel_run(db, run_id)
                    cancelled = True
                    return

                # Get next pending sample
                sample_repo = EvaluationSampleRepository(db)
                pending = sample_repo.get_pending_samples(run_id)
                if not pending:
                    break  # All samples processed

                sample = pending[0]
                if not sample_repo.claim_sample(sample.id):
                    continue  # Another worker claimed it

                # Process with retry
                success = await self._process_with_retry(db, sample, run)

                # Update run counters
                db.expire_all()
                run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
                if run:
                    if success:
                        run.completed_samples += 1
                        completed += 1
                    else:
                        run.failed_samples += 1
                        failed += 1
                    db.commit()

            finally:
                db.close()

            # Throttle between samples
            if self.settings.eval_delay_between_samples_seconds > 0:
                await asyncio.sleep(self.settings.eval_delay_between_samples_seconds)

        # Finalize run
        if not cancelled:
            db = SessionLocal()
            try:
                run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
                if run:
                    await self.eval_service.compute_aggregates(db, run)

                    if run.failed_samples > 0:
                        run.status = "completed_with_errors"
                    else:
                        run.status = "completed"
                    run.completed_at = datetime.utcnow()
                    run.worker_id = None
                    run.worker_lease_expires_at = None
                    db.commit()

                    logger.info(
                        "eval.run.completed",
                        extra={
                            "run_id": run_id,
                            "status": run.status,
                            "completed": completed,
                            "failed": failed,
                        },
                    )
            finally:
                db.close()

    async def _process_with_retry(
        self,
        db,
        sample: EvaluationSample,
        run: EvaluationRun,
    ) -> bool:
        """Process a sample with retry on transient errors."""
        max_attempts = self.settings.eval_max_retries_per_sample + 1
        tag_scope = json.loads(run.tag_scope) if run.tag_scope else None

        for attempt in range(max_attempts):
            try:
                async with asyncio.timeout(self.settings.eval_sample_deadline_seconds):
                    sample = await self.eval_service.process_sample(
                        db, sample, tag_scope, run.admin_bypass_tags
                    )
                db.commit()

                if sample.status == "completed":
                    logger.info(
                        "eval.sample.scored",
                        extra={
                            "run_id": run.id,
                            "sample_id": sample.id,
                            "faithfulness": sample.faithfulness,
                            "answer_relevancy": sample.answer_relevancy,
                        },
                    )
                    return True
                else:
                    logger.warning(
                        "eval.sample.failed",
                        extra={
                            "run_id": run.id,
                            "sample_id": sample.id,
                            "error": sample.error_message,
                        },
                    )
                    return False

            except RETRYABLE_EXCEPTIONS as e:
                is_last = attempt >= max_attempts - 1
                if is_last:
                    sample.status = "failed"
                    sample.error_message = str(e)
                    sample.error_type = type(e).__name__
                    sample.processed_at = datetime.utcnow()
                    db.commit()
                    logger.warning(
                        "eval.sample.failed",
                        extra={
                            "run_id": run.id,
                            "sample_id": sample.id,
                            "error": str(e),
                            "retries_exhausted": True,
                        },
                    )
                    return False
                else:
                    sample.retry_count += 1
                    sample.status = "pending"
                    db.commit()
                    logger.warning(
                        "eval.sample.retried",
                        extra={
                            "run_id": run.id,
                            "sample_id": sample.id,
                            "attempt": attempt + 1,
                            "error": str(e),
                        },
                    )
                    await asyncio.sleep(self.settings.eval_retry_backoff_seconds)
                    # Re-claim sample
                    sample_repo = EvaluationSampleRepository(db)
                    if not sample_repo.claim_sample(sample.id):
                        return False

            except Exception as e:
                sample.status = "failed"
                sample.error_message = str(e)
                sample.error_type = type(e).__name__
                sample.processed_at = datetime.utcnow()
                db.commit()
                logger.warning(
                    "eval.sample.failed",
                    extra={
                        "run_id": run.id,
                        "sample_id": sample.id,
                        "error": str(e),
                        "non_retryable": True,
                    },
                )
                return False

        return False

    async def _cancel_run(self, db, run_id: str) -> None:
        """Cancel a run: skip remaining samples, compute partial aggregates."""
        sample_repo = EvaluationSampleRepository(db)
        skipped = sample_repo.skip_remaining(run_id)

        run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        if run:
            await self.eval_service.compute_aggregates(db, run)
            run.status = "cancelled"
            run.completed_at = datetime.utcnow()
            run.worker_id = None
            run.worker_lease_expires_at = None
            db.commit()

            logger.info(
                "eval.run.cancelled",
                extra={"run_id": run_id, "skipped_samples": skipped},
            )

    def _mark_run_failed(self, run_id: str, error: str) -> None:
        """Mark a run as failed due to unhandled error."""
        db = SessionLocal()
        try:
            now = datetime.utcnow()
            db.query(EvaluationRun).filter(EvaluationRun.id == run_id).update(
                {
                    "status": "failed",
                    "error_message": error[:500],
                    "completed_at": now,
                    "worker_id": None,
                    "worker_lease_expires_at": None,
                    "updated_at": now,
                }
            )
            db.commit()
        except Exception:
            logger.exception("Failed to update run status after error")
        finally:
            db.close()

    async def _lease_renewal_loop(self) -> None:
        """Independent loop that renews lease for current run."""
        renewal_interval = self.settings.eval_lease_renewal_seconds
        while not self._shutdown.is_set():
            try:
                await asyncio.sleep(renewal_interval)

                if self._current_run_id:
                    db = SessionLocal()
                    try:
                        run_repo = EvaluationRunRepository(db)
                        renewed = run_repo.renew_lease(
                            self._current_run_id,
                            self.worker_id,
                            self.settings.eval_lease_duration_minutes,
                        )
                        if renewed:
                            logger.debug(
                                "eval.lease.renewed",
                                extra={"run_id": self._current_run_id},
                            )
                        else:
                            logger.warning(f"Failed to renew lease for run {self._current_run_id}")
                    finally:
                        db.close()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Evaluation lease renewal error: {e}")


async def recover_stale_evaluation_runs(db=None) -> int:
    """Reset evaluation runs with expired leases on server startup.

    Runs with is_cancel_requested=True are cancelled (not re-queued).
    Orphaned 'processing' samples are reset to 'pending'.

    Args:
        db: Optional session (creates own if None).

    Returns number of runs recovered.
    """
    own_session = db is None
    if own_session:
        db = SessionLocal()
    try:
        now = datetime.utcnow()

        # Cancel stale runs that had cancellation requested
        cancelled_count = (
            db.query(EvaluationRun)
            .filter(
                EvaluationRun.status == "running",
                EvaluationRun.worker_lease_expires_at < now,
                EvaluationRun.is_cancel_requested.is_(True),
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
            logger.warning(
                "eval.lease.stale_recovered",
                extra={"cancelled": cancelled_count, "action": "cancelled"},
            )

        # Re-queue remaining stale runs
        count = (
            db.query(EvaluationRun)
            .filter(
                EvaluationRun.status == "running",
                EvaluationRun.worker_lease_expires_at < now,
            )
            .update(
                {
                    "status": "pending",
                    "worker_id": None,
                    "worker_lease_expires_at": None,
                }
            )
        )

        # Reset orphaned 'processing' samples
        if count:
            sample_reset = (
                db.query(EvaluationSample)
                .filter(
                    EvaluationSample.status == "processing",
                    EvaluationSample.run_id.in_(
                        db.query(EvaluationRun.id).filter(EvaluationRun.status == "pending")
                    ),
                )
                .update({"status": "pending"}, synchronize_session=False)
            )
            if sample_reset:
                logger.warning(
                    "eval.lease.stale_recovered",
                    extra={"orphaned_samples_reset": sample_reset},
                )

        db.commit()
        if count:
            logger.warning(
                "eval.lease.stale_recovered",
                extra={"requeued": count, "action": "requeued"},
            )
        return count + cancelled_count
    finally:
        if own_session:
            db.close()
