"""Periodic cleanup service for warming jobs and related resources."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

from ai_ready_rag.config import Settings, get_settings
from ai_ready_rag.db.database import SessionLocal
from ai_ready_rag.db.models import WarmingQueue
from ai_ready_rag.services.sse_buffer_service import prune_old_events

logger = logging.getLogger(__name__)


class WarmingCleanupService:
    """Async background service that periodically cleans up warming resources.

    Cleanup operations:
    1. Delete completed jobs older than warming_completed_retention_days
    2. Delete failed/cancelled jobs older than warming_failed_retention_days
    3. Prune SSE event buffer to sse_event_buffer_size
    4. Clean orphaned staging files older than 1 hour
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize cleanup service.

        Args:
            settings: Optional settings override (uses get_settings() if None).
        """
        self.settings = settings or get_settings()
        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

    async def start(self) -> None:
        """Start cleanup service as asyncio task."""
        self._shutdown.clear()
        self._task = asyncio.create_task(self._cleanup_loop())
        logger.info("WarmingCleanupService started")

    async def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("WarmingCleanupService stopping...")
        self._shutdown.set()
        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except (TimeoutError, asyncio.CancelledError):
                pass
        logger.info("WarmingCleanupService stopped")

    async def _cleanup_loop(self) -> None:
        """Run cleanup every warming_cleanup_interval_hours."""
        interval_seconds = self.settings.warming_cleanup_interval_hours * 3600

        while not self._shutdown.is_set():
            try:
                await self._run_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

            # Wait for next interval or shutdown
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=interval_seconds,
                )
                break  # Shutdown requested
            except TimeoutError:
                pass  # Normal timeout, continue loop

    async def _run_cleanup(self) -> None:
        """Perform all cleanup operations."""
        now = datetime.utcnow()

        db = SessionLocal()
        try:
            # 1. Delete old completed jobs
            completed_count = self._delete_old_jobs(
                db,
                status="completed",
                cutoff=now - timedelta(days=self.settings.warming_completed_retention_days),
            )

            # 2. Delete old failed/cancelled jobs
            failed_count = self._delete_old_jobs(
                db,
                status_list=["failed", "cancelled"],
                cutoff=now - timedelta(days=self.settings.warming_failed_retention_days),
            )

            # 3. Prune SSE event buffer
            pruned_events = prune_old_events(db)

            if completed_count or failed_count or pruned_events:
                logger.info(
                    f"Cleanup: deleted {completed_count} completed, "
                    f"{failed_count} failed/cancelled jobs, "
                    f"pruned {pruned_events} SSE events"
                )
        finally:
            db.close()

        # 4. Clean orphaned staging files (older than 1 hour)
        orphan_count = self._clean_orphaned_staging()
        if orphan_count:
            logger.info(f"Cleanup: removed {orphan_count} orphaned staging files")

    def _delete_old_jobs(
        self,
        db,
        status: str | None = None,
        status_list: list[str] | None = None,
        cutoff: datetime | None = None,
    ) -> int:
        """Delete jobs matching criteria and their associated files.

        Args:
            db: Database session.
            status: Single status to filter (optional).
            status_list: List of statuses to filter (optional).
            cutoff: Delete jobs completed before this datetime.

        Returns:
            Number of jobs deleted.
        """
        count = 0

        query = db.query(WarmingQueue).filter(WarmingQueue.completed_at < cutoff)

        if status:
            query = query.filter(WarmingQueue.status == status)
        elif status_list:
            query = query.filter(WarmingQueue.status.in_(status_list))

        jobs = query.all()

        for job in jobs:
            self._delete_job_and_file(db, job.id, job.file_path)
            count += 1

        return count

    def _delete_job_and_file(self, db, job_id: str, file_path: str | None) -> None:
        """Delete a job from DB and its associated query file.

        Args:
            db: Database session.
            job_id: Job ID to delete.
            file_path: Path to the query file (may be None).
        """
        # Delete the query file first (if exists)
        if file_path:
            try:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                    logger.debug(f"Deleted query file: {file_path}")
            except OSError as e:
                logger.warning(f"Failed to delete query file {file_path}: {e}")

        # Delete job from database (CASCADE deletes related failed_queries)
        db.query(WarmingQueue).filter(WarmingQueue.id == job_id).delete()
        db.commit()

    def _clean_orphaned_staging(self) -> int:
        """Clean staging files older than 1 hour.

        Returns:
            Number of files deleted.
        """
        staging_dir = Path("uploads/warming/.staging")
        count = 0

        if not staging_dir.exists():
            return 0

        cutoff_time = time.time() - 3600  # 1 hour ago

        for file_path in staging_dir.iterdir():
            if file_path.is_file():
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        count += 1
                        logger.debug(f"Deleted orphaned staging file: {file_path.name}")
                except OSError:
                    pass  # File may be in use

        return count
