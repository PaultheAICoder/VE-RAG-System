"""Periodic cleanup service for warming batches and related resources."""

import asyncio
import logging
from datetime import datetime, timedelta

from ai_ready_rag.config import Settings, get_settings
from ai_ready_rag.db.database import SessionLocal
from ai_ready_rag.db.models.warming import WarmingBatch

logger = logging.getLogger(__name__)


class WarmingCleanupService:
    """Async background service that periodically cleans up warming resources.

    Cleanup operations:
    1. Delete completed batches older than warming_completed_retention_days
    2. Delete failed/cancelled batches older than warming_failed_retention_days
    3. Prune SSE event buffer to sse_event_buffer_size
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
        from ai_ready_rag.services.sse_buffer_service import prune_old_events

        now = datetime.utcnow()

        db = SessionLocal()
        try:
            # 1. Delete old completed batches (CASCADE deletes WarmingQuery rows)
            completed_count = self._delete_old_batches(
                db,
                status="completed",
                cutoff=now - timedelta(days=self.settings.warming_completed_retention_days),
            )

            # Also clean completed_with_errors using same retention as completed
            completed_errors_count = self._delete_old_batches(
                db,
                status="completed_with_errors",
                cutoff=now - timedelta(days=self.settings.warming_completed_retention_days),
            )

            # 2. Delete old failed/cancelled batches
            failed_count = self._delete_old_batches(
                db,
                status_list=["failed", "cancelled"],
                cutoff=now - timedelta(days=self.settings.warming_failed_retention_days),
            )

            # 3. Prune SSE event buffer
            pruned_events = prune_old_events(db)

            total_completed = completed_count + completed_errors_count
            if total_completed or failed_count or pruned_events:
                logger.info(
                    f"Cleanup: deleted {total_completed} completed, "
                    f"{failed_count} failed/cancelled batches, "
                    f"pruned {pruned_events} SSE events"
                )
        finally:
            db.close()

    def _delete_old_batches(
        self,
        db,
        status: str | None = None,
        status_list: list[str] | None = None,
        cutoff: datetime | None = None,
    ) -> int:
        """Delete batches matching criteria.

        CASCADE on WarmingQuery foreign key automatically deletes
        associated query rows.

        Args:
            db: Database session.
            status: Single status to filter (optional).
            status_list: List of statuses to filter (optional).
            cutoff: Delete batches completed before this datetime.

        Returns:
            Number of batches deleted.
        """
        query = db.query(WarmingBatch).filter(WarmingBatch.completed_at < cutoff)

        if status:
            query = query.filter(WarmingBatch.status == status)
        elif status_list:
            query = query.filter(WarmingBatch.status.in_(status_list))

        count = query.delete(synchronize_session="fetch")
        db.commit()
        return count
