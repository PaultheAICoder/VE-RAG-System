"""Embedded ARQ worker that runs inside the FastAPI lifespan.

Instead of running `arq` as a separate CLI process, this embeds the worker
as an asyncio.Task managed by FastAPI's startup/shutdown lifecycle.  This
avoids a separate process, shares the existing Redis pool and VectorService,
and sidesteps the circular-import issue that blocks the standalone worker.
"""

import asyncio
import logging
import signal

from arq.connections import ArqRedis
from arq.worker import Worker

from ai_ready_rag.config import Settings
from ai_ready_rag.workers.tasks import (
    process_document,
    process_warming_batch,
    reindex_knowledge_base,
)

logger = logging.getLogger(__name__)


class EmbeddedArqWorker:
    """Wraps an ARQ Worker so it runs as a background task inside FastAPI.

    Key differences from the standalone ``arq`` CLI worker:
    * ``handle_signals=False`` — uvicorn owns process signals.
    * Accepts a **shared** ``redis_pool`` — avoids creating a second connection.
    * Custom ``stop()`` that does **not** close the shared pool.
    """

    def __init__(
        self,
        redis_pool: ArqRedis,
        settings: Settings,
        vector_service: object,
    ) -> None:
        self._redis_pool = redis_pool
        self._settings = settings
        self._vector_service = vector_service
        self._task: asyncio.Task | None = None
        self._worker: Worker | None = None

    async def start(self) -> None:
        """Create the ARQ Worker and launch its poll loop as an asyncio task."""
        self._worker = Worker(
            functions=[process_document, reindex_knowledge_base, process_warming_batch],
            redis_pool=self._redis_pool,
            max_jobs=self._settings.arq_max_jobs,
            job_timeout=self._settings.arq_job_timeout,
            health_check_interval=self._settings.arq_health_check_interval,
            handle_signals=False,
            ctx={
                "settings": self._settings,
                "vector_service": self._vector_service,
            },
        )
        self._task = asyncio.create_task(self._run(), name="arq-worker")
        # Wire up main_task so handle_sig() can cancel the poll loop.
        # Worker.main_task is only set by run()/async_run(), not by main().
        self._worker.main_task = self._task
        logger.info("EmbeddedArqWorker started")

    async def _run(self) -> None:
        """Run worker.main() and log unexpected exits."""
        try:
            await self._worker.main()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("EmbeddedArqWorker crashed")

    async def stop(self) -> None:
        """Gracefully stop the worker without closing the shared Redis pool.

        ``Worker.close()`` calls ``pool.close(close_connection_pool=True)``
        which would destroy the pool shared with the rest of the app.  Instead
        we manually trigger shutdown, wait for in-flight jobs, and clean up.
        """
        if self._worker is None:
            return

        logger.info("EmbeddedArqWorker stopping...")

        # Signal the worker to stop (same mechanism as SIGUSR1 handler)
        self._worker.handle_sig(signal.SIGUSR1)

        # Wait for the main task to finish (gives in-flight jobs time to complete)
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=30)
            except (TimeoutError, asyncio.CancelledError):
                logger.warning("EmbeddedArqWorker did not stop within timeout")

        # Gather any remaining job tasks
        if self._worker.tasks:
            await asyncio.gather(*self._worker.tasks.values(), return_exceptions=True)

        # Clean up health check key without closing the pool
        try:
            await self._redis_pool.delete(self._worker.health_check_key)
        except Exception:
            pass

        self._worker = None
        self._task = None
        logger.info("EmbeddedArqWorker stopped")
