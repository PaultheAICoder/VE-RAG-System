"""Bounded asyncio.Queue with consumer tasks for live RAG query evaluation."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_ready_rag.config import Settings
    from ai_ready_rag.services.evaluation_service import EvaluationService

logger = logging.getLogger(__name__)


class LiveEvaluationQueue:
    """Bounded asyncio.Queue with N consumer tasks scoring live queries via RAGAS."""

    def __init__(self, eval_service: EvaluationService, settings: Settings) -> None:
        self._eval_service = eval_service
        self._settings = settings
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=settings.eval_live_queue_size)
        self._consumers: list[asyncio.Task] = []
        self._shutdown = asyncio.Event()
        self._drops_since_startup = 0
        self._processed_since_startup = 0

    def enqueue(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        model_used: str,
        generation_time_ms: float | None = None,
    ) -> bool:
        """Non-blocking enqueue. Returns False if queue is full (item dropped)."""
        try:
            self._queue.put_nowait(
                {
                    "query": query,
                    "answer": answer,
                    "contexts": contexts,
                    "model_used": model_used,
                    "generation_time_ms": generation_time_ms,
                }
            )
            return True
        except asyncio.QueueFull:
            self._drops_since_startup += 1
            logger.warning(
                "Live eval queue full (capacity=%d), dropping query",
                self._settings.eval_live_queue_size,
            )
            return False

    async def start(self) -> None:
        """Start consumer tasks."""
        self._shutdown.clear()
        for i in range(self._settings.eval_live_max_concurrent):
            task = asyncio.create_task(self._consumer(i))
            self._consumers.append(task)
        logger.info(
            "LiveEvaluationQueue started: %d consumers, capacity %d",
            self._settings.eval_live_max_concurrent,
            self._settings.eval_live_queue_size,
        )

    async def stop(self) -> None:
        """Stop all consumer tasks gracefully."""
        self._shutdown.set()
        for task in self._consumers:
            task.cancel()
        await asyncio.gather(*self._consumers, return_exceptions=True)
        self._consumers.clear()
        logger.info("LiveEvaluationQueue stopped")

    async def _consumer(self, consumer_id: int) -> None:
        """Consumer loop: dequeue items and score them."""
        while not self._shutdown.is_set():
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._eval_service.score_live_query(**item)
                self._processed_since_startup += 1
            except Exception:
                logger.exception("Live eval consumer %d failed to score query", consumer_id)
            finally:
                self._queue.task_done()

    @property
    def depth(self) -> int:
        """Current queue depth."""
        return self._queue.qsize()

    @property
    def capacity(self) -> int:
        """Queue max capacity."""
        return self._settings.eval_live_queue_size

    @property
    def drops_since_startup(self) -> int:
        """Number of items dropped since startup."""
        return self._drops_since_startup

    @property
    def processed_since_startup(self) -> int:
        """Number of items processed since startup."""
        return self._processed_since_startup
