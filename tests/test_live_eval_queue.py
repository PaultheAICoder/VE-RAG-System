"""Tests for LiveEvaluationQueue — bounded asyncio.Queue with consumer tasks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_ready_rag.workers.live_eval_queue import LiveEvaluationQueue


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.eval_live_max_concurrent = 2
    settings.eval_live_queue_size = 3
    return settings


@pytest.fixture
def mock_eval_service():
    service = MagicMock()
    service.score_live_query = AsyncMock()
    return service


@pytest.fixture
def queue(mock_eval_service, mock_settings):
    return LiveEvaluationQueue(mock_eval_service, mock_settings)


def test_enqueue_accepts_when_not_full(queue):
    """Enqueue returns True and depth increments when queue is not full."""
    result = queue.enqueue(
        query="What is RAG?",
        answer="Retrieval-Augmented Generation",
        contexts=["context1"],
        model_used="llama3.2",
        generation_time_ms=150.0,
    )
    assert result is True
    assert queue.depth == 1


def test_enqueue_drops_when_full(queue):
    """Enqueue returns False when queue is at capacity, drops counter increments."""
    # Fill queue to capacity (3)
    for i in range(3):
        assert queue.enqueue(
            query=f"q{i}", answer=f"a{i}", contexts=[], model_used="m", generation_time_ms=None
        )

    assert queue.depth == 3
    assert queue.drops_since_startup == 0

    # Try one more — should be dropped
    result = queue.enqueue(
        query="overflow", answer="answer", contexts=[], model_used="m", generation_time_ms=None
    )
    assert result is False
    assert queue.drops_since_startup == 1
    assert queue.depth == 3


@pytest.mark.asyncio
async def test_consumer_processes_items(mock_eval_service, mock_settings):
    """Consumer dequeues items and calls score_live_query with correct args."""
    queue = LiveEvaluationQueue(mock_eval_service, mock_settings)
    await queue.start()

    queue.enqueue(
        query="test query",
        answer="test answer",
        contexts=["ctx1", "ctx2"],
        model_used="qwen3:8b",
        generation_time_ms=200.0,
    )

    # Wait for consumer to process
    await asyncio.sleep(0.5)
    await queue.stop()

    mock_eval_service.score_live_query.assert_called_once_with(
        query="test query",
        answer="test answer",
        contexts=["ctx1", "ctx2"],
        model_used="qwen3:8b",
        generation_time_ms=200.0,
    )
    assert queue.processed_since_startup == 1


@pytest.mark.asyncio
async def test_start_stop_lifecycle(mock_eval_service, mock_settings):
    """Start creates N consumer tasks, stop clears them."""
    queue = LiveEvaluationQueue(mock_eval_service, mock_settings)

    assert len(queue._consumers) == 0

    await queue.start()
    assert len(queue._consumers) == 2  # eval_live_max_concurrent = 2

    await queue.stop()
    assert len(queue._consumers) == 0


def test_stats_properties(queue):
    """Initial stats are all zero/default."""
    assert queue.depth == 0
    assert queue.capacity == 3  # eval_live_queue_size = 3
    assert queue.drops_since_startup == 0
    assert queue.processed_since_startup == 0
