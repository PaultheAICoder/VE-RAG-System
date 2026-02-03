"""Tests for WarmingWorker background service."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_ready_rag.core.exceptions import (
    ConnectionTimeoutError,
    RateLimitExceededError,
    ServiceUnavailableError,
    WarmingCancelledException,
)
from ai_ready_rag.services.warming_worker import (
    RETRYABLE_EXCEPTIONS,
    WarmingWorker,
    recover_stale_jobs,
)


class TestWarmingWorkerInit:
    """Test worker initialization."""

    def test_creates_unique_worker_id(self):
        """Each worker gets unique ID."""
        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_retry_delays = "5,30,120"
        mock_settings.warming_lease_renewal_seconds = 60

        worker1 = WarmingWorker(mock_rag, mock_settings)
        worker2 = WarmingWorker(mock_rag, mock_settings)

        assert worker1.worker_id != worker2.worker_id
        assert worker1.worker_id.startswith("worker-")
        assert worker2.worker_id.startswith("worker-")

    def test_starts_with_no_current_job(self):
        """Initial state has no job."""
        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_retry_delays = "5,30,120"

        worker = WarmingWorker(mock_rag, mock_settings)

        assert worker._current_job_id is None
        assert worker._task is None
        assert worker._lease_task is None

    def test_parses_retry_delays(self):
        """Retry delays parsed from settings."""
        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_retry_delays = "5,30,120"

        worker = WarmingWorker(mock_rag, mock_settings)

        assert worker._retry_delays == [5, 30, 120]


class TestWarmingWorkerLifecycle:
    """Test start/stop behavior."""

    @pytest.mark.asyncio
    async def test_start_creates_tasks(self):
        """Start creates run and lease tasks."""
        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_retry_delays = "5,30,120"
        mock_settings.warming_lease_renewal_seconds = 60
        mock_settings.warming_scan_interval_seconds = 5

        worker = WarmingWorker(mock_rag, mock_settings)

        # Mock the loop methods to not actually run
        with patch.object(worker, "_run_loop", new_callable=AsyncMock):
            with patch.object(worker, "_lease_renewal_loop", new_callable=AsyncMock):
                await worker.start()

                assert worker._task is not None
                assert worker._lease_task is not None
                assert not worker._shutdown.is_set()

                # Clean up
                await worker.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self):
        """Stop cancels and awaits tasks."""
        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_retry_delays = "5,30,120"
        mock_settings.warming_lease_renewal_seconds = 60
        mock_settings.warming_scan_interval_seconds = 5

        worker = WarmingWorker(mock_rag, mock_settings)

        # Create mock tasks
        async def slow_task():
            await asyncio.sleep(100)

        worker._task = asyncio.create_task(slow_task())
        worker._lease_task = asyncio.create_task(slow_task())

        await worker.stop()

        assert worker._shutdown.is_set()
        assert worker._task.cancelled() or worker._task.done()
        assert worker._lease_task.cancelled() or worker._lease_task.done()


class TestRetryableExceptions:
    """Test retryable exception classification."""

    def test_retryable_exceptions_tuple(self):
        """RETRYABLE_EXCEPTIONS contains correct types."""
        assert ConnectionTimeoutError in RETRYABLE_EXCEPTIONS
        assert ServiceUnavailableError in RETRYABLE_EXCEPTIONS
        assert RateLimitExceededError in RETRYABLE_EXCEPTIONS

    def test_isinstance_checks(self):
        """Exceptions match with isinstance."""
        timeout_err = ConnectionTimeoutError("test")
        unavailable_err = ServiceUnavailableError("test")
        rate_err = RateLimitExceededError("test")

        assert isinstance(timeout_err, RETRYABLE_EXCEPTIONS)
        assert isinstance(unavailable_err, RETRYABLE_EXCEPTIONS)
        assert isinstance(rate_err, RETRYABLE_EXCEPTIONS)

    def test_non_retryable_not_matched(self):
        """Non-retryable exceptions don't match."""
        value_err = ValueError("test")
        cancelled_err = WarmingCancelledException()

        assert not isinstance(value_err, RETRYABLE_EXCEPTIONS)
        assert not isinstance(cancelled_err, RETRYABLE_EXCEPTIONS)


class TestProgressEstimation:
    """Test EMA-based progress estimation."""

    def test_ema_calculation(self):
        """EMA weights recent queries more."""
        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_retry_delays = "5,30,120"
        mock_settings.warming_delay_seconds = 2.0

        worker = WarmingWorker(mock_rag, mock_settings)

        # Add some durations
        worker._query_durations.extend([1.0, 1.0, 1.0, 5.0])  # Last one is spike

        # Create mock job
        mock_job = MagicMock()
        mock_job.total_queries = 100
        mock_job.processed_queries = 50
        mock_job.failed_queries = 0

        estimate = worker._estimate_remaining(mock_job)

        # Should be positive and account for remaining queries
        assert estimate > 0
        assert estimate < 50 * 10  # Reasonable upper bound

    def test_qps_calculation(self):
        """Queries per second from durations."""
        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_retry_delays = "5,30,120"

        worker = WarmingWorker(mock_rag, mock_settings)

        # No data
        assert worker._calculate_qps() == 0.0

        # Add durations (1 second each)
        worker._query_durations.extend([1.0, 1.0, 1.0])

        qps = worker._calculate_qps()
        assert qps == 1.0  # 3 queries in 3 seconds

    def test_estimate_with_no_data(self):
        """Estimate returns -1 with no duration data."""
        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_retry_delays = "5,30,120"

        worker = WarmingWorker(mock_rag, mock_settings)

        mock_job = MagicMock()
        mock_job.total_queries = 100
        mock_job.processed_queries = 0
        mock_job.failed_queries = 0

        assert worker._estimate_remaining(mock_job) == -1


class TestRecoverStaleJobs:
    """Test stale job recovery."""

    @pytest.mark.asyncio
    async def test_recover_expired_leases(self, db):
        """Reset jobs with expired leases."""
        from ai_ready_rag.db.models import WarmingQueue

        # Create a job with expired lease
        job = WarmingQueue(
            file_path="/tmp/test.txt",
            file_checksum="abc123",
            source_type="manual",
            total_queries=10,
            status="running",
            worker_id="old-worker",
            worker_lease_expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        db.add(job)
        db.commit()

        job_id = job.id

        # Override SessionLocal for test
        with patch("ai_ready_rag.services.warming_worker.SessionLocal", return_value=db):
            count = await recover_stale_jobs()

        # Query the job again from the same session
        updated_job = db.query(WarmingQueue).filter(WarmingQueue.id == job_id).first()

        assert count == 1
        assert updated_job.status == "pending"
        assert updated_job.worker_id is None
        assert updated_job.worker_lease_expires_at is None


class TestJobLeaseAcquisition:
    """Test lease acquire/renew/release."""

    def test_acquires_pending_job(self, db):
        """Worker acquires oldest pending job."""
        from ai_ready_rag.db.models import WarmingQueue

        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_retry_delays = "5,30,120"
        mock_settings.warming_lease_duration_minutes = 10

        worker = WarmingWorker(mock_rag, mock_settings)

        # Create pending job
        job = WarmingQueue(
            file_path="/tmp/test.txt",
            file_checksum="abc123",
            source_type="manual",
            total_queries=10,
            status="pending",
        )
        db.add(job)
        db.commit()

        acquired = worker._acquire_job_lease(db)

        assert acquired is not None
        assert acquired.id == job.id
        assert acquired.status == "running"
        assert acquired.worker_id == worker.worker_id
        assert acquired.worker_lease_expires_at is not None

    def test_skips_job_with_active_lease(self, db):
        """Worker skips jobs with non-expired leases."""
        from ai_ready_rag.db.models import WarmingQueue

        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_retry_delays = "5,30,120"
        mock_settings.warming_lease_duration_minutes = 10

        worker = WarmingWorker(mock_rag, mock_settings)

        # Create running job with active lease
        job = WarmingQueue(
            file_path="/tmp/test.txt",
            file_checksum="abc123",
            source_type="manual",
            total_queries=10,
            status="running",
            worker_id="other-worker",
            worker_lease_expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        db.add(job)
        db.commit()

        acquired = worker._acquire_job_lease(db)

        assert acquired is None

    def test_renews_lease_periodically(self, db):
        """Lease renewal extends expiry."""
        from ai_ready_rag.db.models import WarmingQueue

        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_retry_delays = "5,30,120"
        mock_settings.warming_lease_duration_minutes = 10

        worker = WarmingWorker(mock_rag, mock_settings)

        # Create job owned by this worker
        original_expiry = datetime.utcnow() + timedelta(minutes=1)
        job = WarmingQueue(
            file_path="/tmp/test.txt",
            file_checksum="abc123",
            source_type="manual",
            total_queries=10,
            status="running",
            worker_id=worker.worker_id,
            worker_lease_expires_at=original_expiry,
        )
        db.add(job)
        db.commit()

        # Renew lease
        renewed = worker._renew_lease(db, job.id)
        db.refresh(job)

        assert renewed is True
        assert job.worker_lease_expires_at > original_expiry


class TestPauseCancel:
    """Test pause/cancel detection."""

    def test_detects_pause_flag(self, db):
        """Stops processing when is_paused=True."""
        from ai_ready_rag.db.models import WarmingQueue

        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_retry_delays = "5,30,120"

        worker = WarmingWorker(mock_rag, mock_settings)

        # Create paused job
        job = WarmingQueue(
            file_path="/tmp/test.txt",
            file_checksum="abc123",
            source_type="manual",
            total_queries=10,
            is_paused=True,
        )
        db.add(job)
        db.commit()

        should_stop, reason = worker._should_stop(db, job.id)

        assert should_stop is True
        assert reason == "paused"

    def test_detects_cancel_flag(self, db):
        """Stops and marks cancelled when is_cancel_requested=True."""
        from ai_ready_rag.db.models import WarmingQueue

        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_retry_delays = "5,30,120"

        worker = WarmingWorker(mock_rag, mock_settings)

        # Create cancelled job
        job = WarmingQueue(
            file_path="/tmp/test.txt",
            file_checksum="abc123",
            source_type="manual",
            total_queries=10,
            is_cancel_requested=True,
        )
        db.add(job)
        db.commit()

        should_stop, reason = worker._should_stop(db, job.id)

        assert should_stop is True
        assert reason == "cancelled"

    def test_continues_when_not_stopped(self, db):
        """Continues when neither pause nor cancel is set."""
        from ai_ready_rag.db.models import WarmingQueue

        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_retry_delays = "5,30,120"

        worker = WarmingWorker(mock_rag, mock_settings)

        # Create normal job
        job = WarmingQueue(
            file_path="/tmp/test.txt",
            file_checksum="abc123",
            source_type="manual",
            total_queries=10,
            is_paused=False,
            is_cancel_requested=False,
        )
        db.add(job)
        db.commit()

        should_stop, reason = worker._should_stop(db, job.id)

        assert should_stop is False
        assert reason is None
