"""Tests for WarmingQueueService persistence layer.

Comprehensive tests for crash-safe file-based persistence including:
- Atomic writes
- State machine transitions
- File locking
- Quarantine handling
- Job lifecycle operations
"""

import threading
import time
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from ai_ready_rag.services.warming_queue import (
    InvalidStateTransition,
    WarmingJob,
    WarmingQueueService,
    job_lock,
)

# =============================================================================
# TestWarmingJobSerialization
# =============================================================================


class TestWarmingJobSerialization:
    """Tests for WarmingJob to_dict/from_dict serialization."""

    def test_to_dict_serializes_all_fields(self):
        """All WarmingJob fields serialize to dictionary."""
        job = WarmingJob(
            id="test-123",
            queries=["q1", "q2"],
            total=2,
            status="running",
            processed_index=1,
            failed_indices=[0],
            success_count=1,
            triggered_by="api",
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            started_at=datetime(2024, 1, 1, 12, 1, 0, tzinfo=UTC),
            locked_by="worker-1",
            locked_at=datetime(2024, 1, 1, 12, 1, 0, tzinfo=UTC),
        )

        data = job.to_dict()

        assert data["id"] == "test-123"
        assert data["queries"] == ["q1", "q2"]
        assert data["total"] == 2
        assert data["status"] == "running"
        assert data["processed_index"] == 1
        assert data["failed_indices"] == [0]
        assert data["success_count"] == 1
        assert data["triggered_by"] == "api"
        assert data["created_at"] == "2024-01-01T12:00:00+00:00"
        assert data["locked_by"] == "worker-1"
        assert data["version"] == 1
        assert data["results"] == []

    def test_from_dict_deserializes_all_fields(self):
        """All fields are correctly deserialized from dictionary."""
        data = {
            "id": "test-456",
            "queries": ["query1", "query2", "query3"],
            "total": 3,
            "version": 1,
            "status": "completed",
            "processed_index": 3,
            "failed_indices": [1],
            "success_count": 2,
            "triggered_by": "cli",
            "created_at": "2024-01-01T10:00:00+00:00",
            "started_at": "2024-01-01T10:01:00+00:00",
            "completed_at": "2024-01-01T10:05:00+00:00",
            "locked_by": None,
            "locked_at": None,
            "error": None,
            "results": [{"query": "q1", "success": True}],
        }

        job = WarmingJob.from_dict(data)

        assert job.id == "test-456"
        assert job.queries == ["query1", "query2", "query3"]
        assert job.total == 3
        assert job.status == "completed"
        assert job.processed_index == 3
        assert job.failed_indices == [1]
        assert job.success_count == 2
        assert job.triggered_by == "cli"
        assert job.created_at is not None
        assert job.results == [{"query": "q1", "success": True}]

    def test_from_dict_handles_missing_optional_fields(self):
        """Missing optional fields get defaults."""
        data = {
            "id": "minimal-job",
            "queries": ["q1"],
            "total": 1,
            "status": "pending",
        }

        job = WarmingJob.from_dict(data)

        assert job.id == "minimal-job"
        assert job.processed_index == 0
        assert job.failed_indices == []
        assert job.success_count == 0
        assert job.triggered_by == "api"
        assert job.version == 1
        assert job.results == []

    def test_datetime_parsing_iso_format(self):
        """ISO datetime strings are parsed correctly."""
        data = {
            "id": "dt-test",
            "queries": ["q1"],
            "total": 1,
            "status": "pending",
            "created_at": "2024-06-15T14:30:00+00:00",
        }

        job = WarmingJob.from_dict(data)

        assert job.created_at is not None
        assert job.created_at.year == 2024
        assert job.created_at.month == 6
        assert job.created_at.day == 15
        assert job.created_at.hour == 14
        assert job.created_at.minute == 30

    def test_datetime_parsing_with_z_suffix(self):
        """ISO datetime with Z suffix is parsed correctly."""
        data = {
            "id": "z-test",
            "queries": ["q1"],
            "total": 1,
            "status": "pending",
            "created_at": "2024-01-01T00:00:00Z",
        }

        job = WarmingJob.from_dict(data)

        assert job.created_at is not None
        assert job.created_at.tzinfo is not None


# =============================================================================
# TestWarmingJobBackwardCompat
# =============================================================================


class TestWarmingJobBackwardCompat:
    """Tests for backward compatibility properties."""

    def test_processed_property_alias(self):
        """The processed property aliases processed_index."""
        job = WarmingJob(
            id="compat-test",
            queries=["q1", "q2", "q3"],
            total=3,
            processed_index=5,
        )

        # Getter works
        assert job.processed == 5

        # Setter works
        job.processed = 10
        assert job.processed_index == 10

    def test_failed_queries_from_indices(self):
        """failed_queries returns correct query strings from indices."""
        job = WarmingJob(
            id="failed-test",
            queries=["query_a", "query_b", "query_c", "query_d"],
            total=4,
            failed_indices=[0, 2],
        )

        failed = job.failed_queries

        assert len(failed) == 2
        assert "query_a" in failed
        assert "query_c" in failed


# =============================================================================
# TestStateTransitions
# =============================================================================


class TestStateTransitions:
    """Tests for state machine validation."""

    def test_valid_pending_to_running(self, warming_service):
        """Transition from pending to running is valid."""
        # Should not raise
        warming_service._validate_state_transition("pending", "running")

    def test_valid_running_to_completed(self, warming_service):
        """Transition from running to completed is valid."""
        warming_service._validate_state_transition("running", "completed")

    def test_valid_running_to_failed(self, warming_service):
        """Transition from running to failed is valid."""
        warming_service._validate_state_transition("running", "failed")

    def test_valid_running_to_pending_on_stale(self, warming_service):
        """Transition from running to pending (stale lock) is valid."""
        warming_service._validate_state_transition("running", "pending")

    def test_invalid_pending_to_completed_raises(self, warming_service):
        """Cannot skip running state."""
        with pytest.raises(InvalidStateTransition) as exc:
            warming_service._validate_state_transition("pending", "completed")

        assert "pending" in str(exc.value)
        assert "completed" in str(exc.value)

    def test_invalid_completed_to_any_raises(self, warming_service):
        """Cannot transition from completed (terminal state)."""
        with pytest.raises(InvalidStateTransition):
            warming_service._validate_state_transition("completed", "running")

    def test_invalid_failed_to_any_raises(self, warming_service):
        """Cannot transition from failed (terminal state)."""
        with pytest.raises(InvalidStateTransition):
            warming_service._validate_state_transition("failed", "running")


# =============================================================================
# TestAtomicWrite
# =============================================================================


class TestAtomicWrite:
    """Tests for crash-safe atomic writes."""

    def test_atomic_write_creates_file(self, warming_service):
        """Creating a job writes a file."""
        job = warming_service.create_job(queries=["test query"])
        job_path = warming_service.job_path(job.id)

        assert job_path.exists()
        assert job_path.stat().st_size > 0

    def test_atomic_write_no_temp_files_on_success(self, warming_service):
        """No .tmp files left after successful write."""
        warming_service.create_job(queries=["test query"])

        tmp_files = list(warming_service.jobs_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_atomic_write_crash_leaves_original(self, warming_service, sample_job, monkeypatch):
        """If rename fails, original file is preserved."""
        import os

        original_content = warming_service.job_path(sample_job.id).read_text()

        # Mock rename to simulate crash
        def mock_rename(src, dst):
            raise OSError("Simulated disk failure")

        monkeypatch.setattr(os, "rename", mock_rename)

        sample_job.processed_index = 999

        with pytest.raises(OSError):
            warming_service.update_job(sample_job)

        # Original should be unchanged
        current_content = warming_service.job_path(sample_job.id).read_text()
        assert current_content == original_content

    def test_atomic_write_fsync_called(self, warming_service, monkeypatch):
        """Verify fsync is called for durability."""
        import os

        fsync_calls = []
        original_fsync = os.fsync

        def tracking_fsync(fd):
            fsync_calls.append(fd)
            return original_fsync(fd)

        monkeypatch.setattr(os, "fsync", tracking_fsync)

        warming_service.create_job(queries=["test query"])

        assert len(fsync_calls) >= 1


# =============================================================================
# TestQuarantine
# =============================================================================


class TestQuarantine:
    """Tests for quarantine functionality."""

    def test_corrupted_json_quarantined(self, warming_service, warming_queue_dir):
        """Corrupted JSON files are moved to quarantine."""
        # Create corrupted file directly
        job_id = "corrupt-123"
        job_file = warming_queue_dir / "jobs" / f"job_{job_id}.json"
        job_file.write_text("{invalid json", encoding="utf-8")

        result = warming_service.get_job(job_id)

        assert result is None
        assert not job_file.exists()
        quarantined = list((warming_queue_dir / "quarantine").glob(f"job_{job_id}*"))
        assert len(quarantined) >= 1

    def test_missing_required_field_quarantined(self, warming_service, warming_queue_dir):
        """JSON missing required field is quarantined."""
        job_id = "missing-field-123"
        job_file = warming_queue_dir / "jobs" / f"job_{job_id}.json"
        # Missing "id" field
        job_file.write_text('{"queries": [], "total": 0, "status": "pending"}', encoding="utf-8")

        result = warming_service.get_job(job_id)

        assert result is None
        assert not job_file.exists()
        quarantined = list((warming_queue_dir / "quarantine").glob(f"*{job_id}*"))
        assert len(quarantined) >= 1

    def test_invalid_status_quarantined(self, warming_service, warming_queue_dir):
        """JSON with invalid status is quarantined."""
        job_id = "bad-status-123"
        job_file = warming_queue_dir / "jobs" / f"job_{job_id}.json"
        job_file.write_text(
            '{"id": "bad", "queries": [], "total": 0, "status": "bogus"}',
            encoding="utf-8",
        )

        result = warming_service.get_job(job_id)

        assert result is None
        quarantined = list((warming_queue_dir / "quarantine").glob("*bad-status*"))
        assert len(quarantined) >= 1

    def test_quarantine_creates_reason_file(self, warming_service, warming_queue_dir):
        """Reason file explains why file was quarantined."""
        job_id = "reason-test-123"
        job_file = warming_queue_dir / "jobs" / f"job_{job_id}.json"
        job_file.write_text(
            '{"id": "bad", "queries": [], "total": 0, "status": "bogus"}',
            encoding="utf-8",
        )

        warming_service.get_job(job_id)

        reason_files = list((warming_queue_dir / "quarantine").glob("*.reason"))
        assert len(reason_files) >= 1
        reason_text = reason_files[0].read_text()
        assert "Invalid status" in reason_text

    def test_quarantine_timestamp_in_filename(self, warming_service, warming_queue_dir):
        """Quarantined filename has timestamp."""
        job_id = "ts-test-123"
        job_file = warming_queue_dir / "jobs" / f"job_{job_id}.json"
        job_file.write_text("{bad json", encoding="utf-8")

        warming_service.get_job(job_id)

        quarantined = list((warming_queue_dir / "quarantine").glob("*.json"))
        assert len(quarantined) >= 1
        # Filename should contain timestamp pattern like _YYYYMMDD_HHMMSS
        filename = quarantined[0].name
        assert "_202" in filename  # Year 2020+


# =============================================================================
# TestJobLocking
# =============================================================================


class TestJobLocking:
    """Tests for file locking behavior."""

    def test_job_lock_acquired_on_new_file(self, warming_service, sample_job):
        """Lock can be acquired on a job file."""
        job_path = warming_service.job_path(sample_job.id)

        with job_lock(job_path) as acquired:
            assert acquired is True

    def test_job_lock_blocks_concurrent_access(self, warming_service, sample_job):
        """Cannot acquire lock while another process holds it."""
        job_path = warming_service.job_path(sample_job.id)
        second_acquired = None

        def try_acquire():
            nonlocal second_acquired
            with job_lock(job_path) as acquired:
                second_acquired = acquired
                # Hold lock briefly
                time.sleep(0.1)

        # First thread holds lock
        with job_lock(job_path) as first_acquired:
            assert first_acquired is True

            # Second thread tries while first holds
            thread = threading.Thread(target=try_acquire)
            thread.start()
            time.sleep(0.05)  # Give thread time to try
            # Thread should fail to acquire
            thread.join(timeout=0.5)

        # After first releases, second should have failed
        assert second_acquired is False

    def test_job_lock_released_on_exit(self, warming_service, sample_job):
        """Lock is released when context exits."""
        job_path = warming_service.job_path(sample_job.id)

        with job_lock(job_path) as first:
            assert first is True

        # Should be able to acquire again after release
        with job_lock(job_path) as second:
            assert second is True

    def test_acquire_job_sets_lock_metadata(self, warming_service, sample_job):
        """acquire_job sets locked_by and locked_at."""
        acquired = warming_service.acquire_job(sample_job.id, worker_id="worker-1")

        assert acquired is not None
        assert acquired.locked_by == "worker-1"
        assert acquired.locked_at is not None
        assert acquired.status == "running"

    def test_stale_lock_reclaimed_after_timeout(self, warming_queue_dir):
        """Stale running jobs can be reclaimed after external reset to pending.

        Note: When a job's lock becomes stale, an external process (like a
        supervisor or health check) should reset its status to pending so a
        new worker can acquire it. This test verifies that flow works.
        """
        # Use very short timeout for test
        service = WarmingQueueService(
            queue_dir=warming_queue_dir,
            lock_timeout_minutes=0,  # Immediate timeout
        )

        job = service.create_job(queries=["test"])

        # Acquire with first worker
        acquired_first = service.acquire_job(job.id, worker_id="old-worker")
        assert acquired_first is not None
        assert acquired_first.status == "running"

        # Simulate stale lock recovery: reset status to pending
        # (This is what a supervisor/health check would do)
        acquired_first.status = "pending"
        acquired_first.locked_by = None
        acquired_first.locked_at = None
        service._atomic_write(acquired_first)

        # New worker should be able to acquire the reset job
        acquired = service.acquire_job(job.id, worker_id="new-worker")

        assert acquired is not None
        assert acquired.locked_by == "new-worker"
        assert acquired.status == "running"

    def test_fresh_lock_not_reclaimed(self, warming_service, sample_job):
        """Recently locked jobs cannot be reclaimed."""
        acquired = warming_service.acquire_job(sample_job.id, worker_id="worker-1")
        assert acquired is not None

        # Try to acquire again immediately
        second = warming_service.acquire_job(sample_job.id, worker_id="worker-2")

        assert second is None


# =============================================================================
# TestJobCRUD
# =============================================================================


class TestJobCRUD:
    """Tests for create/read/update/delete operations."""

    def test_create_job_generates_uuid(self, warming_service):
        """Job ID is a valid UUID."""
        job = warming_service.create_job(queries=["test"])

        # Should not raise
        uuid.UUID(job.id)

    def test_create_job_sets_defaults(self, warming_service):
        """Created job has correct defaults."""
        job = warming_service.create_job(queries=["q1", "q2"])

        assert job.status == "pending"
        assert job.processed_index == 0
        assert job.total == 2
        assert job.success_count == 0
        assert job.failed_indices == []
        assert job.created_at is not None

    def test_get_job_returns_none_for_missing(self, warming_service):
        """Nonexistent job returns None."""
        result = warming_service.get_job("nonexistent-id")
        assert result is None

    def test_get_job_loads_valid_file(self, warming_service, sample_job):
        """Valid job file is loaded correctly."""
        loaded = warming_service.get_job(sample_job.id)

        assert loaded is not None
        assert loaded.id == sample_job.id
        assert loaded.queries == sample_job.queries
        assert loaded.total == sample_job.total

    def test_delete_job_removes_file_and_lock(self, warming_service, sample_job):
        """Delete removes both .json and .lock files."""
        job_path = warming_service.job_path(sample_job.id)
        lock_path = job_path.with_suffix(".lock")

        # Create lock file
        lock_path.touch()

        warming_service.delete_job(sample_job.id)

        assert not job_path.exists()
        assert not lock_path.exists()

    def test_list_pending_jobs_fifo_order(self, warming_service):
        """Jobs are listed in FIFO order by created_at."""
        # Create jobs with small delays to ensure different timestamps
        job1 = warming_service.create_job(queries=["first"])
        time.sleep(0.01)
        job2 = warming_service.create_job(queries=["second"])
        time.sleep(0.01)
        job3 = warming_service.create_job(queries=["third"])

        pending = warming_service.list_pending_jobs()

        assert len(pending) == 3
        assert pending[0].id == job1.id
        assert pending[1].id == job2.id
        assert pending[2].id == job3.id


# =============================================================================
# TestRestartRecovery
# =============================================================================


class TestRestartRecovery:
    """Tests for crash recovery scenarios."""

    def test_resume_from_exact_position(self, warming_queue_dir):
        """After restart, job resumes from exact position."""
        service1 = WarmingQueueService(queue_dir=warming_queue_dir)
        job = service1.create_job(queries=["q1", "q2", "q3", "q4", "q5"])

        # Simulate partial processing
        acquired = service1.acquire_job(job.id, "worker")
        acquired.processed_index = 3
        acquired.success_count = 3
        service1.update_job(acquired)

        # Simulate crash - create fresh service
        service2 = WarmingQueueService(queue_dir=warming_queue_dir)
        recovered = service2.get_job(job.id)

        assert recovered.processed_index == 3
        assert recovered.success_count == 3

    def test_running_job_survives_restart(self, warming_queue_dir):
        """Running job is listed after restart."""
        service1 = WarmingQueueService(queue_dir=warming_queue_dir)
        job = service1.create_job(queries=["q1"])
        service1.acquire_job(job.id, "worker")

        # Simulate restart
        service2 = WarmingQueueService(queue_dir=warming_queue_dir)
        pending = service2.list_pending_jobs()

        # Running jobs are included in pending list
        assert len(pending) == 1
        assert pending[0].id == job.id
        assert pending[0].status == "running"


# =============================================================================
# TestJobLifecycle
# =============================================================================


class TestJobLifecycle:
    """Tests for complete job lifecycle."""

    def test_complete_job_workflow(self, warming_service, sample_job):
        """Completing a job sets correct state."""
        # Must be running first
        acquired = warming_service.acquire_job(sample_job.id, "worker")

        warming_service.complete_job(acquired)

        completed = warming_service.get_job(sample_job.id)
        assert completed.status == "completed"
        assert completed.locked_by is None
        assert completed.completed_at is not None

    def test_fail_job_workflow(self, warming_service, sample_job):
        """Failing a job sets correct state."""
        acquired = warming_service.acquire_job(sample_job.id, "worker")

        warming_service.fail_job(acquired, error="Test failure reason")

        failed = warming_service.get_job(sample_job.id)
        assert failed.status == "failed"
        assert failed.error == "Test failure reason"
        assert failed.locked_by is None
        assert failed.completed_at is not None

    def test_archive_on_completion_when_enabled(
        self, warming_service_with_archive, warming_queue_dir
    ):
        """Completed jobs are archived when enabled."""
        job = warming_service_with_archive.create_job(queries=["test"])
        acquired = warming_service_with_archive.acquire_job(job.id, "worker")

        warming_service_with_archive.complete_job(acquired)

        archives = list((warming_queue_dir / "archive").glob("*.json"))
        assert len(archives) == 1
        assert job.id in archives[0].name

    def test_no_archive_when_disabled(self, warming_service, warming_queue_dir):
        """Jobs are not archived when archive_completed=False."""
        job = warming_service.create_job(queries=["test"])
        acquired = warming_service.acquire_job(job.id, "worker")

        warming_service.complete_job(acquired)

        archives = list((warming_queue_dir / "archive").glob("*.json"))
        assert len(archives) == 0


# =============================================================================
# TestCleanup
# =============================================================================


class TestCleanup:
    """Tests for cleanup operations."""

    def test_cleanup_old_failed_jobs_by_retention(self, warming_service, sample_job):
        """Failed jobs older than retention are deleted."""
        # Acquire and fail the job
        acquired = warming_service.acquire_job(sample_job.id, "worker")
        warming_service.fail_job(acquired, error="Test failure")

        # Manually backdate completed_at
        job = warming_service.get_job(sample_job.id)
        job.completed_at = datetime.now(UTC) - timedelta(days=30)
        warming_service._atomic_write(job)

        deleted = warming_service.cleanup_old_failed_jobs(retention_days=7)

        assert deleted == 1
        assert warming_service.get_job(sample_job.id) is None

    def test_cleanup_skips_recent_failed_jobs(self, warming_service, sample_job):
        """Recently failed jobs are not deleted."""
        acquired = warming_service.acquire_job(sample_job.id, "worker")
        warming_service.fail_job(acquired, error="Test failure")

        # completed_at is now() so it's recent
        deleted = warming_service.cleanup_old_failed_jobs(retention_days=7)

        assert deleted == 0
        assert warming_service.get_job(sample_job.id) is not None

    def test_cleanup_skips_completed_jobs(self, warming_service, sample_job):
        """Completed (not failed) jobs are not deleted by cleanup."""
        acquired = warming_service.acquire_job(sample_job.id, "worker")
        warming_service.complete_job(acquired)

        # Manually backdate completed_at
        job = warming_service.get_job(sample_job.id)
        job.completed_at = datetime.now(UTC) - timedelta(days=30)
        warming_service._atomic_write(job)

        deleted = warming_service.cleanup_old_failed_jobs(retention_days=7)

        assert deleted == 0
        assert warming_service.get_job(sample_job.id) is not None


# =============================================================================
# TestLoadPerformance (Optional, marked slow)
# =============================================================================


@pytest.mark.slow
class TestLoadPerformance:
    """Optional performance tests."""

    def test_large_job_checkpoint_performance(self, warming_service):
        """Large job checkpoints complete reasonably fast."""
        queries = [f"Query {i}" for i in range(10000)]
        job = warming_service.create_job(queries=queries)
        acquired = warming_service.acquire_job(job.id, "worker")

        start = time.time()
        for i in range(100):
            acquired.processed_index = i * 100
            warming_service.update_job(acquired)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should be fast

    def test_concurrent_job_processing(self, warming_queue_dir):
        """Multiple concurrent workers don't corrupt data."""
        service = WarmingQueueService(queue_dir=warming_queue_dir)
        jobs = [service.create_job(queries=[f"query-{i}"]) for i in range(10)]
        results = []
        errors = []

        def process_job(job_id, worker_id):
            try:
                acquired = service.acquire_job(job_id, worker_id)
                if acquired:
                    # Simulate processing
                    acquired.processed_index = acquired.total
                    acquired.success_count = acquired.total
                    service.complete_job(acquired)
                    results.append(job_id)
            except Exception as e:
                errors.append((job_id, str(e)))

        threads = [
            threading.Thread(target=process_job, args=(job.id, f"worker-{i}"))
            for i, job in enumerate(jobs)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All jobs should complete, no errors
        assert len(errors) == 0
        assert len(results) == 10

        # Verify all completed
        for job in jobs:
            loaded = service.get_job(job.id)
            assert loaded.status == "completed"
