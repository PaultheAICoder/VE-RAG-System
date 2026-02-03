"""Persistent file-based warming queue service.

Provides crash-safe persistence for cache warming jobs with:
- Atomic writes (temp + fsync + rename)
- File locking (fcntl)
- Job state machine enforcement
- Progress tracking by index
- Quarantine for malformed files
- FIFO ordering by created_at
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class InvalidStateTransition(Exception):
    """Raised when an invalid job state transition is attempted."""

    pass


# Valid state transitions
VALID_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"running"},
    "running": {"completed", "failed", "pending", "paused"},  # pending = lock expired
    "paused": {"running"},  # Resume from paused
    "completed": set(),  # Terminal state
    "failed": set(),  # Terminal state
}


@dataclass
class WarmingJob:
    """Tracks state of a file-based cache warming job.

    Fields match spec section 2 (Job File Format).
    """

    id: str
    queries: list[str]
    total: int
    version: int = 1
    status: str = "pending"
    processed_index: int = 0
    failed_indices: list[int] = field(default_factory=list)
    success_count: int = 0
    triggered_by: str = "api"
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    locked_by: str | None = None
    locked_at: datetime | None = None
    error: str | None = None
    # Keep results for SSE compatibility
    results: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Serialize job to dictionary for JSON storage."""
        return {
            "id": self.id,
            "version": self.version,
            "queries": self.queries,
            "total": self.total,
            "status": self.status,
            "processed_index": self.processed_index,
            "failed_indices": self.failed_indices,
            "success_count": self.success_count,
            "triggered_by": self.triggered_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "locked_by": self.locked_by,
            "locked_at": self.locked_at.isoformat() if self.locked_at else None,
            "error": self.error,
            "results": self.results,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WarmingJob:
        """Deserialize job from dictionary."""

        def parse_datetime(val: str | None) -> datetime | None:
            if val is None:
                return None
            # Handle both ISO format with and without timezone
            try:
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return dt
            except (ValueError, AttributeError):
                return None

        return cls(
            id=data["id"],
            queries=data["queries"],
            total=data["total"],
            version=data.get("version", 1),
            status=data["status"],
            processed_index=data.get("processed_index", 0),
            failed_indices=data.get("failed_indices", []),
            success_count=data.get("success_count", 0),
            triggered_by=data.get("triggered_by", "api"),
            created_at=parse_datetime(data.get("created_at")),
            started_at=parse_datetime(data.get("started_at")),
            completed_at=parse_datetime(data.get("completed_at")),
            locked_by=data.get("locked_by"),
            locked_at=parse_datetime(data.get("locked_at")),
            error=data.get("error"),
            results=data.get("results", []),
        )

    # Backward compatibility properties
    @property
    def processed(self) -> int:
        """Alias for processed_index for backward compatibility."""
        return self.processed_index

    @processed.setter
    def processed(self, value: int):
        """Alias for processed_index for backward compatibility."""
        self.processed_index = value

    @property
    def failed_queries(self) -> list[str]:
        """Get failed queries by index for backward compatibility."""
        return [self.queries[i] for i in self.failed_indices if i < len(self.queries)]


@contextmanager
def job_lock(job_path: Path) -> Generator[bool, None, None]:
    """Acquire exclusive file lock for job operations.

    Uses fcntl for POSIX-compatible file locking.

    Yields:
        True if lock was acquired, False if another process holds it.
    """
    lock_path = job_path.with_suffix(".lock")
    lock_file = None
    acquired = False

    try:
        # Create lock file if it doesn't exist
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(lock_path, "w")  # noqa: SIM115 - intentionally not using context manager

        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            acquired = False

        yield acquired

    finally:
        if lock_file:
            if acquired:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            lock_file.close()


class WarmingQueueService:
    """File-based persistent warming queue service.

    Directory structure:
        queue_dir/
        ├── jobs/                # Active job files
        │   └── job_{uuid}.json
        ├── quarantine/          # Malformed/invalid files
        │   └── bad_file.txt
        └── archive/             # Optional: completed job logs
            └── job_{uuid}_completed.json
    """

    def __init__(
        self,
        queue_dir: str | Path,
        lock_timeout_minutes: int = 30,
        checkpoint_interval: int = 1,
        archive_completed: bool = False,
    ):
        """Initialize the warming queue service.

        Args:
            queue_dir: Base directory for queue storage
            lock_timeout_minutes: Minutes before stale lock can be reclaimed
            checkpoint_interval: Save progress every N queries
            archive_completed: Whether to archive completed jobs
        """
        self.queue_dir = Path(queue_dir)
        self.jobs_dir = self.queue_dir / "jobs"
        self.quarantine_dir = self.queue_dir / "quarantine"
        self.archive_dir = self.queue_dir / "archive"
        self.lock_timeout_minutes = lock_timeout_minutes
        self.checkpoint_interval = checkpoint_interval
        self.archive_completed = archive_completed

        # Ensure directories exist
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        if archive_completed:
            self.archive_dir.mkdir(parents=True, exist_ok=True)

    def job_path(self, job_id: str) -> Path:
        """Get the file path for a job."""
        return self.jobs_dir / f"job_{job_id}.json"

    def create_job(self, queries: list[str], triggered_by: str = "api") -> WarmingJob:
        """Create a new warming job.

        Args:
            queries: List of queries to warm
            triggered_by: User ID or 'cli' for folder watcher

        Returns:
            The created WarmingJob
        """
        job_id = str(uuid.uuid4())
        job = WarmingJob(
            id=job_id,
            queries=queries,
            total=len(queries),
            triggered_by=triggered_by,
            created_at=datetime.now(UTC),
        )
        self._atomic_write(job)
        logger.info(f"Created warming job {job_id} with {len(queries)} queries")
        return job

    def get_job(self, job_id: str) -> WarmingJob | None:
        """Load and validate job from file.

        Returns None if job doesn't exist or is corrupted.
        Corrupted jobs are moved to quarantine.
        """
        job_path = self.job_path(job_id)
        if not job_path.exists():
            return None

        try:
            with open(job_path, encoding="utf-8") as f:
                data = json.load(f)

            # Validate required fields
            required = ["id", "queries", "total", "status"]
            for field_name in required:
                if field_name not in data:
                    raise ValueError(f"Missing required field: {field_name}")

            # Validate status
            if data["status"] not in ("pending", "running", "completed", "failed", "paused"):
                raise ValueError(f"Invalid status: {data['status']}")

            return WarmingJob.from_dict(data)

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Corrupted job file {job_id}: {e}")
            self._quarantine_file(job_path, reason=str(e))
            return None

    def update_job(self, job: WarmingJob) -> None:
        """Update job state with atomic write.

        The write is crash-safe: temp file -> fsync -> rename.
        """
        self._atomic_write(job)

    def delete_job(self, job_id: str) -> bool:
        """Delete a job file.

        Returns True if deleted, False if not found.
        """
        job_path = self.job_path(job_id)
        lock_path = job_path.with_suffix(".lock")

        if job_path.exists():
            job_path.unlink()
            logger.info(f"Deleted warming job {job_id}")

        # Clean up lock file
        if lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass

        return True

    def acquire_job(self, job_id: str, worker_id: str) -> WarmingJob | None:
        """Attempt to acquire job for processing.

        Uses file locking to ensure only one worker processes a job.
        Stale locks (older than lock_timeout_minutes) are reclaimed.

        Args:
            job_id: ID of job to acquire
            worker_id: Unique identifier for this worker

        Returns:
            The acquired job, or None if unavailable
        """
        job_path = self.job_path(job_id)

        with job_lock(job_path) as acquired:
            if not acquired:
                return None

            job = self.get_job(job_id)
            if job is None:
                return None

            # Check current status
            if job.status not in ("pending", "running"):
                return None

            # Check for stale lock if running
            if job.status == "running" and job.locked_by and job.locked_at:
                lock_age = datetime.now(UTC) - job.locked_at
                if lock_age < timedelta(minutes=self.lock_timeout_minutes):
                    # Lock still valid, cannot acquire
                    return None
                # Stale lock, reclaim job
                logger.warning(f"Reclaiming stale job {job_id} from {job.locked_by}")

            # Acquire the job
            self._validate_state_transition(job.status, "running")
            job.status = "running"
            job.locked_by = worker_id
            job.locked_at = datetime.now(UTC)
            if job.started_at is None:
                job.started_at = datetime.now(UTC)
            self._atomic_write(job)
            return job

    def release_job(self, job_id: str) -> None:
        """Release lock on a job without changing status.

        Useful for graceful shutdown scenarios.
        """
        job = self.get_job(job_id)
        if job and job.locked_by:
            job.locked_by = None
            job.locked_at = None
            self._atomic_write(job)
            logger.info(f"Released lock on job {job_id}")

    def list_pending_jobs(self) -> list[WarmingJob]:
        """Find all pending/running jobs, ordered by created_at (FIFO)."""
        jobs: list[WarmingJob] = []

        for job_file in self.jobs_dir.glob("job_*.json"):
            job_id = job_file.stem.replace("job_", "")
            job = self.get_job(job_id)
            if job and job.status in ("pending", "running"):
                jobs.append(job)

        # Sort by created_at for FIFO ordering
        return sorted(jobs, key=lambda j: j.created_at or datetime.min.replace(tzinfo=UTC))

    def list_all_jobs(self) -> list[WarmingJob]:
        """List all jobs regardless of status."""
        jobs: list[WarmingJob] = []

        for job_file in self.jobs_dir.glob("job_*.json"):
            job_id = job_file.stem.replace("job_", "")
            job = self.get_job(job_id)
            if job:
                jobs.append(job)

        return sorted(jobs, key=lambda j: j.created_at or datetime.min.replace(tzinfo=UTC))

    def archive_job(self, job: WarmingJob) -> None:
        """Archive a completed job for audit trail."""
        if not self.archive_completed:
            return

        self.archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.archive_dir / f"job_{job.id}_{timestamp}.json"

        archive_data = job.to_dict()
        archive_data["archived_at"] = datetime.now(UTC).isoformat()

        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(archive_data, f, indent=2, default=str)

        logger.info(f"Archived job {job.id} to {archive_path.name}")

    def cleanup_old_failed_jobs(self, retention_days: int = 7) -> int:
        """Delete failed jobs older than retention_days.

        Returns the number of jobs deleted.
        """
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        deleted_count = 0

        for job_file in self.jobs_dir.glob("job_*.json"):
            job_id = job_file.stem.replace("job_", "")
            job = self.get_job(job_id)

            if job and job.status == "failed" and job.completed_at and job.completed_at < cutoff:
                self.delete_job(job_id)
                deleted_count += 1
                logger.info(f"Cleaned up old failed job {job_id}")

        return deleted_count

    def _atomic_write(self, job: WarmingJob) -> None:
        """Atomically write job state to file.

        Pattern: temp file -> fsync -> rename
        This ensures crash safety.
        """
        job_path = self.job_path(job.id)

        # Write to temp file in same directory (same filesystem for atomic rename)
        fd, temp_path = tempfile.mkstemp(
            dir=self.jobs_dir,
            prefix=f".job_{job.id}_",
            suffix=".tmp",
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(job.to_dict(), f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())  # Ensure data on disk

            # Atomic rename
            os.rename(temp_path, job_path)

        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _quarantine_file(self, file_path: Path, reason: str) -> None:
        """Move invalid file to quarantine folder with reason."""
        if not file_path.exists():
            return

        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = self.quarantine_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"

        try:
            file_path.rename(dest)

            # Write reason file
            reason_file = dest.with_suffix(".reason")
            reason_file.write_text(reason, encoding="utf-8")

            logger.info(f"Quarantined {file_path.name}: {reason}")
        except Exception as e:
            logger.error(f"Failed to quarantine {file_path.name}: {e}")

    def _validate_state_transition(self, from_status: str, to_status: str) -> None:
        """Validate that a state transition is allowed.

        Raises InvalidStateTransition if not allowed.
        """
        allowed = VALID_TRANSITIONS.get(from_status, set())
        if to_status not in allowed:
            raise InvalidStateTransition(
                f"Invalid transition from '{from_status}' to '{to_status}'. Allowed: {allowed}"
            )

    def complete_job(self, job: WarmingJob) -> None:
        """Mark job as completed and optionally archive/delete."""
        self._validate_state_transition(job.status, "completed")
        job.status = "completed"
        job.completed_at = datetime.now(UTC)
        job.locked_by = None
        job.locked_at = None
        self._atomic_write(job)

        logger.info(
            f"Warming job {job.id} completed: {job.success_count}/{job.total} queries successful"
        )

        # Archive if configured
        if self.archive_completed:
            self.archive_job(job)

    def fail_job(self, job: WarmingJob, error: str) -> None:
        """Mark job as failed with error message."""
        self._validate_state_transition(job.status, "failed")
        job.status = "failed"
        job.error = error
        job.completed_at = datetime.now(UTC)
        job.locked_by = None
        job.locked_at = None
        self._atomic_write(job)

        logger.error(f"Warming job {job.id} failed: {error}")

    def pause_job(self, job_id: str) -> WarmingJob | None:
        """Pause a running job.

        Args:
            job_id: ID of job to pause

        Returns:
            The paused job, or None if job not found or not running
        """
        job_path = self.job_path(job_id)

        with job_lock(job_path) as acquired:
            if not acquired:
                return None

            job = self.get_job(job_id)
            if job is None:
                return None

            # Can only pause running jobs
            if job.status != "running":
                return None

            # Transition to paused
            self._validate_state_transition(job.status, "paused")
            job.status = "paused"
            job.locked_by = None
            job.locked_at = None
            self._atomic_write(job)

            logger.info(f"Paused warming job {job_id}")
            return job

    def resume_job(self, job_id: str, worker_id: str) -> WarmingJob | None:
        """Resume a paused job.

        Args:
            job_id: ID of job to resume
            worker_id: Unique identifier for the worker resuming the job

        Returns:
            The resumed job, or None if job not found or not paused
        """
        job_path = self.job_path(job_id)

        with job_lock(job_path) as acquired:
            if not acquired:
                return None

            job = self.get_job(job_id)
            if job is None:
                return None

            # Can only resume paused jobs
            if job.status != "paused":
                return None

            # Transition to running
            self._validate_state_transition(job.status, "running")
            job.status = "running"
            job.locked_by = worker_id
            job.locked_at = datetime.now(UTC)
            self._atomic_write(job)

            logger.info(f"Resumed warming job {job_id} by {worker_id}")
            return job
