---
title: Cache Warming Persistent Queue
status: draft
created: 2026-02-02
updated: 2026-02-02
author: admin
type: Enhancement
complexity: MEDIUM
stack: backend
---

# Cache Warming Persistent Queue

## Summary

Replace the in-memory `_warming_jobs` dictionary with a file-based queue that persists warming jobs to disk. Jobs survive server restarts and auto-resume processing on startup. Users can also drop files directly into the queue folder for CLI-based warming.

## Goals

1. Warming jobs survive server restarts without losing progress
2. Auto-resume pending jobs on server startup
3. Support CLI access by dropping files into queue folder
4. Maintain current SSE progress streaming for UI
5. Track progress within job files to resume from exact position
6. **Crash-safe persistence with atomic writes**
7. **Single-writer concurrency model to prevent corruption**

## Scope

### In Scope

- File-based queue in `data/warming_queue/`
- JSON job files with queries + progress metadata
- Auto-resume on server startup (lifespan event)
- Folder watching for new files (CLI access)
- Delete job files after successful completion
- Maintain existing API endpoints (backward compatible)
- **Atomic writes with crash recovery**
- **Job ownership/locking model**
- **State machine enforcement**
- **Malformed file handling with quarantine**

### Out of Scope

- Database-backed queue (SQLite/Redis)
- Multi-server distributed queue
- Job scheduling/cron
- Priority queues (beyond FIFO)
- Partial retry (retry full job only)

---

## Technical Specification

### 1. Queue Directory Structure

```
data/
└── warming_queue/
    ├── jobs/                    # Active job files
    │   ├── job_abc123.json
    │   └── job_def456.json
    ├── quarantine/              # Malformed/invalid files
    │   └── bad_file_20260202.txt
    ├── archive/                 # Optional: completed job logs
    │   └── job_abc123_completed.json
    └── README.txt               # Instructions for CLI users
```

### 2. Job File Format

**File:** `jobs/job_{uuid}.json`

```json
{
  "id": "abc123-uuid",
  "version": 1,
  "queries": [
    "What is our return policy?",
    "How do I request time off?",
    "What are security requirements?"
  ],
  "total": 3,
  "status": "pending",
  "processed_index": 0,
  "failed_indices": [],
  "success_count": 0,
  "triggered_by": "user-uuid-or-cli",
  "created_at": "2026-02-02T10:00:00Z",
  "started_at": null,
  "completed_at": null,
  "locked_by": null,
  "locked_at": null,
  "error": null
}
```

**Key changes from original:**
- `processed_index: int` instead of `processed_queries: list` (prevents duplicate query issues, reduces IO)
- `failed_indices: list[int]` instead of `failed_queries: list[str]` (track by position)
- `locked_by: str | null` — worker ID that owns this job
- `locked_at: str | null` — when lock was acquired
- `version: int` — for future schema migrations

### 3. Job State Machine

```
                    ┌─────────────────────────────────────┐
                    │                                     │
                    ▼                                     │
    ┌─────────┐  acquire  ┌─────────┐  complete  ┌───────────┐
    │ pending │ ────────► │ running │ ─────────► │ completed │
    └─────────┘           └─────────┘            └───────────┘
                               │
                               │ error
                               ▼
                          ┌────────┐
                          │ failed │
                          └────────┘
```

**Allowed transitions:**

| From | To | Trigger |
|------|----|---------|
| `pending` | `running` | Worker acquires lock |
| `running` | `completed` | All queries processed |
| `running` | `failed` | Unrecoverable error |
| `running` | `pending` | Lock expired (worker crashed) |

**Invalid transitions:** Any not listed above raises `InvalidStateTransition` error.

### 4. Concurrency & Locking Strategy

**Single-writer model:** Only one worker processes a job at a time.

```python
import fcntl
from contextlib import contextmanager

@contextmanager
def job_lock(job_path: Path):
    """Acquire exclusive file lock for job operations."""
    lock_path = job_path.with_suffix(".lock")
    with open(lock_path, "w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield True
        except BlockingIOError:
            yield False  # Lock held by another process
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
```

**Job ownership:**

```python
def acquire_job(self, job_id: str, worker_id: str) -> WarmingJob | None:
    """Attempt to acquire job for processing."""
    with job_lock(self.job_path(job_id)) as acquired:
        if not acquired:
            return None

        job = self.get_job(job_id)
        if job is None or job.status != "pending":
            return None

        # Check for stale lock (worker crashed)
        if job.locked_by and job.locked_at:
            lock_age = datetime.now(UTC) - job.locked_at
            if lock_age < timedelta(minutes=30):  # Lock still valid
                return None
            # Stale lock, reclaim job
            logger.warning(f"Reclaiming stale job {job_id} from {job.locked_by}")

        job.status = "running"
        job.locked_by = worker_id
        job.locked_at = datetime.now(UTC)
        job.started_at = job.started_at or datetime.now(UTC)
        self.update_job(job)
        return job
```

**Lock timeout:** 30 minutes. If a worker crashes, another worker can reclaim after timeout.

### 5. Atomic Writes & Crash Recovery

**Write pattern:** temp file → fsync → rename

```python
import os
import tempfile

def update_job(self, job: WarmingJob) -> None:
    """Atomically write job state to file."""
    job_path = self.job_path(job.id)

    # Write to temp file in same directory (ensures same filesystem)
    fd, temp_path = tempfile.mkstemp(
        dir=self.queue_dir / "jobs",
        prefix=f".job_{job.id}_",
        suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
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
```

**Load with validation:**

```python
def get_job(self, job_id: str) -> WarmingJob | None:
    """Load and validate job from file."""
    job_path = self.job_path(job_id)
    if not job_path.exists():
        return None

    try:
        with open(job_path) as f:
            data = json.load(f)

        # Validate required fields
        required = ["id", "queries", "total", "status", "processed_index"]
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Validate status is known
        if data["status"] not in ("pending", "running", "completed", "failed"):
            raise ValueError(f"Invalid status: {data['status']}")

        return WarmingJob.from_dict(data)

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error(f"Corrupted job file {job_id}: {e}")
        self._quarantine_file(job_path, reason=str(e))
        return None
```

### 6. Progress Tracking (By Index)

**Avoids duplicate query issues and reduces IO:**

```python
async def process_warming_job(job: WarmingJob, queue_service: WarmingQueueService):
    """Process job queries starting from processed_index."""
    worker_id = f"worker-{uuid.uuid4().hex[:8]}"

    # Acquire ownership
    job = queue_service.acquire_job(job.id, worker_id)
    if job is None:
        logger.info(f"Job {job.id} already owned by another worker")
        return

    try:
        # Resume from where we left off
        for i in range(job.processed_index, job.total):
            query = job.queries[i]

            try:
                result = await process_single_query(query)
                job.success_count += 1
            except Exception as e:
                logger.error(f"Query {i} failed: {e}")
                job.failed_indices.append(i)

            # Update progress (checkpoint)
            job.processed_index = i + 1
            queue_service.update_job(job)

            # Throttle
            await asyncio.sleep(settings.warming_delay_seconds)

        # Complete
        job.status = "completed"
        job.completed_at = datetime.now(UTC)
        job.locked_by = None
        job.locked_at = None
        queue_service.update_job(job)

        # Delete on success (or archive)
        if settings.warming_archive_completed:
            queue_service.archive_job(job)
        queue_service.delete_job(job.id)

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.locked_by = None
        queue_service.update_job(job)
        raise
```

### 7. Queue Ordering (FIFO)

Jobs processed in `created_at` ascending order:

```python
def list_pending_jobs(self) -> list[WarmingJob]:
    """Find all pending jobs, ordered by created_at (FIFO)."""
    jobs = []
    for job_file in (self.queue_dir / "jobs").glob("job_*.json"):
        job = self.get_job(job_file.stem.replace("job_", ""))
        if job and job.status in ("pending", "running"):
            jobs.append(job)

    # Sort by created_at for deterministic FIFO ordering
    return sorted(jobs, key=lambda j: j.created_at or datetime.min)
```

### 8. SSE Behavior on Resume

When client connects/reconnects to SSE:

```python
@router.get("/cache/warm-progress/{job_id}")
async def warm_progress_sse(job_id: str, ...):
    """SSE endpoint for warming progress."""

    async def event_generator():
        job = queue_service.get_job(job_id)
        if job is None:
            yield {"event": "error", "data": json.dumps({"error": "Job not found"})}
            return

        # Emit current status immediately on connect
        yield {
            "event": "status",
            "data": json.dumps({
                "job_id": job.id,
                "status": job.status,
                "processed": job.processed_index,
                "total": job.total,
                "success_count": job.success_count,
                "failed_count": len(job.failed_indices),
            })
        }

        # If already completed/failed, done
        if job.status in ("completed", "failed"):
            yield {"event": job.status, "data": json.dumps(job.to_dict())}
            return

        # Poll for updates
        last_processed = job.processed_index
        while True:
            await asyncio.sleep(1)
            job = queue_service.get_job(job_id)

            if job is None:
                yield {"event": "error", "data": json.dumps({"error": "Job disappeared"})}
                return

            if job.processed_index > last_processed:
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "processed": job.processed_index,
                        "total": job.total,
                    })
                }
                last_processed = job.processed_index

            if job.status in ("completed", "failed"):
                yield {"event": job.status, "data": json.dumps(job.to_dict())}
                return

    return EventSourceResponse(event_generator())
```

### 9. Supported Input File Formats

| Format | Structure | Example |
|--------|-----------|---------|
| `.txt` | One query per line | `What is the return policy?` |
| `.csv` | One query per line (no headers) | `How do I reset my password?` |

Both formats are treated identically: split by newline, strip numbering prefixes (e.g., "1. Question" → "Question"), skip empty lines.

### 10. Folder Watcher with Validation

```python
# Config
warming_scan_interval_seconds: int = 60
warming_max_file_size_mb: float = 10.0
warming_allowed_extensions: list[str] = [".txt", ".csv"]  # Both: one query per line

async def folder_watcher(queue_service: WarmingQueueService):
    """Watch for new files with validation and quarantine."""
    while True:
        try:
            for file_path in queue_service.queue_dir.iterdir():
                # Skip directories and job files
                if file_path.is_dir() or file_path.suffix == ".json":
                    continue

                # Validate extension
                if file_path.suffix.lower() not in settings.warming_allowed_extensions:
                    logger.warning(f"Invalid extension: {file_path.name}")
                    queue_service._quarantine_file(file_path, "invalid_extension")
                    continue

                # Validate size
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > settings.warming_max_file_size_mb:
                    logger.warning(f"File too large: {file_path.name} ({size_mb:.1f}MB)")
                    queue_service._quarantine_file(file_path, "file_too_large")
                    continue

                # Try to read and parse
                try:
                    content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError as e:
                    logger.warning(f"Encoding error: {file_path.name}")
                    queue_service._quarantine_file(file_path, f"encoding_error: {e}")
                    continue

                # Parse queries
                queries = [
                    _strip_numbering(line)
                    for line in content.strip().split("\n")
                    if line.strip()
                ]

                if not queries:
                    logger.warning(f"Empty file: {file_path.name}")
                    file_path.unlink()
                    continue

                # Create job and remove source file
                job = queue_service.create_job(queries, triggered_by="cli")
                file_path.unlink()
                logger.info(f"Created job {job.id} from {file_path.name} ({len(queries)} queries)")
                asyncio.create_task(process_warming_job(job, queue_service))

        except Exception as e:
            logger.error(f"Folder watcher error: {e}")

        await asyncio.sleep(settings.warming_scan_interval_seconds)

def _quarantine_file(self, file_path: Path, reason: str) -> None:
    """Move invalid file to quarantine folder."""
    quarantine_dir = self.queue_dir / "quarantine"
    quarantine_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = quarantine_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
    file_path.rename(dest)

    # Write reason file
    (dest.with_suffix(".reason")).write_text(reason)
    logger.info(f"Quarantined {file_path.name}: {reason}")
```

### 11. Config Changes

**File:** `ai_ready_rag/config.py`

```python
# Warming queue settings
warming_queue_dir: str = "data/warming_queue"
warming_scan_interval_seconds: int = 60  # Folder watcher interval
warming_failed_job_retention_days: int = 7  # Auto-delete failed jobs
warming_lock_timeout_minutes: int = 30  # Reclaim stale locks after
warming_max_file_size_mb: float = 10.0  # Max uploaded file size
warming_allowed_extensions: list[str] = [".txt", ".csv"]  # Both: one query per line
warming_archive_completed: bool = False  # Archive completed jobs for audit
warming_checkpoint_interval: int = 1  # Save progress every N queries
```

### 12. Audit Logging (Optional)

If `warming_archive_completed = True`:

```python
def archive_job(self, job: WarmingJob) -> None:
    """Archive completed job for audit trail."""
    archive_dir = self.queue_dir / "archive"
    archive_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"job_{job.id}_{timestamp}.json"

    archive_data = job.to_dict()
    archive_data["archived_at"] = datetime.now(UTC).isoformat()

    with open(archive_path, "w") as f:
        json.dump(archive_data, f, indent=2, default=str)

    logger.info(f"Archived job {job.id} to {archive_path.name}")
```

---

## Files to Modify

| File | Change |
|------|--------|
| `ai_ready_rag/services/warming_queue.py` | **NEW** - Queue service with locking, atomic writes |
| `ai_ready_rag/api/admin.py` | Replace `_warming_jobs` dict with service |
| `ai_ready_rag/main.py` | Add auto-resume in lifespan, start folder watcher |
| `ai_ready_rag/config.py` | Add queue settings |

---

## Design Decisions

| Question | Decision |
|----------|----------|
| Failed job retention | Auto-delete after 7 days |
| Folder watcher | Continuous (configurable interval, default 60s) |
| Progress tracking | By index (not string matching) |
| Concurrency | Single-writer with file locks (fcntl) |
| Crash recovery | Atomic writes (temp + fsync + rename) |
| Queue ordering | FIFO by created_at |
| Malformed files | Quarantine folder with reason file |
| Lock timeout | 30 minutes (reclaim stale locks) |

---

## Test Plan

### Unit Tests

| Test | Description |
|------|-------------|
| `test_atomic_write_crash_recovery` | Kill process mid-write, verify no corruption |
| `test_job_state_transitions` | Valid transitions succeed, invalid raise error |
| `test_duplicate_queries_by_index` | Same query at different indices both processed |
| `test_lock_acquisition` | Only one worker acquires pending job |
| `test_stale_lock_reclaim` | Job with expired lock can be reclaimed |
| `test_json_validation_on_load` | Corrupted JSON quarantined, not crash |

### Integration Tests

| Test | Description |
|------|-------------|
| `test_restart_recovery` | Stop server mid-job, restart, verify resume from exact position |
| `test_concurrent_sse_and_updates` | SSE client receives updates while job progresses |
| `test_folder_watcher_ingestion` | Drop .txt file, verify job created within interval |
| `test_folder_watcher_quarantine` | Drop invalid file, verify quarantined with reason |
| `test_retry_flow` | Failed job remains, can be retried |
| `test_cleanup_old_failed_jobs` | Jobs older than 7 days auto-deleted |

### Load Tests

| Test | Description |
|------|-------------|
| `test_large_job_io_performance` | 1000 queries, verify checkpoint performance |
| `test_multiple_concurrent_jobs` | 5 jobs processing simultaneously |

---

## Acceptance Criteria

- [ ] Uploading file creates job in `data/warming_queue/jobs/`
- [ ] Server restart does not lose pending jobs
- [ ] Jobs auto-resume on server startup from exact `processed_index`
- [ ] Dropping `.txt` file in queue folder starts processing within configured interval
- [ ] Job file deleted after successful completion
- [ ] Failed jobs auto-deleted after 7 days
- [ ] Malformed files quarantined with reason
- [ ] Only one worker processes a job at a time (lock enforced)
- [ ] Crash mid-write does not corrupt job file (atomic writes)
- [ ] Duplicate queries at different indices all processed
- [ ] SSE emits current status immediately on connect
- [ ] Existing API endpoints remain backward compatible
- [ ] Queue processes jobs in FIFO order (created_at)
- [ ] Stale locks (>30min) reclaimed automatically

---

**Next Steps**:
1. Review this spec with engineering team
2. Run `/spec-review specs/CACHE_WARMING_PERSISTENT_QUEUE.md --create-issues`
