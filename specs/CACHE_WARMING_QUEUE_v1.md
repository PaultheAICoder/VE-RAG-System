---
title: Cache Warming Persistent Queue
status: DRAFT
version: 1.2
created: 2026-02-03
updated: 2026-02-03
author: jjob
reviewers: Engineering Team
type: Fullstack
complexity: COMPLEX
---

# Cache Warming Persistent Queue

## Summary

A persistent, non-blocking queue system for cache warming that processes queries from multiple sources (manual entry, file upload, SCTP) using an async background worker. The queue survives server restarts and supports pause/resume at the query level with proper concurrency control.

## Goals

- Non-blocking UX: Users can add items while processing continues
- Persistent queue: Survives server restarts, resumes exactly where stopped
- Multiple sources: Manual entry, file upload, SCTP protocol
- Fine-grained control: Pause/resume at query level, not just file level
- Reliability: Crash recovery, atomic checkpoints, proper locking
- DB-agnostic: No dialect-specific SQL (portable across SQLite/Postgres)

## Non-Goals

- Real-time collaborative editing of queue
- Priority ordering (strictly FIFO)
- Distributed processing across multiple workers

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           SOURCES                                   │
├─────────────┬─────────────────┬─────────────────┬───────────────────┤
│ Manual Entry│   File Upload   │      SCTP       │   Future Sources  │
│  (UI form)  │   (UI upload)   │ (external push) │                   │
└──────┬──────┴────────┬────────┴────────┬────────┴───────────────────┘
       │               │                 │
       ▼               ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FILE STORAGE (Immutable Payloads)               │
│                   uploads/warming/                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                   │
│  │manual_*.txt │ │ upload_*.csv│ │ sctp_*.txt  │                   │
│  └─────────────┘ └─────────────┘ └─────────────┘                   │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     QUEUE (SQLite - Source of Truth)                │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ id │ file_path │ total │ processed │ status  │ byte_offset   │ │
│  ├────┼───────────┼───────┼───────────┼─────────┼───────────────┤ │
│  │ 1  │ manual_.. │ 30    │ 18        │ running │ 4096          │ │
│  │ 2  │ upload_.. │ 45    │ 0         │ pending │ 0             │ │
│  │ 3  │ sctp_..   │ 120   │ 0         │ pending │ 0             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  Source of Truth Model:                                             │
│  • DB holds all state (status, progress, byte_offset)               │
│  • Files are immutable payloads (never modified after creation)     │
│  • Checkpoint = atomic DB transaction                               │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ASYNC BACKGROUND WORKER                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • Runs as asyncio.Task (not Thread)                          │   │
│  │ • Acquires job lease before processing                       │   │
│  │ • Streams file line-by-line (not readlines())                │   │
│  │ • Batch checkpoints every 10 queries or 5 seconds            │   │
│  │ • Independent lease renewal every 60s                        │   │
│  │ • Pause state persisted in DB                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     CACHE ENTRIES                                   │
│                   (Existing cache_entries table)                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Worker State Machine

```
                    ┌──────────┐
        ┌──────────►│  IDLE    │◄─────────────────────┐
        │           └────┬─────┘                      │
        │                │ Queue has pending item     │
        │                │ (acquire lease)            │
        │                ▼                            │
        │           ┌──────────┐                      │
        │  ┌───────►│ RUNNING  │                      │
        │  │        └────┬─────┘                      │
        │  │             │                            │
        │  │  ┌──────────┼──────────┬────────────┐   │
        │  │  │          │          │            │   │
        │  │  ▼          ▼          ▼            ▼   │
        │  │┌──────┐ ┌────────┐ ┌────────┐ ┌────────┐│
        │  ││PAUSED│ │COMPLETE│ │ FAILED │ │CANCELED││
        │  │└──┬───┘ └───┬────┘ └───┬────┘ └───┬────┘│
        │  │   │         │          │          │     │
        │  │   │         └──────────┴──────────┴─────┘
        │  │   │              Release lease
        │  └───┘              Move to next
        │ Resume
        │ (DB flag)
        │
        └─── Check DB pause flag each iteration
```

---

## Configuration

### Environment Variables

```env
# Worker settings
WARMING_CHECKPOINT_INTERVAL=10           # Queries between checkpoints
WARMING_CHECKPOINT_TIME_SECONDS=5        # Max seconds between checkpoints
WARMING_LEASE_DURATION_MINUTES=10        # Job lease duration
WARMING_LEASE_RENEWAL_SECONDS=60         # Lease renewal interval

# Retry settings
WARMING_MAX_RETRIES=3
WARMING_RETRY_DELAYS=5,30,120            # Comma-separated seconds

# SCTP settings
SCTP_ENABLED=false
SCTP_HOST=0.0.0.0
SCTP_PORT=9900
SCTP_MAX_FILE_SIZE_MB=10                 # Configurable file size limit
SCTP_MAX_QUERIES_PER_FILE=10000
SCTP_TLS_CERT=/etc/ssl/sctp/server.crt
SCTP_TLS_KEY=/etc/ssl/sctp/server.key
SCTP_TLS_CA=/etc/ssl/sctp/ca.crt
SCTP_SHARED_SECRET=your-256-bit-secret
SCTP_ALLOWED_IPS=10.0.0.0/8,192.168.0.0/16

# Cleanup settings
WARMING_COMPLETED_RETENTION_DAYS=7
WARMING_FAILED_RETENTION_DAYS=30
WARMING_CLEANUP_INTERVAL_HOURS=6

# SSE settings
SSE_EVENT_BUFFER_SIZE=1000               # Ring buffer size for replay
SSE_HEARTBEAT_SECONDS=30
```

---

## Data Model

### New Table: `warming_queue`

```sql
CREATE TABLE warming_queue (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,              -- Path to query file (immutable)
    file_checksum TEXT NOT NULL,          -- SHA256 for integrity verification
    source_type TEXT NOT NULL,            -- 'manual' | 'upload' | 'sctp'
    original_filename TEXT,               -- User-friendly name
    total_queries INTEGER NOT NULL,       -- Total queries in file
    processed_queries INTEGER DEFAULT 0,
    failed_queries INTEGER DEFAULT 0,
    byte_offset INTEGER DEFAULT 0,        -- File position for resume (from f.tell() only)
    status TEXT DEFAULT 'pending',        -- pending | running | paused | completed | failed | cancelled
    is_paused BOOLEAN DEFAULT FALSE,      -- Persisted pause flag (survives restart)
    is_cancel_requested BOOLEAN DEFAULT FALSE,  -- Graceful cancel flag
    worker_id TEXT,                       -- Lease: which worker owns this job
    worker_lease_expires_at TIMESTAMP,    -- Lease expiry (Python datetime, not DB function)
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_by TEXT,                      -- User ID who added
    FOREIGN KEY (created_by) REFERENCES users(id)
);

-- FIFO ordering index
CREATE INDEX idx_warming_queue_fifo ON warming_queue(status, created_at ASC);
CREATE INDEX idx_warming_queue_status ON warming_queue(status);
CREATE INDEX idx_warming_queue_completed ON warming_queue(completed_at);
CREATE INDEX idx_warming_queue_lease ON warming_queue(worker_lease_expires_at);
```

### New Table: `warming_failed_queries`

```sql
CREATE TABLE warming_failed_queries (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    query TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    error_message TEXT,
    error_type TEXT,                      -- Exception class name for retry logic
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES warming_queue(id) ON DELETE CASCADE
);

CREATE INDEX idx_warming_failed_job ON warming_failed_queries(job_id);
```

### New Table: `warming_sse_events` (Ring Buffer for Replay)

```sql
CREATE TABLE warming_sse_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Auto-incrementing for ordering
    event_id TEXT NOT NULL UNIQUE,         -- UUID for client tracking
    event_type TEXT NOT NULL,              -- 'progress', 'job_started', etc.
    job_id TEXT,                           -- Associated job (nullable for heartbeats)
    payload TEXT NOT NULL,                 -- JSON event data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sse_events_job ON warming_sse_events(job_id);
CREATE INDEX idx_sse_events_created ON warming_sse_events(created_at);

-- Trigger to maintain ring buffer size (keeps last 1000 events)
-- Note: Implemented in Python cleanup, not DB trigger for portability
```

### Ring Buffer Management (Python)

```python
async def store_sse_event(
    db: AsyncSession,
    event_type: str,
    job_id: str | None,
    payload: dict
) -> str:
    """Store SSE event and maintain ring buffer."""
    event_id = str(uuid4())

    await db.execute(
        text("""
            INSERT INTO warming_sse_events (event_id, event_type, job_id, payload, created_at)
            VALUES (:event_id, :event_type, :job_id, :payload, :created_at)
        """),
        {
            "event_id": event_id,
            "event_type": event_type,
            "job_id": job_id,
            "payload": json.dumps(payload),
            "created_at": datetime.utcnow()
        }
    )

    # Prune old events beyond buffer size
    buffer_size = settings.SSE_EVENT_BUFFER_SIZE
    await db.execute(
        text("""
            DELETE FROM warming_sse_events
            WHERE id NOT IN (
                SELECT id FROM warming_sse_events
                ORDER BY id DESC
                LIMIT :buffer_size
            )
        """),
        {"buffer_size": buffer_size}
    )

    await db.commit()
    return event_id

async def get_events_since(db: AsyncSession, last_event_id: str) -> list[dict]:
    """Get events after a specific event_id for replay."""
    result = await db.execute(
        text("""
            SELECT event_id, event_type, job_id, payload, created_at
            FROM warming_sse_events
            WHERE id > (
                SELECT COALESCE(
                    (SELECT id FROM warming_sse_events WHERE event_id = :last_id),
                    0
                )
            )
            ORDER BY id ASC
        """),
        {"last_id": last_event_id}
    )
    return [dict(row._mapping) for row in result.fetchall()]
```

---

## Concurrency Control: Job Leasing (DB-Agnostic)

```python
from datetime import datetime, timedelta

async def acquire_job_lease(db: AsyncSession, worker_id: str) -> WarmingJob | None:
    """Atomically acquire next pending job with lease.

    Uses Python datetime for DB-agnostic time calculations.
    """
    now = datetime.utcnow()
    lease_expires = now + timedelta(minutes=settings.WARMING_LEASE_DURATION_MINUTES)

    # First, find the next eligible job
    result = await db.execute(
        text("""
            SELECT id FROM warming_queue
            WHERE status IN ('pending', 'paused')
              AND is_paused = FALSE
              AND is_cancel_requested = FALSE
            ORDER BY created_at ASC
            LIMIT 1
        """)
    )
    row = result.fetchone()
    if not row:
        return None

    job_id = row.id

    # Atomically claim it (check status again to prevent race)
    result = await db.execute(
        text("""
            UPDATE warming_queue
            SET status = 'running',
                worker_id = :worker_id,
                worker_lease_expires_at = :lease_expires,
                started_at = COALESCE(started_at, :now)
            WHERE id = :job_id
              AND status IN ('pending', 'paused')
              AND is_paused = FALSE
            RETURNING *
        """),
        {
            "worker_id": worker_id,
            "lease_expires": lease_expires,
            "now": now,
            "job_id": job_id
        }
    )
    await db.commit()

    row = result.fetchone()
    return WarmingJob(**row._mapping) if row else None

async def renew_lease(db: AsyncSession, job_id: str, worker_id: str) -> bool:
    """Renew lease during processing. Called every 60s independent of checkpoints."""
    lease_expires = datetime.utcnow() + timedelta(minutes=settings.WARMING_LEASE_DURATION_MINUTES)

    result = await db.execute(
        text("""
            UPDATE warming_queue
            SET worker_lease_expires_at = :lease_expires
            WHERE id = :job_id AND worker_id = :worker_id
        """),
        {"job_id": job_id, "worker_id": worker_id, "lease_expires": lease_expires}
    )
    await db.commit()
    return result.rowcount > 0

async def release_lease(db: AsyncSession, job_id: str, new_status: str) -> None:
    """Release lease and update final status."""
    now = datetime.utcnow()

    await db.execute(
        text("""
            UPDATE warming_queue
            SET status = :status,
                worker_id = NULL,
                worker_lease_expires_at = NULL,
                completed_at = :completed_at
            WHERE id = :job_id
        """),
        {
            "job_id": job_id,
            "status": new_status,
            "completed_at": now if new_status in ('completed', 'failed', 'cancelled') else None
        }
    )
    await db.commit()
```

---

## API Endpoints

### Queue Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/warming/queue` | List all queue items (with filters) |
| GET | `/api/admin/warming/queue/completed` | List completed jobs (today by default) |
| POST | `/api/admin/warming/queue/manual` | Add manual queries to queue |
| POST | `/api/admin/warming/queue/upload` | Upload file to queue |
| DELETE | `/api/admin/warming/queue/{id}` | Delete single item |
| DELETE | `/api/admin/warming/queue/bulk` | Delete multiple items |

### Job Control

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/warming/current` | Get currently processing job |
| POST | `/api/admin/warming/current/pause` | Pause current job (persisted) |
| POST | `/api/admin/warming/current/resume` | Resume paused job |
| POST | `/api/admin/warming/current/cancel` | Request graceful cancel |
| GET | `/api/admin/warming/progress` | SSE stream for progress |

### Cancel Behavior (Graceful)

**Problem**: Deleting file while worker reads causes race condition.

**Solution**: Graceful cancel with worker cooperation.

```python
@router.post("/api/admin/warming/current/cancel")
async def cancel_current_job(db: AsyncSession = Depends(get_db)):
    """Request graceful cancellation of current job.

    Sets is_cancel_requested=TRUE. Worker detects this, closes file,
    then transitions to 'cancelled' status. File deletion happens
    AFTER worker releases the file handle.
    """
    # Find running job
    result = await db.execute(
        text("SELECT id FROM warming_queue WHERE status = 'running' LIMIT 1")
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(404, "No running job")

    # Set cancel flag (worker will handle gracefully)
    await db.execute(
        text("""
            UPDATE warming_queue
            SET is_cancel_requested = TRUE
            WHERE id = :job_id
        """),
        {"job_id": row.id}
    )
    await db.commit()

    return {"job_id": row.id, "status": "cancel_requested"}
```

**Worker handles cancel:**
```python
async def _should_stop(self, db: AsyncSession, job_id: str) -> tuple[bool, str]:
    """Check if job should stop. Returns (should_stop, reason)."""
    result = await db.execute(
        text("""
            SELECT is_paused, is_cancel_requested, status
            FROM warming_queue WHERE id = :id
        """),
        {"id": job_id}
    )
    row = result.fetchone()
    if not row:
        return True, "job_deleted"
    if row.is_cancel_requested:
        return True, "cancelled"
    if row.is_paused:
        return True, "paused"
    if row.status == 'cancelled':
        return True, "cancelled"
    return False, ""

# In _process_job, after detecting cancel:
async def _handle_cancel(self, db: AsyncSession, job: WarmingJob, file_handle):
    """Gracefully handle job cancellation."""
    # 1. Close file handle first
    await file_handle.close()

    # 2. Update status
    await release_lease(db, job.id, "cancelled")

    # 3. Delete file (safe now that handle is closed)
    if os.path.exists(job.file_path):
        os.remove(job.file_path)

    # 4. Emit event
    await self._emit_event('job_cancelled', {
        'job_id': job.id,
        'processed': job.processed_queries,
        'total': job.total_queries
    })
```

---

## SSE Progress Contract

### Endpoint: `GET /api/admin/warming/progress`

### Event Types

```typescript
// All events include event_id for replay tracking
interface SSEEvent {
  event_id: string;  // UUID for replay
  // ... other fields
}

// Connection established
event: connected
data: {"event_id": "uuid", "worker_id": "worker-abc123", "timestamp": "2026-02-03T14:30:22Z"}

// Job started
event: job_started
data: {
  "event_id": "uuid",
  "job_id": "abc123",
  "file_path": "uploads/warming/hr_faq.txt",
  "total_queries": 30
}

// Progress update (every 10 queries or 5 seconds)
event: progress
data: {
  "event_id": "uuid",
  "job_id": "abc123",
  "processed": 18,
  "failed": 2,
  "total": 30,
  "percent": 60,
  "estimated_remaining_seconds": 45,
  "queries_per_second": 0.67,
  "current_query": "What is our PTO policy?"
}

// Query failed
event: query_failed
data: {
  "event_id": "uuid",
  "job_id": "abc123",
  "query": "Invalid query here",
  "line_number": 15,
  "error": "Embedding service timeout",
  "error_type": "EmbeddingTimeoutError"
}

// Job completed
event: job_completed
data: {
  "event_id": "uuid",
  "job_id": "abc123",
  "processed": 28,
  "failed": 2,
  "total": 30,
  "duration_seconds": 120,
  "queries_per_second": 0.23
}

// Job paused (by user)
event: job_paused
data: {"event_id": "uuid", "job_id": "abc123", "processed": 18, "total": 30}

// Job cancelled
event: job_cancelled
data: {"event_id": "uuid", "job_id": "abc123", "processed": 18, "total": 30, "file_deleted": true}

// Job failed (unrecoverable error)
event: job_failed
data: {"event_id": "uuid", "job_id": "abc123", "error": "File not found"}

// Heartbeat (every 30 seconds)
event: heartbeat
data: {"event_id": "uuid", "timestamp": "2026-02-03T14:31:00Z"}
```

### Server Reconnection with Replay

```python
@router.get("/api/admin/warming/progress")
async def sse_progress(
    request: Request,
    last_event_id: str | None = Query(None),
    resume_job: str | None = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """SSE endpoint with replay support."""

    async def event_generator():
        # 1. Replay missed events if reconnecting
        if last_event_id:
            missed_events = await get_events_since(db, last_event_id)
            for event in missed_events:
                yield format_sse(event)

        # 2. If resuming specific job, send current status
        if resume_job:
            job = await get_job(db, resume_job)
            if job and job.status == 'running':
                yield format_sse({
                    "event_type": "progress",
                    "event_id": str(uuid4()),
                    "job_id": job.id,
                    "processed": job.processed_queries,
                    "total": job.total_queries,
                    # ...
                })

        # 3. Stream new events
        async for event in worker.subscribe_events():
            yield format_sse(event)

            # Store in ring buffer for future replays
            await store_sse_event(db, event["event_type"], event.get("job_id"), event)

    return EventSourceResponse(event_generator())
```

---

## Background Worker (Async)

### Implementation: `WarmingWorker`

```python
import asyncio
import aiofiles
from uuid import uuid4
from datetime import datetime, timedelta
from collections import deque

# Retryable exception classes (not string matching)
RETRYABLE_EXCEPTIONS = (
    ConnectionTimeoutError,
    ServiceUnavailableError,
    RateLimitExceededError,
)

class WarmingWorker:
    """Async background worker that processes warming queue."""

    def __init__(self, db_session_factory, rag_service, settings):
        self.db_factory = db_session_factory
        self.rag_service = rag_service
        self.settings = settings
        self.worker_id = f"worker-{uuid4().hex[:8]}"
        self._task: asyncio.Task | None = None
        self._lease_task: asyncio.Task | None = None  # Independent lease renewal
        self._shutdown = asyncio.Event()
        self._current_job_id: str | None = None
        self._event_subscribers: list[asyncio.Queue] = []

        # Progress estimation (EMA)
        self._query_durations: deque[float] = deque(maxlen=20)

    async def start(self):
        """Start worker as asyncio task. Called on server boot."""
        self._shutdown.clear()
        self._task = asyncio.create_task(self._run_loop())
        self._lease_task = asyncio.create_task(self._lease_renewal_loop())

    async def stop(self):
        """Graceful shutdown."""
        self._shutdown.set()
        for task in [self._task, self._lease_task]:
            if task:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=10)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

    async def _lease_renewal_loop(self):
        """Independent lease renewal every 60s.

        Prevents lease expiry during long-running queries.
        """
        while not self._shutdown.is_set():
            try:
                await asyncio.sleep(self.settings.WARMING_LEASE_RENEWAL_SECONDS)

                if self._current_job_id:
                    async with self.db_factory() as db:
                        success = await renew_lease(db, self._current_job_id, self.worker_id)
                        if not success:
                            logger.warning(f"Failed to renew lease for job {self._current_job_id}")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Lease renewal error: {e}")

    async def _run_loop(self):
        """Main worker loop."""
        while not self._shutdown.is_set():
            try:
                async with self.db_factory() as db:
                    job = await acquire_job_lease(db, self.worker_id)
                    if not job:
                        await asyncio.sleep(1)
                        continue

                    self._current_job_id = job.id
                    try:
                        await self._process_job(db, job)
                    finally:
                        self._current_job_id = None

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(5)

    async def _process_job(self, db: AsyncSession, job: WarmingJob):
        """Process a single job, streaming line-by-line with batch checkpoints."""

        # Verify file integrity
        if not await self._verify_file(job):
            await self._fail_job(db, job, "File missing or corrupted")
            return

        await self._emit_event('job_started', {
            'job_id': job.id,
            'file_path': job.file_path,
            'total_queries': job.total_queries
        })

        last_checkpoint_time = asyncio.get_event_loop().time()
        queries_since_checkpoint = 0
        file_handle = None

        try:
            file_handle = await aiofiles.open(job.file_path, 'r')

            # Seek to saved byte offset (use f.tell() only, no manual byte math)
            await file_handle.seek(job.byte_offset)
            line_number = job.processed_queries

            async for line in file_handle:
                query_start_time = asyncio.get_event_loop().time()

                # Check for pause/cancel (from DB, not memory)
                should_stop, reason = await self._should_stop(db, job.id)
                if should_stop:
                    # Save current position BEFORE the line we're about to skip
                    job.byte_offset = await file_handle.tell()

                    if reason == "cancelled":
                        await self._handle_cancel(db, job, file_handle)
                        return
                    else:  # paused
                        await self._pause_job(db, job)
                        return

                query = line.strip()
                if not query:
                    continue

                line_number += 1

                # Process query with retry
                try:
                    await self._warm_query_with_retry(query)
                    job.processed_queries += 1

                    # Track duration for EMA
                    duration = asyncio.get_event_loop().time() - query_start_time
                    self._query_durations.append(duration)

                except Exception as e:
                    job.failed_queries += 1
                    await self._save_failed_query(
                        db, job.id, query, line_number,
                        str(e), type(e).__name__
                    )
                    await self._emit_event('query_failed', {
                        'job_id': job.id,
                        'query': query[:100],
                        'line_number': line_number,
                        'error': str(e),
                        'error_type': type(e).__name__
                    })

                queries_since_checkpoint += 1
                current_time = asyncio.get_event_loop().time()

                # Batch checkpoint
                if (queries_since_checkpoint >= self.settings.WARMING_CHECKPOINT_INTERVAL or
                    current_time - last_checkpoint_time >= self.settings.WARMING_CHECKPOINT_TIME_SECONDS):

                    # Use f.tell() for byte offset - safe with UTF-8
                    job.byte_offset = await file_handle.tell()
                    await self._checkpoint(db, job)

                    await self._emit_event('progress', {
                        'job_id': job.id,
                        'processed': job.processed_queries,
                        'failed': job.failed_queries,
                        'total': job.total_queries,
                        'percent': int(job.processed_queries / job.total_queries * 100),
                        'estimated_remaining_seconds': self._estimate_remaining(job),
                        'queries_per_second': self._calculate_qps()
                    })

                    queries_since_checkpoint = 0
                    last_checkpoint_time = current_time

            # Job completed successfully
            await file_handle.close()
            await self._complete_job(db, job)

        except Exception as e:
            if file_handle:
                await file_handle.close()
            await self._fail_job(db, job, str(e))

    def _estimate_remaining(self, job: WarmingJob) -> int:
        """Estimate remaining seconds using Exponential Moving Average (EMA)."""
        if not self._query_durations:
            return 0

        # EMA with alpha=0.3 (weight recent queries more)
        alpha = 0.3
        ema = self._query_durations[0]
        for duration in list(self._query_durations)[1:]:
            ema = alpha * duration + (1 - alpha) * ema

        remaining_queries = job.total_queries - job.processed_queries
        return int(remaining_queries * ema)

    def _calculate_qps(self) -> float:
        """Calculate queries per second from recent durations."""
        if not self._query_durations:
            return 0.0
        avg_duration = sum(self._query_durations) / len(self._query_durations)
        return round(1.0 / avg_duration, 2) if avg_duration > 0 else 0.0

    async def _warm_query_with_retry(self, query: str) -> bool:
        """Warm query with retry logic using exception classes."""
        last_error = None
        delays = [int(d) for d in self.settings.WARMING_RETRY_DELAYS.split(',')]

        for attempt in range(self.settings.WARMING_MAX_RETRIES + 1):
            try:
                await self.rag_service.warm_cache(query)
                return True
            except RETRYABLE_EXCEPTIONS as e:
                last_error = e
                if attempt < self.settings.WARMING_MAX_RETRIES:
                    delay = delays[min(attempt, len(delays) - 1)]
                    await asyncio.sleep(delay)
            except Exception as e:
                raise  # Non-retryable, fail immediately

        raise last_error  # All retries exhausted
```

### Server Integration

```python
# main.py
from contextlib import asynccontextmanager
from ai_ready_rag.services.warming_worker import WarmingWorker
from ai_ready_rag.services.warming_cleanup import WarmingCleanupService

worker: WarmingWorker | None = None
cleanup_service: WarmingCleanupService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker, cleanup_service

    # Startup
    worker = WarmingWorker(get_async_db, rag_service, settings)
    cleanup_service = WarmingCleanupService(get_async_db, settings)

    await worker.start()
    await cleanup_service.start()

    # Recover any jobs that were running when server crashed
    await recover_stale_jobs()

    yield

    # Shutdown
    if worker:
        await worker.stop()
    if cleanup_service:
        await cleanup_service.stop()

app = FastAPI(lifespan=lifespan)

async def recover_stale_jobs():
    """Reset jobs that were running when server crashed."""
    now = datetime.utcnow()

    async with get_async_db() as db:
        # Find jobs with expired leases
        await db.execute(
            text("""
                UPDATE warming_queue
                SET status = 'pending',
                    worker_id = NULL,
                    worker_lease_expires_at = NULL
                WHERE status = 'running'
                  AND worker_lease_expires_at < :now
            """),
            {"now": now}
        )
        await db.commit()
```

---

## SCTP Integration

### Security Model

**Authentication**: Mutual TLS (mTLS) + Shared Secret

```env
# .env - All SCTP settings configurable
SCTP_ENABLED=false
SCTP_HOST=0.0.0.0
SCTP_PORT=9900
SCTP_MAX_FILE_SIZE_MB=10                 # Configurable (was hardcoded)
SCTP_MAX_QUERIES_PER_FILE=10000

# mTLS Configuration
SCTP_TLS_CERT=/etc/ssl/sctp/server.crt
SCTP_TLS_KEY=/etc/ssl/sctp/server.key
SCTP_TLS_CA=/etc/ssl/sctp/ca.crt

# Additional shared secret (defense in depth)
SCTP_SHARED_SECRET=your-256-bit-secret-here

# IP allowlist (optional, additional layer)
SCTP_ALLOWED_IPS=10.0.0.0/8,192.168.0.0/16
```

### Ingestion with Staging (Configurable Limits)

```python
async def handle_sctp_file(stream: SCTPStream, header: FileHeader) -> JobResponse:
    """Handle incoming SCTP file with staging and validation."""

    max_file_size = settings.SCTP_MAX_FILE_SIZE_MB * 1024 * 1024
    max_queries = settings.SCTP_MAX_QUERIES_PER_FILE

    # 1. Verify HMAC signature
    expected_hmac = hmac.new(
        settings.SCTP_SHARED_SECRET.encode(),
        header.payload,
        hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(header.hmac, expected_hmac):
        raise SecurityError("Invalid HMAC signature")

    # 2. Write to staging directory
    staging_path = f"uploads/warming/.staging/{uuid4()}.tmp"
    checksum = hashlib.sha256()
    total_bytes = 0

    async with aiofiles.open(staging_path, 'wb') as f:
        async for chunk in stream.read_chunks():
            await f.write(chunk)
            checksum.update(chunk)
            total_bytes += len(chunk)

            # Configurable file size limit
            if total_bytes > max_file_size:
                os.remove(staging_path)
                raise ValidationError(f"File too large (max {settings.SCTP_MAX_FILE_SIZE_MB}MB)")

    # 3. Verify checksum matches header
    if checksum.hexdigest() != header.checksum:
        os.remove(staging_path)
        raise ValidationError("Checksum mismatch - transfer corrupted")

    # 4. Validate content
    query_count = await validate_query_file(staging_path)
    if query_count > max_queries:
        os.remove(staging_path)
        raise ValidationError(f"Too many queries: {query_count} (max {max_queries})")

    # 5. Move to final location (atomic on same filesystem)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    final_path = f"uploads/warming/sctp_{timestamp}_{header.filename}"
    os.rename(staging_path, final_path)

    # 6. Add to queue
    job = await create_queue_job(
        file_path=final_path,
        file_checksum=checksum.hexdigest(),
        source_type='sctp',
        total_queries=query_count
    )

    return JobResponse(job_id=job.id, position=await get_queue_position(job.id))
```

---

## Retry Policy (Exception Classes)

### Configuration

```python
from ai_ready_rag.core.exceptions import (
    ConnectionTimeoutError,
    ServiceUnavailableError,
    RateLimitExceededError,
    EmbeddingTimeoutError,
)

# Retryable exceptions (class references, not strings)
RETRYABLE_EXCEPTIONS = (
    ConnectionTimeoutError,
    ServiceUnavailableError,
    RateLimitExceededError,
    EmbeddingTimeoutError,
)

# Settings (from env)
WARMING_MAX_RETRIES = 3
WARMING_RETRY_DELAYS = "5,30,120"  # Exponential backoff
```

### Failed Query Schema

```python
async def _save_failed_query(
    self,
    db: AsyncSession,
    job_id: str,
    query: str,
    line_number: int,
    error_message: str,
    error_type: str  # Exception class name
):
    """Save failed query with error type for potential targeted retry."""
    await db.execute(
        text("""
            INSERT INTO warming_failed_queries
            (id, job_id, query, line_number, error_message, error_type, created_at)
            VALUES (:id, :job_id, :query, :line_number, :error_message, :error_type, :created_at)
        """),
        {
            "id": str(uuid4()),
            "job_id": job_id,
            "query": query,
            "line_number": line_number,
            "error_message": error_message,
            "error_type": error_type,  # e.g., "EmbeddingTimeoutError"
            "created_at": datetime.utcnow()
        }
    )
```

---

## Cleanup Service

```python
class WarmingCleanupService:
    """Periodic cleanup of old warming files and records."""

    def __init__(self, db_factory, settings):
        self.db_factory = db_factory
        self.settings = settings
        self._task: asyncio.Task | None = None

    async def start(self):
        """Start cleanup as periodic task."""
        self._task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        if self._task:
            self._task.cancel()

    async def _cleanup_loop(self):
        """Run cleanup every CLEANUP_INTERVAL_HOURS."""
        while True:
            try:
                await self._run_cleanup()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

            await asyncio.sleep(self.settings.WARMING_CLEANUP_INTERVAL_HOURS * 3600)

    async def _run_cleanup(self):
        """Perform cleanup operations."""
        now = datetime.utcnow()

        async with self.db_factory() as db:
            # 1. Delete old completed jobs
            cutoff_completed = now - timedelta(days=self.settings.WARMING_COMPLETED_RETENTION_DAYS)
            old_completed = await db.execute(
                text("""
                    SELECT id, file_path FROM warming_queue
                    WHERE status = 'completed'
                      AND completed_at < :cutoff
                """),
                {"cutoff": cutoff_completed}
            )

            for row in old_completed:
                await self._delete_job_and_file(db, row.id, row.file_path)

            # 2. Delete old failed/cancelled jobs
            cutoff_failed = now - timedelta(days=self.settings.WARMING_FAILED_RETENTION_DAYS)
            old_failed = await db.execute(
                text("""
                    SELECT id, file_path FROM warming_queue
                    WHERE status IN ('failed', 'cancelled')
                      AND completed_at < :cutoff
                """),
                {"cutoff": cutoff_failed}
            )

            for row in old_failed:
                await self._delete_job_and_file(db, row.id, row.file_path)

            # 3. Prune SSE event buffer
            await db.execute(
                text("""
                    DELETE FROM warming_sse_events
                    WHERE id NOT IN (
                        SELECT id FROM warming_sse_events
                        ORDER BY id DESC
                        LIMIT :buffer_size
                    )
                """),
                {"buffer_size": self.settings.SSE_EVENT_BUFFER_SIZE}
            )

            await db.commit()

            # 4. Clean orphaned staging files (older than 1 hour)
            staging_dir = "uploads/warming/.staging"
            if os.path.exists(staging_dir):
                for f in os.listdir(staging_dir):
                    path = os.path.join(staging_dir, f)
                    if os.path.getmtime(path) < time.time() - 3600:
                        try:
                            os.remove(path)
                        except OSError:
                            pass  # File may be in use
```

---

## Migration Plan

### From Existing In-Memory Queue

The current `warming_queue.py` uses an in-memory queue. Migration steps:

1. **Phase 1: Add new tables** (non-breaking)
   - Create `warming_queue`, `warming_failed_queries`, `warming_sse_events` tables
   - New endpoints coexist with old

2. **Phase 2: Dual-write period**
   - New uploads write to both old and new queue
   - Old worker continues processing
   - New worker disabled

3. **Phase 3: Switch to new worker**
   - Enable new async worker
   - Disable old worker
   - Old endpoints redirect to new

4. **Phase 4: Cleanup**
   - Remove old queue code
   - Remove old endpoints
   - Update documentation

---

## Acceptance Criteria

### Core Functionality
- [ ] Manual queries saved to file, added to queue
- [ ] File upload adds to queue
- [ ] SCTP receives files with mTLS + HMAC auth, adds to queue
- [ ] FIFO processing order (ORDER BY created_at ASC)
- [ ] Async background worker starts on boot
- [ ] Worker processes queue continuously

### Concurrency & Reliability
- [ ] Job leasing prevents duplicate processing
- [ ] Stale leases recovered on restart
- [ ] Batch checkpointing (every 10 queries or 5s)
- [ ] Independent lease renewal (every 60s)
- [ ] File streamed line-by-line using f.tell() only
- [ ] DB is source of truth; files are immutable payloads
- [ ] All time math in Python (DB-agnostic)

### Pause/Resume/Cancel
- [ ] Pause persisted in DB (survives restart)
- [ ] Resume continues from exact byte offset
- [ ] Cancel is graceful (worker closes file first, then deletes)

### SSE
- [ ] Events stored in ring buffer table (1000 events)
- [ ] Client reconnection with last_event_id replays missed events
- [ ] Progress includes EMA-based estimated_remaining_seconds
- [ ] Heartbeat every 30 seconds

### Error Handling
- [ ] Retry policy uses exception classes (not string matching)
- [ ] 3 retries with exponential backoff [5s, 30s, 120s]
- [ ] Non-retryable errors fail immediately
- [ ] Failed queries logged with error_type

### Configuration
- [ ] All settings via environment variables
- [ ] SCTP file size limit configurable (SCTP_MAX_FILE_SIZE_MB)

### Cleanup
- [ ] Completed files deleted after 7 days (configurable)
- [ ] Failed files deleted after 30 days (configurable)
- [ ] Orphaned staging files cleaned after 1 hour
- [ ] SSE event buffer pruned to configured size
- [ ] Cleanup runs every 6 hours (configurable)

### Migration
- [ ] New tables created without breaking existing
- [ ] Existing jobs migrated
- [ ] Old code removed after validation

---

## Implementation Phases

### Phase 1: Core Queue (Backend) - 3 days
- [ ] Create new tables with migrations
- [ ] Implement async WarmingWorker with independent lease renewal
- [ ] Add job leasing with Python datetime (DB-agnostic)
- [ ] Add queue management API endpoints
- [ ] Integrate worker with server lifespan

### Phase 2: Sources & Security - 2 days
- [ ] Manual entry → file → queue
- [ ] File upload with checksum
- [ ] SCTP listener with mTLS + HMAC (configurable limits)
- [ ] Staging directory for SCTP

### Phase 3: SSE & UI - 2 days
- [ ] Implement SSE endpoint with ring buffer replay
- [ ] Add EMA progress estimation
- [ ] Merge CacheWarmingCard + WarmingQueueCard
- [ ] Add checkbox selection and bulk delete

### Phase 4: Polish & Migration - 1 day
- [ ] Cleanup service with SSE buffer pruning
- [ ] Migration script
- [ ] Documentation
- [ ] Remove old queue code

---

## References

- Issue #113: Merge WarmingQueueCard into CacheWarmingCard
- Issue #104: Queue Management API (merged)
- Issue #105: Queue Management UI (merged)
- Existing: `ai_ready_rag/services/warming_queue.py`
