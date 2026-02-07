# Cache Warming Queue Redesign — DB-First Architecture

| Field | Value |
|-------|-------|
| **Status** | FINAL |
| **Version** | v1.2 |
| **Author** | Claude (AI) + jjob |
| **Date** | 2026-02-07 |
| **Replaces** | `CACHE_WARMING_QUEUE_v1.md`, `CACHE_WARMING_PERSISTENT_QUEUE.md` |

---

## 1. Problem Statement

The current warming system has **three overlapping data stores** that are out of sync:

| Store | What it holds | Who reads it |
|-------|---------------|-------------|
| Text files on disk (`data/warming_queue/*.txt`) | Raw query strings | WarmingWorker |
| JSON job files (`data/warming_queue/jobs/*.json`) | Job metadata + status | WarmingQueueService (file-based) |
| SQLite `warming_queue` table | Job metadata + status (duplicate) | API endpoints, SSE endpoint (partially) |

### Symptoms

1. **SSE "Connection to server lost"** — The SSE generator reads job state from the file-based `WarmingQueueService`, but manual/upload endpoints create jobs in the DB only. The file-based job doesn't exist, so SSE immediately returns "Job not found."

2. **"Successfully warmed" shown instantly** — Frontend called `completeWarming()` on HTTP 201 response instead of tracking via SSE. (Partially fixed — SSE connection now attempted, but SSE endpoint fails per #1.)

3. **Dual maintenance burden** — Every feature (pause, cancel, retry, progress) must update both the DB record AND the file-based job. State divergence causes silent bugs.

4. **File system complexity** — Atomic writes, quarantine directories, byte-offset resume, file checksums, staging files, archive directories — all to work around the limitations of using files as a database.

### Root Cause

The system was built file-first (v1), then a DB layer was added on top (v2) without removing the file layer. Both layers now fight for source-of-truth status.

---

## 2. Proposed Solution

**Make the database the sole source of truth.** Store individual queries as DB rows. Eliminate all file-based state.

### Architecture

```
Submit (manual/upload)
    → Parse + normalize queries
    → INSERT rows into warming_queries table
    → Enqueue ARQ job: "process batch {batch_id}"
    → If Redis down: return 503 (see Section 6.2)
    → Return batch_id to frontend

ARQ Worker
    → Acquire batch lease (optimistic locking)
    → SELECT pending queries WHERE batch_id = ? (ordered)
    → For each query:
        → Claim via UPDATE ... WHERE status = 'pending' (idempotent)
        → Process through RAG pipeline
        → UPDATE status = completed/failed
    → Determine batch terminal status

SSE Endpoint
    → Poll warming_queries + warming_batches tables
    → Replay missed events via warming_sse_events ring buffer
    → Push progress events to browser

Frontend
    → Show every query with its status
    → Inspect, clear, add, pause, cancel
```

### What Gets Eliminated

| Component | Lines of Code (approx) | Reason |
|-----------|----------------------|--------|
| `services/warming_queue.py` (WarmingQueueService) | ~400 | Replaced by DB queries |
| `WarmingJob` dataclass | ~100 | Replaced by `WarmingQuery` model |
| File I/O in `WarmingWorker._process_job` | ~200 | Worker reads DB instead of files |
| `_sse_event_generator` file-based reads | ~180 | SSE reads DB directly |
| Text file write in `add_manual_warming_queries` | ~15 | Queries go straight to DB |
| Text file write in `upload_warming_file` | ~30 | File parsed, queries go to DB, file discarded |
| Quarantine/archive directory logic | ~50 | No files to quarantine |
| `_ensure_warming_dir()` helper | ~5 | No directories needed |
| 6 warming config settings related to files | - | See section 4 |

**Estimated net reduction: ~800 lines of code.**

---

## 3. Database Schema

### New Table: `warming_queries`

Replaces: `warming_queue` (job-level) + text files (query-level)

```sql
CREATE TABLE warming_queries (
    id            TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    batch_id      TEXT NOT NULL REFERENCES warming_batches(id) ON DELETE CASCADE,
    query_text    TEXT NOT NULL,                     -- The actual question (normalized)
    status        TEXT NOT NULL DEFAULT 'pending',   -- pending | processing | completed | failed | skipped
    error_message TEXT,                              -- If failed
    error_type    TEXT,                              -- Exception class name
    retry_count   INTEGER NOT NULL DEFAULT 0,        -- Number of retry attempts
    sort_order    INTEGER NOT NULL DEFAULT 0,        -- Position within batch
    submitted_by  TEXT REFERENCES users(id) ON DELETE SET NULL,
    processed_at  DATETIME,                          -- When query finished processing
    created_at    DATETIME NOT NULL DEFAULT (datetime('now')),
    updated_at    DATETIME NOT NULL DEFAULT (datetime('now')),

    UNIQUE(batch_id, sort_order)                     -- Prevent duplicate positions
);

CREATE INDEX idx_warming_queries_batch ON warming_queries(batch_id);
CREATE INDEX idx_warming_queries_status ON warming_queries(status, created_at);
CREATE INDEX idx_warming_queries_pending ON warming_queries(batch_id, status)
    WHERE status = 'pending';
CREATE INDEX idx_warming_queries_cleanup ON warming_queries(status, processed_at);
```

### New Table: `warming_batches`

Metadata about each submission (replaces the `warming_queue` table at the job level).

```sql
CREATE TABLE warming_batches (
    id                      TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    source_type             TEXT NOT NULL,               -- 'manual' | 'upload'
    original_filename       TEXT,                        -- For uploads: "queries.txt"
    total_queries           INTEGER NOT NULL,            -- Count at submission time
    status                  TEXT NOT NULL DEFAULT 'pending',
    is_paused               BOOLEAN NOT NULL DEFAULT 0,
    is_cancel_requested     BOOLEAN NOT NULL DEFAULT 0,
    worker_id               TEXT,                        -- Lease: which worker owns this batch
    worker_lease_expires_at DATETIME,                    -- Lease expiry
    error_message           TEXT,                        -- If batch-level failure
    submitted_by            TEXT REFERENCES users(id) ON DELETE SET NULL,
    started_at              DATETIME,
    completed_at            DATETIME,
    created_at              DATETIME NOT NULL DEFAULT (datetime('now')),
    updated_at              DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_warming_batches_status ON warming_batches(status, created_at);
CREATE INDEX idx_warming_batches_lease ON warming_batches(worker_lease_expires_at);
CREATE INDEX idx_warming_batches_cleanup ON warming_batches(status, completed_at);
```

### Modified Table: `warming_sse_events`

The SSE event ring buffer table is kept but **event ordering is changed from UUIDs to a monotonic integer sequence per batch**. This guarantees unambiguous replay ordering.

```sql
-- Existing columns kept. Key change: event_id is now batch-scoped monotonic integer.
-- The auto-increment PK (id) provides global ordering.
-- New: batch_seq provides per-batch ordering for replay.

ALTER TABLE warming_sse_events ADD COLUMN batch_seq INTEGER;
CREATE INDEX idx_sse_events_batch_seq ON warming_sse_events(job_id, batch_seq);
```

**Replay semantics**: `last_event_id` sent by the client is the `batch_seq` integer (not a UUID). On reconnection, the server replays all events for the batch where `batch_seq > last_event_id`. Since `batch_seq` is monotonically increasing per batch, ordering is deterministic regardless of worker identity.

```python
# When storing an SSE event:
max_seq = db.query(func.max(WarmingSSEEvent.batch_seq)).filter_by(job_id=batch_id).scalar() or 0
event.batch_seq = max_seq + 1
event.event_id = str(event.batch_seq)  # Client sees the sequence number
```

### Removed Tables

| Table | Reason |
|-------|--------|
| `warming_queue` | Replaced by `warming_batches` + `warming_queries` |
| `warming_failed_queries` | Merged into `warming_queries` (status = 'failed' + error fields) |

### Query Normalization Rules

Before INSERT, all queries are normalized:

1. Strip leading/trailing whitespace
2. Collapse internal whitespace (multiple spaces → single space)
3. Skip empty strings and comment lines (lines starting with `#` or `//`)

**Duplicate policy**: Duplicate queries within a batch are **allowed**. The same question submitted in different batches is also allowed. Rationale: the user may intentionally want to re-warm a query (e.g., after document updates), and de-duplication adds complexity with unclear UX. A future enhancement could add an optional `deduplicate: bool` flag to the submission endpoint.

### Migration

```python
# Migration script (runs during Phase 1 deployment)
# 1. Create warming_batches and warming_queries tables
# 2. Migrate existing warming_queue completed/failed rows → warming_batches
#    (history only — no query data, just batch metadata)
# 3. Cancel pending/running jobs in warming_queue with message:
#    error_message = "System upgrade: please resubmit warming queries"
# 4. Log cancelled job count to admin notification table (if available)
#    or emit WARNING log visible in structured logging
# 5. Old tables kept read-only until Phase 6 (drop)
# 6. Delete data/warming_queue/ directory contents
```

**Data loss risk**: Pending warming jobs have queries in text files only. During migration, these jobs are cancelled with a user-visible message stored in the batch's `error_message` column: `"System upgrade: please resubmit warming queries"`. This message is displayed in the queue view alongside the `cancelled` status. Admin is also notified via structured log event `warming_migration_cancelled_jobs` with count.

---

## 4. State Machines

### 4.1 Batch State Machine

```
                ┌──────────┐
                │ pending  │
                └────┬─────┘
                     │ Worker acquires lease
                     ▼
                ┌──────────┐
         ┌──────│ running  │──────┐
         │      └────┬─────┘      │
         │           │            │
    is_paused   All queries   is_cancel_requested
         │      resolved          │
         ▼           │            ▼
    ┌────────┐       │      ┌───────────┐
    │ paused │       │      │ cancelled │  (terminal)
    └───┬────┘       │      └───────────┘
        │            ▼
   resume     ┌─────────────────┐
        │     │ Determine final │
        └─►───│ status (4.1.1)  │
              └────────┬────────┘
                       │
              ┌────────┴────────────┐
              │                     │
              ▼                     ▼
     ┌────────────────┐   ┌───────────────────────┐
     │   completed    │   │ completed_with_errors  │
     │ (all succeeded)│   │ (some failed)          │
     └────────────────┘   └───────────────────────┘
              │                     │
              └──── (terminal) ─────┘
```

**Valid batch status values**: `pending`, `running`, `paused`, `completed`, `completed_with_errors`, `cancelled`

Note: there is no `failed` batch status. A batch-level failure (e.g., worker crash without recovery) results in the lease expiring and the batch reverting to `pending` for re-acquisition.

#### 4.1.1 Batch Completion Criteria

When all queries in a batch have a terminal status (`completed`, `failed`, or `skipped`):

```python
failed_count = count(queries WHERE status = 'failed')
total_count = batch.total_queries

if failed_count == 0:
    batch.status = "completed"
elif failed_count == total_count:
    batch.status = "completed_with_errors"  # All failed — still terminal
else:
    batch.status = "completed_with_errors"  # Partial failure
```

The batch always reaches a terminal state. There is no "batch failed" status — the batch completes, but its `completed_with_errors` status tells the admin to inspect.

**Derived `all_failed` flag**: The batch response includes an `all_failed: bool` field computed at read time (`failed_count == total_count`). This lets the frontend distinguish "all 50 failed" from "2 of 50 failed" without a separate status value. The flag is NOT stored — it is derived from query counts.

```python
# In batch response serialization:
response.all_failed = (failed_count == batch.total_queries and failed_count > 0)
```

### 4.2 Query State Machine

```
    ┌─────────┐
    │ pending │
    └────┬────┘
         │  Worker claims (UPDATE WHERE status = 'pending')
         ▼
    ┌────────────┐
    │ processing │
    └─────┬──────┘
          │
    ┌─────┴──────────┐──────────┐
    │                │          │
    ▼                ▼          ▼
┌───────────┐  ┌────────┐  ┌─────────┐
│ completed │  │ failed │  │ skipped │
└───────────┘  └───┬────┘  └─────────┘
                   │
                   │ Admin retries
                   ▼
              ┌─────────┐
              │ pending │  (retry_count incremented)
              └─────────┘
```

**Valid query status values**: `pending`, `processing`, `completed`, `failed`, `skipped`

- `skipped`: Set when a batch is cancelled — remaining `pending` queries become `skipped`
- `failed` → `pending`: Only via explicit admin retry action (increments `retry_count`)

### 4.3 Pause/Cancel Semantics

**Pause** (`is_paused = True`):
- Worker checks `is_paused` flag **before** starting each query (not mid-query)
- The currently processing query **runs to completion** (no interruption)
- After the current query finishes, worker enters wait loop (polls every 2s)
- Wait loop checks for: resume (`is_paused = False`) or cancel (`is_cancel_requested = True`)
- Batch status set to `paused` when worker enters wait loop
- On resume: batch status returns to `running`, processing continues with next pending query

**Cancel** (`is_cancel_requested = True`):
- Worker checks `is_cancel_requested` **before** starting each query
- The currently processing query **runs to completion** (graceful — max `warming_cancel_timeout_seconds`)
- After current query finishes, worker:
  1. Sets all remaining `pending` queries to `skipped`
  2. Sets batch status to `cancelled`
  3. Releases lease
- If cancel requested during pause: worker exits wait loop, follows cancel path above

---

## 5. Concurrency & Idempotency

### 5.1 Query-Level Idempotency Guard

The worker claims queries using an atomic UPDATE with a WHERE clause:

```python
# Claim next pending query — atomic, prevents double-processing
result = db.execute(
    update(WarmingQuery)
    .where(
        WarmingQuery.id == query_row.id,
        WarmingQuery.status == "pending",  # Only claim if still pending
    )
    .values(status="processing", updated_at=datetime.now(UTC))
)
db.commit()

if result.rowcount == 0:
    # Another worker or retry already claimed this query — skip
    continue
```

This ensures:
- **No double-processing**: If two workers somehow overlap (e.g., ARQ retry after crash), only one claims the query
- **Idempotent retries**: If the ARQ job is retried, already-completed queries are skipped (status != 'pending')
- **No row-level locking needed**: The `WHERE status = 'pending'` clause acts as an optimistic lock

### 5.2 Batch Lease Acquisition — Single Worker Per Batch

**Invariant: Only one worker processes a batch at a time.** A second worker cannot acquire a batch that is `running` with a valid (non-expired) lease. This guarantees query ordering is preserved — the single owning worker processes queries in `sort_order` without interleaving.

A second worker can only acquire the batch if:
- Status is `pending` (never started, or reset after stale reclamation)
- The lease has expired (worker crashed — see Section 5.4)

```python
# Acquire batch lease — enforces single-worker-per-batch
result = db.execute(
    update(WarmingBatch)
    .where(
        WarmingBatch.id == batch_id,
        or_(
            # Case 1: Batch is pending (fresh or reclaimed)
            and_(
                WarmingBatch.status == "pending",
            ),
            # Case 2: Already ours (ARQ retry of same worker)
            and_(
                WarmingBatch.status == "running",
                WarmingBatch.worker_id == self.worker_id,
            ),
            # Case 3: Stale lease (previous worker crashed)
            and_(
                WarmingBatch.status == "running",
                WarmingBatch.worker_lease_expires_at < datetime.now(UTC),
            ),
        ),
    )
    .values(
        status="running",
        worker_id=self.worker_id,
        worker_lease_expires_at=datetime.now(UTC) + timedelta(
            minutes=settings.warming_lease_duration_minutes
        ),
        started_at=func.coalesce(WarmingBatch.started_at, datetime.now(UTC)),
    )
)
db.commit()

if result.rowcount == 0:
    return {"success": False, "error": "Batch not available or already leased"}
```

**Why per-query idempotency still matters**: Even with single-worker-per-batch, the worker may crash mid-query. On stale lease reclamation, a new worker acquires the batch and must skip already-completed queries. The `WHERE status = 'pending'` guard (Section 5.1) ensures this — the new worker simply picks up where the crashed worker left off, in `sort_order`.

### 5.3 Lease Renewal

The worker renews its lease on a background interval:

```python
# Runs every warming_lease_renewal_seconds (default: 60s)
# Lease duration: warming_lease_duration_minutes (default: 10 min)
# Invariant: renewal_interval < lease_duration (60s < 600s)

async def _lease_renewal_loop(self):
    while not self._shutdown.is_set():
        if self._current_batch_id:
            db = SessionLocal()
            try:
                db.execute(
                    update(WarmingBatch)
                    .where(
                        WarmingBatch.id == self._current_batch_id,
                        WarmingBatch.worker_id == self.worker_id,
                    )
                    .values(
                        worker_lease_expires_at=datetime.now(UTC) + timedelta(
                            minutes=settings.warming_lease_duration_minutes
                        )
                    )
                )
                db.commit()
            finally:
                db.close()
        await asyncio.sleep(settings.warming_lease_renewal_seconds)
```

### 5.4 Stale Lease Reclamation

On server startup, reclaim batches with expired leases:

```python
async def recover_stale_batches() -> int:
    """Reset batches with expired worker leases to pending."""
    db = SessionLocal()
    try:
        count = (
            db.query(WarmingBatch)
            .filter(
                WarmingBatch.status == "running",
                WarmingBatch.worker_lease_expires_at < datetime.now(UTC),
            )
            .update({
                "status": "pending",
                "worker_id": None,
                "worker_lease_expires_at": None,
            })
        )
        # Also reset any orphaned "processing" queries back to "pending"
        db.query(WarmingQuery).filter(
            WarmingQuery.status == "processing",
            WarmingQuery.batch_id.in_(
                db.query(WarmingBatch.id).filter(WarmingBatch.status == "pending")
            ),
        ).update({"status": "pending"}, synchronize_session=False)
        db.commit()
        return count
    finally:
        db.close()
```

---

## 6. ARQ Integration

### 6.1 Batch ID as Argument (not query data)

```python
# Current (queries in Redis memory — lost on Redis restart):
await redis.enqueue_job("warm_cache", queries=["Q1", "Q2"], triggered_by="user-1")

# Proposed (queries safe in SQLite, ARQ carries only a pointer):
await redis.enqueue_job("process_warming_batch", batch_id="abc-123")
```

### 6.2 Redis Unavailable: Fail-Fast with 503

**Decision**: When Redis is unavailable, the submit endpoint returns `503 Service Unavailable` instead of silently falling back to BackgroundTasks.

**Rationale** (from engineering review):
- BackgroundTasks are "best effort" — lost on server restart, no retry, no lease
- This contradicts the "DB-first reliability" goal of the redesign
- A 503 is honest: "I can't reliably process this right now"
- The admin can retry when Redis is back, and the queries are not lost (not yet in DB)

```python
# Submit endpoint
redis = await get_redis_pool()
if not redis:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Warming queue requires Redis. Please check Redis connection and retry.",
    )

# INSERT queries into DB
# ...

# Enqueue ARQ job
try:
    await redis.enqueue_job("process_warming_batch", batch_id)
except Exception as e:
    # ARQ enqueue failed — batch stays in DB as "pending"
    # It will be picked up when a worker next scans for pending batches
    logger.warning(f"ARQ enqueue failed for batch {batch_id}: {e}")
    # Don't fail the request — batch is safely in DB
```

**Fallback safety net**: If ARQ enqueue fails *after* queries are in DB, the batch stays `pending`. The WarmingWorker's polling loop (which runs on startup and periodically) will discover un-leased pending batches and process them. No data loss.

### 6.3 New ARQ Task: `process_warming_batch`

Replaces: `warm_cache` task + `warm_cache_task` BackgroundTask function + `_warm_file_task` function

```python
async def process_warming_batch(ctx: dict, batch_id: str) -> dict:
    """Process a warming batch by reading queries from DB.

    Idempotent: safe to retry. Already-processed queries are skipped
    via the WHERE status = 'pending' guard.
    """
    settings = ctx.get("settings") or get_settings()
    vector_service = ctx.get("vector_service")
    rag_service = RAGService(settings, vector_service=vector_service)
    db = SessionLocal()

    try:
        # 1. Acquire batch lease (see Section 5.2)
        if not _acquire_batch_lease(db, batch_id, worker_id):
            return {"success": False, "error": "Batch not available"}

        # 2. Process queries
        processed = 0
        failed = 0

        while True:
            # Re-check pause/cancel before each query
            batch = db.query(WarmingBatch).filter_by(id=batch_id).first()

            if batch.is_cancel_requested:
                _cancel_batch(db, batch_id)
                break

            if batch.is_paused:
                await _wait_for_resume_or_cancel(db, batch_id)
                continue

            # Claim next pending query (idempotent — see Section 5.1)
            query_row = _claim_next_query(db, batch_id)
            if not query_row:
                break  # No more pending queries

            # Process with retry
            success = await _warm_query_with_retry(
                rag_service, db, query_row, settings
            )

            if success:
                processed += 1
            else:
                failed += 1

            # Throttle
            if settings.warming_delay_seconds > 0:
                await asyncio.sleep(settings.warming_delay_seconds)

        # 3. Determine batch terminal status (see Section 4.1.1)
        _finalize_batch(db, batch_id)

        return {"success": True, "processed": processed, "failed": failed}

    finally:
        db.close()
```

### 6.4 Retry Policy Per Query

```python
async def _warm_query_with_retry(rag_service, db, query_row, settings) -> bool:
    """Process a single query with exponential backoff retry.

    Retry policy:
    - Max attempts: warming_max_retries (default: 3)
    - Delays: warming_retry_delays (default: "5,30,120" seconds)
    - Retryable errors: ConnectionTimeout, ServiceUnavailable,
      RateLimitExceeded, asyncio.TimeoutError
    - Non-retryable: ValueError, ValidationError, etc.
    """
    delays = [int(d) for d in settings.warming_retry_delays.split(",")]
    max_retries = settings.warming_max_retries

    for attempt in range(max_retries + 1):
        try:
            request = RAGRequest(query=query_row.query_text, user_tags=[])
            await rag_service.generate(request, db)

            query_row.status = "completed"
            query_row.processed_at = datetime.now(UTC)
            db.commit()
            return True

        except RETRYABLE_EXCEPTIONS as e:
            if attempt < max_retries:
                delay = delays[min(attempt, len(delays) - 1)]
                logger.warning(
                    f"Query retry {attempt + 1}/{max_retries}: '{query_row.query_text[:50]}...' "
                    f"error={type(e).__name__}, retrying in {delay}s"
                )
                await asyncio.sleep(delay)
            else:
                # Max retries exhausted
                query_row.status = "failed"
                query_row.error_message = str(e)[:500]
                query_row.error_type = type(e).__name__
                query_row.retry_count = attempt + 1
                query_row.processed_at = datetime.now(UTC)
                db.commit()
                return False

        except Exception as e:
            # Non-retryable error — fail immediately
            query_row.status = "failed"
            query_row.error_message = str(e)[:500]
            query_row.error_type = type(e).__name__
            query_row.retry_count = attempt + 1
            query_row.processed_at = datetime.now(UTC)
            db.commit()
            return False

RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    # Add project-specific: ServiceUnavailableError, RateLimitExceededError
)
```

---

## 7. SSE Redesign

### 7.1 Current Problem

`_sse_event_generator` calls `queue_service.get_job(job_id)` which reads from the file-based `WarmingQueueService`. Manual/upload endpoints create DB records only — no file-based job exists — so SSE immediately returns "Job not found."

### 7.2 Proposed: SSE reads from DB with replay support

The SSE generator has two modes:
1. **Replay**: If `last_event_id` is provided, replay missed events from `warming_sse_events` ring buffer
2. **Live polling**: Poll `warming_queries` + `warming_batches` tables for current state

```python
async def _sse_event_generator(batch_id: str, last_event_id: str | None = None):
    """Generate SSE events by polling DB state.

    Supports reconnection via last_event_id for replaying missed events.
    """
    event_sequence = 0  # Monotonic event counter for this connection

    # Phase 1: Replay missed events (if reconnecting)
    if last_event_id:
        db = SessionLocal()
        try:
            missed_events = get_events_for_job(db, batch_id, since_event_id=last_event_id)
            for event in missed_events:
                yield _format_sse_event(
                    event["event_type"],
                    event["payload"],
                    event["event_id"],
                )
            logger.info(f"[SSE] Replayed {len(missed_events)} events for batch {batch_id}")
        finally:
            db.close()

    # Phase 2: Live polling
    last_processed_count = -1

    while True:
        db = SessionLocal()
        try:
            batch = db.query(WarmingBatch).filter_by(id=batch_id).first()
            if not batch:
                event_id = str(uuid.uuid4())
                yield _format_sse_event("error", {"error": "Batch not found"}, event_id)
                return

            # Aggregate query statuses
            from sqlalchemy import func as sa_func
            counts = dict(
                db.query(WarmingQuery.status, sa_func.count())
                .filter_by(batch_id=batch_id)
                .group_by(WarmingQuery.status)
                .all()
            )

            completed = counts.get("completed", 0)
            failed = counts.get("failed", 0)
            processing = counts.get("processing", 0)
            skipped = counts.get("skipped", 0)
            total = batch.total_queries
            processed_count = completed + failed + skipped

            # Emit progress only when it changes
            if processed_count != last_processed_count:
                last_processed_count = processed_count
                event_sequence += 1
                event_id = str(uuid.uuid4())
                progress_data = {
                    "batch_id": batch_id,
                    "processed": processed_count,
                    "completed": completed,
                    "failed": failed,
                    "processing": processing,
                    "skipped": skipped,
                    "total": total,
                    "percent": int(processed_count / total * 100) if total > 0 else 0,
                    "batch_status": batch.status,
                }
                # Store for replay
                store_sse_event(db, "progress", batch_id, progress_data)
                yield _format_sse_event("progress", progress_data, event_id)

            # Terminal states
            if batch.status in ("completed", "completed_with_errors", "cancelled"):
                event_id = str(uuid.uuid4())
                complete_data = {
                    "batch_id": batch_id,
                    "status": batch.status,
                    "completed": completed,
                    "failed": failed,
                    "skipped": skipped,
                    "total": total,
                }
                store_sse_event(db, "complete", batch_id, complete_data)
                yield _format_sse_event("complete", complete_data, event_id)
                prune_old_events(db)
                return

            # Paused — emit but keep connection open
            if batch.status == "paused":
                event_id = str(uuid.uuid4())
                yield _format_sse_event("paused", {
                    "batch_id": batch_id,
                    "processed": processed_count,
                    "total": total,
                }, event_id)

        finally:
            db.close()

        await asyncio.sleep(1.0)  # Poll interval

        # Heartbeat every 30s (sse_heartbeat_seconds)
        # (heartbeat logic interleaved with polling, omitted for brevity)
```

### 7.3 SSE Event Types

| Event | When | Data |
|-------|------|------|
| `connected` | SSE connection established | `{ worker_id, timestamp }` |
| `progress` | Query status changes | `{ batch_id, processed, completed, failed, total, percent, batch_status }` |
| `paused` | Batch paused | `{ batch_id, processed, total }` |
| `complete` | Batch reached terminal state | `{ batch_id, status, completed, failed, skipped, total }` |
| `error` | Batch not found or system error | `{ error }` |
| `heartbeat` | Every `sse_heartbeat_seconds` | `{ timestamp }` |

### 7.4 Replay Guarantee

- Every `progress` and `complete` event is stored in `warming_sse_events` ring buffer
- On reconnection, client sends `last_event_id` → server replays all events after that ID
- Ring buffer is pruned to `sse_event_buffer_size` (default: 1000) entries
- If `last_event_id` is too old (pruned), full current state is sent instead of replay

---

## 8. Configuration Changes

### Removed Settings

| Setting | Reason |
|---------|--------|
| `warming_queue_dir` | No file storage |
| `warming_max_file_size_mb` | Replaced by `warming_max_upload_size_mb` |
| `warming_allowed_extensions` | Only relevant during file parsing, not a config setting |
| `warming_archive_completed` | No files to archive (DB has full history) |
| `warming_checkpoint_interval` | No file checkpointing (DB commits per-query) |
| `warming_scan_interval_seconds` | No folder scanning (ARQ triggers work) |

### Kept Settings

| Setting | Purpose |
|---------|---------|
| `warming_delay_seconds` | Throttle between Ollama calls |
| `warming_lock_timeout_minutes` | Reclaim stale batch leases |
| `warming_max_concurrent_queries` | Semaphore for concurrent Ollama calls |
| `warming_lease_duration_minutes` | Batch lease duration (default: 10 min) |
| `warming_lease_renewal_seconds` | Lease renewal interval (default: 60s). **Invariant: must be < lease_duration** |
| `warming_max_retries` | Max retry attempts per query (default: 3) |
| `warming_retry_delays` | Exponential backoff delays in seconds (default: "5,30,120") |
| `warming_cancel_timeout_seconds` | Grace period for cancel |
| `warming_completed_retention_days` | Cleanup: delete old completed batches + queries |
| `warming_failed_retention_days` | Cleanup: delete old failed batches + queries |
| `warming_cleanup_interval_hours` | Cleanup service interval |
| `warming_checkpoint_time_seconds` | Max seconds between SSE event emissions |

### New Settings

| Setting | Type | Default | Purpose |
|---------|------|---------|---------|
| `warming_max_queries_per_batch` | int | 10000 | Max queries in a single submission. Returns 400 if exceeded. |
| `warming_max_upload_size_mb` | float | 10.0 | Max file size for upload parsing (before discard). Returns 400 if exceeded. |

**Validation**: Both limits are checked before any DB writes. If exceeded, the endpoint returns `400 Bad Request` with a descriptive message:

```python
# Manual submission
if len(queries) > settings.warming_max_queries_per_batch:
    raise HTTPException(
        status_code=400,
        detail=f"Batch exceeds maximum of {settings.warming_max_queries_per_batch} queries. "
               f"Submitted: {len(queries)}. Split into smaller batches.",
    )

# File upload — check file size before parsing
if file.size > settings.warming_max_upload_size_mb * 1024 * 1024:
    raise HTTPException(
        status_code=400,
        detail=f"File exceeds maximum size of {settings.warming_max_upload_size_mb} MB.",
    )
```

---

## 9. API Changes

### Modified Endpoints

#### POST `/api/admin/warming/queue/manual`

**Before**: Parse queries → write text file → create DB `WarmingQueue` record
**After**: Parse + normalize queries → INSERT rows into `warming_queries` + `warming_batches` → enqueue ARQ job

```python
# Request (unchanged)
{ "queries": ["Who is head of IT?", "What is PTO policy?"] }

# Response (updated)
{
    "batch_id": "abc-123",
    "total_queries": 2,
    "status": "pending",
    "source_type": "manual",
    "created_at": "2026-02-07T08:30:00Z"
}

# Error: Redis unavailable
# 503 { "detail": "Warming queue requires Redis. Please check Redis connection and retry." }
```

#### POST `/api/admin/warming/queue/upload`

**Before**: Save file to disk → create DB `WarmingQueue` record
**After**: Parse file in memory → normalize → INSERT rows → discard file → enqueue ARQ job

#### GET `/api/admin/warming/progress`

**Before**: SSE generator reads from `WarmingQueueService.get_job()` (file-based)
**After**: SSE generator queries DB directly with replay support via `last_event_id`

#### GET `/api/admin/warming/queue`

**Before**: Returns list of `WarmingQueue` records (job-level only)
**After**: Returns list of `warming_batches` with aggregated query counts

### New Endpoints

#### GET `/api/admin/warming/batch/{batch_id}/queries`

List individual queries in a batch with their status.

```python
# Response
{
    "batch_id": "abc-123",
    "queries": [
        { "id": "q-001", "query_text": "Who is head of IT?", "status": "completed", "processed_at": "..." },
        { "id": "q-002", "query_text": "What is PTO policy?", "status": "pending", "processed_at": null }
    ],
    "total": 2,
    "completed": 1,
    "failed": 0,
    "pending": 1
}
```

#### DELETE `/api/admin/warming/batch/{batch_id}/queries/{query_id}`

Remove a single query from the queue (only if status = 'pending').

#### POST `/api/admin/warming/batch/{batch_id}/retry`

Re-set all `failed` queries in a batch to `pending` (increments `retry_count`). Re-enqueues ARQ job.

#### POST `/api/admin/warming/batch/{batch_id}/queries/{query_id}/retry`

Re-set a **single** `failed` query to `pending` (increments `retry_count`). If the batch is in a terminal state (`completed_with_errors`), changes batch status back to `pending` and re-enqueues ARQ job.

```python
# Response
{
    "query_id": "q-002",
    "batch_id": "abc-123",
    "status": "pending",
    "retry_count": 2,
    "batch_requeued": true   # true if batch status was also changed
}

# Error: query is not in 'failed' status
# 409 { "detail": "Query is not in failed status" }
```

### Removed Endpoints (with deprecation window)

Legacy endpoints are removed in two phases:

1. **Phase 2 (this release)**: Legacy endpoints return `410 Gone` with a JSON body directing the caller to the replacement endpoint. This allows any external scripts or bookmarks to fail with a clear message rather than a silent 404.

```python
# Example: legacy endpoint stub
@router.post("/cache/warm")
async def warm_cache_legacy():
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="This endpoint has been removed. Use POST /api/admin/warming/queue/manual instead.",
    )
```

2. **Phase 6 (next release)**: Legacy endpoint stubs are deleted entirely.

| Endpoint | Replacement | Status |
|----------|-------------|--------|
| `POST /api/admin/cache/warm` | `POST /warming/queue/manual` | 410 Gone (Phase 2) → Deleted (Phase 6) |
| `GET /api/admin/cache/warm-progress/{job_id}` | `GET /warming/progress` | 410 Gone (Phase 2) → Deleted (Phase 6) |
| `POST /api/admin/cache/warm-retry` | `POST /warming/batch/{id}/retry` | 410 Gone (Phase 2) → Deleted (Phase 6) |
| `GET /api/admin/cache/warm-status/{job_id}` | `GET /warming/queue/{batch_id}` | 410 Gone (Phase 2) → Deleted (Phase 6) |

### Unchanged Endpoints

- `GET /api/admin/warming/queue` (list batches)
- `GET /api/admin/warming/queue/completed` (list completed)
- `GET /api/admin/warming/queue/{batch_id}` (get batch)
- `DELETE /api/admin/warming/queue/{batch_id}` (delete batch + queries via CASCADE)
- `DELETE /api/admin/warming/queue/bulk` (bulk delete)
- `GET /api/admin/warming/current` (current running batch)
- `POST /api/admin/warming/current/pause`
- `POST /api/admin/warming/current/resume`
- `POST /api/admin/warming/current/cancel`

---

## 10. Frontend Changes

### Queue View (CacheWarmingCard)

**Before**: Shows jobs (batch-level only, no query visibility)
**After**: Shows batches with expandable query list

```
┌─────────────────────────────────────────────────────────────┐
│  Warming Queue                                     Refresh  │
│                                                             │
│  ▼ Batch abc-123 (Manual, 2 queries)     Running  1/2      │
│    ├── "Who is the head of IT?"          [completed]        │
│    └── "What is our PTO policy?"         [processing]       │
│                                                             │
│  ► Batch xyz-789 (Upload: queries.txt)   Pending  0/100    │
│                                                             │
│  [Delete Selected]                                          │
└─────────────────────────────────────────────────────────────┘
```

### UX Text Changes Checklist

- [ ] Batch status badge: add `completed_with_errors` variant (yellow/warning)
- [ ] 503 error: show "Redis unavailable — please retry" instead of generic error
- [ ] "Successfully warmed" → "Warming complete: X/Y queries succeeded"
- [ ] File upload: remove `.txt/.csv only` restriction from help text (still enforced server-side, but UI shouldn't limit)

### Changes to `CacheWarmingCard.tsx`

1. `handleManualWarm` — Already fixed to use `startFileJob` + `connectToSSE`
2. Queue table — Add expandable rows to show individual queries (lazy-loaded via `getBatchQueries`)
3. SSE event handling — Handle new `paused` event type
4. Batch status badges — Add `completed_with_errors` status

### New API Call

```typescript
export async function getBatchQueries(batchId: string): Promise<BatchQueriesResponse> {
    return apiClient.get(`/api/admin/warming/batch/${batchId}/queries`);
}

export async function retryBatch(batchId: string): Promise<void> {
    return apiClient.post(`/api/admin/warming/batch/${batchId}/retry`);
}

export async function retryQuery(batchId: string, queryId: string): Promise<void> {
    return apiClient.post(`/api/admin/warming/batch/${batchId}/queries/${queryId}/retry`);
}
```

---

## 11. Files Modified / Created / Deleted

### Created

| File | Purpose |
|------|---------|
| `ai_ready_rag/db/models/warming.py` | New `WarmingBatch` + `WarmingQuery` models |
| `ai_ready_rag/workers/tasks/warming_batch.py` | New `process_warming_batch` ARQ task |

### Modified

| File | Changes |
|------|---------|
| `ai_ready_rag/api/admin.py` | Rewrite warming endpoints to use DB, remove file I/O, remove legacy endpoints, add 503 for Redis unavailable |
| `ai_ready_rag/workers/warming_worker.py` | Simplify: read from DB, idempotent query claiming, batch lease with renewal |
| `ai_ready_rag/workers/warming_cleanup.py` | Simplify: delete old batches + queries (CASCADE), remove file cleanup |
| `ai_ready_rag/workers/tasks/__init__.py` | Replace `warm_cache` with `process_warming_batch` |
| `ai_ready_rag/workers/settings.py` | Update registered tasks |
| `ai_ready_rag/schemas/admin.py` | Update response schemas, add `BatchQueriesResponse`, add `completed_with_errors` status |
| `ai_ready_rag/config.py` | Remove 6 file-based settings, add 2 new settings |
| `ai_ready_rag/db/models/__init__.py` | Export new models |
| `frontend/src/components/features/admin/CacheWarmingCard.tsx` | Add query-level view, `completed_with_errors` badge |
| `frontend/src/api/cache.ts` | Add `getBatchQueries()`, `retryBatch()`, remove legacy endpoints |
| `frontend/src/types/index.ts` | Update types, add `completed_with_errors` to `WarmingJobStatus` |

### Deleted

| File | Reason |
|------|--------|
| `ai_ready_rag/services/warming_queue.py` | Entire file-based queue service |
| `ai_ready_rag/workers/tasks/warming.py` | Old `warm_cache` task (replaced by `warming_batch.py`) |
| `data/warming_queue/` (directory) | No file storage |

---

## 12. Implementation Phases

### Phase 1: DB Schema + New ARQ Task (backend only)

1. Create `WarmingBatch` and `WarmingQuery` models with indexes and constraints
2. Create `process_warming_batch` ARQ task with idempotency guards
3. Register task in worker settings
4. Write unit tests for: idempotent claiming, batch completion criteria, lease acquisition, retry policy

**Risk**: Low. Additive — doesn't break existing system.
**Effort**: 1.5 days

### Phase 2: Rewrite API Endpoints (backend)

1. Modify `POST /warming/queue/manual` — normalize, INSERT, enqueue ARQ, 503 if no Redis
2. Modify `POST /warming/queue/upload` — parse file, normalize, INSERT, discard, enqueue ARQ
3. Modify `GET /warming/queue` — query `warming_batches` with aggregated counts
4. Modify queue management endpoints (delete, pause, cancel, resume)
5. Add `GET /warming/batch/{id}/queries` endpoint
6. Add `POST /warming/batch/{id}/retry` endpoint (batch-level retry)
7. Add `POST /warming/batch/{id}/queries/{query_id}/retry` endpoint (single-query retry)
8. Add validation: 400 for over-limit submissions and over-size uploads
9. Replace legacy endpoints with 410 Gone stubs (deprecation window, removed in Phase 6)

**Risk**: Medium. Breaks existing warming if done incrementally. Do as atomic swap.
**Effort**: 2 days

### Phase 3: Rewrite SSE Generator (backend)

1. Replace `_sse_event_generator` — read from DB with replay support
2. Remove `get_warming_queue()` usage from SSE
3. Test SSE with manual and file submissions
4. Test SSE reconnection with `last_event_id` replay

**Risk**: Medium. SSE is the user-visible indicator that warming works.
**Effort**: 1 day

### Phase 4: Simplify WarmingWorker (backend)

1. Remove file-reading logic from `_process_job`
2. Worker queries `warming_queries` for pending rows with idempotent claiming
3. Implement pause/cancel semantics per Section 4.3
4. Implement batch lease renewal per Section 5.3
5. Implement stale lease reclamation per Section 5.4
6. Remove `WarmingQueueService` dependency

**Risk**: Medium. Worker is the most complex component.
**Effort**: 1.5 days

### Phase 5: Frontend Updates

1. Update `CacheWarmingCard` — expandable query view (lazy-loaded)
2. Add `getBatchQueries()` and `retryBatch()` API calls
3. Add `completed_with_errors` status badge
4. Update UX text per checklist
5. Handle 503 Redis unavailable error in submit handler
6. Verify SSE progress tracking works end-to-end

**Risk**: Low. Frontend is mostly consuming data — shape changes are small.
**Effort**: 1 day

### Phase 6: Cleanup + Migration

1. Run migration: create new tables, cancel pending old jobs with admin notification
2. Delete `services/warming_queue.py`
3. Delete `workers/tasks/warming.py`
4. Drop old DB tables (`warming_queue`, `warming_failed_queries`)
5. Remove file-based config settings
6. Delete `data/warming_queue/` directory
7. Update all tests

**Risk**: Low. Everything should already be using new paths.
**Effort**: 1 day

**Total estimated effort: 8 days**

---

## 13. Migration Plan

### Step 1: Deploy new tables alongside old

Both `warming_queue` (old) and `warming_batches` + `warming_queries` (new) exist simultaneously. New endpoints write to new tables. Old tables are read-only.

### Step 2: Cancel pending old jobs with notification

```python
# Migration script
stuck_count = (
    db.query(WarmingQueue)
    .filter(WarmingQueue.status.in_(["pending", "running"]))
    .update({
        "status": "cancelled",
        "error_message": "System upgrade: please resubmit warming queries",
    })
)
db.commit()

if stuck_count:
    logger.warning(
        "warming_migration_cancelled_jobs",
        extra={"cancelled_count": stuck_count},
    )
```

**Admin notification**: The structured log event `warming_migration_cancelled_jobs` is visible in the log output. If the system has an admin notification UI in the future, this event should trigger a banner.

### Step 3: Drop old tables

After confirming new system works (Phase 6), drop `warming_queue` and `warming_failed_queries`.

### Rollback

If issues are found after Phase 2, revert the API endpoints to use old tables. Old tables are not dropped until Phase 6.

---

## 14. Acceptance Criteria

### Functional

- [ ] Manual entry creates DB rows and shows real-time SSE progress
- [ ] File upload parses file, creates DB rows, discards file, shows SSE progress
- [ ] Queue view shows every query with its status (expandable batch rows)
- [ ] Individual queries can be deleted from queue (pending only)
- [ ] Batch pause works: current query completes, then worker waits
- [ ] Batch resume works: processing continues from next pending query
- [ ] Batch cancel works: current query completes, remaining set to `skipped`
- [ ] Batch completes as `completed` when all queries succeed
- [ ] Batch completes as `completed_with_errors` when some queries fail
- [ ] Failed queries show error details (message + type)
- [ ] Retry (batch) sets all failed queries in batch back to `pending` and re-enqueues
- [ ] Retry (single query) sets one failed query back to `pending` and re-enqueues batch
- [ ] Over-limit submissions (> `warming_max_queries_per_batch`) return 400 with descriptive message
- [ ] Over-size file uploads (> `warming_max_upload_size_mb`) return 400 with descriptive message
- [ ] Comment lines (`#` and `//`) are skipped during file parsing

### Reliability

- [ ] ARQ processes batches when Redis available
- [ ] Submit returns 503 when Redis unavailable (not silent fallback)
- [ ] ARQ enqueue failure doesn't lose data (batch stays pending in DB)
- [ ] Query claiming is idempotent (no double-processing on ARQ retry)
- [ ] Stale batch leases are reclaimed on server restart
- [ ] Orphaned `processing` queries reset to `pending` on lease reclaim
- [ ] SSE reconnection with `last_event_id` replays missed events

### Cleanup

- [ ] No files created in `data/warming_queue/`
- [ ] `services/warming_queue.py` deleted
- [ ] Old tables dropped after migration confirmed
- [ ] All warming-related tests pass
- [ ] Legacy endpoints return 404

### UX

- [ ] `completed_with_errors` badge shown in queue view
- [ ] `all_failed` batches show distinct messaging (e.g., "All queries failed" vs "2 of 50 failed")
- [ ] 503 error shows "Redis unavailable" message to admin
- [ ] Migration cancels pending jobs with user-visible message in `error_message` field
- [ ] Legacy endpoints return 410 Gone with redirect guidance (not 404)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| v1.0 | 2026-02-07 | Claude + jjob | Initial draft |
| v1.1 | 2026-02-07 | Claude + jjob | Address engineering review: add state machines (Sec 4), idempotency guards (Sec 5), batch completion criteria (Sec 4.1.1), pause/cancel semantics (Sec 4.3), 503 fail-fast for Redis down (Sec 6.2), SSE replay policy (Sec 7.2), retry algorithm (Sec 6.4), uniqueness constraints, cleanup indexes, lease renewal spec, stale reclamation, migration notification, UX text checklist. Effort estimate 6d → 8d. |
| v1.2 | 2026-02-07 | Claude + jjob | Final review: single worker per batch invariant (Sec 5.2), monotonic SSE event ordering via `batch_seq` (Sec 3 warming_sse_events), derived `all_failed` flag (Sec 4.1.1), single-query retry endpoint (Sec 9), comment style support `#` and `//` (Sec 3 normalization), migration cancel reason in `error_message` (Sec 3 migration), legacy endpoint deprecation window with 410 Gone (Sec 9), over-limit validation with 400 (Sec 8). |
