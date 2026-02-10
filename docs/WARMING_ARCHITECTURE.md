# Cache Warming Architecture — Proposed

## What is ARQ? What is SSE?

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ARQ (Async Redis Queue)                                           │
│  ═══════════════════════                                           │
│  A task queue. Think of it as a to-do list in Redis.               │
│                                                                     │
│  Producer says: "Hey, process warming batch ABC"                   │
│  Redis holds the message until a worker picks it up.               │
│  Worker says: "Got it, I'll run the queries now."                  │
│                                                                     │
│  PURPOSE: Reliable "go do this work" signal                        │
│  DIRECTION: Backend → Redis → Worker                               │
│                                                                     │
│  If Redis is down? BackgroundTasks runs it in-process instead.     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SSE (Server-Sent Events)                                          │
│  ════════════════════════                                          │
│  A live stream from server to browser. Like a news ticker.         │
│                                                                     │
│  Browser says: "I want updates on batch ABC"                       │
│  Server holds the connection open.                                 │
│  Server pushes: "2/10 done... 3/10 done... complete!"             │
│                                                                     │
│  PURPOSE: Live progress updates to the UI                          │
│  DIRECTION: Server → Browser (one-way)                             │
│                                                                     │
│  Browser uses EventSource API. No polling needed.                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Complete Flow — Manual Entry

```
 BROWSER                    FASTAPI SERVER                   REDIS        WORKER
 ═══════                    ══════════════                   ═════        ══════

 ┌──────────────┐
 │ User types:  │
 │ "Who is the  │
 │  head of IT?"│
 │ "What is our │
 │  PTO policy?"│
 │              │
 │ Clicks       │
 │ [Warm Cache] │
 └──────┬───────┘
        │
        │ POST /api/admin/warming/queue/manual
        │ { queries: ["Who is the head of IT?", "What is our PTO policy?"] }
        │
        ├──────────────────►┌──────────────────────────────┐
        │                   │ 1. Parse & validate queries   │
        │                   │                               │
        │                   │ 2. Generate batch_id          │
        │                   │                               │
        │                   │ 3. INSERT into DB:            │
        │                   │    ┌─────────────────────┐    │
        │                   │    │ warming_queries      │    │
        │                   │    ├─────────────────────┤    │
        │                   │    │ batch_id: "abc-123"  │    │
        │                   │    │ query: "Who is the   │    │
        │                   │    │   head of IT?"       │    │
        │                   │    │ status: pending      │    │
        │                   │    ├─────────────────────┤    │
        │                   │    │ batch_id: "abc-123"  │    │
        │                   │    │ query: "What is our  │    │
        │                   │    │   PTO policy?"       │    │
        │                   │    │ status: pending      │    │
        │                   │    └─────────────────────┘    │
        │                   │                               │
        │                   │ 4. Enqueue ARQ job ──────────────────────►┌─────────┐
        │                   │    (or BackgroundTasks        │          │  Redis   │
        │                   │     if Redis down)            │          │          │
        │                   │                               │          │ "process │
        │                   └──────────────────────────────┘          │  batch   │
        │                                                              │  abc-123"│
   ◄────┤ 201 Created { batch_id: "abc-123", total: 2 }               └────┬─────┘
        │                                                                   │
        │                                                                   │
        │  ┌─ SSE CONNECTION ────────────────────────────────────┐          │
        │  │                                                     │          │
        │  │ GET /api/admin/warming/progress?batch_id=abc-123    │          │
        │  │                                                     │          │
        │  │ Browser opens EventSource (persistent connection)   │          │
        │  │                                                     │          │
        │  └─────────────────┬───────────────────────────────────┘          │
        │                    │                                              │
        │                    ▼                                              ▼
        │        ┌──────────────────────┐                       ┌──────────────────┐
        │        │  SSE Endpoint        │                       │  ARQ Worker      │
        │        │                      │                       │                  │
        │        │  Polls DB every 0.5s │                       │  Picks up job    │
        │        │  for status changes  │                       │  from Redis      │
        │        │                      │                       │                  │
        │        └──────────┬───────────┘                       └────────┬─────────┘
        │                   │                                            │
        │                   │                                            │
        │                   │           ┌────────────────────────────────┘
        │                   │           │
        │                   │           ▼
        │                   │  ┌─────────────────────────────────────────┐
        │                   │  │  Worker processes queries one by one:   │
        │                   │  │                                         │
        │                   │  │  FOR each query WHERE status = pending: │
        │                   │  │    1. UPDATE status = 'processing'      │
        │                   │  │    2. Send query to RAG pipeline        │
        │                   │  │       (embed → search → LLM → cache)   │
        │                   │  │    3. UPDATE status = 'completed'       │
        │                   │  │       (or 'failed' + error_message)     │
        │                   │  │    4. Sleep (warming_delay_seconds)     │
        │                   │  │                                         │
        │                   │  └──────────────────┬──────────────────────┘
        │                   │                     │
        │                   │     DB is updated   │
        │                   │         ◄───────────┘
        │                   │
        │                   ▼
        │        ┌──────────────────────┐
        │        │  SSE reads DB:       │
        │        │  "1/2 completed"     │
        │        └──────────┬───────────┘
        │                   │
   ◄────┤  event: progress  │
        │  data: {          │
        │    processed: 1,  │
        │    total: 2       │
        │  }                │
        │                   │
        │        ┌──────────────────────┐
        │        │  SSE reads DB:       │
        │        │  "2/2 completed"     │
        │        └──────────┬───────────┘
        │                   │
   ◄────┤  event: progress  │
        │  data: {          │
        │    processed: 2,  │
        │    total: 2       │
        │  }                │
        │                   │
   ◄────┤  event: complete  │
        │  data: {          │
        │    total: 2,      │
        │    failed: 0      │
        │  }                │
        │                   │
        │   Connection closes
        │
 ┌──────┴───────┐
 │ UI shows:    │
 │ "Successfully│
 │  warmed 2/2  │
 │  queries"    │
 └──────────────┘
```

## Complete Flow — File Upload

```
 BROWSER                    FASTAPI SERVER                   REDIS        WORKER
 ═══════                    ══════════════                   ═════        ══════

 ┌──────────────┐
 │ User uploads │
 │ queries.txt: │
 │              │
 │ Who is CEO?  │
 │ PTO policy?  │
 │ IT helpdesk? │
 │ ...100 lines │
 │              │
 │ Clicks       │
 │ [Start]      │
 └──────┬───────┘
        │
        │ POST /api/admin/warming/queue/upload  (multipart/form-data)
        │ file: queries.txt
        │
        ├──────────────────►┌──────────────────────────────┐
        │                   │ 1. Read file contents         │
        │                   │                               │
        │                   │ 2. Parse lines, skip blanks/  │
        │                   │    comments                   │
        │                   │                               │
        │                   │ 3. Generate batch_id          │
        │                   │                               │
        │                   │ 4. INSERT 100 rows into DB    │
        │                   │    (one per query, all        │
        │                   │     status: pending)          │
        │                   │                               │
        │                   │ 5. DISCARD file               │  ◄── File no longer needed!
        │                   │    (DB is source of truth)    │
        │                   │                               │
        │                   │ 6. Enqueue ARQ job ──────────────────►  Redis
        │                   │                               │
        │                   └──────────────────────────────┘
        │
   ◄────┤ 201 Created { batch_id: "xyz-789", total: 100 }
        │
        │    ... same SSE flow as manual entry ...
        │    (connect → progress events → complete)
```

## The Queue Is Inspectable

```
┌─────────────────────────────────────────────────────────────────┐
│  Warming Queue  (what the admin sees in the UI)                 │
│                                                          Refresh│
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Batch "abc-123" (Manual, 2 queries)      ⏳ Processing   │  │
│  │   ☐ "Who is the head of IT?"             ✅ completed    │  │
│  │   ☐ "What is our PTO policy?"            ⏳ processing   │  │
│  │                                                          │  │
│  │  [Cancel]  [Pause]                                       │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │ Batch "xyz-789" (File: queries.txt, 100 queries) Pending │  │
│  │   100 queries waiting...                                 │  │
│  │                                                          │  │
│  │  [Delete]  [View Queries]                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Admin can:                                                     │
│    • See every query and its status                             │
│    • Delete individual queries or entire batches                │
│    • Add more queries (manual or upload) while batch runs       │
│    • Pause/resume/cancel a batch                                │
│    • Retry failed queries                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## System Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER                                   │
│                                                                  │
│   Manual Entry          File Upload           Queue Inspector    │
│   [textarea]            [file picker]         [table view]       │
│        │                     │                      ▲            │
│        └──── POST ───────────┘                      │            │
│                │                                SSE stream       │
│                ▼                                    │            │
├────────────────────────────────────────────────────────────────  │
│                     FASTAPI SERVER                               │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐    │
│  │ Submit      │  │ SSE         │  │ Queue Management     │    │
│  │ Endpoint    │  │ Endpoint    │  │ Endpoints            │    │
│  │             │  │             │  │                      │    │
│  │ Parse →     │  │ Poll DB →   │  │ List / Delete /      │    │
│  │ INSERT →    │  │ Push events │  │ Pause / Cancel /     │    │
│  │ Enqueue ARQ │  │             │  │ Retry                │    │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬───────────┘    │
│         │                │                     │                 │
│         ▼                ▼                     ▼                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    SQLite DATABASE                        │   │
│  │                                                          │   │
│  │  warming_queries                                         │   │
│  │  ┌────────┬──────────────────────┬──────────┬─────────┐  │   │
│  │  │ id     │ query_text           │ status   │ batch   │  │   │
│  │  ├────────┼──────────────────────┼──────────┼─────────┤  │   │
│  │  │ q-001  │ Who is head of IT?   │ completed│ abc-123 │  │   │
│  │  │ q-002  │ What is PTO policy?  │ processing│ abc-123│  │   │
│  │  │ q-003  │ Who is CEO?          │ pending  │ xyz-789 │  │   │
│  │  │ q-004  │ IT helpdesk number?  │ pending  │ xyz-789 │  │   │
│  │  │ ...    │ ...                  │ ...      │ ...     │  │   │
│  │  └────────┴──────────────────────┴──────────┴─────────┘  │   │
│  │                                                          │   │
│  │  ◄── Source of truth. Survives crashes. Inspectable. ──► │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                        │
│         │  Enqueue: "process batch abc-123"                      │
│         ▼                                                        │
│  ┌──────────────┐          ┌─────────────────────────────────┐  │
│  │    Redis     │ ──────►  │  ARQ Worker                     │  │
│  │              │          │                                 │  │
│  │  Job queue:  │          │  1. Dequeue job                 │  │
│  │  - batch     │          │  2. SELECT pending queries      │  │
│  │    abc-123   │          │  3. For each query:             │  │
│  │              │          │     → Send to Ollama (RAG)      │  │
│  │              │          │     → UPDATE DB row             │  │
│  │              │          │  4. Mark batch done             │  │
│  └──────────────┘          └─────────────────────────────────┘  │
│                                                                  │
│  If Redis is down:                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  BackgroundTasks fallback (same logic, runs in-process)  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## What Gets Eliminated

```
  CURRENT (complex)                    PROPOSED (simple)
  ═════════════════                    ═════════════════

  ┌──────────────────┐                 ┌──────────────────┐
  │ Text files on    │                 │                  │
  │ disk (queries)   │  ──► GONE       │   SQLite DB      │
  ├──────────────────┤                 │   (one table)    │
  │ JSON job files   │                 │                  │
  │ (warming_queue/) │  ──► GONE       │   That's it.     │
  ├──────────────────┤                 │                  │
  │ WarmingQueue-    │                 └──────────────────┘
  │ Service (file    │
  │ based)           │  ──► GONE
  ├──────────────────┤
  │ File byte-offset │
  │ resume logic     │  ──► GONE
  ├──────────────────┤
  │ Quarantine dir   │  ──► GONE
  ├──────────────────┤
  │ Archive dir      │  ──► GONE
  └──────────────────┘
```
