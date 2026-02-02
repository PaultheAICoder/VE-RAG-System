---
title: Cache Warming File Upload
status: implemented
created: 2026-02-01
author: admin
type: Enhancement
complexity: SIMPLE
stack: fullstack
---

# Cache Warming File Upload

## Summary

Enhance the existing cache warming functionality to allow administrators to upload a file containing questions to seed the cache, with real-time progress monitoring via Server-Sent Events (SSE). This replaces the need for manual typing of questions one-by-one.

## Goals

- Enable bulk cache warming via file upload (.txt or .csv)
- Provide real-time progress feedback during warming
- Show success/failure status for each query
- Maintain existing manual text entry as fallback option

## Scope

### In Scope

- File upload UI component in CacheWarmingCard
- Support for .txt files (one question per line)
- Support for .csv files (requires 'question' or 'query' column header)
- SSE endpoint for real-time progress updates
- Progress bar with count (e.g., "Processing 15/100 queries")
- Per-query success/fail status display
- Maximum 100 questions per file
- File validation (format, size, question count)

### Out of Scope

- Scheduling/automation of file-based warming
- Question deduplication against existing cache
- Multi-file upload in single operation
- Question preview/editing before warming
- Persisting uploaded files

## Technical Specification

### Backend

#### New Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/admin/cache/warm-file` | Upload file and start warming |
| GET | `/api/admin/cache/warm-progress/{job_id}` | SSE stream for progress |
| POST | `/api/admin/cache/warm-retry` | Retry specific failed queries |

#### Request/Response Schemas

**POST /api/admin/cache/warm-file**

Request: `multipart/form-data`
- `file`: File (.txt or .csv, max 100KB)

Response (202 Accepted):
```json
{
  "job_id": "uuid",
  "queued": 45,
  "message": "Cache warming started. Connect to SSE for progress.",
  "sse_url": "/api/admin/cache/warm-progress/{job_id}"
}
```

**GET /api/admin/cache/warm-progress/{job_id}** (SSE Stream)

Event types:
```
event: progress
data: {"processed": 5, "total": 45, "current_query": "What is..."}

event: result
data: {"query": "What is...", "status": "success", "cached": true}

event: result
data: {"query": "How do I...", "status": "failed", "error": "Low confidence (32%)"}

event: complete
data: {"total": 45, "success": 42, "failed": 3, "duration_seconds": 180, "failed_queries": ["query1", "query2", "query3"]}
```

**Estimated Time Calculation:**
- Track rolling average of last 5 query processing times
- `remaining_seconds = avg_time_per_query * (total - processed)`
- Update estimate with each `progress` event

**POST /api/admin/cache/warm-retry**

Request:
```json
{
  "queries": ["failed query 1", "failed query 2", "failed query 3"]
}
```

Response (202 Accepted):
```json
{
  "job_id": "uuid",
  "queued": 3,
  "message": "Retry warming started. Connect to SSE for progress.",
  "sse_url": "/api/admin/cache/warm-progress/{job_id}"
}
```

Note: Uses same SSE progress stream as file upload.

#### File Parsing Logic

**Text files (.txt)**:
- Split by newlines
- Trim whitespace
- Filter empty lines
- Limit to first 100 non-empty lines

**CSV files (.csv)**:
- Parse with csv module
- Look for column header: 'question', 'query', or 'text' (case-insensitive)
- If no header match, use first column
- Limit to first 100 rows (excluding header)

#### Validation Rules

| Rule | Error Message |
|------|---------------|
| File size > 100KB | "File too large. Maximum size is 100KB." |
| No valid questions found | "No valid questions found in file." |
| Questions > 100 | "Too many questions. Maximum is 100 per file." |
| Invalid file type | "Invalid file type. Supported: .txt, .csv" |
| Empty questions after parsing | "File contains no valid questions after parsing." |

#### Models Affected

| Model | Changes |
|-------|---------|
| None | No database changes required |

Note: Warming jobs are ephemeral (in-memory tracking during SSE stream). No persistence needed.

### Frontend

#### Components to Modify

| Component | Changes |
|-----------|---------|
| `CacheWarmingCard.tsx` | Add file upload input, progress display |

#### New State

| State | Type | Initial | Purpose |
|-------|------|---------|---------|
| `uploadedFile` | `File \| null` | `null` | Selected file |
| `warmingJob` | `WarmingJob \| null` | `null` | Active job with progress |
| `results` | `QueryResult[]` | `[]` | Per-query results |
| `failedQueries` | `string[]` | `[]` | Queries that failed (for retry) |
| `estimatedTimeRemaining` | `number \| null` | `null` | Seconds remaining |
| `queryTimes` | `number[]` | `[]` | Rolling window for time estimate |

#### New Types

```typescript
interface WarmingJob {
  job_id: string;
  total: number;
  processed: number;
  status: 'pending' | 'running' | 'complete' | 'error';
}

interface QueryResult {
  query: string;
  status: 'success' | 'failed';
  error?: string;
  cached: boolean;
}
```

#### UI Layout Changes

Current layout (preserve):
```
[Flame Icon] Cache Warming
[Description text]
[Textarea for manual entry]
[Query count] [Warm Cache button]
```

Enhanced layout:
```
[Flame Icon] Cache Warming
[Description text]

--- Tab: Manual Entry ---
[Textarea for manual entry]
[Query count] [Warm Cache button]

--- Tab: File Upload ---
[File dropzone / input]
[Selected: questions.txt (45 queries)]
[Upload & Warm button]

--- Progress Section (when warming) ---
[Progress bar: 15/45]
[Estimated time remaining: ~2m 30s]
[Current: "What is the return policy?"]
[Results list with success/fail badges]

--- Completion Section (when done) ---
[Summary: 42 succeeded, 3 failed]
[Retry Failed (3)] button  ← only if failures exist
[Failed queries collapsible list]
```

#### SSE Connection Logic

```typescript
// Connect to SSE when job starts
const eventSource = new EventSource(`/api/admin/cache/warm-progress/${jobId}`);

eventSource.addEventListener('progress', (e) => {
  const data = JSON.parse(e.data);
  setWarmingJob(prev => ({ ...prev, processed: data.processed }));
});

eventSource.addEventListener('result', (e) => {
  const data = JSON.parse(e.data);
  setResults(prev => [...prev, data]);
});

eventSource.addEventListener('complete', (e) => {
  const data = JSON.parse(e.data);
  setWarmingJob(prev => ({ ...prev, status: 'complete' }));
  eventSource.close();
});

eventSource.onerror = () => {
  setWarmingJob(prev => ({ ...prev, status: 'error' }));
  eventSource.close();
};
```

### API Contract

#### File Upload Request

```
POST /api/admin/cache/warm-file
Content-Type: multipart/form-data
Authorization: Bearer {token}

------boundary
Content-Disposition: form-data; name="file"; filename="questions.txt"
Content-Type: text/plain

What is the return policy?
How do I request time off?
...
------boundary--
```

#### File Upload Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "queued": 45,
  "message": "Cache warming started. Connect to SSE for progress.",
  "sse_url": "/api/admin/cache/warm-progress/550e8400-e29b-41d4-a716-446655440000"
}
```

## Risk Flags

### SSE_CONNECTION Risk
First SSE endpoint in codebase:
- [ ] Verify FastAPI SSE support (use `sse-starlette` or manual `StreamingResponse`)
- [ ] Handle client disconnection gracefully
- [ ] Set appropriate timeouts

### LONG_RUNNING_TASK Risk
Cache warming can take 3-15 minutes:
- [ ] Use background task to avoid request timeout
- [ ] Track job state in memory (dict with job_id key)
- [ ] Clean up completed jobs after 1 hour

### FILE_UPLOAD Risk
Accepting user files:
- [ ] Validate file extension
- [ ] Limit file size (100KB)
- [ ] Parse safely (handle malformed CSV)
- [ ] No file storage (process in memory only)

## Acceptance Criteria

- [ ] Admin can upload .txt file with one question per line
- [ ] Admin can upload .csv file with 'question' column
- [ ] File validation rejects files > 100KB
- [ ] File validation rejects files with > 100 questions
- [ ] Progress bar updates in real-time during warming
- [ ] Estimated time remaining displays and updates during warming
- [ ] Each query shows success/fail status as it completes
- [ ] Final summary shows total success/failed counts
- [ ] Failed queries are listed and can be retried with "Retry Failed" button
- [ ] Retry uses same progress UI as initial warming
- [ ] Manual text entry still works as before (preserved)
- [ ] SSE connection closes gracefully on completion
- [ ] SSE connection handles client disconnect

## Resolved Questions

- **Show estimated time remaining?** → Yes, calculate based on rolling average of completed query times
- **Retry failed queries?** → Yes, save failed queries and allow "Retry Failed" button at the end

---

**Next Steps**:
1. Review this spec
2. Fill any [TODO] sections
3. Run `/spec-review specs/CACHE_WARMING_FILE_UPLOAD.md`
