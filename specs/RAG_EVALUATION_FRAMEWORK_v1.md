# RAG Evaluation Framework — RAGAS Integration

| Field | Value |
|-------|-------|
| **Status** | FINAL — Ready for Implementation |
| **Version** | v1.4 |
| **Author** | Claude (AI) + jjob |
| **Date** | 2026-02-11 |
| **Depends On** | RAG Service (rag_service.py), WarmingWorker pattern (warming_worker.py) |

---

## 1. Problem Statement

The system has **no way to measure RAG quality**. When documents are updated, models change, or prompts are tuned, there is no objective signal on whether retrieval and generation improved or regressed.

### Symptoms

1. **No regression detection** — After updating chunking strategy, embedding model, or prompt templates, there is no automated way to detect quality regressions before they reach users.

2. **No production quality signal** — Live RAG responses are not evaluated. Admins rely on user complaints to detect quality issues.

3. **No dataset management** — No infrastructure for storing evaluation datasets (question/answer pairs with ground truth) for repeatable benchmarking.

4. **Competitive gap** — Competitor analysis (Feb 2026) identified RAG evaluation as a P0 gap. Enterprise buyers increasingly ask "How do I know this is working well?" during pilots.

### Root Cause

The system was built output-first: documents in, answers out. Quality measurement was deferred. RAGAS (the industry-standard RAG evaluation library) supports Ollama via LangChain wrappers, making local evaluation feasible without external API calls — aligning with the air-gap requirement.

---

## 2. Proposed Solution

**Integrate RAGAS as the evaluation engine.** Provide batch evaluation against stored datasets, live production monitoring via sampling, and an admin dashboard for visibility.

### Architecture

```
Batch Evaluation Flow:
    Admin triggers run (dataset_id, tag_scope)
    → EvaluationWorker claims run (lease + heartbeat)
    → For each sample in dataset:
        → Check cancel flag
        → Run RAG pipeline with tag_scope (generate answer + retrieve contexts)
        → Run RAGAS metrics (Faithfulness, AnswerRelevancy, LLMContextPrecision, LLMContextRecall)
        → Store per-sample scores in EvaluationSample
        → Retry on transient failure (1 retry, 10s backoff)
    → Compute aggregate scores → Mark run complete
    → Admin views results in dashboard

Live Monitoring Flow:
    User sends chat query
    → RAGService.generate() returns response + eval payload
    → Sampling decision (configurable rate, default 0%)
    → Submit to bounded evaluation queue (max 2 concurrent, drop if full)
    → Worker picks up:
        → Run RAGAS (Faithfulness + AnswerRelevancy only — no ground truth)
        → Store in LiveEvaluationScore (auto-purged after retention TTL)
    → Never blocks user response

Dataset Management:
    → Manual: Admin creates Q&A pairs via API
    → RAGBench: Import from pre-downloaded HuggingFace files (air-gap)
    → Synthetic: RAGAS TestsetGenerator creates Q&A from user documents
```

### RAGAS Metrics

| Metric | What It Measures | Requires Ground Truth | RAGAS Input Fields | Used In |
|--------|-----------------|----------------------|-------------------|---------|
| **Faithfulness** | Does the answer faithfully represent the context without hallucination? | No | `user_input`, `response`, `retrieved_contexts` | Batch + Live |
| **AnswerRelevancy** | How relevant is the answer to the question? | No | `user_input`, `response`, `retrieved_contexts` | Batch + Live |
| **LLMContextPrecision** | Are retrieved contexts ranked correctly (most relevant first)? | Yes | `user_input`, `retrieved_contexts`, `reference` | Batch only |
| **LLMContextRecall** | Does the context contain enough information to answer? | Yes | `user_input`, `retrieved_contexts`, `reference` | Batch only |

**Why these 4**: They cover both retrieval quality (Context*) and generation quality (Faithfulness, AnswerRelevancy). Faithfulness and AnswerRelevancy work without ground truth, enabling live monitoring. LLMContextPrecision and LLMContextRecall require reference answers, limiting them to batch evaluation.

**RAGAS v0.4 input contract**: RAGAS v0.4 uses `reference` (a string) for LLMContextRecall — it does NOT use a separate `reference_contexts` list. The `reference_contexts` field in our schema is stored for **human review and dataset provenance only**, not passed to RAGAS. RAGAS v0.4 field names: `user_input` (question), `response` (answer), `retrieved_contexts` (list of context strings), `reference` (ground truth string). See Section 5.2 for the exact field mapping.

### Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Evaluation engine | RAGAS v0.4 via `llm_factory()` + OpenAI-compatible Ollama endpoint | Industry standard, direct Ollama integration via `/v1` API, no LangChain wrapper needed |
| LLM for evaluation | Same Ollama model as RAG | Air-gap compliant, no additional infrastructure |
| Processing model | Sequential (not parallel) | Ollama is the bottleneck — parallelism causes resource contention |
| Ollama timeout | 120 seconds per RAGAS call | Local evaluation is slow (10-60s per metric on typical hardware) |
| Live monitoring | Bounded async queue with drop policy | Must never block user-facing RAG responses; backpressure prevents runaway |
| Default sample rate | 0% (disabled) | Don't double Ollama load without explicit admin opt-in |
| Live metrics | Faithfulness + AnswerRelevancy only | No ground truth available for live queries |
| Dataset storage | Pre-downloaded files at `data/ragbench/` | Air-gap requirement — no runtime HuggingFace downloads |
| Access control | Tag-scoped evaluation with opt-in admin bypass | Evaluation must reflect real user experience by default |
| Worker concurrency | Lease-based claiming with heartbeat | Safe for multi-process deployments |
| ID format | Hex UUIDs (`generate_uuid()`) | Matches existing codebase convention; no prefixed IDs |

---

## 3. Database Schema

### ID Format Convention

All tables use the existing `generate_uuid()` pattern from `ai_ready_rag.db.models.base`:

```python
def generate_uuid() -> str:
    return str(uuid.uuid4())  # e.g. "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

id = Column(String, primary_key=True, default=generate_uuid)
```

IDs are **hyphenated UUID v4 strings** (e.g., `"a1b2c3d4-e5f6-7890-abcd-ef1234567890"`). This matches the existing codebase convention (`db/models/base.py:11`). API examples in this spec use realistic UUIDs.

**Note on DDL vs SQLAlchemy**: The SQL DDL snippets below are **illustrative** for schema review. The actual implementation uses SQLAlchemy models with `default=generate_uuid` — the DDL `DEFAULT` clauses are never executed at runtime. To avoid confusion, DDL snippets omit the `DEFAULT` clause on `id` columns; the Python model definition is the canonical source.

### New Table: `evaluation_runs`

Tracks batch evaluation executions. Mirrors the `warming_batches` pattern including lease-based worker claiming.

```sql
CREATE TABLE evaluation_runs (
    id                      TEXT PRIMARY KEY,                    -- generate_uuid() in SQLAlchemy model
    name                    TEXT NOT NULL,                       -- "Nightly regression v12"
    description             TEXT,
    dataset_id              TEXT NOT NULL REFERENCES evaluation_datasets(id) ON DELETE RESTRICT,
    status                  TEXT NOT NULL DEFAULT 'pending',     -- pending | running | completed | completed_with_errors | failed | cancelled
    total_samples           INTEGER NOT NULL DEFAULT 0,
    completed_samples       INTEGER NOT NULL DEFAULT 0,
    failed_samples          INTEGER NOT NULL DEFAULT 0,

    -- Access control scope for evaluation
    tag_scope               TEXT,                                -- JSON list of tags, e.g. '["hr","finance"]'. NULL = use admin_bypass_tags mode
    admin_bypass_tags       BOOLEAN NOT NULL DEFAULT 0,          -- If true, evaluation runs with user_tags=None (full corpus). Labeled in results.

    -- Aggregate scores (computed on completion)
    avg_faithfulness        REAL,
    avg_answer_relevancy    REAL,
    avg_llm_context_precision REAL,
    avg_llm_context_recall  REAL,

    -- Reproducibility snapshot (see Section 3.1 for required fields)
    model_used              TEXT NOT NULL,                       -- e.g. "qwen3:8b"
    embedding_model_used    TEXT NOT NULL,                       -- e.g. "nomic-embed-text"
    config_snapshot         TEXT NOT NULL,                       -- JSON, required — see Section 3.1

    -- Worker lease fields (multi-process safe)
    worker_id               TEXT,                                -- Which worker owns this run
    worker_lease_expires_at DATETIME,                            -- Lease expiry for stale detection
    is_cancel_requested     BOOLEAN NOT NULL DEFAULT 0,          -- Admin cancel flag

    -- Capacity controls
    max_duration_hours      REAL,                                -- Auto-pause if exceeded (NULL = use global default)

    error_message           TEXT,
    triggered_by            TEXT REFERENCES users(id) ON DELETE SET NULL,
    started_at              DATETIME,
    completed_at            DATETIME,
    created_at              DATETIME NOT NULL DEFAULT (datetime('now')),
    updated_at              DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_evaluation_runs_status ON evaluation_runs(status, created_at);
CREATE INDEX idx_evaluation_runs_dataset ON evaluation_runs(dataset_id);
CREATE INDEX idx_evaluation_runs_lease ON evaluation_runs(worker_lease_expires_at);
```

### 3.1 Config Snapshot Schema (Mandatory)

The `config_snapshot` column stores a JSON object with the following **required** fields. Runs are only comparable when these values match.

```json
{
    "chat_model": "qwen3:8b",
    "embedding_model": "nomic-embed-text",
    "temperature": 0.1,
    "chunking_strategy": "hybrid",
    "chunk_max_tokens": 512,
    "chunk_overlap_tokens": 50,
    "retrieval_top_k": 5,
    "reranker_enabled": false,
    "reranker_model": null,
    "prompt_template_hash": "sha256:a1b2c3...",
    "corpus_doc_count": 142,
    "corpus_last_ingested_at": "2026-02-09T14:30:00Z",
    "rag_timeout_seconds": 60,
    "eval_timeout_seconds": 120
}
```

**`prompt_template_hash`**: SHA-256 of the `RAG_SYSTEM_PROMPT` template string (computed at run creation time). Changes to the prompt template produce a different hash, flagging non-comparable runs.

**`corpus_doc_count`** and **`corpus_last_ingested_at`**: Queried from the documents table at run creation. If the corpus changes between runs, these values differ, alerting the reviewer.

The snapshot is captured at run creation time and is **immutable** — it records the state of the system when the evaluation was triggered, not the current state.

### New Table: `evaluation_samples`

Per-question results within a run. Mirrors the `warming_queries` pattern.

```sql
CREATE TABLE evaluation_samples (
    id                      TEXT PRIMARY KEY,                    -- generate_uuid() in SQLAlchemy model
    run_id                  TEXT NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    sort_order              INTEGER NOT NULL DEFAULT 0,
    status                  TEXT NOT NULL DEFAULT 'pending',     -- pending | processing | completed | failed | skipped

    -- Input (from dataset)
    question                TEXT NOT NULL,
    ground_truth            TEXT,                                -- May be NULL for datasets without reference answers
    reference_contexts      TEXT,                                -- JSON list — stored for human review only, NOT passed to RAGAS

    -- RAG output (captured during evaluation)
    generated_answer        TEXT,
    retrieved_contexts      TEXT,                                -- JSON list of retrieved context strings

    -- RAGAS metric scores (0.0 - 1.0, NULL if not computed)
    faithfulness            REAL,
    answer_relevancy        REAL,
    llm_context_precision   REAL,
    llm_context_recall      REAL,

    generation_time_ms      REAL,                               -- Time for RAG pipeline (float, matches RAGResponse)
    retry_count             INTEGER NOT NULL DEFAULT 0,         -- Number of retry attempts
    error_message           TEXT,
    error_type              TEXT,                                -- Exception class name
    processed_at            DATETIME,
    created_at              DATETIME NOT NULL DEFAULT (datetime('now')),
    updated_at              DATETIME NOT NULL DEFAULT (datetime('now')),

    UNIQUE(run_id, sort_order)
);

CREATE INDEX idx_evaluation_samples_run ON evaluation_samples(run_id);
CREATE INDEX idx_evaluation_samples_status ON evaluation_samples(run_id, status);
```

### New Table: `evaluation_datasets`

Stores named collections of Q&A pairs for repeatable evaluation.

```sql
CREATE TABLE evaluation_datasets (
    id                      TEXT PRIMARY KEY,                    -- generate_uuid() in SQLAlchemy model
    name                    TEXT NOT NULL UNIQUE,                -- "RAGBench-TechQA-50"
    description             TEXT,
    source_type             TEXT NOT NULL,                       -- manual | ragbench | synthetic | live_sample
    source_config           TEXT,                                -- JSON: {"subset": "techqa", "max_samples": 50}
    sample_count            INTEGER NOT NULL DEFAULT 0,          -- Denormalized. MUST be updated on dataset_samples insert/delete.
    created_by              TEXT REFERENCES users(id) ON DELETE SET NULL,
    created_at              DATETIME NOT NULL DEFAULT (datetime('now')),
    updated_at              DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_evaluation_datasets_source ON evaluation_datasets(source_type);
```

### New Table: `dataset_samples`

Individual Q&A pairs within a dataset.

```sql
CREATE TABLE dataset_samples (
    id                      TEXT PRIMARY KEY,                    -- generate_uuid() in SQLAlchemy model
    dataset_id              TEXT NOT NULL REFERENCES evaluation_datasets(id) ON DELETE CASCADE,
    question                TEXT NOT NULL,
    ground_truth            TEXT,                                -- Reference answer (optional)
    reference_contexts      TEXT,                                -- JSON list — for human review/provenance only
    metadata                TEXT,                                -- JSON: source info, difficulty, category
    sort_order              INTEGER NOT NULL DEFAULT 0,
    created_at              DATETIME NOT NULL DEFAULT (datetime('now')),

    UNIQUE(dataset_id, sort_order)
);

CREATE INDEX idx_dataset_samples_dataset ON dataset_samples(dataset_id);
```

### New Table: `live_evaluation_scores` (Phase 3)

Stores evaluation scores for sampled live queries. Subject to automatic retention purging.

```sql
CREATE TABLE live_evaluation_scores (
    id                      TEXT PRIMARY KEY,                    -- generate_uuid() in SQLAlchemy model
    message_id              TEXT REFERENCES chat_messages(id) ON DELETE SET NULL,  -- NOTE: Always NULL in Phase 3 (message_id not known at enqueue time). Retained for future backfill.
    question                TEXT NOT NULL,
    generated_answer        TEXT NOT NULL,
    faithfulness            REAL,
    answer_relevancy        REAL,
    generation_time_ms      REAL,                               -- Float, matches RAGResponse.generation_time_ms
    eval_time_ms            REAL,
    created_at              DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_live_eval_created ON live_evaluation_scores(created_at);
```

**Privacy & Retention (Review Finding #12):**
- Rows are automatically purged after `eval_live_retention_days` (default 30 days) by the cleanup job.
- The `GET /api/evaluations/live/scores` endpoint is **admin-only** (requires admin role).
- Future: Add `eval_live_redact_pii` setting (default false). When enabled, `question` and `generated_answer` are replaced with `"[REDACTED]"` after metric scores are computed and stored. Scores remain; raw text is discarded.

---

## 4. State Machines

### 4.1 Evaluation Run State Machine

```
            ┌──────────┐
            │ pending  │
            └────┬─────┘
                 │ Worker acquires lease (Section 6.1)
                 ▼
            ┌──────────┐
     ┌──────│ running  │──────┬──────────────┐
     │      └────┬─────┘      │              │
     │           │             │              │
  All samples  Error    Cancel requested   Max duration
  resolved      │        (Section 4.3)     exceeded
     │          ▼             │              │
     │    ┌──────────┐  ┌───────────┐       │
     │    │  failed   │  │ cancelled │  ◄────┘
     │    └──────────┘  └───────────┘
     │                       (terminal)
     ▼
┌─────────────────┐
│ Determine final │
│ status          │
└────────┬────────┘
         │
    ┌────┴────────────────┐
    │                     │
    ▼                     ▼
┌────────────────┐  ┌───────────────────────┐
│   completed    │  │ completed_with_errors  │
│ (all succeeded)│  │ (some samples failed)  │
└────────────────┘  └───────────────────────┘
```

**Valid status values**: `pending`, `running`, `completed`, `completed_with_errors`, `failed`, `cancelled`

#### 4.1.1 Run Completion Criteria

```python
failed_count = count(samples WHERE status = 'failed')
total_count = run.total_samples

if failed_count == 0:
    run.status = "completed"
else:
    run.status = "completed_with_errors"
```

A run-level `failed` status is reserved for infrastructure failures (e.g., Ollama unreachable for all samples, worker crash without recovery after lease expiry).

### 4.2 Evaluation Sample State Machine

```
    ┌─────────┐
    │ pending │
    └────┬────┘
         │  Worker claims sample (UPDATE WHERE status = 'pending')
         ▼
    ┌────────────┐
    │ processing │
    └─────┬──────┘
          │
    ┌─────┼──────────┐
    │     │          │
    ▼     ▼          ▼
┌──────┐ ┌────────┐ ┌─────────┐
│ done │ │ failed │ │ skipped │
└──────┘ └───┬────┘ └─────────┘
              │
              │ Transient error + retries left
              ▼
         ┌─────────┐
         │ pending │  (retry_count incremented)
         └─────────┘
```

**Valid status values**: `pending`, `processing`, `completed`, `failed`, `skipped`

- `skipped`: Set when a run is cancelled — remaining `pending` samples become `skipped`
- Retry: On transient error with retries remaining, status resets to `pending` with incremented `retry_count`

### 4.2.1 Distributed State Transition Invariants

All status transitions are enforced via atomic `UPDATE ... WHERE status = <expected>` queries. If the `WHERE` clause matches 0 rows, the transition is rejected (another worker or cancel won the race). No in-memory status checks — the database is the source of truth.

**Run transitions (atomic SQL guards):**

| From | To | Guard (`WHERE` clause) | Trigger |
|------|----|----------------------|---------|
| `pending` | `running` | `status = 'pending'` | Worker lease claim (Section 6.1) |
| `running` | `completed` | `status = 'running' AND worker_id = self.worker_id` | All samples resolved, 0 failures |
| `running` | `completed_with_errors` | `status = 'running' AND worker_id = self.worker_id` | All samples resolved, some failures |
| `running` | `failed` | `status = 'running' AND worker_id = self.worker_id` | Infrastructure error (catch-all) |
| `running` | `cancelled` | `status = 'running' AND worker_id = self.worker_id` | `is_cancel_requested = True` or max duration |
| `pending` | `cancelled` | `status = 'pending'` | Cancel on un-started run |
| `running` (stale) | `pending` | `status = 'running' AND worker_lease_expires_at < now()` | Stale recovery (Section 6.4) |

**Sample transitions (atomic SQL guards):**

| From | To | Guard (`WHERE` clause) | Trigger |
|------|----|----------------------|---------|
| `pending` | `processing` | `status = 'pending'` | Worker claims sample |
| `processing` | `completed` | `status = 'processing'` | RAGAS scores computed |
| `processing` | `failed` | `status = 'processing'` | Non-retryable error or retries exhausted |
| `failed` | `pending` | `status = 'failed' AND retry_count < max_retries` | Transient error retry |
| `pending` | `skipped` | `status = 'pending'` | Run cancelled |

**Race resolution rules:**
1. **Cancel vs. processing:** Cancel sets `is_cancel_requested` on the run. The currently processing sample finishes normally. Worker checks the flag before starting the *next* sample.
2. **Cancel vs. retry:** If a sample fails with a retryable error and the run is simultaneously cancelled, the retry resets the sample to `pending`, then the cancel sweep sets it to `skipped`. The cancel sweep's `UPDATE WHERE status = 'pending'` catches it.
3. **Lease expiry vs. active processing:** If a worker's lease expires while it is processing a sample, another worker may reclaim the run. The original worker's subsequent `UPDATE WHERE worker_id = self.worker_id` on the run will match 0 rows — it detects it lost the lease and stops processing.
4. **Double-claim prevention:** Two workers racing to claim the same run: both issue `UPDATE WHERE status = 'pending'`. SQLite's write serialization ensures exactly one succeeds (`rowcount = 1`); the other gets `rowcount = 0` and moves on.

### 4.3 Cancellation Semantics

**Cancel** (`is_cancel_requested = True`):
- Admin calls `POST /api/evaluations/runs/{run_id}/cancel`
- Worker checks `is_cancel_requested` **before** starting each sample
- The currently processing sample **runs to completion** (no mid-inference interruption)
- After current sample finishes, worker:
  1. Sets all remaining `pending` samples to `skipped`
  2. Computes aggregates for completed samples
  3. Sets run status to `cancelled`
  4. Releases lease

**Idempotency**: Calling cancel on an already-cancelled or completed run returns 200 (no-op). Calling cancel on a `pending` run transitions directly to `cancelled`.

### 4.4 Idempotency Contract

All state-mutating operations are **replay-safe** — re-executing the same operation produces no side effects if the entity has already transitioned past the expected state.

**Sample processing idempotency:**
- `_claim_sample()` uses `UPDATE WHERE status = 'pending'` — if the sample is already `processing`/`completed`/`failed`/`skipped`, the UPDATE matches 0 rows and returns `False`. No double-processing.
- `process_sample()` is a no-op if `sample.status != 'processing'` at entry — the method checks status before running RAGAS.
- The `UNIQUE(run_id, sort_order)` constraint on `evaluation_samples` prevents duplicate sample rows for the same position in a run.

**Run claiming idempotency:**
- `_claim_run()` uses `UPDATE WHERE (status = 'pending' OR (status = 'running' AND lease_expired))` — if another worker already claimed the run, the UPDATE matches 0 rows. No duplicate claiming.
- Re-claiming the same run by the same worker (restart scenario) is explicitly allowed: `OR (status = 'running' AND worker_id = self.worker_id)`.

**Live score idempotency:**
- Live scores are append-only (INSERT, never UPDATE). Duplicate scores for the same query are tolerable — they represent independent evaluations of separate RAG invocations, not retries.
- No deduplication constraint is needed because the sampling hook fires at most once per `generate()` call.

**Aggregate computation idempotency:**
- `compute_aggregates()` is a pure function that reads completed samples and overwrites the aggregate columns. Re-running it produces the same result — safe to call after retries or partial failures.

---

## 5. EvaluationService

### 5.1 Service Design

```python
class EvaluationService:
    """Wraps RAGAS evaluation with Ollama via OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings, vector_service=None):
        self.settings = settings
        self._vector_service = vector_service
        self._rag_service: RAGService | None = None

    def _get_ragas_llm(self):
        """Get RAGAS-compatible LLM via Ollama's OpenAI-compatible endpoint."""
        from openai import OpenAI
        from ragas.llms import llm_factory

        client = OpenAI(
            api_key="ollama",  # Ollama ignores API key but field is required
            base_url=f"{self.settings.ollama_base_url}/v1",
            timeout=self.settings.eval_timeout_seconds,
        )
        return llm_factory(
            model=self.settings.chat_model,
            provider="openai",
            client=client,
        )

    def _get_ragas_embeddings(self):
        """Get RAGAS-compatible embeddings via Ollama's OpenAI-compatible endpoint."""
        from openai import OpenAI
        from ragas.embeddings import embedding_factory

        client = OpenAI(
            api_key="ollama",
            base_url=f"{self.settings.ollama_base_url}/v1",
        )
        return embedding_factory(
            model=self.settings.embedding_model,
            provider="openai",
            client=client,
        )

    async def create_run(self, db: Session, dataset_id: str, name: str,
                         description: str | None, triggered_by: str | None,
                         tag_scope: list[str] | None = None,
                         admin_bypass_tags: bool = False) -> EvaluationRun:
        """Create a new evaluation run from a dataset.

        Args:
            tag_scope: Tags to use for retrieval filtering. Evaluation results
                reflect what a user with these tags would see.
            admin_bypass_tags: If True, run with user_tags=None (full corpus access).
                Requires admin role. Results are labeled as "admin bypass" in responses.

        Validation rules (mutually exclusive):
            - tag_scope provided AND admin_bypass_tags=False → scoped evaluation (default)
            - tag_scope=None AND admin_bypass_tags=True → admin bypass (full corpus)
            - tag_scope provided AND admin_bypass_tags=True → 422 (mutually exclusive)
            - tag_scope=None AND admin_bypass_tags=False → 422 (must specify scope)

        Copies dataset samples into evaluation_samples for this run.
        Captures config_snapshot at creation time (Section 3.1).
        Returns the run in 'pending' status for the worker to pick up.
        """
        # Validate: tag_scope and admin_bypass_tags are mutually exclusive
        # Validate: one of tag_scope or admin_bypass_tags must be set
        # Validate: admin_bypass_tags requires admin role on triggered_by user
        # Validate: dataset sample count <= eval_max_samples_per_run
        # Capture config_snapshot (Section 3.1)
        ...

    async def process_sample(self, db: Session, sample: EvaluationSample,
                             run: EvaluationRun) -> bool:
        """Process a single evaluation sample.

        1. Run RAG pipeline with run.tag_scope to get answer + contexts
        2. Run RAGAS metrics against the sample
        3. Store scores in the sample row
        Returns True on success, False on failure.
        """
        ...

    async def evaluate_single(self, question: str, answer: str,
                               contexts: list[str],
                               ground_truth: str | None = None) -> dict[str, float]:
        """Run RAGAS metrics on a single Q&A pair.

        Used by both batch processing and live monitoring.
        Returns dict of metric_name → score (0.0-1.0).

        RAGAS v0.4 field mapping (see Section 5.2):
        - question → "user_input"
        - answer → "response"
        - contexts → "retrieved_contexts" (list of context strings)
        - ground_truth → "reference" (string, for LLMContextRecall)
        """
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import Faithfulness, AnswerRelevancy

        ragas_llm = self._get_ragas_llm()
        ragas_embeddings = self._get_ragas_embeddings()

        metrics = [
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ]
        if ground_truth:
            from ragas.metrics import LLMContextPrecision, LLMContextRecall
            metrics.extend([
                LLMContextPrecision(llm=ragas_llm),
                LLMContextRecall(llm=ragas_llm),
            ])

        eval_sample = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        }
        if ground_truth:
            eval_sample["reference"] = ground_truth

        dataset = EvaluationDataset.from_list([eval_sample])

        # IMPORTANT: ragas.evaluate() is synchronous and CPU/IO-bound (LLM calls).
        # Run in thread pool to avoid blocking the async event loop (heartbeat,
        # SSE streams, other workers).
        result = await asyncio.to_thread(evaluate, dataset=dataset, metrics=metrics)
        df = result.to_pandas()

        score_columns = [c for c in df.columns if c not in
                         ("user_input", "response", "retrieved_contexts", "reference")]
        return {col: float(df[col].iloc[0]) for col in score_columns}

    async def compute_aggregates(self, db: Session, run_id: str) -> None:
        """Calculate average scores across all completed samples in a run."""
        ...

    async def score_live_query(self, question: str, answer: str,
                                contexts: list[str],
                                message_id: str | None = None,
                                generation_time_ms: float | None = None) -> None:
        """Score a live query (Faithfulness + AnswerRelevancy only).

        Called from the live evaluation worker (Section 7.2).
        Opens and closes its own DB session.
        Stores result in live_evaluation_scores table.
        """
        db = SessionLocal()
        try:
            start = time.perf_counter()
            scores = await self.evaluate_single(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=None,
            )
            eval_time_ms = int((time.perf_counter() - start) * 1000)

            score = LiveEvaluationScore(
                question=question,
                generated_answer=answer,
                message_id=message_id,
                faithfulness=scores.get("faithfulness"),
                answer_relevancy=scores.get("answer_relevancy"),
                generation_time_ms=generation_time_ms,
                eval_time_ms=eval_time_ms,
            )
            db.add(score)
            db.commit()
        finally:
            db.close()
```

### 5.2 RAG Pipeline Interface for Evaluation

**Problem (Review Finding #2):** `RAGResponse` (rag_service.py:158) has no `contexts` field. The evaluation service needs access to retrieved context chunks, which are internal to `generate()`.

**Solution:** Add a dedicated evaluation entry point that returns both the response and the retrieved contexts, without changing the existing `RAGResponse` dataclass.

```python
@dataclass
class EvalRAGResult:
    """Extended RAG result for evaluation — includes retrieved contexts."""
    response: RAGResponse
    retrieved_contexts: list[str]   # Plain text of each retrieved chunk
    retrieval_scores: list[float]   # Similarity scores for each chunk

# New method on RAGService:
async def generate_for_eval(self, request: RAGRequest, db: Session) -> EvalRAGResult:
    """Run the full retrieval+generation pipeline and return contexts alongside the response.

    Unlike generate(), this method:
    - Skips curated Q&A lookup (evaluation needs fresh RAG answers)
    - Skips cache lookup/store (evaluation must not use cached responses)
    - Skips direct routing (evaluation always requires retrieved contexts)

    Runs steps 2-9 of generate() directly: retrieve → token budget →
    build prompt → LLM generate → extract citations → confidence →
    action → build RAGResponse. Returns both the response and the
    intermediate context chunks that generate() normally discards.

    Used by EvaluationService.process_sample() and nowhere else.
    """
    # Implementation note: Do NOT refactor generate() itself.
    # Instead, extract the retrieval+generation core (rag_service.py lines
    # ~1546-1704) into a private _run_rag_pipeline() that returns
    # (RAGResponse, list[QualityContext]). Then:
    #   - generate() calls _run_rag_pipeline() after its early-return checks
    #   - generate_for_eval() calls _run_rag_pipeline() directly
    # This avoids touching the hot path while sharing the core logic.
    ...
```

### 5.3 RAGAS Execution Detail

```python
# Within process_sample():

# 1. Determine access scope from run configuration
tag_scope = json.loads(run.tag_scope) if run.tag_scope else None
user_tags = None if run.admin_bypass_tags else tag_scope

# 2. Build RAG request with appropriate tag scope
rag_request = RAGRequest(
    query=sample.question,
    user_tags=user_tags,        # Scoped to run's tag_scope (or None if admin bypass)
    is_warming=False,
)

# 3. Run RAG pipeline via evaluation entry point (Section 5.2)
eval_result = await self._rag_service.generate_for_eval(rag_request, db)

# 4. Run RAGAS evaluation with correct field mapping
scores = await self.evaluate_single(
    question=sample.question,
    answer=eval_result.response.answer,
    contexts=eval_result.retrieved_contexts,      # From EvalRAGResult
    ground_truth=sample.ground_truth,             # String passed to RAGAS LLMContextRecall
    # NOTE: sample.reference_contexts is NOT passed to RAGAS — it is stored
    # for human review only (see Section 2, RAGAS v0.4 input contract)
)

# 5. Store results
sample.generated_answer = eval_result.response.answer
sample.retrieved_contexts = json.dumps(eval_result.retrieved_contexts)
sample.faithfulness = scores.get("faithfulness")
sample.answer_relevancy = scores.get("answer_relevancy")
sample.llm_context_precision = scores.get("llm_context_precision")
sample.llm_context_recall = scores.get("llm_context_recall")
sample.generation_time_ms = eval_result.response.generation_time_ms
sample.status = "completed"
sample.processed_at = datetime.utcnow()
db.commit()
```

---

## 6. EvaluationWorker

Mirrors the `WarmingWorker` pattern: background asyncio task that polls for pending runs, with lease-based claiming for multi-process safety.

### 6.1 Lease-Based Run Claiming

```python
# Atomic claim — prevents duplicate processing in multi-worker deployments
result = db.execute(
    update(EvaluationRun)
    .where(
        EvaluationRun.id == run_id,
        or_(
            # Case 1: Run is pending (never started)
            EvaluationRun.status == "pending",
            # Case 2: Already ours (restart/retry of same worker)
            and_(
                EvaluationRun.status == "running",
                EvaluationRun.worker_id == self.worker_id,
            ),
            # Case 3: Stale lease (previous worker crashed)
            and_(
                EvaluationRun.status == "running",
                EvaluationRun.worker_lease_expires_at < datetime.now(UTC),
            ),
        ),
    )
    .values(
        status="running",
        worker_id=self.worker_id,
        worker_lease_expires_at=datetime.now(UTC) + timedelta(
            minutes=settings.eval_lease_duration_minutes
        ),
        started_at=func.coalesce(EvaluationRun.started_at, datetime.now(UTC)),
    )
)
db.commit()

if result.rowcount == 0:
    return None  # Run not available — another worker claimed it
```

### 6.2 Worker Implementation

```python
class EvaluationWorker:
    """Background worker that processes pending evaluation runs.

    Uses lease-based claiming for multi-process safety (Section 6.1).
    Processes samples sequentially with retry support (Section 6.3).
    Supports cancellation (Section 4.3) and max duration limits.
    """

    def __init__(self, eval_service: EvaluationService, settings: Settings | None = None):
        self.eval_service = eval_service
        self.settings = settings or get_settings()
        self.worker_id = f"eval-worker-{uuid4().hex[:8]}"
        self._task: asyncio.Task | None = None
        self._lease_task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._current_run_id: str | None = None

    async def start(self):
        """Start the evaluation worker background task."""
        self._shutdown.clear()
        self._task = asyncio.create_task(self._run_loop())
        self._lease_task = asyncio.create_task(self._lease_renewal_loop())
        logger.info(f"EvaluationWorker {self.worker_id} started")

    async def stop(self):
        """Graceful shutdown."""
        logger.info(f"EvaluationWorker {self.worker_id} stopping...")
        self._shutdown.set()
        for task in [self._task, self._lease_task]:
            if task:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=10.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

    async def _lease_renewal_loop(self):
        """Renew lease periodically. Mirrors WarmingWorker._lease_renewal_loop()."""
        while not self._shutdown.is_set():
            if self._current_run_id:
                db = SessionLocal()
                try:
                    db.execute(
                        update(EvaluationRun)
                        .where(
                            EvaluationRun.id == self._current_run_id,
                            EvaluationRun.worker_id == self.worker_id,
                        )
                        .values(
                            worker_lease_expires_at=datetime.now(UTC) + timedelta(
                                minutes=self.settings.eval_lease_duration_minutes
                            )
                        )
                    )
                    db.commit()
                finally:
                    db.close()
            await asyncio.sleep(self.settings.eval_lease_renewal_seconds)

    async def _run_loop(self):
        """Main loop: find pending runs, claim via lease, process samples sequentially."""
        while not self._shutdown.is_set():
            db = SessionLocal()
            try:
                # Find next claimable run: pending OR running with expired lease
                run = (
                    db.query(EvaluationRun)
                    .filter(
                        or_(
                            EvaluationRun.status == "pending",
                            and_(
                                EvaluationRun.status == "running",
                                EvaluationRun.worker_lease_expires_at < datetime.now(UTC),
                            ),
                        )
                    )
                    .order_by(EvaluationRun.created_at.asc())
                    .first()
                )

                if not run:
                    await asyncio.sleep(self.settings.eval_scan_interval_seconds)
                    continue

                # Claim via atomic lease (Section 6.1) — prevents race conditions
                claimed = self._claim_run(db, run.id)
                if not claimed:
                    continue

                self._current_run_id = run.id
                run_start_time = time.monotonic()

                # Process each sample SEQUENTIALLY (Ollama bottleneck)
                samples = (
                    db.query(EvaluationSample)
                    .filter_by(run_id=run.id, status="pending")
                    .order_by(EvaluationSample.sort_order.asc())
                    .all()
                )

                for sample in samples:
                    if self._shutdown.is_set():
                        break

                    # Check cancel flag before each sample (Section 4.3)
                    db.refresh(run)
                    if run.is_cancel_requested:
                        await self._cancel_run(db, run)
                        break

                    # Check max duration
                    max_hours = run.max_duration_hours or self.settings.eval_max_run_duration_hours
                    if max_hours and (time.monotonic() - run_start_time) / 3600 > max_hours:
                        run.status = "cancelled"
                        run.error_message = f"Auto-cancelled: exceeded max duration of {max_hours}h"
                        self._skip_remaining(db, run.id)
                        await self.eval_service.compute_aggregates(db, run.id)
                        run.completed_at = datetime.utcnow()
                        db.commit()
                        break

                    # Claim sample (idempotent — skip if already claimed)
                    claimed = self._claim_sample(db, sample)
                    if not claimed:
                        continue

                    # Process with retry (Section 6.3)
                    success = await self._process_with_retry(db, sample, run)

                    if success:
                        run.completed_samples += 1
                    else:
                        run.failed_samples += 1
                    db.commit()

                    # Throttle between samples
                    if self.settings.eval_delay_between_samples_seconds > 0:
                        await asyncio.sleep(
                            self.settings.eval_delay_between_samples_seconds
                        )
                else:
                    # All samples processed (loop completed without break)
                    await self.eval_service.compute_aggregates(db, run.id)
                    if run.failed_samples == 0:
                        run.status = "completed"
                    else:
                        run.status = "completed_with_errors"
                    run.completed_at = datetime.utcnow()
                    db.commit()

                self._current_run_id = None

            except Exception as e:
                logger.error(f"EvaluationWorker error: {e}")
                if self._current_run_id:
                    try:
                        run = db.query(EvaluationRun).filter_by(id=self._current_run_id).first()
                        if run and run.status == "running":
                            run.status = "failed"
                            run.error_message = str(e)[:500]
                            run.completed_at = datetime.utcnow()
                            db.commit()
                    except Exception:
                        pass
                self._current_run_id = None
            finally:
                db.close()

    def _claim_sample(self, db: Session, sample: EvaluationSample) -> bool:
        """Claim a sample via atomic UPDATE WHERE status = 'pending'."""
        result = db.execute(
            update(EvaluationSample)
            .where(
                EvaluationSample.id == sample.id,
                EvaluationSample.status == "pending",
            )
            .values(status="processing", updated_at=datetime.utcnow())
        )
        db.commit()
        return result.rowcount > 0

    async def _cancel_run(self, db: Session, run: EvaluationRun) -> None:
        """Cancel a run: skip remaining samples, compute partial aggregates."""
        self._skip_remaining(db, run.id)
        await self.eval_service.compute_aggregates(db, run.id)
        run.status = "cancelled"
        run.completed_at = datetime.utcnow()
        db.commit()

    def _skip_remaining(self, db: Session, run_id: str) -> None:
        """Set all pending samples in a run to 'skipped'."""
        db.execute(
            update(EvaluationSample)
            .where(
                EvaluationSample.run_id == run_id,
                EvaluationSample.status == "pending",
            )
            .values(status="skipped")
        )
        db.commit()
```

### 6.3 Retry Policy

```python
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)

async def _process_with_retry(self, db: Session, sample: EvaluationSample,
                               run: EvaluationRun) -> bool:
    """Process a sample with retry on transient errors.

    Retry policy:
    - Max retries: eval_max_retries_per_sample (default: 1)
    - Backoff: eval_retry_backoff_seconds (default: 10)
    - Retryable: ConnectionError, TimeoutError
    - Non-retryable: ValueError, KeyError, RAGAS internal errors → fail immediately
    """
    max_retries = self.settings.eval_max_retries_per_sample
    backoff = self.settings.eval_retry_backoff_seconds

    for attempt in range(max_retries + 1):
        try:
            success = await self.eval_service.process_sample(db, sample, run)
            return success
        except RETRYABLE_EXCEPTIONS as e:
            if attempt < max_retries:
                sample.retry_count = attempt + 1
                sample.status = "pending"
                db.commit()
                logger.warning(
                    f"Eval sample retry {attempt + 1}/{max_retries}: "
                    f"'{sample.question[:50]}...' error={type(e).__name__}, "
                    f"retrying in {backoff}s"
                )
                await asyncio.sleep(backoff)
                # Re-claim sample
                self._claim_sample(db, sample)
            else:
                sample.status = "failed"
                sample.error_message = str(e)[:500]
                sample.error_type = type(e).__name__
                sample.retry_count = attempt + 1
                sample.processed_at = datetime.utcnow()
                db.commit()
                return False
        except Exception as e:
            # Non-retryable — fail immediately
            sample.status = "failed"
            sample.error_message = str(e)[:500]
            sample.error_type = type(e).__name__
            sample.retry_count = attempt
            sample.processed_at = datetime.utcnow()
            db.commit()
            return False
```

### 6.4 Stale Lease Recovery

On server startup, recover runs with expired leases:

```python
async def recover_stale_evaluation_runs() -> int:
    """Reset evaluation runs with expired worker leases to pending."""
    db = SessionLocal()
    try:
        count = (
            db.query(EvaluationRun)
            .filter(
                EvaluationRun.status == "running",
                EvaluationRun.worker_lease_expires_at < datetime.now(UTC),
            )
            .update({
                "status": "pending",
                "worker_id": None,
                "worker_lease_expires_at": None,
            })
        )
        # Also reset orphaned "processing" samples back to "pending"
        db.query(EvaluationSample).filter(
            EvaluationSample.status == "processing",
            EvaluationSample.run_id.in_(
                db.query(EvaluationRun.id).filter(EvaluationRun.status == "pending")
            ),
        ).update({"status": "pending"}, synchronize_session=False)
        db.commit()
        if count:
            logger.warning(f"Recovered {count} stale evaluation run(s)")
        return count
    finally:
        db.close()
```

---

## 7. Live Monitoring (Phase 3)

### 7.1 Evaluation Payload from RAG Pipeline

**Problem (Review Finding #2, applied to live path):** Live monitoring needs retrieved contexts, which `RAGResponse` does not expose.

**Solution:** `generate_for_eval()` (Section 5.2) is used for batch evaluation. For the live path, we add a lightweight context capture that avoids changing the hot-path return type:

```python
# In RAGService.generate(), at the NORMAL return path ONLY (after step 9, ~line 1704).
#
# The 4 early-return paths (curated Q&A, cache hit, direct routing, zero-context)
# do NOT capture eval_payload — they return before this code runs.
# This is intentional: live evaluation only scores fresh RAG-pipeline responses
# with retrieved contexts.

# Capture eval payload for live sampling (lightweight — just text strings)
eval_payload = None
if not request.is_warming and final_chunks:
    eval_payload = {
        "contexts": [c.chunk_text for c in final_chunks],
    }

# Pass eval_payload to sampling hook (not returned to caller)
self._maybe_queue_for_evaluation(request, response, eval_payload)

return response  # RAGResponse unchanged
```

### 7.2 Bounded Evaluation Queue

**Problem (Review Finding #3):** Unbounded `asyncio.create_task()` with no concurrency limit, no backpressure, and no DB session lifecycle management.

**Solution:** Use `asyncio.Queue` with a bounded size and a dedicated consumer task. If the queue is full, samples are silently dropped (backpressure).

```python
class LiveEvaluationQueue:
    """Bounded async queue for live evaluation scoring.

    - Max concurrent evaluations: eval_live_max_concurrent (default: 2)
    - Queue capacity: eval_live_queue_size (default: 10)
    - Drop policy: if queue full, silently drop sample (log at debug level)
    - DB sessions: opened and closed per-evaluation in the consumer
    """

    def __init__(self, eval_service: EvaluationService, settings: Settings):
        self.eval_service = eval_service
        self.settings = settings
        self._queue: asyncio.Queue | None = None
        self._workers: list[asyncio.Task] = []
        self._shutdown = asyncio.Event()

    async def start(self):
        """Start queue and consumer workers."""
        self._queue = asyncio.Queue(maxsize=self.settings.eval_live_queue_size)
        for i in range(self.settings.eval_live_max_concurrent):
            task = asyncio.create_task(self._consumer(i))
            self._workers.append(task)

    async def stop(self):
        """Graceful shutdown."""
        self._shutdown.set()
        for task in self._workers:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    def try_enqueue(self, question: str, answer: str, contexts: list[str],
                    message_id: str | None, generation_time_ms: float | None) -> bool:
        """Non-blocking enqueue. Returns False if queue is full (dropped)."""
        try:
            self._queue.put_nowait({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "message_id": message_id,
                "generation_time_ms": generation_time_ms,
            })
            return True
        except asyncio.QueueFull:
            logger.debug("Live eval queue full — dropping sample")
            return False

    async def _consumer(self, worker_id: int):
        """Consume items from queue, evaluate, store results."""
        while not self._shutdown.is_set():
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self.eval_service.score_live_query(
                    question=item["question"],
                    answer=item["answer"],
                    contexts=item["contexts"],
                    message_id=item["message_id"],
                    generation_time_ms=item["generation_time_ms"],
                )
            except Exception as e:
                logger.debug(f"Live evaluation failed (non-blocking): {e}")
```

**Scaling note:** `LiveEvaluationQueue` uses in-process `asyncio.Queue`, which is correct for the single-process DGX Spark deployment. For future multi-instance deployments, replace with ARQ/Redis consumer group — the `try_enqueue()` interface is designed as a single swap point. The batch `EvaluationWorker` already uses DB-based lease claiming and is multi-instance safe.

### 7.3 Sampling Hook in RAGService

**Injection:** `LiveEvaluationQueue` is injected into `RAGService` via an optional constructor parameter, matching the existing pattern for `cache_service` and `vector_service`:

```python
# In RAGService.__init__():
def __init__(
    self,
    settings: Settings,
    vector_service: VectorServiceProtocol | None = None,
    cache_service: CacheService | None = None,
    default_model: str | None = None,
    live_eval_queue: "LiveEvaluationQueue | None" = None,  # NEW — Phase 3
):
    ...
    self._live_eval_queue = live_eval_queue
```

**Hook method:**

```python
def _maybe_queue_for_evaluation(
    self, request: RAGRequest, response: RAGResponse, eval_payload: dict | None
) -> None:
    """Probabilistically queue this query for live evaluation.

    Non-blocking. Never awaited. Never raises.
    """
    if eval_payload is None or self._live_eval_queue is None:
        return

    try:
        sample_rate = self._get_live_sample_rate()  # Reads from AdminSetting cache
        if sample_rate <= 0.0:
            return
        if random.random() > sample_rate:
            return

        # Non-blocking enqueue — drops if queue full
        self._live_eval_queue.try_enqueue(
            question=request.query,
            answer=response.answer,
            contexts=eval_payload["contexts"],
            message_id=None,  # message_id assigned after DB commit in chat flow
            generation_time_ms=response.generation_time_ms,
        )
    except Exception:
        pass  # Never block user response
```

**Wiring in `main.py` lifespan** (Phase 3): After creating `LiveEvaluationQueue`, inject it into `RAGService`:

```python
# In lifespan(), after LiveEvaluationQueue is created:
rag_service._live_eval_queue = live_eval_queue
```

**`_get_live_sample_rate()` implementation:** Uses the existing `get_rag_setting()` pattern from `settings_service.py` — a standalone function that creates its own DB session, suitable for use from `RAGService` which may not have a session in the hot path:

```python
def _get_live_sample_rate(self) -> float:
    """Read eval_live_sample_rate from AdminSetting (DB). Default 0.0 (disabled)."""
    from ai_ready_rag.services.settings_service import get_rag_setting
    return float(get_rag_setting("eval_live_sample_rate", default=0.0))
```

### 7.4 Admin Configuration

Live monitoring is controlled via `AdminSetting` keys (runtime-configurable without restart):

| Setting Key | Type | Default | Description |
|------------|------|---------|-------------|
| `eval_live_sample_rate` | float | 0.0 | Fraction of queries to evaluate (0.0 = disabled, 0.1 = 10%) |
| `eval_live_metrics` | str | `"faithfulness,answer_relevancy"` | Comma-separated metrics for live evaluation |

### 7.5 Retention & Privacy

- **Automatic purge**: A cleanup job (similar to `WarmingCleanupService`) deletes `live_evaluation_scores` rows older than `eval_live_retention_days` (default: 30 days). Runs on the same interval as `warming_cleanup_interval_hours`.
- **Access control**: `GET /api/evaluations/live/scores` and `GET /api/evaluations/live/stats` require **admin role**.
- **Future PII redaction**: When `eval_live_redact_pii` is enabled (default: false, Phase 3+), `question` and `generated_answer` are replaced with `"[REDACTED]"` after scores are computed. Metric scores are retained; raw text is discarded. This allows quality trend analysis without retaining user queries.

---

## 8. Configuration Changes

### New Settings (in `config.py`)

| Setting | Type | Default | Purpose |
|---------|------|---------|---------|
| `eval_enabled` | bool | `True` | Master switch for evaluation framework |
| `eval_scan_interval_seconds` | int | 30 | Worker polling interval for pending runs |
| `eval_timeout_seconds` | int | 120 | Per-sample RAGAS evaluation timeout |
| `eval_delay_between_samples_seconds` | float | 1.0 | Throttle between samples (Ollama breathing room) |
| `eval_lease_duration_minutes` | int | 15 | Worker lease duration for run claiming |
| `eval_lease_renewal_seconds` | int | 60 | Lease renewal interval. **Invariant: must be < lease_duration** |
| `eval_max_retries_per_sample` | int | 1 | Max retry attempts on transient error per sample |
| `eval_retry_backoff_seconds` | int | 10 | Backoff delay between retries |
| `eval_max_samples_per_run` | int | 500 | Max samples allowed in a single run. Returns 400 if exceeded. |
| `eval_max_run_duration_hours` | float | 8.0 | Auto-cancel run if exceeded. NULL = unlimited. |
| `eval_live_max_concurrent` | int | 2 | Max concurrent live evaluation tasks |
| `eval_live_queue_size` | int | 10 | Bounded queue capacity; drops if full |
| `eval_live_retention_days` | int | 30 | Auto-purge live scores older than this |

### New Dependencies

Add to `requirements-wsl.txt` and `requirements-spark.txt`:

```
ragas>=0.4.0,<0.5.0
openai>=1.0.0,<2.0.0
```

**Pin rationale**: Upper bounds prevent breaking changes from RAGAS API redesigns (v0.5) or OpenAI client major versions (v2). RAGAS v0.4 uses `llm_factory()` with OpenAI-compatible endpoints — the `openai` package is used only as an HTTP client to Ollama's `/v1` endpoint, NOT for external API calls.

**Note**: The `openai` Python package is a lightweight HTTP client. It connects to `localhost:11434/v1` (Ollama) — no external network calls, fully air-gap compliant. The HuggingFace `datasets` library is a transitive dependency of RAGAS and is also used for RAGBench import in Phase 2.

### 8.1 Observability Requirements

Evaluation must emit structured log events at `INFO`/`WARNING` level for operational visibility. No external metrics infrastructure (Prometheus, Grafana) is required — logs are the primary observability channel for the single-node deployment.

**Required log events (structured, JSON-parseable):**

| Event | Level | Fields | When |
|-------|-------|--------|------|
| `eval.run.started` | INFO | `run_id`, `dataset_id`, `total_samples`, `worker_id` | Worker claims run |
| `eval.run.completed` | INFO | `run_id`, `status`, `completed_samples`, `failed_samples`, `duration_seconds` | Run finishes |
| `eval.sample.scored` | INFO | `run_id`, `sample_id`, `faithfulness`, `answer_relevancy`, `duration_ms` | Sample scored |
| `eval.sample.failed` | WARNING | `run_id`, `sample_id`, `error_type`, `error_message`, `retry_count` | Sample fails |
| `eval.sample.retried` | WARNING | `run_id`, `sample_id`, `attempt`, `error_type`, `backoff_seconds` | Transient retry |
| `eval.lease.renewed` | DEBUG | `run_id`, `worker_id`, `new_expiry` | Heartbeat renewal |
| `eval.lease.stale_recovered` | WARNING | `run_id`, `old_worker_id` | Stale recovery on startup |
| `eval.live.enqueued` | DEBUG | `queue_depth` | Live sample accepted |
| `eval.live.dropped` | WARNING | `queue_depth`, `queue_capacity` | Live sample dropped (full) |
| `eval.live.scored` | INFO | `message_id`, `faithfulness`, `answer_relevancy`, `eval_time_ms` | Live score stored |

**Health endpoint additions** (`GET /api/health`):

```python
# Add to existing health response:
"evaluation": {
    "worker_running": true,
    "current_run_id": "abc123..." | null,
    "live_queue_depth": 3,           # Current items in LiveEvaluationQueue
    "live_queue_capacity": 10,       # Max capacity (eval_live_queue_size)
    "live_drops_since_startup": 0,   # Counter of dropped samples
}
```

---

## 9. API Endpoints

### 9.1 Evaluation Runs (Phase 1)

#### POST `/api/evaluations/runs`

Trigger a new evaluation run.

```python
# Request
{
    "dataset_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "name": "Nightly regression v12",
    "description": "After embedding model update",       # optional
    "tag_scope": ["hr", "finance"],                       # required unless admin_bypass_tags=true
    "admin_bypass_tags": false                            # optional, default false, admin-only
}

# Response (201 Created)
{
    "id": "f9e8d7c6-b5a4-3210-fedc-ba0987654321",
    "name": "Nightly regression v12",
    "dataset_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "status": "pending",
    "total_samples": 50,
    "tag_scope": ["hr", "finance"],
    "admin_bypass_tags": false,
    "model_used": "qwen3:8b",
    "embedding_model_used": "nomic-embed-text",
    "created_at": "2026-02-10T08:30:00Z"
}

# Error: neither tag_scope nor admin_bypass_tags provided
# 422 { "detail": "tag_scope is required when admin_bypass_tags is false" }

# Error: both tag_scope and admin_bypass_tags provided (mutually exclusive)
# 422 { "detail": "tag_scope and admin_bypass_tags are mutually exclusive" }

# Error: admin_bypass_tags true but user is not admin
# 403 { "detail": "admin_bypass_tags requires admin role" }

# Error: dataset sample count exceeds eval_max_samples_per_run
# 400 { "detail": "Dataset has 750 samples, exceeding maximum of 500 per run" }
```

#### GET `/api/evaluations/runs`

List evaluation runs with optional filters.

```python
# Query params: ?status=completed&limit=20&offset=0

# Response
{
    "runs": [...],
    "total": 42,
    "limit": 20,
    "offset": 0
}
```

#### GET `/api/evaluations/runs/{run_id}`

Get a single run with aggregate scores, config snapshot, and ETA.

```python
# Response
{
    "id": "f9e8d7c6-b5a4-3210-fedc-ba0987654321",
    "name": "Nightly regression v12",
    "status": "running",
    "total_samples": 50,
    "completed_samples": 30,
    "failed_samples": 2,
    "tag_scope": ["hr", "finance"],
    "admin_bypass_tags": false,
    "avg_faithfulness": 0.87,
    "avg_answer_relevancy": 0.92,
    "avg_llm_context_precision": 0.81,
    "avg_llm_context_recall": 0.75,
    "model_used": "qwen3:8b",
    "config_snapshot": { ... },
    "eta_seconds": 2400,                              -- Estimated time remaining
    "started_at": "2026-02-10T08:30:05Z",
    "completed_at": null
}
```

**ETA computation**: `(remaining_samples * avg_sample_time_ms) / 1000`. `avg_sample_time_ms` is derived from completed samples in this run. Returns `null` if no samples completed yet.

#### GET `/api/evaluations/runs/{run_id}/samples`

Paginated sample results for a run.

```python
# Query params: ?status=failed&limit=20&offset=0

# Response
{
    "samples": [
        {
            "id": "1a2b3c4d-5e6f-7890-abcd-ef1234567890",
            "sort_order": 0,
            "status": "completed",
            "question": "What is the PTO policy?",
            "ground_truth": "Employees receive 15 days PTO...",
            "generated_answer": "According to the HR handbook...",
            "faithfulness": 0.95,
            "answer_relevancy": 0.88,
            "llm_context_precision": 0.90,
            "llm_context_recall": 0.82,
            "generation_time_ms": 3400,
            "retry_count": 0,
            "processed_at": "2026-02-10T08:31:12Z"
        }
    ],
    "total": 50,
    "limit": 20,
    "offset": 0
}
```

#### DELETE `/api/evaluations/runs/{run_id}`

Delete a run and all its samples (CASCADE). Running runs must be cancelled first (returns 409).

```python
# Response (204 No Content)

# Error: run is still running
# 409 { "detail": "Cannot delete a running evaluation. Cancel it first." }
```

#### POST `/api/evaluations/runs/{run_id}/cancel`

Request cancellation of a running evaluation.

```python
# Response (200 OK)
{
    "id": "f9e8d7c6-b5a4-3210-fedc-ba0987654321",
    "status": "running",                                 -- Still running until worker processes cancel
    "is_cancel_requested": true
}

# Idempotent: calling cancel on already-cancelled or completed run returns 200
# On pending run: transitions directly to cancelled (no worker involved)
```

#### GET `/api/evaluations/summary`

Dashboard summary: latest run, totals, score trends.

```python
# Response
{
    "latest_run": { ... },  # Most recent completed run
    "total_runs": 42,
    "total_datasets": 5,
    "avg_scores": {
        "faithfulness": 0.85,
        "answer_relevancy": 0.89,
        "llm_context_precision": 0.78,
        "llm_context_recall": 0.72
    },
    "score_trend": [  # Last 10 runs
        {"run_id": "...", "completed_at": "...", "avg_faithfulness": 0.82, ...},
        {"run_id": "...", "completed_at": "...", "avg_faithfulness": 0.85, ...}
    ]
}
```

### 9.2 Datasets (Phase 1 — CRUD stubs; Phase 2 — import/generate)

#### GET `/api/evaluations/datasets`

```python
# Response
{
    "datasets": [
        {
            "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
            "name": "RAGBench-TechQA-50",
            "source_type": "ragbench",
            "sample_count": 50,
            "created_at": "2026-02-08T10:00:00Z"
        }
    ]
}
```

#### POST `/api/evaluations/datasets`

Create a manual dataset with inline samples.

```python
# Request
{
    "name": "HR Policy QA Set",
    "description": "Hand-curated Q&A for HR documents",
    "samples": [
        {
            "question": "What is the PTO policy?",
            "ground_truth": "Employees receive 15 days PTO annually.",
            "reference_contexts": ["Section 4.2 of HR Handbook..."]
        }
    ]
}

# Response (201 Created)
{
    "id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
    "name": "HR Policy QA Set",
    "source_type": "manual",
    "sample_count": 1,
    "created_at": "2026-02-10T10:00:00Z"
}
```

#### DELETE `/api/evaluations/datasets/{dataset_id}`

Delete dataset and all its samples (CASCADE). Fails with 409 if any evaluation runs reference this dataset.

#### GET `/api/evaluations/datasets/{dataset_id}/samples`

Paginated list of samples in a dataset.

### 9.3 Dataset Import & Generation (Phase 2)

#### POST `/api/evaluations/datasets/import-ragbench`

Import from pre-downloaded RAGBench files.

**Dataset integrity validation:** Before ingestion, the import endpoint verifies file integrity against a manifest:

```
data/ragbench/
├── manifest.json              # SHA-256 checksums for all subset files
├── techqa/
│   └── dataset.parquet
├── covidqa/
│   └── dataset.parquet
└── ...
```

```python
# manifest.json format:
{
    "version": "1.0",
    "generated_at": "2026-02-10T10:00:00Z",
    "subsets": {
        "techqa": {
            "file": "techqa/dataset.parquet",
            "sha256": "a1b2c3d4e5f6...",
            "sample_count": 500,
            "source_url": "https://huggingface.co/datasets/galileo-ai/ragbench"
        }
    }
}
```

**Validation rules:**
1. `manifest.json` must exist in `data/ragbench/` — 400 if missing
2. Requested subset must exist in manifest — 404 if not found
3. File SHA-256 must match manifest checksum — 422 if mismatch (corrupt/tampered file)
4. Manifest is generated offline during air-gap package preparation (not at runtime)

```python
# Request
{
    "subset": "techqa",         # RAGBench subset name
    "max_samples": 50,          # Cap for manageable evaluation
    "name": "RAGBench-TechQA-50"
}

# Response (201 Created)
{
    "id": "d4e5f6a7-b8c9-0123-def0-234567890123",
    "name": "RAGBench-TechQA-50",
    "source_type": "ragbench",
    "sample_count": 50,
    "source_config": {"subset": "techqa", "max_samples": 50, "sha256": "a1b2c3d4e5f6..."}
}

# Error: checksum mismatch
# 422 { "detail": "Checksum mismatch for techqa/dataset.parquet: expected a1b2c3..., got f6e5d4..." }
```

#### POST `/api/evaluations/datasets/generate-synthetic`

Generate synthetic Q&A using RAGAS TestsetGenerator from user documents.

```python
# Request
{
    "name": "Synthetic-HR-Docs-30",
    "document_ids": ["e5f6a7b8-c9d0-1234-ef01-345678901234", "f6a7b8c9-d0e1-2345-f012-456789012345"],
    "num_samples": 30
}

# Response (202 Accepted — generation runs async)
{
    "id": "a7b8c9d0-e1f2-3456-0123-567890123456",
    "name": "Synthetic-HR-Docs-30",
    "source_type": "synthetic",
    "status": "generating",
    "sample_count": 0  # Updated when generation completes
}
```

### 9.4 Live Monitoring (Phase 3)

#### GET `/api/evaluations/live/stats`

Aggregated live evaluation statistics. **Admin-only.**

```python
# Query params: ?window_hours=24

# Response
{
    "window_hours": 24,
    "sample_count": 47,
    "avg_faithfulness": 0.88,
    "avg_answer_relevancy": 0.91,
    "trend": [  # Hourly buckets
        {"hour": "2026-02-10T08:00:00Z", "count": 5, "avg_faithfulness": 0.85, ...},
        {"hour": "2026-02-10T09:00:00Z", "count": 8, "avg_faithfulness": 0.90, ...}
    ]
}
```

#### GET `/api/evaluations/live/scores`

Paginated list of individual live evaluation scores. **Admin-only.**

```python
# Query params: ?limit=20&offset=0

# Response
{
    "scores": [
        {
            "id": "b8c9d0e1-f2a3-4567-1234-678901234567",
            "question": "Who is the IT director?",
            "faithfulness": 0.92,
            "answer_relevancy": 0.88,
            "generation_time_ms": 2800,
            "eval_time_ms": 45000,
            "created_at": "2026-02-10T09:15:00Z"
        }
    ],
    "total": 47,
    "limit": 20,
    "offset": 0
}
```

---

## 10. Frontend Changes (Phase 4)

### 10.1 Admin Dashboard Integration

Add an **"Evaluations"** tab to `RAGQualityView.tsx` alongside existing tabs (Synonyms, Curated Q&A).

```
┌───────────────────────────────────────────────────────────────────────┐
│  RAG Quality  │ Synonyms │ Curated Q&A │ ▶ Evaluations │            │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐        │
│  │ Faithful.  │ │ Answer Rel │ │ LLM Ctx P. │ │ LLM Ctx R. │        │
│  │   0.87     │ │   0.92     │ │   0.81     │ │   0.75     │        │
│  │  ▲ +0.03   │ │  ▼ -0.01  │ │  ▲ +0.05   │ │  ── 0.00  │        │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘        │
│                                                                       │
│  Evaluation Runs                              [▶ Run Evaluation]      │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │ Name              │ Scope     │ Status    │ F    │ AR  │...│      │
│  │ Nightly v12       │ hr,fin    │ ✓ Done   │ 0.87 │ 0.92│   │      │
│  │ Post-embed update │ [BYPASS]  │ ⚠ Errors │ 0.82 │ 0.89│   │      │
│  │ Baseline v1       │ all-tags  │ ✓ Done   │ 0.79 │ 0.85│   │      │
│  └─────────────────────────────────────────────────────────────┘      │
│                                                                       │
│  ► Datasets (3)                                                       │
│  ► Live Quality (last 24h: avg F=0.88, AR=0.91)                     │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

**Scope column**: Shows `tag_scope` tags or `[BYPASS]` badge for admin bypass runs. This makes access scope visible at a glance.

### 10.2 New React Components

| Component | Purpose |
|-----------|---------|
| `EvaluationManager.tsx` | Main container: summary cards, runs table, dataset accordion |
| `EvaluationRunCard.tsx` | Single run display with status badge, scope, aggregate scores, ETA |
| `EvaluationSampleTable.tsx` | Drill-down table for per-sample results within a run |
| `DatasetManager.tsx` | Dataset CRUD: list, create manual, import RAGBench, generate synthetic |
| `LiveQualityChart.tsx` | Time-series chart for live monitoring (hourly buckets) |
| `MetricScoreCard.tsx` | Reusable metric card with score, trend arrow, and color coding |

### 10.3 New Frontend API Client

`frontend/src/api/evaluations.ts` — follows existing patterns (e.g., `synonyms.ts`):

```typescript
// Evaluation Runs
export async function getEvaluationRuns(params?: { status?: string; limit?: number }): Promise<EvaluationRunListResponse>;
export async function getEvaluationRun(runId: string): Promise<EvaluationRunResponse>;
export async function triggerEvaluationRun(request: TriggerEvaluationRequest): Promise<EvaluationRunResponse>;
export async function cancelEvaluationRun(runId: string): Promise<EvaluationRunResponse>;
export async function deleteEvaluationRun(runId: string): Promise<void>;
export async function getEvaluationSamples(runId: string, params?: PaginationParams): Promise<EvaluationSampleListResponse>;
export async function getEvaluationSummary(): Promise<EvaluationSummaryResponse>;

// Datasets
export async function getDatasets(): Promise<DatasetListResponse>;
export async function createDataset(request: CreateDatasetRequest): Promise<DatasetResponse>;
export async function deleteDataset(datasetId: string): Promise<void>;
export async function importRagbench(request: ImportRagbenchRequest): Promise<DatasetResponse>;
export async function generateSyntheticDataset(request: GenerateSyntheticRequest): Promise<DatasetResponse>;

// Live Monitoring
export async function getLiveStats(windowHours?: number): Promise<LiveStatsResponse>;
export async function getLiveScores(params?: PaginationParams): Promise<LiveScoreListResponse>;
```

---

## 11. Files Modified / Created / Deleted

### Phase 1: Created

| File | Purpose |
|------|---------|
| `ai_ready_rag/db/models/evaluation.py` | `EvaluationRun`, `EvaluationSample`, `EvaluationDataset`, `DatasetSample` models |
| `ai_ready_rag/db/repositories/evaluation.py` | `EvaluationRepository` following `BaseRepository[T]` pattern |
| `ai_ready_rag/services/evaluation_service.py` | `EvaluationService` wrapping RAGAS |
| `ai_ready_rag/workers/evaluation_worker.py` | `EvaluationWorker` with lease-based claiming |
| `ai_ready_rag/schemas/evaluation.py` | Pydantic request/response models |
| `ai_ready_rag/api/evaluations.py` | REST route handlers (including cancel endpoint) |
| `tests/test_evaluation_service.py` | Service unit tests |
| `tests/test_evaluations_api.py` | API integration tests |

### Phase 1: Modified

| File | Changes |
|------|---------|
| `ai_ready_rag/db/models/__init__.py` | Export new evaluation models |
| `ai_ready_rag/main.py` | Add `EvaluationWorker` to lifespan startup/shutdown, mount evaluation router, call `recover_stale_evaluation_runs()` |
| `ai_ready_rag/config.py` | Add `eval_*` settings (Section 8) |
| `ai_ready_rag/services/rag_service.py` | Add `generate_for_eval()` method and `EvalRAGResult` dataclass (Section 5.2) |
| `requirements-wsl.txt` | Add `ragas>=0.4.0,<0.5.0`, `openai>=1.0.0,<2.0.0` |
| `requirements-spark.txt` | Add `ragas>=0.4.0,<0.5.0`, `openai>=1.0.0,<2.0.0` |

### Phase 3: Created

| File | Purpose |
|------|---------|
| `ai_ready_rag/workers/live_eval_queue.py` | `LiveEvaluationQueue` bounded async queue |

### Phase 3: Modified

| File | Changes |
|------|---------|
| `ai_ready_rag/db/models/evaluation.py` | Add `LiveEvaluationScore` model |
| `ai_ready_rag/services/rag_service.py` | Add `live_eval_queue` optional `__init__` param, `_maybe_queue_for_evaluation()` hook, `eval_payload` capture |
| `ai_ready_rag/main.py` | Add `LiveEvaluationQueue` to lifespan startup/shutdown |
| `ai_ready_rag/workers/warming_cleanup.py` | Add live evaluation score purge job |

### Phase 4: Created

| File | Purpose |
|------|---------|
| `frontend/src/components/features/admin/EvaluationManager.tsx` | Main evaluation UI container |
| `frontend/src/components/features/admin/EvaluationRunCard.tsx` | Run display component (with scope badge, ETA) |
| `frontend/src/components/features/admin/EvaluationSampleTable.tsx` | Per-sample drill-down table |
| `frontend/src/components/features/admin/DatasetManager.tsx` | Dataset CRUD UI |
| `frontend/src/components/features/admin/LiveQualityChart.tsx` | Live monitoring chart |
| `frontend/src/components/features/admin/MetricScoreCard.tsx` | Reusable metric card |
| `frontend/src/api/evaluations.ts` | API client functions |

### Phase 4: Modified

| File | Changes |
|------|---------|
| `frontend/src/views/RAGQualityView.tsx` | Add "Evaluations" tab |
| `frontend/src/types/index.ts` | Add evaluation-related TypeScript types |

---

## 12. Implementation Phases

### Phase 1: Core Evaluation Service (Backend) — 5-7 days

1. Create SQLAlchemy models: `EvaluationRun` (with lease + cancel + scope fields), `EvaluationSample` (with retry fields), `EvaluationDataset`, `DatasetSample`
2. Create `EvaluationRepository` with `BaseRepository[T]` pattern
3. Implement `EvalRAGResult` dataclass and `generate_for_eval()` on `RAGService`
4. Implement `EvaluationService` with RAGAS v0.4 wrapper (`llm_factory()` + OpenAI-compatible Ollama endpoint), tag-scoped evaluation, config snapshot capture
5. Implement `EvaluationWorker` with lease-based claiming, heartbeat renewal, stale recovery, cancellation checks, retry policy, max duration enforcement
6. Create Pydantic schemas for all request/response models
7. Implement 10 REST endpoints under `/api/evaluations/*` (including cancel)
8. Wire into `main.py` lifespan (start/stop worker, stale recovery)
9. Add `eval_*` config settings
10. Add `ragas>=0.4.0,<0.5.0` and `openai>=1.0.0,<2.0.0` to requirements files
11. Write unit tests for service + API integration tests (including lease claiming, retry, cancel)

**Risk**: Medium — RAGAS + Ollama integration may require timeout tuning. Lease logic adds complexity.
**Effort**: 5-7 days

### Phase 2: Dataset Management — 3-5 days

1. RAGBench import: load from pre-downloaded HuggingFace files at `data/ragbench/`
2. RAGAS `TestsetGenerator`: generate synthetic Q&A from user-uploaded documents
3. Implement `POST /datasets/import-ragbench` endpoint
4. Implement `POST /datasets/generate-synthetic` endpoint (async generation)
5. Test with actual RAGBench subsets and user documents

**Risk**: Medium — TestsetGenerator quality depends on document content and Ollama model capability.
**Effort**: 3-5 days

### Phase 3: Live Monitoring — 3-4 days

1. Implement `LiveEvaluationQueue` with bounded concurrency and drop policy
2. Add eval payload capture in `RAGService.generate()` (lightweight context extraction)
3. Add `_maybe_queue_for_evaluation()` sampling hook
4. Create `LiveEvaluationScore` model
5. Add `eval_live_sample_rate` and `eval_live_metrics` to AdminSetting
6. Add live score retention purge to cleanup job
7. Implement `GET /evaluations/live/stats` aggregation endpoint (admin-only)
8. Implement `GET /evaluations/live/scores` paginated endpoint (admin-only)
9. Test: verify live evaluation never blocks user response, verify drop policy under load

**Risk**: Low — Bounded queue pattern is simple. Main risk is Ollama load at higher sample rates.
**Effort**: 3-4 days

### Phase 4: Admin Dashboard — 4-6 days

1. Create `EvaluationManager.tsx` main container with summary cards
2. Create `EvaluationRunCard.tsx` and `EvaluationSampleTable.tsx` for run management (including scope badge, ETA, cancel button)
3. Create `DatasetManager.tsx` for dataset CRUD
4. Create `LiveQualityChart.tsx` for time-series monitoring
5. Create `MetricScoreCard.tsx` reusable component
6. Add "Evaluations" tab to `RAGQualityView.tsx`
7. Create `frontend/src/api/evaluations.ts` API client (including `cancelEvaluationRun`)
8. Add TypeScript types to `frontend/src/types/index.ts`

**Risk**: Low — Frontend consumes existing API, no new architectural patterns.
**Effort**: 4-6 days

**Total estimated effort: 15-22 days**

---

## 13. Acceptance Criteria

### Functional — Batch Evaluation

- [ ] Admin can create evaluation datasets (manual Q&A pairs)
- [ ] Admin can trigger evaluation runs against a dataset **with a tag scope**
- [ ] Admin can trigger admin-bypass evaluation (labeled in results, requires admin role)
- [ ] Worker claims runs via atomic lease (no duplicate processing in multi-worker)
- [ ] Worker processes samples sequentially through RAG pipeline + RAGAS
- [ ] RAG pipeline is called with tag_scope from the run (not `user_tags=None` by default)
- [ ] Retrieved contexts are captured via `generate_for_eval()` (not from `RAGResponse`)
- [ ] Per-sample scores (Faithfulness, AnswerRelevancy, LLMContextPrecision, LLMContextRecall) are stored
- [ ] RAGAS receives `reference` string for LLMContextRecall (not `reference_contexts`)
- [ ] `reference_contexts` stored for human review only, not passed to RAGAS
- [ ] Config snapshot captured at run creation with all required fields (Section 3.1)
- [ ] Aggregate scores are computed on run completion
- [ ] Run status transitions correctly: pending → running → completed/completed_with_errors/failed/cancelled
- [ ] Failed samples show error details (message + type + retry_count)
- [ ] Transient errors retry once with 10s backoff before failing
- [ ] Runs can be cancelled via `POST /runs/{id}/cancel` (idempotent)
- [ ] Cancelled runs skip remaining samples, compute partial aggregates
- [ ] Max duration enforcement auto-cancels long-running evaluations
- [ ] Runs can be deleted (cascades to samples; running runs must be cancelled first)
- [ ] Summary endpoint returns latest run, totals, and score trends
- [ ] ETA is computed and exposed on running runs

### Functional — Dataset Management (Phase 2)

- [ ] RAGBench subsets can be imported from pre-downloaded files
- [ ] Synthetic datasets can be generated from user documents via TestsetGenerator
- [ ] Datasets can be listed, viewed, and deleted
- [ ] Deleting a dataset with active runs returns 409

### Functional — Live Monitoring (Phase 3)

- [ ] Sampling hook fires after RAG response via bounded queue (never blocks)
- [ ] Queue drops samples silently when full (backpressure)
- [ ] Max 2 concurrent live evaluations (configurable)
- [ ] DB sessions opened and closed per-evaluation in consumer (no leaks)
- [ ] Sample rate is configurable via AdminSetting (default 0%)
- [ ] Live evaluation scores only Faithfulness + AnswerRelevancy (no ground truth)
- [ ] Stats endpoint returns hourly aggregated scores (admin-only)
- [ ] Scores endpoint returns paginated individual scores (admin-only)
- [ ] Live scores auto-purged after retention TTL (default 30 days)

### Functional — Dashboard (Phase 4)

- [ ] "Evaluations" tab appears in RAG Quality admin view
- [ ] Metric score cards show latest aggregate scores with trend arrows
- [ ] Runs table shows all runs with status badges, scope column, and scores
- [ ] Admin bypass runs labeled with `[BYPASS]` badge
- [ ] Drill-down shows per-sample results for a run
- [ ] Running runs show ETA and cancel button
- [ ] Dataset manager allows CRUD operations
- [ ] Live quality chart shows time-series data

### Non-Functional

- [ ] Evaluation never blocks user-facing RAG responses
- [ ] 120-second timeout per RAGAS evaluation call
- [ ] Sequential sample processing (no Ollama contention)
- [ ] Air-gap compatible: no external API calls, no runtime downloads
- [ ] Config snapshot stored per run for reproducibility (mandatory, explicit schema)
- [ ] All IDs are unprefixed hex UUIDs (consistent with existing codebase)
- [ ] Worker lease prevents duplicate processing across multiple app instances
- [ ] Stale leases recovered on server startup
- [ ] Dependency versions pinned with upper bounds: `ragas>=0.4.0,<0.5.0`, `openai>=1.0.0,<2.0.0`

---

## 14. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| RAGAS + Ollama timeout | Samples fail silently | 120s timeout, 1 retry with 10s backoff on transient errors, fail with error_type + error_message on non-transient |
| Ollama resource contention | RAG responses slow during evaluation | Sequential processing, configurable delay, admin can cancel runs, max duration auto-cancel |
| RAGAS library breaking changes | Build failures | Pin `ragas>=0.4.0,<0.5.0` in all requirements files |
| Large datasets overwhelm Ollama | Worker runs for hours/days | `eval_max_samples_per_run` (default 500) enforced at submission, `eval_max_run_duration_hours` (default 8h) auto-cancels, ETA visibility |
| TestsetGenerator poor quality | Useless synthetic datasets | Validate with manual spot-checks, allow dataset deletion |
| Live sampling Ollama load | User-facing latency | Default 0% rate, bounded queue (max 2 concurrent, drop if full), admin opt-in only |
| Multi-worker duplicate processing | Wasted compute, inconsistent results | Lease-based run claiming with heartbeat renewal, stale recovery on startup |
| Access control bypass in evaluation | Unrealistic scores, sensitive data exposure | Tag-scoped evaluation by default, admin bypass requires explicit opt-in + labeling |
| Live score privacy/compliance | PII retention risk | 30-day auto-purge, admin-only access, future PII redaction setting |
| Non-comparable runs | Misleading regression signals | Mandatory config snapshot with explicit schema; compare only runs with matching snapshots |

---

## 15. Engineering Review Resolutions

This section documents the resolutions to the 12 findings from engineering review of v1.0.

| # | Priority | Finding | Resolution | Sections Updated |
|---|----------|---------|------------|-----------------|
| 1 | P0 | Access control bypass — `user_tags=None` produces unrealistic scores | Added `tag_scope` + `admin_bypass_tags` fields on `evaluation_runs`. Default mode uses tag scope; admin bypass is explicit opt-in and labeled in results. | 3, 5.3, 9.1, 10.1, 13 |
| 2 | P0 | `rag_response.contexts` doesn't exist on `RAGResponse` | Added `EvalRAGResult` dataclass and `generate_for_eval()` method on `RAGService`. Live path uses lightweight `eval_payload` capture. | 5.2, 5.3, 7.1, 11 |
| 3 | P0 | Fire-and-forget lacks guardrails — unbounded tasks, no DB session lifecycle | Replaced with `LiveEvaluationQueue`: bounded `asyncio.Queue`, max 2 consumers, drop policy, DB sessions opened/closed per-evaluation. | 7.2, 7.3, 8, 11, 13 |
| 4 | P1 | No lease/locking — unsafe for multi-process | Added full lease system: atomic `UPDATE WHERE` claiming, `worker_id` + `worker_lease_expires_at`, heartbeat renewal, stale recovery on startup. Mirrors WarmingWorker. | 3, 6.1, 6.2, 6.4, 8 |
| 5 | P1 | Cancellation specified but not implementable | Added `POST /api/evaluations/runs/{run_id}/cancel`, `is_cancel_requested` column, worker cancel check before each sample, skip-remaining + partial aggregate logic. | 3, 4.3, 6.2, 9.1, 10.3, 13 |
| 6 | P1 | Retrieval metrics underspecified — `reference_contexts` not wired | Clarified RAGAS v0.4 input contract: `reference` (string) is what RAGAS uses for LLMContextRecall. `reference_contexts` stored for human review only, explicitly NOT passed to RAGAS. | 2, 3, 5.2, 5.3, 13 |
| 7 | P1 | Config snapshot too vague | Defined explicit `EvalConfigSnapshot` schema (Section 3.1) with 12 required fields including prompt template hash, corpus state, and reranker config. Made `config_snapshot` NOT NULL. | 3, 3.1, 9.1, 13, 14 |
| 8 | P2 | ID format inconsistency (prefixed vs hex) | Standardized: all IDs are unprefixed hex UUIDs via `generate_uuid()`. Added Section 3 ID format convention. Updated all API examples. | 3, 9 (all examples), 13 |
| 9 | P2 | Retry promised but not designed | Added explicit retry taxonomy: `RETRYABLE_EXCEPTIONS` (ConnectionError, TimeoutError), 1 retry with 10s backoff, non-retryable fail immediately. Added `retry_count` + `error_type` to samples. | 3, 4.2, 6.3, 8, 13, 14 |
| 10 | P2 | Dependency pinning inconsistent | Standardized to `ragas>=0.4.0,<0.5.0` and `openai>=1.0.0,<2.0.0` everywhere. Upgraded from v0.2 to v0.4 for simpler Ollama integration via `llm_factory()`. | 8, 12, 13, 14 |
| 11 | P2 | No runtime/capacity planning | Added `eval_max_samples_per_run` (500), `eval_max_run_duration_hours` (8h, auto-cancel), ETA computation on running runs. | 3, 6.2, 8, 9.1, 13, 14 |
| 12 | P2 | Live score privacy/retention gap | Added `eval_live_retention_days` (30d) with auto-purge, admin-only access on live endpoints, future `eval_live_redact_pii` setting. | 3, 7.5, 8, 9.4, 11, 13, 14 |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| v1.0 | 2026-02-10 | Claude + jjob | Initial draft |
| v1.1 | 2026-02-10 | Claude + jjob | Address all 12 engineering review findings (3 P0, 3 P1, 6 P2). Added: tag-scoped access control with admin bypass (Sec 3, 5.3); `EvalRAGResult` + `generate_for_eval()` interface (Sec 5.2); bounded `LiveEvaluationQueue` with drop policy (Sec 7.2); lease-based worker claiming with heartbeat + stale recovery (Sec 6.1-6.4); cancel endpoint + `is_cancel_requested` flow (Sec 4.3, 9.1); RAGAS v0.2 input contract clarification (Sec 2, 5.2); mandatory config snapshot schema (Sec 3.1); standardized hex UUID IDs (Sec 3); retry taxonomy with backoff (Sec 6.3); pinned deps `<0.3.0`/`<3.0.0` (Sec 8); capacity controls: max samples, max duration, ETA (Sec 6.2, 8, 9.1); live score retention + privacy (Sec 7.5). Added Section 15 (review resolution traceability). |
| v1.2 | 2026-02-10 | Claude + jjob | Resolve 4 spec-finalization blockers + 1 ambiguity: (1) ID format — removed `lower(hex(randomblob(16)))` from DDL, added note that SQLAlchemy `default=generate_uuid` is canonical, DDL is illustrative (Sec 3); (2) Stale-lease recovery at runtime — worker poll loop now queries `pending OR (running AND expired lease)` each cycle, not only on startup (Sec 6.2); (3) Missing `updated_at` on `evaluation_samples` — added column (Sec 3); (4) Async contract — added `await` to all `compute_aggregates()` calls, made `_cancel_run()` async (Sec 6.2); (5) `tag_scope`/`admin_bypass_tags` mutual exclusivity — defined explicit 4-case validation matrix, added 422 for "both set" case (Sec 5.1, 9.1). |
| v1.3 | 2026-02-10 | Claude + jjob | **P0 fixes:** (1) **RAGAS v0.2 → v0.4 upgrade**: Replaced `LangchainLLMWrapper` with `llm_factory()` + OpenAI-compatible Ollama endpoint; updated metric classes (`ContextPrecision` → `LLMContextPrecision`, `ContextRecall` → `LLMContextRecall`); updated field names (`question` → `user_input`, `answer` → `response`, `contexts` → `retrieved_contexts`, `ground_truth` → `reference`); rewrote `_get_ragas_llm()`, `_get_ragas_embeddings()`, `evaluate_single()` for v0.4 API; changed deps from `ragas>=0.2.0,<0.3.0` + `datasets` to `ragas>=0.4.0,<0.5.0` + `openai>=1.0.0,<2.0.0` (Sec 2, 5.1, 5.3, 8, 11, 13, 14, 15). (2) Corrected `RAGQualityView.tsx` path from `components/features/admin/` to `views/` (Sec 11). (3) Changed `generation_time_ms` DDL from `INTEGER` to `REAL` and type hints from `int` to `float` to match codebase (Sec 3, 5.1, 7.2). (4) Clarified `generate_for_eval()` skips curated Q&A/cache/direct routing; implementation via `_run_rag_pipeline()` shared helper (Sec 5.2). **P1 fixes:** (5) Renamed DDL columns and API fields from `context_precision`/`context_recall` to `llm_context_precision`/`llm_context_recall` to match RAGAS v0.4 metric class output names (Sec 3, 5.3, 9.1, 9.4). (6) `LiveEvaluationQueue` injected via `RAGService.__init__()` parameter instead of `app.state` access (Sec 7.3, 11). (7) Defined `_get_live_sample_rate()` using existing `get_rag_setting()` pattern (Sec 7.3). (8) Wrapped `ragas.evaluate()` in `asyncio.to_thread()` to prevent event loop blocking (Sec 5.1). (9) Clarified `eval_payload` capture applies only to normal return path, not 4 early-return paths (Sec 7.1). **P2 fixes:** (10) Config snapshot field renames for industry alignment: `rag_temperature` → `temperature`, `chunk_strategy` → `chunking_strategy`, `eval_ragas_timeout_seconds` → `eval_timeout_seconds` (Sec 3.1, 5.1, 8). (11) Added note that `live_evaluation_scores.message_id` is always NULL in Phase 3, retained for future backfill (Sec 3). (12) Added note that `evaluation_datasets.sample_count` is denormalized and must be updated on `dataset_samples` insert/delete (Sec 3). (13) Updated UI wireframe metric labels to reflect RAGAS v0.4 names: "LLM Ctx P." / "LLM Ctx R." (Sec 10.1). |
| v1.4 | 2026-02-11 | Claude + jjob | **Engineering team review — 4 accepted findings (of 12):** (1) **Idempotency contract** (Sec 4.4): Defined replay-safe behavior for sample processing, run claiming, live scores, and aggregate computation; documented that DB row = idempotency key, `UPDATE WHERE status = <expected>` prevents double-processing. (2) **Distributed state transition invariants** (Sec 4.2.1): Added explicit state transition tables for runs and samples with atomic SQL `WHERE` guards; documented 4 race resolution rules (cancel vs processing, cancel vs retry, lease expiry vs active, double-claim prevention). (3) **Observability requirements** (Sec 8.1): Added 10 structured log events (`eval.run.started`, `eval.sample.scored`, `eval.live.dropped`, etc.) with required fields; added health endpoint evaluation status block. (4) **Dataset integrity checksums** (Sec 9.3): Added `manifest.json` with SHA-256 checksums for RAGBench files; validation on import with 400/404/422 error responses. Also added multi-instance scaling note on `LiveEvaluationQueue` (Sec 7.2). |
