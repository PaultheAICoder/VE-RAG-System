# HYBRID_SEARCH_v1 — Qdrant Hybrid Retrieval Specification

**Version:** 1.1
**Date:** 2026-02-16
**Status:** Draft
**Depends on:** VectorService, Qdrant 1.16+, Ollama embedding pipeline
**Type:** Enhancement
**Changelog:** v1.1 — Address engineering review: score normalization contract, unified config precedence, degraded indexing with backfill, blue/green cutover runbook, per-query sparse fallback, dual thresholds, capability detection, fastembed resilience, filter parity tests, migration idempotency

---

## 1. Problem Statement

The current retrieval pipeline uses **dense semantic search only** (nomic-embed-text, 768-dim, cosine similarity). This works well for natural language queries but fails for:

- **Exact keyword matches**: Searching for "ACORD 25" or "D&O" relies on semantic proximity, which may rank conceptually similar but wrong documents higher.
- **Acronyms and codes**: Industry-specific terms (policy numbers, form codes, carrier abbreviations) have weak semantic embeddings.
- **Proper nouns**: Names like "Bethany Terrace" or "Continental Casualty" may not embed distinctly enough to reliably surface.
- **Short queries**: Single-word or two-word queries produce low-confidence embeddings with poor recall.

**Goal:** Add sparse (BM25) vectors alongside dense vectors in Qdrant to enable hybrid keyword+semantic retrieval with configurable fusion scoring, without changing the RAG service interface.

---

## 2. Solution: Qdrant Native Hybrid Search

### 2.1 Architecture

```
Query
  ├─ Dense embedding (Ollama nomic-embed-text, 768-dim)
  ├─ Sparse embedding (fastembed BM25, variable-dim)
  │   └─ On failure: skip sparse, fallback to dense-only (degraded)
  ▼
Qdrant query_points() with prefetch + fusion
  ├─ Dense search → top-N candidates (with access filter)
  ├─ Sparse search → top-N candidates (with same access filter)
  ├─ Reciprocal Rank Fusion (RRF)
  ▼
Raw RRF scores (rank-based, NOT cosine-calibrated)
  ▼
Score normalization (min-max → 0.0–1.0)
  ▼
Normalized SearchResult[] → RAG service (unchanged)
```

### 2.2 Why Qdrant Native

Qdrant 1.16+ supports named vectors (dense + sparse in same collection), prefetch queries, and server-side fusion — all in a single API call. No external search engine, no result merging in Python, no second round-trip.

---

## 3. Current State (What Changes)

### 3.1 Files Affected

| File | Change | Impact |
|------|--------|--------|
| `ai_ready_rag/services/vector_service.py` | Collection config, indexing, search | **Primary** — all changes here |
| `ai_ready_rag/config.py` | New settings for hybrid search | Config additions |
| `ai_ready_rag/services/protocols.py` | No change | Protocol compatible (search signature unchanged) |
| `ai_ready_rag/services/rag_service.py` | No change | Consumes `SearchResult.score` as before |
| `ai_ready_rag/api/health.py` | Report hybrid search status | Minor addition |

### 3.2 Existing Code (Unchanged)

- `SearchResult` dataclass — `.score` field holds **normalized** score (0-1 range preserved via normalization layer)
- `VectorServiceProtocol.search()` — same signature, same return type
- `RAGService.get_quality_context()` — no changes needed
- `_build_access_filter()` — same Qdrant filter, applied identically to both dense and sparse prefetch queries
- All downstream consumers (confidence scoring, citation extraction, routing)

---

## 4. Technical Specification

### 4.1 Collection Configuration

**Current:**
```python
await self._qdrant.create_collection(
    collection_name=self.collection_name,
    vectors_config=models.VectorParams(
        size=self.embedding_dimension,
        distance=models.Distance.COSINE,
    ),
)
```

**New:**
```python
await self._qdrant.create_collection(
    collection_name=self.collection_name,
    vectors_config={
        "dense": models.VectorParams(
            size=self.embedding_dimension,          # 768
            distance=models.Distance.COSINE,
        ),
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF,           # BM25-style IDF weighting
        ),
    },
)
```

### 4.2 Sparse Vector Generation

Use `fastembed` (Qdrant's recommended library) for BM25 sparse encoding:

```python
from fastembed import SparseTextEmbedding

class VectorService:
    def __init__(self, ...):
        ...
        self._sparse_model: SparseTextEmbedding | None = None

    def _get_sparse_model(self) -> SparseTextEmbedding:
        """Lazy-load sparse model (first use)."""
        if self._sparse_model is None:
            self._sparse_model = SparseTextEmbedding(
                model_name="Qdrant/bm25",           # BM25 tokenizer
            )
        return self._sparse_model

    def sparse_embed(self, text: str) -> models.SparseVector:
        """Generate sparse BM25 vector for text."""
        model = self._get_sparse_model()
        result = list(model.embed([text]))[0]
        return models.SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist(),
        )

    def sparse_embed_batch(self, texts: list[str]) -> list[models.SparseVector]:
        """Generate sparse vectors for multiple texts."""
        model = self._get_sparse_model()
        results = list(model.embed(texts))
        return [
            models.SparseVector(
                indices=r.indices.tolist(),
                values=r.values.tolist(),
            )
            for r in results
        ]
```

**Dependency:** `pip install fastembed` (lightweight, no GPU required, runs CPU-only).

The `Qdrant/bm25` model is a subword tokenizer that produces sparse vectors compatible with Qdrant's sparse index. It downloads once (~5MB) and runs locally.

### 4.3 Indexing (add_document)

**Current** (line 442-448 of vector_service.py):
```python
points.append(
    models.PointStruct(
        id=chunk_id,
        vector=embedding,          # Single dense vector
        payload=payload,
    )
)
```

**New:**
```python
points.append(
    models.PointStruct(
        id=chunk_id,
        vector={
            "dense": embedding,                 # 768-dim dense
            "sparse": sparse_vectors[i],        # BM25 sparse
        },
        payload=payload,
    )
)
```

The `sparse_embed_batch()` call is added alongside the existing `embed_batch()`:

```python
# Generate embeddings (existing)
embeddings = await self.embed_batch(chunks)

# Generate sparse vectors (new)
sparse_vectors = self.sparse_embed_batch(chunks)
```

### 4.4 Search (Hybrid Query with Prefetch + RRF)

**Current:**
```python
response = await self._qdrant.query_points(
    collection_name=self.collection_name,
    query=query_embedding,
    query_filter=access_filter,
    limit=limit,
    score_threshold=score_threshold,
    with_payload=True,
)
```

**New (hybrid mode):**
```python
from qdrant_client.models import Prefetch, FusionQuery, Fusion

# Generate both embeddings
query_dense = await self.embed(query)
query_sparse = self.sparse_embed(query)

response = await self._qdrant.query_points(
    collection_name=self.collection_name,
    prefetch=[
        Prefetch(
            query=query_dense,
            using="dense",
            limit=prefetch_limit,              # 2x final limit for candidate pool
            filter=access_filter,
        ),
        Prefetch(
            query=query_sparse,
            using="sparse",
            limit=prefetch_limit,
            filter=access_filter,
        ),
    ],
    query=FusionQuery(fusion=Fusion.RRF),      # Reciprocal Rank Fusion
    limit=limit,
    with_payload=True,
)
```

**Reciprocal Rank Fusion (RRF)** merges the two result sets server-side. Each document's fused score is:

```
RRF_score = Σ(1 / (k + rank_i))
```

Where `k=60` (Qdrant default) and `rank_i` is the document's rank in each result set.

**IMPORTANT:** RRF scores are **rank-based fusion scores, NOT cosine similarity probabilities.** Raw RRF values typically range from ~0.008 to ~0.033 and are NOT directly comparable to cosine similarity (0.0–1.0). They must be normalized before passing to downstream consumers.

### 4.5 Score Normalization

Raw RRF scores are normalized to 0.0–1.0 using min-max normalization within the result set:

```python
def _normalize_scores(self, points: list) -> list:
    """Normalize RRF scores to 0.0-1.0 range for downstream compatibility.

    RRF scores are rank-based (typically 0.008-0.033) and NOT cosine-calibrated.
    Min-max normalization preserves relative ordering while producing values
    compatible with existing confidence scoring, routing thresholds, and
    score_threshold filtering.
    """
    if not points:
        return points

    scores = [p.score for p in points]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        # All same rank — assign 1.0 (best possible)
        for p in points:
            p.score = 1.0
        return points

    for p in points:
        p.score = (p.score - min_score) / (max_score - min_score)

    return points
```

**Applied:** After `query_points()` returns, before converting to `SearchResult` objects. Only applied in hybrid mode — dense-only mode returns native cosine scores unchanged.

**Score contract:**

| Mode | Score Source | Score Range | Score Meaning |
|------|------------|-------------|---------------|
| Dense-only | Cosine similarity | 0.0–1.0 | Absolute similarity (calibrated) |
| Hybrid (RRF) | Min-max normalized RRF | 0.0–1.0 | Relative rank within result set (not calibrated) |

The distinction matters for `score_threshold` filtering — see Section 5 for dual threshold configuration.

### 4.6 Per-Query Sparse Fallback

If `sparse_embed()` fails during a hybrid query (timeout, model error, etc.), the search **automatically degrades to dense-only** for that request:

```python
async def search(self, query, user_tags, limit=5, score_threshold=0.0, tenant_id=None):
    query_dense = await self.embed(query)
    degraded = False

    if self.hybrid_enabled and self._collection_has_sparse:
        try:
            query_sparse = self.sparse_embed(query)
        except Exception as e:
            logger.warning(f"Sparse embed failed, falling back to dense-only: {e}")
            metrics.increment("search.sparse_fallback")
            degraded = True
            query_sparse = None

        if query_sparse is not None:
            # Hybrid path
            response = await self._qdrant.query_points(prefetch=[...], ...)
            points = self._normalize_scores(response.points)
        else:
            # Degraded: dense-only fallback
            response = await self._qdrant.query_points(
                query=query_dense, using="dense", ...
            )
            points = response.points  # Native cosine scores, no normalization
    else:
        # Dense-only (config disabled or collection not migrated)
        response = await self._qdrant.query_points(
            query=query_dense, using="dense", ...
        )
        points = response.points

    # Convert to SearchResult (same for all paths)
    results = self._to_search_results(points)
    if degraded:
        logger.info(f"Search completed in degraded mode (dense-only): {len(results)} results")
    return results
```

**Invariant:** A sparse embedding failure **never** causes the user's search request to fail. The user always gets results.

### 4.7 Dense-Only Mode

When hybrid is disabled or the collection lacks sparse vectors, search uses the dense-only path (see Section 4.6). This is the default state before migration and the automatic fallback on sparse errors.

### 4.8 Degraded Indexing

When `add_document()` fails to generate sparse vectors (fastembed error, model not loaded, etc.), the document is indexed with **dense vectors only** and flagged for backfill:

```python
# In add_document():
try:
    sparse_vectors = self.sparse_embed_batch(chunks)
    sparse_indexed = True
except Exception as e:
    logger.error(f"Sparse embedding failed for {document_id}, indexing dense-only: {e}")
    metrics.increment("index.sparse_embed_failed")
    sparse_vectors = [None] * len(chunks)
    sparse_indexed = False

# Each point's payload includes the flag:
payload["sparse_indexed"] = sparse_indexed

# Point vector:
if sparse_indexed:
    vector = {"dense": embedding, "sparse": sparse_vectors[i]}
else:
    vector = {"dense": embedding}  # Sparse vector omitted
```

**Structured error code:** `SPARSE_EMBED_FAILED` — logged with document_id, chunk_count, error message.

**Backfill job:**

```python
async def backfill_sparse_vectors(self, batch_size: int = 100) -> int:
    """Scan for points with sparse_indexed=false and add sparse vectors.

    Called manually via CLI or scheduled job. Idempotent.
    Returns count of points backfilled.
    """
    # Scroll points where sparse_indexed == false
    # For each batch: read chunk_text, generate sparse vector, update point
    # Set sparse_indexed = true after successful update
```

**Health reporting:**

```json
{
  "sparse_coverage": {
    "total_points": 10500,
    "sparse_indexed": 10200,
    "sparse_missing": 300,
    "coverage_pct": 97.1
  }
}
```

**Alert thresholds:**
- `sparse_coverage_pct < 95%` → Warning in health endpoint
- `sparse_coverage_pct < 80%` → Health endpoint reports degraded status

### 4.9 Prefetch Limit Strategy

The `prefetch_limit` controls how many candidates each search path produces before fusion:

```python
prefetch_limit = max(20, min(100, limit * prefetch_multiplier))
```

| Parameter | Default | Configurable | Notes |
|-----------|---------|--------------|-------|
| `prefetch_multiplier` | `3` | Yes (runtime setting) | 3x oversampling for good fusion overlap |
| Floor | `20` | No (hardcoded) | Below this, RRF has too few candidates |
| Ceiling | `100` | No (hardcoded) | Above this, diminishing returns vs latency cost |

**v1 note:** Adaptive prefetch sizing based on filter selectivity or corpus size is deferred to v2. The fixed formula is sufficient for deployments up to ~100K chunks. If real workloads demonstrate under-fetching or over-fetching, the multiplier can be tuned via runtime settings without code changes.

---

## 5. Configuration

### 5.1 Unified Configuration Precedence

There is **one flag** for hybrid search enablement, resolved via this precedence chain:

```
DB runtime setting → Environment variable → Default value
```

| Setting | DB Key | Env Var (seed) | Default | Description |
|---------|--------|----------------|---------|-------------|
| Hybrid enabled | `retrieval_hybrid_enabled` | `HYBRID_SEARCH_ENABLED` | `false` | Master switch |
| Prefetch multiplier | `retrieval_prefetch_multiplier` | `HYBRID_SEARCH_PREFETCH_MULTIPLIER` | `3` | Candidate pool size |
| Dense min score | `retrieval_min_score_dense` | `RETRIEVAL_MIN_SCORE_DENSE` | `0.3` | Threshold for dense-only mode |
| Hybrid min score | `retrieval_min_score_hybrid` | `RETRIEVAL_MIN_SCORE_HYBRID` | `0.05` | Threshold for hybrid mode |

**Resolution rules:**
1. On first boot, env var values seed the DB settings table (if not already present).
2. After first boot, **DB value is the single source of truth**. Env vars are ignored for settings that already exist in DB.
3. Admin changes via `PUT /api/settings/retrieval` update DB immediately — no restart required.
4. The code reads from DB via `get_rag_setting()` (existing pattern), never directly from env.

**No dual flags.** The old `hybrid_search_enabled` in config.py is removed. Only `retrieval_hybrid_enabled` exists.

### 5.2 Dual Score Thresholds

The active threshold depends on the current search mode:

```python
@property
def min_similarity_score(self) -> float:
    """Get the appropriate score threshold based on active search mode."""
    if self.hybrid_enabled and self._collection_has_sparse:
        return get_rag_setting("retrieval_min_score_hybrid", 0.05)
    else:
        return get_rag_setting("retrieval_min_score_dense", 0.3)
```

| Mode | Setting | Default | Rationale |
|------|---------|---------|-----------|
| Dense-only | `retrieval_min_score_dense` | `0.3` | Cosine similarity — well-calibrated |
| Hybrid (RRF) | `retrieval_min_score_hybrid` | `0.05` | Normalized RRF — lower baseline due to rank-based scoring |

**Admin UI:** The retrieval settings panel shows both thresholds with help text explaining the difference. The active threshold is highlighted based on current mode.

**Migration note:** When enabling hybrid for the first time, the system auto-creates `retrieval_min_score_hybrid=0.05` in the DB if it doesn't exist.

### 5.3 Collection Capability Detection

The system detects whether the active collection supports sparse vectors:

```python
class VectorService:
    _collection_has_sparse: bool = False
    _capabilities_checked_at: datetime | None = None

    async def _detect_collection_capabilities(self) -> None:
        """Check collection for sparse vector support. Called at startup and on config change."""
        try:
            info = await self._qdrant.get_collection(self.collection_name)
            self._collection_has_sparse = (
                info.config.params.sparse_vectors is not None
                and "sparse" in info.config.params.sparse_vectors
            )
            self._capabilities_checked_at = datetime.now(UTC)
            logger.info(
                f"Collection capabilities: sparse={self._collection_has_sparse}"
            )
        except Exception as e:
            logger.warning(f"Failed to detect collection capabilities: {e}")
            self._collection_has_sparse = False
```

**Detection policy:**

| Trigger | Action |
|---------|--------|
| Application startup (`initialize()`) | Detect capabilities, cache result |
| Setting change (`retrieval_hybrid_enabled` toggled) | Re-detect capabilities |
| Migration complete (manual trigger) | Re-detect capabilities |
| Health check | Report cached capabilities (no re-check) |

**No per-request detection** — too expensive. Capability state is cached and only refreshed on known state-change events.

### 5.4 Health Check

```json
{
  "vector_service": {
    "hybrid_search": {
      "enabled": true,
      "collection_has_sparse": true,
      "capabilities_checked_at": "2026-02-16T10:00:00Z",
      "sparse_model": "Qdrant/bm25",
      "fusion_method": "rrf",
      "active_threshold": 0.05,
      "active_mode": "hybrid"
    },
    "sparse_coverage": {
      "total_points": 10500,
      "sparse_indexed": 10200,
      "sparse_missing": 300,
      "coverage_pct": 97.1,
      "status": "ok"
    }
  }
}
```

**Status values:**
- `"ok"` — coverage ≥ 95%
- `"degraded"` — coverage 80-95%
- `"critical"` — coverage < 80%

---

## 6. Collection Migration

### 6.1 Migration Strategy: Blue/Green

Existing collections use a single unnamed vector. Qdrant does **not** support in-place migration from unnamed to named vectors. The migration uses a **blue/green approach** — create a new collection alongside the old one, verify, then swap.

| Phase | Collection | Status |
|-------|-----------|--------|
| Pre-migration | `documents` (blue) | Active, dense-only |
| During migration | `documents` (blue) + `documents_v2` (green) | Blue active, green building |
| Post-cutover | `documents_v2` (green) | Active, hybrid-capable |
| Post-verification | `documents_v2` (green) | Active; blue deleted after 24hr window |

### 6.2 Migration Script

```bash
python -m ai_ready_rag.cli.migrate_hybrid \
    --source-collection documents \
    --target-collection documents_v2 \
    --batch-size 100 \
    --verify                          # Run verification gates after migration
```

**Steps:**

1. **Create target** — `documents_v2` with named vectors config (dense + sparse) + payload indexes
2. **Scroll source** — iterate `documents` collection in batches of 100 using offset cursor
3. **Transform** — for each point: preserve original point ID, keep dense vector, generate sparse vector from `chunk_text` payload, copy full payload, set `sparse_indexed=true`
4. **Upsert** — write batch to `documents_v2` (upsert by point ID — idempotent)
5. **Checkpoint** — write cursor position to `.migrate_cursor` after each batch commit
6. **Verify** — run cutover gates (see Section 6.4)
7. **Report** — print migration summary with pass/fail for each gate

### 6.3 Idempotency and Resume

| Concern | Design |
|---------|--------|
| **Upsert key** | Original point ID (deterministic, same across runs) |
| **Batch checkpoint** | `.migrate_cursor` file updated after each batch commit with `{"offset": N, "migrated": M, "timestamp": "..."}` |
| **Resume** | On restart, read cursor and skip already-migrated batches |
| **Partial batch** | If interrupted mid-batch, the partial upsert is safe — next run re-upserts the same point IDs (idempotent) |
| **Replay** | Running migration twice produces identical result (same point IDs, same vectors) |

**Integrity verification:**

After migration completes, sample 100 random points from both collections and verify:
1. `chunk_text` payload SHA256 hash matches
2. Dense vector L2 distance < 1e-6 (floating point tolerance)
3. All payload fields present and equal

### 6.4 Cutover Gates (Hard Requirements)

The migration script's `--verify` flag runs these gates. **All must pass before cutover.**

| Gate | Check | Pass Criteria |
|------|-------|---------------|
| **G1: Point count parity** | Compare `count(source)` vs `count(target)` | Exact match |
| **G2: Payload integrity sample** | Hash `chunk_text` for 100 random points in both collections | 100% match |
| **G3: Search parity** | Run 10 canary queries against both collections (dense-only mode) | Result overlap ≥ 80% (same documents in top-5) |
| **G4: Latency guardrail** | Run 10 queries against target, measure p99 | p99 ≤ 2x source p99 |
| **G5: Sparse vector presence** | Sample 100 points in target, check sparse vector exists | 100% have sparse vectors |

**Gate failure behavior:** Script prints failed gates and exits without modifying config. Operator must investigate and re-run.

### 6.5 Cutover Procedure

After all gates pass:

```bash
# 1. Update collection name in DB settings
python -m ai_ready_rag.cli.migrate_hybrid --cutover \
    --target-collection documents_v2

# 2. Trigger capability re-detection (no restart needed)
curl -X POST http://localhost:8502/api/admin/vector/refresh-capabilities \
    -H "Authorization: Bearer $ADMIN_TOKEN"

# 3. Enable hybrid search
curl -X PUT http://localhost:8502/api/settings/retrieval \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"retrieval_hybrid_enabled": true}'
```

### 6.6 Rollback

If issues are detected post-cutover:

```bash
# Revert to old collection
python -m ai_ready_rag.cli.migrate_hybrid --rollback \
    --source-collection documents

# Disable hybrid search
curl -X PUT http://localhost:8502/api/settings/retrieval \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"retrieval_hybrid_enabled": false}'
```

**Rollback is instant** — only changes the collection_name setting. Old collection is untouched throughout the process.

### 6.7 Post-Cutover Monitoring

| Check | Timing | Action if Failed |
|-------|--------|------------------|
| Search latency p99 | Continuous for 24 hours | Rollback to old collection |
| Search result quality (spot-check) | Manual, within 4 hours | Rollback |
| Error rate (`SPARSE_QUERY_FALLBACK` metric) | Continuous for 24 hours | Investigate sparse model; rollback if > 5% |
| Old collection deletion | After 24 hours of clean operation | `python -m ai_ready_rag.cli.migrate_hybrid --cleanup --source-collection documents` |

### 6.8 Migration Performance

| Metric | Estimate |
|--------|----------|
| Sparse embedding speed | ~500 chunks/sec (CPU, fastembed BM25) |
| Qdrant upsert speed | ~1000 points/sec (batch of 100) |
| 10K chunks | ~30 seconds |
| 100K chunks | ~5 minutes |
| Storage increase | ~10-15% (sparse vectors are compact) |
| Temporary storage during migration | 2x (both collections exist simultaneously) |

### 6.9 New Collection Indexing

After migration, all new `add_document()` calls automatically produce both dense and sparse vectors. Points that fail sparse embedding are flagged with `sparse_indexed=false` and backfilled later (see Section 4.8).

---

## 7. Dependency

### 7.1 New Package

```
fastembed==0.4.1
```

Install: `pip install fastembed==0.4.1`

| Property | Value |
|----------|-------|
| Package size | ~15MB |
| Model (`Qdrant/bm25`) | ~5MB (downloaded on first use) |
| Model cache path | `FASTEMBED_CACHE_PATH` env var, default `~/.cache/fastembed` |
| Runtime | CPU only, no GPU required |
| Cold-start time | ~3-5s (model download + load) |
| Concurrent calls | Safe (model is read-only after load) |

### 7.2 Resilience

| Scenario | Behavior |
|----------|----------|
| Model not yet downloaded | Download on first `sparse_embed()` call; 30s timeout |
| Model download fails (no internet) | Log error, set `_sparse_available=false`, all indexing/search runs dense-only |
| Model cache corrupt/missing | Re-download on next call; if fails, degrade to dense-only |
| Cold-start timeout (>30s) | Graceful fallback — sparse model marked unavailable until next restart |
| Concurrent `sparse_embed()` calls | Safe — model loaded once via lazy init with threading lock |

```python
import threading

class VectorService:
    _sparse_model: SparseTextEmbedding | None = None
    _sparse_available: bool = True
    _sparse_lock = threading.Lock()

    def _get_sparse_model(self) -> SparseTextEmbedding | None:
        """Thread-safe lazy-load of sparse model."""
        if not self._sparse_available:
            return None
        if self._sparse_model is not None:
            return self._sparse_model
        with self._sparse_lock:
            if self._sparse_model is not None:
                return self._sparse_model
            try:
                self._sparse_model = SparseTextEmbedding(
                    model_name="Qdrant/bm25",
                    cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"),
                )
            except Exception as e:
                logger.error(f"Failed to load sparse model: {e}")
                self._sparse_available = False
                return None
        return self._sparse_model
```

### 7.3 Existing Packages (No Changes)

- `qdrant-client==1.16.2` — already supports sparse vectors and prefetch queries
- `httpx` — unchanged (Ollama dense embeddings)
- No changes to `nomic-embed-text` or Ollama setup

---

## 8. Backward Compatibility

| Concern | Resolution |
|---------|-----------|
| Existing collection (unnamed vector) | Migration script converts to named vectors; dense-only fallback until migrated |
| `SearchResult.score` range | RRF scores are min-max normalized to 0-1 before returning (Section 4.5) |
| `SearchResult.score` semantics | Dense mode: absolute cosine similarity. Hybrid mode: relative rank within result set. Both 0-1. |
| `VectorServiceProtocol.search()` | Same signature, same return type |
| `RAGService.get_quality_context()` | No changes — consumes `SearchResult` as before |
| Confidence scoring | Uses normalized `SearchResult.score` average — works in both modes |
| `score_threshold` parameter | Dual thresholds: `retrieval_min_score_dense` (0.3) and `retrieval_min_score_hybrid` (0.05) — see Section 5.2 |
| `update_document_tags()` | Unchanged — payload-only operation, no vector changes |
| `delete_document()` | Unchanged — deletes all vectors for document |

**Key semantic difference:** In dense-only mode, a score of 0.8 means "80% cosine similarity." In hybrid mode, a normalized score of 0.8 means "ranked in the top ~20% of this result set." The RAG service's confidence scoring still works because it uses relative comparisons (average across results), not absolute thresholds. The per-mode `min_score` settings account for this difference.

---

## 9. Testing

### 9.1 Unit Tests

| Test | Assertion |
|------|-----------|
| `test_sparse_embed_produces_valid_vector` | Returns `SparseVector` with non-empty indices and values |
| `test_sparse_embed_batch` | Batch output matches individual embed results |
| `test_sparse_embed_deterministic` | Same input always produces same sparse vector |
| `test_collection_created_with_named_vectors` | Collection has "dense" and "sparse" vector configs |
| `test_add_document_indexes_both_vectors` | Points have both dense and sparse vectors + `sparse_indexed=true` |
| `test_add_document_degraded_sparse_failure` | On sparse failure: dense indexed, `sparse_indexed=false` in payload |
| `test_search_hybrid_returns_results` | Hybrid search returns fused results with normalized scores |
| `test_search_hybrid_scores_normalized_0_1` | All returned scores are in [0.0, 1.0] range |
| `test_search_fallback_dense_only` | Dense-only mode works when hybrid disabled |
| `test_search_sparse_failure_falls_back_to_dense` | Sparse embed exception → dense-only results returned (no error) |
| `test_keyword_query_finds_exact_match` | "ACORD 25" ranks ACORD 25 documents highest |
| `test_semantic_query_finds_conceptual_match` | "insurance coverage" finds policy documents |
| `test_score_normalization_single_result` | Single result gets score 1.0 |
| `test_score_normalization_preserves_ordering` | Normalized scores maintain same relative order as raw scores |
| `test_capability_detection_sparse_collection` | `_collection_has_sparse=True` for hybrid collection |
| `test_capability_detection_dense_only_collection` | `_collection_has_sparse=False` for legacy collection |
| `test_dual_threshold_selection` | Dense mode uses `min_score_dense`, hybrid uses `min_score_hybrid` |

### 9.2 fastembed Resilience Tests

| Test | Assertion |
|------|-----------|
| `test_sparse_model_lazy_load` | Model not loaded until first `sparse_embed()` call |
| `test_sparse_model_thread_safe_init` | Concurrent first calls produce single model instance |
| `test_sparse_model_load_failure_degrades` | If model fails to load, `_sparse_available=False`, no exceptions raised |
| `test_sparse_embed_after_model_failure` | Returns `None`, logs warning, doesn't retry until restart |
| `test_sparse_model_cache_path_configurable` | `FASTEMBED_CACHE_PATH` env var is respected |

### 9.3 Access Filter Parity Tests

| Test | Assertion |
|------|-----------|
| `test_filter_parity_multi_tag_user` | User with tags [hr, finance]: hybrid results are subset of accessible documents |
| `test_filter_parity_single_tag_user` | User with tag [hr]: no documents from other tags in hybrid results |
| `test_filter_parity_admin_no_filter` | Admin user (tags=None): hybrid returns all documents, same as dense-only |
| `test_filter_parity_public_tag` | Documents with "public" tag appear for all users in both dense and hybrid |
| `test_fused_results_only_contain_accessible_docs` | Every document in fused result set has at least one tag matching user's tags |

### 9.4 Integration Tests

| Test | Assertion |
|------|-----------|
| `test_hybrid_improves_keyword_recall` | Exact keyword queries have higher recall with hybrid vs dense-only |
| `test_hybrid_preserves_semantic_quality` | Semantic queries don't regress in quality |
| `test_migration_preserves_point_count` | Point count exact match source vs target |
| `test_migration_preserves_payload_integrity` | 100 random points: chunk_text hash matches |
| `test_migration_preserves_dense_vectors` | 100 random points: dense vector L2 distance < 1e-6 |
| `test_migration_adds_sparse_vectors` | 100 random points: sparse vector exists in target |
| `test_migration_idempotent` | Running migration twice: target unchanged |
| `test_migration_resume_after_interrupt` | Kill mid-migration, resume: final count correct |
| `test_cutover_gates_pass_on_valid_migration` | All 5 gates pass for correctly migrated collection |
| `test_cutover_gates_fail_on_partial_migration` | Gates fail if target has fewer points |
| `test_rag_pipeline_with_hybrid` | End-to-end RAG produces answers with citations |
| `test_backfill_sparse_vectors` | Points with `sparse_indexed=false` get sparse vectors added |

---

## 10. Implementation Plan

### Phase 1: Foundation + fastembed
- [ ] Add `fastembed==0.4.1` dependency
- [ ] Implement `sparse_embed()` and `sparse_embed_batch()` with thread-safe lazy loading
- [ ] Add `FASTEMBED_CACHE_PATH` env var support
- [ ] Implement graceful degradation when sparse model fails to load
- [ ] Unit tests: sparse embedding, thread safety, model failure, cache path
- [ ] fastembed resilience tests (Section 9.2)

### Phase 2: Configuration + Capability Detection
- [ ] Add unified config: `retrieval_hybrid_enabled`, `retrieval_min_score_dense`, `retrieval_min_score_hybrid`, `retrieval_prefetch_multiplier` to DB settings
- [ ] Implement env-var-seeds-DB precedence (Section 5.1)
- [ ] Implement `_detect_collection_capabilities()` with startup + event-driven refresh
- [ ] Add hybrid search status and sparse coverage to health endpoint
- [ ] Unit tests: config precedence, capability detection, dual thresholds

### Phase 3: Collection + Indexing
- [ ] Update `initialize()` to create collection with named dense + sparse vectors
- [ ] Update `add_document()` to index both vectors with `sparse_indexed` payload flag
- [ ] Implement degraded indexing path (dense-only on sparse failure, Section 4.8)
- [ ] Implement `backfill_sparse_vectors()` method
- [ ] Handle existing unnamed-vector collections (detect and fall back to dense-only)
- [ ] Unit tests: dual indexing, degraded indexing, backfill

### Phase 4: Hybrid Search
- [ ] Update `search()` with prefetch + RRF fusion path
- [ ] Implement `_normalize_scores()` min-max normalization (Section 4.5)
- [ ] Implement per-query sparse fallback on embed failure (Section 4.6)
- [ ] Add dense-only fallback for non-migrated collections
- [ ] Unit tests: hybrid search, score normalization, sparse fallback, dual thresholds
- [ ] Access filter parity tests (Section 9.3)

### Phase 5: Migration CLI
- [ ] Write `migrate_hybrid` CLI script with blue/green approach
- [ ] Implement cursor-based resume with `.migrate_cursor` file
- [ ] Implement 5 cutover verification gates (Section 6.4)
- [ ] Implement `--cutover`, `--rollback`, `--cleanup` commands
- [ ] Implement integrity verification (payload hash + vector distance sampling)
- [ ] Integration tests: migration, resume, idempotency, gate pass/fail

### Phase 6: Tuning & Documentation
- [ ] Document score threshold tuning guidance for admins
- [ ] Add help text to admin retrieval settings panel (dual thresholds, mode indicator)
- [ ] Write deployment runbook (pre-migration, migration, cutover, monitoring, rollback)
- [ ] Performance benchmark (see Section 12 for benchmark profile)

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| RRF scores not cosine-calibrated | Confidence/routing thresholds break | Min-max normalization + dual thresholds (`min_score_dense`, `min_score_hybrid`) |
| fastembed model download on first use | Cold start delay (~5s) | Thread-safe lazy load; 30s timeout; graceful degradation to dense-only |
| fastembed model cache corrupt | Sparse embedding fails | Re-download on next call; `_sparse_available` flag prevents repeated failures |
| Migration interruption | Partial target collection | Cursor-based resume; idempotent upsert by point ID; source untouched |
| Migration cutover to bad collection | Search quality regression | 5 hard verification gates; instant rollback; 24hr monitoring window |
| Sparse vectors increase storage | ~10-15% more disk + 2x during migration | Compact BM25 vectors; old collection deleted after 24hr verification |
| fastembed version incompatibility | Import errors | Pinned `fastembed==0.4.1`; CI tests verify import + embed |
| Per-query sparse failure | Search fails | Auto-fallback to dense-only per request; metric tracking; never blocks user |
| Stale capability detection | Wrong search mode used | Startup detection + event-driven refresh on config/migration changes |
| Dense-only queries regress with named vectors | Slight latency increase | Named vector overhead negligible per Qdrant benchmarks; verified in Phase 6 |
| Sparse coverage degrades over time | Hybrid quality drops | Health endpoint reports coverage %; alerts at <95%; backfill job available |

---

## 12. Success Metrics

### 12.1 Standard Benchmark Profile

All performance SLOs are measured against this standardized profile:

| Parameter | Value |
|-----------|-------|
| **Hardware** | Spark server (4-core, 8GB RAM, SSD) |
| **Collection size** | 10,000 chunks (~500 documents) |
| **Concurrency** | 5 concurrent queries |
| **Filter complexity** | 3 user tags (typical account manager) |
| **Qdrant deployment** | Single-node, same host |
| **Embedding model** | nomic-embed-text (Ollama, local) |
| **Sparse model** | Qdrant/bm25 (fastembed, CPU) |

### 12.2 Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Keyword query recall (exact terms) | +30% vs dense-only | 20 keyword queries (policy numbers, form codes, proper nouns) |
| Semantic query quality | No regression (±5%) | Same 20 semantic queries, compare top-5 overlap with dense-only |
| Search latency (p50) | < 100ms | Standard benchmark profile, 100 queries |
| Search latency (p99) | < 300ms | Standard benchmark profile, 100 queries |
| Index time per document (10 chunks) | < 5% increase vs dense-only | 50 documents, measure add_document wall time |
| Migration time (10K chunks) | < 60 seconds | Timed migration run on benchmark hardware |
| Migration gate pass rate | 5/5 gates pass | Automated verification after migration |
| Sparse coverage after migration | 100% | All points have `sparse_indexed=true` |
| Post-cutover error rate | < 1% `SPARSE_QUERY_FALLBACK` events | 24hr monitoring window |
