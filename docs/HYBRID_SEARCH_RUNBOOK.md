# Hybrid Search Deployment Runbook

## Overview

Hybrid search combines dense vector search (semantic similarity via embeddings) with sparse BM25 keyword search (exact term matching). Results from both methods are fused using Reciprocal Rank Fusion (RRF). This improves retrieval quality for queries that include specific identifiers, codes, names, or technical terms that pure semantic search may miss.

**Architecture**: Qdrant named vectors (`dense` + `text` sparse) per point, with RRF fusion at query time via Qdrant's prefetch mechanism.

---

## Pre-Migration Checklist

Before starting migration, verify all prerequisites:

| Requirement | Check | Command |
|---|---|---|
| Qdrant >= 1.13 | Sparse vector + named vector support | `curl http://localhost:6333/healthz` |
| fastembed installed | Sparse embedding generation | `pip show fastembed` |
| Disk space | ~2x current collection size during migration | `df -h` |
| Qdrant backup | Snapshot source collection | Qdrant dashboard or API |
| Maintenance window | Estimate: ~1 min per 1000 points | Count points in source collection |
| Backend stopped | No writes during migration | Stop the FastAPI server |

---

## Migration Steps

### Step 1: Run Migration

```bash
# Basic migration (source collection -> source_hybrid)
python -m ai_ready_rag.cli.migrate_hybrid \
  --source-collection documents

# With custom target and options
python -m ai_ready_rag.cli.migrate_hybrid \
  --source-collection documents \
  --target-collection documents_hybrid \
  --batch-size 100 \
  --embedding-dimension 768 \
  --verbose
```

The migration:
- Creates target collection with named vectors (`dense` for embeddings, `text` for BM25 sparse)
- Copies all points from source, converting unnamed vectors to named `dense` vectors
- Generates sparse BM25 vectors from each point's text payload
- Supports resume via `.migrate_cursor` file if interrupted

### Step 2: Verify Migration

```bash
python -m ai_ready_rag.cli.migrate_hybrid \
  --source-collection documents \
  --verify
```

This runs 5 verification gates:

| Gate | Check | Pass Criteria |
|------|-------|--------------|
| G1: Point Count | Target has same number of points as source | Exact match |
| G2: Payload Integrity | Random sample of payloads match between collections | 100% match |
| G3: Search Parity | Same queries return similar results on both collections | Top-k overlap >= 80% |
| G4: Latency | Target collection query latency is acceptable | < 2x source latency |
| G5: Sparse Presence | Target points have sparse vectors | 100% of sampled points |

All 5 gates must pass before proceeding to cutover.

### Step 3: Cutover

```bash
python -m ai_ready_rag.cli.migrate_hybrid \
  --source-collection documents \
  --cutover
```

Cutover performs:
1. Re-runs all 5 verification gates
2. Updates the `qdrant_collection_name` setting in the database to point to the target collection
3. Refreshes the vector service capability cache

After cutover, restart the FastAPI server to pick up the new collection.

### Step 4: Enable Hybrid Search

After cutover, enable hybrid search via:

**Admin UI**: Settings > Retrieval Settings > Enable Hybrid Search toggle

**API**:
```bash
curl -X PUT http://localhost:8502/api/admin/settings/retrieval \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"retrieval_hybrid_enabled": true}'
```

---

## Rollback Procedure

If issues are detected after cutover:

### Step 1: Disable Hybrid Search

```bash
curl -X PUT http://localhost:8502/api/admin/settings/retrieval \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"retrieval_hybrid_enabled": false}'
```

### Step 2: Revert Collection

```bash
python -m ai_ready_rag.cli.migrate_hybrid \
  --source-collection documents \
  --rollback
```

This reverts the database setting to point back to the source collection.

### Step 3: Restart Server

Restart FastAPI to pick up the reverted collection setting.

### When to Rollback

- Query latency increased significantly (>3x baseline)
- Search quality degradation reported by users
- Qdrant errors in logs related to sparse vectors
- Verification gates failed during cutover

---

## Cleanup

Once you are confident the hybrid collection is working correctly (recommended: wait at least 48 hours after cutover):

```bash
python -m ai_ready_rag.cli.migrate_hybrid \
  --source-collection documents \
  --cleanup
```

This permanently deletes the source collection. This is irreversible -- ensure rollback is no longer needed before running cleanup.

Also remove the cursor file:
```bash
rm -f .migrate_cursor
```

---

## Score Threshold Tuning Guide

### Understanding Score Types

| Mode | Score Type | Range | Default | Description |
|------|-----------|-------|---------|-------------|
| Dense Only | Cosine similarity | 0.0 - 1.0 | 0.30 | Direct vector similarity; 1.0 = identical |
| Hybrid | Normalized RRF | 0.0 - 1.0 | 0.05 | Fused rank score; naturally lower than cosine |

### Why Hybrid Scores Are Lower

RRF scores are computed as `1 / (k + rank)` summed across result sets, then normalized. Even highly relevant results rarely exceed 0.3 in RRF space. This is normal -- do not set hybrid thresholds as high as dense thresholds.

### Recommended Settings

| Scenario | Dense Threshold | Hybrid Threshold | Prefetch Multiplier |
|----------|----------------|-------------------|---------------------|
| General knowledge base | 0.30 | 0.05 | 3 |
| Technical docs (codes, IDs) | 0.25 | 0.03 | 4 |
| High-precision required | 0.45 | 0.10 | 3 |
| Large corpus (>10k docs) | 0.35 | 0.05 | 4 |
| Small corpus (<100 docs) | 0.20 | 0.02 | 3 |

### When to Use Hybrid vs Dense-Only

**Use Hybrid when:**
- Users search for specific identifiers, policy numbers, employee IDs
- Documents contain technical jargon or acronyms
- Exact keyword matching matters alongside semantic understanding
- You have a hybrid-enabled collection (migrated)

**Stay with Dense-Only when:**
- Queries are primarily natural language questions
- Collection has not been migrated to hybrid format
- Latency is critical and you need fastest possible response

### Tuning Process

1. **Start with defaults**: Dense 0.30, Hybrid 0.05, Prefetch 3
2. **Monitor for 24 hours**: Check confidence scores in chat responses
3. **If too many low-confidence results**: Lower the active threshold slightly
4. **If too many irrelevant results**: Raise the active threshold slightly
5. **Adjust prefetch**: Increase to 4-5 if result quality is inconsistent; decrease to 2 if latency is a concern

### Common Pitfalls

- **Setting hybrid threshold too high** (>0.15): Most results will be filtered out since RRF scores are naturally low
- **Setting dense threshold too low** (<0.15): Irrelevant chunks may be included in context
- **Forgetting to enable hybrid after migration**: The collection supports hybrid, but the search mode defaults to dense-only
- **Not restarting after cutover**: The server caches collection capabilities at startup

---

## Post-Cutover Monitoring (24-Hour Window)

### First Hour
- Verify search returns results (test 5-10 representative queries)
- Check FastAPI logs for Qdrant errors
- Compare response latency to baseline

### First 24 Hours
- Monitor confidence score distribution in chat responses
- Check for user-reported quality issues
- Review admin settings page to confirm hybrid mode badge shows correctly
- Verify query expansion still works with hybrid mode

### Metrics to Watch

| Metric | Healthy | Warning | Action |
|--------|---------|---------|--------|
| Query latency | < 2x baseline | 2-3x baseline | Check Qdrant resources |
| Avg confidence | > 40% | < 30% | Adjust thresholds |
| Zero-result queries | < 5% | > 15% | Lower threshold |
| Error rate | 0% | > 1% | Check logs, consider rollback |

---

## Performance Benchmarks

Use the evaluation framework to measure hybrid search impact:

### Running Benchmarks

1. Create or select an evaluation dataset (Admin > Evaluations > Datasets)
2. Run evaluation with dense-only mode (baseline)
3. Enable hybrid search
4. Run same evaluation with hybrid mode
5. Compare results

### Benchmark Template

Record results in this format:

| Mode | Threshold | Avg Faithfulness | Avg Answer Relevancy | Avg Latency (ms) | Notes |
|------|-----------|------------------|----------------------|-------------------|-------|
| Dense | 0.30 | -- | -- | -- | Baseline |
| Hybrid | 0.05 | -- | -- | -- | After migration |
| Hybrid | 0.03 | -- | -- | -- | Lowered threshold |

### Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Faithfulness | >= baseline | Hybrid should not degrade factual accuracy |
| Answer Relevancy | >= baseline + 5% | Expect improvement from keyword matching |
| Latency | < 2x baseline | Prefetch adds overhead but should be bounded |
| Keyword query recall | >= baseline + 15% | Primary benefit of hybrid search |

---

## Troubleshooting

### Migration fails midway
The CLI supports resume. Simply re-run the same migration command. It reads the `.migrate_cursor` file and continues from the last batch.

### "Collection not found" after cutover
The database setting was updated but the server has not been restarted. Restart the FastAPI server.

### Sparse vectors missing on some points
Re-run migration for the affected batch. If widespread, delete the target collection and re-run the full migration.

### Search returns no results after enabling hybrid
Check that the hybrid score threshold is not too high. Start with 0.05 or lower. RRF scores are naturally much lower than cosine similarity scores.

### High latency after enabling hybrid
- Reduce prefetch multiplier from 5 to 3
- Check Qdrant resource usage (CPU, memory)
- Ensure Qdrant has enough RAM for the sparse index

### Confidence scores dropped after enabling hybrid
This may indicate the hybrid threshold is too low (letting in irrelevant results) or too high (filtering out relevant ones). Adjust `retrieval_min_score_hybrid` incrementally.
