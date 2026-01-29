# Document Update & Reindex Spec (Draft)

| Field | Value |
|-------|-------|
| **Status** | DRAFT |
| **Version** | 0.1 |
| **Created** | 2026-01-29 |
| **Type** | Backend Service + Ops |
| **Owner** | — |

## Summary
Define how customer document updates are applied to the vector database in an air-gapped deployment. The system must support safe, auditable updates with minimal downtime and clear rollback behavior.

---

## Goals
- Support incremental updates when documents change
- Maintain integrity of citations and access controls
- Provide clear rollback and audit trails
- Minimize downtime for on-prem installations

## Non-Goals
- Real-time streaming updates from customer systems
- Full content governance policies (covered elsewhere)

---

## Update Strategies (Options)

### Option A: Full Reindex (Replace All)
Reprocess all documents and rebuild vectors from scratch.
- **Pros:** Simple, consistent
- **Cons:** Slow, downtime risk
- **Use when:** Small corpora or major model changes

### Option B: Incremental Update (Document-level)
Detect changed docs and reprocess only those; delete and re-insert vectors for that `document_id`.
- **Pros:** Fast, scalable
- **Cons:** Requires robust change detection
- **Use when:** Most customer deployments

### Option C: Patch-Level Update (Chunk-level)
Compute diffs per section and update only affected chunks.
- **Pros:** Minimal compute
- **Cons:** Complex, fragile
- **Use when:** Very large corpora with frequent updates

### Option D: Blue/Green Reindex
Build new collection, switch traffic when complete.
- **Pros:** Zero downtime, easy rollback
- **Cons:** Double storage
- **Use when:** High-availability required

---

## Decision Matrix

| Scenario | Recommended Strategy | Rationale |
|----------|----------------------|-----------|
| Small corpus, infrequent updates | Full Reindex | Simple, low cost |
| Medium/large corpus, regular updates | Incremental | Efficient with good integrity |
| Large corpus, strict uptime | Blue/Green + Incremental | Zero downtime + fast deltas |
| High change frequency within docs | Patch-Level (optional) | Minimizes reprocessing |
| Model upgrade or chunker change | Full Reindex | Ensures consistency |

---

## Recommended Default
**Incremental Update (Option B)** with periodic full rebuilds for model upgrades.

---

## Change Detection

### Content Hash
- Compute SHA-256 of file content at ingest
- Store `content_hash` in Document record
- On update, compare hashes:
  - **Same hash**: skip processing
  - **Different hash**: reprocess + reindex

### Version Tracking
Add fields:
- `document_version` (int)
- `embedding_model_version` (string)
- `chunker_version` (string)

---

## Update Workflow (Incremental)

1. **Upload updated file** (same document_id or mapped by manifest)
2. **Validate** file type, size, tags
3. **Compare content_hash**
4. **If changed:**
   - Mark status `processing`
   - Delete existing vectors for document_id
   - Process + chunk
   - Insert new vectors
   - Update metadata, chunk_count, processed_at
   - Mark status `ready`
5. **If unchanged:**
   - No-op, return status `unchanged`

---

## Offline Update Package (Air-Gapped)

### Update Bundle Layout
```
update_bundle/
├── manifest.json
├── documents/
│   ├── doc-123.pdf
│   └── doc-456.docx
└── signatures/ (optional)
```

### Manifest Format (draft)
```json
{
  "bundle_version": "1.0",
  "generated_at": "2026-01-29T12:00:00Z",
  "documents": [
    {
      "document_id": "doc-123",
      "filename": "doc-123.pdf",
      "content_hash": "sha256:...",
      "document_version": 4,
      "tags": ["hr", "policy"],
      "title": "Employee Handbook"
    }
  ]
}
```

---

## API Endpoints (Draft)

### POST /api/documents/update
- Accepts file + document_id + tags
- Applies incremental update workflow

### POST /api/documents/reindex
- Triggers full reindex (admin only)

### GET /api/documents/{id}/versions
- Returns version history for a document

---

## Rollback Strategy

- If update fails, preserve previous vectors (do not delete until new index succeeds)
- Optionally maintain `document_version` pointer to last good version
- Blue/green (optional) for strict zero-downtime

---

## Operational Playbook (Draft)

### Roles
- **Customer Admin**: Provides update bundle and approves update window
- **System Operator**: Runs update, validates results, initiates rollback if needed

### Pre-Update Checklist
- [ ] Backup vector store / snapshot (if supported)
- [ ] Verify update bundle checksums
- [ ] Confirm available disk space
- [ ] Confirm correct `ENV_PROFILE` and settings

### Update Steps
1. Upload or mount update bundle
2. Run update job (incremental or full)
3. Verify document status = ready
4. Run sampling QA queries
5. Log completion + metrics

### Rollback Steps
1. Revert to last good snapshot or previous collection
2. Mark documents as `ready` with previous version
3. Record rollback in audit log

### Success Criteria
- 0 failed documents after retries
- Expected chunk counts match manifest
- QA queries return correct citations

---

## Acceptance Criteria
- [ ] Incremental updates work with hash detection
- [ ] No changes = no reprocessing
- [ ] Failed updates do not leave partial vectors
- [ ] Version metadata stored and visible
- [ ] Audit log for each update

---

## Change Log
| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-01-29 | Initial draft |
