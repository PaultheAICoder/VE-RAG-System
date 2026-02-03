---
title: Reranker Integration
status: DRAFT
version: 1.0
created: 2026-02-03
author: codex
type: Fullstack
complexity: MEDIUM
---

# Reranker Integration (Configurable in Settings)

## Summary
Add a configurable **local Python reranker** to the RAG pipeline with a settings UI toggle. Support two models:
- `BAAI/bge-reranker-v2-m3`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`

Reranker should be optional and disabled by default for minimal overhead. When enabled, it reorders retrieved chunks before final context assembly.

---

## Goals
- Allow admins to select reranker model in **Settings UI**.
- Run reranker **in-process (Python)** with local model files.
- Improve retrieval precision by reordering top‑k chunks.
- Preserve existing pipeline and allow fallback when reranker unavailable.

## Non‑Goals
- No external API rerankers.
- No distributed reranking service.
- No changes to vector DB or embedding model selection.

---

## User Experience

### Settings UI
Add a new **Reranker** section under Settings:
- Toggle: **Enable Reranker** (on/off)
- Dropdown: **Reranker Model**
  - `BAAI/bge-reranker-v2-m3`
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Info text: “Reranker reorders top‑k chunks for higher precision. May add latency.”

### Health View
- Show reranker model and status (loaded/disabled/error).

---

## Backend Changes

### 1) Config / Settings
Add new settings keys (DB settings + env defaults):
- `reranker_enabled` (bool, default false)
- `reranker_model_name` (str, default `BAAI/bge-reranker-v2-m3`)

### 2) Reranker Service
Create a new module: `ai_ready_rag/services/reranker_service.py`

Responsibilities:
- Lazy‑load reranker model on first use
- Provide `rerank(query, candidates)` returning reordered candidates
- Handle errors and fallback if model not available

**Model loading** (Python in‑process):
- Use `sentence-transformers` CrossEncoder
- Models are local or cached in HF cache

Pseudo‑code:
```python
from sentence_transformers import CrossEncoder

class RerankerService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    def load(self):
        if not self.model:
            self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if not candidates:
            return candidates
        self.load()
        pairs = [(query, c.chunk_text) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c.rerank_score = float(s)
        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
```

### 3) RAG Pipeline Integration
Integrate reranking **after retrieval + dedup + per‑doc limiting** and **before final top‑k selection**.

Current flow (simplified):
```
retrieve -> filter -> dedup -> limit_per_doc -> top_k
```
New flow:
```
retrieve -> filter -> dedup -> limit_per_doc -> rerank (optional) -> top_k
```

### 4) Fallback Behavior
- If reranker enabled but model fails to load, log error and continue without rerank.
- Health endpoint should show reranker error status.

---

## Frontend Changes

### SettingsView
Add form controls for reranker settings:
- Checkbox: Enable reranker
- Dropdown: Model selection

Add API calls to get/update settings:
- `GET /api/admin/settings` (existing or add)
- `PATCH /api/admin/settings` for reranker fields

### HealthView
Display reranker status:
- Enabled/disabled
- Model name
- Load errors

---

## API Additions

If not existing, extend settings endpoints:

- `GET /api/admin/settings`
  - includes `reranker_enabled`, `reranker_model_name`
- `PATCH /api/admin/settings`
  - accepts changes to reranker fields

---

## Dependencies
Add Python dependency:
- `sentence-transformers`

Optional: allow model caching path via env
- `HF_HOME` or `TRANSFORMERS_CACHE`

---

## Acceptance Criteria
- [ ] Reranker can be enabled/disabled in Settings UI
- [ ] Model selection persists and affects runtime behavior
- [ ] Rerank stage runs only when enabled
- [ ] Reranker failures do not crash pipeline (fallback to non‑reranked results)
- [ ] Health view shows reranker status

---

## Rollout Plan
1. Add settings keys + UI controls (disabled by default)
2. Add reranker service and pipeline integration
3. Verify with small dataset; measure latency
4. Enable for internal demo

---

## Risks
- **Latency**: Reranking adds compute time; must monitor response time.
- **Memory**: Large models may increase memory usage.
- **Model availability**: Ensure models are pre‑cached for air‑gapped installs.

---

## Open Questions
- Should reranker apply to **all queries** or only when `top_k > N`?
- Do we want a **threshold** for rerank score or a full reorder?
