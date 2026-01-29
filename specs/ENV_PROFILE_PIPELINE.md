# Environment-Profiled RAG Pipeline Spec

| Field | Value |
|-------|-------|
| **Status** | DRAFT |
| **Version** | 0.2 |
| **Created** | 2026-01-29 |
| **Updated** | 2026-01-28 |
| **Type** | Architecture/Config |
| **Owner** | — |
| **Related Specs** | DOCUMENT_MANAGEMENT.md |

## Summary
Provide a configurable ingestion + retrieval pipeline that automatically selects the correct services for **Laptop** or **Spark** deployments based on `ENV_PROFILE` in `.env`.

## Goals
- Single codebase runs on laptop or Spark without code changes
- Profile-driven selection of vector DB, chunker, OCR, and model sizes
- Fast iteration on laptop; high-fidelity processing on Spark
- Clear, centralized configuration

## Non-Goals
- Auto-detecting hardware at runtime
- Implementing a full deployment system

---

## Configuration

### .env
```
ENV_PROFILE=laptop
```

### Profiles
Two profiles are supported: `laptop`, `spark`.

**Laptop Profile (default dev)**
- **Vector DB:** Chroma (local)
- **Chunking:** Fast splitter (RecursiveCharacterTextSplitter)
- **OCR:** Disabled (unless explicitly enabled)
- **LLM:** Small model (e.g., `llama3.2:latest`)
- **Embeddings:** `nomic-embed-text`
- **Token budgets:** conservative

**Spark Profile (production/scale)**
- **Vector DB:** Qdrant
- **Chunking:** Docling + HybridChunker
- **OCR:** Enabled (Tesseract/EasyOCR)
- **LLM:** Larger model (e.g., `qwen3:8b` or `deepseek-r1:32b`)
- **Embeddings:** `nomic-embed-text` (or higher-quality embedding if available)
- **Token budgets:** larger

---

## Required Settings (config.py)

```
ENV_PROFILE
VECTOR_BACKEND         # chroma | qdrant
CHUNKER_BACKEND        # simple | docling
ENABLE_OCR             # true | false
CHAT_MODEL
EMBEDDING_MODEL
RAG_MAX_CONTEXT_TOKENS
RAG_MAX_HISTORY_TOKENS
RAG_MAX_RESPONSE_TOKENS
```

---

## Profile Map (example)

```
profiles = {
  "laptop": {
    "vector_backend": "chroma",
    "chunker_backend": "simple",
    "enable_ocr": False,
    "chat_model": "llama3.2:latest",
    "embedding_model": "nomic-embed-text",
    "rag_max_context_tokens": 2000,
    "rag_max_history_tokens": 600,
    "rag_max_response_tokens": 512,
  },
  "spark": {
    "vector_backend": "qdrant",
    "chunker_backend": "docling",
    "enable_ocr": True,
    "chat_model": "qwen3:8b",
    "embedding_model": "nomic-embed-text",
    "rag_max_context_tokens": 6000,
    "rag_max_history_tokens": 1500,
    "rag_max_response_tokens": 2048,
  }
}
```

---

## Pipeline Selection

### Ingestion
- **Laptop:** simple text extraction + splitter
- **Spark:** Docling conversion + HybridChunker (OCR on)

### Vector Store
- **Laptop:** Chroma (local persist)
- **Spark:** Qdrant (remote)

### Retrieval/LLM
- **Laptop:** smaller LLM
- **Spark:** larger LLM

---

## Integration Points
- **Settings loader** reads `.env` and applies `ENV_PROFILE`
- **Service factory** returns correct implementations based on profile

### Example Factory API
```
VectorStore = get_vector_backend(settings)
Chunker = get_chunker(settings)
```

---

## Acceptance Criteria
- [ ] Setting `ENV_PROFILE=laptop` uses Chroma + simple chunker
- [ ] Setting `ENV_PROFILE=spark` uses Qdrant + Docling chunker
- [ ] All required settings override defaults cleanly
- [ ] Profile selection is logged on startup

---

## Open Questions
- Should `spark` profile default to GPU-enabled Qdrant indexing?
- Should OCR be enabled for specific file types only?

---

## Implementation Issues

| Issue | Title | Complexity |
|-------|-------|------------|
| #040 | Profile-Aware Settings | SIMPLE |
| #041 | Service Protocols (Abstract Interfaces) | SIMPLE |
| #042 | Vector Service Factory + Chroma Implementation | MODERATE |
| #043 | Chunker Factory + SimpleChunker Implementation | MODERATE |
| #044 | Profile Pipeline Integration | MODERATE |
| #045 | Profile Pipeline Tests | MODERATE |

### Implementation Order

```
#040 Profile Settings
  └── #041 Service Protocols
        ├── #042 Vector Factory + Chroma
        └── #043 Chunker Factory + SimpleChunker
              └── #044 Integration
                    └── #045 Tests
```

---

## Change Log
| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-01-29 | Initial spec |
| 0.2 | 2026-01-28 | Added implementation issues, factory pattern details, relationship to DOCUMENT_MANAGEMENT |
