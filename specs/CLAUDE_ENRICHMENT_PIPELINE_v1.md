---
title: "Claude Enrichment Pipeline"
status: DRAFT
version: v1.0
created: 2026-02-19
author: —
type: Integration
complexity: COMPLEX
---

# Claude Enrichment Pipeline

## Summary

Replace the existing Ollama-based document processing pipeline (summary generation, auto-tagging, document classification) with a Claude API-powered enrichment pipeline that runs during ingestion. Claude processes each document holistically — generating rich summaries, extracting structured entities, classifying document types, and producing auto-tags — creating pre-digested context that enables the local Ollama model to answer queries quickly and accurately at query time.

The system supports two deployment modes: (1) enrichment on a network-connected workstation with database transfer to the air-gapped Spark, and (2) direct enrichment on the Spark with temporary API access during ingestion.

## Problem Statement

### Current State

The existing ingestion pipeline uses local Ollama models (qwen3:8b) for:
- **Summary generation** (`SummaryGenerator`): Samples 9 chunks, generates a document synopsis via Ollama. Quality is limited by the 8B model's comprehension of complex insurance documents.
- **Auto-tagging** (`DocumentClassifier`): Uses Ollama to classify document type and assign tags. Limited by model's domain knowledge.
- **Chunk storage**: Raw Docling text stored in Qdrant with minimal metadata (page_number, section heading only).

**Result**: All 16 demo questions returned confidence 30 and were routed to human. The local model cannot reliably extract specific values (policy numbers, limits, insurer names) from raw chunk text.

### Root Cause

1. **Chunks lack structured context**: Raw text stored without entity extraction. The query-time LLM must interpret complex insurance language from scratch for every query.
2. **Summaries are shallow**: The 8B model produces generic summaries that miss key details (specific limits, insurer names, policy numbers).
3. **No cross-document awareness**: Each document processed in isolation. No relationship metadata for comparative queries.

### Required State

Each chunk in Qdrant should contain:
- **Original text** (for citation fidelity)
- **Claude-generated summary** (plain-English interpretation)
- **Extracted entities** (key-value pairs: insured, insurer, limits, policy numbers, dates)
- **Document-level synopsis** (answers "what is this document about?" queries instantly)
- **Accurate tags** (document type, coverage types, client name)

The query-time Ollama model's job becomes: "Read this pre-interpreted chunk and answer the question" — a trivially easy task for even a small model.

## Goals

1. **Improve answer quality**: Increase demo question confidence from 30 → 70+ for structured queries
2. **Reduce query-time LLM burden**: Pre-digest context so Ollama performs simple read-and-cite tasks
3. **Support air-gap + cloud modes**: Toggle enrichment on/off per deployment
4. **Maintain pipeline compatibility**: Enrichment is an optional stage; existing pipeline works without it
5. **Cost efficiency**: Batch API calls to minimize Claude API spend (~$0.02-0.05 per document)

## Scope

### In Scope

- Claude API integration service (`claude_enrichment_service.py`)
- Per-chunk entity extraction and summary generation
- Per-document synopsis generation
- Document type classification and auto-tag generation
- Enriched chunk format for Qdrant storage
- Configuration settings and feature flag
- Both deployment modes (workstation transfer + direct Spark)
- Offline batch re-enrichment command for existing documents
- Cost tracking and logging

### Out of Scope

- Changes to the query-time RAG pipeline (prompt template, retrieval logic)
- Changes to the Docling/OCR chunking stage (Claude enriches *after* chunking)
- Cross-document relationship extraction (future Phase 2)
- Claude-based query answering at runtime (Ollama stays for queries)
- Frontend UI for enrichment configuration (admin API settings only)
- Anthropic prompt caching for chunk-batch calls (synopsis as cached prefix)

## Technical Specification

### Architecture

```
CURRENT PIPELINE (Ollama-only):
Document → Docling → Chunks → [Ollama Summary] → [Ollama Auto-Tag] → Qdrant

NEW PIPELINE (Claude-enriched):
Document → Docling → Chunks → [Claude Enrichment] → Qdrant
                                    │
                                    ├─ Document Synopsis (1 API call)
                                    ├─ Chunk Enrichment  (batched API calls)
                                    │   ├─ Summary per chunk
                                    │   ├─ Entity extraction per chunk
                                    │   └─ Coverage classification per chunk
                                    ├─ Document Classification + Auto-Tags
                                    └─ Structured Fields → SQLite
```

When `claude_enrichment_enabled = True`, the Claude enrichment service **replaces**:
- `SummaryGenerator` (Ollama summary generation)
- `DocumentClassifier` + auto-tagging conflict resolution
- Manual tag classification
- `FormsProcessingService` (ingestkit-forms ACORD extraction) — Claude handles all structured extraction including ACORD forms, eliminating the forms dependency

When `claude_enrichment_enabled = False`, the existing Ollama + ingestkit-forms pipeline runs unchanged.

### Integration Point

The enrichment service plugs into `ProcessingService.process_document()` **early in the method** (before the ingestkit-forms routing at line 128), short-circuiting the entire existing pipeline:

```python
# NEW: Claude enrichment replaces ALL downstream processing
if self.settings.claude_enrichment_enabled and self._should_use_claude_enrichment():
    enrichment = ClaudeEnrichmentService(self.settings)
    result = await enrichment.process_and_index(
        document=document,
        file_path=file_path,
        db=db,
        chunker=self.chunker,  # Still use Docling for parsing/chunking
    )
    if result is not None:
        return result
    # Fallback: if enrichment failed, continue to existing pipeline below
    logger.info("Claude enrichment failed, falling back to standard pipeline")

# EXISTING PIPELINE (unchanged, runs when enrichment disabled or fails):
# ... ingestkit-forms routing ...
# ... standard chunking ...
# ... Ollama summary generation ...
# ... LLM auto-tagging ...
```

**Key**: Docling still handles document parsing and chunking. Claude enrichment operates on the parsed chunks, replacing summary generation, auto-tagging, AND forms extraction in a single unified pass.

### New Service: `ClaudeEnrichmentService`

**File**: `ai_ready_rag/services/claude_enrichment_service.py`

```python
class ClaudeEnrichmentService:
    """Enriches document chunks using Claude API during ingestion.

    Replaces Ollama-based summary generation and auto-tagging with
    Claude-powered enrichment that produces pre-digested context
    for fast query-time retrieval.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.claude_api_key
        self.model = settings.claude_enrichment_model  # default: claude-sonnet-4-6
        self.base_url = settings.claude_api_base_url

    async def enrich_document(
        self,
        chunks: list[ChunkInfo],
        filename: str,
        document_id: str,
    ) -> tuple[list[EnrichedChunk], ChunkInfo, list[AutoTagResult]]:
        """Full document enrichment pipeline.

        Phase 1: Document synopsis (1 API call, all chunks batched)
        Phase 2: Per-chunk entity extraction (batched API calls)
        Phase 3: Auto-tag generation (derived from Phase 1 + 2 results)

        Returns:
            Tuple of (enriched_chunks, synopsis_chunk, auto_tags)
        """

    async def _generate_synopsis(
        self, chunks: list[ChunkInfo], filename: str
    ) -> DocumentSynopsis:
        """Generate document-level synopsis from all chunks.

        Single API call. Sends full document text (or sampled chunks
        if over token budget) to Claude for holistic interpretation.
        """

    async def _enrich_chunks(
        self, chunks: list[ChunkInfo], synopsis: DocumentSynopsis
    ) -> list[EnrichedChunk]:
        """Extract entities and generate summaries per chunk.

        Batches chunks into groups of N (default 5) per API call
        to balance cost and quality. Each batch includes the document
        synopsis as context for consistent entity extraction.
        """

    async def _generate_auto_tags(
        self, synopsis: DocumentSynopsis, enriched_chunks: list[EnrichedChunk]
    ) -> list[AutoTagResult]:
        """Derive auto-tags from enrichment results.

        No additional API call needed — tags are extracted from
        synopsis metadata (document_type, client_name, coverage_types).
        """
```

### Data Models

```python
@dataclass
class DocumentSynopsis:
    """Document-level summary produced by Claude."""
    text: str                          # 100-300 word summary
    document_type: str                 # policy, certificate, loss_run, etc.
    client_name: str                   # Named insured / client
    insurers: list[str]                # All insurer names found
    coverage_types: list[str]          # GL, WC, D&O, umbrella, etc.
    policy_period: str | None          # e.g., "01/01/2025-01/01/2026"
    total_premium: str | None          # e.g., "$45,230"
    key_limits: dict[str, str]         # {"gl_per_occurrence": "$1,000,000", ...}
    key_facts: list[str]              # Bullet points of important details


@dataclass
class EnrichedChunk:
    """A chunk with Claude-generated enrichment."""
    original_text: str                 # Raw Docling text (for citation)
    summary: str                       # Plain-English interpretation (1-3 sentences)
    entities: dict[str, str | list]    # Extracted key-value pairs
    coverage_type: str | None          # If chunk discusses specific coverage
    chunk_index: int
    page_number: int | None
    section: str | None

    @property
    def enriched_text(self) -> str:
        """Combined text stored in Qdrant for embedding + retrieval."""
        parts = []
        if self.summary:
            parts.append(f"[SUMMARY] {self.summary}")
        if self.entities:
            entity_str = "; ".join(f"{k}={v}" for k, v in self.entities.items())
            parts.append(f"[ENTITIES] {entity_str}")
        parts.append(f"[ORIGINAL] {self.original_text}")
        return "\n".join(parts)


@dataclass
class AutoTagResult:
    """Auto-tag derived from Claude enrichment."""
    tag_name: str                      # e.g., "client:bethany-terrace"
    display_name: str                  # e.g., "Bethany Terrace"
    namespace: str                     # e.g., "client", "doctype", "coverage"
    confidence: float                  # Always 1.0 (Claude extraction, not classification)
    source: str                        # "claude_enrichment"
```

### Claude API Call Design

#### Call 1: Document Synopsis (per document)

**Input**: All chunks concatenated (or sampled if >100K tokens), plus filename.

**Prompt**:
```
You are analyzing an insurance/business document for a retrieval system.
Document: {filename}

Here is the full document text, divided into chunks:

{chunk_texts}

Produce a structured analysis:

DOCUMENT_TYPE: (policy|certificate|loss_run|reserve_study|ccr|endorsement|
  declaration|application|schedule_of_values|financial|letter|other)
CLIENT_NAME: (the named insured, HOA, or primary entity)
INSURERS: (comma-separated list of all insurance companies/carriers mentioned)
COVERAGE_TYPES: (comma-separated: gl, wc, do, epli, umbrella, property, crime, auto, cyber, other)
POLICY_PERIOD: (start-end dates, or "N/A")
TOTAL_PREMIUM: (dollar amount, or "N/A")
KEY_LIMITS: (JSON object mapping coverage to limit amounts)
KEY_FACTS: (3-8 bullet points of the most important details a user might ask about)
SYNOPSIS: (100-300 word summary describing what this document contains,
  its key coverages/limits/deductibles, insurer names, and what questions it answers)
```

**Estimated tokens**: ~2000-8000 input, ~500 output per document.

#### Call 2: Chunk Entity Extraction (batched, 5 chunks per call)

**Input**: 5 chunks + document synopsis for context.

**Prompt**:
```
You are extracting structured data from document chunks for a retrieval system.

Document context: {synopsis.text}
Document type: {synopsis.document_type}
Client: {synopsis.client_name}

For each chunk below, extract:
1. SUMMARY: 1-3 sentence plain-English interpretation
2. ENTITIES: Key-value pairs (names, numbers, dates, limits, policy numbers, etc.)
3. COVERAGE_TYPE: Which coverage this chunk relates to (or "general")

---
CHUNK {index}: {chunk_text}
---

Respond in this exact format for each chunk:

CHUNK {index}:
SUMMARY: ...
ENTITIES: key1=value1; key2=value2; ...
COVERAGE_TYPE: ...
```

**Estimated tokens**: ~3000 input, ~500 output per batch of 5 chunks.

#### Prompt Caching Strategy

The chunk enrichment calls (Call 2) share a common prefix: the system prompt + document synopsis. Anthropic's prompt caching allows this prefix to be cached on the first call and reused for subsequent batch calls within the same document, reducing input token costs by ~90% on the cached portion.

**Implementation**:
```python
# Using the Anthropic SDK's cache_control feature
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": system_prompt + synopsis_context,
                "cache_control": {"type": "ephemeral"},  # Cache this prefix
            },
            {
                "type": "text",
                "text": chunk_batch_text,  # Varies per batch (not cached)
            },
        ],
    }
]
```

**Cost impact**: For a 30-chunk document (6 batch calls), the synopsis context (~1500 tokens) is sent once at full price and 5 times at the cached rate (10% of input cost). Estimated savings: ~30-40% on total input tokens per document.

**Cache TTL**: Anthropic's ephemeral cache lasts 5 minutes, which is sufficient since all batch calls for a single document complete within ~60 seconds.

### Enriched Qdrant Payload

The enriched chunk text stored in Qdrant combines summary, entities, and original text:

```
[SUMMARY] This chunk describes the General Liability coverage for Bethany Terrace HOA.
The per-occurrence limit is $1,000,000 with a $2,000,000 aggregate.
[ENTITIES] insured=Bethany Terrace HOA; insurer=Philadelphia Indemnity;
gl_per_occurrence=$1,000,000; gl_aggregate=$2,000,000; policy_number=PHPK2345678;
deductible=$10,000
[ORIGINAL] The General Liability coverage provides protection against bodily
injury and property damage claims. The limit per occurrence is $1,000,000
with an annual aggregate of $2,000,000...
```

**Why this format works**:
1. The `[SUMMARY]` block matches semantic queries ("What is the GL coverage?")
2. The `[ENTITIES]` block matches specific value queries ("What is the GL limit?")
3. The `[ORIGINAL]` block preserves citation fidelity (LLM can cite exact source text)
4. All three are embedded together, so the vector captures all dimensions of meaning

### Qdrant Payload Metadata

Additional metadata fields stored per chunk (not embedded, used for filtering):

```python
{
    # Existing fields (unchanged)
    "chunk_id": "doc-id:chunk-index",
    "document_id": "uuid",
    "document_name": "filename",
    "chunk_text": enriched_chunk.enriched_text,  # NEW: enriched format
    "tags": ["client:bethany-terrace", "doctype:certificate", "coverage:gl"],
    "tenant_id": "default",
    "uploaded_by": "user-id",
    "page_number": 5,
    "section": "General Liability",

    # NEW enrichment metadata
    "enrichment_version": "claude-sonnet-4-6-v1",
    "original_text": "raw docling text...",  # Preserved for citation
    "entities_json": '{"insured": "Bethany Terrace HOA", ...}',  # Searchable
    "coverage_type": "gl",
    "is_synopsis": False,
}
```

### Synopsis Chunk

The document synopsis is stored as a special chunk (index -1) with high retrieval priority:

```python
{
    "chunk_text": "DOCUMENT SYNOPSIS: This is the Bethany Terrace HOA Certificate...",
    "is_synopsis": True,
    "is_summary": True,  # Backward compat with existing summary chunks
    "document_type": "certificate",
    "client_name": "Bethany Terrace HOA",
    "insurers_json": '["Philadelphia Indemnity", "Great American"]',
    "coverage_types_json": '["gl", "wc", "umbrella", "do"]',
    "key_limits_json": '{"gl_per_occurrence": "$1,000,000", ...}',
    "total_premium": "$45,230",
    "policy_period": "01/01/2025-01/01/2026",
}
```

### Structured Data in SQLite

In addition to Qdrant enrichment, extracted entities are stored in a dedicated SQLite table for direct lookup queries (similar to existing `forms_data.db` pattern):

```sql
CREATE TABLE enrichment_entities (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    entity_key TEXT NOT NULL,       -- e.g., "gl_per_occurrence"
    entity_value TEXT NOT NULL,     -- e.g., "$1,000,000"
    entity_type TEXT,               -- e.g., "currency", "policy_number", "date", "name"
    coverage_type TEXT,             -- e.g., "gl", "wc", "do"
    confidence REAL DEFAULT 1.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(document_id, chunk_index, entity_key)
);

CREATE INDEX idx_enrichment_doc ON enrichment_entities(document_id);
CREATE INDEX idx_enrichment_key ON enrichment_entities(entity_key);
CREATE INDEX idx_enrichment_coverage ON enrichment_entities(coverage_type);

CREATE TABLE enrichment_synopses (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,
    synopsis_text TEXT NOT NULL,
    document_type TEXT,
    client_name TEXT,
    insurers_json TEXT,              -- JSON array
    coverage_types_json TEXT,        -- JSON array
    policy_period TEXT,
    total_premium TEXT,
    key_limits_json TEXT,            -- JSON object
    key_facts_json TEXT,             -- JSON array
    enrichment_model TEXT,           -- e.g., "claude-sonnet-4-6"
    enrichment_version TEXT,         -- e.g., "v1"
    tokens_used INTEGER,
    cost_usd REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME
);

CREATE INDEX idx_synopsis_client ON enrichment_synopses(client_name);
CREATE INDEX idx_synopsis_doctype ON enrichment_synopses(document_type);
```

### Configuration Settings

New settings in `config.py`:

```python
# Claude Enrichment Pipeline
claude_enrichment_enabled: bool = False          # Master feature flag
claude_api_key: str | None = None                # ANTHROPIC_API_KEY env var
claude_api_base_url: str = "https://api.anthropic.com"
claude_enrichment_model: str = "claude-sonnet-4-6"  # Model for enrichment
claude_enrichment_batch_size: int = 5            # Chunks per API call
claude_enrichment_max_doc_tokens: int = 100000   # Max tokens for synopsis call
claude_enrichment_timeout_seconds: int = 120     # Per API call timeout
claude_enrichment_max_retries: int = 2           # Retry on transient failures
claude_enrichment_db_path: str = "./data/enrichment.db"  # SQLite for entities
```

**Environment variable mapping**:
```bash
CLAUDE_ENRICHMENT_ENABLED=true
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_ENRICHMENT_MODEL=claude-sonnet-4-6
```

### Deployment Modes

#### Mode 1: Direct Enrichment on Spark

The Spark has temporary internet access during document ingestion.

```
Upload → Docling → Chunks → Claude API → Enriched Chunks → Qdrant + SQLite
```

**Config**: Set `CLAUDE_ENRICHMENT_ENABLED=true` and `ANTHROPIC_API_KEY` on the Spark.

#### Mode 2: Workstation Enrichment + Transfer

Documents are processed on a network-connected workstation, then the enriched databases are transferred to the air-gapped Spark.

```
WORKSTATION (network connected):
Upload → Docling → Chunks → Claude API → Enriched Qdrant + SQLite

TRANSFER:
Qdrant snapshot + enrichment.db + ai_ready_rag.db → USB/SCP → Spark

SPARK (air-gapped):
Receive databases → Serve queries via Ollama (no internet needed)
```

**Implementation**: A CLI command for export/import:
```bash
# On workstation: export enriched data
python -m ai_ready_rag.cli export-enriched --output /path/to/export/

# On Spark: import enriched data
python -m ai_ready_rag.cli import-enriched --input /path/to/export/
```

The export includes:
- Qdrant collection snapshot (binary)
- `enrichment.db` (SQLite entities + synopses)
- `ai_ready_rag.db` (document records + tags)
- Upload files (original documents)

### Batch Re-Enrichment

A CLI command to re-enrich existing documents without re-uploading:

```bash
# Re-enrich all documents
python -m ai_ready_rag.cli enrich --all

# Re-enrich specific document
python -m ai_ready_rag.cli enrich --document-id <uuid>

# Re-enrich documents missing enrichment
python -m ai_ready_rag.cli enrich --missing-only

# Dry run (estimate cost without calling API)
python -m ai_ready_rag.cli enrich --all --dry-run
```

### Cost Tracking

Each enrichment operation logs:
- Tokens used (input + output)
- Estimated cost (based on model pricing)
- Time elapsed

Stored in `enrichment_synopses.tokens_used` and `cost_usd` per document.

Aggregate cost available via admin API:
```
GET /api/admin/enrichment/stats
→ {"total_documents": 50, "total_tokens": 125000, "total_cost_usd": 1.87}
```

### Error Handling

| Scenario | Behavior |
|----------|----------|
| Claude API key missing | Skip enrichment, fall back to Ollama pipeline |
| Claude API timeout | Retry up to `max_retries`, then fall back to Ollama pipeline |
| Claude API rate limit (429) | Exponential backoff with jitter, retry |
| Claude API server error (5xx) | Retry up to `max_retries`, then fall back to Ollama pipeline |
| Parsing failure (bad response format) | Log warning, store raw response as summary, skip entity extraction for that batch |
| Token budget exceeded (huge document) | Sample chunks (first 3 + middle 3 + last 3 per section), note sampling in metadata |

**Critical invariant**: Enrichment failure **never** blocks document processing. The system always falls back to the existing Ollama pipeline.

### Document Model Changes

New nullable fields on `Document` model:

```python
# Claude enrichment fields
enrichment_status = Column(String, nullable=True)   # null|pending|completed|failed|skipped
enrichment_model = Column(String, nullable=True)     # e.g., "claude-sonnet-4-6"
enrichment_version = Column(String, nullable=True)   # e.g., "v1"
enrichment_tokens_used = Column(Integer, nullable=True)
enrichment_cost_usd = Column(Float, nullable=True)
enrichment_completed_at = Column(DateTime, nullable=True)
```

### Query-Time Impact

**No changes to the RAG query pipeline are required in this spec.** The enrichment improves query results through better vector content:

1. **Re-embedded enriched text**: Chunks are re-embedded using the enriched `[SUMMARY] + [ENTITIES] + [ORIGINAL]` text. This produces vectors that match both semantic queries ("What is the GL coverage?") and specific-value queries ("What is the GL limit for Bethany Terrace?"). The ~30s Ollama embedding cost per document is negligible compared to the quality improvement.
2. **Better LLM context**: When Ollama receives enriched chunks, the `[SUMMARY]` and `[ENTITIES]` sections make it trivial to extract answers.
3. **Synopsis chunks**: Document-level synopses answer broad questions ("Summarize the coverages") without needing chunk retrieval.

**Future optimization** (out of scope): Modify the RAG prompt template to instruct Ollama to prioritize `[ENTITIES]` sections for specific-value queries.

## Files Affected

### New Files
| File | Purpose |
|------|---------|
| `ai_ready_rag/services/claude_enrichment_service.py` | Core enrichment service |
| `ai_ready_rag/db/models/enrichment.py` | SQLAlchemy models for enrichment_entities and enrichment_synopses |
| `ai_ready_rag/cli/enrich.py` | CLI commands for batch enrichment and export/import |
| `tests/test_claude_enrichment.py` | Unit tests with mocked Claude API |

### Modified Files
| File | Changes |
|------|---------|
| `ai_ready_rag/services/processing_service.py` | Add enrichment branch (lines 278-313) |
| `ai_ready_rag/config.py` | Add Claude enrichment settings |
| `ai_ready_rag/db/models/document.py` | Add enrichment_* fields |
| `ai_ready_rag/db/database.py` | Register new enrichment models |
| `requirements-wsl.txt` | Add `anthropic` package |
| `requirements-spark.txt` | Add `anthropic` package |

### Unchanged Files
| File | Why Unchanged |
|------|---------------|
| `ai_ready_rag/services/rag_service.py` | Query pipeline reads enriched chunks transparently |
| `ai_ready_rag/services/vector_service.py` | `add_document()` API unchanged; enriched text passed as chunks |
| `ai_ready_rag/services/summary_generator.py` | Kept for fallback when enrichment disabled |
| `ai_ready_rag/services/auto_tagging/` | Kept for fallback when enrichment disabled |
| `ai_ready_rag/services/forms_processing_service.py` | Kept for fallback when enrichment disabled; bypassed when enabled |
| `ai_ready_rag/services/forms_query_service.py` | Kept for fallback; enrichment entities served via enrichment_entities table instead |

## Cost Estimate

### Per-Document Cost (Claude Sonnet, with prompt caching)

| Operation | Input Tokens | Output Tokens | Cost |
|-----------|-------------|---------------|------|
| Synopsis (1 call) | ~4,000 | ~500 | ~$0.015 |
| Chunk enrichment — 1st batch (cache miss) | ~3,000 | ~500 | ~$0.012 |
| Chunk enrichment — remaining batches (cache hit, ~30-40% savings) | ~1,500 effective × 5 | ~500 × 5 | ~$0.035 |
| **Total per document** | ~15,500 effective | ~3,500 | **~$0.06** |

*With prompt caching, the synopsis context (~1,500 tokens) is cached after the first batch call and reused at 10% input cost for subsequent batches. Net savings: ~25% vs. uncached.*

### Batch Cost Estimates

| Corpus Size | Estimated Cost | Time (sequential) |
|-------------|---------------|-------------------|
| 10 documents | ~$0.80 | ~5 min |
| 50 documents | ~$4.00 | ~25 min |
| 100 documents | ~$8.00 | ~50 min |
| 500 documents | ~$40.00 | ~4 hours |

*Note: Costs based on Claude Sonnet 4.6 pricing ($3/MTok input, $15/MTok output). Actual costs may vary with document size. Using Claude Haiku would reduce costs by ~10x.*

## Risk Flags

### ⚠️ API Key Security
- Claude API key is a secret. MUST be stored as environment variable, NEVER in config files or committed to git.
- Existing pattern: follows `jwt_secret_key` precedent in `config.py`.

### ⚠️ Data Leaves Device
- Document text is sent to Anthropic's API during enrichment.
- **Mitigation**: Feature flag is OFF by default. Must be explicitly enabled. Admin should verify client data handling agreements before enabling.
- **Mitigation**: Mode 2 (workstation transfer) keeps documents on the office network; only the workstation talks to Claude API.

### ⚠️ Enrichment Adds Latency to Ingestion
- Each document takes ~30-60 seconds longer to process (Claude API calls).
- **Mitigation**: Enrichment runs as part of the existing async/background processing pipeline (ARQ worker). User sees "processing" status until complete. No impact on interactive responsiveness.

### ⚠️ Fallback Complexity
- Two code paths (enriched vs. Ollama) must both work correctly.
- **Mitigation**: Feature flag makes the switch clean. Extensive tests for both paths. Enrichment failure always falls back gracefully.

### ⚠️ Qdrant Chunk Format Change
- Enriched chunks have different text format (`[SUMMARY]...[ENTITIES]...[ORIGINAL]...`) than raw chunks.
- **Mitigation**: Re-enrichment command handles migration. Enrichment metadata tracks version. Mixed collections (some enriched, some raw) work fine — enriched chunks just score higher for relevant queries.

## Acceptance Criteria

- [ ] `claude_enrichment_enabled=false` (default): existing pipeline works identically, no regressions
- [ ] `claude_enrichment_enabled=true` with valid API key: documents are enriched with synopsis + entities
- [ ] `claude_enrichment_enabled=true` without API key: logs warning, falls back to Ollama pipeline
- [ ] Claude API failure during enrichment: falls back to Ollama pipeline, document still processes successfully
- [ ] Enriched chunks contain `[SUMMARY]`, `[ENTITIES]`, and `[ORIGINAL]` sections
- [ ] Synopsis chunk stored with `is_synopsis=True` and structured metadata
- [ ] Entity data stored in `enrichment_entities` SQLite table
- [ ] Synopsis data stored in `enrichment_synopses` SQLite table
- [ ] Document model updated with enrichment_status, enrichment_model, enrichment_cost_usd
- [ ] `python -m ai_ready_rag.cli enrich --all --dry-run` reports estimated cost without API calls
- [ ] `python -m ai_ready_rag.cli enrich --all` enriches all documents
- [ ] `python -m ai_ready_rag.cli export-enriched` exports databases for transfer
- [ ] `python -m ai_ready_rag.cli import-enriched` imports databases on target machine
- [ ] Cost tracking: per-document and aggregate cost available via admin API
- [ ] All 16 demo questions achieve confidence > 60 after enrichment (measured via cache warming)
- [ ] No hardcoded API keys in codebase
- [ ] Tests pass with mocked Claude API responses

## Design Decisions (Resolved)

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | **Prompt caching** | Include in v1 | Synopsis context is repeated per chunk batch. Ephemeral cache reduces input costs ~30-40%. Straightforward SDK flag. |
| 2 | **Model selection** | Sonnet for everything | Consistent quality across synopsis and entity extraction. Insurance docs are nuanced enough to justify frontier model. ~$0.08/doc is acceptable. |
| 3 | **Re-embedding** | Re-embed with enriched text | Enriched text produces vectors that match both semantic and specific-value queries. ~30s extra Ollama compute per document is negligible. |
| 4 | **Forms pipeline interaction** | Replace entirely | Claude handles all structured extraction including ACORD forms. Eliminates ingestkit-forms dependency when enrichment is enabled. Forms pipeline preserved as fallback when enrichment disabled. |

## Open Questions

*(None — all design questions resolved.)*

---

**Next Steps**:
1. Review this spec
2. Fill any [TODO] sections
3. Run `/spec-review specs/CLAUDE_ENRICHMENT_PIPELINE_v1.md`
