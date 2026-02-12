# Document Summary Chunks — Design Document

**Status:** Proposed
**Date:** 2026-02-11
**Branch:** TBD (feat/issue-XXX-summary-chunks)

## Problem

When users ask broad or document-selection questions ("What's the earthquake deductible?", "What does the D&O policy cover?"), the RAG retriever pulls chunks from multiple documents that mention similar terms. The LLM receives competing information from quotes, bound policies, submissions, and loss runs — leading to confused or incorrect answers.

The root cause: the retriever has no way to identify the *right document* before diving into granular chunks. Every chunk looks the same regardless of whether it comes from a bound policy or an expired quote.

## Solution

Generate a structured summary for each document at processing time. Store the summary as a special chunk in Qdrant alongside the regular chunks. The summary participates in normal vector search with no retrieval code changes required.

## Pipeline Integration

```
Upload → Docling Parse → HybridChunker → Generate Summary → Embed All → Qdrant
```

The summary generation step slots into `ProcessingService.process_document()` after chunking and before embedding. It adds one Ollama LLM call per document.

### Processing Flow

1. `DoclingChunker.chunk_document()` produces `N` chunks (existing)
2. **NEW:** `SummaryGenerator.generate()` samples chunks, calls Ollama, returns a summary chunk dict
3. Summary chunk is appended to the chunk list
4. All chunks (regular + summary) are embedded and stored in Qdrant (existing)

## Summary Chunk Structure

Stored in Qdrant with the same schema as regular chunks, plus additional metadata fields:

```python
{
    "chunk_text": "DOCUMENT SUMMARY: Directors & Officers and Crime Liability Policy issued by CNA for Cervantes Villas HOA. Policy period 12/01/2024 to 12/01/2025. Coverage includes D&O liability ($2M aggregate), Employment Practices Liability, and Crime/Fidelity ($500K). Key exclusions: bodily injury, property damage, pollution. Deductible: $5,000 per claim. This document answers questions about D&O coverage limits, exclusions, crime coverage, policy conditions, and claim reporting procedures.",
    "chunk_index": -1,
    "document_id": "abc-123",
    "is_summary": true,
    "document_type": "policy",
    "carrier": "CNA",
    "policy_period": "2024-2025",
    "page_number": null,
    "section": "Document Summary",
    "tags": ["cervantes"],
}
```

### Metadata Fields

| Field | Type | Purpose |
|-------|------|---------|
| `is_summary` | bool | Distinguishes summary from regular chunks |
| `chunk_index` | int | `-1` signals non-positional chunk |
| `document_type` | string | LLM-extracted: policy, quote, submission, loss_run, financial, sov, certificate, application, other |
| `carrier` | string | LLM-extracted carrier/market name (nullable) |
| `policy_period` | string | LLM-extracted period (nullable) |

## LLM Prompt Design

The prompt is tuned for insurance documents but works generically:

```
You are summarizing a document for a retrieval-augmented generation (RAG) system.
Your summary will be embedded and used to match user questions to the right document.

Document: {filename}
Total chunks: {chunk_count}
Total words: {word_count}

Here are representative samples from the document:

{first 3 chunks}
{middle 3 chunks}
{last 3 chunks}

Produce the following (be precise, use exact numbers/names from the document):

DOCUMENT_TYPE: (one of: policy, quote, submission, loss_run, financial, sov, certificate, application, other)
CARRIER: (carrier or market name, or "unknown")
POLICY_PERIOD: (e.g., "12/01/2024-12/01/2025", or "unknown")
SUMMARY: (one paragraph, 100-200 words) Describe what this document contains, its key
coverages/limits/deductibles if applicable, and what types of questions it could answer.
Include specific numbers, names, and dates from the content.
```

### Chunk Sampling Strategy

For the prompt context, sample up to 9 chunks:
- First 3 chunks (title pages, declarations, table of contents)
- Middle 3 chunks (core content)
- Last 3 chunks (endorsements, signatures, schedules)

This gives the LLM a representative view without exceeding context limits.

### Response Parsing

Parse the structured fields from the LLM response:
1. Extract `DOCUMENT_TYPE:` line → `document_type` metadata
2. Extract `CARRIER:` line → `carrier` metadata
3. Extract `POLICY_PERIOD:` line → `policy_period` metadata
4. Extract `SUMMARY:` paragraph → prepend "DOCUMENT SUMMARY: " → `chunk_text`

If parsing fails, fall back to using the entire response as the summary text with `document_type="other"`.

## Why It Works

### Example: "What's the earthquake deductible for Cervantes?"

**Without summary chunks:**
Retriever returns chunks from:
- EQ Quote.pdf (proposed deductible options)
- 2025 Coverage Summary.xlsx (summary line item)
- CondoLogic Package Quote.pdf (competing quote)
- DIC SOV.xlsx (schedule of values)

The LLM sees conflicting numbers from quotes vs bound coverage.

**With summary chunks:**
The summary chunk for the bound DIC/EQ policy scores high because it contains "earthquake", "deductible", and the specific dollar amount. The retriever pulls this summary + its sibling regular chunks, giving the LLM focused context from the correct document.

### Example: "Compare D&O coverage between 2024 and 2025"

Summary chunks for both D&O policies surface because they contain "D&O", the respective years, and coverage details. The LLM receives two clear document summaries plus supporting chunks from each, enabling comparison.

## Implementation Details

### New File: `ai_ready_rag/services/summary_generator.py`

```python
class SummaryGenerator:
    """Generate document summaries via Ollama for RAG indexing."""

    def __init__(self, ollama_url: str, model: str = "qwen3:8b"):
        self.ollama_url = ollama_url
        self.model = model

    async def generate(
        self, chunks: list[dict], filename: str, document_id: str
    ) -> dict | None:
        """Generate a summary chunk from sampled document chunks.

        Returns a chunk dict ready for embedding, or None on failure.
        """
        ...
```

### Modified: `ai_ready_rag/services/processing_service.py`

After chunking, before embedding:

```python
# Generate summary chunk
summary_generator = SummaryGenerator(
    ollama_url=settings.ollama_base_url,
    model=settings.chat_model,
)
summary_chunk = await summary_generator.generate(chunks, document.original_filename, document.id)
if summary_chunk:
    chunks.append(summary_chunk)
```

### Modified: `ai_ready_rag/services/vector_service.py` (if needed)

May need to handle `is_summary` metadata in payload. The existing `store_chunks()` method should work as-is since it stores arbitrary metadata dicts.

### Configuration

Add to `config.py`:

```python
generate_summaries: bool = True  # Enable/disable summary generation
summary_model: str | None = None  # Override model (default: use chat_model)
```

## Impact Assessment

| Aspect | Impact |
|--------|--------|
| Processing time | +5-10s per document (one Ollama call) |
| Storage | +1 chunk per document in Qdrant (negligible) |
| Embedding cost | +1 embedding per document (negligible) |
| Code changes | ~150 lines new, ~20 lines modified |
| Retrieval changes | None — summary is just another searchable chunk |
| Existing documents | Require one-time reprocessing to generate summaries |

## What This Does NOT Solve

- **Cross-document comparison** — Helps surface the right documents, but multi-step retrieval is needed for true comparison workflows
- **Structured table data** — SOVs and financials still lose structure when chunked as text
- **Insurance synonym matching** — "deductible" vs "retention" vs "SIR" still needs a glossary in the system prompt or query expansion

These are separate improvements that can be layered on top.

## Testing Plan

1. **Unit test:** `SummaryGenerator` with mocked Ollama response → verify chunk dict structure
2. **Unit test:** Response parsing with edge cases (missing fields, malformed response)
3. **Integration test:** Process a test document → verify summary chunk appears in Qdrant with correct metadata
4. **Manual validation:** Reprocess Cervantes documents → review summary quality in exported markdown
5. **Retrieval test:** Ask broad questions before/after summaries → compare retrieval precision

## Rollout

1. Implement behind `generate_summaries=True` config flag (default: True)
2. Deploy to Spark
3. Reprocess Cervantes documents (53 docs, ~5-10 min total)
4. Export chunks again to review summary quality
5. Test with real user questions
