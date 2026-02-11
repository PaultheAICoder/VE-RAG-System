# GitHub Open-Source RAG Pattern Benchmark (2026-02-11)

## Purpose
Validate that `VE-RAG-System` is aligned with strong open-source patterns for enterprise, on-prem/private-VPC, SMB-friendly RAG ingestion and retrieval.

## Benchmark Shortlist (Direct + Adjacent)
1. **Unstructured** (`Unstructured-IO/unstructured`)  
   Focus: production document partitioning + connectors + normalization.
2. **Docling** (`docling-project/docling`)  
   Focus: high-fidelity PDF/doc parsing pipelines and structured extraction.
3. **MarkItDown** (`microsoft/markitdown`)  
   Focus: broad-format conversion into markdown/text for downstream chunking.
4. **Haystack** (`deepset-ai/haystack`)  
   Focus: modular RAG pipeline architecture, evaluation, components.
5. **LlamaIndex** (`run-llama/llama_index`)  
   Focus: ingestion abstractions, indices, retrieval orchestration.
6. **LangChain + Unstructured integration** (`langchain-ai/langchain`, `langchain-ai/langchain-unstructured`)  
   Focus: parser/loader composition and fallback chains.
7. **Enterprise/Cloud-adjacent templates** (`aws-ia/...`, `stackitcloud/...`)  
   Focus: secure ingestion blueprint patterns (even when infra differs).

## Best Patterns Observed Across Strong OSS Projects
1. **Parser fallback chains, not single-parser dependency**  
   Pattern: primary parser -> secondary parser -> final text fallback with reason codes.
2. **Strict stage contracts**  
   Pattern: classify -> parse -> normalize -> chunk -> embed -> index with typed outputs.
3. **Idempotent ingestion keys**  
   Pattern: content-hash + source-id dedupe to avoid duplicate embedding cost.
4. **Recoverable async workflow**  
   Pattern: explicit status machine, retries, stale-lease reclaim, dead-letter handling.
5. **Normalized metadata schema**  
   Pattern: stable metadata fields (source, page, section, doc_type, confidence, parser_used).
6. **Evaluation hooks from day one**  
   Pattern: ingest quality metrics (parse coverage, chunk quality, recall@k proxies).
7. **PII-safe observability**  
   Pattern: structured logs with redaction and stage-level durations/error codes.
8. **Format-specific chunking strategies**  
   Pattern: tables, headers, legal clauses, and forms chunked differently than prose.

## Fit vs `VE-RAG-System` (Current Direction)
### Already aligned
1. Queue-based background processing and batch-oriented workflow.
2. Enterprise constraints (on-prem/private VPC and air-gapped assumptions).
3. Active work on warming/recovery reliability and evaluation framework.

### Gaps to close to match strongest OSS patterns
1. **Ingestion fallback chain standardization**  
   Current risk: parser failure path can be inconsistent between file types.
2. **Canonical parser decision record**  
   Current risk: limited auditability of why parser/model path was chosen.
3. **End-to-end idempotency contract**  
   Current risk: potential reprocessing/re-embedding duplicates under retries.
4. **Typed pipeline stage outputs**  
   Current risk: stage drift and harder regression testing.
5. **Evaluation signals tightly coupled to ingestion stages**  
   Current risk: difficult to prove quality change before/after pipeline updates.

## Recommended Implementation Backlog (Prioritized)
### P0 (next 2-4 weeks)
1. Add **parser fallback chain policy** per document class with explicit reason codes.
2. Add **idempotency key**: `tenant_id + source_uri + content_hash + parser_version`.
3. Add **structured stage result schema** persisted for each ingest step.
4. Add **PII-safe structured logging** with stage timing + normalized error taxonomy.

### P1 (next 1-2 months)
1. Add **format-aware chunking profiles** (legal contracts, financial statements, manufacturing SOPs, forms/tables).
2. Add **ingestion quality dashboard**: parse success by type, avg chunk token variance, duplicate rate, retry rate.
3. Add **automated regression corpus** for top target verticals (legal, manufacturing, financial services, agriculture, landscaping).

### P2 (next 2-3 months)
1. Add **policy-driven parser routing** (deterministic rules with optional ML classifier hints).
2. Add **cross-parser A/B capability** for quality/cost tuning in pilots.
3. Add **operator controls** for fail-open vs fail-closed modes in air-gapped deployments.

## Architecture Guardrails for Your Market Position
1. Keep primary deployment paths strictly **on-prem/private VPC**.
2. Keep a **sub-$1k/month SMB starter profile** with controlled ingestion throughput and model footprint.
3. Separate enterprise-only features from core path so SMB offering stays operationally simple.
4. Preserve deterministic/offline fallback behavior for all critical ingestion steps.

## Competitiveness Impact (Practical)
1. Reduces ingestion failures and support burden in paid pilots.
2. Lowers reprocessing/embedding spend through idempotency and dedupe.
3. Improves trust for security-sensitive buyers via auditable parser and retrieval lineage.
4. Strengthens win-rate against cloud-first platforms where air-gapped and private-VPC constraints are mandatory.

## Suggested Success Metrics (90-day)
1. Parse success rate by priority file types: `>= 98%`.
2. Duplicate embed operations reduced by `>= 30%`.
3. Mean ingest retry count reduced by `>= 25%`.
4. Median ingest-to-search availability time improved by `>= 20%`.
5. Pilot defect rate tied to ingestion/parsing reduced month-over-month.

## Notes
- This benchmark focuses on architectural patterns and execution models, not feature-count parity.
- Cloud-only features from adjacent projects should be treated as reference patterns, not deployment requirements.
