---
title: "Insurance AI Platform — Intelligent Document Management & Knowledge System"
status: DRAFT
version: v1.1
created: 2026-02-19
updated: 2026-02-20
author: —
type: Fullstack
complexity: COMPLEX
supersedes: CLAUDE_ENRICHMENT_PIPELINE_v1.md
---

# Insurance AI Platform — Intelligent Document Management & Knowledge System

## Summary

Transform the VE-RAG System from a general-purpose RAG application into a purpose-built **Insurance AI Platform** that combines intelligent document ingestion, structured data extraction, and natural-language querying to give insurance agents instant access to every fact across their book of business.

The platform is offered in **two tiers**:

- **Standard (Hosted)**: Cloud-deployed as a **single-tenant instance per customer** — each agency gets its own isolated application, database, and vector store. Claude API handles both ingestion enrichment and query-time answering. No hardware purchase, highest answer quality, monthly subscription model.
- **Enterprise (Air-Gapped)**: On-prem DGX Spark deployment. Claude enriches during ingestion only; Ollama answers queries locally with no internet. Maximum data privacy, hardware purchase model.

Both tiers share the same core: Claude API extracts structured insurance data (policies, coverages, limits, carriers, claims) into SQL tables during ingestion, while simultaneously enriching unstructured content for semantic search via **pgvector** (PostgreSQL's native vector extension). Structured data and vectors live in a single PostgreSQL database, enabling SQL JOINs between insurance tables and vector similarity results. At query time, a deterministic router tries SQL first for structured questions, then falls back to RAG for analytical queries.

## Problem Statement

### Current State

Insurance agents manage hundreds of documents per property — policies, certificates, loss runs, endorsements, CC&Rs, reserve studies, correspondence. Today:

1. **Manual lookup**: Agent opens 3-4 PDFs to answer a single question about limits or carriers
2. **No structured data**: Policy details live inside PDF text, not queryable databases
3. **Scattered files**: Documents organized by brokerage workflow (Sub/Quote/Bind), not by what the agent needs to know
4. **Renewal prep is manual**: 2-3 hours to assemble submission packages by pulling data from multiple folders
5. **No compliance checking**: Agent manually cross-references CC&R requirements against current coverage
6. **Loss history fragmented**: Loss runs scattered across year folders; no consolidated view

### Root Cause

Documents contain both **structured data** (policy numbers, limits, premiums, dates) and **unstructured content** (legal language, coverage descriptions, exclusion clauses). Current RAG systems treat everything as unstructured text, forcing the LLM to re-extract facts from raw chunks on every query. This is slow, unreliable, and wastes inference compute.

### Required State

A dual-path system where:
- **Structured facts** are extracted once during ingestion and stored in SQL tables → deterministic, instant lookups
- **Unstructured content** is enriched with summaries and entities → high-quality semantic search
- **Smart routing** directs each query to the optimal path → best answer, fastest response

## Goals

1. **Sub-second answers for structured queries**: Policy limits, carriers, premiums, policy numbers via SQL lookup
2. **High-confidence RAG for analytical queries**: CC&R analysis, coverage comparisons, reserve study interpretation
3. **Automated document classification and filing**: Drop a PDF, system classifies and organizes it
4. **Structured insurance database**: Queryable SQL tables for accounts, policies, coverages, claims
5. **Renewal prep automation**: One-click generation of coverage summaries and submission data
6. **Compliance gap detection**: Automated cross-reference of coverage vs. CC&R/bylaw requirements
7. **Two deployment tiers**: Standard (hosted, Claude for queries) and Enterprise (air-gapped, Ollama for queries)
8. **Cost efficient**: Claude enrichment at ~$0.04/document with prompt caching

## Scope

### In Scope — Phase 1 (Core Platform)

- Claude API enrichment service (replaces Ollama summary generator, auto-tagger, forms pipeline)
- Insurance-specific SQL schema (accounts, policies, coverages, claims, certificates)
- Structured entity extraction during ingestion → SQL table population
- Enriched chunk storage in pgvector (summary + entities + original text in same PostgreSQL DB)
- Deterministic query router: SQL-first for structured queries, RAG fallback for unstructured
- Document type classification (policy, certificate, loss run, endorsement, etc.)
- Per-document synopsis generation
- Configuration settings, feature flags, cost tracking
- Standard tier: single-tenant hosted instances with Claude query-time answering + PostgreSQL
- Enterprise tier: air-gapped DGX Spark with Ollama query-time + local PostgreSQL
- CLI commands for batch enrichment, re-enrichment, export/import
- Gold evaluation harness (16+ questions, automated scoring)
- Data lifecycle management (soft-delete, versioning, cascade rules)
- Low-confidence review workflow with human-in-the-loop gates

### In Scope — Phase 2 (Automation)

- Renewal summary generation from structured data
- Coverage gap detection (current coverage vs. CC&R requirements)
- Loss history consolidation and trend analysis
- Certificate tracking and issuance data
- Cross-account portfolio queries
- Unit owner letter template generation

### Out of Scope

- Changes to Docling/OCR chunking stage (enrichment happens after chunking)
- Email (.msg/.eml) content analysis (future phase)
- Carrier API integrations (IVANS, etc.)
- Premium calculation or rating
- Policy issuance or binding workflows

---

## Technical Specification

### Tier Comparison

| Capability | Standard (Hosted) | Enterprise (Air-Gapped) |
|---|---|---|
| **Deployment** | Cloud VM (Docker), one instance per customer | DGX Spark on-prem |
| **Tenant Model** | Single-tenant (isolated instance per agency) | Single agency |
| **Ingestion LLM** | Claude API (Sonnet) | Claude API (Sonnet) |
| **Query-time LLM** | Claude API (Haiku/Sonnet) | Ollama (local 8B model) |
| **Database** | PostgreSQL + pgvector (unified) | SQLite + pgvector or Qdrant |
| **Vector Storage** | pgvector (same PostgreSQL instance) | pgvector (same database) |
| **Document Parsing** | Docling | Docling + OCR |
| **Answer Quality** | Highest (Claude Sonnet/Haiku) | Good (enriched chunks + 8B model) |
| **Data Location** | Encrypted cloud (US), customer-isolated | Customer's office |
| **Internet Required** | Always (for queries) | Only during ingestion |
| **Customization** | Per-customer config, branding, schema extensions | Full local control |
| **Pricing Model** | Monthly subscription | Hardware + license |
| **Setup Time** | Same day (automated provisioning) | Hardware procurement + install |
| **Concurrent Users** | 50+ (cloud scales) | 8-10 (Ollama bottleneck) |

### System Architecture

#### Shared Ingestion Pipeline (Both Tiers)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE (identical)                   │
│                                                                     │
│  Upload → Docling Parse → Chunk → Claude Enrichment → Store        │
│                                       │                             │
│                           ┌───────────┴───────────┐                 │
│                           │                       │                 │
│                    Structured Data          Enriched Chunks          │
│                           │                       │                 │
│                     ┌─────▼─────────────────▼─────┐                 │
│                     │     PostgreSQL + pgvector     │                │
│                     │                               │                │
│                     │  SQL tables:    Vector index:  │                │
│                     │  accounts       chunk_vectors  │                │
│                     │  policies       (768-dim)      │                │
│                     │  coverages                     │                │
│                     │  claims         JOINs across   │                │
│                     │  synopses       SQL ↔ vectors  │                │
│                     └───────────────────────────────┘                │
│                                                                     │
│  Key: SQL tables and vectors share the SAME database.               │
│  Enables: SELECT ... FROM chunk_vectors v                           │
│           JOIN insurance_policies p ON v.document_id = p.source_id  │
│           ORDER BY v.embedding <=> query_embedding                  │
└─────────────────────────────────────────────────────────────────────┘
```

#### Standard Tier — Query Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                QUERY PIPELINE (Standard / Hosted)                   │
│                                                                     │
│  User Question → Intent Classifier / Query Router                   │
│                           │                                         │
│              ┌────────────┴────────────┐                            │
│              │                         │                            │
│       Structured Query           Unstructured Query                 │
│              │                         │                            │
│        SQL Lookup               RAG Retrieval                       │
│        (2ms, exact)             (enriched chunks)                   │
│              │                         │                            │
│              └────────────┬────────────┘                            │
│                           │                                         │
│              Claude API Response (Haiku or Sonnet)                   │
│              ├── Haiku: simple lookups, formatting ($0.001/query)    │
│              └── Sonnet: analysis, comparison ($0.01-0.03/query)    │
└─────────────────────────────────────────────────────────────────────┘
```

#### Enterprise Tier — Query Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│               QUERY PIPELINE (Enterprise / Air-Gapped)              │
│                                                                     │
│  User Question → Intent Classifier / Query Router                   │
│                           │                                         │
│              ┌────────────┴────────────┐                            │
│              │                         │                            │
│       Structured Query           Unstructured Query                 │
│              │                         │                            │
│        SQL Lookup               RAG Retrieval                       │
│        (2ms, exact)             (enriched chunks)                   │
│              │                         │                            │
│              └────────────┬────────────┘                            │
│                           │                                         │
│                    Ollama Response (local, no internet)              │
│                    (cite sources from enriched chunks)               │
└─────────────────────────────────────────────────────────────────────┘
```

### Deployment Architectures

#### Standard Tier: Cloud Hosted (Single-Tenant)

```
Per-Customer Instance (isolated Docker Compose stack)
┌─────────────────────────────────────────────────┐
│          customer-abc.vaultiq.app                │
│                                                  │
│  FastAPI (:8502)       React (served by FastAPI) │
│  ├── /api/auth/*                                 │
│  ├── /api/chat/*  ──► Claude API (queries)       │
│  ├── /api/documents/* ──► Claude API (enrich)    │
│  └── /api/insurance/*                            │
│                                                  │
│  PostgreSQL + pgvector (single database)         │
│  ├── users, chat_*, enrichment_*                 │
│  ├── insurance_* (policies, coverages, claims)   │
│  └── chunk_vectors (pgvector 768-dim index)      │
│                                                  │
│  No cross-customer data sharing.                 │
│  Each instance is independently configurable.    │
└─────────────────────────────────────────────────┘
         │
         ▼
   Claude API (Anthropic)
   ├── Sonnet: enrichment + complex queries
   └── Haiku: simple lookups + classification

Orchestration Layer (internal)
┌─────────────────────────────────────────────────┐
│  Provisioning service: spin up new instances     │
│  Monitoring: health checks per instance          │
│  Backup: per-customer PostgreSQL snapshots        │
│  Updates: rolling deployment across instances    │
└─────────────────────────────────────────────────┘
```

#### Enterprise Tier: Air-Gapped (Mode A — Workstation Transfer)

```
Workstation (internet)              DGX Spark (air-gapped)
┌──────────────────┐                ┌──────────────────┐
│ 1. Upload PDF    │                │                  │
│ 2. Docling parse │                │                  │
│ 3. Claude enrich │──── export ───►│ 4. Import DB     │
│    → SQL tables  │   (signed PG   │    (pg_restore)  │
│    → enriched    │    dump with   │ 5. Query locally │
│      chunks      │    checksum)   │    (Ollama)      │
└──────────────────┘                └──────────────────┘
```

See **Air-Gap Transfer Protocol** section below for manifest format and integrity checks.

#### Enterprise Tier: Air-Gapped (Mode B — Temporary Internet)

```
DGX Spark (temporary internet during ingestion)
┌──────────────────────────────────────────┐
│ 1. Upload PDF                            │
│ 2. Docling parse                         │
│ 3. Claude enrich (API call)              │
│ 4. Store in local PostgreSQL + pgvector  │
│ 5. Disconnect internet                   │
│ 6. Query locally (Ollama, no internet)   │
└──────────────────────────────────────────┘
```

### Claude Model Routing (Standard Tier)

The Standard tier uses different Claude models based on query complexity to optimize cost:

```python
class ClaudeModelRouter:
    """Route queries to the most cost-effective Claude model."""

    HAIKU_PATTERNS = [
        # Simple structured lookups — Haiku is fast and cheap
        r"what (?:is|are) the .+ (?:limit|deductible|premium|number)",
        r"who is the .+ (?:carrier|insurer|broker|agent)",
        r"when does .+ (?:expire|renew|effective)",
        r"is .+ (?:covered|included|excluded)",
        r"list (?:all|the) .+",
    ]

    SONNET_PATTERNS = [
        # Complex analysis — Sonnet for quality
        r"compare",
        r"summarize",
        r"explain",
        r"what does .+ (?:say|require|mean)",
        r"difference between",
        r"recommend",
        r"comply|compliance|gap",
    ]

    def select_model(self, query: str, intent: QueryIntent) -> str:
        if intent == QueryIntent.STRUCTURED:
            return "claude-haiku-4-5-20251001"  # Format SQL results
        elif intent == QueryIntent.COMPARISON:
            return "claude-sonnet-4-20250514"   # Analytical comparison
        elif intent == QueryIntent.ANALYTICAL:
            return "claude-sonnet-4-20250514"   # Deep document analysis
        else:
            return "claude-haiku-4-5-20251001"  # Default to cheap
```

---

## Data Model: Insurance SQL Schema

### Entity Relationship Diagram

```
accounts ──< policies ──< coverages
    │             │
    │             └──< claims
    │
    ├──< certificates
    │
    └──< account_documents (junction → documents)

documents ──< enrichment_entities
    │
    └──< enrichment_synopses
```

### Table Definitions

#### `insurance_accounts`

The top-level entity representing a property or insured entity.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| name | TEXT | No | Named insured (e.g., "Marshall Wells Lofts Condominium") |
| account_type | TEXT | Yes | "condo_association", "commercial", "residential", "hoa" |
| address | TEXT | Yes | Primary property address |
| city | TEXT | Yes | City |
| state | TEXT | Yes | State code |
| zip_code | TEXT | Yes | ZIP code |
| units_residential | INTEGER | Yes | Number of residential units |
| units_commercial | INTEGER | Yes | Number of commercial units |
| year_built | INTEGER | Yes | Construction year |
| construction_type | TEXT | Yes | Frame, masonry, fire-resistive, etc. |
| management_company | TEXT | Yes | Property manager name |
| management_contact | TEXT | Yes | Manager contact info |
| agent_name | TEXT | Yes | Servicing agent |
| agent_email | TEXT | Yes | Agent email |
| tenant_id | TEXT | No | Instance identifier (single-tenant; used for export/import integrity) |
| created_at | DATETIME | No | Timestamp |
| updated_at | DATETIME | Yes | Timestamp |
| is_deleted | BOOLEAN | No | Soft-delete flag (default false) |
| deleted_at | DATETIME | Yes | When soft-deleted |
| valid_from | DATETIME | No | When this record became active |
| valid_to | DATETIME | Yes | When superseded (null = current) |
| source_document_id | TEXT FK | Yes | Document this was extracted from |
| extraction_model | TEXT | Yes | Claude model version used for extraction |
| extraction_confidence | REAL | Yes | Confidence of extraction (0-1) |

#### `insurance_policies`

One row per coverage line per policy period.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| account_id | TEXT FK | No | → insurance_accounts.id |
| line_of_business | TEXT | No | "gl", "property", "do", "crime", "umbrella", "wc", "equipment_breakdown", "epli", "cyber", "auto" |
| carrier | TEXT | Yes | Insurance company name |
| policy_number | TEXT | Yes | Policy number |
| effective_date | DATE | Yes | Policy inception |
| expiration_date | DATE | Yes | Policy expiration |
| status | TEXT | No | "active", "expired", "cancelled", "pending" |
| annual_premium | REAL | Yes | Total annual premium |
| payment_plan | TEXT | Yes | Annual, semi-annual, quarterly, monthly |
| program_name | TEXT | Yes | e.g., "CondoLogic", "HOA Shield" |
| broker | TEXT | Yes | Wholesale broker if applicable |
| is_admitted | BOOLEAN | Yes | Admitted vs. surplus lines |
| layer_position | INTEGER | Yes | 1=primary, 2=first excess, etc. |
| program_group_id | TEXT | Yes | UUID linking layers in same tower |
| tenant_id | TEXT | No | Instance identifier (single-tenant) |
| created_at | DATETIME | No | Timestamp |
| updated_at | DATETIME | Yes | Timestamp |
| is_deleted | BOOLEAN | No | Soft-delete flag (default false) |
| deleted_at | DATETIME | Yes | When soft-deleted |
| valid_from | DATETIME | No | When this record became active |
| valid_to | DATETIME | Yes | When superseded (null = current) |
| source_document_id | TEXT FK | Yes | Document this was extracted from |
| extraction_model | TEXT | Yes | Claude model version used |
| extraction_confidence | REAL | Yes | Confidence of extraction (0-1) |

**Index**: `(account_id, line_of_business, effective_date)`

#### `insurance_coverages`

Granular coverage details within a policy. Multiple rows per policy (e.g., GL has per-occurrence, aggregate, products/completed ops, personal/advertising injury, medical payments, damage to rented premises).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| policy_id | TEXT FK | No | → insurance_policies.id |
| coverage_type | TEXT | No | "per_occurrence", "aggregate", "products_completed_ops", "personal_advertising_injury", "medical_payments", "fire_damage", "building", "contents", "business_income", "each_employee", "each_accident", "disease_policy_limit", "per_claim", "retention" |
| limit_amount | REAL | Yes | Coverage limit in dollars |
| deductible_amount | REAL | Yes | Deductible in dollars |
| deductible_type | TEXT | Yes | "per_claim", "per_occurrence", "annual_aggregate", "percentage" |
| coinsurance_pct | REAL | Yes | Coinsurance percentage |
| valuation | TEXT | Yes | "replacement_cost", "actual_cash_value", "agreed_value" |
| sublimit | REAL | Yes | Sublimit if applicable |
| description | TEXT | Yes | Free-text coverage description |
| source_document_id | TEXT FK | Yes | Document this was extracted from |
| created_at | DATETIME | No | Timestamp |

**Index**: `(policy_id, coverage_type)`

#### `insurance_claims`

Loss history from loss runs and claims reports.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| policy_id | TEXT FK | Yes | → insurance_policies.id (if identifiable) |
| account_id | TEXT FK | No | → insurance_accounts.id |
| line_of_business | TEXT | Yes | Coverage line |
| claim_number | TEXT | Yes | Carrier claim number |
| date_of_loss | DATE | Yes | When loss occurred |
| date_reported | DATE | Yes | When claim was reported |
| status | TEXT | Yes | "open", "closed", "reopened" |
| claimant | TEXT | Yes | Claimant name |
| description | TEXT | Yes | Loss description |
| paid_amount | REAL | Yes | Total paid |
| reserved_amount | REAL | Yes | Outstanding reserves |
| total_incurred | REAL | Yes | Paid + reserved |
| recovery_amount | REAL | Yes | Subrogation/salvage |
| source_document_id | TEXT FK | Yes | Loss run document ID |
| created_at | DATETIME | No | Timestamp |

**Index**: `(account_id, date_of_loss)`

#### `insurance_certificates`

Certificate of insurance tracking.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| account_id | TEXT FK | No | → insurance_accounts.id |
| certificate_type | TEXT | Yes | "acord_24", "acord_25", "acord_27", "acord_28" |
| holder_name | TEXT | Yes | Certificate holder |
| holder_address | TEXT | Yes | Holder address |
| issued_date | DATE | Yes | Date issued |
| lines_included | TEXT | Yes | JSON array of coverage lines shown |
| additional_insured | BOOLEAN | Yes | Whether holder is AI |
| waiver_of_subrogation | BOOLEAN | Yes | WOS endorsed |
| source_document_id | TEXT FK | Yes | Certificate document ID |
| created_at | DATETIME | No | Timestamp |

#### `insurance_requirements`

Coverage requirements from CC&Rs, bylaws, loan agreements.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| account_id | TEXT FK | No | → insurance_accounts.id |
| requirement_source | TEXT | No | "ccr", "bylaws", "loan_agreement", "management_agreement" |
| coverage_line | TEXT | Yes | Which line this applies to |
| requirement_text | TEXT | No | Exact text of the requirement |
| min_limit | REAL | Yes | Minimum limit required |
| min_limit_type | TEXT | Yes | "per_occurrence", "aggregate", etc. |
| is_met | BOOLEAN | Yes | Whether current coverage meets this (computed) |
| current_limit | REAL | Yes | What we currently carry (computed) |
| gap_amount | REAL | Yes | Shortfall if not met (computed) |
| source_document_id | TEXT FK | Yes | CC&R/bylaw document ID |
| created_at | DATETIME | No | Timestamp |
| last_checked_at | DATETIME | Yes | When compliance was last verified |

#### `account_documents`

Junction table linking documents to accounts.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| account_id | TEXT FK | No | → insurance_accounts.id |
| document_id | TEXT FK | No | → documents.id |
| document_role | TEXT | Yes | "policy", "certificate", "loss_run", "endorsement", "ccr", "bylaws", "reserve_study", "appraisal", "proposal", "submission", "bind_order", "bor", "unit_owner_letter", "correspondence" |
| policy_year | TEXT | Yes | e.g., "2025-26" |
| is_current | BOOLEAN | No | True if this is the active version |
| superseded_by | TEXT FK | Yes | → documents.id (newer version) |

**Composite PK**: `(account_id, document_id)`

#### `enrichment_synopses`

Claude-generated document summaries stored for fast retrieval.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| document_id | TEXT FK | No | → documents.id (unique) |
| synopsis_text | TEXT | No | 200-400 word synopsis |
| document_type | TEXT | Yes | Classified type |
| document_subtype | TEXT | Yes | e.g., "acord_24", "occurrence_policy" |
| carrier | TEXT | Yes | Extracted carrier |
| named_insured | TEXT | Yes | Extracted named insured |
| policy_period | TEXT | Yes | Extracted period |
| key_facts | TEXT | Yes | JSON array of key facts |
| enrichment_model | TEXT | Yes | "claude-sonnet-4-20250514" |
| enrichment_version | TEXT | Yes | Prompt version |
| tokens_used | INTEGER | Yes | Total tokens consumed |
| cost_usd | REAL | Yes | API cost |
| created_at | DATETIME | No | Timestamp |

#### `enrichment_entities`

Structured entities extracted per-chunk by Claude.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| document_id | TEXT FK | No | → documents.id |
| chunk_index | INTEGER | No | Which chunk |
| entity_type | TEXT | No | "insured", "insurer", "limit", "deductible", "premium", "policy_number", "effective_date", "expiration_date", "coverage_type", "exclusion", "endorsement", "claim", "address", "agent", "broker" |
| entity_value | TEXT | No | Extracted value |
| entity_context | TEXT | Yes | Surrounding context for disambiguation |
| confidence | REAL | Yes | Extraction confidence 0-1 |
| created_at | DATETIME | No | Timestamp |

**Index**: `(document_id, entity_type)`

---

## Claude Enrichment Pipeline

### Processing Flow

```
Document Upload
    │
    ▼
1. Docling Parse + Chunk (existing)
    │
    ▼
2. Claude API Call #1: Document Synopsis
   Input: First/middle/last chunks + filename + metadata
   Output: {
     synopsis, document_type, document_subtype,
     named_insured, carrier, policy_period,
     key_facts[], coverage_lines[]
   }
    │
    ▼
3. Claude API Call #2: Chunk Enrichment (batched)
   Input: Synopsis (cached prefix) + batch of 5-10 chunks
   Output per chunk: {
     summary: "Plain-English interpretation",
     entities: [
       {type: "limit", value: "$1,000,000", context: "GL per occurrence"},
       {type: "insurer", value: "Travelers", context: "D&O carrier"},
       ...
     ]
   }
    │
    ├──► SQL: Populate insurance_* tables from entities
    │         - Match/create account by named_insured
    │         - Create/update policies by line + period
    │         - Insert coverages from limit/deductible entities
    │         - Insert claims from loss run entities
    │         - Insert requirements from CC&R entities
    │
    └──► pgvector: Store enriched chunks (same PostgreSQL database)
              Format per chunk:
              "[SUMMARY] {chunk_summary}
               [ENTITIES] insured: X | insurer: Y | limit: $Z
               [ORIGINAL] {raw_chunk_text}"

              Re-embed with enriched text for better vector matching.
              Vectors stored in `chunk_vectors` table with document_id FK,
              enabling JOINs against insurance_policies, insurance_accounts, etc.
```

### Claude API Call Design

#### Call #1: Document Synopsis

```python
{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "system": [
        {
            "type": "text",
            "text": SYNOPSIS_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}  # Cache across calls
        }
    ],
    "messages": [
        {
            "role": "user",
            "content": f"""Analyze this insurance document.

Filename: {filename}
Total chunks: {chunk_count}

Representative content:
{sampled_chunks}

Return JSON:
{{
  "synopsis": "200-400 word summary...",
  "document_type": "policy|certificate|loss_run|endorsement|...",
  "document_subtype": "acord_24|occurrence|claims_made|...",
  "named_insured": "...",
  "carrier": "...",
  "policy_number": "...",
  "policy_period": "MM/DD/YYYY - MM/DD/YYYY",
  "coverage_lines": ["gl", "property", ...],
  "key_facts": ["168 residential units", "$1M/$2M GL limits", ...],
  "premium_total": 12345.00
}}"""
        }
    ]
}
```

#### Call #2: Chunk Enrichment (Batched)

```python
{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 4096,
    "system": [
        {
            "type": "text",
            "text": CHUNK_ENRICHMENT_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}
        }
    ],
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Document synopsis:\n{synopsis_text}",
                    "cache_control": {"type": "ephemeral"}  # Cache synopsis
                },
                {
                    "type": "text",
                    "text": f"""Enrich these chunks. For each, provide a plain-English
summary and extract all structured entities.

{formatted_chunk_batch}

Return JSON array, one object per chunk:
[
  {{
    "chunk_index": 0,
    "summary": "This section shows the GL coverage...",
    "entities": [
      {{"type": "limit", "value": "$1,000,000", "context": "GL per occurrence limit"}},
      {{"type": "insurer", "value": "CondoLogic", "context": "GL carrier"}}
    ]
  }}
]"""
                }
            ]
        }
    ]
}
```

### Entity-to-SQL Mapping

After Claude returns entities, the system maps them to SQL tables:

```python
ENTITY_TO_TABLE_MAP = {
    # Policy-level entities
    "insured": ("insurance_accounts", "name"),
    "address": ("insurance_accounts", "address"),
    "insurer": ("insurance_policies", "carrier"),
    "carrier": ("insurance_policies", "carrier"),
    "policy_number": ("insurance_policies", "policy_number"),
    "effective_date": ("insurance_policies", "effective_date"),
    "expiration_date": ("insurance_policies", "expiration_date"),
    "premium": ("insurance_policies", "annual_premium"),
    "program": ("insurance_policies", "program_name"),

    # Coverage-level entities
    "limit": ("insurance_coverages", "limit_amount"),
    "deductible": ("insurance_coverages", "deductible_amount"),
    "coinsurance": ("insurance_coverages", "coinsurance_pct"),
    "valuation": ("insurance_coverages", "valuation"),
    "sublimit": ("insurance_coverages", "sublimit"),

    # Claims entities
    "claim": ("insurance_claims", None),  # Complex mapping
    "loss_date": ("insurance_claims", "date_of_loss"),
    "paid": ("insurance_claims", "paid_amount"),
    "reserved": ("insurance_claims", "reserved_amount"),

    # Requirement entities
    "requirement": ("insurance_requirements", "requirement_text"),
    "min_coverage": ("insurance_requirements", "min_limit"),
}
```

### Prompt Caching Strategy

Claude's prompt caching reduces costs when processing multiple chunks from the same document:

```
Call #1 (Synopsis):    System prompt cached        → ~30% savings
Call #2a (Chunks 1-8): System + synopsis cached    → ~40% savings
Call #2b (Chunks 9-16): System + synopsis cached   → ~40% savings
Call #2c (Chunks 17-24): System + synopsis cached  → ~40% savings
```

**Estimated cost per document** (20 chunks average):
- Synopsis call: ~$0.01
- 3 chunk batch calls: ~$0.05
- **Total: ~$0.06 per document**

---

## Query Router

### Intent Classification

The query router determines whether a question should be answered from SQL or RAG:

```python
class QueryIntent(str, Enum):
    STRUCTURED = "structured"      # SQL lookup
    COMPARISON = "comparison"      # SQL join/aggregate
    ANALYTICAL = "analytical"      # RAG retrieval
    HYBRID = "hybrid"              # SQL + RAG
    CONVERSATIONAL = "conversational"  # Chat history only

STRUCTURED_PATTERNS = [
    # Direct lookups
    (r"what (?:is|are) the .+ limits?", "structured"),
    (r"what (?:is|are) the .+ deductible", "structured"),
    (r"what (?:is|are) the .+ premium", "structured"),
    (r"who is the .+ (?:carrier|insurer)", "structured"),
    (r"what is the .+ policy number", "structured"),
    (r"when does the .+ (?:expire|renew)", "structured"),
    (r"is .+ covered", "structured"),

    # Comparisons
    (r"compare .+ (?:to|with|vs|versus)", "comparison"),
    (r"how (?:do|does) .+ compare", "comparison"),
    (r"difference between", "comparison"),
    (r"(?:higher|lower|more|less) than", "comparison"),

    # Analytical (RAG)
    (r"what does the .+ (?:say|require|state)", "analytical"),
    (r"summarize", "analytical"),
    (r"explain", "analytical"),
    (r"describe", "analytical"),
    (r"what are the exclusions", "analytical"),
    (r"what are the (?:conditions|requirements)", "analytical"),
]
```

### SQL Query Generation

For structured queries, the system generates parameterized SQL:

```python
async def handle_structured_query(
    self, question: str, account_name: str | None, user_tags: list[str]
) -> StructuredAnswer:
    """Generate SQL query from natural language question."""

    # Step 1: Identify the account (from question or context)
    account = self._resolve_account(question, account_name)

    # Step 2: Identify what's being asked
    field = self._classify_field(question)  # limit, carrier, premium, etc.
    line = self._classify_line(question)    # gl, property, do, etc.

    # Step 3: Build and execute query
    if field == "limit":
        result = self.db.execute("""
            SELECT c.coverage_type, c.limit_amount, c.deductible_amount,
                   p.carrier, p.policy_number, p.effective_date, p.expiration_date
            FROM insurance_coverages c
            JOIN insurance_policies p ON c.policy_id = p.id
            WHERE p.account_id = :account_id
              AND p.line_of_business = :line
              AND p.status = 'active'
            ORDER BY c.coverage_type
        """, {"account_id": account.id, "line": line})

    # Step 4: Format response with source citation
    return StructuredAnswer(
        answer=self._format_coverage_table(result),
        source_document_id=result[0].source_document_id,
        confidence=95,  # SQL lookups are high confidence
        query_type="structured"
    )
```

### Comparison Queries

```python
async def handle_comparison_query(
    self, question: str, user_tags: list[str]
) -> StructuredAnswer:
    """Compare coverage across accounts or time periods."""

    accounts = self._extract_comparison_targets(question)
    line = self._classify_line(question)

    result = self.db.execute("""
        SELECT a.name, p.carrier, p.policy_number,
               p.effective_date, p.expiration_date, p.annual_premium,
               c.coverage_type, c.limit_amount, c.deductible_amount
        FROM insurance_coverages c
        JOIN insurance_policies p ON c.policy_id = p.id
        JOIN insurance_accounts a ON p.account_id = a.id
        WHERE a.name IN (:names)
          AND p.line_of_business = :line
          AND p.status = 'active'
        ORDER BY a.name, c.coverage_type
    """, {"names": accounts, "line": line})

    return StructuredAnswer(
        answer=self._format_comparison_table(result),
        confidence=95,
        query_type="comparison"
    )
```

### Deterministic Hybrid Routing Algorithm

All queries follow a **SQL-first, then RAG** deterministic pipeline. No LLM classification is needed to choose the path.

```
User Question
    │
    ▼
Step 1: SQL Lookup (always runs first)
    - Parse question for entity types (limit, carrier, premium, etc.)
    - Execute parameterized SQL against insurance_* tables
    - If SQL returns results with sufficient data → format answer (done)
    │
    ▼ (SQL returned empty or partial results)
Step 2: RAG Retrieval (fallback)
    - Vector search via pgvector with tag-based access control
    - Retrieve enriched chunks with source citations
    - If SQL had partial results, inject them as additional context
    │
    ▼
Step 3: LLM Response
    - Standard tier: Claude API (Haiku for simple, Sonnet for complex)
    - Enterprise tier: Ollama (local model)
    - Synthesize answer from SQL data + RAG chunks
    - Cite sources from both structured data and documents
```

**Example: "Do our current limits meet the CC&R requirements?"**

```
Step 1 (SQL): Fetch current GL limit = $1M per occurrence
              Fetch CC&R requirement = "not less than $1M per occurrence"
              → Partial answer (structured data found)
Step 2 (RAG): Retrieve CC&R text for exact language and context
Step 3 (LLM): Synthesize answer citing both structured data and document text
```

### Router SLO/SLI Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| SQL lookup latency (p95) | < 50ms | Time from query parse to SQL result |
| RAG retrieval latency (p95) | < 500ms | Time from vector search to chunk return |
| End-to-end latency — structured (p95) | < 1 second | Total time for SQL-only answers |
| End-to-end latency — analytical (p95) | < 5 seconds | Total time for RAG answers |
| End-to-end latency — hybrid (p95) | < 6 seconds | Total time for SQL + RAG answers |
| Claude API timeout | 15 seconds | Hard cutoff, fallback to cached/partial |
| Ollama timeout | 30 seconds | Hard cutoff, return "still processing" |
| Monthly query cost cap | Configurable per customer | Default $50, admin-adjustable |

**Fallback behavior**: If the Claude API is unavailable (Standard tier), the system returns cached answers for previously-asked questions and queues new questions with an estimated wait time. See **API Degradation Behavior** section.

---

## Automation Features (Phase 2)

### Renewal Summary Generation

```python
async def generate_renewal_summary(self, account_id: str) -> RenewalSummary:
    """Generate a complete renewal summary from structured data."""

    account = self.get_account(account_id)
    policies = self.get_active_policies(account_id)
    claims = self.get_claims_history(account_id, years=5)

    return RenewalSummary(
        named_insured=account.name,
        address=account.address,
        property_info={
            "units_residential": account.units_residential,
            "units_commercial": account.units_commercial,
            "year_built": account.year_built,
            "construction": account.construction_type,
        },
        coverage_schedule=[
            {
                "line": p.line_of_business,
                "carrier": p.carrier,
                "policy_number": p.policy_number,
                "effective": p.effective_date,
                "expiration": p.expiration_date,
                "premium": p.annual_premium,
                "limits": self.get_coverage_limits(p.id),
            }
            for p in policies
        ],
        total_premium=sum(p.annual_premium or 0 for p in policies),
        claims_summary={
            "total_claims": len(claims),
            "total_incurred": sum(c.total_incurred or 0 for c in claims),
            "open_claims": sum(1 for c in claims if c.status == "open"),
        },
        loss_runs_available=[p.line_of_business for p in policies],
    )
```

### Compliance Gap Detection

```python
async def check_compliance(self, account_id: str) -> list[ComplianceGap]:
    """Cross-reference coverage against requirements."""

    requirements = self.get_requirements(account_id)
    policies = self.get_active_policies(account_id)
    gaps = []

    for req in requirements:
        coverage = self.find_matching_coverage(policies, req.coverage_line)
        if not coverage:
            gaps.append(ComplianceGap(
                requirement=req,
                status="missing",
                message=f"No active {req.coverage_line} policy found"
            ))
        elif req.min_limit and coverage.limit_amount < req.min_limit:
            gaps.append(ComplianceGap(
                requirement=req,
                status="insufficient",
                current=coverage.limit_amount,
                required=req.min_limit,
                gap=req.min_limit - coverage.limit_amount,
                message=f"{req.coverage_line} limit ${coverage.limit_amount:,.0f} "
                        f"below required ${req.min_limit:,.0f}"
            ))

    return gaps
```

### Loss History Consolidation

```python
async def get_loss_summary(
    self, account_id: str, years: int = 5
) -> LossSummary:
    """Consolidated loss history across all lines."""

    cutoff = date.today() - timedelta(days=years * 365)

    claims = self.db.execute("""
        SELECT c.*, p.line_of_business, p.carrier
        FROM insurance_claims c
        LEFT JOIN insurance_policies p ON c.policy_id = p.id
        WHERE c.account_id = :account_id
          AND c.date_of_loss >= :cutoff
        ORDER BY c.date_of_loss DESC
    """, {"account_id": account_id, "cutoff": cutoff})

    by_line = defaultdict(list)
    for claim in claims:
        by_line[claim.line_of_business].append(claim)

    return LossSummary(
        total_claims=len(claims),
        total_incurred=sum(c.total_incurred or 0 for c in claims),
        open_claims=[c for c in claims if c.status == "open"],
        by_line={
            line: {
                "count": len(line_claims),
                "total_incurred": sum(c.total_incurred or 0 for c in line_claims),
            }
            for line, line_claims in by_line.items()
        },
        trend="improving" if len(claims) < 3 else "stable",
    )
```

---

## Configuration

### New Settings (config.py)

```python
# Deployment Tier
deployment_tier: Literal["standard", "enterprise"] = "enterprise"  # standard=hosted, enterprise=air-gapped

# Claude Enrichment (both tiers)
claude_enrichment_enabled: bool = False          # Master toggle
claude_api_key: str | None = None                # Anthropic API key
claude_enrichment_model: str = "claude-sonnet-4-20250514"
claude_enrichment_batch_size: int = 8            # Chunks per API call
claude_enrichment_max_retries: int = 3           # Retry on API failure
claude_enrichment_timeout: int = 60              # Seconds per call
claude_enrichment_cost_limit_usd: float = 10.0   # Daily cost cap

# Claude Query-Time (Standard tier only)
claude_query_enabled: bool = False               # True for Standard tier
claude_query_model_simple: str = "claude-haiku-4-5-20251001"   # Simple lookups
claude_query_model_complex: str = "claude-sonnet-4-20250514"   # Analysis queries
claude_query_cost_limit_usd: float = 50.0        # Monthly query cost cap

# Structured Query Router
structured_query_enabled: bool = True            # Enable SQL-first routing
structured_query_confidence: int = 95            # Confidence for SQL answers

# Insurance Schema
insurance_schema_enabled: bool = True            # Enable insurance tables
insurance_auto_link_accounts: bool = True        # Auto-match docs to accounts

# Database — PostgreSQL + pgvector for both tiers
database_url: str = "postgresql://localhost/vaultiq"  # Per-customer database
pgvector_dimension: int = 768                    # Embedding dimension
pgvector_index_type: str = "ivfflat"             # ivfflat or hnsw (hnsw better at scale)
pgvector_lists: int = 100                        # IVFFlat lists (tune per collection size)
pgvector_probes: int = 10                        # IVFFlat probes (accuracy vs speed)

# Legacy (backward compat for existing enterprise deployments)
database_backend: Literal["sqlite", "postgresql"] = "postgresql"  # Default changed to PG
```

### Profile Defaults

```python
PROFILE_DEFAULTS = {
    "laptop": {
        # Development — minimal, SQLite for easy setup
        "claude_enrichment_enabled": False,
        "claude_query_enabled": False,
        "structured_query_enabled": True,
        "insurance_schema_enabled": True,
        "database_backend": "sqlite",            # SQLite for dev convenience
        "vector_backend": "chroma",              # Lightweight dev vector store
    },
    "spark": {
        # Enterprise tier — air-gapped, Ollama for queries
        "deployment_tier": "enterprise",
        "claude_enrichment_enabled": True,
        "claude_query_enabled": False,           # Ollama handles queries
        "structured_query_enabled": True,
        "insurance_schema_enabled": True,
        "database_backend": "postgresql",        # PostgreSQL + pgvector
        "vector_backend": "pgvector",            # Unified with SQL
    },
    "hosted": {
        # Standard tier — single-tenant cloud, Claude for everything
        "deployment_tier": "standard",
        "claude_enrichment_enabled": True,
        "claude_query_enabled": True,            # Claude handles queries
        "structured_query_enabled": True,
        "insurance_schema_enabled": True,
        "database_backend": "postgresql",        # PostgreSQL + pgvector
        "vector_backend": "pgvector",            # Unified with SQL
        # No Ollama needed
        "chat_model": None,                      # Not used — Claude answers queries
        "rag_enable_hallucination_check": False,  # Claude self-validates
    },
}
```

---

## Document Model Changes

Add to `documents` table:

| Column | Type | Description |
|--------|------|-------------|
| enrichment_status | TEXT | null, "pending", "enriching", "completed", "failed" |
| enrichment_model | TEXT | Claude model used |
| enrichment_version | TEXT | Prompt version for re-enrichment tracking |
| enrichment_tokens_used | INTEGER | Total tokens consumed |
| enrichment_cost_usd | REAL | API cost for this document |
| enrichment_completed_at | DATETIME | When enrichment finished |
| insurance_account_id | TEXT FK | → insurance_accounts.id |
| document_role | TEXT | "policy", "certificate", "loss_run", etc. |

---

## CLI Commands

### Batch Enrichment

```bash
# Enrich all unenriched documents
python -m ai_ready_rag.cli enrich --all

# Enrich specific document
python -m ai_ready_rag.cli enrich --document-id <uuid>

# Re-enrich with updated prompts
python -m ai_ready_rag.cli enrich --re-enrich --version v2

# Dry run (show what would be enriched, estimate cost)
python -m ai_ready_rag.cli enrich --dry-run

# Export for air-gap transfer (signed manifest + checksum)
python -m ai_ready_rag.cli transfer export --output /path/to/transfer/
# Exports: pg_dump (SQL + pgvector data) + signed manifest + SHA-256 checksums

# Verify transfer integrity
python -m ai_ready_rag.cli transfer verify --input /path/to/transfer/

# Import on air-gapped Spark (atomic with rollback)
python -m ai_ready_rag.cli transfer import --input /path/to/transfer/
```

### Insurance Data Queries

```bash
# List all accounts
python -m ai_ready_rag.cli accounts list

# Show account details
python -m ai_ready_rag.cli accounts show "Marshall Wells Lofts"

# Check compliance
python -m ai_ready_rag.cli compliance check --account "Marshall Wells Lofts"

# Generate renewal summary
python -m ai_ready_rag.cli renewal summary --account "Marshall Wells Lofts"
```

---

## Integration Points with Existing Code

### ProcessingService Changes

```python
# ai_ready_rag/services/processing_service.py

async def process_document(self, document_id: str, ...):
    # ... existing Docling parsing ...

    # NEW: Claude enrichment (replaces SummaryGenerator + DocumentClassifier)
    if settings.claude_enrichment_enabled:
        enrichment = await self.claude_enrichment_service.enrich_document(
            document_id=document_id,
            chunks=chunks,
            filename=document.original_filename,
        )

        # Store structured data in SQL
        await self.insurance_data_service.ingest_entities(
            document_id=document_id,
            synopsis=enrichment.synopsis,
            entities=enrichment.entities,
        )

        # Use enriched chunks for vector storage
        chunks_for_indexing = enrichment.enriched_chunks
    else:
        # Fallback: existing Ollama pipeline
        chunks_for_indexing = chunks

    # Index to pgvector (same PostgreSQL database)
    await self.vector_service.add_document(
        document_id=document_id,
        chunks=chunks_for_indexing,
        tags=tags,
    )
```

### RAGService Changes

```python
# ai_ready_rag/services/rag_service.py

async def generate(self, request: RAGRequest) -> RAGResponse:
    # NEW: Check query router first
    if settings.structured_query_enabled:
        intent = self.query_router.classify(request.query)

        if intent in (QueryIntent.STRUCTURED, QueryIntent.COMPARISON):
            return await self.handle_structured_query(request)
        elif intent == QueryIntent.HYBRID:
            sql_context = await self.get_sql_context(request)
            # Inject SQL data into RAG context
            request.additional_context = sql_context

    # Tier-dependent LLM call
    if settings.claude_query_enabled:
        # STANDARD TIER: Claude API for response generation
        model = self.model_router.select_model(request.query, intent)
        return await self._claude_generate(request, model=model)
    else:
        # ENTERPRISE TIER: Ollama local model (existing pipeline)
        return await self._ollama_generate(request)
```

### Claude Query Service (Standard Tier)

```python
# ai_ready_rag/services/claude_query_service.py

class ClaudeQueryService:
    """Query-time Claude API integration for Standard tier."""

    async def generate_response(
        self,
        query: str,
        context_chunks: list[str],
        sql_data: dict | None,
        chat_history: list[dict],
        model: str = "claude-haiku-4-5-20251001",
    ) -> ClaudeQueryResponse:
        """Generate a response using Claude API."""

        messages = [
            {
                "role": "user",
                "content": self._build_prompt(
                    query=query,
                    context=context_chunks,
                    sql_data=sql_data,
                    history=chat_history,
                ),
            }
        ]

        response = await self.client.messages.create(
            model=model,
            max_tokens=2048,
            system=INSURANCE_QUERY_SYSTEM_PROMPT,
            messages=messages,
        )

        return ClaudeQueryResponse(
            answer=response.content[0].text,
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost=self._calculate_cost(model, response.usage),
        )
```

---

## New Files to Create

| File | Purpose | Tier |
|------|---------|------|
| `ai_ready_rag/services/claude_enrichment_service.py` | Claude API integration, batch enrichment | Both |
| `ai_ready_rag/services/claude_query_service.py` | Claude query-time answering | Standard only |
| `ai_ready_rag/services/claude_model_router.py` | Haiku/Sonnet model selection | Standard only |
| `ai_ready_rag/services/insurance_data_service.py` | Insurance SQL CRUD, entity-to-table mapping | Both |
| `ai_ready_rag/services/query_router.py` | Deterministic SQL-first routing, SQL query generation | Both |
| `ai_ready_rag/services/pgvector_service.py` | pgvector integration (replaces Qdrant vector_service) | Both |
| `ai_ready_rag/services/review_service.py` | Low-confidence review workflow | Both |
| `ai_ready_rag/db/models/insurance.py` | SQLAlchemy models for insurance tables | Both |
| `ai_ready_rag/db/models/vectors.py` | SQLAlchemy model for chunk_vectors (pgvector) | Both |
| `ai_ready_rag/db/models/review.py` | Review queue model for low-confidence items | Both |
| `ai_ready_rag/db/models/provenance.py` | Provenance/audit trail models | Both |
| `ai_ready_rag/db/migrations/` | Alembic migration scripts | Both |
| `ai_ready_rag/schemas/insurance.py` | Pydantic schemas for insurance data | Both |
| `ai_ready_rag/api/insurance.py` | REST endpoints for insurance data | Both |
| `ai_ready_rag/api/review.py` | REST endpoints for review queue | Both |
| `ai_ready_rag/cli/enrich.py` | CLI commands for enrichment | Both |
| `ai_ready_rag/cli/insurance.py` | CLI commands for insurance data | Both |
| `ai_ready_rag/cli/transfer.py` | Air-gap export/import with signed manifests | Enterprise only |
| `docker-compose.customer.yml` | Per-customer Docker Compose template | Standard only |
| `scripts/provision_customer.sh` | Automated customer instance provisioning | Standard only |
| `tests/eval/gold_set.json` | Gold evaluation set (16+ questions with expected answers) | Both |
| `tests/eval/eval_runner.py` | Automated evaluation harness | Both |

---

## Migration & Rollout

### Phase 1: Core Platform (Weeks 1-4)

1. **Week 1**: PostgreSQL + pgvector migration + Insurance SQL schema
   - Migrate from Qdrant to pgvector (new `chunk_vectors` table with vector column)
   - Create SQLAlchemy models for insurance tables (shared across tiers)
   - Implement pgvector service replacing existing VectorService
   - Alembic migration scripts for schema management
   - Gold evaluation harness (16+ Marshall Wells questions, automated scoring, CI gates)

2. **Week 2**: Claude enrichment + entity-to-SQL mapping
   - Implement Claude API client with prompt caching
   - Entity extraction prompt engineering and testing
   - Build entity mapping pipeline (Claude entities → SQL rows)
   - Named insured fuzzy matching for account linking
   - Data lifecycle: soft-delete, versioning, cascade rules, re-enrichment idempotency
   - Provenance tracking (valid_from/valid_to, extraction metadata, source chain)

3. **Week 3**: Query router + Standard tier services
   - Deterministic SQL-first query router
   - SQL execution safety layer (allowlisted templates, parameterization, timeouts, row caps)
   - Claude query-time service with Haiku/Sonnet model routing
   - Query cost tracking and configurable caps
   - Low-confidence review workflow (threshold-based routing to human review queue)
   - Privacy controls (PII detection, prompt redaction, retention policies)

4. **Week 4**: Integration testing + deployment
   - Wire enrichment into ProcessingService (both tiers)
   - Wire query router into RAGService (tier-aware)
   - Test with Marshall Wells Lofts documents (both tiers)
   - Air-gap transfer CLI with signed manifests, checksums, atomic rollback
   - Single-tenant Docker Compose template + provisioning script
   - API degradation behavior (cache TTL, user notification, fallback)

### Phase 2: Automation (Weeks 5-6)

5. **Week 5**: Automation features
   - Renewal summary generation from structured data
   - Compliance gap detection (coverage vs. CC&R requirements)
   - Loss history consolidation and trend analysis

6. **Week 6**: Polish + documentation
   - API endpoints for insurance data + review queue
   - Frontend dashboard components
   - Per-customer isolation verification testing
   - Document taxonomy versioning (fixed enum, unknown handling)
   - Deployment documentation for both tiers
   - Cost control dashboards (enrichment + query spend per customer)

---

## Cost Estimate

### Per-Document Enrichment Cost (Both Tiers)

| Component | Input Tokens | Output Tokens | Cost |
|-----------|-------------|---------------|------|
| Synopsis call | ~2,000 | ~500 | ~$0.01 |
| Chunk batch (×3 calls, 8 chunks each) | ~8,000 | ~2,000 | ~$0.05 |
| **Total per document** | ~10,000 | ~2,500 | **~$0.06** |

With prompt caching (~35% input savings): **~$0.04 per document**

### Portfolio Scale — Enrichment (One-Time)

| Portfolio Size | Documents | Enrichment Cost | Time (sequential) |
|---|---|---|---|
| Single property | 20-50 | $1-3 | 5-15 min |
| 10 properties | 200-500 | $12-30 | 1-2 hours |
| 50 properties | 1,000-2,500 | $60-150 | 4-10 hours |
| 100 properties | 2,000-5,000 | $120-300 | 8-20 hours |

### Query Costs — Standard Tier Only

| Query Type | Model | Cost/Query | Example |
|---|---|---|---|
| Simple lookup | Haiku | ~$0.001 | "What are the GL limits?" |
| Formatted response | Haiku | ~$0.002 | "List all carriers for Marshall Wells" |
| Comparison | Sonnet | ~$0.015 | "Compare GL limits across all properties" |
| Deep analysis | Sonnet | ~$0.03 | "Does our coverage meet CC&R requirements?" |

**Estimated monthly query costs (Standard tier):**

| Usage Level | Queries/Day | Haiku % | Sonnet % | Monthly Cost |
|---|---|---|---|---|
| Light (1 agent) | 20 | 70% | 30% | ~$5 |
| Moderate (3 agents) | 75 | 70% | 30% | ~$18 |
| Heavy (10 agents) | 250 | 70% | 30% | ~$60 |

### Total Monthly Cost Comparison (Per Customer)

| | Standard (Hosted) | Enterprise (Air-Gapped) |
|---|---|---|
| Hardware | $0 | ~$3,000+ (one-time) |
| Infrastructure | $30-150/mo (cloud VM + PostgreSQL) | $0 (local) |
| Enrichment | Same (one-time per doc) | Same (one-time per doc) |
| Query costs | $5-60/mo (Claude API) | $0 (Ollama, local) |
| **Total monthly** | **$35-210/mo** | **$0 after hardware** |
| **Answer quality** | **Highest** | **Good** |

Note: Standard tier infrastructure cost is lower than previous estimates because pgvector eliminates the need for a separate Qdrant service. Single PostgreSQL instance handles all data.

---

## Risk Flags

### API_DEPENDENCY Risk
Enterprise tier: Claude API required for enrichment only, NOT for queries.
Standard tier: Claude API required for both enrichment AND queries.
- [ ] Enterprise: graceful fallback to Ollama pipeline when API unavailable for enrichment
- [ ] Standard: graceful degradation when Claude API is down (queue queries, show cached results)
- [ ] Cost tracking with configurable daily/monthly caps (separate for enrichment vs. queries)
- [ ] Batch retry logic for API failures
- [ ] Standard tier: rate limiting to prevent runaway query costs

### SCHEMA_MIGRATION Risk
Insurance tables and pgvector added to PostgreSQL.
- [ ] Alembic migration scripts that preserve existing data
- [ ] Insurance tables are additive (no existing table changes)
- [ ] Re-enrichment doesn't duplicate SQL rows (upsert logic)
- [ ] pgvector extension auto-created on first migration

### ENTITY_ACCURACY Risk
Claude entity extraction may produce incorrect values.
- [ ] Confidence scoring per entity
- [ ] Human review flag for low-confidence extractions
- [ ] Source document citation for every SQL row

### DATA_STALENESS Risk
SQL data could become stale if documents are re-uploaded.
- [ ] Track source_document_id on every row
- [ ] When document is deleted/replaced, cascade to SQL data
- [ ] Re-enrichment updates existing rows vs. creating duplicates

### SINGLE_TENANT_OPS Risk
Each customer gets their own instance — operational complexity scales linearly.
- [ ] Automated provisioning script (database, app, DNS, TLS)
- [ ] Rolling update strategy across customer instances
- [ ] Per-customer backup and restore procedures
- [ ] Monitoring dashboard aggregating all instances
- [ ] Customer-specific configuration management

---

## Acceptance Criteria

### Phase 1 — Both Tiers

- [ ] Claude enrichment processes a document and populates insurance SQL tables
- [ ] All vectors stored in pgvector (same PostgreSQL database as insurance tables)
- [ ] SQL JOIN queries across insurance tables and vector results work correctly
- [ ] Structured query ("What are the GL limits?") returns answer from SQL in <500ms
- [ ] Deterministic router: SQL runs first, RAG only if SQL insufficient
- [ ] Cost tracking shows per-document and cumulative API spend
- [ ] Existing documents can be batch-enriched via CLI
- [ ] Fallback to Ollama pipeline works when Claude enrichment API unavailable
- [ ] Gold evaluation harness: 16+ questions with automated scoring and rubric
- [ ] Evaluation gates: all gold-set questions pass minimum confidence threshold
- [ ] Data lifecycle: soft-delete works, document replacement cascades to SQL data
- [ ] Provenance: every SQL row tracks source_document_id, extraction metadata, valid_from/valid_to
- [ ] Low-confidence answers route to review queue (not shown to user as definitive)
- [ ] SQL queries use allowlisted templates with parameterization (no raw SQL generation)
- [ ] PII fields encrypted at rest, not included in LLM prompts

### Phase 1 — Standard Tier

- [ ] Claude API answers queries with Haiku/Sonnet model routing
- [ ] All 16 demo questions return confidence >90 (Claude query quality)
- [ ] Single-tenant: each customer instance is fully isolated (separate database, app, config)
- [ ] Docker Compose template deploys a new customer instance
- [ ] Provisioning script automates database creation, DNS, TLS
- [ ] Monthly query cost tracking with configurable per-customer cap
- [ ] Graceful degradation when Claude query API is unavailable (cached results, queue)
- [ ] Per-customer cost dashboards (enrichment spend + query spend)

### Phase 1 — Enterprise Tier

- [ ] RAG query ("What does the CC&R require?") returns enriched-chunk answer with confidence >70
- [ ] All 16 demo questions return confidence >70 (up from 30) via Ollama + enriched chunks
- [ ] Enrichment works in both deployment modes (workstation transfer + direct Spark)
- [ ] Air-gap transfer: signed manifest + SHA-256 checksums + atomic import with rollback
- [ ] Transfer verify command catches corruption before import

### Phase 2 — Both Tiers

- [ ] Renewal summary generated from SQL data for any account
- [ ] Compliance gap detection identifies coverage shortfalls vs. CC&R requirements
- [ ] Loss history query returns consolidated 5-year view across all lines
- [ ] Cross-account comparison query works ("Compare GL limits across all properties")
- [ ] Document taxonomy versioning: fixed enum with unknown handling and F1 tracking

---

## Design Decisions (Resolved)

1. **Account matching heuristic**: Named insured fuzzy match (>90% string similarity) with manual override option. Claude extracts the named insured during synopsis generation; system matches against existing `insurance_accounts.name` using difflib.SequenceMatcher. If no match above threshold, creates new account. Agent can reassign via UI.

2. **Historical data retention**: Track all policies, all years. Every policy period ever uploaded gets a row in `insurance_policies` with appropriate `status` (active/expired/cancelled). Enables premium trending, coverage change history, and complete renewal prep across years.

3. **Multi-carrier layered programs**: Separate policies with `layer_position` column. Each layer is its own row in `insurance_policies` with `layer_position` (1=primary, 2=first excess, etc.) and `program_group_id` (UUID linking layers in the same tower). Clean and queryable per-layer or per-program.

4. **Certificate handling**: Phase 1 = track only. Record certificates found during document ingestion (holder, lines, dates, AI/WOS status). No PDF generation — that's Phase 3 scope.

5. **Target audience**: Agency principals. Primary pitch is efficiency + scale: "Your team handles 2x the book without hiring."

6. **Pricing in presentations**: Discuss in person only. No pricing in written materials.

7. **Demo data**: Use Marshall Wells Lofts real data for demos and presentations.

8. **Market positioning**: Complement to AMS. "Your AMS tracks transactions. VaultIQ understands your documents." Non-threatening to existing Epic/HawkSoft/Applied workflows.

9. **Hybrid routing strategy**: Deterministic SQL-first, then RAG. No LLM-based intent classification to choose the path. SQL always runs first; if it returns sufficient data, answer is formatted directly. If SQL returns empty/partial, RAG retrieval runs as fallback. Simpler, faster, more predictable than LLM-routed approaches.

10. **Tenant model**: Single-tenant hosted. Each customer gets their own isolated application instance (FastAPI + PostgreSQL + pgvector). More expensive to operate than multi-tenant, but provides: complete data isolation, per-customer customization capability, simpler security model, independent scaling and backup.

11. **Vector database**: PostgreSQL + pgvector (replacing Qdrant). Single database for structured data AND vectors. Eliminates Qdrant dependency. Enables SQL JOINs between vector similarity results and insurance tables. At VaultIQ's per-customer scale (5K-50K vectors), pgvector performance is equivalent to dedicated vector databases. IVFFlat index for most customers; HNSW for large portfolios (>20K vectors).

---

## Engineering Review Addenda

The following sections address gaps identified during engineering review.

### Data Lifecycle Management

**Soft-delete**: All insurance_* tables use `is_deleted` flag + `deleted_at` timestamp. Soft-deleted rows are excluded from queries by default but remain for audit/recovery. Hard-delete after configurable retention period (default 90 days).

**Versioning**: When a document is re-uploaded or re-enriched:
1. Existing SQL rows from that document are marked with `superseded_at` timestamp
2. New rows are created with `valid_from` = now
3. `enrichment_version` tracks which prompt version generated the data
4. Re-enrichment is idempotent: same prompt version + same document = no-op

**Cascade rules**:
- Document deleted → soft-delete all insurance_* rows with that `source_document_id`
- Account deleted → soft-delete all policies, coverages, claims, certificates, requirements
- Policy deleted → soft-delete all coverages, claims linked to that policy

### Gold Evaluation Harness

A fixed set of 16+ questions with expected answers, graded automatically:

```json
{
  "questions": [
    {
      "id": "gl_limit",
      "question": "What are the GL limits for Marshall Wells?",
      "expected_answer": "$1,000,000 per occurrence / $2,000,000 aggregate",
      "expected_source": "structured",
      "min_confidence": 90,
      "rubric": {
        "exact_match_fields": ["per_occurrence", "aggregate"],
        "must_cite_source": true
      }
    }
  ]
}
```

**Evaluation scoring**:
- Factual accuracy (exact match on key fields): 50%
- Source citation present and correct: 25%
- Confidence score above threshold: 15%
- Latency within SLO: 10%

**CI gates**: Evaluation runs automatically on every PR that touches RAG, query router, or enrichment code. PR blocked if any gold-set question drops below minimum score.

### Low-Confidence Review Workflow

When the system produces an answer with confidence below the review threshold:

```
Answer confidence < 70 (configurable)
    │
    ▼
Route to review queue (not shown to user as definitive)
    │
    ▼
User sees: "I found some relevant information but I'm not fully confident
           in this answer. [Show tentative answer] [Flag for review]"
    │
    ├── User accepts → answer logged as "user_accepted"
    ├── User flags → added to review queue for admin
    └── Admin reviews → corrects answer → correction stored for training
```

**Review queue table** (`review_items`):

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | UUID |
| query | TEXT | Original question |
| tentative_answer | TEXT | System's best answer |
| confidence | REAL | System confidence score |
| review_status | TEXT | "pending", "accepted", "corrected", "dismissed" |
| corrected_answer | TEXT | Admin-provided correction |
| reviewer_id | TEXT FK | → users.id |
| created_at | DATETIME | When flagged |
| resolved_at | DATETIME | When reviewed |

### Provenance and Audit Schema

Every extracted entity and SQL row maintains a full audit trail:

| Column (added to insurance_* tables) | Type | Description |
|---------------------------------------|------|-------------|
| valid_from | DATETIME | When this record became active |
| valid_to | DATETIME | When this record was superseded (null = current) |
| source_document_id | TEXT FK | Document that produced this record |
| extraction_model | TEXT | Claude model version used |
| extraction_prompt_version | TEXT | Prompt template version |
| extraction_confidence | REAL | Claude's reported confidence |
| correction_history | TEXT | JSON array of admin corrections |

**Query pattern**: To get current records, always filter `WHERE valid_to IS NULL AND is_deleted = FALSE`.

### SQL Execution Safety

All SQL queries against insurance tables use a safety layer:

1. **Allowlisted templates**: Only pre-defined query templates execute. No arbitrary SQL generation.
2. **Parameterization**: All user-provided values passed via bind parameters, never string interpolation.
3. **Timeouts**: 5-second query timeout. Queries exceeding this are killed.
4. **Row caps**: Maximum 1,000 rows returned per query. Pagination required for larger result sets.
5. **Read-only**: Query-time SQL uses a read-only database connection. Write operations only during enrichment.

```python
ALLOWED_QUERY_TEMPLATES = {
    "coverage_by_account_line": """
        SELECT c.coverage_type, c.limit_amount, c.deductible_amount,
               p.carrier, p.policy_number
        FROM insurance_coverages c
        JOIN insurance_policies p ON c.policy_id = p.id
        WHERE p.account_id = :account_id
          AND p.line_of_business = :line
          AND p.status = 'active'
          AND p.valid_to IS NULL
        ORDER BY c.coverage_type
        LIMIT :row_cap
    """,
    # ... additional templates ...
}
```

### Privacy and PII Controls

**Encryption at rest**: PII fields (named insured, addresses, claimant names) stored encrypted using Fernet symmetric encryption. Decrypted only when needed for display or query matching.

**Prompt redaction**: When sending context to Claude (query-time, Standard tier):
- SSN, EIN, bank account numbers are redacted from context chunks
- Claimant names replaced with anonymized identifiers in prompts
- Full context available in SQL response (not sent to Claude)

**Retention policies**:
- Chat history: configurable per-customer (default 90 days)
- Enrichment data: retained as long as source document exists
- Audit logs: 1 year minimum, configurable
- Review queue items: retained indefinitely for training data

**Data export/delete**: Per-customer data export (GDPR-style) and full customer data deletion available via admin CLI.

### Air-Gap Transfer Protocol

For Enterprise tier workstation-to-Spark transfers:

**Export format**:
```
transfer_bundle/
├── manifest.json          # Signed manifest with metadata
├── manifest.sig           # HMAC-SHA256 signature
├── database.sql.gz        # pg_dump of all tables (compressed)
├── database.sql.gz.sha256 # SHA-256 checksum
└── metadata.json          # Export timestamp, versions, row counts
```

**Manifest schema**:
```json
{
  "version": "1.0",
  "exported_at": "2026-02-20T10:00:00Z",
  "source_instance": "workstation-abc",
  "target_instance": "spark-001",
  "database_version": "v1.1",
  "tables": {
    "insurance_accounts": {"rows": 5, "checksum": "sha256:abc..."},
    "insurance_policies": {"rows": 23, "checksum": "sha256:def..."},
    "chunk_vectors": {"rows": 1200, "checksum": "sha256:ghi..."}
  },
  "total_size_bytes": 12345678
}
```

**Import process**:
1. Verify manifest signature (shared secret between workstation and Spark)
2. Verify all file checksums match manifest
3. Begin transaction
4. Import database dump (pg_restore)
5. Verify row counts match manifest
6. Commit transaction (or rollback on any failure)

### API Degradation Behavior

When the Claude API is unavailable (Standard tier):

| Scenario | Behavior | User Experience |
|----------|----------|-----------------|
| Claude API timeout (>15s) | Return cached answer if available | "Here's what I found previously..." |
| Claude API down | Queue query, return estimate | "System is processing, ETA ~2 min" |
| Cost cap reached | Restrict to SQL-only answers | "Detailed analysis unavailable, here are the facts from our database" |
| Partial outage | Haiku available, Sonnet down | Route all queries through Haiku with quality warning |

**Cache TTL**: Answers cached for 24 hours (configurable). Cache key = normalized query + account context.

**SLO**: 99.5% availability for SQL-only answers (no external dependency). 99% availability for full RAG answers (depends on Claude API uptime).

### Document Taxonomy Versioning

Document types use a **fixed enum** with explicit versioning:

```python
class DocumentType(str, Enum):
    """v1.0 — Insurance document taxonomy."""
    POLICY = "policy"
    CERTIFICATE = "certificate"
    LOSS_RUN = "loss_run"
    ENDORSEMENT = "endorsement"
    CCR = "ccr"
    BYLAWS = "bylaws"
    RESERVE_STUDY = "reserve_study"
    APPRAISAL = "appraisal"
    PROPOSAL = "proposal"
    SUBMISSION = "submission"
    BIND_ORDER = "bind_order"
    BOR = "bor"                    # Broker of record letter
    UNIT_OWNER_LETTER = "unit_owner_letter"
    CORRESPONDENCE = "correspondence"
    UNKNOWN = "unknown"            # Catch-all for unrecognized types
```

**Unknown handling**: Documents classified as `UNKNOWN` are flagged for admin review. System tracks classification F1 score per document type. Target: >0.90 F1 for common types (policy, certificate, loss_run).

**Versioning**: Taxonomy version stored in `enrichment_synopses.enrichment_version`. When taxonomy changes, re-enrichment only targets documents whose `enrichment_version` differs from current.

### Cost Controls

**Hard stops and alerts**:

| Control | Default | Behavior |
|---------|---------|----------|
| Daily enrichment cost cap | $10 | Enrichment paused, admin notified, queue preserved |
| Monthly query cost cap | $50 | Complex queries restricted, SQL-only mode |
| Per-document cost limit | $1 | Document flagged as "too expensive", admin override required |
| Alert at 80% of cap | — | Admin notification via email/dashboard |
| Alert at 95% of cap | — | Warning banner in UI |

**Admin override**: Admin can temporarily lift cost caps via dashboard or CLI (`python -m ai_ready_rag.cli costs override --daily-limit 50`).

**Cost tracking granularity**: Per-document enrichment cost, per-query cost, daily/monthly aggregates, per-customer totals (Standard tier).

---

## pgvector Schema

### `chunk_vectors` Table

Replaces the Qdrant collection. Stored in the same PostgreSQL database as all other tables.

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | UUID |
| document_id | TEXT FK | → documents.id |
| chunk_index | INTEGER | Position within document |
| chunk_text | TEXT | Original chunk text |
| enriched_text | TEXT | [SUMMARY] + [ENTITIES] + [ORIGINAL] format |
| embedding | vector(768) | nomic-embed-text embedding of enriched_text |
| metadata | JSONB | Tags, tenant_id, document_type, etc. |
| created_at | TIMESTAMP | Insertion time |

**Indexes**:
- `CREATE INDEX ON chunk_vectors USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);`
- `CREATE INDEX ON chunk_vectors (document_id);`
- `CREATE INDEX ON chunk_vectors USING gin (metadata);` — for tag-based filtering

**Vector search query** (with access control):
```sql
SELECT cv.id, cv.chunk_text, cv.enriched_text,
       cv.embedding <=> :query_embedding AS distance,
       d.original_filename, d.id AS doc_id
FROM chunk_vectors cv
JOIN documents d ON cv.document_id = d.id
JOIN document_tags dt ON d.id = dt.document_id
JOIN tags t ON dt.tag_id = t.id
WHERE t.name = ANY(:user_tags)
ORDER BY cv.embedding <=> :query_embedding
LIMIT :top_k;
```

**Cross-table JOIN** (vector search + insurance data):
```sql
SELECT cv.enriched_text, cv.embedding <=> :query_embedding AS distance,
       p.carrier, p.policy_number, p.line_of_business,
       c.coverage_type, c.limit_amount
FROM chunk_vectors cv
JOIN insurance_policies p ON cv.document_id = p.source_document_id
LEFT JOIN insurance_coverages c ON p.id = c.policy_id
WHERE cv.embedding <=> :query_embedding < :max_distance
ORDER BY distance
LIMIT :top_k;
```

### Migration from Qdrant

For existing deployments currently using Qdrant:

1. Export all vectors from Qdrant collection via API
2. Create `chunk_vectors` table with pgvector extension
3. Bulk insert vectors with metadata mapping
4. Verify row counts and sample vector distances match
5. Update `vector_backend` config to `pgvector`
6. Deprecate Qdrant dependency

Existing Qdrant-based deployments (Enterprise tier already in production) will be supported during a transition period. New deployments default to pgvector.

---

**Next Steps**:
1. Review this spec (v1.1 — includes pgvector migration, single-tenant architecture, engineering review items)
2. Run `/spec-review specs/INSURANCE_AI_PLATFORM_v1.md`
