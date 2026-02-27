---
title: "VaultIQ — Intelligent Insurance Document Management Platform"
status: DRAFT
version: v1.1
created: 2026-02-27
updated: 2026-02-27
author: —
type: Fullstack
complexity: COMPLEX
supersedes: INSURANCE_AI_PLATFORM_v1.md, CLAUDE_ENRICHMENT_PIPELINE_v1.md
review_notes: v1.1 incorporates engineering review feedback — 5 P0 gaps, 5 P1 gaps, 3 P2 gaps resolved
---

# VaultIQ — Intelligent Insurance Document Management Platform

## Product Vision

VaultIQ is a purpose-built AI platform that transforms how independent insurance agencies manage documents and access knowledge across their book of business. By combining intelligent document ingestion, structured data extraction, and natural-language querying, VaultIQ gives every agent in an agency instant access to every fact across every property they manage.

**The core problem**: Insurance agents manage hundreds of documents per property — policies, certificates, loss runs, endorsements, CC&Rs, reserve studies. Today, agents manually open 3-4 PDFs to answer a single question about limits or carriers. Policy details live inside PDF text, not queryable databases. Renewal prep takes 2-3 hours per property. Compliance checking is manual and often skipped.

**The solution**: A dual-path system where structured facts (policy numbers, limits, premiums, carriers, dates) are extracted once during ingestion and stored in SQL tables for instant deterministic lookup, while unstructured content is enriched with AI-generated summaries and entities for high-quality semantic search. A smart query router directs each question to the optimal path.

**The market**: 39,000 independent P&C agencies in the US. 23,000 in VaultIQ's serviceable market across six verticals. No incumbent provides AI-powered document intelligence for independent agencies — AMS platforms (Applied Epic, Vertafore, HawkSoft) track transactions but cannot read, understand, or analyze document content. VaultIQ creates a new category: **Insurance Knowledge Intelligence**.

---

## Product Tiers

VaultIQ is offered in two deployment tiers that share the same core:

| Capability | Standard (Hosted) | Enterprise (Air-Gapped) |
|---|---|---|
| **Deployment** | Cloud VM per customer (Docker) | DGX Spark on-prem |
| **Tenant Model** | Single-tenant (isolated instance per agency) | Single agency |
| **Ingestion LLM** | Claude API (Sonnet) — primary | Claude API (Sonnet) — primary |
| **Query-time LLM** | Claude API (Haiku/Sonnet) — primary | Ollama (local 8B) — primary |
| **Ollama Role** | Fallback when Claude API unavailable | Primary query engine |
| **Database** | PostgreSQL + pgvector | PostgreSQL + pgvector |
| **Internet Required** | Always (for queries) | Only during ingestion (Mode B, v1.0) |
| **Answer Quality** | Highest (Claude) | Good (enriched chunks + Ollama) |
| **Data Location** | Encrypted cloud (US), customer-isolated | Customer's office |
| **Customization** | Full tenant customization layer | Full local control |
| **Pricing** | Monthly subscription | Hardware + license |
| **Setup Time** | Same day (automated provisioning) | Hardware procurement + install |

---

## Model Governance

### Pinned Model IDs

All code, configuration, and acceptance criteria **must** use exact pinned provider model IDs. Aliases, short names, and generic references are forbidden in any production code path, config file, or test assertion.

| Role | Tier | Pinned Model ID |
|---|---|---|
| Document enrichment (ingestion) | Both | `claude-sonnet-4-6` |
| Query — simple structured lookups | Standard | `claude-haiku-4-5-20251001` |
| Query — comparison and analytical | Standard | `claude-sonnet-4-6` |
| Query — primary | Enterprise | `qwen3-rag` (local Ollama model) |
| Query — fallback | Standard | `qwen3-rag` (local Ollama model) |

### Governance Rules

1. **Only pinned IDs in code**: `claude-sonnet-4-6` is the Sonnet model for this codebase. Do not use `claude-sonnet`, `sonnet`, or dated variants like `claude-sonnet-4-20250514` anywhere in code, config, or tests.
2. **Model upgrades require a spec change**: Changing a model ID requires updating this table, bumping the spec version, and re-running the gold evaluation set before deploying.
3. **Environment variable override permitted**: `CLAUDE_ENRICHMENT_MODEL` and `CLAUDE_QUERY_MODEL_*` env vars may override pinned IDs for testing, but defaults in `config.py` must match this table exactly.
4. **No aliases in acceptance artifacts**: Gold set rubrics, PROVE reports, and CI scripts must reference the pinned ID, not a generic name.

---

## Architecture: The 3-Tier Layered System

### Overview

VaultIQ is built on a **3-tier configuration hierarchy** that enables a single codebase to serve multiple industries and be deeply customized per customer — without forking code.

```
┌─────────────────────────────────────────────────────────────────┐
│  Tier 1: Core Platform                                          │
│                                                                 │
│  Universal capabilities shared by every customer, every         │
│  vertical. Document ingestion, Claude enrichment, SQL           │
│  insurance schema, query router, access control, audit.         │
│  This tier never contains vertical or customer-specific code.   │
└─────────────────────────────────┬───────────────────────────────┘
                                  ↓  extended by
┌─────────────────────────────────────────────────────────────────┐
│  Tier 2: Vertical Modules                                       │
│                                                                 │
│  Industry-specific packages that add document classifiers,      │
│  Claude extraction prompts, compliance rule sets, and           │
│  dashboard templates for a specific insurance niche.            │
│  Each module is a self-describing folder loaded at startup.     │
│  A 4-person team can support 5+ verticals simultaneously.       │
└─────────────────────────────────┬───────────────────────────────┘
                                  ↓  overridden by
┌─────────────────────────────────────────────────────────────────┐
│  Tier 3: Tenant Customization Layer                             │
│                                                                 │
│  Per-customer configuration: branding, custom fields,           │
│  prompt overrides, feature flags, integrations, AI model        │
│  preferences, cost caps. Each customer instance has its own     │
│  tenant.json and customizations/ directory. Customization       │
│  is data, not code — no code deployment required for most       │
│  customer-specific changes.                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Model

**Not multiple products**: Separate codebases per vertical or customer means every bug fix is shipped N times, features diverge within months, and a small engineering team cannot maintain them.

**Not a monolith with embedded vertical logic**: `if vertical == "construction":` scattered through core code is untestable, unmaintainable, and unsellable as modular add-ons.

**Configuration as data**: The critical design principle is that vertical differences and customer customizations are *data* (YAML rules, prompt files, JSON config), not code. Adding a new vertical is a new folder. Customizing a customer's interface or prompts requires no code deployment.

**Single-tenant isolation**: Each customer gets their own dedicated Docker Compose instance — separate application, separate PostgreSQL database, separate configuration. No cross-customer data sharing.

---

## Storage Architecture

### Mandatory Baseline (v1.0)

PostgreSQL + pgvector is the **required** storage backend for all production profiles. There is no runtime choice of vector backend in v1.0 — Qdrant is fully removed from scope and the `chroma` and `qdrant` backend paths are not implemented.

| Profile | Database | Vector Store | Notes |
|---|---|---|---|
| `hosted` | PostgreSQL | pgvector (same DB) | Production — Standard tier |
| `spark` | PostgreSQL | pgvector (same DB) | Production — Enterprise tier |
| `laptop` | SQLite | In-memory / stub | **Development only** — see gap note |

**Laptop profile gap (explicit)**: The `laptop` profile uses SQLite for developer convenience (zero-dependency setup). Insurance SQL tables and pgvector-based SQL JOINs are **not available** in this profile. Tests that exercise the enrichment pipeline, structured query router, or vector JOINs **must** run against a PostgreSQL test database. The `laptop` profile is sufficient only for auth, document upload, basic RAG, and frontend development.

Any test asserting insurance schema behavior must declare `@pytest.mark.requires_postgres` and be skipped when `database_backend == "sqlite"`.

---

## Tier 1: Core Platform

### Core Capabilities

| Capability | Description |
|---|---|
| **Document Intelligence** | Upload any document → Claude classifies, enriches, and extracts structured data |
| **Insurance SQL Schema** | Policies, coverages, claims, certificates, requirements in queryable tables |
| **Natural Language Q&A** | Ask any question in plain English, get cited answers |
| **Deterministic Query Router** | SQL-first for facts (no LLM to choose path), RAG fallback for analysis |
| **Claude Enrichment Pipeline** | Two-call pipeline: synopsis + batch chunk enrichment with prompt caching |
| **pgvector Semantic Search** | Vector search co-located in same PostgreSQL database as insurance tables |
| **Access Control** | Tag-based visibility — agents see only their assigned properties |
| **Audit Trail** | Every query, every answer, every source cited, configurable audit level |
| **Cost Tracking** | Per-document enrichment cost, per-query cost, configurable caps |
| **Review Workflow** | Low-confidence answers routed to human review queue |

---

## Tier 2: Vertical Module System

### Architecture

```
ai_ready_rag/
├── core/                           ← Never touches vertical logic
└── modules/                        ← Vertical packages
    ├── registry.py                 ← Module loader
    ├── community_associations/     ← LAUNCH VERTICAL
    │   ├── manifest.json
    │   ├── classifiers.yaml
    │   ├── prompts/
    │   ├── migrations/
    │   ├── compliance.py
    │   └── dashboard.json
    ├── construction/               ← Year 1
    ├── real_estate/                ← Year 1
    ├── transportation/             ← Year 2
    └── healthcare/                 ← Year 2
```

### Module Manifest

```json
{
  "name": "community_associations",
  "version": "1.0",
  "display_name": "Community Associations",
  "document_types": ["ccr", "bylaws", "reserve_study", "appraisal", "board_minutes", "unit_owner_letter"],
  "compliance_rules": true,
  "schema_migrations": ["001_community_tables.py"],
  "prompts": ["ccr_extraction", "reserve_study", "board_minutes"],
  "dashboard_views": ["compliance_dashboard", "ccr_requirements", "unit_owner_report"],
  "requires_core_version": ">=1.0"
}
```

### The 4 Extension Points Per Vertical

| Extension Point | Format |
|---|---|
| Document classifiers | YAML rule files |
| Claude prompts | Text prompt files (loaded via PromptResolver) |
| Compliance rules | Python classes implementing `ComplianceChecker` protocol |
| Dashboard views | JSON view config + React component registry |

### Vertical Capabilities

#### Community Associations (Launch Vertical)
CC&R/bylaw compliance checking, reserve study analysis, unit owner letter generation, board presentation package, multi-policy program view. **Unique docs**: CC&Rs, bylaws, reserve studies, appraisals. **Build effort**: Already built at Marshall Wells.

#### Construction (Year 1, ~2-3 weeks)
Subcontractor certificate tracking, certificate compliance verification (limits/AI/WOS), OCIP management, builder's risk tracking. **Unique docs**: Subcontractor agreements, OCIP schedules, builder's risk.

#### Real Estate / Property Management (Year 1, ~2 weeks)
Tenant certificate tracking, lease insurance requirement extraction, landlord/tenant coverage comparison. **Unique docs**: Lease abstracts, tenant COIs, SOVs.

#### Transportation (Year 2, ~2-3 weeks)
MCS-90/DOT filing tracking, fleet schedule management, cargo coverage tracking.

#### Healthcare / Professional Liability (Year 2, ~2 weeks)
Claims-made vs. occurrence tracking, consent-to-settle flagging, professional liability limit trending.

---

## Tier 3: Tenant Customization Layer

### tenant.json Schema

```json
{
  "tenant_id": "sunrise-insurance",
  "name": "Sunrise Insurance Agency",
  "modules": ["core", "community_associations"],
  "ui": {
    "brand": {
      "name": "Sunrise Insurance Agency",
      "logo_url": "/tenant/assets/logo.png",
      "primary_color": "#2563EB",
      "accent_color": "#F59E0B"
    },
    "layout": {
      "sidebar_items": ["dashboard", "accounts", "renewals", "compliance"],
      "default_view": "accounts",
      "hide_features": []
    }
  },
  "custom_fields": {
    "insurance_policies": [
      { "key": "internal_job_code", "label": "Job Code", "type": "string", "searchable": true }
    ]
  },
  "features": {
    "certificate_automation": true,
    "renewal_prep": true,
    "compliance_checking": true,
    "client_portal": false,
    "ams_integration": null
  },
  "ai": {
    "primary": "claude",
    "fallback": "ollama",
    "enrichment_model": "claude-sonnet-4-6",
    "query_model_simple": "claude-haiku-4-5-20251001",
    "query_model_analytical": "claude-sonnet-4-6",
    "monthly_query_cap_usd": 150,
    "daily_enrichment_cap_usd": 10
  },
  "integrations": {
    "webhooks": [],
    "ams": null
  }
}
```

### Config Reload Policy (v1.0)

**Restart required** for all tenant config and prompt changes. Hot-reload is deferred to v1.1.

| Action | Required procedure |
|---|---|
| Branding, feature flag, model cap changes | Update `tenant.json` → `docker compose restart api` |
| Prompt override added or modified | Add/update file in `customizations/prompts/` → restart |
| Module list change | Update `modules` array in `tenant.json` → restart |
| Expected downtime | < 30 seconds per instance |
| Audit log event | `CONFIG_RELOAD` — timestamp, operator, list of changed files |

Config reload must be logged at `INFO` level with the diff of changed keys (excluding secrets). On next startup, the `TenantConfigResolver` re-reads all files and logs the resolved effective config.

### The 5 Customization Domains

**1. Branding & UI Theme** — CSS custom properties, logo, company name from `ui.brand`. Frontend calls `GET /api/tenant/config` at startup and applies CSS variables. Full white-labeling supported.

**2. Custom Fields** — JSONB extension columns on all core insurance entities:
```sql
ALTER TABLE insurance_policies  ADD COLUMN custom_fields JSONB DEFAULT '{}';
ALTER TABLE insurance_accounts  ADD COLUMN custom_fields JSONB DEFAULT '{}';
ALTER TABLE insurance_coverages ADD COLUMN custom_fields JSONB DEFAULT '{}';
```
Field definitions in `tenant.json` drive frontend rendering. Claude extraction prompts can be extended to auto-populate these fields — no code change required.

**3. Prompt Overrides** — The `PromptResolver` uses a layered lookup:
```
Tenant override → Vertical default → Core base prompt
```
```python
class PromptResolver:
    def get_prompt(self, prompt_name: str, tenant_id: str, vertical: str) -> str:
        tenant_path = f"tenant-instances/{tenant_id}/customizations/prompts/{prompt_name}.txt"
        if exists(tenant_path): return load(tenant_path)
        vertical_path = f"modules/{vertical}/prompts/{prompt_name}.txt"
        if exists(vertical_path): return load(vertical_path)
        return load(f"core/prompts/{prompt_name}.txt")
```
A missing prompt file at any tier falls through to the next — it is never an error unless the core base prompt is also missing.

**4. Feature Flags** — Every backend endpoint and frontend component checks `tenant.features.X`. Missing keys default to `false` (deny by default for optional capabilities).

**5. Integrations & Webhooks** — AMS connectors and webhook endpoints configured in `tenant.json`. Connector code (`applied_epic.py`, `vertafore.py`) in `ai_ready_rag/integrations/` is loaded only for tenants with that integration enabled.

### Tenant Config Resolver

```python
class TenantConfigResolver:
    def __init__(self, tenant_id: str):
        self.core    = load_core_defaults()
        self.vertical = load_vertical_config(get_active_modules(tenant_id))
        self.tenant  = load_tenant_config(tenant_id)

    def resolve(self) -> ResolvedConfig:
        return deep_merge(self.core, self.vertical, self.tenant)

    def get_prompt(self, prompt_name: str) -> str:
        return PromptResolver(self.tenant_id, self.vertical_name).get(prompt_name)

    def feature_enabled(self, feature: str) -> bool:
        return self.resolve().features.get(feature, False)
```

---

## System Architecture

### Directory Structure

```
ai_ready_rag/
├── main.py
├── config.py
├── api/
│   ├── auth.py, chat.py, documents.py, tags.py, users.py, admin.py
│   └── insurance.py              ← NEW
├── core/
├── db/
│   ├── models/
│   │   ├── insurance.py          ← NEW
│   │   └── vectors.py            ← NEW (pgvector)
│   └── migrations/               ← NEW (Alembic)
├── services/
│   ├── rag_service.py            ← MODIFIED
│   ├── processing_service.py     ← MODIFIED
│   ├── claude_enrichment_service.py  ← NEW
│   ├── claude_query_service.py       ← NEW (Standard tier)
│   ├── claude_model_router.py        ← NEW
│   ├── insurance_data_service.py     ← NEW
│   ├── query_router.py               ← NEW
│   └── pgvector_service.py           ← NEW (replaces Qdrant)
├── modules/                      ← NEW
│   ├── registry.py
│   └── community_associations/ …
├── tenant/                       ← NEW
│   ├── config.py
│   ├── resolver.py
│   └── api.py
└── integrations/                 ← NEW
    ├── base.py
    └── webhook.py

tenant-instances/                 ← NEW (outside app package)
└── {agency-slug}/
    ├── tenant.json
    └── customizations/prompts/
```

### Ingestion Pipeline (Both Tiers)

```
Upload → Docling Parse → Chunk
                           │
                    PromptResolver selects prompts
                    (tenant override → vertical → core)
                           │
               Claude API Call #1: Synopsis
               Model: claude-sonnet-4-6
               Output: document_type, named_insured, carrier, key_facts
                           │
               Claude API Call #2: Chunk Enrichment (batched)
               Model: claude-sonnet-4-6
               Input: synopsis (cached prefix) + 8 chunks
               Output: { summary, entities[] } per chunk
                           │
               ┌───────────┴───────────┐
               │                       │
       Entity-to-SQL mapping    Enrich chunk text
       (Canonicalization        "[SUMMARY] ... [ENTITIES] ... [ORIGINAL] ..."
        Contract applied)
               │                       │
       insurance_* tables        chunk_vectors (pgvector)
       (PostgreSQL)              (same PostgreSQL database)
               │
       Fallback: if Claude API unavailable → Ollama pipeline
       (no SQL population; RAG only)
```

### Query Pipeline — Standard Tier (Claude Primary)

```
User Question → Deterministic Query Router
                        │
           ┌────────────┴────────────┐
           │                         │
    SQL Lookup (always first)   RAG Retrieval (fallback)
    (pattern match → template   (pgvector + tag ACL)
     → execute → sufficiency    (enriched chunks)
     check)                          │
           │                         │
           └────────────┬────────────┘
                        │
           Claude API (primary)
           ├── Haiku (claude-haiku-4-5-20251001): structured/simple
           └── Sonnet (claude-sonnet-4-6): analytical/comparison
                        │
           Ollama / qwen3-rag (fallback if Claude unavailable)
```

### Query Pipeline — Enterprise Tier (Ollama Primary)

```
User Question → Deterministic Query Router
                        │
           ┌────────────┴────────────┐
           │                         │
    SQL Lookup (always first)   RAG Retrieval (fallback)
    (same logic, no internet)   (pgvector + tag ACL)
           │                         │
           └────────────┬────────────┘
                        │
           Ollama / qwen3-rag (primary, no internet required)
```

### Deployment: Standard Tier

```
Per-Customer Docker Compose Instance
┌─────────────────────────────────────────────────┐
│  {agency}.vaultiq.app                           │
│                                                 │
│  FastAPI (:8502) + React                        │
│  Claude API → enrichment + queries              │
│  Ollama → fallback queries                      │
│                                                 │
│  PostgreSQL + pgvector (single database)        │
│  tenant.json + customizations/                  │
└─────────────────────────────────────────────────┘
```

### Deployment: Enterprise Tier

**v1.0 scope: Mode B only.**

```
Mode B — Temporary Internet (v1.0)
DGX Spark with internet during ingestion:
  1. Upload PDF
  2. Docling parse
  3. Claude enrich (API call while internet available)
  4. Store in local PostgreSQL + pgvector
  5. Disconnect internet
  6. Query locally via Ollama (no internet required)
```

**Mode A — Air-Gap Transfer is deferred to v1.1.**
The signed pg_dump + HMAC manifest + atomic pg_restore workflow (`transfer export/verify/import` CLI) is not in v1.0 acceptance scope. Enterprise tier v1.0 requires Mode B only.

---

## Data Model: Insurance SQL Schema

### Entity Relationship

```
accounts ──< policies ──< coverages
    │             │
    │             └──< claims
    │
    ├──< certificates
    ├──< requirements
    └──< account_documents (junction → documents)

documents ──< enrichment_entities
    │
    └──< enrichment_synopses

chunk_vectors (pgvector) ──► documents (FK)
```

### Core Tables

#### `insurance_accounts`

| Column | Type | Description |
|---|---|---|
| id | TEXT PK | UUID |
| name | TEXT | Named insured |
| account_type | TEXT | "condo_association", "commercial", "residential", "hoa" |
| address, city, state, zip_code | TEXT | Property address |
| units_residential, units_commercial | INTEGER | Unit counts |
| year_built | INTEGER | Construction year |
| construction_type | TEXT | Frame, masonry, fire-resistive |
| management_company, management_contact | TEXT | Property manager |
| agent_name, agent_email | TEXT | Servicing agent |
| custom_fields | JSONB | Tenant-defined extension fields |
| tenant_id | TEXT | Instance identifier |
| is_deleted, deleted_at | BOOLEAN/DATETIME | Soft-delete |
| valid_from, valid_to | DATETIME | Versioning (NULL valid_to = current) |
| source_document_id | TEXT FK | Document that produced this record |
| extraction_model | TEXT | Pinned model ID used |
| extraction_confidence | REAL | 0.0–1.0 |
| created_at, updated_at | DATETIME | Timestamps |

#### `insurance_policies`

| Column | Type | Description |
|---|---|---|
| id | TEXT PK | UUID |
| account_id | TEXT FK | → insurance_accounts.id |
| line_of_business | TEXT | "gl", "property", "do", "crime", "umbrella", "wc", "equipment_breakdown", "epli", "cyber", "auto" |
| carrier | TEXT | Canonical carrier name (see Canonicalization Contract) |
| policy_number | TEXT | Normalized policy number |
| effective_date, expiration_date | DATE | ISO 8601 (YYYY-MM-DD) |
| status | TEXT | "active", "expired", "cancelled", "pending" |
| annual_premium | REAL | Dollars (no formatting) |
| program_name, broker | TEXT | Program and wholesale broker |
| is_admitted | BOOLEAN | Admitted vs. surplus |
| layer_position | INTEGER | 1=primary, 2=first excess |
| program_group_id | TEXT | UUID linking layers in same tower |
| custom_fields | JSONB | Tenant-defined extension fields |
| tenant_id | TEXT | Instance identifier |
| is_deleted, deleted_at, valid_from, valid_to | — | Soft-delete + versioning |
| source_document_id, extraction_model, extraction_confidence | — | Provenance |

**Idempotency key**: `(account_id, line_of_business, effective_date, carrier)` — exact match = update in place.

**Index**: `(account_id, line_of_business, effective_date)`

#### `insurance_coverages`

| Column | Type | Description |
|---|---|---|
| id | TEXT PK | UUID |
| policy_id | TEXT FK | → insurance_policies.id |
| coverage_type | TEXT | "per_occurrence", "aggregate", "products_completed_ops", "building", "contents", "per_claim", "retention" |
| limit_amount | REAL | Dollars (no formatting) |
| deductible_amount | REAL | Dollars |
| deductible_type | TEXT | "per_claim", "per_occurrence", "annual_aggregate", "percentage" |
| coinsurance_pct | REAL | Percentage as decimal (0.80 = 80%) |
| valuation | TEXT | "replacement_cost", "actual_cash_value", "agreed_value" |
| sublimit | REAL | Dollars |
| description | TEXT | Free-text |
| custom_fields | JSONB | Tenant-defined extension fields |
| source_document_id | TEXT FK | Provenance |

**Idempotency key**: `(policy_id, coverage_type)` — exact match = update in place.

**Index**: `(policy_id, coverage_type)`

#### `insurance_claims`

| Column | Type | Description |
|---|---|---|
| id | TEXT PK | UUID |
| account_id | TEXT FK | → insurance_accounts.id |
| policy_id | TEXT FK | Optional |
| line_of_business, claim_number | TEXT | Line and carrier claim number |
| date_of_loss, date_reported | DATE | ISO 8601 |
| status | TEXT | "open", "closed", "reopened" |
| claimant, description | TEXT | |
| paid_amount, reserved_amount, total_incurred, recovery_amount | REAL | Dollars |
| source_document_id | TEXT FK | Loss run document |

**Idempotency key**: `(account_id, claim_number)` when claim_number is non-null; otherwise `(account_id, date_of_loss, line_of_business, total_incurred)`.

**Index**: `(account_id, date_of_loss)`

#### `insurance_certificates`

| Column | Type | Description |
|---|---|---|
| id | TEXT PK | UUID |
| account_id | TEXT FK | |
| certificate_type | TEXT | "acord_24", "acord_25", "acord_27", "acord_28" |
| holder_name, holder_address | TEXT | |
| issued_date | DATE | ISO 8601 |
| additional_insured, waiver_of_subrogation | BOOLEAN | |
| source_document_id | TEXT FK | |

#### `insurance_requirements`

| Column | Type | Description |
|---|---|---|
| id | TEXT PK | UUID |
| account_id | TEXT FK | |
| requirement_source | TEXT | "ccr", "bylaws", "loan_agreement", "management_agreement" |
| coverage_line | TEXT | |
| requirement_text | TEXT | Exact text |
| min_limit | REAL | Dollars |
| min_limit_type | TEXT | "per_occurrence", "aggregate" |
| is_met, current_limit, gap_amount | BOOLEAN/REAL | Computed compliance fields |
| source_document_id | TEXT FK | |
| last_checked_at | DATETIME | |

**Idempotency key**: `(account_id, requirement_source, coverage_line)` — exact match = update in place.

#### `enrichment_synopses` and `enrichment_entities`

See Claude Enrichment Pipeline section for column definitions. These tables are append-only during enrichment; existing rows are superseded (not deleted) on re-enrichment.

#### `chunk_vectors` (pgvector)

| Column | Type | Description |
|---|---|---|
| id | TEXT PK | UUID |
| document_id | TEXT FK | → documents.id |
| chunk_index | INTEGER | Position in document |
| chunk_text | TEXT | Original chunk |
| enriched_text | TEXT | [SUMMARY] + [ENTITIES] + [ORIGINAL] |
| embedding | vector(768) | nomic-embed-text embedding |
| metadata | JSONB | Tags, tenant_id, document_type, module_context |
| created_at | TIMESTAMP | |

**Indexes**:
```sql
CREATE INDEX ON chunk_vectors USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON chunk_vectors (document_id);
CREATE INDEX ON chunk_vectors USING gin (metadata);
```

**Cross-table JOIN** (vector + insurance data):
```sql
SELECT cv.enriched_text,
       cv.embedding <=> :query_embedding AS distance,
       p.carrier, p.policy_number, p.line_of_business,
       c.coverage_type, c.limit_amount
FROM chunk_vectors cv
JOIN insurance_policies p ON cv.document_id = p.source_document_id
LEFT JOIN insurance_coverages c ON p.id = c.policy_id
WHERE cv.embedding <=> :query_embedding < :max_distance
ORDER BY distance
LIMIT :top_k;
```

### Document Model Additions

| Column | Type | Description |
|---|---|---|
| enrichment_status | TEXT | null / "pending" / "enriching" / "completed" / "failed" |
| enrichment_model | TEXT | Pinned model ID used |
| enrichment_version | TEXT | Prompt version (re-enrichment tracking) |
| enrichment_tokens_used | INTEGER | |
| enrichment_cost_usd | REAL | |
| enrichment_completed_at | DATETIME | |
| insurance_account_id | TEXT FK | → insurance_accounts.id |
| document_role | TEXT | "policy", "certificate", "loss_run", "endorsement", "ccr", etc. |

### Document Type Taxonomy

```python
class DocumentType(str, Enum):
    """v1.0 — Base taxonomy. Modules extend via classifiers.yaml."""
    POLICY          = "policy"
    CERTIFICATE     = "certificate"
    LOSS_RUN        = "loss_run"
    ENDORSEMENT     = "endorsement"
    CCR             = "ccr"
    BYLAWS          = "bylaws"
    RESERVE_STUDY   = "reserve_study"
    APPRAISAL       = "appraisal"
    PROPOSAL        = "proposal"
    SUBMISSION      = "submission"
    BIND_ORDER      = "bind_order"
    BOR             = "bor"
    UNIT_OWNER_LETTER = "unit_owner_letter"
    CORRESPONDENCE  = "correspondence"
    UNKNOWN         = "unknown"   # Flagged for admin review; F1 tracked
```

---

## Claude Enrichment Pipeline

Claude (`claude-sonnet-4-6`) is the **primary intelligence layer** for document ingestion on both tiers. Ollama (`qwen3-rag`) is the fallback when the Claude API is unavailable.

### Processing Flow

```
1. Docling Parse + Chunk (existing)
2. PromptResolver: load active prompts
3. Claude Call #1: Document Synopsis  (claude-sonnet-4-6)
4. Claude Call #2: Chunk Enrichment   (claude-sonnet-4-6, batched 8 chunks)
5. Canonicalization Contract applied to all extracted values
6. Entity-to-SQL mapping → insurance_* tables
7. Enriched chunk text → chunk_vectors (pgvector)
8. Fallback: if Claude API unavailable → Ollama pipeline (RAG only, no SQL)
```

### Claude API Call #1: Document Synopsis

```python
{
    "model": "claude-sonnet-4-6",
    "max_tokens": 1024,
    "system": [{"type": "text", "text": SYNOPSIS_SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
    "messages": [{"role": "user", "content": f"""
Filename: {filename}
Total chunks: {chunk_count}
Representative content: {sampled_chunks}

Return JSON:
{{
  "synopsis": "200-400 word summary",
  "document_type": "policy|certificate|loss_run|...",
  "document_subtype": "acord_24|occurrence|claims_made|...",
  "named_insured": "...",
  "carrier": "...",
  "policy_number": "...",
  "policy_period": "YYYY-MM-DD / YYYY-MM-DD",
  "coverage_lines": ["gl", "property"],
  "key_facts": ["168 residential units", "$1M/$2M GL limits"],
  "premium_total": 12345.00
}}"""}]
}
```

### Claude API Call #2: Chunk Enrichment (Batched)

```python
{
    "model": "claude-sonnet-4-6",
    "max_tokens": 4096,
    "system": [{"type": "text", "text": CHUNK_ENRICHMENT_SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": f"Document synopsis:\n{synopsis_text}",
         "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": f"""
Enrich these chunks. Return JSON array, one object per chunk:
[{{
  "chunk_index": 0,
  "summary": "Plain-English interpretation",
  "entities": [
    {{"type": "limit", "value": "$1,000,000", "context": "GL per occurrence"}},
    {{"type": "insurer", "value": "Travelers", "context": "GL carrier"}}
  ]
}}]
{formatted_chunk_batch}"""}
    ]}]
}
```

### Prompt Caching Strategy

```
Call #1 (Synopsis):      System prompt cached            → ~30% input savings
Call #2a (Chunks 1-8):   System + synopsis cached        → ~40% input savings
Call #2b (Chunks 9-16):  System + synopsis cached        → ~40% input savings
```

**Estimated cost**: ~$0.06/document gross, ~$0.04/document with caching.

### Entity-to-SQL Mapping

```python
ENTITY_TO_TABLE_MAP = {
    "insured":        ("insurance_accounts",  "name"),
    "address":        ("insurance_accounts",  "address"),
    "insurer":        ("insurance_policies",  "carrier"),
    "policy_number":  ("insurance_policies",  "policy_number"),
    "effective_date": ("insurance_policies",  "effective_date"),
    "expiration_date":("insurance_policies",  "expiration_date"),
    "premium":        ("insurance_policies",  "annual_premium"),
    "limit":          ("insurance_coverages", "limit_amount"),
    "deductible":     ("insurance_coverages", "deductible_amount"),
    "valuation":      ("insurance_coverages", "valuation"),
    "claim":          ("insurance_claims",    None),   # complex mapping
    "loss_date":      ("insurance_claims",    "date_of_loss"),
    "requirement":    ("insurance_requirements", "requirement_text"),
    "min_coverage":   ("insurance_requirements", "min_limit"),
}
```

---

## Canonicalization Contract

All entity values extracted by Claude **must** be canonicalized before writing to insurance tables. Raw Claude output is never written directly to SQL.

### Rules by Data Type

| Data Type | Canonical Form | Examples |
|---|---|---|
| **Dates** | ISO 8601 `YYYY-MM-DD` | `"1/1/2026"` → `"2026-01-01"` |
| **Monetary / limits** | REAL in dollars, no formatting | `"$1M"` → `1000000.0`; `"$1,000,000.00"` → `1000000.0`; `"1.5M"` → `1500000.0` |
| **Percentages** | REAL as decimal | `"80%"` → `0.80`; `"100"` → `1.00` |
| **Policy numbers** | Strip leading/trailing whitespace; collapse internal runs to single space | `"  CPP-029 4618 "` → `"CPP-029 4618"` |
| **Carrier names** | Lookup in `modules/core/data/carrier_aliases.csv`; if no match, store as-is | `"Travelers Casualty"` → `"Travelers"` |
| **Coverage types** | Map to enum values | `"each occurrence"` → `"per_occurrence"` |
| **Line of business** | Map to enum values | `"General Liability"` → `"gl"` |

### Carrier Alias Table

`modules/core/data/carrier_aliases.csv` — a simple two-column CSV: `raw_name, canonical_name`. Maintained by the engineering team. Common entries seeded at launch (Travelers, Chubb, Liberty Mutual, Hartford, Zurich, AIG, etc.). Unmatched carrier names stored as-is and flagged with `extraction_confidence < 0.7` for admin review.

### Validation Failure Behavior

When an entity value cannot be canonicalized (unparseable date, non-numeric monetary value, etc.):
1. Entity is stored in `enrichment_entities` with `confidence = 0.0`
2. Entity is **not** written to insurance_* tables
3. A `review_items` entry is created with `review_type = "canonicalization_failure"`
4. Processing continues — one bad entity does not fail the entire document

---

## Account Matching

Account matching resolves a Claude-extracted `named_insured` string to an existing `insurance_accounts` row or creates a new one. Single-threshold fuzzy matching was rejected as fragile. v1.0 uses a 3-tier deterministic decision process.

### 3-Tier Decision Process

**Tier 1 — Auto-link (high confidence)**

Auto-link without human review when ANY of the following are true:
- Name similarity ≥ 95% (difflib.SequenceMatcher ratio) AND address matches existing account
- Exact policy number match on any document already linked to an existing account
- Name similarity ≥ 98% regardless of other signals

Action: Link document to existing account. Log `ACCOUNT_MATCH_AUTO` audit event.

**Tier 2 — Flag for review (medium confidence)**

Flag when:
- Name similarity 75%–95% with no corroborating signal (no address, no policy number match)
- Multiple existing accounts score above 75% (ambiguous — more than one candidate)

Action: Document linked to a **provisional account** (a new account record with `status = "pending_merge_review"`). Admin review queue entry created showing both the provisional account and the candidate(s). Agent or admin confirms or reassigns. Log `ACCOUNT_MATCH_PENDING` audit event.

**Tier 3 — Auto-create (low confidence)**

When:
- Name similarity < 75% against all existing accounts
- No policy number evidence pointing to an existing account

Action: New `insurance_accounts` row created automatically. Log `ACCOUNT_CREATED` audit event.

### Implementation Note

The similarity calculation uses `difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()` after stripping legal suffixes ("LLC", "Inc", "Corporation", "Condominium Association", "Homeowners Association") from both strings before comparison. This prevents "Marshall Wells Lofts LLC" from failing to match "Marshall Wells Lofts" due to the legal suffix.

---

## Query Router

### Confidence Score Definition

Every answer produced by the system carries a confidence score (0–100 integer). The score is a weighted composite of three inputs:

```
confidence = clamp(round(
    sql_completeness  × 0.50 +
    rag_similarity    × 0.30 +
    llm_self_report   × 0.20
), 0, 100)
```

| Input | Definition | Range |
|---|---|---|
| `sql_completeness` | `(non_null_primary_fields / total_primary_fields_requested) × 100`. 100 if SQL sufficient; 0 if SQL returned no rows; proportional if partial. | 0–100 |
| `rag_similarity` | `(1.0 - mean_cosine_distance_of_top_k_chunks) × 100`. 0 if RAG not invoked (SQL-only answer). | 0–100 |
| `llm_self_report` | Extracted from LLM response via prompt instruction ("On a scale of 0–100, rate your confidence in this answer"). Default 70 if not extractable. | 0–100 |

**Thresholds**:

| Threshold | Value | Behavior |
|---|---|---|
| Review queue trigger | < 70 | Answer held in review queue; user sees tentative answer with flag option |
| Standard tier acceptance (gold set) | ≥ 90 | All 16 gold set questions must meet or exceed this on first run |
| Enterprise tier acceptance (gold set) | ≥ 70 | All 16 gold set questions must meet or exceed this on first run |

**Calibration dataset**: 16 Marshall Wells gold set questions with known correct answers, expected sources (SQL vs. RAG), and expected confidence bands. The calibration set is fixed at spec commit — new questions are additive only.

### Deterministic Routing Specification

The router never calls an LLM to choose a path. The decision is rule-based, executed in order. The first matching rule wins.

#### Step 1: Account Resolution

Extract account name from: (a) explicit mention in query text, (b) session context (active account). If no account can be resolved → `CONVERSATIONAL` intent. Proceed to LLM directly with chat history only.

#### Step 2: Entity Type Detection (ordered pattern matching)

Rules are evaluated in order; first match wins:

| Priority | Pattern match on lowercased query | Intent |
|---|---|---|
| 1 | `compare \| vs \| versus \| difference between \| how do .+ compare` | `COMPARISON` |
| 2 | `what .+(limit\|deductible\|premium\|policy number\|carrier\|insurer\|expire\|renew\|effective)` | `STRUCTURED` |
| 3 | `list (all\|the) .+` \| `who is the .+` \| `when does .+` \| `is .+ covered` | `STRUCTURED` |
| 4 | `what does .+(say\|require\|state\|mean)` \| `summarize\|explain\|describe\|exclusion\|condition` | `ANALYTICAL` |
| 5 | Matches both a STRUCTURED and ANALYTICAL pattern | `HYBRID` |
| 6 | Default (no pattern match) | `ANALYTICAL` |

#### Step 3: SQL Execution and Sufficiency Check

For `STRUCTURED`, `COMPARISON`, and `HYBRID` intents: execute the mapped SQL template (see SQL Template Catalog). SQL result is **sufficient** when:
- Result set has ≥ 1 row AND
- All primary fields for the question type are non-null (e.g., for a limit query: `limit_amount` is non-null; for a carrier query: `carrier` is non-null)

SQL result is **insufficient** (trigger RAG fallback) when:
- Result set is empty, OR
- All rows have NULL in the primary field being asked about

#### Step 4: Routing Decision

| Intent | SQL Sufficient? | Route |
|---|---|---|
| `STRUCTURED` | Yes | Format SQL result → LLM formatting call (Haiku) |
| `STRUCTURED` | No | RAG retrieval → LLM synthesis (Haiku) |
| `COMPARISON` | Yes | Format comparison table → LLM formatting call (Haiku) |
| `COMPARISON` | No | RAG retrieval → LLM synthesis (Sonnet) |
| `ANALYTICAL` | N/A | RAG retrieval → LLM synthesis (Sonnet) |
| `HYBRID` | Yes | SQL as context + RAG retrieval → LLM synthesis (Sonnet) |
| `HYBRID` | No | RAG retrieval → LLM synthesis (Sonnet) |
| `CONVERSATIONAL` | N/A | Chat history only → LLM (Haiku) |

Unmapped queries (no SQL template for the detected entity type): treated as `ANALYTICAL`, fall through to RAG. Never raise an error to the user.

### Claude Model Routing (Standard Tier)

```python
class ClaudeModelRouter:
    def select_model(self, intent: QueryIntent) -> str:
        if intent in (QueryIntent.STRUCTURED, QueryIntent.CONVERSATIONAL):
            return "claude-haiku-4-5-20251001"
        return "claude-sonnet-4-6"
```

Model selection is overridable per tenant via `ai.query_model_simple` and `ai.query_model_analytical` in `tenant.json` — but overrides must also use pinned IDs.

### SQL Template Catalog

The following 8 templates are required for v1.0. No SQL executes outside this catalog. An incoming query that maps to no template falls through to RAG (`ANALYTICAL` path) — it is never an error.

| Template ID | Intent | Answers |
|---|---|---|
| `coverage_by_account_line` | STRUCTURED | "What are the GL limits?" |
| `carrier_lookup` | STRUCTURED | "Who is the carrier for property?" |
| `premium_query` | STRUCTURED | "What is the annual premium for..." |
| `policy_dates` | STRUCTURED | "When does the D&O policy expire?" |
| `claims_history` | STRUCTURED | "What claims have been filed against...?" |
| `compliance_gap` | STRUCTURED | "Does current coverage meet CC&R requirements?" |
| `coverage_schedule` | STRUCTURED | "List all coverages for Marshall Wells" |
| `comparison_by_line` | COMPARISON | "Compare GL limits across all properties" |

Example template (all follow the same safety pattern):

```python
ALLOWED_QUERY_TEMPLATES = {
    "coverage_by_account_line": """
        SELECT c.coverage_type, c.limit_amount, c.deductible_amount,
               p.carrier, p.policy_number, p.effective_date, p.expiration_date
        FROM insurance_coverages c
        JOIN insurance_policies p ON c.policy_id = p.id
        WHERE p.account_id = :account_id
          AND p.line_of_business = :line
          AND p.status = 'active'
          AND p.valid_to IS NULL
          AND p.is_deleted = FALSE
        ORDER BY c.coverage_type
        LIMIT :row_cap
    """,
    # ... 7 additional templates ...
}
```

### SQL Execution Safety

1. **Allowlisted templates only** — no arbitrary SQL generation, ever
2. **Parameterized bindings** — no string interpolation; all user values via bind params
3. **5-second timeout** — queries exceeding this are killed; fallback to RAG
4. **1,000-row cap** — `LIMIT :row_cap` in every template; pagination required beyond
5. **Read-only connection** — query-time SQL uses a read-only database role; writes during enrichment only
6. **Unmapped query behavior** — no template match → `structured_data_unavailable` flag set → fall through to RAG path; no error raised to user

### Router SLO Targets

| Metric | Target |
|---|---|
| SQL lookup (p95) | < 50ms |
| RAG retrieval (p95) | < 500ms |
| End-to-end structured (p95) | < 1 second |
| End-to-end analytical (p95) | < 5 seconds |
| End-to-end hybrid (p95) | < 6 seconds |
| Claude API timeout | 15 seconds → Ollama fallback |
| Ollama timeout | 30 seconds → return partial result |

---

## Data Governance

### Soft-Delete and Versioning

All insurance_* tables use `is_deleted` + `deleted_at` for soft-delete. Hard-delete after configurable retention (default 90 days).

**Current record filter**: Always `WHERE valid_to IS NULL AND is_deleted = FALSE`.

**Cascade rules**:
- Document deleted → soft-delete all insurance_* rows with that `source_document_id`
- Account deleted → soft-delete all linked policies, coverages, claims, certificates
- Policy deleted → soft-delete all coverages and claims linked to that policy

### Idempotency and Re-enrichment

Re-enrichment is idempotent. The same document uploaded twice must not create duplicate SQL rows.

**Conflict resolution order**:

1. **Same document re-uploaded** (same content hash): no-op if `enrichment_version` matches current prompt version. If prompt version differs, supersede existing rows (set `valid_to = now()`) and create new rows with `valid_from = now()`.

2. **New document for same policy period** (different document, same policy): existing policy row's `valid_to` is set to `now()`; new row created with `valid_from = now()` and new `source_document_id`.

3. **Same document, metadata update only**: update `extraction_confidence`, `extraction_model`, `updated_at` in place without versioning.

**Worked example — same policy document uploaded twice**:

```
Upload 1: "Marshall_Wells_GL_2025.pdf"
  → insurance_policies row created: id=uuid1, valid_from=2026-01-01, valid_to=NULL

Upload 2 (identical file, same prompt version):
  → No-op. Existing row unchanged.

Upload 2 (identical file, new prompt version v2):
  → Row uuid1: valid_to=2026-02-01
  → New row uuid2: valid_from=2026-02-01, valid_to=NULL, enrichment_version=v2
```

### Low-Confidence Review Workflow

```
confidence < 70 (configurable threshold)
    │
    ▼
Route to review queue
User sees: "I found relevant information but I'm not fully confident.
           [Show tentative answer] [Flag for review]"
    │
    ├── User accepts → logged as "user_accepted"
    ├── User flags → review queue entry for admin
    └── Admin reviews → corrects answer → correction stored
```

**`review_items` table**:

| Column | Type | Description |
|---|---|---|
| id | TEXT PK | UUID |
| review_type | TEXT | "low_confidence_answer", "account_match_pending", "canonicalization_failure", "unknown_document_type" |
| query | TEXT | Original question (if applicable) |
| tentative_answer | TEXT | System's best answer |
| confidence | REAL | System confidence score |
| review_status | TEXT | "pending", "accepted", "corrected", "dismissed" |
| corrected_answer | TEXT | Admin-provided correction |
| reviewer_id | TEXT FK | → users.id |
| created_at, resolved_at | DATETIME | |

### PII Controls (v1.0 Scope)

**v1.0 implementation**:
- Encryption key sourced from environment variable `VAULTIQ_ENCRYPTION_KEY` — never hardcoded, never committed to source control
- Fernet symmetric encryption applied to: `named insured`, `management_contact`, `agent_email`, `claimant` (in insurance_claims)
- Decrypt operations logged as `PII_DECRYPT` audit events with user ID and timestamp
- SSN, EIN, and bank account numbers redacted from Claude context chunks at query time via regex patterns before any API call

**Redaction patterns (minimum)**:
- SSN: `\b\d{3}-\d{2}-\d{4}\b`
- EIN: `\b\d{2}-\d{7}\b`
- Bank account: `\b\d{8,17}\b` (contextual — only when preceded by "account number", "acct #", etc.)

**v1.1 deferred**: KMS integration (AWS KMS or HashiCorp Vault), key rotation policy, per-role access policy, per-field audit trail. These are required before handling more than 5 live customers.

### API Degradation Behavior (Standard Tier)

| Scenario | Behavior | Pass/fail signal |
|---|---|---|
| Claude API timeout > 15s | Return cached answer (24h TTL) or route to Ollama fallback | Response field `"source": "cache"` or `"source": "ollama_fallback"` |
| Claude API down | All queries routed to Ollama; status field set | Response field `"degraded": true` |
| Cost cap reached | Restrict to SQL-only answers; Sonnet queries rejected | HTTP 200 with `"mode": "sql_only"` |
| Partial outage (Haiku up, Sonnet down) | Route all queries through Haiku | Response field `"quality_reduced": true` |

---

## Automation Features (Phase 2)

### Renewal Summary Generation

Pure SQL aggregation — no AI required:

```python
async def generate_renewal_summary(self, account_id: str) -> RenewalSummary:
    account = self.get_account(account_id)
    policies = self.get_active_policies(account_id)
    claims   = self.get_claims_history(account_id, years=5)
    return RenewalSummary(
        named_insured=account.name,
        coverage_schedule=[{
            "line": p.line_of_business,
            "carrier": p.carrier,
            "policy_number": p.policy_number,
            "effective": p.effective_date,
            "expiration": p.expiration_date,
            "premium": p.annual_premium,
            "limits": self.get_coverage_limits(p.id),
        } for p in policies],
        total_premium=sum(p.annual_premium or 0 for p in policies),
        claims_summary={
            "total_claims": len(claims),
            "total_incurred": sum(c.total_incurred or 0 for c in claims),
            "open_claims": sum(1 for c in claims if c.status == "open"),
        },
    )
```

### Compliance Gap Detection

```python
async def check_compliance(self, account_id: str) -> list[ComplianceGap]:
    for req in self.get_requirements(account_id):
        coverage = self.find_matching_coverage(self.get_active_policies(account_id), req.coverage_line)
        if not coverage:
            yield ComplianceGap(requirement=req, status="missing")
        elif req.min_limit and coverage.limit_amount < req.min_limit:
            yield ComplianceGap(
                requirement=req, status="insufficient",
                current=coverage.limit_amount,
                required=req.min_limit,
                gap=req.min_limit - coverage.limit_amount,
            )
```

### Loss History Consolidation

```sql
SELECT c.*, p.line_of_business, p.carrier
FROM insurance_claims c
LEFT JOIN insurance_policies p ON c.policy_id = p.id
WHERE c.account_id = :account_id
  AND c.date_of_loss >= :cutoff
ORDER BY c.date_of_loss DESC
```

---

## Configuration

### Settings (`config.py`)

```python
# Deployment
deployment_tier: Literal["standard", "enterprise"] = "enterprise"

# Claude Enrichment (both tiers — Claude is primary)
claude_enrichment_enabled: bool = True
claude_api_key: str | None = None                          # from ANTHROPIC_API_KEY env var
claude_enrichment_model: str = "claude-sonnet-4-6"        # pinned
claude_enrichment_batch_size: int = 8
claude_enrichment_max_retries: int = 3
claude_enrichment_timeout: int = 60
claude_enrichment_cost_limit_usd: float = 10.0            # daily cap

# Claude Query-Time (Standard tier primary)
claude_query_enabled: bool = False
claude_query_model_simple: str = "claude-haiku-4-5-20251001"  # pinned
claude_query_model_complex: str = "claude-sonnet-4-6"         # pinned
claude_query_cost_limit_usd: float = 50.0                     # monthly cap

# Ollama (Enterprise primary / Standard fallback)
ollama_base_url: str = "http://localhost:11434"
chat_model: str = "qwen3-rag"

# Query Router
structured_query_enabled: bool = True
structured_query_row_cap: int = 1000
structured_query_timeout_seconds: int = 5

# Insurance Schema
insurance_schema_enabled: bool = True
insurance_auto_link_accounts: bool = True

# Database (PostgreSQL mandatory for hosted/spark)
database_url: str = "postgresql://localhost/vaultiq"
database_backend: Literal["sqlite", "postgresql"] = "postgresql"
pgvector_dimension: int = 768
pgvector_index_type: str = "ivfflat"                      # hnsw for >20K vectors
pgvector_lists: int = 100
pgvector_probes: int = 10

# Tenant
active_modules: list[str] = ["core"]
tenant_config_path: str = "tenant-instances/{tenant_id}/tenant.json"
```

### Profile Defaults

```python
PROFILE_DEFAULTS = {
    "laptop": {
        # Developer convenience only — insurance SQL and pgvector NOT available
        "deployment_tier": "enterprise",
        "claude_enrichment_enabled": False,    # Avoid API costs in dev
        "claude_query_enabled": False,
        "structured_query_enabled": False,     # No insurance schema in SQLite
        "insurance_schema_enabled": False,
        "database_backend": "sqlite",
        "vector_backend": "chroma",
        # Tests requiring insurance schema must use requires_postgres marker
    },
    "spark": {
        # Enterprise tier — air-gapped, Ollama for queries
        "deployment_tier": "enterprise",
        "claude_enrichment_enabled": True,
        "claude_query_enabled": False,
        "structured_query_enabled": True,
        "insurance_schema_enabled": True,
        "database_backend": "postgresql",
        "vector_backend": "pgvector",
        "claude_enrichment_model": "claude-sonnet-4-6",
        "chat_model": "qwen3-rag",
    },
    "hosted": {
        # Standard tier — single-tenant cloud, Claude for everything
        "deployment_tier": "standard",
        "claude_enrichment_enabled": True,
        "claude_query_enabled": True,
        "structured_query_enabled": True,
        "insurance_schema_enabled": True,
        "database_backend": "postgresql",
        "vector_backend": "pgvector",
        "claude_enrichment_model": "claude-sonnet-4-6",
        "claude_query_model_simple": "claude-haiku-4-5-20251001",
        "claude_query_model_complex": "claude-sonnet-4-6",
        "chat_model": "qwen3-rag",             # Ollama fallback
        "rag_enable_hallucination_check": False,
    },
}
```

---

## Current Codebase Transition

### What Already Exists

| Component | Status | Notes |
|---|---|---|
| FastAPI backend, JWT auth, RBAC | Production-ready | Carries forward unchanged |
| Tag-based access control | Production-ready | Pre-retrieval filtering intact |
| ProcessingService factory | Solid | Enrichment hook slots in after chunking |
| ARQ/Redis async workers | Solid | Enrichment runs as background tasks |
| RAGService (routing, confidence, citations) | Solid | Extend, not replace |
| Auto-tagging pipeline | Solid | Repurpose for document type detection |
| Audit logging (3-level) | Production-ready | No changes needed |
| React frontend (Vite + TypeScript) | Solid | Add insurance views, tenant config load |
| Evaluation framework (RAGAS) | Exists | Add insurance gold set |

### What Needs to Be Built

| Component | Priority |
|---|---|
| `ClaudeEnrichmentService` | P0 |
| `InsuranceDataService` + entity mapper | P0 |
| `QueryRouter` (rule engine + SQL execution) | P0 |
| `PgVectorService` (replaces Qdrant) | P0 |
| `ModuleRegistry` | P0 |
| Insurance SQLAlchemy models | P0 |
| Alembic migration system | P0 |
| `TenantConfigResolver` + `PromptResolver` | P1 |
| `ClaudeQueryService` | P1 |
| `ClaudeModelRouter` | P1 |
| `CanonicalizedEntityMapper` (Contract enforcement) | P1 |
| 3-tier account matching service | P1 |
| `ProcessingService` — enrichment hook | P0 (modify) |
| `RAGService` — router + tier branching | P0 (modify) |
| React — insurance dashboard views | P1 |
| React — tenant config load at startup | P1 |
| Air-gap transfer CLI (Mode A) | **v1.1 only** |
| Provisioning scripts | P2 |

### Integration Points

#### `ProcessingService` — Enrichment Hook

```python
async def process_document(self, document_id: str, ...):
    # ... existing Docling parsing (unchanged) ...

    if settings.claude_enrichment_enabled:
        prompts = self.prompt_resolver.get_all_prompts()
        enrichment = await self.claude_enrichment_service.enrich_document(
            document_id=document_id, chunks=chunks,
            filename=document.original_filename, prompts=prompts,
        )
        # Canonicalization applied inside ingest_entities
        await self.insurance_data_service.ingest_entities(
            document_id=document_id,
            synopsis=enrichment.synopsis,
            entities=enrichment.entities,
        )
        chunks_for_indexing = enrichment.enriched_chunks
    else:
        chunks_for_indexing = chunks   # Ollama fallback — RAG only

    await self.vector_service.add_document(
        document_id=document_id, chunks=chunks_for_indexing, tags=tags,
    )
```

#### `RAGService` — Router and Tier Branching

```python
async def generate(self, request: RAGRequest) -> RAGResponse:
    if settings.structured_query_enabled:
        intent = self.query_router.classify(request.query)
        if intent in (QueryIntent.STRUCTURED, QueryIntent.COMPARISON):
            return await self.handle_structured_query(request)
        elif intent == QueryIntent.HYBRID:
            request.additional_context = await self.get_sql_context(request)

    if settings.claude_query_enabled:
        model = self.model_router.select_model(intent)
        try:
            return await self._claude_generate(request, model=model)
        except ClaudeAPIError:
            return await self._ollama_generate(request)   # fallback
    else:
        return await self._ollama_generate(request)        # Enterprise primary
```

---

## Implementation Phases

### Phase 1 — Core Platform (Weeks 1-4)

**Week 1: Database + Schema**
- Alembic migration system (replaces hand-rolled `ALTER TABLE` list)
- PostgreSQL + pgvector for `hosted` and `spark` profiles
- SQLite preserved for `laptop` (with `@pytest.mark.requires_postgres` test marker)
- `chunk_vectors` table with `vector(768)`, IVFFlat index
- SQLAlchemy models for all insurance tables
- Module registry scaffold
- Gold evaluation harness (16 Marshall Wells questions, automated scoring)

**Week 2: Claude Enrichment**
- `ClaudeEnrichmentService` — two-call design, prompt caching
- `PromptResolver` — layered lookup (tenant → vertical → core)
- `CanonicalizedEntityMapper` — implements Canonicalization Contract
- Entity-to-SQL mapper with idempotency (upsert by idempotency keys)
- 3-tier account matching service
- Data lifecycle — soft-delete, versioning, cascade rules
- Ollama fallback when Claude API unavailable

**Week 3: Query Router + Standard Tier**
- Rule-based query router (Step 1–4 of Routing Specification)
- SQL template catalog (all 8 required templates)
- SQL execution safety layer
- `ClaudeQueryService` with pinned model IDs
- `TenantConfigResolver` — 3-tier merge
- Tenant config API endpoint (`GET /api/tenant/config`)
- React: tenant branding + feature flag loading at startup
- Query cost tracking and configurable caps
- Confidence score implementation and calibration against gold set

**Week 4: Integration + Deployment**
- Wire enrichment into `ProcessingService`
- Wire router + tier branching into `RAGService`
- `community_associations` module (repackages Marshall Wells work)
- End-to-end test with Marshall Wells documents (both tiers)
- Docker Compose customer template
- API degradation behavior (Ollama fallback, cost-cap SQL-only mode)

### Phase 2 — Automation + Verticals (Weeks 5-8)

**Week 5**: Renewal summary, compliance gap detection, loss history consolidation, `/api/insurance/*` endpoints.

**Week 6**: Frontend insurance dashboard (account overview, coverage schedule, compliance gap report, cost dashboards).

**Week 7**: `construction` and `real_estate` vertical modules; custom fields end-to-end; prompt override tested.

**Week 8**: Provisioning automation, monitoring aggregation, `transportation` and `healthcare` module scaffolds.

### v1.1 Scope (Deferred from v1.0)

- Mode A air-gap transfer (signed pg_dump + HMAC manifest + atomic pg_restore)
- Hot-reload for tenant config and prompt changes
- KMS integration for PII encryption key management
- Per-role access policies for PII decryption

---

## Cost Model

### Per-Document Enrichment (Both Tiers)

| Component | Tokens | Cost |
|---|---|---|
| Synopsis call | ~2,000 in / ~500 out | ~$0.01 |
| Chunk batch calls (×3, 8 chunks) | ~8,000 in / ~2,000 out | ~$0.05 |
| **Gross total** | | **~$0.06** |
| **With prompt caching (~35%)** | | **~$0.04** |

### Query Costs — Standard Tier Only

| Query Type | Model | Cost/Query |
|---|---|---|
| Structured lookup | `claude-haiku-4-5-20251001` | ~$0.001 |
| Analytical / comparison | `claude-sonnet-4-6` | ~$0.015–$0.03 |

**Estimated monthly (Standard tier, 3 agents / 75 queries/day)**: ~$18/month.

---

## Risk Flags

### ENTITY_ACCURACY
- [ ] Confidence scoring per entity; low-confidence entities quarantined (not written to SQL)
- [ ] Canonicalization Contract enforced — validation failures flagged for review
- [ ] Source document citation for every SQL row
- [ ] Carrier alias table seeded with top 50 US P&C carriers before launch

### API_DEPENDENCY
- [ ] Ollama fallback tested in both enrichment and query paths
- [ ] Cost tracking with configurable daily (enrichment) and monthly (query) caps
- [ ] Batch retry logic (3 retries with exponential backoff)
- [ ] Answer cache (24h TTL) for Standard tier degradation

### DATA_STALENESS
- [ ] Idempotency keys defined per table; re-upload tested
- [ ] Document deleted/replaced → cascade soft-delete verified
- [ ] Versioning (valid_from/valid_to) transitions tested with worked example

### ACCOUNT_MATCHING
- [ ] 3-tier matching tested against known edge cases (legal suffix variants, abbreviations)
- [ ] Tier 2 (pending merge) review queue surfaced in admin UI
- [ ] Audit log events for all three tiers

### SINGLE_TENANT_OPS
- [ ] Automated provisioning script (database, app, DNS, TLS, module selection)
- [ ] Rolling update strategy defined
- [ ] Per-customer backup and restore tested

### MODULE_ISOLATION
- [ ] Core never imports directly from module packages (only `ModuleRegistry` does)
- [ ] Module removal leaves core data intact

---

## Acceptance Criteria

### Phase 1 — Core Platform

All acceptance criteria are measurable against the Marshall Wells gold set (16 questions) unless otherwise noted.

- [ ] **Enrichment populates SQL**: Upload `Marshall_Wells_GL_2025.pdf` → `insurance_policies` contains a row with `line_of_business="gl"`, non-null `carrier`, `effective_date`, `expiration_date`. Pass signal: `SELECT COUNT(*) FROM insurance_policies WHERE source_document_id = :id AND is_deleted = FALSE` returns ≥ 1.
- [ ] **pgvector storage**: After enrichment, `SELECT COUNT(*) FROM chunk_vectors WHERE document_id = :id` returns > 0.
- [ ] **SQL JOIN works**: `SELECT cv.enriched_text, p.carrier FROM chunk_vectors cv JOIN insurance_policies p ON cv.document_id = p.source_document_id LIMIT 1` returns a row without error.
- [ ] **Structured query latency**: "What are the GL limits for Marshall Wells?" → SQL path → response in < 500ms (measured from API request receipt to response bytes sent). Measured with `pytest-benchmark` against a seeded PostgreSQL test database.
- [ ] **Router takes SQL-first path**: When SQL returns sufficient data, `response.meta.query_path == "sql"`. When SQL returns empty, `response.meta.query_path == "rag"`.
- [ ] **Cost tracking**: After enriching 1 document, `GET /api/admin/costs` returns `enrichment_cost_usd > 0.0`.
- [ ] **Batch enrichment CLI**: `python -m ai_ready_rag.cli enrich --all` completes without error against a seeded database.
- [ ] **Ollama fallback**: When `ANTHROPIC_API_KEY` is unset, document upload completes (Ollama path); no 500 errors. Query endpoints return 200 with `"source": "ollama"`.
- [ ] **Gold set — Standard tier**: All 16 gold set questions return `confidence >= 90`. Measured by `tests/eval/eval_runner.py` against a PostgreSQL database seeded with Marshall Wells documents.
- [ ] **Gold set — Enterprise tier**: All 16 gold set questions return `confidence >= 70` via Ollama + enriched chunks.
- [ ] **Data lifecycle**: Upload same document twice → `SELECT COUNT(*) FROM insurance_policies WHERE source_document_id = :id AND is_deleted = FALSE` returns same count as after first upload (no duplicates).
- [ ] **Cascade delete**: Delete a document → `SELECT COUNT(*) FROM insurance_policies WHERE source_document_id = :id AND is_deleted = FALSE` returns 0.
- [ ] **Provenance**: Every row in `insurance_policies` has non-null `source_document_id`, `extraction_model`, `valid_from`.
- [ ] **Review queue routing**: Submit a query against an empty database → response has `confidence < 70` → `SELECT COUNT(*) FROM review_items WHERE review_status = 'pending'` returns ≥ 1.
- [ ] **SQL safety**: Calling `QueryRouter` with a query that matches no template → response has `"mode": "rag_fallback"`, no SQL error raised.
- [ ] **Module registry**: Start application with `active_modules: ["core", "community_associations"]` → no startup error; `GET /health` returns 200; `ModuleRegistry.active_modules` contains both entries.
- [ ] **Tenant branding**: `GET /api/tenant/config` returns `ui.brand.name` matching `tenant.json`; frontend `<title>` renders tenant name, not "VaultIQ".
- [ ] **Feature flag**: Set `"compliance_checking": false` in `tenant.json`, restart → `GET /api/insurance/compliance/:account_id` returns HTTP 403.
- [ ] **Canonicalization**: Upload a document where Claude returns `"$1,000,000"` as a limit → `insurance_coverages.limit_amount = 1000000.0` (REAL, no formatting).
- [ ] **Account matching Tier 1**: Upload document with named insured "Marshall Wells Lofts Condominium" when "Marshall Wells Lofts" already exists → auto-linked (no review queue entry).
- [ ] **Account matching Tier 2**: Upload document with named insured "Marsh Wells Lofts" (70% similarity) when "Marshall Wells Lofts" exists → `review_items` entry created with `review_type = "account_match_pending"`.
- [ ] **Config reload**: Update `tenant.json`, restart → `GET /api/tenant/config` reflects updated values; `audit_log` contains `CONFIG_RELOAD` event.

### Phase 1 — Standard Tier

- [ ] **Claude query routing**: `GET /api/chat/sessions/:id/messages` with "What are the GL limits?" → response `meta.model` is `claude-haiku-4-5-20251001`.
- [ ] **Analytical routing**: "Summarize coverage gaps" → response `meta.model` is `claude-sonnet-4-6`.
- [ ] **Single-tenant isolation**: Two instances with different `tenant_id` values → `SELECT COUNT(*) FROM insurance_accounts` in instance A is unaffected by uploads to instance B.
- [ ] **Cost cap enforcement**: Set `monthly_query_cap_usd: 0.001`, submit 5 Sonnet queries → responses after cap is reached return `"mode": "sql_only"`.
- [ ] **Degradation — Claude down**: Unset `ANTHROPIC_API_KEY`, submit query → response returns 200 with `"degraded": true` and Ollama-generated answer.

### Phase 1 — Enterprise Tier (Mode B)

- [ ] **Mode B ingestion**: On Spark with internet, upload document → Claude enrichment completes → `enrichment_status = "completed"` in documents table.
- [ ] **Mode B query**: After enrichment, disconnect internet → query returns answer from Ollama + enriched chunks with `confidence >= 70`.

### Phase 2

- [ ] **Renewal summary**: `GET /api/insurance/accounts/:id/renewal-summary` returns JSON with `coverage_schedule`, `total_premium`, `claims_summary` — all fields non-null for a seeded account.
- [ ] **Compliance gap**: `GET /api/insurance/accounts/:id/compliance` returns `gaps` array; for a seeded account where GL limit < CC&R minimum, gap entry has `status = "insufficient"` and correct `gap_amount`.
- [ ] **Custom fields round-trip**: Define `internal_job_code` in `tenant.json`, upload policy, verify `insurance_policies.custom_fields->'internal_job_code'` is populated by Claude extraction.
- [ ] **Prompt override**: Add override file to `customizations/prompts/policy_extraction.txt`, restart, re-enrich document → `enrichment_synopses.enrichment_version` contains override identifier.

---

## Operations (Stub)

The following components require operational runbooks. Runbooks are maintained as separate documents (`docs/runbooks/`) and referenced here by name. The spec defines ownership roles only.

| Component | Owner Role | Runbook |
|---|---|---|
| Customer provisioning (new instance) | Infrastructure lead | `docs/runbooks/provision-customer.md` |
| Customer backup and restore | Infrastructure lead | `docs/runbooks/backup-restore.md` |
| Rolling instance updates | Engineering lead | `docs/runbooks/rolling-update.md` |
| Incident response (API outage, data issue) | On-call engineer | `docs/runbooks/incident-response.md` |
| Cost cap alert response | On-call engineer | `docs/runbooks/cost-cap-alert.md` |
| Config reload procedure | Any engineer | Documented inline in Config Reload Policy section above |

Runbooks are required before the first production customer deployment. Provisioning and incident response runbooks are required before the end of Phase 1.

---

## Design Decisions (Resolved)

1. **Claude as primary LLM for ingestion**: Both tiers use Claude API for document enrichment. Ollama-only enrichment produces inferior entity extraction. Ollama is fallback for enrichment (API unavailable) and primary for query-time responses in Enterprise tier.

2. **PostgreSQL mandatory for production**: All `hosted` and `spark` deployments use PostgreSQL + pgvector. SQLite is permitted for `laptop` dev profile only, with documented gaps. No Qdrant, no Chroma in v1.0 production paths.

3. **Deterministic routing without LLM**: The query router uses ordered pattern matching rules, not an LLM call, to classify intent. This is faster, cheaper, and reproducible. The full decision table is in the Routing Specification section.

4. **Confidence is a formula, not a feeling**: The composite confidence score is defined with specific inputs, weights, and thresholds. Acceptance criteria are tied to this formula and calibrated on a fixed gold set.

5. **Only pinned model IDs**: Aliases and short names are forbidden in code. Model upgrades require a spec change and re-calibration.

6. **Mode A deferred to v1.1**: The air-gap transfer workflow adds significant engineering complexity. Mode B (upload while internet is temporarily available) satisfies the Enterprise tier requirement for v1.0.

7. **3-tier account matching**: Single-threshold fuzzy match was rejected for account matching. Tier 1 auto-links high-confidence matches, Tier 2 flags ambiguous matches for human review, Tier 3 auto-creates low-confidence new accounts.

8. **Canonicalization Contract before SQL**: Raw Claude output is never written to SQL. All monetary, date, carrier, and limit values pass through the Canonicalization Contract. Validation failures are quarantined to review queue.

9. **Config reload requires restart (v1.0)**: Hot-reload is deferred to v1.1. Restart is the documented procedure for all config and prompt changes. Expected downtime < 30 seconds per instance.

10. **Single-tenant isolation**: Each customer gets a dedicated Docker Compose instance — separate application, separate PostgreSQL database. More expensive to operate than multi-tenant but provides complete data isolation and per-customer customization.

11. **JSONB custom fields**: Tenant-specific metadata fields use JSONB extension columns, not per-customer schema migrations. This enables zero-code-deployment customization at the cost of losing strict column typing on custom fields.

12. **PII encryption key from env var (v1.0)**: KMS integration is deferred to v1.1. The encryption key is stored in `VAULTIQ_ENCRYPTION_KEY` environment variable. This is sufficient for a controlled early-customer deployment; it is not sufficient for >5 customers.

---

*Spec: VaultIQ Platform v1.1*
*Supersedes: INSURANCE_AI_PLATFORM_v1.md, CLAUDE_ENRICHMENT_PIPELINE_v1.md*
*Status: DRAFT — engineering review incorporated; ready for final review before issue creation*
