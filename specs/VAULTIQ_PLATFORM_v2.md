---
title: "VaultIQ — Core Platform Specification"
status: FINAL - Ready for Implementation (Core Platform)
version: v2.0
created: 2026-02-27
updated: 2026-02-27
author: —
type: Fullstack
complexity: COMPLEX
supersedes: VAULTIQ_PLATFORM_v1.md, INSURANCE_AI_PLATFORM_v1.md, CLAUDE_ENRICHMENT_PIPELINE_v1.md
scope: Core Platform Layer Only — vertical-specific content lives in module specs
---

# VaultIQ — Core Platform Specification (v2.0)

---

## Architecture Contract

This section defines the non-negotiable boundary between the core platform and vertical modules. Every engineer working on this codebase must read this before writing any code.

### The Rules

1. **Core never imports from vertical modules.** The core platform package (`ai_ready_rag/core/`, `ai_ready_rag/services/`, `ai_ready_rag/api/`) must never contain an import of anything from `ai_ready_rag/modules/<vertical>/`. The only permitted coupling point is the `ModuleRegistry` API.

2. **Vertical modules never modify core files.** A module ships as a self-contained package. It registers its capabilities by calling `ModuleRegistry` APIs at startup. It never patches, monkey-patches, or modifies core platform code.

3. **All vertical-specific schema lives in module packages.** Core owns only the tables defined in this spec. Every table specific to an industry vertical is defined and migrated by that vertical's module package. Core Alembic migrations never reference vertical table names.

4. **All vertical-specific classifiers, prompts, templates, and compliance rules live in module packages.** Core provides the loader infrastructure and dispatch APIs; it never hardcodes industry-specific patterns, rules, or vocabulary.

5. **The ModuleRegistry is the only coupling point.** Core discovers what modules provide by calling `ModuleRegistry.get_classifiers()`, `ModuleRegistry.get_entity_map()`, `ModuleRegistry.get_sql_templates()`, and `ModuleRegistry.get_compliance_checker()`. These are the four and only four extension points.

### What Module Developers Can Rely On (The Core Contract)

When you build a vertical module, you can rely on the following from the core platform:

> **Note**: The guarantees in this section describe the runtime contract once Phase 1 core platform work is complete. During active development, check `DEVELOPMENT_PLANS.md` for current implementation status of each component.

- `ModuleRegistry` is initialized and available at application startup before any module `register_*` call is made.
- `PromptResolver` will execute its 3-tier lookup (tenant override → vertical → core base) for any prompt name you register; you only need to provide the vertical-tier prompt file.
- `TenantConfigResolver` will deep-merge your module's default config into the resolved tenant config at startup.
- `ClaudeEnrichmentService` will call your registered prompts during enrichment and pass extracted entities back through your registered `entity_map` for SQL insertion into your module's tables.
- `QueryRouter` will execute any SQL template you register via `register_sql_templates()` using the same safety layer (allowlist, parameterized bindings, 5s timeout, 1000-row cap, read-only connection) that governs all templates.
- The `enrichment_synopses`, `enrichment_entities`, `chunk_vectors`, `documents`, and `review_items` tables are always present and stable — you may FK reference `documents.id` from your module's tables.
- The confidence scoring formula (defined in this spec) is applied uniformly — your module does not compute confidence.
- API degradation handling (Claude → cache → Ollama fallback) is handled by core — your module receives a result, it does not manage fallback chains.

---

## Product Vision

VaultIQ is a multi-vertical AI platform that transforms how knowledge-intensive businesses manage documents and access structured information. By combining intelligent document ingestion, structured data extraction, and natural-language querying, VaultIQ gives any user instant access to every fact across every document in their instance.

**The platform problem**: Knowledge workers in regulated industries manage large volumes of structured documents — policies, contracts, compliance records, financial statements, certificates — and spend significant time manually locating facts that should be instantly queryable. Document details live inside PDF text, not in queryable databases. Structured reports require manual assembly. Compliance checking is manual and often skipped.

**The platform solution**: A dual-path system where structured facts are extracted once during ingestion and stored in vertical-specific SQL tables for deterministic lookup, while unstructured content is enriched with AI-generated summaries and entities for high-quality semantic search. A deterministic query router directs each question to the optimal path. A modular vertical system makes the platform re-deployable across industries without code duplication.

**The deployment model**: 39,000+ potential customers across six target verticals. No incumbent provides AI-powered document intelligence that combines deterministic SQL lookup with semantic RAG in a single system. VaultIQ creates a new category: **Domain Knowledge Intelligence**, deployable as a hosted SaaS instance or an air-gapped on-premises system.

---

## Deployment Tiers

VaultIQ is offered in two deployment tiers that share the same core platform code:

| Capability | Standard (Hosted) | Enterprise (Air-Gapped) |
|---|---|---|
| **Deployment** | Cloud VM per customer (Docker Compose) | DGX Spark on-prem |
| **Tenant Model** | Single-tenant (isolated instance per customer) | Single customer |
| **Ingestion LLM** | Claude API (Sonnet) — primary | Claude API (Sonnet) — primary |
| **Query-time LLM** | Claude API (Haiku/Sonnet) — primary | Ollama (local 8B) — primary |
| **Ollama Role** | Fallback when Claude API unavailable | Primary query engine |
| **Database** | PostgreSQL + pgvector | PostgreSQL + pgvector |
| **Internet Required** | Always (for queries) | Only during ingestion (Mode B, v1.0) |
| **Answer Quality** | Highest (Claude) | Good (enriched chunks + Ollama) |
| **Data Location** | Encrypted cloud (US), customer-isolated | Customer's premises |
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

1. **Only pinned IDs in code**: `claude-sonnet-4-6` is the Sonnet model for this codebase. Do not use `claude-sonnet`, `sonnet`, or dated variants anywhere in code, config, or tests.
2. **Model upgrades require a spec change**: Changing a model ID requires updating this table, bumping the spec version, and re-running the gold evaluation set before deploying.
3. **Environment variable override permitted**: `CLAUDE_ENRICHMENT_MODEL` and `CLAUDE_QUERY_MODEL_*` env vars may override pinned IDs for testing, but defaults in `config.py` must match this table exactly.
4. **No aliases in acceptance artifacts**: Gold set rubrics, evaluation reports, and CI scripts must reference the pinned ID, not a generic name.

---

## Architecture: The 3-Tier Layered System

### Overview

VaultIQ is built on a **3-tier configuration hierarchy** that enables a single codebase to serve multiple industries and be deeply customized per customer — without forking code.

```
┌─────────────────────────────────────────────────────────────────┐
│  Tier 1: Core Platform                                          │
│                                                                 │
│  Universal capabilities shared by every customer, every         │
│  vertical. Document ingestion, Claude enrichment, query         │
│  router, access control, audit, confidence scoring.             │
│  This tier never contains vertical or customer-specific code.   │
└─────────────────────────────────┬───────────────────────────────┘
                                  ↓  extended by
┌─────────────────────────────────────────────────────────────────┐
│  Tier 2: Vertical Modules                                       │
│                                                                 │
│  Industry-specific packages that add document classifiers,      │
│  Claude extraction prompts, SQL schema migrations, SQL query     │
│  templates, compliance rule sets, and automation endpoints      │
│  for a specific industry niche.                                 │
│  Each module is a self-describing package loaded at startup     │
│  via the ModuleRegistry. A 4-person team can support            │
│  5+ verticals simultaneously.                                   │
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

**Not a monolith with embedded vertical logic**: `if vertical == "insurance":` scattered through core code is untestable, unmaintainable, and unsellable as modular add-ons.

**Configuration as data**: The critical design principle is that vertical differences and customer customizations are *data* (YAML rules, prompt files, JSON config), not code. Adding a new vertical is a new module package. Customizing a customer's interface or prompts requires no code deployment.

**Single-tenant isolation**: Each customer gets their own dedicated Docker Compose instance — separate application, separate PostgreSQL database, separate configuration. No cross-customer data sharing.

---

## Storage Architecture

### Mandatory Baseline (v1.0)

PostgreSQL + pgvector is the **required** storage backend for all production profiles. In v1.0, the `chroma` and `qdrant` backend paths will be deprecated; `pgvector` is required for all production profiles. Existing SQLite + Qdrant/Chroma deployments are supported only on the `laptop` dev profile during the transition.

| Profile | Database | Vector Store | Notes |
|---|---|---|---|
| `hosted` | PostgreSQL | pgvector (same DB) | Production — Standard tier |
| `spark` | PostgreSQL | pgvector (same DB) | Production — Enterprise tier |
| `laptop` | SQLite | In-memory / stub | **Development only** — see gap note |

**Laptop profile gap (explicit)**: The `laptop` profile uses SQLite for developer convenience (zero-dependency setup). Module SQL tables and pgvector-based SQL JOINs are **not available** in this profile. Tests that exercise the enrichment pipeline, structured query router, or vector JOINs **must** run against a PostgreSQL test database. The `laptop` profile is sufficient only for auth, document upload, basic RAG, and frontend development.

Any test asserting vertical schema behavior must declare `@pytest.mark.requires_postgres` and be skipped when `database_backend == "sqlite"`.

---

## Tier 2: Vertical Module System

### Architecture

```
ai_ready_rag/
├── core/                           ← Never touches vertical logic
└── modules/                        ← Vertical packages
    ├── registry.py                 ← Module loader (ModuleRegistry class)
    ├── community_associations/     ← LAUNCH VERTICAL (see MODULE_COMMUNITY_ASSOCIATIONS spec)
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

### Module Manifest Format

Every vertical module ships a `manifest.json` at its package root. Core reads this manifest at startup to validate the module before calling any registration method.

```json
{
  "name": "<module_name>",
  "version": "<semver>",
  "display_name": "<Human-readable name>",
  "document_types": ["<type_id>", "..."],
  "compliance_rules": true,
  "schema_migrations": ["<migration_file.py>", "..."],
  "prompts": ["<prompt_name>", "..."],
  "dashboard_views": ["<view_id>", "..."],
  "requires_core_version": ">=2.0"
}
```

**Manifest fields:**

| Field | Type | Description |
|---|---|---|
| `name` | string | Unique module identifier (snake_case). Used as the key in `active_modules` config. |
| `version` | string | Semver module version. Logged at startup; used for migration tracking. |
| `display_name` | string | Human-readable name for admin UI. |
| `document_types` | string[] | Document type IDs this module can classify. Core merges all modules' lists at startup. |
| `compliance_rules` | boolean | Whether this module provides a `ComplianceChecker` implementation. |
| `schema_migrations` | string[] | Alembic migration files in this module's `migrations/` directory. Applied in order during startup. |
| `prompts` | string[] | Prompt names this module provides. `PromptResolver` will find them in `modules/<name>/prompts/`. |
| `dashboard_views` | string[] | View IDs this module contributes to the frontend dashboard registry. |
| `requires_core_version` | string | Semver constraint against core platform version. Module load fails with a clear error if not satisfied. |

### The 4 Extension Points

`ModuleRegistry` exposes exactly four registration methods. These are the only approved mechanisms for a module to contribute capabilities to the core platform.

| Extension Point | Method | Format |
|---|---|---|
| Document classifiers | `register_document_classifiers(module_name, classifiers)` | YAML rule list loaded from `classifiers.yaml` |
| Entity-to-SQL map | `register_entity_map(module_name, entity_map)` | Dict mapping entity type strings to `(table_name, column_name)` tuples |
| SQL query templates | `register_sql_templates(module_name, templates)` | Dict mapping template ID strings to parameterized SQL strings |
| Compliance checker | `register_compliance_checker(module_name, checker)` | Instance implementing the `ComplianceChecker` protocol |

### ModuleRegistry Contract

```python
class ModuleRegistry:
    """
    The sole coupling point between the core platform and vertical modules.

    Core services call get_* methods to retrieve the merged superset of all
    registered capabilities from all active modules. Modules call register_*
    methods during their startup initialization phase.

    Core never imports module packages directly. This class is the only
    permitted import surface.
    """

    def register_document_classifiers(self, module_name: str, classifiers: list[dict]) -> None:
        """Register YAML-loaded classifier rules for this module's document types."""

    def register_entity_map(self, module_name: str, entity_map: dict[str, tuple[str, str | None]]) -> None:
        """
        Register entity type → (table, column) mappings.
        ClaudeEnrichmentService uses this to route extracted entities to SQL tables.
        table must be a table owned by this module's migrations.
        """

    def register_sql_templates(self, module_name: str, templates: dict[str, str]) -> None:
        """
        Register named, parameterized SQL query templates.
        All templates are merged into the QueryRouter's allowlist.
        Templates must use :param_name binding syntax — no string interpolation.
        """

    def register_compliance_checker(self, module_name: str, checker: ComplianceChecker) -> None:
        """Register a ComplianceChecker implementation for this module."""

    def get_classifiers(self) -> list[dict]:
        """Return merged classifiers from all active modules."""

    def get_entity_map(self) -> dict[str, tuple[str, str | None]]:
        """Return merged entity map from all active modules."""

    def get_sql_templates(self) -> dict[str, str]:
        """Return merged SQL template catalog from all active modules."""

    def get_compliance_checker(self, module_name: str) -> ComplianceChecker | None:
        """Return the compliance checker for a specific module, or None."""

    @property
    def active_modules(self) -> list[str]:
        """Return list of loaded module names."""
```

### Vertical Schema Extensions

Core platform owns only the tables listed in the "Core Data Model" section of this spec. Every vertical module is responsible for:

1. Providing its own Alembic migration files in `modules/<name>/migrations/`.
2. Running those migrations via the schema migration framework at application startup (before the module's `register_*` calls are made).
3. Never modifying core-owned tables via module migrations (adding columns, indexes, or constraints to core tables is forbidden).
4. Using `tenant_id` columns in all module tables for row-level tenant isolation.
5. Using `source_document_id` (FK → `documents.id`) as the provenance reference for any record created during enrichment.

Core applies module migrations in the order returned by `manifest.json`'s `schema_migrations` list. If a module migration fails, the module is not loaded and an error is logged. Other modules and core continue loading.

### Automation Framework

Vertical modules may register automation endpoints that generate structured output (renewal summaries, compliance reports, data exports). The architectural boundary is:

- **Automation services** (in module packages) query module SQL tables and return structured Python dicts or Pydantic models.
- **Core `DocumentRenderer`** owns all PDF and PPTX output generation. Automation services never directly generate binary files — they call `DocumentRenderer` APIs with structured data.
- **Core API router** mounts module automation endpoints under `/api/modules/<module_name>/` when a module is active. Module endpoint handlers are registered via a fifth (optional) registration method: `register_api_router(module_name, router)`.

Module automation pseudocode follows this pattern:
```python
# In module package — returns structured data, never PDF bytes
async def generate_account_summary(account_id: str, db: Session) -> AccountSummary:
    ...  # query module tables, return Pydantic model

# Core DocumentRenderer — called by module endpoint handler
summary_pdf = await document_renderer.render_pdf("account_summary", summary.dict())
```

This boundary means core can improve PDF rendering quality, templating engine, or branding injection without touching any module code.

---

## Tier 3: Tenant Customization Layer

### tenant.json Schema

```json
{
  "tenant_id": "<agency-slug>",
  "name": "<Display Name>",
  "modules": ["core", "<vertical_module_name>"],
  "ui": {
    "brand": {
      "name": "<Agency Name>",
      "logo_url": "/tenant/assets/logo.png",
      "primary_color": "#2563EB",
      "accent_color": "#F59E0B"
    },
    "layout": {
      "sidebar_items": ["dashboard", "documents", "search"],
      "default_view": "documents",
      "hide_features": []
    }
  },
  "custom_fields": {
    "<module_table_name>": [
      { "key": "<field_key>", "label": "<Display Label>", "type": "string", "searchable": true }
    ]
  },
  "features": {
    "<feature_key>": true
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

**Key `tenant.json` rules:**

- `modules` array controls which vertical modules are loaded for this tenant. Core is always included.
- `custom_fields` keys must be table names owned by an active module — core validates at startup.
- `ai` model IDs must be pinned IDs from the Model Governance table. `TenantConfigResolver` rejects aliases.
- Feature flag keys not recognized by any active module are logged as warnings and ignored.

### Config Reload Policy (v1.0)

**Restart required** for all tenant config and prompt changes. Hot-reload is deferred to v1.1.

| Action | Required procedure |
|---|---|
| Branding, feature flag, model cap changes | Update `tenant.json` → `docker compose restart api` |
| Prompt override added or modified | Add/update file in `customizations/prompts/` → restart |
| Module list change | Update `modules` array in `tenant.json` → restart |
| Expected downtime | < 30 seconds per instance |
| Audit log event | `CONFIG_RELOAD` — timestamp, operator, list of changed files |

Config reload must be logged at `INFO` level with the diff of changed keys (excluding secrets). On next startup, `TenantConfigResolver` re-reads all files and logs the resolved effective config.

### The 5 Customization Domains

**1. Branding & UI Theme** — CSS custom properties, logo, company name from `ui.brand`. Frontend calls `GET /api/tenant/config` at startup and applies CSS variables. Full white-labeling supported.

**2. Custom Fields** — JSONB extension columns on module entity tables:
```sql
ALTER TABLE <module_table>  ADD COLUMN custom_fields JSONB DEFAULT '{}';
```
Field definitions in `tenant.json` drive frontend rendering and are injected into Claude extraction prompts automatically — no code change required. Core injects the tenant's custom field definitions into the enrichment prompt context before calling the Claude API.

**3. Prompt Overrides** — The `PromptResolver` uses a layered lookup:
```
Tenant override → Vertical module default → Core base prompt
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

**4. Feature Flags** — Every backend endpoint and frontend component checks `tenant.features.<key>`. Missing keys default to `false` (deny by default for optional capabilities). Vertical modules register their feature flag keys via their manifest; core enforces the default-false behavior.

**5. Integrations & Webhooks** — AMS connectors and webhook endpoints configured in `tenant.json`. Connector code in `ai_ready_rag/integrations/` is loaded only for tenants with that integration enabled.

### Tenant Config Resolver

```python
class TenantConfigResolver:
    def __init__(self, tenant_id: str):
        self.core     = load_core_defaults()
        self.vertical = load_vertical_config(get_active_modules(tenant_id))
        self.tenant   = load_tenant_config(tenant_id)

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
│   └── tenant.py                     ← tenant config endpoint
├── core/
├── db/
│   ├── models/
│   │   └── vectors.py                ← chunk_vectors (pgvector)
│   └── migrations/                   ← Alembic (core migrations only)
├── services/
│   ├── rag_service.py                ← MODIFIED (router + tier branching)
│   ├── processing_service.py         ← MODIFIED (enrichment hook)
│   ├── claude_enrichment_service.py  ← NEW
│   ├── claude_query_service.py       ← NEW (Standard tier)
│   ├── claude_model_router.py        ← NEW
│   ├── query_router.py               ← NEW
│   └── pgvector_service.py           ← NEW (replaces Qdrant)
├── modules/                          ← NEW
│   ├── registry.py                   ← ModuleRegistry class
│   └── <vertical>/                   ← One package per vertical
├── tenant/                           ← NEW
│   ├── config.py
│   ├── resolver.py
│   └── api.py
└── integrations/                     ← NEW
    ├── base.py
    └── webhook.py

tenant-instances/                     ← Outside app package
└── {customer-slug}/
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
               Claude API Call #1: Document Synopsis
               Model: claude-sonnet-4-6
               Output: document_type, key facts per vertical prompt
                           │
               Claude API Call #2: Chunk Enrichment (batched, 8 chunks)
               Model: claude-sonnet-4-6
               Input: synopsis (cached prefix) + 8 chunks
               Output: { summary, entities[] } per chunk
                           │
               ┌───────────┴───────────┐
               │                       │
       Entity-to-SQL mapping    Enrich chunk text
       (ModuleRegistry.         "[SUMMARY] ... [ENTITIES] ... [ORIGINAL] ..."
        get_entity_map()
        → module tables)
               │                       │
       Module SQL tables         chunk_vectors (pgvector)
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
│  {customer}.vaultiq.app                          │
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
  1. Upload document
  2. Docling parse
  3. Claude enrich (API call while internet available)
  4. Store in local PostgreSQL + pgvector
  5. Disconnect internet
  6. Query locally via Ollama (no internet required)
```

**Mode A — Air-Gap Transfer is deferred to v1.1.**
The signed pg_dump + HMAC manifest + atomic pg_restore workflow (`transfer export/verify/import` CLI) is not in v1.0 acceptance scope. Enterprise tier v1.0 requires Mode B only.

---

## Core Data Model

Core platform owns the following tables. These tables are present in every VaultIQ instance regardless of which vertical modules are active. **Vertical modules must not add columns to these tables via module migrations.**

### `documents` (existing — additions below)

The existing `documents` table is extended with enrichment tracking columns:

| Column | Type | Description |
|---|---|---|
| enrichment_status | TEXT | `null` / `"pending"` / `"enriching"` / `"completed"` / `"failed"` |
| enrichment_model | TEXT | Pinned model ID used (e.g., `"claude-sonnet-4-6"`) |
| enrichment_version | TEXT | Prompt version string (used for re-enrichment change detection) |
| enrichment_tokens_used | INTEGER | Total tokens consumed by enrichment calls |
| enrichment_cost_usd | REAL | USD cost of enrichment (synopsis + all chunk batches) |
| enrichment_completed_at | DATETIME | Timestamp of successful enrichment completion |
| document_role | TEXT | Detected document role string (e.g., `"policy"`, `"contract"`, `"report"`). Value set is defined by the active module's classifier. |

`document_role` values are not hardcoded in core. The active module's `classifiers.yaml` defines what values are valid for its document types. Core stores whatever string the classifier assigns.

### `enrichment_synopses`

One row per enrichment run per document. Rows are never deleted — superseded rows remain with non-null `superseded_at`.

| Column | Type | Description |
|---|---|---|
| id | TEXT PK | UUID |
| document_id | TEXT FK | → documents.id |
| synopsis_text | TEXT | Full synopsis output from Claude Call #1 |
| document_type | TEXT | Detected type string (from active module's type list) |
| key_facts | JSONB | Array of key fact strings extracted by synopsis prompt |
| enrichment_model | TEXT | Pinned model ID |
| enrichment_version | TEXT | Prompt version |
| tokens_used | INTEGER | |
| cost_usd | REAL | |
| created_at | DATETIME | |
| superseded_at | DATETIME | NULL if current; set when re-enrichment produces a newer synopsis |

### `enrichment_entities`

One row per entity extracted per chunk per enrichment run. Entities are immutable once written; re-enrichment writes new rows and marks old ones superseded.

| Column | Type | Description |
|---|---|---|
| id | TEXT PK | UUID |
| document_id | TEXT FK | → documents.id |
| chunk_index | INTEGER | Which chunk this entity came from |
| entity_type | TEXT | Entity type string (defined by active module's entity map) |
| raw_value | TEXT | Raw Claude-extracted value before canonicalization |
| canonical_value | TEXT | Canonicalized value (or null if canonicalization failed) |
| confidence | REAL | 0.0–1.0 extraction confidence |
| context | TEXT | Contextual phrase from chunk |
| mapped_table | TEXT | SQL table this entity was written to (null if unmapped or failed) |
| mapped_column | TEXT | SQL column this entity was written to |
| created_at | DATETIME | |
| superseded_at | DATETIME | NULL if current |

### `chunk_vectors` (pgvector)

| Column | Type | Description |
|---|---|---|
| id | TEXT PK | UUID |
| document_id | TEXT FK | → documents.id |
| chunk_index | INTEGER | Position in document |
| chunk_text | TEXT | Original chunk text |
| enriched_text | TEXT | `[SUMMARY] ... [ENTITIES] ... [ORIGINAL] ...` concatenation |
| embedding | vector(768) | nomic-embed-text embedding of `enriched_text` |
| metadata | JSONB | tags, tenant_id, document_type, module_context |
| created_at | TIMESTAMP | |

**Indexes:**
```sql
CREATE INDEX ON chunk_vectors USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON chunk_vectors (document_id);
CREATE INDEX ON chunk_vectors USING gin (metadata);
```

**Cross-table JOIN pattern** (vector + module SQL data):

Core provides this JOIN as a query primitive. Module entity tables are joined via `document_id → source_document_id`:
```sql
SELECT cv.enriched_text,
       cv.embedding <=> :query_embedding AS distance,
       mt.*
FROM chunk_vectors cv
JOIN <module_table> mt ON cv.document_id = mt.source_document_id
WHERE cv.embedding <=> :query_embedding < :max_distance
  AND cv.metadata->>'tenant_id' = :tenant_id
ORDER BY distance
LIMIT :top_k;
```
The specific `<module_table>` and columns in `mt.*` are parameterized per template and registered by the vertical module.

### `review_items`

Holds items requiring human review from any subsystem (confidence routing, account matching, canonicalization failures, unknown document types).

| Column | Type | Description |
|---|---|---|
| id | TEXT PK | UUID |
| review_type | TEXT | `"low_confidence_answer"`, `"account_match_pending"`, `"canonicalization_failure"`, `"unknown_document_type"` |
| query | TEXT | Original question (if applicable) |
| tentative_answer | TEXT | System's best answer |
| confidence | REAL | System confidence score (0.0–1.0) |
| review_status | TEXT | `"pending"`, `"accepted"`, `"corrected"`, `"dismissed"` |
| corrected_answer | TEXT | Admin-provided correction |
| reviewer_id | TEXT FK | → users.id |
| module_context | TEXT | Which module's data was involved (for routing to correct admin view) |
| created_at | DATETIME | |
| resolved_at | DATETIME | |

---

## Claude Enrichment Pipeline

Claude (`claude-sonnet-4-6`) is the **primary intelligence layer** for document ingestion on both tiers. Ollama (`qwen3-rag`) is the fallback when the Claude API is unavailable.

### Processing Flow

```
1. Docling Parse + Chunk (existing infrastructure)
2. PromptResolver: load active prompts for tenant + vertical
3. Claude Call #1: Document Synopsis  (claude-sonnet-4-6)
4. Claude Call #2: Chunk Enrichment   (claude-sonnet-4-6, batched 8 chunks)
5. Canonicalization Contract applied to all extracted entity values
6. Entity-to-SQL mapping via ModuleRegistry.get_entity_map() → module tables
7. Enriched chunk text → chunk_vectors (pgvector)
8. Fallback: if Claude API unavailable → Ollama pipeline (RAG only, no SQL population)
```

### Claude API Call #1: Document Synopsis

```python
{
    "model": "claude-sonnet-4-6",
    "max_tokens": 1024,
    "system": [{"type": "text", "text": synopsis_system_prompt,
                 "cache_control": {"type": "ephemeral"}}],
    "messages": [{"role": "user", "content": f"""
Filename: {filename}
Total chunks: {chunk_count}
Representative content: {sampled_chunks}

Return JSON:
{{
  "synopsis": "200-400 word summary",
  "document_type": "<type from active module classifiers>",
  "document_subtype": "<subtype if applicable>",
  "key_facts": ["<fact 1>", "<fact 2>", "..."]
}}"""}]
}
```

The `synopsis_system_prompt` is loaded via `PromptResolver` using the name `"synopsis_system"`. The JSON fields requested in the user message are also injected by `PromptResolver` from the vertical module's synopsis prompt template — core provides the skeleton; the module populates the vertical-specific entity fields. This means the synopsis call structure above is illustrative; actual fields depend on the active module's prompt.

### Claude API Call #2: Chunk Enrichment (Batched)

```python
{
    "model": "claude-sonnet-4-6",
    "max_tokens": 4096,
    "system": [{"type": "text", "text": chunk_enrichment_system_prompt,
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
    {{"type": "<entity_type>", "value": "<raw_value>", "context": "<contextual phrase>"}}
  ]
}}]
{formatted_chunk_batch}"""}
    ]}]
}
```

Entity types in the `entities` array are defined by the active module's entity map. Core does not define entity types — it passes extracted entities through `ModuleRegistry.get_entity_map()` to determine where each entity routes in SQL.

### Prompt Caching Strategy

```
Call #1 (Synopsis):      System prompt cached            → ~30% input savings
Call #2a (Chunks 1-8):   System + synopsis cached        → ~40% input savings
Call #2b (Chunks 9-16):  System + synopsis cached        → ~40% input savings
```

**Estimated cost**: ~$0.06/document gross, ~$0.04/document with caching.

### `ClaudeEnrichmentService`

```python
class ClaudeEnrichmentService:
    """
    Orchestrates the two-call Claude enrichment pipeline.

    Receives chunks from ProcessingService, loads prompts via PromptResolver,
    makes Claude API calls with prompt caching, applies the Canonicalization
    Contract, and dispatches enriched entities to the entity mapper.

    Does not know what vertical is active. All vertical specifics arrive
    via the PromptResolver and entity_map from ModuleRegistry.
    """

    async def enrich_document(
        self,
        document_id: str,
        chunks: list[Chunk],
        filename: str,
        prompts: ResolvedPrompts,
        entity_map: dict[str, tuple[str, str | None]],
    ) -> EnrichmentResult:
        ...
```

### Canonicalization Contract

All entity values extracted by Claude **must** be canonicalized before writing to any SQL table. Raw Claude output is never written directly to SQL.

| Data Type | Canonical Form | Examples |
|---|---|---|
| **Dates** | ISO 8601 `YYYY-MM-DD` | `"1/1/2026"` → `"2026-01-01"` |
| **Monetary / numeric amounts** | REAL in base units, no formatting | `"$1M"` → `1000000.0`; `"$1,000,000.00"` → `1000000.0` |
| **Percentages** | REAL as decimal | `"80%"` → `0.80`; `"100"` → `1.00` |
| **Identifiers** | Strip leading/trailing whitespace; collapse internal runs to single space | `"  ID-029 4618 "` → `"ID-029 4618"` |
| **Enumerated values** | Map to registered enum values via active module's canonicalization rules | Module-specific; provided by `register_entity_map()` |
| **Name canonicalization** | Module-specific alias tables (e.g., carrier aliases for insurance) injected by module | Core applies the lookup; module provides the data |

**Validation failure behavior**: When an entity value cannot be canonicalized:
1. Entity is stored in `enrichment_entities` with `confidence = 0.0` and `canonical_value = null`
2. Entity is **not** written to any module SQL table
3. A `review_items` entry is created with `review_type = "canonicalization_failure"`
4. Processing continues — one bad entity does not fail the entire document

---

## Account Matching Algorithm

Account matching resolves a Claude-extracted primary entity name (e.g., named insured, company name, property name) to an existing account record in the active module's account table, or creates a new one.

Core provides the matching algorithm. The active module provides:
- The table name and name column to match against (via entity map)
- Legal/organizational suffix strings to strip before comparison (registered in entity map metadata)

### 3-Tier Decision Process

**Tier 1 — Auto-link (high confidence)**

Auto-link without human review when ANY of the following are true:
- Name similarity ≥ 95% (difflib.SequenceMatcher ratio) AND a corroborating signal matches (e.g., address, identifier on an existing document already linked to the account)
- An exact unique identifier match (e.g., policy number, account number) on any document already linked to an existing account
- Name similarity ≥ 98% regardless of other signals

Action: Link document to existing account. Log `ACCOUNT_MATCH_AUTO` audit event.

**Tier 2 — Flag for review (medium confidence)**

Flag when:
- Name similarity 75%–95% with no corroborating signal
- Multiple existing accounts score above 75% (ambiguous — more than one candidate)

Action: Document linked to a **provisional account** (a new account record with `status = "pending_merge_review"`). Admin review queue entry created. Log `ACCOUNT_MATCH_PENDING` audit event.

**Tier 3 — Auto-create (low confidence)**

When:
- Name similarity < 75% against all existing accounts
- No identifier evidence pointing to an existing account

Action: New account row created automatically. Log `ACCOUNT_CREATED` audit event.

### Implementation

```python
def compute_name_similarity(a: str, b: str, suffix_list: list[str]) -> float:
    """
    Strip organizational suffixes from both strings before comparison.
    suffix_list is provided by the active module's entity map metadata.
    Comparison is case-insensitive.
    """
    a_clean = strip_suffixes(a.lower(), suffix_list)
    b_clean = strip_suffixes(b.lower(), suffix_list)
    return SequenceMatcher(None, a_clean, b_clean).ratio()
```

The `suffix_list` is module-specific. The insurance vertical provides `["LLC", "Inc", "Corporation", "Condominium Association", "Homeowners Association", ...]`. A construction vertical might provide different suffixes. Core does not hardcode any suffix list — it accepts the list from the module's entity map registration.

---

## Query Router

### Confidence Score Definition

Every answer produced by the system carries a confidence score (0–100 integer). The score is a weighted composite:

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

**Thresholds:**

| Threshold | Value | Behavior |
|---|---|---|
| Review queue trigger | < 70 | Answer held in review queue; user sees tentative answer with flag option |
| Standard tier gold set acceptance | ≥ 90 | All gold set questions must meet or exceed this on first run |
| Enterprise tier gold set acceptance | ≥ 70 | All gold set questions must meet or exceed this on first run |

### Deterministic Routing Specification

The router never calls an LLM to choose a path. The decision is rule-based, executed in order. The first matching rule wins.

#### Step 1: Account Resolution

Extract account name from: (a) explicit mention in query text, (b) session context (active account). If no account can be resolved → `CONVERSATIONAL` intent. Proceed to LLM directly with chat history only.

#### Step 2: Entity Type Detection (ordered pattern matching)

Rules are evaluated in order; first match wins. The base rules below are always present. Modules may register additional patterns via the entity map.

| Priority | Pattern match on lowercased query | Intent |
|---|---|---|
| 1 | `compare \| vs \| versus \| difference between \| how do .+ compare` | `COMPARISON` |
| 2 | Matches a structured entity pattern registered by any active module | `STRUCTURED` |
| 3 | `list (all\|the) .+` \| `who is the .+` \| `when does .+` \| `is .+ covered` | `STRUCTURED` |
| 4 | `what does .+(say\|require\|state\|mean)` \| `summarize\|explain\|describe\|exclusion\|condition` | `ANALYTICAL` |
| 5 | Matches both a STRUCTURED and ANALYTICAL pattern | `HYBRID` |
| 6 | Default (no pattern match) | `ANALYTICAL` |

#### Step 3: SQL Execution and Sufficiency Check

For `STRUCTURED`, `COMPARISON`, and `HYBRID` intents: look up the matching SQL template from `ModuleRegistry.get_sql_templates()`. Execute via the SQL safety layer. SQL result is **sufficient** when:
- Result set has ≥ 1 row AND
- All primary fields for the question type are non-null

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

### SQL Template Registry

Modules register named, parameterized SQL query templates via `register_sql_templates()`. These templates are merged into the QueryRouter's allowlist at startup. The router executes **only** registered templates — no ad-hoc SQL generation ever occurs.

**Template requirements (enforced at registration):**
- Must use `:param_name` binding syntax — no string interpolation or f-strings
- Must include `LIMIT :row_cap` as the final clause
- Must not contain DML (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, etc.)
- Must reference only tables owned by the registering module (core validates FK presence at startup)

**Example template registration:**
```python
registry.register_sql_templates("my_vertical", {
    "my_lookup_query": """
        SELECT col_a, col_b, col_c
        FROM my_vertical_table
        WHERE tenant_id = :tenant_id
          AND account_id = :account_id
          AND is_deleted = FALSE
          AND valid_to IS NULL
        ORDER BY col_a
        LIMIT :row_cap
    """,
})
```

### Entity Pattern Registry

Modules register entity detection patterns via `register_entity_map()`. The entity map drives two behaviors:

1. **At enrichment time**: Core passes the map to `ClaudeEnrichmentService`, which uses it to route Claude-extracted entities to the correct module SQL tables.
2. **At query time**: Core uses entity type labels from the map to generate Step 2 pattern matching rules (in addition to the base rules in this spec).

**Entity map entry format:**
```python
{
    "<entity_type_string>": ("<module_table_name>", "<column_name_or_None>"),
    # column_name is None for complex entity types that require a
    # dedicated mapper method (e.g., entities that map to multiple columns)
}
```

Modules that require complex mapping logic (multi-column writes, conditional routing) implement a `EntityMapper` protocol method alongside the entity map registration.

### SQL Execution Safety

The following rules apply to every SQL template executed by the QueryRouter, regardless of which module registered it:

1. **Allowlisted templates only** — no arbitrary SQL generation, ever
2. **Parameterized bindings** — no string interpolation; all user values via bind params
3. **5-second timeout** — queries exceeding this are killed; fallback to RAG
4. **1,000-row cap** — `LIMIT :row_cap` in every template; pagination required beyond
5. **Read-only connection** — query-time SQL uses a read-only database role; writes during enrichment only
6. **Unmapped query behavior** — no template match → `structured_data_unavailable` flag set → fall through to RAG path; no error raised to user

### `ClaudeModelRouter`

```python
class ClaudeModelRouter:
    def select_model(self, intent: QueryIntent) -> str:
        if intent in (QueryIntent.STRUCTURED, QueryIntent.CONVERSATIONAL):
            return "claude-haiku-4-5-20251001"
        return "claude-sonnet-4-6"
```

Model selection is overridable per tenant via `ai.query_model_simple` and `ai.query_model_analytical` in `tenant.json` — but overrides must also use pinned IDs.

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

All module entity tables must use `is_deleted` + `deleted_at` for soft-delete. Hard-delete after configurable retention (default 90 days). This is a core governance requirement that modules must implement in their schemas.

**Current record filter**: Always `WHERE valid_to IS NULL AND is_deleted = FALSE`.

**Cascade rules (core enforces via enrichment pipeline):**
- Document deleted → soft-delete all module table rows with that `source_document_id`
- Module account deleted → soft-delete all child records linked to that account

### Idempotency and Re-enrichment

Re-enrichment is idempotent. The same document uploaded twice must not create duplicate SQL rows.

**Conflict resolution order:**

1. **Same document re-uploaded** (same content hash): no-op if `enrichment_version` matches current prompt version. If prompt version differs, supersede existing rows (set `valid_to = now()`) and create new rows with `valid_from = now()`.

2. **New document for same entity period** (different document, same logical record): existing row's `valid_to` is set to `now()`; new row created with `valid_from = now()` and new `source_document_id`.

3. **Same document, metadata update only**: update `extraction_confidence`, `extraction_model`, `updated_at` in place without versioning.

**Worked example — same document uploaded twice:**
```
Upload 1: "contract_2025.pdf"
  → Module table row created: id=uuid1, valid_from=2026-01-01, valid_to=NULL

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
    ├── User flags → review_items entry for admin
    └── Admin reviews → corrects answer → correction stored
```

### PII Controls (v1.0 Scope)

- Encryption key sourced from environment variable `VAULTIQ_ENCRYPTION_KEY` — never hardcoded, never committed to source control
- Fernet symmetric encryption applied to columns declared as `pii=True` in module entity map registrations — core handles encryption/decryption transparently
- Decrypt operations logged as `PII_DECRYPT` audit events with user ID and timestamp
- SSN, EIN, and financial account numbers redacted from Claude context chunks at query time via regex patterns before any API call

**Redaction patterns (minimum — always active):**
- SSN: `\b\d{3}-\d{2}-\d{4}\b`
- EIN: `\b\d{2}-\d{7}\b`
- Bank account: `\b\d{8,17}\b` (contextual — only when preceded by "account number", "acct #", etc.)

**v1.1 deferred**: KMS integration (AWS KMS or HashiCorp Vault), key rotation policy, per-role access policy, per-field audit trail. These are required before handling more than 5 live customers.

### API Degradation Behavior (Standard Tier)

| Scenario | Behavior | Signal |
|---|---|---|
| Claude API timeout > 15s | Return cached answer (24h TTL) or route to Ollama fallback | `"source": "cache"` or `"source": "ollama_fallback"` |
| Claude API down | All queries routed to Ollama; status field set | `"degraded": true` |
| Cost cap reached | Restrict to SQL-only answers; Sonnet queries rejected | HTTP 200 with `"mode": "sql_only"` |
| Partial outage (Haiku up, Sonnet down) | Route all queries through Haiku | `"quality_reduced": true` |

---

## Alembic Migration Framework

### Core Migrations

Core platform owns an Alembic `env.py` and migration chain in `ai_ready_rag/db/migrations/`. Core migrations create and modify only the core-owned tables (`documents`, `enrichment_synopses`, `enrichment_entities`, `chunk_vectors`, `review_items`, and the existing tables from the pre-VaultIQ codebase).

Core migration naming convention: `NNNN_core_<description>.py`

### Module Migration Integration

At application startup, after core migrations are applied, the `ModuleRegistry` applies each active module's migrations in declaration order from `manifest.json`:

```python
# Startup sequence (simplified)
run_alembic_upgrade("head", scope="core")  # core migrations only
for module_name in active_modules:
    module = registry.load_module(module_name)
    run_alembic_upgrade("head", scope=f"module:{module_name}",
                         migration_dir=module.migrations_path)
    module.register_all(registry)  # 4 extension points + optional router
```

Module migrations are isolated — they run against their own Alembic version table (`alembic_version_<module_name>`) so module migration state is tracked independently of core migration state.

**Module migration rules:**
- May create new tables (must include `tenant_id`, `is_deleted`, `deleted_at`, `valid_from`, `valid_to`, `source_document_id` columns per data governance requirements)
- May add indexes to module-owned tables
- May NOT add columns, indexes, or constraints to core-owned tables
- May NOT reference core-internal implementation details (may FK reference `documents.id` and `users.id`)

---

## Configuration

### Platform-Level Settings (`config.py`)

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

# Database (PostgreSQL mandatory for hosted/spark)
database_url: str = "postgresql://localhost/vaultiq"
database_backend: Literal["sqlite", "postgresql"] = "postgresql"
pgvector_dimension: int = 768
pgvector_index_type: str = "ivfflat"                      # hnsw for >20K vectors
pgvector_lists: int = 100
pgvector_probes: int = 10

# Tenant / Module
active_modules: list[str] = ["core"]
tenant_config_path: str = "tenant-instances/{tenant_id}/tenant.json"
```

### Profile Defaults

```python
PROFILE_DEFAULTS = {
    "laptop": {
        # Developer convenience only — module SQL tables and pgvector NOT available
        "deployment_tier": "enterprise",
        "claude_enrichment_enabled": False,    # Avoid API costs in dev
        "claude_query_enabled": False,
        "structured_query_enabled": False,     # No module schema in SQLite
        "database_backend": "sqlite",
        "vector_backend": "chroma",
        # Tests requiring module schema must use @pytest.mark.requires_postgres
    },
    "spark": {
        # Enterprise tier — air-gapped, Ollama for queries
        "deployment_tier": "enterprise",
        "claude_enrichment_enabled": True,
        "claude_query_enabled": False,
        "structured_query_enabled": True,
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
        "database_backend": "postgresql",
        "vector_backend": "pgvector",
        "claude_enrichment_model": "claude-sonnet-4-6",
        "claude_query_model_simple": "claude-haiku-4-5-20251001",
        "claude_query_model_complex": "claude-sonnet-4-6",
        "chat_model": "qwen3-rag",             # Ollama fallback
    },
}
```

---

## Codebase Transition

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
| React frontend (Vite + TypeScript) | Solid | Add module views, tenant config load |
| Evaluation framework (RAGAS) | Exists | Add module-specific gold sets |

### What Needs to Be Built (Core Platform)

| Component | Priority |
|---|---|
| `ClaudeEnrichmentService` | P0 |
| `QueryRouter` (rule engine + SQL execution) | P0 |
| `PgVectorService` (replaces Qdrant) | P0 |
| `ModuleRegistry` | P0 |
| Alembic migration system (core + module integration) | P0 |
| `TenantConfigResolver` + `PromptResolver` | P1 |
| `ClaudeQueryService` | P1 |
| `ClaudeModelRouter` | P1 |
| `CanonicalizedEntityMapper` (Contract enforcement) | P1 |
| Account matching service (3-tier algorithm) | P1 |
| `ProcessingService` — enrichment hook | P0 (modify) |
| `RAGService` — router + tier branching | P0 (modify) |
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
        entity_map = self.module_registry.get_entity_map()
        enrichment = await self.claude_enrichment_service.enrich_document(
            document_id=document_id, chunks=chunks,
            filename=document.original_filename,
            prompts=prompts, entity_map=entity_map,
        )
        # Canonicalization applied inside enrich_document
        # Entity-to-SQL dispatch uses entity_map to route to module tables
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

**Week 1: Database + Schema Foundation**
- Alembic migration system (core migrations + module migration integration)
- PostgreSQL + pgvector for `hosted` and `spark` profiles
- SQLite preserved for `laptop` (with `@pytest.mark.requires_postgres` test marker)
- `chunk_vectors` table with `vector(768)`, IVFFlat index
- `enrichment_synopses`, `enrichment_entities`, `review_items` core tables
- `ModuleRegistry` scaffold with all 4 (+ 1 optional) extension point methods
- Evaluation harness scaffold (gold set runner structure; gold set data provided by each vertical module spec)

**Week 2: Claude Enrichment**
- `ClaudeEnrichmentService` — two-call design, prompt caching
- `PromptResolver` — layered lookup (tenant → vertical → core)
- `CanonicalizedEntityMapper` — implements Canonicalization Contract
- Entity-to-SQL dispatcher using `ModuleRegistry.get_entity_map()` with idempotency
- Account matching service (3-tier algorithm; suffix list injected by module)
- Data lifecycle — soft-delete, versioning, cascade rules
- Ollama fallback when Claude API unavailable

**Week 3: Query Router + Standard Tier**
- Rule-based query router (Step 1–4 of Routing Specification)
- SQL template execution safety layer (uses `ModuleRegistry.get_sql_templates()`)
- `ClaudeQueryService` with pinned model IDs
- `TenantConfigResolver` — 3-tier merge
- Tenant config API endpoint (`GET /api/tenant/config`)
- React: tenant branding + feature flag loading at startup
- Query cost tracking and configurable caps
- Confidence score implementation (formula from this spec)

**Week 4: Integration + Deployment**
- Wire enrichment into `ProcessingService`
- Wire router + tier branching into `RAGService`
- Load first vertical module (`community_associations`) end-to-end
- End-to-end test with sample documents (both tiers)
- Docker Compose customer template
- API degradation behavior (Ollama fallback, cost-cap SQL-only mode)

### Phase 2 — Verticals + Frontend (Weeks 5-8)

**Weeks 5-6**: First vertical module fully operational; automation framework; frontend dashboard module view system.

**Weeks 7-8**: Second and third vertical modules; custom fields end-to-end; prompt override tested across verticals; provisioning automation.

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

Specific usage estimates (agents per customer, queries per day) are vertical-specific and belong in each vertical module's business case documentation.

---

## Risk Flags

### ENTITY_ACCURACY
- [ ] Confidence scoring per entity; low-confidence entities quarantined (not written to SQL)
- [ ] Canonicalization Contract enforced — validation failures flagged for review
- [ ] Source document citation for every SQL row in every module table

### API_DEPENDENCY
- [ ] Ollama fallback tested in both enrichment and query paths
- [ ] Cost tracking with configurable daily (enrichment) and monthly (query) caps
- [ ] Batch retry logic (3 retries with exponential backoff)
- [ ] Answer cache (24h TTL) for Standard tier degradation

### DATA_STALENESS
- [ ] Idempotency keys defined per module table; re-upload tested
- [ ] Document deleted/replaced → cascade soft-delete verified across all module tables
- [ ] Versioning (valid_from/valid_to) transitions tested with worked example

### ACCOUNT_MATCHING
- [ ] 3-tier matching tested against known edge cases (suffix variants, abbreviations)
- [ ] Tier 2 (pending merge) review queue surfaced in admin UI
- [ ] Audit log events for all three tiers

### SINGLE_TENANT_OPS
- [ ] Automated provisioning script (database, app, DNS, TLS, module selection)
- [ ] Rolling update strategy defined
- [ ] Per-customer backup and restore tested

### MODULE_ISOLATION
- [ ] Core never imports directly from module packages (only `ModuleRegistry` does)
- [ ] Module removal leaves core data intact
- [ ] Module migration failure does not block core startup or other modules

---

## Acceptance Criteria

### Phase 1 — Core Platform Infrastructure

- [ ] **pgvector storage**: After enrichment of any document, `SELECT COUNT(*) FROM chunk_vectors WHERE document_id = :id` returns > 0.
- [ ] **Enrichment pipeline completes**: Upload any document with an active module → `documents.enrichment_status = "completed"` and `enrichment_synopses` contains a row for that document.
- [ ] **Entity routing**: After enrichment, `enrichment_entities` contains rows with non-null `mapped_table` and `mapped_column` for entity types registered by the active module.
- [ ] **Router takes SQL-first path**: When SQL returns sufficient data, `response.meta.query_path == "sql"`. When SQL returns empty, `response.meta.query_path == "rag"`.
- [ ] **SQL safety**: Calling `QueryRouter` with a query that matches no registered template → response has `"mode": "rag_fallback"`, no SQL error raised.
- [ ] **Cost tracking**: After enriching 1 document, `GET /api/admin/costs` returns `enrichment_cost_usd > 0.0`.
- [ ] **Ollama fallback**: When `ANTHROPIC_API_KEY` is unset, document upload completes (Ollama path); no 500 errors. Query endpoints return 200 with `"source": "ollama"`.
- [ ] **Data lifecycle**: Upload same document twice → SQL rows in module tables have same count as after first upload (no duplicates).
- [ ] **Cascade delete**: Delete a document → all `enrichment_entities` and module table rows with `source_document_id = :id` have `is_deleted = TRUE`.
- [ ] **Provenance**: Every row in every module table has non-null `source_document_id`, `extraction_model`, `valid_from`.
- [ ] **Review queue routing**: Submit a query against an empty database → response has `confidence < 70` → `SELECT COUNT(*) FROM review_items WHERE review_status = 'pending'` returns ≥ 1.
- [ ] **Module registry**: Start application with `active_modules: ["core", "<any_vertical>"]` → no startup error; `GET /health` returns 200; `ModuleRegistry.active_modules` contains both entries.
- [ ] **Tenant branding**: `GET /api/tenant/config` returns `ui.brand.name` matching `tenant.json`; frontend `<title>` renders tenant name.
- [ ] **Feature flag**: Set any feature to `false` in `tenant.json`, restart → corresponding API endpoint returns HTTP 403.
- [ ] **Canonicalization**: Upload a document where Claude returns a monetary value with formatting → module table stores the value as REAL with no formatting characters.
- [ ] **Account matching Tier 1**: Upload document with high-similarity name (≥ 98%) against existing account → auto-linked (no review queue entry).
- [ ] **Account matching Tier 2**: Upload document with medium-similarity name (70–95%) against existing account → `review_items` entry created with `review_type = "account_match_pending"`.
- [ ] **Config reload**: Update `tenant.json`, restart → `GET /api/tenant/config` reflects updated values; `audit_log` contains `CONFIG_RELOAD` event.
- [ ] **Module migration isolation**: Apply a module migration, then remove the module from `active_modules` → core starts successfully; core tables unaffected.

### Phase 1 — Standard Tier

- [ ] **Model routing**: Structured intent query → `response.meta.model == "claude-haiku-4-5-20251001"`. Analytical intent query → `response.meta.model == "claude-sonnet-4-6"`.
- [ ] **Single-tenant isolation**: Two instances with different `tenant_id` values → data in instance A is unaffected by uploads to instance B.
- [ ] **Cost cap enforcement**: Set `monthly_query_cap_usd: 0.001`, submit 5 Sonnet queries → responses after cap reached return `"mode": "sql_only"`.
- [ ] **Degradation — Claude down**: Unset `ANTHROPIC_API_KEY`, submit query → response returns 200 with `"degraded": true` and Ollama-generated answer.

### Phase 1 — Enterprise Tier (Mode B)

- [ ] **Mode B ingestion**: On Spark with internet, upload document → Claude enrichment completes → `enrichment_status = "completed"` in documents table.
- [ ] **Mode B query**: After enrichment, disconnect internet → query returns answer from Ollama + enriched chunks with `confidence >= 70`.

---

## Operations (Stub)

The following components require operational runbooks. Runbooks are maintained as separate documents (`docs/runbooks/`) and referenced here by name.

| Component | Owner Role | Runbook |
|---|---|---|
| Customer provisioning (new instance) | Infrastructure lead | `docs/runbooks/provision-customer.md` |
| Customer backup and restore | Infrastructure lead | `docs/runbooks/backup-restore.md` |
| Rolling instance updates | Engineering lead | `docs/runbooks/rolling-update.md` |
| Incident response (API outage, data issue) | On-call engineer | `docs/runbooks/incident-response.md` |
| Cost cap alert response | On-call engineer | `docs/runbooks/cost-cap-alert.md` |
| Config reload procedure | Any engineer | Documented inline in Config Reload Policy section |

Runbooks are required before the first production customer deployment. Provisioning and incident response runbooks are required before the end of Phase 1.

---

## Design Decisions (Resolved)

1. **Claude as primary LLM for ingestion**: Both tiers use Claude API for document enrichment. Ollama-only enrichment produces inferior entity extraction. Ollama is fallback for enrichment (API unavailable) and primary for query-time responses in Enterprise tier.

2. **PostgreSQL mandatory for production**: All `hosted` and `spark` deployments use PostgreSQL + pgvector. SQLite is permitted for `laptop` dev profile only, with documented gaps. No Qdrant, no Chroma in v1.0 production paths.

3. **Deterministic routing without LLM**: The query router uses ordered pattern matching rules, not an LLM call, to classify intent. This is faster, cheaper, and reproducible.

4. **Confidence is a formula, not a feeling**: The composite confidence score is defined with specific inputs, weights, and thresholds. Acceptance criteria are tied to this formula.

5. **Only pinned model IDs**: Aliases and short names are forbidden in code. Model upgrades require a spec change and re-calibration.

6. **Mode A deferred to v1.1**: The air-gap transfer workflow adds significant engineering complexity. Mode B satisfies the Enterprise tier requirement for v1.0.

7. **3-tier account matching**: Single-threshold fuzzy match was rejected. Tier 1 auto-links high-confidence, Tier 2 flags ambiguous, Tier 3 auto-creates.

8. **Canonicalization Contract before SQL**: Raw Claude output is never written to SQL. All values pass through the Contract. Validation failures are quarantined to review queue.

9. **Config reload requires restart (v1.0)**: Hot-reload is deferred to v1.1.

10. **Single-tenant isolation**: Each customer gets a dedicated Docker Compose instance. More expensive to operate than multi-tenant but provides complete data isolation and per-customer customization.

11. **JSONB custom fields**: Tenant-specific metadata fields use JSONB extension columns, not per-customer schema migrations. Zero-code-deployment customization at the cost of losing strict column typing on custom fields.

12. **PII encryption key from env var (v1.0)**: KMS integration is deferred to v1.1. Sufficient for early-customer deployment; not sufficient for >5 customers.

13. **Module architecture over embedded vertical logic**: The ModuleRegistry pattern was chosen over embedded `if vertical == X` logic specifically because it allows the core platform to be independently testable, deployable, and maintainable. A team of 4 engineers can support 5+ verticals simultaneously because each vertical's complexity is fully contained in its module package.

---

*Spec: VaultIQ Core Platform v2.0*
*Supersedes: VAULTIQ_PLATFORM_v1.md, INSURANCE_AI_PLATFORM_v1.md, CLAUDE_ENRICHMENT_PIPELINE_v1.md*
*Status: FINAL - Ready for Implementation (Core Platform)*
*Scope: Core platform layer only. Vertical-specific content is defined in module specs (e.g., MODULE_COMMUNITY_ASSOCIATIONS_v1.md).*
