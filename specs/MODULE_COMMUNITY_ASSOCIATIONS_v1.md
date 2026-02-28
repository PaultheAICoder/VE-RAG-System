---
title: "VaultIQ Module — Community Associations Insurance"
status: FINAL - Ready for Implementation
version: v1.2
created: 2026-02-27
updated: 2026-02-27
author: —
type: Vertical Module Spec
parent_spec: specs/VAULTIQ_PLATFORM_v2.md (v2.0)
complexity: COMPLEX
stack: backend + frontend
changes: |
  v1.1 — Engineering review: plugin interface contract, Fannie Mae persistence/versioning,
         HO-6 batch safety, classifier ambiguity gate, idempotency enforcement, requirement
         precedence, template governance, account matching guards, schema constraints,
         renderer boundary, test fixtures, Fannie Mae governance, unit owner PII policy
  v1.2 — Absorbed from VAULTIQ_PLATFORM_v2.md: Insurance SQL schema (5 tables with full
         column contracts), complete document type taxonomy (CA + insurance types), CA-specific
         canonicalization contract, expanded SQL template catalog (8 templates), account matching
         suffix list registered via register_entity_map(), query router entity patterns, CA feature
         flags, full automation output contracts for all 4 services, CA gold set (16 questions)
---

# VaultIQ Module — Community Associations Insurance

## Summary

The Community Associations module is the **launch vertical** for VaultIQ. It extends the core platform with six unique document types, community-association-specific Claude extraction prompts, a CC&R and bylaw compliance engine, reserve study analysis, and automation features (unit owner letter generation, board presentation packages). All capabilities are implemented as a self-describing module package loaded at platform startup — extending core only through the four formally-defined plugin registration APIs; no ad-hoc core edits.

**Target customer**: Insurance agencies specializing in condominium associations, homeowner associations (HOAs), and community associations. Marshall Wells Lofts is the primary test case.

**Why this vertical first**: Community associations are document-intensive (CC&Rs, bylaws, reserve studies, board minutes, appraisals, HO-6 coordination), compliance-driven (Fannie Mae/FHA certification requirements, CC&R minimum coverage mandates), and renewal-predictable (annual programs). Automation ROI is highest in this segment.

---

## Scope

### In Scope — v1.0

- Module manifest, classifiers, and file structure
- Six unique CA document types plus insurance document types (full taxonomy below)
- Classifier ambiguity gate — routes mixed documents to review queue before extraction
- Five insurance SQL tables registered via Alembic migration (`insurance_accounts`, `insurance_policies`, `insurance_coverages`, `insurance_claims`, `insurance_certificates`) plus four CA-specific tables (`ca_reserve_studies`, `ca_unit_owners`, `ca_board_resolutions`, `ca_letter_batches`)
- Enforced idempotency: UNIQUE indexes + ON CONFLICT semantics on all CA tables
- CC&R/bylaw compliance engine with deterministic requirement-source precedence
- Fannie Mae/FHA requirement source — persisted in `insurance_requirements` with versioning
- Renewal prep automation — one-click coverage schedule for annual submissions
- Unit owner letter generation — failure-safe batch workflow with letter-level status tracking
- Board presentation package — coverage summary + compliance check + renewal recommendation
- Eight module-specific SQL query templates with parameter/output contracts
- CA-specific canonicalization rules applied during entity-to-SQL mapping
- CA-specific account name suffix list registered via `register_entity_map()`
- Query router entity patterns registered by this module
- CA feature flags in `tenant.json`
- Dashboard view configuration (multi-policy program view)
- Rendering boundary: all automation outputs rendered by core `DocumentRenderer` service
- Unit owner PII classification and retention policy
- Fannie Mae requirements governance (version, owner, update procedure)
- CA gold set: 16 Marshall Wells evaluation questions for quality validation
- 13 acceptance criteria mapped to test fixtures

### Out of Scope — v1.0

- Live Fannie Mae/FHA certification status lookup (future: carrier API integration)
- Unit owner certificate tracking (future: matches Construction subcontractor cert module)
- Reserve study financial modeling (extract data only; projections are out of scope)
- Email integration for unit owner letter delivery
- Board meeting scheduling or calendar features

---

## Core Platform Dependencies

This module extends the core platform through the four registered extension points only. It depends on the following core contracts and must not import core internals directly.

### Core APIs This Module Calls

| Core Interface | Where Used | Notes |
|---|---|---|
| `ModuleRegistry.register_document_classifiers(module_id, path)` | `module.register()` startup | Loads `classifiers.yaml` into the core document classifier |
| `ModuleRegistry.register_entity_map(module_id, map)` | `module.register()` startup | Merges `CA_ENTITY_TO_TABLE_MAP` into the core enrichment service; also registers CA suffix list for account matching |
| `ModuleRegistry.register_sql_templates(module_id, path)` | `module.register()` startup | Loads 8 templates from `sql_templates.yaml` into the core SQL template registry with `ca_` prefix |
| `ModuleRegistry.register_compliance_checker(module_id, cls)` | `module.register()` startup | Registers `CommunityAssociationsComplianceChecker` |
| `QueryRouter` | Core query pipeline | Executes the SQL templates this module registers; module does not call router directly |
| `ClaudeEnrichmentService` | Core ingestion pipeline | Calls the extraction prompts this module provides via `PromptResolver` |
| `PromptResolver` | Core ingestion pipeline | Loads prompts from `modules/community_associations/prompts/`; supports tenant override layer above module layer |
| `DocumentRenderer` | Automation output | Core renders PDF/PPTX from structured dict returned by automation services; module never calls renderer directly |

### Core Tables This Module Reads/Writes

| Table | Relationship |
|---|---|
| `documents` | FK target for `source_document_id` on all CA tables and insurance tables |
| `enrichment_synopses` | Written by core enrichment; read by compliance engine for document_type |
| `enrichment_entities` | Written by core enrichment; read by entity-to-SQL mapper |
| `chunk_vectors` | Written by core vector service after enrichment; read by RAG path |
| `review_items` | Written by ambiguity gate and canonicalization validation failure handler |

> **Module-owned table note**: `insurance_requirements` is **not** a core table. It is created and owned by this module via `modules/community_associations/migrations/001_insurance_tables.py`. It is listed separately here to document that the CA compliance engine both **creates** this table (via migration) **and** queries it at runtime (written by the compliance engine; read by the compliance checker; Fannie Mae/FHA rows injected at account creation).

### Module Registration Invariant

Once the four extension points are implemented in core (Phase 2 platform work), no subsequent vertical module requires any core file changes. Any feature request that cannot be satisfied through these four points requires a platform spec change, not an ad-hoc core edit.

---

## Module File Structure

```
ai_ready_rag/modules/community_associations/
├── manifest.json              ← Self-description + Fannie Mae governance metadata
├── module.py                  ← register() entry point
├── classifiers.yaml           ← Document type detection rules + ambiguity gate config
├── prompts/
│   ├── ccr_bylaws.txt         ← CC&R and bylaw extraction prompt
│   ├── reserve_study.txt      ← Reserve study extraction prompt
│   ├── board_minutes.txt      ← Board minutes extraction prompt
│   ├── appraisal.txt          ← Appraisal/valuation extraction prompt
│   ├── unit_owner_letter.txt  ← Unit owner correspondence extraction prompt
│   └── fannie_mae_reqs.txt    ← Fannie Mae/FHA requirement injection template
├── migrations/
│   ├── 001_insurance_tables.py   ← Alembic migration: 5 insurance tables (accounts, policies,
│   │                                coverages, claims, certificates) + insurance_requirements
│   └── 002_ca_tables.py          ← Alembic migration: 4 CA tables + all UNIQUE/CHECK constraints
├── compliance.py              ← ComplianceChecker implementation + precedence engine
├── automations/
│   ├── renewal_prep.py        ← Coverage schedule + submission data generator
│   ├── unit_owner_letter.py   ← HO-6 requirement letter batch generator (failure-safe)
│   └── board_presentation.py  ← Board package assembler
├── sql_templates.yaml         ← 8 module-specific SQL query templates with contracts
└── dashboard.json             ← Multi-policy program view configuration
```

---

## Plugin Interface Contract

### Overview

The module integrates with core via **four registered extension points** only. No direct imports of core internals. No core file edits per module added.

The module exposes a single `register(registry: ModuleRegistry)` entry point called at startup:

```python
# community_associations/module.py — required entry point
def register(registry: ModuleRegistry) -> None:
    """Called once at startup by ModuleRegistry.load_module()."""
    registry.register_document_classifiers("community_associations", "classifiers.yaml")
    registry.register_entity_map("community_associations", CA_ENTITY_TO_TABLE_MAP)
    registry.register_sql_templates("community_associations", "sql_templates.yaml")
    registry.register_compliance_checker("community_associations", CommunityAssociationsComplianceChecker)
    registry.register_api_router("community_associations", ca_router)
```

### The Four Extension Points

| Method | What It Does | Core Contract |
|---|---|---|
| `register_document_classifiers(module_id, path)` | Loads `classifiers.yaml` into the core document classifier; document types become available to enrichment pipeline | Core evaluates module classifiers after core classifiers; first match wins within module; ambiguity gate applied across all classifiers |
| `register_entity_map(module_id, map)` | Merges `CA_ENTITY_TO_TABLE_MAP` into the core entity routing map in the core enrichment service; also registers CA legal suffix list for account matching | Core raises a startup error if module adds a key already in the core map |
| `register_sql_templates(module_id, path)` | Loads templates from `sql_templates.yaml` into the core SQL template registry; template IDs prefixed with `ca_` to avoid collision | Core validates parameter schema on registration; rejects malformed templates at startup |
| `register_compliance_checker(module_id, cls)` | Registers `CommunityAssociationsComplianceChecker` with the core compliance dispatch layer; checker is invoked for any account with this module enabled | Core calls `checker.check(account_id, db)` and merges result into the global compliance output |

---

## Module Manifest

```json
{
  "module_id": "community_associations",
  "version": "1.0",
  "display_name": "Community Associations",
  "description": "CC&R compliance, reserve study analysis, unit owner automation for HOA/condo agencies.",
  "document_types": [
    "ccr", "bylaws", "reserve_study", "appraisal", "board_minutes", "unit_owner_letter",
    "policy", "certificate", "loss_run", "endorsement", "proposal", "submission",
    "bind_order", "correspondence", "unknown"
  ],
  "entity_types": [
    "unit_count", "reserve_fund_balance", "reserve_fund_percent_funded",
    "replacement_cost_new", "association_name", "management_company",
    "board_member", "fannie_mae_certification", "fha_certification",
    "ccr_requirement", "ho6_requirement"
  ],
  "compliance_rules": true,
  "schema_migrations": ["001_insurance_tables.py", "002_ca_tables.py"],
  "feature_flags": {
    "ca_compliance_engine": true,
    "ca_renewal_prep": true,
    "ca_unit_owner_letters": true,
    "ca_board_presentation": true,
    "ca_fannie_mae_tracking": true
  },
  "required_core_version": "1.0",
  "sql_templates": "sql_templates.yaml",
  "dashboard": "dashboard.json",
  "fannie_mae_governance": {
    "fannie_mae_reqs_version": "2026-Q1",
    "last_reviewed_at": "2026-02-27",
    "review_owner": "platform-team",
    "next_review_due": "2027-01-01"
  }
}
```

---

## Document Type Taxonomy

The CA module owns the following document type classifiers. Core CA governing document types are defined in `classifiers.yaml`. Insurance document types that appear in CA accounts are registered alongside them.

### Full CA Document Type Taxonomy

```python
# CA vertical owns all of these — registered by this module
CA_DOCUMENT_TYPES = {
    # Core CA governing documents
    "ccr": {
        "display_name": "CC&Rs",
        "filename_patterns": ["ccr", "cc&r", "covenants", "declaration", "restated declaration"],
        "content_keywords": [
            "covenants, conditions and restrictions", "declaration of covenants",
            "homeowners association", "condominium declaration", "association shall maintain"
        ],
        "confidence_threshold": 0.80,
        "extraction_prompt": "prompts/ccr_bylaws.txt",
    },
    "bylaws": {
        "display_name": "Bylaws",
        "filename_patterns": ["bylaws", "by-laws", "by laws"],
        "content_keywords": [
            "bylaws of the", "article i. name", "board of directors",
            "annual meeting", "officers of the association"
        ],
        "confidence_threshold": 0.80,
        "extraction_prompt": "prompts/ccr_bylaws.txt",
    },
    "reserve_study": {
        "display_name": "Reserve Study",
        "filename_patterns": ["reserve study", "reserve analysis", "reserve fund", "capital improvement"],
        "content_keywords": [
            "percent funded", "fully funded balance", "reserve fund balance",
            "replacement cost new", "remaining useful life",
            "annual reserve contribution", "30-year projection"
        ],
        "confidence_threshold": 0.85,
        "extraction_prompt": "prompts/reserve_study.txt",
    },
    "appraisal": {
        "display_name": "Property Appraisal",
        "filename_patterns": ["appraisal", "valuation", "replacement cost", "insured value"],
        "content_keywords": [
            "replacement cost new", "insured replacement value",
            "certified appraisal", "replacement cost appraisal",
            "marshall & swift", "e2value"
        ],
        "negative_keywords": ["loss run", "premium"],
        "confidence_threshold": 0.80,
        "extraction_prompt": "prompts/appraisal.txt",
    },
    "board_minutes": {
        "display_name": "Board Meeting Minutes",
        "filename_patterns": ["minutes", "board meeting", "board minutes", "meeting minutes"],
        "content_keywords": [
            "board of directors", "called to order", "quorum",
            "motion was made", "seconded by", "meeting adjourned"
        ],
        "confidence_threshold": 0.80,
        "extraction_prompt": "prompts/board_minutes.txt",
    },
    "unit_owner_letter": {
        "display_name": "Unit Owner Letter",
        "filename_patterns": ["ho6 letter", "unit owner", "ho-6 requirement", "insurance requirement letter"],
        "content_keywords": [
            "unit owners are required", "ho-6", "homeowners policy",
            "personal property coverage", "loss assessment coverage",
            "you are required to maintain"
        ],
        "confidence_threshold": 0.75,
        "extraction_prompt": "prompts/unit_owner_letter.txt",
    },
    # Insurance document types appearing in CA accounts
    "policy": {
        "display_name": "Insurance Policy",
        "filename_patterns": ["policy", "declarations", "dec page"],
        "content_keywords": ["declarations page", "policy number", "named insured", "policy period"],
        "confidence_threshold": 0.80,
        "extraction_prompt": "core/prompts/policy_extraction.txt",
    },
    "certificate": {
        "display_name": "Certificate of Insurance",
        "filename_patterns": ["certificate", "acord 25", "acord 24", "acord 27", "acord 28", "coi"],
        "content_keywords": ["certificate of liability insurance", "this is to certify", "acord"],
        "confidence_threshold": 0.85,
        "extraction_prompt": "core/prompts/certificate_extraction.txt",
    },
    "loss_run": {
        "display_name": "Loss Run",
        "filename_patterns": ["loss run", "claims history", "loss history"],
        "content_keywords": ["loss run", "claims experience", "date of loss", "incurred", "paid"],
        "confidence_threshold": 0.80,
        "extraction_prompt": "core/prompts/loss_run_extraction.txt",
    },
    "endorsement": {
        "display_name": "Policy Endorsement",
        "filename_patterns": ["endorsement", "form", "rider"],
        "content_keywords": ["it is agreed that", "in consideration of", "this endorsement modifies"],
        "confidence_threshold": 0.75,
        "extraction_prompt": "core/prompts/endorsement_extraction.txt",
    },
    "proposal": {
        "display_name": "Insurance Proposal",
        "filename_patterns": ["proposal", "quote", "quotation"],
        "content_keywords": ["proposed premium", "subject to underwriting", "this proposal"],
        "confidence_threshold": 0.75,
        "extraction_prompt": "core/prompts/policy_extraction.txt",
    },
    "submission": {
        "display_name": "Underwriting Submission",
        "filename_patterns": ["submission", "acord 80", "acord 126"],
        "content_keywords": ["applicant information", "requested limits", "underwriting information"],
        "confidence_threshold": 0.80,
        "extraction_prompt": "core/prompts/policy_extraction.txt",
    },
    "bind_order": {
        "display_name": "Bind Order",
        "filename_patterns": ["bind order", "binder", "confirmation of coverage"],
        "content_keywords": ["bind coverage", "coverage is bound", "effective immediately"],
        "confidence_threshold": 0.80,
        "extraction_prompt": "core/prompts/policy_extraction.txt",
    },
    "correspondence": {
        "display_name": "Correspondence",
        "filename_patterns": ["letter", "correspondence", "memo"],
        "content_keywords": [],
        "confidence_threshold": 0.60,
        "extraction_prompt": "core/prompts/synopsis_only.txt",
    },
    "unknown": {
        "display_name": "Unknown",
        "description": "Unclassified document — flagged for admin review",
        "confidence_threshold": 0.0,  # Always fallback
        "extraction_prompt": "core/prompts/synopsis_only.txt",
    },
}
```

### `classifiers.yaml`

```yaml
# Community Associations — Document Type Classifiers
# Applied by the core document classifier after core classification passes.
# Ambiguity gate: if top-two candidate scores are within 0.10 of each other,
# route to review queue with review_reason='ambiguous_classification' instead of
# proceeding to extraction. The agent resolves classification before extraction runs.

module: community_associations
ambiguity_threshold: 0.10   # ← Gap required between top-two candidates to auto-proceed

classifiers:

  - document_type: ccr
    display_name: "CC&Rs"
    description: "Covenants, Conditions & Restrictions governing document"
    patterns:
      filename:
        - "ccr"
        - "cc&r"
        - "covenants"
        - "declaration"
        - "restated declaration"
      content_keywords:
        - "covenants, conditions and restrictions"
        - "declaration of covenants"
        - "homeowners association"
        - "condominium declaration"
        - "association shall maintain"
      negative_keywords: []
    confidence_threshold: 0.80

  - document_type: bylaws
    display_name: "Bylaws"
    description: "Association bylaws governing operations and insurance requirements"
    patterns:
      filename:
        - "bylaws"
        - "by-laws"
        - "by laws"
      content_keywords:
        - "bylaws of the"
        - "article i. name"
        - "board of directors"
        - "annual meeting"
        - "officers of the association"
      negative_keywords: []
    confidence_threshold: 0.80

  - document_type: reserve_study
    display_name: "Reserve Study"
    description: "Reserve fund analysis and capital replacement schedule"
    patterns:
      filename:
        - "reserve study"
        - "reserve analysis"
        - "reserve fund"
        - "capital improvement"
      content_keywords:
        - "percent funded"
        - "fully funded balance"
        - "reserve fund balance"
        - "replacement cost new"
        - "remaining useful life"
        - "annual reserve contribution"
        - "30-year projection"
      negative_keywords: []
    confidence_threshold: 0.85

  - document_type: appraisal
    display_name: "Property Appraisal"
    description: "Real property appraisal establishing replacement cost or market value"
    patterns:
      filename:
        - "appraisal"
        - "valuation"
        - "replacement cost"
        - "insured value"
      content_keywords:
        - "replacement cost new"
        - "insured replacement value"
        - "certified appraisal"
        - "replacement cost appraisal"
        - "marshall & swift"
        - "e2value"
      negative_keywords:
        - "loss run"
        - "premium"
    confidence_threshold: 0.80

  - document_type: board_minutes
    display_name: "Board Meeting Minutes"
    description: "Board of directors meeting minutes"
    patterns:
      filename:
        - "minutes"
        - "board meeting"
        - "board minutes"
        - "meeting minutes"
      content_keywords:
        - "board of directors"
        - "called to order"
        - "quorum"
        - "motion was made"
        - "seconded by"
        - "meeting adjourned"
      negative_keywords: []
    confidence_threshold: 0.80

  - document_type: unit_owner_letter
    display_name: "Unit Owner Letter"
    description: "Correspondence to unit owners regarding insurance requirements (HO-6)"
    patterns:
      filename:
        - "ho6 letter"
        - "unit owner"
        - "ho-6 requirement"
        - "insurance requirement letter"
      content_keywords:
        - "unit owners are required"
        - "ho-6"
        - "homeowners policy"
        - "personal property coverage"
        - "loss assessment coverage"
        - "you are required to maintain"
      negative_keywords: []
    confidence_threshold: 0.75
```

**Ambiguity gate behavior** (board packet example):
A board meeting packet may score `board_minutes: 0.78` and `reserve_study: 0.73` — gap of 0.05, below the 0.10 threshold. Instead of proceeding, the system creates a `review_items` row with `review_reason = 'ambiguous_classification'`, `candidate_types = ['board_minutes', 'reserve_study']`, and `candidate_scores = [0.78, 0.73]`. The agent selects the correct type; extraction runs only after review resolves.

---

## Insurance SQL Schema

All five insurance tables are registered by this module via Alembic migration at `modules/community_associations/migrations/001_insurance_tables.py`. These tables are additive — no changes to any existing core tables.

All tables share these common columns unless otherwise noted:
- `tenant_id TEXT` — instance identifier
- `is_deleted BOOLEAN DEFAULT FALSE` — soft-delete flag
- `deleted_at DATETIME` — soft-delete timestamp
- `valid_from DATETIME NOT NULL DEFAULT NOW()` — versioning start
- `valid_to DATETIME` — versioning end (NULL = current record)
- `created_at DATETIME NOT NULL DEFAULT NOW()`
- `updated_at DATETIME NOT NULL DEFAULT NOW()`

**Current record filter**: Always `WHERE valid_to IS NULL AND is_deleted = FALSE`.

### `insurance_accounts`

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| account_name | TEXT | No | Named insured (display name) |
| account_type | TEXT | No | Enum: `"condo_association"`, `"hoa"`, `"planned_community"` |
| named_insured | TEXT | Yes | Legal name as appears on policy |
| primary_address | TEXT | Yes | Street address |
| city | TEXT | Yes | |
| state | TEXT | Yes | Two-letter state code |
| zip | TEXT | Yes | |
| units_residential | INTEGER | Yes | Total residential units in the association |
| units_commercial | INTEGER | Yes | Total commercial units (if mixed-use) |
| account_manager | TEXT | Yes | Servicing agent name |
| custom_fields | JSONB | Yes | Tenant-defined extension fields |
| source_document_id | TEXT FK | Yes | Document that produced this record → documents.id |
| extraction_confidence | REAL | Yes | 0.0–1.0 |

**CHECK constraints**:
```sql
CHECK (account_type IN ('condo_association', 'hoa', 'planned_community'))
CHECK (extraction_confidence IS NULL OR (extraction_confidence >= 0 AND extraction_confidence <= 1))
```

**Idempotency key**: `(account_name, tenant_id)` — same named insured in same tenant updates in place.

### `insurance_policies`

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| account_id | TEXT FK | No | → insurance_accounts.id |
| policy_number | TEXT | No | Normalized policy number (trimmed) |
| carrier_name | TEXT | Yes | Canonical carrier name (via `ca_carrier_aliases.csv`) |
| carrier_naic | TEXT | Yes | NAIC company code |
| line_of_business | TEXT | No | Enum: see LOB enum below |
| policy_type | TEXT | Yes | Enum: `"occurrence"`, `"claims_made"` |
| inception_date | DATE | Yes | ISO 8601 |
| expiration_date | DATE | Yes | ISO 8601 |
| premium | REAL | Yes | Annual premium in dollars (no formatting) |
| billing_type | TEXT | Yes | Enum: `"annual"`, `"installment"` |
| policy_status | TEXT | No | Enum: `"active"`, `"expired"`, `"cancelled"`, `"pending"` |
| source_document_id | TEXT FK | Yes | → documents.id |
| extraction_model | TEXT | Yes | Pinned model ID used (e.g., `"claude-sonnet-4-6"`) |
| extraction_confidence | REAL | Yes | 0.0–1.0 |
| idempotency_key | TEXT | Yes | `SHA256(account_id + policy_number + inception_date)` |

**Line of business enum values** (stored as strings, not Python names):
`"commercial_property"`, `"gl"`, `"do"`, `"crime"`, `"umbrella"`, `"fidelity"`, `"residential"`, `"wc"`, `"epli"`, `"cyber"`, `"auto"`, `"equipment_breakdown"`

**CHECK constraints**:
```sql
CHECK (policy_status IN ('active', 'expired', 'cancelled', 'pending'))
CHECK (policy_type IS NULL OR policy_type IN ('occurrence', 'claims_made'))
CHECK (billing_type IS NULL OR billing_type IN ('annual', 'installment'))
```

**Idempotency key / UNIQUE index**: `UNIQUE (account_id, policy_number, inception_date)` — same policy uploaded twice updates in place.

**Index**: `(account_id, line_of_business, inception_date)`

### `insurance_coverages`

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| policy_id | TEXT FK | No | → insurance_policies.id |
| coverage_type | TEXT | No | Enum: see coverage type enum below |
| limit_per_occurrence | REAL | Yes | Dollars |
| limit_aggregate | REAL | Yes | Dollars |
| deductible | REAL | Yes | Dollars |
| sublimit | REAL | Yes | Dollars |
| exclusions | JSONB | Yes | Array of exclusion strings |
| endorsements | JSONB | Yes | Array of endorsement form numbers/descriptions |
| source_document_id | TEXT FK | Yes | → documents.id |
| extraction_confidence | REAL | Yes | 0.0–1.0 |

**Coverage type enum values** (stored as strings):
`"property"`, `"general_liability"`, `"directors_and_officers"`, `"crime"`, `"umbrella"`, `"fidelity"`, `"ho6_unit_owner"`, `"workers_comp"`, `"epli"`, `"cyber"`, `"auto_liability"`, `"equipment_breakdown"`

**Idempotency key / UNIQUE index**: `UNIQUE (policy_id, coverage_type)` — exact match = update in place.

**Index**: `(policy_id, coverage_type)`

### `insurance_claims`

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| account_id | TEXT FK | No | → insurance_accounts.id |
| policy_id | TEXT FK | Yes | → insurance_policies.id (optional — claim may precede policy link) |
| claimant | TEXT | Yes | Claimant name — PII (Fernet-encrypted) |
| date_of_loss | DATE | Yes | ISO 8601 |
| date_reported | DATE | Yes | ISO 8601 |
| claim_status | TEXT | Yes | Enum: `"open"`, `"closed"`, `"reopened"` |
| reserve_amount | REAL | Yes | Dollars |
| paid_amount | REAL | Yes | Dollars |
| closed_amount | REAL | Yes | Total incurred at close |
| line_of_business | TEXT | Yes | LOB enum value |
| description | TEXT | Yes | Claim description |
| source_document_id | TEXT FK | Yes | Loss run document → documents.id |

**Idempotency key**: When claim_number is present: `UNIQUE (account_id, claim_number)`. When absent: `(account_id, date_of_loss, line_of_business, closed_amount)`.

**Index**: `(account_id, date_of_loss)`

### `insurance_certificates`

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| account_id | TEXT FK | No | → insurance_accounts.id |
| policy_id | TEXT FK | Yes | → insurance_policies.id |
| cert_holder_name | TEXT | Yes | Certificate holder name |
| cert_holder_address | TEXT | Yes | Certificate holder address |
| additional_insured | BOOLEAN | Yes | Whether additional insured is indicated |
| waiver_of_subrogation | BOOLEAN | Yes | Whether WOS is endorsed |
| effective_date | DATE | Yes | ISO 8601 |
| expiration_date | DATE | Yes | ISO 8601 |
| acord_form_type | TEXT | Yes | Enum: `"acord_24"`, `"acord_25"`, `"acord_27"`, `"acord_28"` |
| source_document_id | TEXT FK | Yes | → documents.id |
| extraction_confidence | REAL | Yes | 0.0–1.0 |

**Idempotency key / UNIQUE index**: `UNIQUE (account_id, cert_holder_name, effective_date, acord_form_type)`.

---

## CA-Specific Schema Extensions

Four tables are added by the CA module's second migration (`002_ca_tables.py`). They are **additive only** — no changes to the insurance tables above.

### `ca_reserve_studies`

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| account_id | TEXT FK | No | → insurance_accounts.id |
| document_id | TEXT FK | No | → documents.id |
| study_date | DATE | Yes | Date of reserve study |
| study_firm | TEXT | Yes | Firm that prepared the study |
| percent_funded | REAL | Yes | Reserve fund percent funded (0-100) |
| fully_funded_balance | REAL | Yes | Target fully-funded balance in dollars |
| actual_reserve_balance | REAL | Yes | Current reserve fund balance |
| annual_contribution | REAL | Yes | Recommended annual reserve contribution |
| replacement_cost_new | REAL | Yes | Total replacement cost new from study |
| component_count | INTEGER | Yes | Number of components in study |
| study_type | TEXT | Yes | Enum: `"full"`, `"update"` |
| next_study_date | DATE | Yes | Recommended next study date |
| funding_plan | TEXT | Yes | Enum: `"baseline"`, `"threshold"`, `"full_funding"`, `"reserve_specialist"` |
| notes | TEXT | Yes | Analyst notes or caveats |
| created_at | DATETIME | No | Timestamp |

**Idempotency key / UNIQUE index**: `UNIQUE (account_id, study_date, study_firm)`

**ON CONFLICT**: `DO UPDATE SET` all mutable columns + `updated_at = NOW()` — same study from same firm on same date updates in place.

**CHECK constraints**:
```sql
CHECK (study_type IN ('full', 'update'))
CHECK (funding_plan IN ('baseline', 'threshold', 'full_funding', 'reserve_specialist'))
CHECK (percent_funded IS NULL OR (percent_funded >= 0 AND percent_funded <= 100))
```

---

### `ca_unit_owners`

PII columns (`owner_name`, `owner_email`, `mailing_address`) are encrypted at rest via the platform Fernet key. See [Unit Owner Data Privacy](#unit-owner-data-privacy) section.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| account_id | TEXT FK | No | → insurance_accounts.id |
| unit_number | TEXT | No | Unit identifier (e.g., "101", "B-4") |
| owner_name | TEXT (encrypted) | Yes | Unit owner name — PII |
| owner_email | TEXT (encrypted) | Yes | For letter delivery (future) — PII |
| mailing_address | TEXT (encrypted) | Yes | Owner mailing address if different from unit — PII |
| ho6_required | BOOLEAN | No | Whether HO-6 is required for this unit. Default: `false` |
| ho6_minimum_amount | REAL | Yes | Minimum required HO-6 coverage amount |
| source_document_id | TEXT FK | Yes | Document this was extracted from |
| created_at | DATETIME | No | Timestamp |
| updated_at | DATETIME | No | Last update timestamp |

**Idempotency key / UNIQUE index**: `UNIQUE (account_id, unit_number)`

**ON CONFLICT**: `DO UPDATE SET owner_name, owner_email, mailing_address, ho6_required, ho6_minimum_amount, updated_at = NOW()` — unit owner details may change between uploads.

**CHECK constraints**:
```sql
CHECK (ho6_required IN (true, false))
```

---

### `ca_board_resolutions`

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| account_id | TEXT FK | No | → insurance_accounts.id |
| document_id | TEXT FK | No | → documents.id (board minutes) |
| resolution_date | DATE | Yes | Date of meeting where resolution passed |
| resolution_type | TEXT | Yes | Enum: `"coverage_approval"`, `"carrier_change"`, `"deductible_change"`, `"special_assessment"`, `"coverage_waiver"`, `"other"` |
| description | TEXT | No | Full text of resolution |
| motion_by | TEXT | Yes | Director who made the motion |
| vote_result | TEXT | Yes | Enum: `"approved"`, `"denied"`, `"tabled"` |
| effective_date | DATE | Yes | When resolution takes effect |
| created_at | DATETIME | No | Timestamp |

**Idempotency key / UNIQUE index**: `UNIQUE (account_id, resolution_date, resolution_type, vote_result)`

**ON CONFLICT**: `DO NOTHING` — a passed resolution does not change retroactively; duplicate extractions are silently ignored.

**CHECK constraints**:
```sql
CHECK (resolution_type IN ('coverage_approval','carrier_change','deductible_change',
                           'special_assessment','coverage_waiver','other'))
CHECK (vote_result IN ('approved', 'denied', 'tabled'))
```

---

### `ca_letter_batches`

Failure-safe batch tracking for HO-6 letter generation.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID — the `letter_batch_id` |
| account_id | TEXT FK | No | → insurance_accounts.id |
| initiated_by | TEXT FK | No | → users.id |
| initiated_at | DATETIME | No | When batch generation was requested |
| status | TEXT | No | Enum: `"pending"`, `"generating"`, `"generated"`, `"failed"` |
| total_units | INTEGER | No | Total letters in this batch |
| generated_count | INTEGER | No | Letters successfully generated so far |
| failed_count | INTEGER | No | Letters that failed generation |
| completed_at | DATETIME | Yes | When status reached `"generated"` or `"failed"` |
| failure_reason | TEXT | Yes | Set if status = `"failed"` |

**Letter-level rows** are stored in `ca_letter_batch_items` (child table):

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| batch_id | TEXT FK | No | → ca_letter_batches.id |
| unit_number | TEXT | No | Unit identifier |
| letter_status | TEXT | No | Enum: `"pending"`, `"generated"`, `"failed"` |
| letter_text | TEXT | Yes | Generated letter content (set when `generated`) |
| generated_at | DATETIME | Yes | When this letter was generated |
| failure_reason | TEXT | Yes | Set if `letter_status = 'failed'` |

**Composite PK** on `ca_letter_batch_items`: `(batch_id, unit_number)`.

**Generation workflow**:
1. Create `ca_letter_batches` row with `status = 'pending'`, `total_units = N`
2. For each unit, generate letter text; insert `ca_letter_batch_items` with `letter_status = 'generated'` or `'failed'`
3. On successful completion of all items: update batch `status = 'generated'`, `completed_at = NOW()`
4. On partial failure: batch `status = 'failed'`; re-run only retries `letter_status = 'pending'` or `'failed'` items within the existing batch
5. A new batch is created only when the caller explicitly requests a new generation run — retries reuse the existing batch row

**CHECK constraints**:
```sql
CHECK (status IN ('pending', 'generating', 'generated', 'failed'))
CHECK (letter_status IN ('pending', 'generated', 'failed'))  -- on ca_letter_batch_items
```

---

## Canonicalization Contract (CA-Specific)

All entity values extracted by Claude **must** be canonicalized before writing to insurance tables. Raw Claude output is never written directly to SQL. The CA module applies the following rules during entity-to-SQL mapping. Validation failures are quarantined — see failure behavior below.

### Rules by Data Type

| Data Type | Canonical Form | CA Examples |
|---|---|---|
| **Dates** | ISO 8601 `YYYY-MM-DD` | `"1/1/2026"` → `"2026-01-01"`; `"Jan 1, 2026"` → `"2026-01-01"` |
| **Monetary values / limits** | REAL in dollars, no formatting | `"$1M"` → `1000000.0`; `"$1,000,000.00"` → `1000000.0`; `"1.5M"` → `1500000.0`; strip `$` and commas before parsing |
| **Percentages** | REAL as decimal | `"85%"` → `0.85`; `"100"` → `1.00` (when context is a percentage field) |
| **Policy numbers** | Strip leading/trailing whitespace; collapse internal runs to single space | `"  CPP-029 4618 "` → `"CPP-029 4618"` |
| **Carrier names** | Lookup in `modules/community_associations/data/ca_carrier_aliases.csv`; if no match, store as-is | `"Travelers Casualty and Surety"` → `"Travelers"`; `"Zurich North America"` → `"Zurich"` |
| **Coverage types** | Map to enum string values | `"each occurrence"` → `"per_occurrence"`; `"general aggregate"` → `"aggregate"` |
| **Line of business** | Map to enum string values | `"General Liability"` → `"gl"`; `"Commercial Property"` → `"commercial_property"`; `"D&O"` → `"do"` |

### CA Carrier Alias File

`modules/community_associations/data/ca_carrier_aliases.csv` — two-column CSV: `raw_name, canonical_name`.

Common CA carrier aliases seeded at launch:

| Raw Name | Canonical Name |
|---|---|
| Travelers Casualty and Surety Company | Travelers |
| Travelers Casualty and Surety | Travelers |
| The Travelers Indemnity Company | Travelers |
| Chubb National Insurance | Chubb |
| ACE American Insurance | Chubb |
| Federal Insurance Company | Chubb |
| Liberty Mutual Fire Insurance | Liberty Mutual |
| Zurich American Insurance | Zurich |
| Zurich North America | Zurich |
| Hartford Fire Insurance | Hartford |
| The Hartford | Hartford |
| AIG Specialty Insurance | AIG |
| Markel Insurance Company | Markel |
| Philadelphia Indemnity Insurance | Philadelphia |
| State Auto Insurance | State Auto |

Unmatched carrier names are stored as-is and flagged with `extraction_confidence < 0.7` for admin review.

### Validation Failure Behavior

When an entity value cannot be canonicalized (unparseable date, non-numeric monetary value, unrecognized enum, etc.):

1. Entity is stored in `enrichment_entities` with `confidence = 0.0`
2. Entity is **not** written to insurance_* tables or CA tables
3. A `review_items` entry is created with `review_type = "canonicalization_failure"`
4. Processing continues — one bad entity does not fail the entire document

---

## Entity-to-SQL Mapping Extensions

The CA module registers these additional entity type mappings via `register_entity_map()`. No collisions with core entity keys are permitted — core raises a startup error if any key is duplicated.

```python
CA_ENTITY_TO_TABLE_MAP = {
    "unit_count":                  ("insurance_accounts",   "units_residential"),
    "reserve_fund_balance":        ("ca_reserve_studies",   "actual_reserve_balance"),
    "reserve_fund_percent_funded": ("ca_reserve_studies",   "percent_funded"),
    "replacement_cost_new":        ("ca_reserve_studies",   "replacement_cost_new"),
    "association_name":            ("insurance_accounts",   "account_name"),
    "management_company":          ("insurance_accounts",   "custom_fields"),  # JSONB
    "ccr_requirement":             ("insurance_requirements", "requirement_text"),
    "ho6_requirement":             ("insurance_requirements", "requirement_text"),
    "fannie_mae_certification":    ("insurance_accounts",   "custom_fields"),  # JSONB
    "fha_certification":           ("insurance_accounts",   "custom_fields"),  # JSONB
}
```

---

## Account Matching — CA Configuration

The core 3-tier matching algorithm handles all document types. The CA module registers a CA-specific suffix list and corroboration rules via `register_entity_map()`.

### Legal Suffixes Registered by This Module

These suffixes are stripped from both the document's extracted `named_insured` and the candidate account name before difflib similarity scoring:

```python
CA_LEGAL_SUFFIXES_TO_STRIP = [
    "LLC", "Inc", "Inc.", "Corporation", "Corp", "Corp.",
    "Association", "Condominium Association", "Condo Association",
    "HOA", "Community Association", "Homeowners Association",
    "Homeowner's Association", "Owners Association", "Owner's Association",
    "Planned Community", "Ltd", "Ltd.",
]

CA_NORMALIZATIONS = {
    "Homeowners": "HOA",
    "Homeowner's": "HOA",
    "Condominium": "Condo",
}
```

**Protected-qualifier exception**: If the canonical name in the system contains a geographic or distinguishing qualifier (e.g., "River Oaks HOA Portland"), **do not auto-link** a document whose post-normalization name lacks that qualifier. Route to review (Tier 2) instead. This prevents false merges between same-named associations in different cities.

### Minimum Corroboration for Auto-Link (95-99% Band)

When normalized name similarity is between 95-99%, **at least 2 corroborating signals** are required before auto-link:

| Signal | Match Criterion |
|---|---|
| Property address | Street number + street name match (city/state optional) |
| Unit count | Within ±10% of count in system |
| Management company | Exact string match in JSONB `custom_fields` |
| CA-type document from same property | Another document already linked to this account |

If fewer than 2 corroborating signals are available: route to review (Tier 2) regardless of name similarity score.

**Sample name variations handled**:

| Document Name | Post-normalization | System Name | Result |
|---|---|---|---|
| "River Oaks Homeowners Association, Inc." | "River Oaks" | "River Oaks HOA" | Matches |
| "Marshall Wells Lofts Owners Association" | "Marshall Wells Lofts" | "Marshall Wells Lofts" | Exact match |
| "Marshall Wells Lofts Condominium" | "Marshall Wells Lofts" | "Marshall Wells Lofts" | Matches |

---

## Entity Patterns and Query Router Integration

The CA module registers query entity patterns and SQL template mappings via `register_sql_templates()`. The core `QueryRouter` evaluates these patterns using the same rule-based intent classifier described in `VAULTIQ_PLATFORM_v2.md`.

### CA-Specific Structured Query Triggers

The following entity patterns cause the router to route to SQL-first path:

| Entity Pattern | Query Keywords | Maps to Template |
|---|---|---|
| Coverage limits | `limit`, `deductible`, `coverage` | `ca_coverage_by_account_line` |
| Premium | `premium`, `cost`, `rate` | `ca_premium_query` |
| Carrier | `carrier`, `insurer`, `who insures` | `ca_carrier_lookup` |
| Policy dates | `expiration`, `expires`, `renewal`, `effective` | `ca_policy_dates` |
| Policy number | `policy number`, `policy #` | `ca_carrier_lookup` |
| Claims history | `claims`, `losses`, `loss history`, `incidents` | `ca_claims_history` |
| Compliance | `compliance`, `cc&r`, `requirement`, `missing coverage`, `gap` | `ca_compliance_gap` |
| Coverage schedule | `coverage schedule`, `all lines`, `program summary`, `what coverage` | `ca_coverage_schedule` |
| Comparison | `compare`, `vs`, `versus`, `difference between`, `how do .+ compare` | `ca_comparison_by_line` |

### Account Resolution for CA Queries

Step 1 of routing resolves account context from:
- Explicit mention in query text: "for Marshall Wells", "Marshall Wells Lofts"
- Active session context (current account in UI session)
- If neither resolves: `CONVERSATIONAL` intent — no SQL attempted

---

## SQL Template Catalog (CA Module)

All 8 templates are registered by this module via `register_sql_templates()`. Template IDs are prefixed with `ca_` to avoid collision with core templates. All templates are allowlisted, parameterized, read-only. Expected latency < 500ms. Row cap: 1,000.

### Template 1: `ca_coverage_by_account_line`

| | |
|---|---|
| **Parameters** | `:account_id TEXT`, `:line_of_business TEXT` |
| **Output columns** | `coverage_type, limit_per_occurrence, limit_aggregate, deductible, sublimit, carrier_name, policy_number, inception_date, expiration_date` |
| **Triggers on** | "limit", "deductible", "coverage", "what are the GL limits", "what is the property deductible" |

```sql
SELECT c.coverage_type, c.limit_per_occurrence, c.limit_aggregate,
       c.deductible, c.sublimit,
       p.carrier_name, p.policy_number, p.inception_date, p.expiration_date
FROM insurance_coverages c
JOIN insurance_policies p ON c.policy_id = p.id
WHERE p.account_id = :account_id
  AND p.line_of_business = :line_of_business
  AND p.policy_status = 'active'
  AND p.valid_to IS NULL AND p.is_deleted = FALSE
ORDER BY c.coverage_type
LIMIT :row_cap
```

### Template 2: `ca_carrier_lookup`

| | |
|---|---|
| **Parameters** | `:account_id TEXT` |
| **Output columns** | `line_of_business, carrier_name, carrier_naic, policy_number, inception_date, expiration_date, policy_status` |
| **Triggers on** | "carrier", "insurer", "who insures", "policy number" |

```sql
SELECT p.line_of_business, p.carrier_name, p.carrier_naic,
       p.policy_number, p.inception_date, p.expiration_date, p.policy_status
FROM insurance_policies p
WHERE p.account_id = :account_id
  AND p.valid_to IS NULL AND p.is_deleted = FALSE
ORDER BY p.line_of_business, p.inception_date DESC
LIMIT :row_cap
```

### Template 3: `ca_premium_query`

| | |
|---|---|
| **Parameters** | `:account_id TEXT` |
| **Output columns** | `line_of_business, carrier_name, policy_number, premium, inception_date, expiration_date` |
| **Triggers on** | "premium", "cost", "rate", "how much does", "total premium" |

```sql
SELECT p.line_of_business, p.carrier_name, p.policy_number,
       p.premium, p.inception_date, p.expiration_date
FROM insurance_policies p
WHERE p.account_id = :account_id
  AND p.policy_status = 'active'
  AND p.valid_to IS NULL AND p.is_deleted = FALSE
ORDER BY p.line_of_business
LIMIT :row_cap
```

### Template 4: `ca_policy_dates`

| | |
|---|---|
| **Parameters** | `:account_id TEXT` |
| **Output columns** | `line_of_business, carrier_name, policy_number, inception_date, expiration_date, days_until_expiration` |
| **Triggers on** | "expiration", "expires", "renewal", "effective", "when does" |

```sql
SELECT p.line_of_business, p.carrier_name, p.policy_number,
       p.inception_date, p.expiration_date,
       (p.expiration_date - CURRENT_DATE) AS days_until_expiration
FROM insurance_policies p
WHERE p.account_id = :account_id
  AND p.valid_to IS NULL AND p.is_deleted = FALSE
ORDER BY p.expiration_date ASC
LIMIT :row_cap
```

### Template 5: `ca_claims_history`

| | |
|---|---|
| **Parameters** | `:account_id TEXT`, `:cutoff_date DATE` (default: 5 years ago) |
| **Output columns** | `line_of_business, date_of_loss, date_reported, claim_status, reserve_amount, paid_amount, closed_amount, description` |
| **Triggers on** | "claims", "losses", "loss history", "incidents", "what claims" |

```sql
SELECT c.line_of_business, c.date_of_loss, c.date_reported,
       c.claim_status, c.reserve_amount, c.paid_amount, c.closed_amount,
       c.description
FROM insurance_claims c
WHERE c.account_id = :account_id
  AND c.date_of_loss >= :cutoff_date
  AND c.is_deleted = FALSE
ORDER BY c.date_of_loss DESC
LIMIT :row_cap
```

### Template 6: `ca_compliance_gap`

| | |
|---|---|
| **Parameters** | `:account_id TEXT` |
| **Output columns** | `coverage_line, requirement_source, requirement_text, min_limit, is_met, current_limit, gap_amount` |
| **Triggers on** | "compliance", "cc&r", "requirement", "missing coverage", "gap", "does current coverage meet" |

```sql
SELECT r.coverage_line, r.requirement_source, r.requirement_text,
       r.min_limit, r.is_met, r.current_limit, r.gap_amount
FROM insurance_requirements r
WHERE r.account_id = :account_id
  AND r.is_met = false
  AND r.superseded_at IS NULL
ORDER BY r.coverage_line
LIMIT :row_cap
```

### Template 7: `ca_coverage_schedule`

| | |
|---|---|
| **Parameters** | `:account_id TEXT` |
| **Output columns** | `line_of_business, carrier_name, policy_number, inception_date, expiration_date, premium, coverage_type, limit_per_occurrence, limit_aggregate, deductible, valuation` |
| **Triggers on** | "coverage schedule", "all lines", "program summary", "what coverage do we have", "list all coverages" |

```sql
SELECT p.line_of_business, p.carrier_name, p.policy_number,
       p.inception_date, p.expiration_date, p.premium,
       c.coverage_type, c.limit_per_occurrence, c.limit_aggregate,
       c.deductible
FROM insurance_policies p
JOIN insurance_coverages c ON c.policy_id = p.id
WHERE p.account_id = :account_id
  AND p.policy_status = 'active'
  AND p.valid_to IS NULL AND p.is_deleted = FALSE
ORDER BY p.line_of_business, c.coverage_type
LIMIT :row_cap
```

### Template 8: `ca_comparison_by_line`

| | |
|---|---|
| **Parameters** | `:account_id TEXT` |
| **Output columns** | `policy_number, inception_date, expiration_date, line_of_business, carrier_name, premium, coverage_type, limit_per_occurrence, limit_aggregate, deductible` |
| **Triggers on** | "compare", "vs", "versus", "difference between", "how do .+ compare", "side by side" |

```sql
SELECT p.policy_number, p.inception_date, p.expiration_date,
       p.line_of_business, p.carrier_name, p.premium,
       c.coverage_type, c.limit_per_occurrence, c.limit_aggregate, c.deductible
FROM insurance_policies p
JOIN insurance_coverages c ON c.policy_id = p.id
WHERE p.account_id = :account_id
  AND p.valid_to IS NULL AND p.is_deleted = FALSE
ORDER BY p.line_of_business, p.inception_date DESC
LIMIT :row_cap
```

**SQL Execution Safety** (enforced by core `QueryRouter` for all registered templates):
- Allowlisted templates only — no arbitrary SQL generation
- Parameterized bindings — no string interpolation
- 5-second query timeout — fallback to RAG on timeout
- 1,000-row cap enforced via `LIMIT :row_cap` in every template
- Read-only database connection for all query-time SQL

---

## CA Feature Flags

The following feature flags control CA-specific capabilities. They are declared in `manifest.json` and effective only when this module is active. They appear in the tenant's `tenant.json` under the `features` key and are evaluated by the platform's `TenantConfigResolver`.

Missing keys default to `false` — deny by default for optional capabilities.

```json
{
  "features": {
    "ca_compliance_engine": true,
    "ca_renewal_prep": true,
    "ca_unit_owner_letters": true,
    "ca_board_presentation": true,
    "ca_fannie_mae_tracking": true
  }
}
```

| Flag | Controls |
|---|---|
| `ca_compliance_engine` | CC&R/bylaw/Fannie Mae compliance checking; `GET /api/insurance/compliance/:account_id`. Returns HTTP 403 if false. |
| `ca_renewal_prep` | Renewal package generation endpoint; `GET /api/insurance/accounts/:id/renewal-prep`. Returns HTTP 403 if false. |
| `ca_unit_owner_letters` | HO-6 letter batch generation; `POST /api/insurance/accounts/:id/unit-owner-letters`. Returns HTTP 403 if false. |
| `ca_board_presentation` | Board presentation package generation; `GET /api/insurance/accounts/:id/board-presentation`. Returns HTTP 403 if false. |
| `ca_fannie_mae_tracking` | Enables Fannie Mae requirement injection at account creation; if false, Fannie Mae rows are not written to `insurance_requirements`. |

> **API Routing**: The CA module mounts its automation endpoints under `/api/modules/community_associations/` via `register_api_router()`. The paths shown above are logical names; actual mounted paths follow the `ModuleRegistry` convention.

---

## Claude Extraction Prompts

### `prompts/ccr_bylaws.txt`

```
You are analyzing a community association governing document (CC&Rs or Bylaws).
Extract all insurance-related requirements and obligations.

For EACH insurance requirement found, extract:
- The exact quoted text of the requirement
- Which coverage line it applies to (property, gl, do, crime, umbrella, workers_comp, flood, earthquake, ho6)
- Whether it specifies a minimum limit (extract dollar amount if present)
- The limit type (per_occurrence, aggregate, replacement_cost, etc.)
- Whether it references Fannie Mae, FHA, or another external standard
- The document section or article number where this appears

Also extract:
- Association legal name (exact)
- Property address
- Number of units (residential and commercial separately if specified)
- Construction type if mentioned (wood_frame, masonry, fire_resistive)
- Year built if mentioned
- Any mention of deductible assessment provisions for unit owners

Return JSON:
{
  "association_name": "...",
  "property_address": "...",
  "units_residential": null,   // maps to insurance_accounts.units_residential
  "units_commercial": null,    // maps to insurance_accounts.units_commercial
  "construction_type": null,
  "year_built": null,
  "document_type": "ccr" | "bylaws",
  "insurance_requirements": [
    {
      "requirement_text": "exact quoted text",
      "coverage_line": "property | gl | do | crime | umbrella | workers_comp | flood | earthquake | ho6",
      "min_limit": 1000000.00,
      "min_limit_type": "per_occurrence | aggregate | replacement_cost | per_unit",
      "references_external_standard": "fannie_mae | fha | none",
      "section_reference": "Article VIII, Section 3",
      "ho6_required_for_unit_owners": true | false | null,
      "ho6_minimum_amount": null
    }
  ],
  "deductible_assessment_provision": true | false,
  "deductible_assessment_limit": null
}
```

### `prompts/reserve_study.txt`

```
You are analyzing a community association reserve study document.
Extract the key financial and structural findings.

Extract:
- Study preparation date and firm name
- Percent funded (as a number 0-100, not a percentage string)
- Fully funded balance (dollar amount)
- Current reserve fund balance (dollar amount)
- Recommended annual reserve contribution (dollar amount)
- Total replacement cost new for all components (dollar amount)
- Number of components analyzed
- Study type: "full" (involves site visit) or "update" (desktop/paper update)
- Funding plan type if stated (baseline, threshold, full_funding)
- Recommended date of next study
- Any notes about adequacy of current coverage (if the study comments on insurance)
- Top 5 components by replacement cost (component name + replacement cost + remaining useful life)

Return JSON:
{
  "study_date": "YYYY-MM-DD",
  "study_firm": "...",
  "percent_funded": 67.5,
  "fully_funded_balance": 850000.00,
  "actual_reserve_balance": 571000.00,
  "annual_contribution": 45000.00,
  "replacement_cost_new": 2400000.00,
  "component_count": 28,
  "study_type": "full" | "update",
  "funding_plan": "baseline | threshold | full_funding | reserve_specialist",
  "next_study_date": "YYYY-MM-DD",
  "top_components": [
    {"component": "Roof - Flat membrane", "replacement_cost": 180000.00, "remaining_useful_life_years": 8}
  ],
  "insurance_notes": "The current insured replacement cost of $3.2M appears adequate..."
}
```

### `prompts/board_minutes.txt`

```
You are analyzing community association board meeting minutes.
Extract only insurance-related resolutions, decisions, and discussions.

For each insurance-related item found:
- Date of the meeting
- What was decided or discussed (exact text when available)
- Resolution type: coverage_approval | carrier_change | deductible_change | special_assessment | coverage_waiver | other
- Who made the motion (if stated)
- Vote result: approved | denied | tabled
- Effective date (if stated)

Also note:
- Any mention of claims (claim number, loss date, carrier, status)
- Any discussion of coverage gaps or compliance issues
- Any mention of renewal, quote requests, or broker changes

Return JSON:
{
  "meeting_date": "YYYY-MM-DD",
  "insurance_resolutions": [
    {
      "resolution_type": "coverage_approval | carrier_change | deductible_change | special_assessment | coverage_waiver | other",
      "description": "Full description of what was decided...",
      "motion_by": "Director Smith",
      "vote_result": "approved | denied | tabled",
      "effective_date": "YYYY-MM-DD"
    }
  ],
  "claims_discussed": [
    {
      "claim_description": "...",
      "claim_number": null,
      "date_of_loss": null,
      "status": "open | closed | pending"
    }
  ],
  "coverage_issues_noted": []
}
```

### `prompts/appraisal.txt`

```
You are analyzing a property appraisal for a community association.
Extract the replacement cost and valuation data used for insurance purposes.

Extract:
- Appraisal date, appraiser name and firm
- Property address
- Total insured replacement value (dollar amount)
- Per-unit replacement value if stated
- Methodology (Marshall & Swift, E2Value, manual estimate)
- Exclusions (contents, land, etc.)
- Whether replacement cost or actual cash value
- Number of buildings and total units appraised

Return JSON:
{
  "appraisal_date": "YYYY-MM-DD",
  "appraiser_name": "...",
  "appraiser_firm": "...",
  "property_address": "...",
  "total_insured_replacement_value": 3200000.00,
  "per_unit_replacement_value": 19000.00,
  "number_of_buildings": 1,
  "total_units": 168,
  "valuation_type": "replacement_cost | actual_cash_value | agreed_value",
  "methodology": "Marshall & Swift | E2Value | Manual",
  "exclusions": ["land", "contents"],
  "building_components": [
    {"component": "Structure", "value": 2800000.00},
    {"component": "HVAC", "value": 250000.00}
  ]
}
```

---

## CC&R Compliance Engine

### `compliance.py` — ComplianceChecker Implementation

The `CommunityAssociationsComplianceChecker` implements the core `ComplianceChecker` protocol.

**Requirement sources** (all read from `insurance_requirements` table):
- `requirement_source = 'ccr'` — extracted from CC&R documents
- `requirement_source = 'bylaws'` — extracted from bylaws
- `requirement_source = 'fannie_mae'` — injected at account creation (see Fannie Mae section)
- `requirement_source = 'fha'` — injected at account creation

### Requirement Source Precedence

When two sources specify conflicting limits for the same `(account_id, coverage_line, min_limit_type)`:

**Precedence ladder** (highest wins):
1. External statutory/regulatory: `fannie_mae`, `fha`, `loan_agreement`
2. Governing documents: `ccr`, `bylaws`
3. Advisory: `management_agreement`, `board_resolution`

**Merge rule**: Apply the **highest minimum limit** regardless of source (most protective). Both requirements remain persisted and visible in compliance output. The agent sees both — the system never discards either silently.

**Example**: CC&Rs require $1M GL per occurrence; Fannie Mae requires $1M GL per occurrence (same); result: $1M required, `is_met` evaluated against $1M. If CC&Rs required $500K and Fannie Mae required $1M, effective minimum is $1M (Fannie Mae wins on highest-limit rule).

**Advisory requirements** (`advisory_only: true`) are surfaced as `INFO` items — they are displayed in compliance output but do not set `is_met = false`.

### CA-Specific Compliance Checks

Beyond the core limit-comparison algorithm:

- **Reserve adequacy flag**: If `ca_reserve_studies.percent_funded < 10`, emit `WARNING` on property coverage line with `ca_context = 'reserve_underfunded'`
- **Appraisal currency**: If no appraisal document exists OR most recent appraisal is >3 years old, emit `WARNING` on property coverage with `ca_context = 'appraisal_stale'`
- **HO-6 letter gap**: If CC&Rs require HO-6 for unit owners AND any `ca_unit_owners` row has `ho6_required = true` with no completed batch in `ca_letter_batches`, emit `ACTION_REQUIRED` with `ca_context = 'ho6_letters_not_sent'`

**Output**: `list[ComplianceGap]` — same structure as core, with additional `ca_context` field.

---

## Fannie Mae / FHA Requirements

### Persistence and Versioning

Fannie Mae and FHA requirements are **injected as `insurance_requirements` rows** at account creation — not evaluated from static code at runtime. The Python dict `FANNIE_MAE_REQUIREMENTS` is the source of injection; the database is the authoritative runtime source.

**Required fields on injected rows**:

| Field | Value |
|---|---|
| `requirement_source` | `'fannie_mae'` or `'fha'` |
| `requirements_version` | e.g., `'2026-Q1'` — from `manifest.json:fannie_mae_governance.fannie_mae_reqs_version` |
| `effective_date` | Date the guidelines took effect (from module manifest) |
| `injected_at` | Timestamp of injection |
| `injected_by_module_version` | Module version string (e.g., `'1.0'`) |
| `superseded_at` | NULL until an updated version supersedes this row |

### Fannie Mae Reference Requirements (v2026-Q1)

```python
FANNIE_MAE_REQUIREMENTS = [
    {
        "coverage_line": "property",
        "requirement_text": "100% of insurable replacement cost (IRC) — no coinsurance clause",
        "min_limit_type": "replacement_cost",
        "min_limit_pct_of_replacement_cost": 100,
    },
    {
        "coverage_line": "property",
        "requirement_text": "Maximum deductible 5% of insured value for all perils including wind/hail",
        "max_deductible_pct": 5,
    },
    {
        "coverage_line": "gl",
        "requirement_text": "Minimum $1,000,000 per occurrence / $2,000,000 aggregate commercial general liability",
        "min_limit": 1_000_000,
        "min_limit_type": "per_occurrence",
    },
    {
        "coverage_line": "do",
        "requirement_text": "D&O liability coverage recommended — Fannie Mae-favorable but not required",
        "advisory_only": True,
    },
    {
        "coverage_line": "fidelity",
        "requirement_text": "Fidelity/crime coverage equal to 3 months maximum assessments plus reserve fund balance",
        "formula": "3_months_assessments_plus_reserves",
    },
]
```

### Governance and Update Procedure

| Field | Value |
|---|---|
| **Version** | `2026-Q1` |
| **Last reviewed** | 2026-02-27 |
| **Review owner** | Platform team |
| **Next review due** | 2027-01-01 |
| **Source** | Fannie Mae Single Family Selling Guide, B7-3 and B7-4 |

**Update procedure** when Fannie Mae publishes new guidelines:
1. Update `FANNIE_MAE_REQUIREMENTS` dict in `compliance.py`
2. Bump `fannie_mae_reqs_version` in `manifest.json` (e.g., `2026-Q1` → `2026-Q3`)
3. Update `last_reviewed_at` and `next_review_due` in `manifest.json`
4. Write a module upgrade migration: set `superseded_at = NOW()` on all existing `fannie_mae` rows for all accounts; re-inject new requirements with new version
5. Run regression tests (AC-CA-06 and AC-CA-05 must pass)
6. Bump module version (e.g., `1.0` → `1.1`) in `manifest.json`

**Rollback**: If new requirements contain errors, re-run migration in reverse: set `superseded_at = NULL` on prior version rows, set `superseded_at = NOW()` on the bad rows.

---

## Automation Features

### Rendering Boundary

All four automation services return **structured Python dicts**. They are not PDF/PPTX generators.

**Rendering is owned by the core `DocumentRenderer` service** (`ai_ready_rag/services/document_renderer.py`). Modules declare a `template_name` in `dashboard.json`; core maps template names to Jinja2/PPTX templates. The module never calls a renderer directly.

**Failure semantics**: If rendering fails (missing template, renderer error), the API returns the raw structured dict with `Content-Type: application/json` and `X-Render-Failed: true` so the agent can at least access the data. Failed renders are logged as `RENDER_ERROR` audit events.

**Artifact lifecycle**: Rendered outputs are ephemeral (memory for the request lifetime). They are not persisted to storage. The agent downloads the artifact from the API response; the server holds nothing after the response completes.

### Renewal Prep (`automations/renewal_prep.py`)

**Class**: `RenewalPrepService`
**Method**: `generate(account_id: str) -> dict`

**Input**: `account_id` (TEXT UUID)

**Output contract**:
```python
{
  "account": {
    "name": str,
    "primary_address": str,
    "city": str,
    "state": str,
    "account_type": str,   # "condo_association" | "hoa" | "planned_community"
    "custom_fields": dict, # management company, unit counts, year built, construction type
  },
  "coverage_schedule": [
    {
      "line_of_business": str,
      "carrier_name": str,
      "policy_number": str,
      "inception_date": str,         # ISO 8601
      "expiration_date": str,        # ISO 8601
      "premium": float,              # dollars
      "limits": [
        {
          "coverage_type": str,
          "limit_per_occurrence": float | None,
          "limit_aggregate": float | None,
          "deductible": float | None,
        }
      ]
    }
  ],
  "total_premium": float,            # sum of all active policy premiums
  "claims_summary": {
    "total_claims": int,
    "total_incurred": float,
    "open_claims": int,
    "by_line": {
      "<line_of_business>": {
        "count": int,
        "total_incurred": float,
        "open": int,
      }
    }
  },
  "reserve_summary": {
    "study_date": str | None,        # ISO 8601
    "percent_funded": float | None,
    "actual_reserve_balance": float | None,
    "replacement_cost_new": float | None,
    "study_firm": str | None,
  },
  "appraisal_summary": {
    "appraisal_date": str | None,    # ISO 8601
    "total_insured_replacement_value": float | None,
    "appraiser_firm": str | None,
  },
  "compliance_gaps": [               # list[ComplianceGap] as dicts
    {
      "coverage_line": str,
      "requirement_source": str,
      "requirement_text": str,
      "min_limit": float | None,
      "is_met": bool,
      "current_limit": float | None,
      "gap_amount": float | None,
      "ca_context": str | None,      # CA-specific warning context
    }
  ]
}
```

**All 6 top-level keys must be present**. `coverage_schedule` must be a non-empty list for the acceptance test to pass (AC-CA-09).

### CC&R Compliance Checker (`compliance.py`)

**Class**: `CommunityAssociationsComplianceChecker`
**Method**: `check(account_id: str, db: Session) -> list[ComplianceGap]`

**Output contract** (each `ComplianceGap`):
```python
{
  "coverage_line": str,            # e.g., "gl", "property", "do"
  "requirement_source": str,       # "ccr" | "bylaws" | "fannie_mae" | "fha"
  "requirement_text": str,         # exact text of the requirement
  "min_limit": float | None,       # required minimum in dollars
  "min_limit_type": str | None,    # "per_occurrence" | "aggregate" | "replacement_cost"
  "is_met": bool,                  # False = gap exists
  "current_limit": float | None,   # actual coverage limit found
  "gap_amount": float | None,      # min_limit - current_limit (positive = gap)
  "advisory_only": bool,           # True = INFO only, does not set is_met=False
  "ca_context": str | None,        # "reserve_underfunded" | "appraisal_stale" | "ho6_letters_not_sent"
}
```

**Severity levels**:
- `is_met = false` and not `advisory_only`: hard compliance gap — must resolve before renewal
- `advisory_only = true`: INFO — surfaced but does not block
- `ca_context` set: CA-specific contextual warning (may be on a met requirement)

### Unit Owner Letter Generator (`automations/unit_owner_letter.py`)

**Class**: `UnitOwnerLetterService`
**Method**: `generate(account_id: str, unit_number: str | None = None) -> dict`

**Input**: `account_id`, optional `unit_number` (if omitted, generates for all units with `ho6_required = true` and no completed batch in `ca_letter_batches`).

**Batch workflow**:
1. Create `ca_letter_batches` row: `status = 'generating'`, `total_units = N`
2. For each unit: generate letter; write `ca_letter_batch_items` row with `letter_status = 'generated'` or `'failed'`
3. On all items complete with no failures: set batch `status = 'generated'`, `completed_at = NOW()`
4. On any failure: set batch `status = 'failed'`; failed items can be retried within the same batch

**Letter content populated from**:
- Association name (from `insurance_accounts.account_name`)
- Minimum HO-6 amount (from `ca_unit_owners.ho6_minimum_amount` or CC&R requirement)
- Loss assessment coverage requirement (from `insurance_requirements` where `coverage_line = 'ho6'`)
- Agent name and contact (from tenant config)

**Output contract**:
```python
{
  "batch_id": str,                 # UUID
  "status": str,                   # "pending" | "generating" | "generated" | "failed"
  "total_units": int,
  "generated_count": int,
  "failed_count": int,
  "completed_at": str | None,      # ISO 8601 datetime
  "letters": [
    {
      "unit_number": str,
      "owner_name": str | None,    # decrypted PII (only in response, not stored plaintext)
      "letter_text": str | None,   # generated letter content
      "letter_status": str,        # "pending" | "generated" | "failed"
      "failure_reason": str | None,
    }
  ]
}
```

### Board Presentation Package (`automations/board_presentation.py`)

**Class**: `BoardPresentationService`
**Method**: `generate(account_id: str) -> dict`

**Output contract**:
```python
{
  "program_overview": {            # same structure as coverage_schedule in RenewalPrepService
    "lines": [
      {
        "line_of_business": str,
        "carrier_name": str,
        "policy_number": str,
        "inception_date": str,
        "expiration_date": str,
        "premium": float,
        "limits": list[dict],
      }
    ]
  },
  "premium_summary": {
    "current_year_total": float,
    "prior_year_total": float | None,  # None if no prior year data
    "change_amount": float | None,
    "change_pct": float | None,        # decimal (0.12 = 12% increase)
  },
  "compliance_status": {
    "ccr": str,                    # "green" | "yellow" | "red"
    "fannie_mae": str,             # "green" | "yellow" | "red"
    "fha": str,                    # "green" | "yellow" | "red"
    "open_gap_count": int,
  },
  "reserve_adequacy": {
    "percent_funded": float | None,
    "actual_reserve_balance": float | None,
    "replacement_cost_new": float | None,
    "last_study_date": str | None, # ISO 8601
    "trend": str | None,           # "improving" | "declining" | "stable" | None
  },
  "loss_history": {
    "total_claims": int,
    "total_incurred": float,
    "open_claims": int,
    "by_line": dict,               # same structure as RenewalPrepService.claims_summary.by_line
  },
  "renewal_recommendations": [str] # list of action items derived from open compliance gaps
}
```

**Status color thresholds for `compliance_status`**:
- `"green"`: 0 open gaps for that source
- `"yellow"`: 1–2 open gaps
- `"red"`: 3 or more open gaps

---

## Dashboard View Configuration

### `dashboard.json`

```json
{
  "module": "community_associations",
  "views": [
    {
      "view_id": "ca_program_overview",
      "display_name": "Insurance Program",
      "icon": "shield",
      "layout": "cards",
      "cards": [
        {
          "card_id": "coverage_schedule",
          "title": "Coverage Schedule",
          "data_source": "ca_coverage_schedule",
          "display_type": "table",
          "columns": ["line_of_business", "carrier_name", "policy_number", "expiration_date", "premium"]
        },
        {
          "card_id": "total_premium",
          "title": "Total Program Premium",
          "data_source": "ca_premium_query",
          "display_type": "metric",
          "aggregate": "sum(premium)",
          "format": "currency"
        },
        {
          "card_id": "compliance_status",
          "title": "Compliance Status",
          "data_source": "ca_compliance_gap",
          "display_type": "status_indicator",
          "green_condition": "count = 0",
          "yellow_condition": "count <= 2",
          "red_condition": "count > 2"
        }
      ]
    },
    {
      "view_id": "ca_reserve_study",
      "display_name": "Reserve Study",
      "icon": "chart",
      "layout": "detail",
      "data_source": "ca_reserve_status",
      "display_type": "metric_group",
      "metrics": [
        {"field": "percent_funded", "label": "Percent Funded", "format": "percent",
         "threshold_green": 70, "threshold_yellow": 30},
        {"field": "actual_reserve_balance", "label": "Current Balance", "format": "currency"},
        {"field": "fully_funded_balance", "label": "Fully Funded Target", "format": "currency"},
        {"field": "replacement_cost_new", "label": "Replacement Cost New", "format": "currency"}
      ]
    },
    {
      "view_id": "ca_compliance",
      "display_name": "Compliance",
      "icon": "checkmark",
      "layout": "table",
      "data_source": "ca_compliance_gap",
      "show_all_requirements": true,
      "actions": [
        {"action_id": "export_compliance_report", "label": "Export Report"},
        {"action_id": "generate_board_presentation", "label": "Board Presentation"}
      ]
    },
    {
      "view_id": "ca_unit_owners",
      "display_name": "Unit Owners",
      "icon": "users",
      "layout": "table",
      "data_source": "ca_unit_owner_status",
      "actions": [
        {"action_id": "generate_ho6_letters", "label": "Generate HO-6 Letters"},
        {"action_id": "export_unit_list", "label": "Export Unit List"}
      ]
    }
  ],
  "automation_actions": [
    {
      "action_id": "ca_renewal_prep",
      "display_name": "Renewal Package",
      "description": "Generate complete renewal submission data",
      "service": "renewal_prep",
      "renderer": "core.document_renderer",
      "template_name": "ca_renewal_package",
      "output_format": "pdf",
      "fallback": "json"
    },
    {
      "action_id": "ca_board_presentation",
      "display_name": "Board Presentation",
      "description": "Coverage summary + compliance + renewal recommendations",
      "service": "board_presentation",
      "renderer": "core.document_renderer",
      "template_name": "ca_board_presentation",
      "output_format": "pptx",
      "fallback": "json"
    },
    {
      "action_id": "ca_unit_owner_letters",
      "display_name": "HO-6 Letters",
      "description": "Generate HO-6 requirement letters for all units",
      "service": "unit_owner_letter",
      "renderer": "core.document_renderer",
      "template_name": "ca_ho6_letter",
      "output_format": "pdf",
      "fallback": "json"
    }
  ]
}
```

---

## Module-Specific SQL Query Templates (Legacy — Reserve Study and Requirements)

These two templates are in addition to the 8 templates in the main SQL Template Catalog above. They address CA-specific data not covered by the primary insurance templates.

### Template: `ca_reserve_status`

| | |
|---|---|
| **Parameters** | `:account_id TEXT` |
| **Output columns** | `study_date, study_firm, percent_funded, actual_reserve_balance, fully_funded_balance, annual_contribution, replacement_cost_new` |
| **Triggers on** | "reserve study", "percent funded", "reserve balance", "reserve fund" |

```sql
SELECT rs.study_date, rs.study_firm, rs.percent_funded,
       rs.actual_reserve_balance, rs.fully_funded_balance,
       rs.annual_contribution, rs.replacement_cost_new
FROM ca_reserve_studies rs
WHERE rs.account_id = :account_id
ORDER BY rs.study_date DESC
LIMIT 1
```

### Template: `ca_requirements_by_source`

| | |
|---|---|
| **Parameters** | `:account_id TEXT`, `:requirement_source TEXT` — must be one of `'ccr','bylaws','fannie_mae','fha','loan_agreement'` |
| **Output columns** | `coverage_line, requirement_text, min_limit, min_limit_type, is_met, current_limit, gap_amount` |
| **Triggers on** | "what do the cc&rs require", "bylaw requirements", "fannie mae requirements" |

```sql
SELECT r.coverage_line, r.requirement_text, r.min_limit,
       r.min_limit_type, r.is_met, r.current_limit, r.gap_amount
FROM insurance_requirements r
WHERE r.account_id = :account_id
  AND r.requirement_source = :requirement_source
  AND r.superseded_at IS NULL
ORDER BY r.coverage_line
```

### Template: `ca_unit_owner_status`

| | |
|---|---|
| **Parameters** | `:account_id TEXT` |
| **Output columns** | `unit_number, ho6_required, ho6_minimum_amount, latest_batch_status, latest_batch_date` |
| **Triggers on** | "unit owner letters", "ho-6 requirements", "which units need letters" |

```sql
SELECT u.unit_number, u.ho6_required, u.ho6_minimum_amount,
       b.status AS latest_batch_status, b.completed_at AS latest_batch_date
FROM ca_unit_owners u
LEFT JOIN ca_letter_batches b ON b.account_id = u.account_id
  AND b.id = (SELECT id FROM ca_letter_batches
              WHERE account_id = u.account_id
              ORDER BY initiated_at DESC LIMIT 1)
WHERE u.account_id = :account_id
ORDER BY u.unit_number
```

### Template: `ca_board_resolutions`

| | |
|---|---|
| **Parameters** | `:account_id TEXT`, `:limit INTEGER` (default 20) |
| **Output columns** | `resolution_date, resolution_type, description, vote_result, effective_date` |
| **Triggers on** | "board resolutions", "board approved", "what did the board decide" |

```sql
SELECT br.resolution_date, br.resolution_type, br.description,
       br.vote_result, br.effective_date
FROM ca_board_resolutions br
WHERE br.account_id = :account_id
ORDER BY br.resolution_date DESC
LIMIT :limit
```

---

## Unit Owner Data Privacy

### PII Classification

`ca_unit_owners` introduces three PII columns. These are subject to the platform-level Fernet encryption (see `VAULTIQ_PLATFORM_v2.md`, PII Controls section). The columns are declared here so the encryption layer knows what to encrypt.

| Column | PII Classification | Encrypted at Rest |
|---|---|---|
| `owner_name` | Personal identifier | Yes — Fernet |
| `owner_email` | Contact information | Yes — Fernet |
| `mailing_address` | Physical location | Yes — Fernet |
| `unit_number` | Not PII (property identifier) | No |
| `ho6_required` | Not PII | No |
| `ho6_minimum_amount` | Not PII (coverage data) | No |

### Access Controls

Access to `ca_unit_owners` data follows the platform's tag-based access control — agents see only unit owners for accounts matching their assigned tags.

### Retention Policy

| Event | Retention Action |
|---|---|
| Customer account cancelled | PII columns (`owner_name`, `owner_email`, `mailing_address`) set to NULL within 30 days; row skeleton retained for audit trail (unit_number, ho6_required, timestamps) |
| Unit owner data update | Prior values are overwritten (no history kept) — unit owner PII is not versioned |
| Export/delete request | PII columns set to NULL immediately; audit event `UNIT_OWNER_PII_DELETED` logged |

**v1.0 scope**: Retention actions are triggered manually by platform admin. Automated retention sweep is deferred to v1.1.

---

## Quality Validation — CA Gold Set

The following 16 questions are the canonical Marshall Wells evaluation set for this module. All 16 must pass the confidence threshold before v1.0 ships.

**Acceptance thresholds** (from platform spec):
- Standard tier (Claude primary): `confidence >= 90` on all 16 questions
- Enterprise tier (Ollama primary): `confidence >= 70` on all 16 questions

**Source of answer** is annotated: `SQL` = deterministic query path; `RAG` = semantic search path; `HYBRID` = SQL context + RAG synthesis.

| # | Question | Expected Path | Expected Data Source |
|---|---|---|---|
| 1 | What is the GL per-occurrence limit for Marshall Wells Lofts? | SQL | `insurance_coverages` via `ca_coverage_by_account_line` |
| 2 | Who is the property insurance carrier for Marshall Wells? | SQL | `insurance_policies` via `ca_carrier_lookup` |
| 3 | What is the total annual premium for the Marshall Wells program? | SQL | `insurance_policies` via `ca_premium_query` |
| 4 | When does the D&O policy expire for Marshall Wells? | SQL | `insurance_policies` via `ca_policy_dates` |
| 5 | What is the property deductible for Marshall Wells? | SQL | `insurance_coverages` via `ca_coverage_by_account_line` |
| 6 | Does Marshall Wells have a fidelity/crime policy? | SQL | `insurance_policies` via `ca_carrier_lookup` |
| 7 | What claims has Marshall Wells had in the last 5 years? | SQL | `insurance_claims` via `ca_claims_history` |
| 8 | Does Marshall Wells meet Fannie Mae insurance requirements? | SQL | `insurance_requirements` via `ca_compliance_gap` |
| 9 | What does the CC&R require for GL coverage? | SQL | `insurance_requirements` via `ca_requirements_by_source` |
| 10 | What is the reserve fund percent funded for Marshall Wells? | SQL | `ca_reserve_studies` via `ca_reserve_status` |
| 11 | How many units does Marshall Wells have? | SQL | `insurance_accounts` via entity map |
| 12 | What are the insured replacement values from the most recent appraisal? | SQL/RAG | `ca_reserve_studies` or RAG on appraisal document |
| 13 | Summarize the coverage gaps for Marshall Wells compared to CC&R requirements | HYBRID | SQL compliance gap + RAG on CC&R text |
| 14 | What did the board approve at the last meeting regarding insurance? | RAG | `ca_board_resolutions` or RAG on board minutes |
| 15 | Generate a coverage schedule for the Marshall Wells renewal submission | SQL | `ca_coverage_schedule` template |
| 16 | Compare the Marshall Wells GL limits to what Fannie Mae requires | HYBRID | SQL on both policies and requirements |

**Evaluation harness**: Questions are run by `tests/eval/eval_runner.py` against a PostgreSQL test database seeded with Marshall Wells fixture documents. Each question has a rubric:
- **Correct answer**: the specific fact(s) the answer must contain
- **Expected confidence band**: minimum acceptable confidence score
- **Expected query path**: SQL, RAG, or HYBRID — path mismatch is a routing bug, not just an answer quality issue
- **Acceptable latency**: < 500ms for SQL-path questions; < 5 seconds for analytical questions

---

## Implementation Notes

### Depends On (Core Platform)

- `insurance_accounts`, `insurance_policies`, `insurance_coverages`, `insurance_requirements` tables — managed by this module's `001_insurance_tables.py` migration
- `ComplianceChecker` protocol (core `services/compliance_service.py`)
- `PromptResolver` (core `tenant/resolver.py`)
- `ModuleRegistry` with `register()` entry point protocol (core `modules/registry.py`)
- `DocumentRenderer` service (core `services/document_renderer.py`) — renders automation outputs
- The core SQL template registry (core `services/query_router.py`) — `register_sql_templates()` method
- The core entity routing map — `register_entity_map()` method in the core enrichment service
- The core document classifier — `register_document_classifiers()` method

The four `register_*` methods on `ModuleRegistry` must be implemented as part of the core Platform Phase 2 work **before** this module can be loaded.

### Depends On (Internal Order)

1. `001_insurance_tables.py` migration runs before `002_ca_tables.py` (FK dependencies)
2. `002_ca_tables.py` migration runs before any CA data is written
3. `module.register()` called at startup: classifiers → entity map → SQL templates → compliance checker
4. Fannie Mae requirements injected as `insurance_requirements` rows on account creation; `requirements_version` set from `manifest.json`
5. `CommunityAssociationsComplianceChecker` registered after tables exist

---

## Test Fixtures

The following fixture files must exist before acceptance criteria can be validated. Marshall Wells Lofts documents are the primary source. Synthetic fixtures are acceptable for types not available from Marshall Wells.

| Fixture File | Type | Used By |
|---|---|---|
| `test_data/mwl_ccr.pdf` | CC&Rs | AC-CA-02, AC-CA-03, AC-CA-05, AC-CA-11 |
| `test_data/mwl_bylaws.pdf` | Bylaws | AC-CA-02, AC-CA-03 |
| `test_data/mwl_reserve_study.pdf` | Reserve study | AC-CA-04, AC-CA-07 |
| `test_data/mwl_appraisal.pdf` | Appraisal | AC-CA-09 (appraisal_summary) |
| `test_data/mwl_board_minutes.pdf` | Board minutes | AC-CA-12 |
| `test_data/mwl_board_packet_mixed.pdf` | Mixed (ambiguous) | AC-CA-13 |
| `test_data/25-26 D&O Crime Policy.pdf` | Policy | AC-CA-09 (coverage_schedule) |
| `test_data/ACORD 25 fillable.pdf` | Certificate | AC-CA-01 (module load test) |

**Status**: Marshall Wells policy and certificate files exist in `test_data/`. CC&R, bylaws, reserve study, appraisal, board minutes, and mixed packet files must be sourced or created before v1.0 ships.

---

## Acceptance Criteria

### AC-CA-01: Module Loads Without Error
```
Pass: `python -m ai_ready_rag.modules.registry --list` shows `community_associations`;
      four registration calls complete without error in startup logs.
```

### AC-CA-02: Classifiers Registered — CC&R
Upload `test_data/mwl_ccr.pdf`. System classifies as `document_type = 'ccr'`.
```sql
SELECT document_type FROM enrichment_synopses WHERE document_id = '<uploaded_id>';
-- Expected: 'ccr'
```

### AC-CA-03: CC&R Requirement Extraction
After enriching `mwl_ccr.pdf`, `insurance_requirements` has ≥1 row with `requirement_source = 'ccr'` for the associated account.
```sql
SELECT COUNT(*) FROM insurance_requirements WHERE account_id = :id AND requirement_source = 'ccr';
-- Expected: >= 1
```

### AC-CA-04: Reserve Study Extraction
After enriching `mwl_reserve_study.pdf`, `ca_reserve_studies` has a row with non-null `percent_funded` and `replacement_cost_new`.
```sql
SELECT percent_funded, replacement_cost_new FROM ca_reserve_studies WHERE account_id = :id;
-- Expected: row exists, both non-null
```

### AC-CA-05: Compliance Gap Detection — Limit Shortfall
For an account with CC&R requiring $2M GL aggregate and active policy showing $1M:
```sql
SELECT is_met, gap_amount FROM insurance_requirements
WHERE account_id = :id AND coverage_line = 'gl' AND min_limit_type = 'aggregate'
  AND requirement_source = 'ccr';
-- Expected: is_met=false, gap_amount=1000000.00
```

### AC-CA-06: Fannie Mae Requirements Persisted with Version
For a Fannie Mae–certified account, rows exist with `requirement_source = 'fannie_mae'`, non-null `requirements_version`, and non-null `injected_by_module_version`.
```sql
SELECT COUNT(*), requirements_version, injected_by_module_version
FROM insurance_requirements
WHERE account_id = :id AND requirement_source = 'fannie_mae'
GROUP BY requirements_version, injected_by_module_version;
-- Expected: >= 3 rows, requirements_version = '2026-Q1', injected_by_module_version = '1.0'
```

### AC-CA-07: Reserve Adequacy Warning
For an account where `ca_reserve_studies.percent_funded < 10`, `check_compliance()` returns ≥1 `ComplianceGap` with `ca_context = 'reserve_underfunded'`.
```
Pass: Warning present in compliance output for property coverage line.
```

### AC-CA-08: CA Coverage Schedule Query Routes to SQL
Natural language query "What is the coverage schedule for [account]?" routes to `ca_coverage_schedule` SQL template.
```
Pass: SQL path taken (no RAG), ≥1 row per active policy, latency < 500ms.
```

### AC-CA-09: Renewal Package Generation
`RenewalPrepService.generate(account_id)` returns dict with all 6 required keys.
```
Pass: Keys present: account, coverage_schedule, total_premium, claims_summary,
      reserve_summary, compliance_gaps. coverage_schedule is non-empty list.
      appraisal_summary key also present (dict; values may be None if no appraisal uploaded).
```

### AC-CA-10: HO-6 Letter Batch Safety
Calling `UnitOwnerLetterService.generate(account_id)` for 5 units creates a `ca_letter_batches` row and 5 `ca_letter_batch_items` rows. On success, batch `status = 'generated'`. `ca_unit_owners` rows are unchanged.
```sql
SELECT status, total_units, generated_count FROM ca_letter_batches
WHERE account_id = :id ORDER BY initiated_at DESC LIMIT 1;
-- Expected: status='generated', total_units=5, generated_count=5
```

### AC-CA-11: Account Name Normalization
Uploading a document with insured "River Oaks Homeowners Association, Inc." where account "River Oaks HOA" exists and ≥2 corroborating signals match: document auto-links to existing account.
```sql
SELECT insurance_account_id FROM documents WHERE id = :doc_id;
-- Expected: River Oaks HOA account_id (not NULL, not a new account)
```

### AC-CA-12: Board Resolution Extraction
After enriching `mwl_board_minutes.pdf` containing a coverage approval resolution:
```sql
SELECT resolution_type, vote_result FROM ca_board_resolutions WHERE account_id = :id;
-- Expected: row with resolution_type='coverage_approval', vote_result='approved'
```

### AC-CA-13: Classifier Ambiguity Gate
Uploading `mwl_board_packet_mixed.pdf` (mixed board minutes + reserve study content) produces a `review_items` row rather than proceeding to extraction.
```sql
SELECT review_reason, candidate_types FROM review_items WHERE document_id = :doc_id;
-- Expected: review_reason='ambiguous_classification',
--           candidate_types contains 'board_minutes' and 'reserve_study'
```

### AC-CA-14: Insurance Policy Extraction — Canonicalization
Upload `test_data/25-26 D&O Crime Policy.pdf`. After enrichment:
- `insurance_policies` has a row with non-null `carrier_name`, `policy_number`, `inception_date`, `expiration_date`
- Monetary values stored as REAL (no `$` or commas): `insurance_coverages.limit_per_occurrence` is a number, not a string
- Dates stored as ISO 8601: `inception_date` matches `YYYY-MM-DD` pattern
```sql
SELECT carrier_name, policy_number, inception_date, expiration_date
FROM insurance_policies WHERE source_document_id = :doc_id;
-- Expected: non-null values, inception_date format '20XX-XX-XX'
SELECT limit_per_occurrence FROM insurance_coverages
WHERE policy_id = (SELECT id FROM insurance_policies WHERE source_document_id = :doc_id LIMIT 1);
-- Expected: numeric REAL value, not null, no string formatting
```

### AC-CA-15: Gold Set — SQL Path Latency
Run gold set questions 1–9 (SQL-path questions) against a seeded PostgreSQL test database. All must return within 500ms as measured from API request receipt to response bytes sent.
```
Pass: eval_runner.py reports p95 latency < 500ms for SQL-path questions.
```

### AC-CA-16: CA Feature Flag Enforcement
Set `"ca_compliance_engine": false` in `tenant.json`, restart. `GET /api/insurance/compliance/:account_id` returns HTTP 403.
```
Pass: HTTP 403 returned. Setting back to true and restarting restores access.
```

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Formal plugin interface (4 registration methods) | `module.register()` entry point only | Removes core-modularity contradiction; makes "no core edits per module" a hard guarantee |
| Insurance tables in this module's migration | `001_insurance_tables.py` owned by CA module | CA is the launch vertical; these tables are insurance-vertical tables, not core platform tables. Future verticals may add their own extension tables or share these. |
| Fannie Mae requirements persisted, not runtime-evaluated | Rows in `insurance_requirements` with version fields | Auditable, replaceable on guideline update, testable with SQL; static dict is source of injection only |
| HO-6 batch tracking | `ca_letter_batches` + `ca_letter_batch_items` | Failure-safe; supports partial retries; distinguishes generated vs. delivered |
| Ambiguity gate at 0.10 confidence gap | Review queue routing for ambiguous docs | Prevents wrong extraction from common board packets; consistent with platform review workflow |
| UNIQUE indexes + ON CONFLICT per table | Enforced at DB level, not application level | Idempotency can't be a documentation promise; DB must enforce it under concurrency |
| Highest-limit merge rule across requirement sources | Apply max across all sources for same (account, line, limit_type) | Most protective; both requirements remain visible; agent sees conflict |
| Rendering owned by core DocumentRenderer | Module declares template_name; never calls renderer directly | Single rendering surface; consistent failure handling; module never owns binary output |
| Minimum 2 corroboration signals for auto-link in 95-99% band | Required before auto-link | Prevents false merges between same-named associations in different cities |
| Unit owner PII in dedicated encrypted columns | Fernet + explicit retention policy | Module introduces PII; can't defer to implicit platform controls without declaring which fields |
| Reserve study as separate table | History across years | Multiple studies per account over time; keeps history |
| CA carrier alias file separate from core | `ca_carrier_aliases.csv` in module's `data/` folder | CA vertical has its own carrier universe; avoids polluting core alias table; can be updated independently |
| 16-question gold set fixed at spec commit | Additive only — new questions do not replace old | Fixed calibration set ensures regressions are detectable; additions are permitted, removals require spec change |

---

*Spec: Community Associations Module v1.2*
*Parent: `specs/VAULTIQ_PLATFORM_v2.md` (v2.0)*
*Created: 2026-02-27 | Updated: 2026-02-27*
