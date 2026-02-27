---
title: "VaultIQ Module — Community Associations Insurance"
status: DRAFT
version: v1.1
created: 2026-02-27
updated: 2026-02-27
author: —
type: Vertical Module Spec
parent_spec: specs/VAULTIQ_PLATFORM_v1.md (v1.1)
complexity: COMPLEX
stack: backend + frontend
changes: v1.1 — Engineering review: plugin interface contract, Fannie Mae persistence/versioning,
         HO-6 batch safety, classifier ambiguity gate, idempotency enforcement, requirement
         precedence, template governance, account matching guards, schema constraints,
         renderer boundary, test fixtures, Fannie Mae governance, unit owner PII policy
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
- Six unique document types with classifiers and Claude extraction prompts
- Classifier ambiguity gate — routes mixed documents to review queue before extraction
- Four CA-specific database tables (`ca_reserve_studies`, `ca_unit_owners`, `ca_board_resolutions`, `ca_letter_batches`)
- Enforced idempotency: UNIQUE indexes + ON CONFLICT semantics on all CA tables
- CC&R/bylaw compliance engine with deterministic requirement-source precedence
- Fannie Mae/FHA requirement source — persisted in `insurance_requirements` with versioning
- Renewal prep automation — one-click coverage schedule for annual submissions
- Unit owner letter generation — failure-safe batch workflow with letter-level status tracking
- Board presentation package — coverage summary + compliance check + renewal recommendation
- Six module-specific SQL query templates with parameter/output contracts
- Dashboard view configuration (multi-policy program view)
- Rendering boundary: all automation outputs rendered by core `DocumentRenderer` service
- Unit owner PII classification and retention policy
- Fannie Mae requirements governance (version, owner, update procedure)
- 13 acceptance criteria mapped to test fixtures

### Out of Scope — v1.0

- Live Fannie Mae/FHA certification status lookup (future: carrier API integration)
- Unit owner certificate tracking (future: matches Construction subcontractor cert module)
- Reserve study financial modeling (extract data only; projections are out of scope)
- Email integration for unit owner letter delivery
- Board meeting scheduling or calendar features

---

## Module File Structure

```
ai_ready_rag/modules/community_associations/
├── manifest.json              ← Self-description + Fannie Mae governance metadata
├── classifiers.yaml           ← Document type detection rules + ambiguity gate config
├── prompts/
│   ├── ccr_bylaws.txt         ← CC&R and bylaw extraction prompt
│   ├── reserve_study.txt      ← Reserve study extraction prompt
│   ├── board_minutes.txt      ← Board minutes extraction prompt
│   ├── appraisal.txt          ← Appraisal/valuation extraction prompt
│   ├── unit_owner_letter.txt  ← Unit owner correspondence extraction prompt
│   └── fannie_mae_reqs.txt    ← Fannie Mae/FHA requirement injection template
├── migrations/
│   └── 001_ca_tables.py       ← Alembic migration: 4 tables + all UNIQUE/CHECK constraints
├── compliance.py              ← ComplianceChecker implementation + precedence engine
├── automations/
│   ├── renewal_prep.py        ← Coverage schedule + submission data generator
│   ├── unit_owner_letter.py   ← HO-6 requirement letter batch generator (failure-safe)
│   └── board_presentation.py  ← Board package assembler
├── sql_templates.yaml         ← 6 module-specific SQL query templates with contracts
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
```

### The Four Extension Points

| Method | What It Does | Core Contract |
|---|---|---|
| `register_document_classifiers(module_id, path)` | Loads `classifiers.yaml` into `DocumentClassifier`; document types become available to enrichment pipeline | Core evaluates module classifiers after core classifiers; first match wins within module; ambiguity gate applied across all classifiers |
| `register_entity_map(module_id, map)` | Merges `CA_ENTITY_TO_TABLE_MAP` into `ENTITY_TO_TABLE_MAP` in `EnrichmentService`; no key collisions with core keys permitted | Core raises `DuplicateEntityKeyError` on startup if module adds a key already in core map |
| `register_sql_templates(module_id, path)` | Loads templates from `sql_templates.yaml` into `SQL_TEMPLATE_CATALOG`; template IDs prefixed with `ca_` to avoid collision | Core validates parameter schema on registration; rejects malformed templates at startup |
| `register_compliance_checker(module_id, cls)` | Registers `CommunityAssociationsComplianceChecker` with `ComplianceService`; checker is invoked for any account with this module enabled | Core calls `checker.check(account_id, db)` and merges result into the global compliance output |

### "No Further Core Edits" Guarantee

Once the four extension points are implemented in core (part of Phase 2 platform work), **no subsequent vertical module requires any core file changes**. This is the invariant that makes the module architecture viable at scale. Any feature request that cannot be satisfied through these four points requires a platform spec change, not an ad-hoc core edit.

---

## Module Manifest

```json
{
  "module_id": "community_associations",
  "version": "1.0",
  "display_name": "Community Associations",
  "description": "CC&R compliance, reserve study analysis, unit owner automation for HOA/condo agencies.",
  "document_types": [
    "ccr", "bylaws", "reserve_study", "appraisal", "board_minutes", "unit_owner_letter"
  ],
  "entity_types": [
    "unit_count", "reserve_fund_balance", "reserve_fund_percent_funded",
    "replacement_cost_new", "association_name", "management_company",
    "board_member", "fannie_mae_certification", "fha_certification",
    "ccr_requirement", "ho6_requirement"
  ],
  "compliance_rules": true,
  "schema_migrations": ["001_ca_tables.py"],
  "feature_flags": {
    "ca_compliance_engine": true,
    "ca_renewal_prep": true,
    "ca_unit_owner_letters": true,
    "ca_board_presentation": true
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

## Document Types and Classifiers

### `classifiers.yaml`

```yaml
# Community Associations — Document Type Classifiers
# Applied by DocumentClassifier after core classification passes.
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

## Schema Extensions

Four tables are added by the Community Associations module. They are **additive only** — no changes to core insurance tables.

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

**CHECK constraints** (enforced in migration):
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

**Removed from v1.0**: `letter_sent_date` — replaced by `ca_letter_batches` tracking. Letter delivery status is tracked at the batch level, not the unit level.

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

Failure-safe batch tracking for HO-6 letter generation. Replaces the `letter_sent_date` column that was on `ca_unit_owners` in v1.0 drafts.

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
  "units_residential": null,
  "units_commercial": null,
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

All three automation services return **structured Python dicts**. They are not PDF/PPTX generators.

**Rendering is owned by the core `DocumentRenderer` service** (`ai_ready_rag/services/document_renderer.py`). Modules declare a `template_name` in `dashboard.json`; core maps template names to Jinja2/PPTX templates. The module never calls a renderer directly.

**Failure semantics**: If rendering fails (missing template, renderer error), the API returns the raw structured dict with `Content-Type: application/json` and `X-Render-Failed: true` so the agent can at least access the data. Failed renders are logged as `RENDER_ERROR` audit events.

**Artifact lifecycle**: Rendered outputs are ephemeral (memory for the request lifetime). They are not persisted to storage. The agent downloads the artifact from the API response; the server holds nothing after the response completes.

### Renewal Prep (`automations/renewal_prep.py`)

**Returns**:
```python
{
  "account": {name, address, units_residential, units_commercial, year_built, construction_type},
  "coverage_schedule": [
    {line_of_business, carrier, policy_number, effective_date, expiration_date, annual_premium, limits: [...]}
  ],
  "total_premium": 48500.00,
  "claims_summary": {total_claims, total_incurred, open_claims, by_line: {...}},
  "reserve_summary": {study_date, percent_funded, actual_reserve_balance, replacement_cost_new},
  "appraisal_summary": {appraisal_date, total_insured_replacement_value},
  "compliance_gaps": [...]
}
```

### Unit Owner Letter Generator (`automations/unit_owner_letter.py`)

**Input**: `account_id`, optional `unit_number` (if omitted, generates for all units with `ho6_required = true` and no completed batch in `ca_letter_batches`).

**Batch workflow**:
1. Create `ca_letter_batches` row: `status = 'generating'`, `total_units = N`
2. For each unit: generate letter; write `ca_letter_batch_items` row with `letter_status = 'generated'` or `'failed'`
3. On all items complete with no failures: set batch `status = 'generated'`, `completed_at = NOW()`
4. On any failure: set batch `status = 'failed'`; failed items can be retried within the same batch

**Letter content populated from**:
- Association name (from `insurance_accounts.name`)
- Minimum HO-6 amount (from `ca_unit_owners.ho6_minimum_amount` or CC&R requirement)
- Loss assessment coverage requirement (from `insurance_requirements` where `coverage_line = 'ho6'`)
- Agent name and contact (from tenant config)

**Returns**: `{batch_id, status, total_units, letters: [{unit_number, owner_name, letter_text, letter_status}]}`

### Board Presentation Package (`automations/board_presentation.py`)

**Returns**:
```python
{
  "program_overview": coverage_schedule_dict,
  "premium_summary": {current_year_total, prior_year_total},
  "compliance_status": {ccr: "green|yellow|red", fannie_mae: "green|yellow|red", fha: "green|yellow|red"},
  "reserve_adequacy": {percent_funded, balance, trend, last_study_date},
  "loss_history": {total_claims, total_incurred, open_claims, by_line: {...}},
  "renewal_recommendations": ["list of action items from open compliance gaps"]
}
```

---

## Module-Specific SQL Query Templates

All templates are allowlisted, parameterized, read-only. Unmatched queries fall through to RAG per platform spec behavior (VAULTIQ_PLATFORM_v1.md, Deterministic Routing Specification).

### Template 1: `ca_coverage_schedule`

| | |
|---|---|
| **Parameters** | `:account_id TEXT` |
| **Output columns** | `line_of_business, carrier, policy_number, effective_date, expiration_date, annual_premium, coverage_type, limit_amount, deductible_amount, valuation` |
| **Triggers on** | "coverage schedule", "all lines", "program summary", "what coverage do we have" |
| **Row cap** | 1,000 (platform default) |

```sql
SELECT p.line_of_business, p.carrier, p.policy_number,
       p.effective_date, p.expiration_date, p.annual_premium,
       c.coverage_type, c.limit_amount, c.deductible_amount, c.valuation
FROM insurance_policies p
JOIN insurance_coverages c ON c.policy_id = p.id
WHERE p.account_id = :account_id AND p.status = 'active'
ORDER BY p.line_of_business, c.coverage_type
```

### Template 2: `ca_compliance_gaps`

| | |
|---|---|
| **Parameters** | `:account_id TEXT` |
| **Output columns** | `coverage_line, requirement_source, requirement_text, min_limit, is_met, current_limit, gap_amount` |
| **Triggers on** | "compliance gaps", "requirements not met", "what are we missing", "cc&r compliance" |

```sql
SELECT r.coverage_line, r.requirement_source, r.requirement_text,
       r.min_limit, r.is_met, r.current_limit, r.gap_amount
FROM insurance_requirements r
WHERE r.account_id = :account_id AND r.is_met = false AND r.superseded_at IS NULL
ORDER BY r.coverage_line
```

### Template 3: `ca_reserve_status`

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

### Template 4: `ca_requirements_by_source`

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

### Template 5: `ca_unit_owner_status`

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

### Template 6: `ca_board_resolutions`

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
          "columns": ["line_of_business", "carrier", "policy_number", "expiration_date", "annual_premium"]
        },
        {
          "card_id": "total_premium",
          "title": "Total Program Premium",
          "data_source": "ca_coverage_schedule",
          "display_type": "metric",
          "aggregate": "sum(annual_premium)",
          "format": "currency"
        },
        {
          "card_id": "compliance_status",
          "title": "Compliance Status",
          "data_source": "ca_compliance_gaps",
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
      "data_source": "ca_compliance_gaps",
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

## Entity-to-SQL Mapping Extensions

```python
CA_ENTITY_TO_TABLE_MAP = {
    "unit_count":                  ("insurance_accounts", "units_residential"),
    "reserve_fund_balance":        ("ca_reserve_studies", "actual_reserve_balance"),
    "reserve_fund_percent_funded": ("ca_reserve_studies", "percent_funded"),
    "replacement_cost_new":        ("ca_reserve_studies", "replacement_cost_new"),
    "association_name":            ("insurance_accounts", "name"),
    "management_company":          ("insurance_accounts", "custom_fields"),  # JSONB
    "ccr_requirement":             ("insurance_requirements", "requirement_text"),
    "ho6_requirement":             ("insurance_requirements", "requirement_text"),
    "fannie_mae_certification":    ("insurance_accounts", "custom_fields"),  # JSONB
    "fha_certification":           ("insurance_accounts", "custom_fields"),  # JSONB
}
```

---

## Account Matching Considerations

The core 3-tier matching handles most documents. CA documents require pre-processing before the core similarity check.

**CA-specific stripping rules** (applied before difflib):
- Remove: "Homeowners Association", "Homeowner's Association"
- Remove: "Condominium Association", "Condo Association"
- Remove: "Owners Association", "Owner's Association"
- Remove: ", Inc.", ", LLC", ", Ltd."
- Normalize: "Homeowners" → "HOA", "Condominium" → "Condo"

**Protected-qualifier exception**: If the canonical name in the system contains a geographic or distinguishing qualifier (e.g., "River Oaks HOA Portland"), **do not auto-link** a document whose post-normalization name lacks that qualifier. Route to review (Tier 2) instead. This prevents false merges between same-named associations in different cities.

**Minimum corroboration requirement for auto-link** (Tier 1, 95-99% band): When normalized name similarity is between 95-99% (not the unambiguous ≥99% case), **at least 2 corroborating signals** are required before auto-link:

| Signal | Match Criterion |
|---|---|
| Property address | Street number + street name match (city/state optional) |
| Unit count | Within ±10% of count in system |
| Management company | Exact string match in JSONB `custom_fields` |
| CA-type document from same property | Another document already linked to this account |

If <2 corroborating signals are available: route to review (Tier 2) regardless of name similarity score.

**Sample name variations handled**:
| Document name | Post-normalization | System name |
|---|---|---|
| "River Oaks Homeowners Association, Inc." | "River Oaks" | "River Oaks HOA" → matches |
| "Marshall Wells Lofts Owners Association" | "Marshall Wells Lofts" | "Marshall Wells Lofts" → exact |

---

## Unit Owner Data Privacy

### PII Classification

`ca_unit_owners` introduces three PII columns. These are subject to the platform-level Fernet encryption (see `VAULTIQ_PLATFORM_v1.md`, PII Controls section). The columns are declared here so the encryption layer knows what to encrypt.

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

## Implementation Notes

### Depends On (Core Platform)

- `insurance_accounts`, `insurance_policies`, `insurance_coverages`, `insurance_requirements` tables (core schema)
- `ComplianceChecker` protocol (core `services/compliance_service.py`)
- `PromptResolver` (core `tenant/resolver.py`)
- `ModuleRegistry` with `register()` entry point protocol (core `modules/registry.py`)
- `DocumentRenderer` service (core `services/document_renderer.py`) — renders automation outputs
- `SQL_TEMPLATE_CATALOG` (core `services/query_router.py`) — `register_sql_templates()` method
- `ENTITY_TO_TABLE_MAP` — `register_entity_map()` method in `EnrichmentService`
- `DocumentClassifier` — `register_document_classifiers()` method

The four `register_*` methods on `ModuleRegistry` must be implemented as part of the core Platform Phase 2 work **before** this module can be loaded.

### Depends On (Internal Order)

1. `001_ca_tables.py` migration runs before any CA data is written
2. `module.register()` called at startup: classifiers → entity map → SQL templates → compliance checker
3. Fannie Mae requirements injected as `insurance_requirements` rows on account creation; `requirements_version` set from `manifest.json`
4. `CommunityAssociationsComplianceChecker` registered after tables exist

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
```

### AC-CA-10: HO-6 Letter Batch Safety
Calling `UnitOwnerLetterService.generate(account_id)` for 5 units creates a `ca_letter_batches` row and 5 `ca_letter_batch_items` rows. On success, batch `status = 'generated'`. `ca_unit_owners` rows are unchanged (no `letter_sent_date` column).
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

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Formal plugin interface (4 registration methods) | `module.register()` entry point only | Removes core-modularity contradiction; makes "no core edits per module" a hard guarantee |
| Fannie Mae requirements persisted, not runtime-evaluated | Rows in `insurance_requirements` with version fields | Auditable, replaceable on guideline update, testable with SQL; static dict is source of injection only |
| HO-6 batch tracking | `ca_letter_batches` + `ca_letter_batch_items` replaces `letter_sent_date` column | Failure-safe; supports partial retries; distinguishes generated vs. delivered |
| Ambiguity gate at 0.10 confidence gap | Review queue routing for ambiguous docs | Prevents wrong extraction from common board packets; consistent with platform review workflow |
| UNIQUE indexes + ON CONFLICT per table | Enforced at DB level, not application level | Idempotency can't be a documentation promise; DB must enforce it under concurrency |
| Highest-limit merge rule across requirement sources | Apply max across all sources for same (account, line, limit_type) | Most protective; both requirements remain visible; agent sees conflict |
| Rendering owned by core DocumentRenderer | Module declares template_name; never calls renderer directly | Single rendering surface; consistent failure handling; module never owns binary output |
| Minimum 2 corroboration signals for auto-link in 95-99% band | Required before auto-link | Prevents false merges between same-named associations in different cities |
| Unit owner PII in dedicated encrypted columns | Fernet + explicit retention policy | Module introduces PII; can't defer to implicit platform controls without declaring which fields |
| Reserve study as separate table | History across years | Multiple studies per account over time; keeps history |

---

*Spec: Community Associations Module v1.1*
*Parent: `specs/VAULTIQ_PLATFORM_v1.md` (v1.1)*
*Created: 2026-02-27 | Updated: 2026-02-27*
