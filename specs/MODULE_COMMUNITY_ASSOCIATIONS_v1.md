---
title: "VaultIQ Module — Community Associations Insurance"
status: DRAFT
version: v1.0
created: 2026-02-27
updated: 2026-02-27
author: —
type: Vertical Module Spec
parent_spec: specs/VAULTIQ_PLATFORM_v1.md (v1.1)
complexity: COMPLEX
stack: backend + frontend
---

# VaultIQ Module — Community Associations Insurance

## Summary

The Community Associations module is the **launch vertical** for VaultIQ. It extends the core platform with six unique document types, community-association-specific Claude extraction prompts, a CC&R and bylaw compliance engine, reserve study analysis, and automation features (unit owner letter generation, board presentation packages). All capabilities are implemented as a self-describing module package loaded at platform startup — no changes to core platform code.

**Target customer**: Insurance agencies specializing in condominium associations, homeowner associations (HOAs), and community associations. Marshall Wells Lofts is the primary test case.

**Why this vertical first**: Community associations are document-intensive (CC&Rs, bylaws, reserve studies, board minutes, appraisals, HO-6 coordination), compliance-driven (Fannie Mae/FHA certification requirements, CC&R minimum coverage mandates), and renewal-predictable (annual programs). Every new commercial lines account follows the same document pattern. Automation ROI is highest in this segment.

---

## Scope

### In Scope — v1.0

- Module manifest, classifiers, and file structure
- Six unique document types with classifiers and Claude extraction prompts
- Three module-specific database tables (`ca_reserve_studies`, `ca_unit_owners`, `ca_board_resolutions`)
- CC&R/bylaw compliance engine — extract requirements, cross-reference with current coverage
- Fannie Mae/FHA requirement source (external requirement injection, no live API)
- Renewal prep automation — one-click coverage schedule for annual submissions
- Unit owner letter generation — HO-6 requirement notices from current coverage data
- Board presentation package — coverage summary + compliance check + renewal recommendation
- Six module-specific SQL query templates
- Dashboard view configuration (multi-policy program view)
- Acceptance criteria (12 checks)

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
├── manifest.json              ← Self-description (document types, migrations, feature flags)
├── classifiers.yaml           ← Document type detection rules
├── prompts/
│   ├── ccr_bylaws.txt         ← CC&R and bylaw extraction prompt
│   ├── reserve_study.txt      ← Reserve study extraction prompt
│   ├── board_minutes.txt      ← Board minutes extraction prompt
│   ├── appraisal.txt          ← Appraisal/valuation extraction prompt
│   ├── unit_owner_letter.txt  ← Unit owner correspondence extraction prompt
│   └── fannie_mae_reqs.txt    ← Fannie Mae/FHA requirement injection template
├── migrations/
│   └── 001_ca_tables.py       ← Alembic migration for 3 CA-specific tables
├── compliance.py              ← ComplianceChecker implementation
├── automations/
│   ├── renewal_prep.py        ← Coverage schedule + submission data generator
│   ├── unit_owner_letter.py   ← HO-6 requirement letter generator
│   └── board_presentation.py  ← Board package assembler
├── sql_templates.yaml         ← 6 module-specific SQL query templates
└── dashboard.json             ← Multi-policy program view configuration
```

---

## Module Manifest

```json
{
  "module_id": "community_associations",
  "version": "1.0",
  "display_name": "Community Associations",
  "description": "CC&R compliance, reserve study analysis, unit owner automation for HOA/condo agencies.",
  "document_types": [
    "ccr",
    "bylaws",
    "reserve_study",
    "appraisal",
    "board_minutes",
    "unit_owner_letter"
  ],
  "entity_types": [
    "unit_count",
    "reserve_fund_balance",
    "reserve_fund_percent_funded",
    "replacement_cost_new",
    "association_name",
    "management_company",
    "board_member",
    "fannie_mae_certification",
    "fha_certification",
    "ccr_requirement",
    "ho6_requirement"
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
  "dashboard": "dashboard.json"
}
```

---

## Document Types and Classifiers

### `classifiers.yaml`

```yaml
# Community Associations — Document Type Classifiers
# Applied by DocumentClassifier after core classification passes.
# Each classifier is evaluated in order; first match wins.
# If no classifier matches, the document falls through to core classification.

module: community_associations

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

---

## Schema Extensions

Three tables are added by the Community Associations module. They are **additive only** — no changes to core insurance tables.

### `ca_reserve_studies`

Reserve study data extracted from reserve study documents.

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
| study_type | TEXT | Yes | "full" (site visit) or "update" (desktop) |
| next_study_date | DATE | Yes | Recommended next study date |
| funding_plan | TEXT | Yes | "baseline", "threshold", "full_funding", "reserve_specialist" |
| notes | TEXT | Yes | Analyst notes or caveats |
| created_at | DATETIME | No | Timestamp |

**Idempotency key**: `(account_id, study_date, study_firm)` — same study from same firm on same date is a no-op on re-upload.

### `ca_unit_owners`

Unit owner roster for HO-6 requirement tracking.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| account_id | TEXT FK | No | → insurance_accounts.id |
| unit_number | TEXT | No | Unit identifier (e.g., "101", "B-4") |
| owner_name | TEXT | Yes | Unit owner name |
| owner_email | TEXT | Yes | For letter delivery (future) |
| mailing_address | TEXT | Yes | Owner mailing address if different from unit |
| ho6_required | BOOLEAN | No | Whether HO-6 is required for this unit |
| ho6_minimum_amount | REAL | Yes | Minimum required HO-6 coverage amount |
| letter_sent_date | DATE | Yes | When last HO-6 requirement letter was sent |
| source_document_id | TEXT FK | Yes | Document this was extracted from |
| created_at | DATETIME | No | Timestamp |
| updated_at | DATETIME | No | Last update timestamp |

**Idempotency key**: `(account_id, unit_number)` — unit number is unique within an account.

### `ca_board_resolutions`

Insurance-related board resolutions extracted from meeting minutes.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | TEXT PK | No | UUID |
| account_id | TEXT FK | No | → insurance_accounts.id |
| document_id | TEXT FK | No | → documents.id (board minutes) |
| resolution_date | DATE | Yes | Date of meeting where resolution passed |
| resolution_type | TEXT | Yes | "coverage_approval", "carrier_change", "deductible_change", "special_assessment", "coverage_waiver" |
| description | TEXT | No | Full text of resolution |
| motion_by | TEXT | Yes | Director who made the motion |
| vote_result | TEXT | Yes | "approved", "denied", "tabled" |
| effective_date | DATE | Yes | When resolution takes effect |
| created_at | DATETIME | No | Timestamp |

**Idempotency key**: `(account_id, resolution_date, resolution_type, vote_result)`.

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
- Appraisal date
- Appraiser name and firm
- Property address
- Total insured replacement value (dollar amount)
- Per-unit replacement value if stated
- Building components and their replacement costs if itemized
- Methodology used (e.g., Marshall & Swift, E2Value, manual estimate)
- Any exclusions from the appraisal (contents, land, etc.)
- Whether this is a full replacement cost appraisal or actual cash value
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

**Requirement sources checked** (in priority order):
1. **CC&R requirements** — extracted from association governing documents (`insurance_requirements` where `requirement_source = 'ccr'`)
2. **Bylaw requirements** — extracted from association bylaws (`insurance_requirements` where `requirement_source = 'bylaws'`)
3. **Fannie Mae requirements** — injected as static requirements for any account with `fannie_mae_certification = true` (see Fannie Mae Reference below)
4. **FHA requirements** — injected as static requirements for any account with `fha_certification = true`

**Compliance check logic** (see core spec `VAULTIQ_PLATFORM_v1.md` for `check_compliance` algorithm):

Additional CA-specific checks:
- **Reserve study adequacy flag**: If `ca_reserve_studies.percent_funded < 10`, emit `WARNING` for all property coverages (underfunded reserves = higher insured risk)
- **Appraisal currency**: If no appraisal document exists OR most recent appraisal is >3 years old, emit `WARNING` on property coverage (replacement cost may be stale)
- **HO-6 requirement tracking**: If CC&Rs require unit owners to carry HO-6 and `ca_unit_owners` has rows with `ho6_required = true` and `letter_sent_date IS NULL`, emit `ACTION_REQUIRED` — letters not yet sent

**Output**: `list[ComplianceGap]` — same structure as core, with additional `ca_context` field for reserve/appraisal warnings.

### Fannie Mae Reference Requirements

These are static requirements injected for Fannie Mae–certified accounts. No live API call — Fannie Mae warrantability guidelines as of publication date.

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
        "requirement_text": "Directors & Officers liability coverage recommended (not required but Fannie Mae-favorable)",
        "advisory_only": True,
    },
    {
        "coverage_line": "fidelity",
        "requirement_text": "Fidelity/crime coverage equal to 3 months maximum assessments plus reserve fund balance",
        "formula": "3_months_assessments_plus_reserves",
    },
]
```

**Note**: Fannie Mae periodically updates warrantability guidelines. `FANNIE_MAE_REQUIREMENTS` is pinned at the time of v1.0 publication and should be reviewed annually. A `fannie_mae_reqs_version` field in the module manifest will track last review date.

---

## Automation Features

### Renewal Prep (`automations/renewal_prep.py`)

Generates a structured submission package for annual renewal.

**Data assembled**:
1. Account overview: name, address, unit count, year built, construction type
2. Coverage schedule: all active policies (line, carrier, policy number, effective/expiration, premium, key limits)
3. Total program premium (sum across all lines)
4. 5-year loss summary (count, total incurred, open claims, by line)
5. Reserve study summary: percent funded, balance, replacement cost new (from most recent `ca_reserve_studies` row)
6. Appraisal summary: replacement value, appraisal date (from most recent appraisal document)
7. Compliance status: any open gaps from `ComplianceChecker`

**Output format**: Structured dict (rendered to PDF/DOCX template by caller, not this service).

**SQL query used** (registered as `ca_renewal_package` template):
```sql
SELECT
    a.name, a.address, a.units_residential, a.units_commercial,
    a.year_built, a.construction_type,
    p.line_of_business, p.carrier, p.policy_number,
    p.effective_date, p.expiration_date, p.annual_premium,
    p.program_name
FROM insurance_accounts a
JOIN insurance_policies p ON p.account_id = a.id
WHERE a.id = :account_id
  AND p.status = 'active'
ORDER BY p.line_of_business
```

### Unit Owner Letter Generator (`automations/unit_owner_letter.py`)

Generates HO-6 requirement notices for all unit owners in `ca_unit_owners`.

**Input**: `account_id`, optional `unit_number` (if omitted, generates for all units with `ho6_required = true` and `letter_sent_date IS NULL`)

**Letter content populated from**:
- Association name (from `insurance_accounts.name`)
- Association property address
- Minimum HO-6 coverage amount (from `ca_unit_owners.ho6_minimum_amount` or CC&R requirement)
- Loss assessment coverage requirement (from `insurance_requirements` where `coverage_line = 'ho6'`)
- Agent name and contact (from tenant config)
- Effective date of requirement

**Output**: List of `{unit_number, owner_name, letter_text}` dicts. Caller handles delivery.

**After generation**: Sets `ca_unit_owners.letter_sent_date = today()` for all letters generated.

### Board Presentation Package (`automations/board_presentation.py`)

Assembles a presentation-ready coverage summary for board meetings.

**Sections**:
1. **Program Overview** — coverage schedule table (all active policies, one row per line)
2. **Total Premium Summary** — current year vs. prior year comparison (if prior year data exists)
3. **Compliance Status** — green/yellow/red for each requirement source (CC&Rs, Fannie Mae, FHA)
4. **Reserve Adequacy** — percent funded, balance, trend from most recent reserve study
5. **Loss History** — 5-year summary table (count, total incurred, open claims by line)
6. **Renewal Recommendations** — bullet list of open compliance gaps and actions needed

**Output**: Structured dict for template rendering. Not a PDF generator itself.

---

## Module-Specific SQL Query Templates

Registered in `sql_templates.yaml` and added to the core `SQL_TEMPLATE_CATALOG`.

### Template 1: `ca_coverage_schedule`
Full coverage schedule for an account, including all active lines.
```sql
SELECT p.line_of_business, p.carrier, p.policy_number,
       p.effective_date, p.expiration_date, p.annual_premium,
       c.coverage_type, c.limit_amount, c.deductible_amount, c.valuation
FROM insurance_policies p
JOIN insurance_coverages c ON c.policy_id = p.id
WHERE p.account_id = :account_id AND p.status = 'active'
ORDER BY p.line_of_business, c.coverage_type
```
**Triggers on**: "coverage schedule", "all lines", "program summary", "what coverage do we have"

### Template 2: `ca_compliance_gaps`
All open compliance gaps for an account.
```sql
SELECT r.coverage_line, r.requirement_source, r.requirement_text,
       r.min_limit, r.is_met, r.current_limit, r.gap_amount
FROM insurance_requirements r
WHERE r.account_id = :account_id AND r.is_met = false
ORDER BY r.coverage_line
```
**Triggers on**: "compliance gaps", "requirements not met", "what are we missing", "cc&r compliance"

### Template 3: `ca_reserve_status`
Most recent reserve study summary for an account.
```sql
SELECT rs.study_date, rs.study_firm, rs.percent_funded,
       rs.actual_reserve_balance, rs.fully_funded_balance,
       rs.annual_contribution, rs.replacement_cost_new
FROM ca_reserve_studies rs
WHERE rs.account_id = :account_id
ORDER BY rs.study_date DESC
LIMIT 1
```
**Triggers on**: "reserve study", "percent funded", "reserve balance", "reserve fund"

### Template 4: `ca_requirements_by_source`
All requirements from a specific source (CC&Rs, bylaws, Fannie Mae, FHA).
```sql
SELECT r.coverage_line, r.requirement_text, r.min_limit,
       r.min_limit_type, r.is_met, r.current_limit, r.gap_amount
FROM insurance_requirements r
WHERE r.account_id = :account_id
  AND r.requirement_source = :requirement_source
ORDER BY r.coverage_line
```
**Triggers on**: "what do the cc&rs require", "bylaw requirements", "fannie mae requirements"

### Template 5: `ca_unit_owner_status`
Unit owner HO-6 letter tracking status.
```sql
SELECT unit_number, owner_name, ho6_required,
       ho6_minimum_amount, letter_sent_date
FROM ca_unit_owners
WHERE account_id = :account_id
ORDER BY unit_number
```
**Triggers on**: "unit owner letters", "ho-6 requirements", "which units need letters"

### Template 6: `ca_board_resolutions`
Insurance-related board resolutions for an account.
```sql
SELECT br.resolution_date, br.resolution_type, br.description,
       br.vote_result, br.effective_date
FROM ca_board_resolutions br
WHERE br.account_id = :account_id
ORDER BY br.resolution_date DESC
LIMIT :limit
```
**Triggers on**: "board resolutions", "board approved", "what did the board decide"

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
          "columns": ["line_of_business", "carrier", "policy_number", "expiration_date", "annual_premium"],
          "sort": "line_of_business"
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
      "sections": [
        {
          "section_id": "reserve_summary",
          "title": "Reserve Fund Summary",
          "data_source": "ca_reserve_status",
          "display_type": "metric_group",
          "metrics": [
            {"field": "percent_funded", "label": "Percent Funded", "format": "percent",
             "threshold_green": 70, "threshold_yellow": 30},
            {"field": "actual_reserve_balance", "label": "Current Balance", "format": "currency"},
            {"field": "fully_funded_balance", "label": "Fully Funded Target", "format": "currency"},
            {"field": "replacement_cost_new", "label": "Replacement Cost New", "format": "currency"}
          ]
        }
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
      "output_format": "pdf"
    },
    {
      "action_id": "ca_board_presentation",
      "display_name": "Board Presentation",
      "description": "Coverage summary + compliance + renewal recommendations",
      "service": "board_presentation",
      "output_format": "pptx"
    },
    {
      "action_id": "ca_unit_owner_letters",
      "display_name": "HO-6 Letters",
      "description": "Generate HO-6 requirement letters for all units",
      "service": "unit_owner_letter",
      "output_format": "pdf"
    }
  ]
}
```

---

## Entity-to-SQL Mapping Extensions

Additional entity types registered by this module (extends core `ENTITY_TO_TABLE_MAP`):

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

The core 3-tier account matching (≥95% auto-link, 75-95% review, <75% auto-create) handles most community association documents. However, CA documents present specific name variation patterns that require pre-processing before matching:

**Legal entity variations commonly seen**:
| Raw Name in Document | Canonical Name in System |
|---|---|
| "River Oaks Homeowners Association" | "River Oaks HOA" |
| "River Oaks HOA, Inc." | "River Oaks HOA" |
| "Association of River Oaks" | "River Oaks HOA" |
| "Marshall Wells Lofts Condominium" | "Marshall Wells Lofts" |
| "Marshall Wells Lofts Owners Association" | "Marshall Wells Lofts" |

**Additional CA-specific stripping rules** (applied before difflib comparison):
- Remove: "Homeowners Association", "Homeowner's Association", "HOA"
- Remove: "Condominium Association", "Condo Association"
- Remove: "Owners Association", "Owner's Association"
- Remove: ", Inc.", ", LLC", ", Ltd."
- Normalize: "Homeowners" → "HOA", "Condominium" → "Condo"

These rules are applied by the CA module's `AccountMatcher` subclass before delegating to the core 3-tier logic.

**Corroborating signals** used to break ties at 75-95% name similarity:
- Same property address (street number + street name match)
- Same unit count (if extractable from both documents)
- Same management company name (JSONB custom field)

---

## Implementation Notes

### Depends On (Core Platform)

- `insurance_accounts`, `insurance_policies`, `insurance_coverages`, `insurance_requirements` tables (core schema)
- `ComplianceChecker` protocol (core `services/compliance_service.py`)
- `PromptResolver` (core `tenant/resolver.py`) — resolves prompts with tenant-level overrides
- `ModuleRegistry` (core `modules/registry.py`) — loads this module at startup
- `SQL_TEMPLATE_CATALOG` (core `services/query_router.py`) — accepts module template additions
- `DocumentType` enum extension — core uses `classifiers.yaml` from loaded modules
- `ENTITY_TO_TABLE_MAP` — core accepts module additions via `register_entity_map()`

### Depends On (This Module — Internal Order)

1. `001_ca_tables.py` migration must run before any CA data is written
2. CC&R/bylaw prompts must be registered before document enrichment begins
3. `CommunityAssociationsComplianceChecker` registered with compliance service after tables exist
4. Fannie Mae static requirements injected as `insurance_requirements` rows with `requirement_source = 'fannie_mae'` during account creation (not enrichment)

### Marshall Wells Lofts — Primary Test Case

Marshall Wells Lofts (168 residential units, Portland OR) is the development test case for this module. All extraction prompts and compliance rules must be validated against Marshall Wells documents before v1.0 ships.

Test documents available in `test_data/`:
- `25-26 D&O Crime Policy.pdf` — policy extraction
- `ACORD 25 fillable.pdf`, `acord_24_2016-03.pdf`, `Acord-80.pdf` — certificate extraction
- Additional Marshall Wells documents to be added during development

---

## Acceptance Criteria

### AC-CA-01: Module Loads Without Error
Running `python -m ai_ready_rag.modules.registry --list` shows `community_associations` in loaded modules.
```
Pass: Module appears in registry output, no startup errors.
```

### AC-CA-02: Classifiers Registered
Upload a CC&R PDF with filename "cc&r-riveroaks.pdf". System classifies as `document_type = 'ccr'`.
```sql
SELECT document_type FROM enrichment_synopses WHERE document_id = '<uploaded_id>';
-- Expected: 'ccr'
```

### AC-CA-03: CC&R Requirement Extraction
After enriching a CC&R document, `insurance_requirements` contains at least one row with `requirement_source = 'ccr'` for the associated account.
```sql
SELECT COUNT(*) FROM insurance_requirements WHERE account_id = :id AND requirement_source = 'ccr';
-- Expected: >= 1
```

### AC-CA-04: Reserve Study Extraction
After enriching a reserve study document, `ca_reserve_studies` has a row with non-null `percent_funded` and `replacement_cost_new`.
```sql
SELECT percent_funded, replacement_cost_new FROM ca_reserve_studies WHERE account_id = :id;
-- Expected: row exists, both columns non-null
```

### AC-CA-05: Compliance Gap Detection
For an account where CC&Rs require $2M GL aggregate and active policy shows $1M: `insurance_requirements.is_met = false`, `gap_amount = 1000000`.
```sql
SELECT is_met, gap_amount FROM insurance_requirements
WHERE account_id = :id AND coverage_line = 'gl' AND min_limit_type = 'aggregate';
-- Expected: is_met=false, gap_amount=1000000.00
```

### AC-CA-06: Fannie Mae Requirements Injected
For an account with `custom_fields->>'fannie_mae_certification' = 'true'`, `insurance_requirements` contains rows with `requirement_source = 'fannie_mae'`.
```sql
SELECT COUNT(*) FROM insurance_requirements WHERE account_id = :id AND requirement_source = 'fannie_mae';
-- Expected: >= 3 (property IRC, GL limit, fidelity)
```

### AC-CA-07: Reserve Adequacy Warning
For an account where `ca_reserve_studies.percent_funded < 10`, `check_compliance()` returns at least one `ComplianceGap` with `ca_context = 'reserve_underfunded'`.
```
Pass: Warning appears in compliance output for property coverage line.
```

### AC-CA-08: CA Coverage Schedule Query
Natural language query "What is the coverage schedule for [account]?" routes to `ca_coverage_schedule` SQL template and returns table with all active lines.
```
Pass: SQL path taken (no RAG), result has >= 1 row per active policy, latency < 500ms.
```

### AC-CA-09: Renewal Package Generation
Calling `RenewalPrepService.generate(account_id)` returns a dict with keys: `account`, `coverage_schedule`, `total_premium`, `claims_summary`, `reserve_summary`, `compliance_gaps`.
```
Pass: All 6 keys present, coverage_schedule is non-empty list.
```

### AC-CA-10: Unit Owner Letter Generation
Calling `UnitOwnerLetterService.generate(account_id)` for an account with 5 rows in `ca_unit_owners` where `ho6_required = true` and `letter_sent_date IS NULL` returns 5 letter dicts and updates `letter_sent_date` on all 5 rows.
```sql
SELECT COUNT(*) FROM ca_unit_owners WHERE account_id = :id AND letter_sent_date IS NOT NULL;
-- Expected: 5
```

### AC-CA-11: Account Name Normalization
Uploading a document with insured name "River Oaks Homeowners Association, Inc." where account "River Oaks HOA" exists in the system: name similarity ≥ 95% after CA stripping rules applied, resulting in auto-link.
```sql
SELECT insurance_account_id FROM documents WHERE id = :doc_id;
-- Expected: River Oaks HOA account_id (not NULL, not a new account)
```

### AC-CA-12: Board Resolution Extraction
After enriching board minutes containing a coverage approval resolution, `ca_board_resolutions` has a row with `resolution_type = 'coverage_approval'` and `vote_result = 'approved'`.
```sql
SELECT resolution_type, vote_result FROM ca_board_resolutions WHERE account_id = :id;
-- Expected: row with coverage_approval / approved
```

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Fannie Mae requirements as static data | Pinned in `compliance.py`, not DB | Avoids migration on each Fannie Mae guideline update; reviewed annually via module version |
| Reserve study as separate table | `ca_reserve_studies` not inline on `insurance_accounts` | One account may have multiple studies over time; keeps history |
| Unit owner roster optional | `ca_unit_owners` populated from documents, not required for enrichment | Not all agencies have unit owner lists; system degrades gracefully without them |
| HO-6 letter delivery out of scope | Generate text only; no email/portal delivery | Avoids SMTP dependencies and deliverability complexity in v1.0 |
| Board resolutions as structured table | Not just appended to notes | Enables SQL queries ("what did the board decide about deductibles") |
| CA name stripping applied before core matching | Module subclass, not modifying core | Preserves core algorithm; module can override without touching shared code |

---

*Spec: Community Associations Module v1.0*
*Parent: `specs/VAULTIQ_PLATFORM_v1.md` (v1.1)*
*Created: 2026-02-27*
