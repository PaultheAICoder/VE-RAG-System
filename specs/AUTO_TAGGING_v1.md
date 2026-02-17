# AUTO_TAGGING_v1 — Hybrid Auto-Tagging Specification

**Version:** 1.2
**Date:** 2026-02-16
**Status:** Draft
**Depends on:** Document upload flow, Tag system, User model, LLM integration, Processing pipeline
**Changelog:** v1.2 — Address engineering review: access policy enforcement, conflict resolution, strategy pinning, approval-mode decision table, YAML schema contract, operational guardrails, batch semantics, LLM failure handling, .msg scope reduction

---

## 1. Problem Statement

Organizations store documents in structured folder hierarchies that encode metadata: customer names, time periods, document categories, and workflow stages. Different industries use different conventions:

```
# Insurance Agency
Bethany Terrace (12-13)/24 NB/Quote/CNA/D&O Quote.pdf

# Law Firm
Acme Corp/2025/Discovery/Depositions/Smith_deposition_01-15.pdf

# Construction
Riverdale Phase 2/Bids/Subcontractor/HVAC_bid_johnson.pdf

# Generic/Flat
Company Reports/Q4_2024_Financial_Review.pdf
```

Currently, uploading these files requires **manual tag selection per document**. With 100+ files per customer and dozens of customers, this is impractical.

**Goal:** Automatically derive tags from folder structure (instant, zero cost) and document content (LLM-based, async) using **pluggable, industry-specific strategies** defined in YAML configuration.

---

## 2. Solution: Hybrid Auto-Tagging with Pluggable Strategies

### 2.1 Two-Stage Pipeline

| Stage | Trigger | Source | Cost | Accuracy |
|-------|---------|--------|------|----------|
| **Path-based** | Upload time (sync) | Folder structure | Zero (string parsing) | Deterministic |
| **LLM-based** | Post-chunking (async) | Document content | 1 LLM call per doc | Model-dependent |

### 2.2 Strategy Pattern

Each deployment selects an **auto-tagging strategy** that defines:

1. **Tag namespaces** — What categories of tags exist
2. **Path parsing rules** — How to extract tags from folder paths
3. **LLM classification prompt** — How to classify document content
4. **Document type taxonomy** — What document types are recognized
5. **Entity extraction fields** — What domain-specific entities to extract (carriers, opposing counsel, subcontractors, etc.)

Strategies are defined in **YAML files** stored in `data/auto_tag_strategies/`. No code changes needed to add a new industry.

### 2.3 Namespaced Tags

All auto-tags use a `namespace:value` format to avoid collisions with manual access-control tags:

| Namespace | Purpose | Example | Universal? |
|-----------|---------|---------|------------|
| `client:` | Customer/client/project name | `client:bethany-terrace` | Yes |
| `year:` | Time period | `year:2024-2025` | Yes |
| `doctype:` | Document classification | `doctype:contract` | Yes |
| `stage:` | Workflow stage | `stage:review` | Yes |
| `entity:` | Domain-specific entity | `entity:cna`, `entity:johnson-hvac` | Strategy-defined |
| `topic:` | Subject area / coverage line | `topic:liability`, `topic:zoning` | Strategy-defined |

The `client:`, `year:`, `doctype:`, and `stage:` namespaces are **universal** across all strategies. The `entity:` and `topic:` namespaces carry **strategy-specific semantics** (carrier vs opposing counsel vs subcontractor).

---

## 3. Strategy Definition Format (YAML)

### 3.1 File Location

```
data/auto_tag_strategies/
├── generic.yaml              # Built-in: works for any folder structure
├── insurance_agency.yaml     # Built-in: insurance brokerage
├── law_firm.yaml             # Built-in: legal practice
├── construction.yaml         # Built-in: construction/project management
└── custom.yaml               # User-created (optional, see Section 12 for examples)
```

### 3.2 Strategy Schema

```yaml
# ============================================================
# Auto-Tagging Strategy Definition
# ============================================================
strategy:
  id: "insurance_agency"
  name: "Insurance Agency"
  description: "For insurance brokerages managing customer policy folders"
  version: "1.0"

# ============================================================
# Tag Namespaces — what categories of tags this strategy produces
# ============================================================
namespaces:
  client:
    display: "Client"
    color: "#6366f1"          # Indigo
    description: "Customer or insured name"

  year:
    display: "Policy Year"
    color: "#f59e0b"          # Amber
    description: "Policy effective period"

  doctype:
    display: "Document Type"
    color: "#10b981"          # Emerald
    description: "Classification of document"

  stage:
    display: "Workflow Stage"
    color: "#64748b"          # Slate
    description: "Where in the workflow this document belongs"

  entity:
    display: "Carrier"
    color: "#0ea5e9"          # Sky
    description: "Insurance carrier or underwriter"

  topic:
    display: "Coverage Line"
    color: "#f43f5e"          # Rose
    description: "Type of insurance coverage"

# ============================================================
# Path Parsing Rules — extract tags from folder structure
# ============================================================
path_rules:
  # Client name: extracted from the root customer folder
  # Captures text before optional parenthetical date
  - namespace: "client"
    level: 0                  # 0 = first folder in relative path
    pattern: "^(.+?)(?:\\s*\\(.*\\))?$"
    capture_group: 1
    transform: "slugify"      # "Bethany Terrace" → "bethany-terrace"

  # Policy year: extracted from "24 NB" or "25 Renewal" folders
  - namespace: "year"
    level: 1
    pattern: "^(\\d{2})\\s+(?:NB|Renewal)$"
    capture_group: 1
    transform: "year_range"   # "24" → "2024-2025"

  # Workflow stage: mapped from known folder names
  - namespace: "stage"
    level: 2
    mapping:
      "Bind": "bind"
      "Quote": "quote"
      "Sub": "submission"
      "Policy": "policy"
      "Misc": "miscellaneous"
      "Docs": "governing-docs"
      "Pics": "photos"

  # Entity (carrier): extracted from subfolder under Quote/ or Sub/
  - namespace: "entity"
    level: 3
    parent_match: "^(Quote|Sub)$"   # Only extract when parent is Quote or Sub
    transform: "slugify"

# ============================================================
# Document Type Taxonomy — valid doctype: values for this strategy
# ============================================================
document_types:
  policy:
    display: "Policy"
    description: "Active insurance policy with terms and conditions"
    keywords: ["policy", "declarations", "dec page", "coverage form"]

  certificate:
    display: "Certificate"
    description: "ACORD 25/28 certificate of insurance"
    keywords: ["certificate of liability", "acord 25", "acord 28", "evidence of"]

  quote:
    display: "Quote"
    description: "Carrier proposal with pricing"
    keywords: ["quote", "proposal", "indication", "premium"]

  application:
    display: "Application"
    description: "Completed insurance application"
    keywords: ["application", "supplemental", "acord 125", "acord 126"]

  loss_run:
    display: "Loss Run"
    description: "Claims history report from carrier"
    keywords: ["loss run", "loss history", "claims experience", "loss ratio"]

  financial:
    display: "Financial Statement"
    description: "Balance sheet, budget, P&L, or financial report"
    keywords: ["balance sheet", "budget", "financial", "p&l", "income statement"]

  reserve_study:
    display: "Reserve Study"
    description: "Capital reserve study or maintenance plan"
    keywords: ["reserve study", "reserve fund", "capital plan", "component list"]

  coverage_summary:
    display: "Coverage Summary"
    description: "Side-by-side carrier or year-over-year comparison"
    keywords: ["coverage summary", "comparison", "expiring vs", "side by side"]

  bor_letter:
    display: "BOR Letter"
    description: "Broker of record change letter"
    keywords: ["broker of record", "bor", "agent of record"]

  endorsement:
    display: "Endorsement"
    description: "Policy amendment, rider, or endorsement"
    keywords: ["endorsement", "amendment", "rider", "schedule change"]

  governing_docs:
    display: "Governing Documents"
    description: "CC&Rs, bylaws, articles of incorporation, plat maps"
    keywords: ["cc&r", "bylaws", "covenants", "plat", "declaration"]

  correspondence:
    display: "Correspondence"
    description: "Emails, memos, and general communications"
    keywords: ["re:", "fwd:", "dear", "please find"]

  invoice:
    display: "Invoice"
    description: "Premium invoice or billing statement"
    keywords: ["invoice", "premium due", "billing", "payment"]

  sov:
    display: "Schedule of Values"
    description: "Property schedule with building values"
    keywords: ["schedule of values", "sov", "building schedule", "tiv"]

  unknown:
    display: "Unknown"
    description: "Could not classify"
    keywords: []

# ============================================================
# Entity Extraction — domain-specific entities from content
# ============================================================
entity_extraction:
  namespace: "entity"
  prompt_label: "carrier"     # Used in LLM prompt: "carrier name"
  prompt_instruction: "Identify the insurance carrier or underwriting company. Use the carrier's legal name, not the MGA or broker name. Return null if not carrier-specific."
  aliases:                    # Normalize variant names
    "Continental Casualty Company": "cna"
    "Continental Casualty": "cna"
    "CNA": "cna"
    "LIO Insurance Company": "lio"
    "LIO Insurance": "lio"
    "USLI": "usli"
    "United States Liability Insurance": "usli"
    "Philadelphia Indemnity": "phly"
    "PHLY": "phly"
    "Chubb": "chubb"
    "ACE American Insurance": "chubb"

# ============================================================
# Topic Extraction — domain-specific subject areas
# ============================================================
topic_extraction:
  namespace: "topic"
  prompt_label: "coverage lines"
  prompt_instruction: "List the insurance coverage lines discussed in this document."
  values:
    - id: "gl"
      display: "General Liability"
      keywords: ["general liability", "cgl", "bodily injury", "property damage"]
    - id: "property"
      display: "Property"
      keywords: ["property coverage", "building", "blanket", "bpp"]
    - id: "do"
      display: "D&O"
      keywords: ["directors and officers", "d&o", "management liability"]
    - id: "crime"
      display: "Crime"
      keywords: ["crime", "fidelity", "employee theft", "dishonesty"]
    - id: "auto"
      display: "Auto"
      keywords: ["auto", "automobile", "hired and non-owned"]
    - id: "wc"
      display: "Workers Comp"
      keywords: ["workers comp", "workers' compensation", "wc"]
    - id: "umbrella"
      display: "Umbrella/Excess"
      keywords: ["umbrella", "excess liability"]
    - id: "epli"
      display: "EPLI"
      keywords: ["employment practices", "epli", "wrongful termination"]
    - id: "cyber"
      display: "Cyber"
      keywords: ["cyber", "data breach", "network security"]
    - id: "earthquake"
      display: "Earthquake"
      keywords: ["earthquake", "seismic"]

# ============================================================
# LLM Classification Prompt Template
# ============================================================
llm_prompt: |
  Classify this document from an insurance agency file system. Return JSON only.

  Document filename: {filename}
  Source path: {source_path}
  First 2000 characters of content:
  {content_preview}

  Return this exact JSON structure:
  {{
    "document_type": "<one of: {document_type_ids}>",
    "entity": "<{entity_extraction.prompt_label} or null>",
    "topics": [<list of: {topic_ids}>],
    "year_start": "<YYYY or null>",
    "year_end": "<YYYY or null>",
    "confidence": <0.0 to 1.0>
  }}

  Classification rules:
  {document_type_rules}

  Entity rules:
  {entity_extraction.prompt_instruction}

  Topic rules:
  {topic_extraction.prompt_instruction}

# ============================================================
# Email Subject Patterns — for .msg file auto-tagging
# ============================================================
email_patterns:
  - pattern: "(?i)bind|bound"
    tags: [{namespace: "stage", value: "bind"}]
  - pattern: "(?i)quote|proposal|indication"
    tags: [{namespace: "stage", value: "quote"}]
  - pattern: "(?i)submit|submission"
    tags: [{namespace: "stage", value: "submission"}]
  - pattern: "(?i)policy|certificate"
    tags: [{namespace: "stage", value: "policy"}]
  - pattern: "(?i)loss.?run|claims"
    tags: [{namespace: "doctype", value: "loss-run"}]
  - pattern: "(?i)endorse|change"
    tags: [{namespace: "doctype", value: "endorsement"}]
```

### 3.3 YAML Schema Contract

#### Required Fields

Every strategy YAML **must** contain these top-level keys:

| Key | Type | Description |
|-----|------|-------------|
| `strategy.id` | string | Unique identifier, matches filename (without `.yaml`) |
| `strategy.name` | string | Human-readable display name |
| `strategy.version` | string | Semver (e.g., `"1.0"`) — used for strategy pinning |
| `namespaces` | dict | At least one namespace definition |
| `document_types` | dict | At least `unknown` entry |
| `llm_prompt` | string | LLM prompt template with `{document_type_ids}` placeholder |

#### Optional Fields

| Key | Type | Default |
|-----|------|---------|
| `strategy.description` | string | `""` |
| `path_rules` | list | `[]` (no path extraction) |
| `entity_extraction` | object \| null | `null` |
| `topic_extraction` | object \| null | `null` |
| `email_patterns` | list | `[]` |

#### Validation Rules

1. **Unknown fields rejected.** Any top-level key not in the schema causes a validation error on load. Nested unknown fields within `strategy`, `namespaces`, `path_rules`, `document_types`, `entity_extraction`, `topic_extraction`, and `email_patterns` are also rejected.
2. **`strategy.id` must match filename.** `insurance_agency.yaml` must contain `strategy.id: "insurance_agency"`.
3. **Namespace IDs** must be lowercase alphanumeric (`[a-z][a-z0-9_]*`), max 20 characters.
4. **Document type IDs** must be lowercase alphanumeric with underscores, max 30 characters.
5. **`llm_prompt`** must contain `{document_type_ids}` placeholder.
6. **Path rule `level`** must be non-negative integer.

#### Startup Behavior

| Condition | Behavior |
|-----------|----------|
| Active strategy YAML is valid | Load and use |
| Active strategy YAML has validation errors | Log error, fall back to `generic` strategy, set health warning |
| Active strategy YAML file missing | Log error, fall back to `generic` strategy, set health warning |
| `generic.yaml` is missing or invalid | **Fatal error** — application refuses to start |

#### Version Compatibility

- Strategies with the same `strategy.id` and same major version (e.g., `1.0` and `1.3`) are backward-compatible.
- Major version bumps (e.g., `1.x` → `2.0`) may change namespace semantics. Documents tagged with v1 retain their tags; new uploads use v2 rules.
- No automatic migration between strategy versions. Retagging requires explicit admin action via bulk-retag endpoint (future).

### 3.4 Built-In Strategies

Four strategies ship with the application. Each is a YAML file.

#### `generic.yaml` — Universal Fallback

Works with any folder structure. Minimal assumptions.

```yaml
strategy:
  id: "generic"
  name: "Generic"
  description: "Universal strategy — extracts client from top folder, basic doc classification"

namespaces:
  client:
    display: "Client"
    color: "#6366f1"
  year:
    display: "Year"
    color: "#f59e0b"
  doctype:
    display: "Document Type"
    color: "#10b981"

path_rules:
  - namespace: "client"
    level: 0
    pattern: "^(.+)$"
    capture_group: 1
    transform: "slugify"

document_types:
  contract:
    display: "Contract"
    keywords: ["contract", "agreement", "terms and conditions", "msa"]
  report:
    display: "Report"
    keywords: ["report", "summary", "analysis", "findings"]
  financial:
    display: "Financial"
    keywords: ["invoice", "budget", "p&l", "balance sheet", "financial"]
  correspondence:
    display: "Correspondence"
    keywords: ["dear", "re:", "fwd:", "please find", "attached"]
  proposal:
    display: "Proposal"
    keywords: ["proposal", "quote", "estimate", "bid"]
  form:
    display: "Form"
    keywords: ["application", "form", "questionnaire", "survey"]
  legal:
    display: "Legal Document"
    keywords: ["amendment", "addendum", "exhibit", "declaration", "resolution"]
  reference:
    display: "Reference Material"
    keywords: ["manual", "guide", "handbook", "specification", "standard"]
  photo:
    display: "Photo/Image"
    keywords: []
  unknown:
    display: "Unknown"
    keywords: []

entity_extraction: null       # No domain-specific entities
topic_extraction: null        # No domain-specific topics

llm_prompt: |
  Classify this document. Return JSON only.

  Document filename: {filename}
  First 2000 characters of content:
  {content_preview}

  Return this exact JSON structure:
  {{
    "document_type": "<one of: {document_type_ids}>",
    "year_start": "<YYYY or null>",
    "year_end": "<YYYY or null>",
    "confidence": <0.0 to 1.0>
  }}

email_patterns:
  - pattern: "(?i)invoice|payment"
    tags: [{namespace: "doctype", value: "financial"}]
```

#### `law_firm.yaml` — Legal Practice

```yaml
strategy:
  id: "law_firm"
  name: "Law Firm"
  description: "For legal practices managing client matter folders"

namespaces:
  client:
    display: "Client"
    color: "#6366f1"
  year:
    display: "Year"
    color: "#f59e0b"
  doctype:
    display: "Document Type"
    color: "#10b981"
  stage:
    display: "Matter Stage"
    color: "#64748b"
  entity:
    display: "Opposing Party"
    color: "#0ea5e9"
  topic:
    display: "Practice Area"
    color: "#f43f5e"

path_rules:
  - namespace: "client"
    level: 0
    pattern: "^(.+)$"
    capture_group: 1
    transform: "slugify"

  - namespace: "stage"
    level: 1
    mapping:
      "Pleadings": "pleadings"
      "Discovery": "discovery"
      "Depositions": "depositions"
      "Motions": "motions"
      "Correspondence": "correspondence"
      "Billing": "billing"
      "Research": "research"
      "Trial": "trial"
      "Appeals": "appeals"
      "Settlement": "settlement"
      "Close-Out": "close-out"

document_types:
  pleading:
    display: "Pleading"
    keywords: ["complaint", "answer", "motion", "brief", "memorandum"]
  deposition:
    display: "Deposition"
    keywords: ["deposition", "transcript", "testimony"]
  contract:
    display: "Contract"
    keywords: ["agreement", "contract", "settlement", "release"]
  discovery:
    display: "Discovery"
    keywords: ["interrogatory", "request for production", "subpoena"]
  correspondence:
    display: "Correspondence"
    keywords: ["letter", "dear", "re:", "counsel"]
  court_order:
    display: "Court Order"
    keywords: ["order", "ruling", "judgment", "decree"]
  billing:
    display: "Billing"
    keywords: ["invoice", "time entry", "retainer", "trust account"]
  research:
    display: "Research Memo"
    keywords: ["research", "memorandum", "analysis", "case law"]
  evidence:
    display: "Evidence"
    keywords: ["exhibit", "evidence", "photograph", "record"]
  unknown:
    display: "Unknown"
    keywords: []

entity_extraction:
  namespace: "entity"
  prompt_label: "opposing party"
  prompt_instruction: "Identify the opposing party or adverse entity in this legal document. Return null if not identifiable."
  aliases: {}

topic_extraction:
  namespace: "topic"
  prompt_label: "practice areas"
  prompt_instruction: "List the areas of law relevant to this document."
  values:
    - id: "litigation"
      display: "Litigation"
      keywords: ["litigation", "lawsuit", "court", "trial"]
    - id: "corporate"
      display: "Corporate"
      keywords: ["corporate", "formation", "merger", "acquisition"]
    - id: "real-estate"
      display: "Real Estate"
      keywords: ["property", "lease", "deed", "easement", "zoning"]
    - id: "employment"
      display: "Employment"
      keywords: ["employment", "termination", "discrimination", "wage"]
    - id: "ip"
      display: "Intellectual Property"
      keywords: ["patent", "trademark", "copyright", "trade secret"]
    - id: "tax"
      display: "Tax"
      keywords: ["tax", "irs", "assessment", "deduction"]
    - id: "estate"
      display: "Estate Planning"
      keywords: ["trust", "will", "estate", "probate", "beneficiary"]

llm_prompt: |
  Classify this legal document. Return JSON only.

  Document filename: {filename}
  First 2000 characters of content:
  {content_preview}

  Return this exact JSON structure:
  {{
    "document_type": "<one of: {document_type_ids}>",
    "entity": "<opposing party name or null>",
    "topics": [<list of: {topic_ids}>],
    "year_start": "<YYYY or null>",
    "year_end": "<YYYY or null>",
    "confidence": <0.0 to 1.0>
  }}
```

#### `construction.yaml` — Construction/Project Management

```yaml
strategy:
  id: "construction"
  name: "Construction"
  description: "For construction companies managing project documentation"

namespaces:
  client:
    display: "Project"
    color: "#6366f1"
  year:
    display: "Year"
    color: "#f59e0b"
  doctype:
    display: "Document Type"
    color: "#10b981"
  stage:
    display: "Project Phase"
    color: "#64748b"
  entity:
    display: "Subcontractor/Vendor"
    color: "#0ea5e9"
  topic:
    display: "Trade/Discipline"
    color: "#f43f5e"

path_rules:
  - namespace: "client"
    level: 0
    pattern: "^(.+?)(?:\\s*Phase.*)?$"
    capture_group: 1
    transform: "slugify"

  - namespace: "stage"
    level: 1
    mapping:
      "Bids": "bidding"
      "Contracts": "contract"
      "Submittals": "submittal"
      "RFIs": "rfi"
      "Change Orders": "change-order"
      "Pay Applications": "pay-app"
      "Inspections": "inspection"
      "Punch List": "punch-list"
      "Close-Out": "close-out"
      "Safety": "safety"
      "Photos": "photos"
      "Drawings": "drawings"

document_types:
  bid:
    display: "Bid/Proposal"
    keywords: ["bid", "proposal", "estimate", "scope of work"]
  contract:
    display: "Contract"
    keywords: ["contract", "agreement", "aia", "subcontract"]
  change_order:
    display: "Change Order"
    keywords: ["change order", "co", "modification", "amendment"]
  submittal:
    display: "Submittal"
    keywords: ["submittal", "shop drawing", "product data", "sample"]
  rfi:
    display: "RFI"
    keywords: ["request for information", "rfi", "clarification"]
  pay_app:
    display: "Pay Application"
    keywords: ["pay application", "aia g702", "g703", "schedule of values"]
  inspection:
    display: "Inspection Report"
    keywords: ["inspection", "report", "punch list", "deficiency"]
  drawing:
    display: "Drawing/Plan"
    keywords: ["drawing", "plan", "elevation", "detail", "blueprint"]
  permit:
    display: "Permit"
    keywords: ["permit", "building permit", "variance", "approval"]
  safety:
    display: "Safety Document"
    keywords: ["safety", "osha", "incident", "toolbox talk"]
  correspondence:
    display: "Correspondence"
    keywords: ["letter", "email", "memo", "notice"]
  unknown:
    display: "Unknown"
    keywords: []

entity_extraction:
  namespace: "entity"
  prompt_label: "subcontractor or vendor"
  prompt_instruction: "Identify the subcontractor, vendor, or supplier named in this document. Return null if not identifiable."
  aliases: {}

topic_extraction:
  namespace: "topic"
  prompt_label: "construction trades"
  prompt_instruction: "List the construction trades or disciplines relevant to this document."
  values:
    - id: "electrical"
      display: "Electrical"
      keywords: ["electrical", "wiring", "conduit", "panel"]
    - id: "plumbing"
      display: "Plumbing"
      keywords: ["plumbing", "piping", "fixture", "drain"]
    - id: "hvac"
      display: "HVAC"
      keywords: ["hvac", "mechanical", "ductwork", "air handling"]
    - id: "structural"
      display: "Structural"
      keywords: ["structural", "foundation", "framing", "steel"]
    - id: "concrete"
      display: "Concrete"
      keywords: ["concrete", "formwork", "rebar", "pour"]
    - id: "roofing"
      display: "Roofing"
      keywords: ["roofing", "membrane", "flashing", "gutter"]
    - id: "sitework"
      display: "Sitework"
      keywords: ["grading", "excavation", "paving", "landscaping"]
    - id: "fire-protection"
      display: "Fire Protection"
      keywords: ["fire", "sprinkler", "alarm", "suppression"]

llm_prompt: |
  Classify this construction project document. Return JSON only.

  Document filename: {filename}
  First 2000 characters of content:
  {content_preview}

  Return this exact JSON structure:
  {{
    "document_type": "<one of: {document_type_ids}>",
    "entity": "<subcontractor or vendor name or null>",
    "topics": [<list of: {topic_ids}>],
    "year_start": "<YYYY or null>",
    "year_end": "<YYYY or null>",
    "confidence": <0.0 to 1.0>
  }}
```

---

## 4. Feature Configuration

### 4.1 Settings (config.py)

```python
# Auto-tagging
auto_tagging_enabled: bool = False                    # Master switch
auto_tagging_strategy: str = "generic"                # Strategy ID (YAML filename)
auto_tagging_path_enabled: bool = True                # Path-based (no cost)
auto_tagging_llm_enabled: bool = True                 # LLM-based (1 call per doc)
auto_tagging_llm_model: str = "qwen3:8b"             # Model for classification
auto_tagging_require_approval: bool = False           # If True, ALL LLM tags are suggestions
auto_tagging_create_missing_tags: bool = True         # Auto-create tags that don't exist
auto_tagging_confidence_threshold: float = 0.7        # Auto-apply above this (when !require_approval)
auto_tagging_suggestion_threshold: float = 0.4        # Suggest above this (when !require_approval)
auto_tagging_strategies_dir: str = "./data/auto_tag_strategies"  # Strategy YAML directory

# Operational guardrails
auto_tagging_max_tags_per_doc: int = 20               # Hard limit on auto-tags per document
auto_tagging_max_tag_name_length: int = 100           # Max chars for tag_name (namespace:value)
auto_tagging_max_client_tags: int = 500               # Max unique client: tags before warning
auto_tagging_llm_timeout_seconds: int = 30            # LLM call timeout
auto_tagging_llm_max_retries: int = 1                 # LLM retry count (exponential backoff)
```

### 4.2 Environment Variables

```env
AUTO_TAGGING_ENABLED=true
AUTO_TAGGING_STRATEGY=insurance_agency
AUTO_TAGGING_PATH_ENABLED=true
AUTO_TAGGING_LLM_ENABLED=true
AUTO_TAGGING_LLM_MODEL=qwen3:8b
AUTO_TAGGING_REQUIRE_APPROVAL=false
AUTO_TAGGING_CONFIDENCE_THRESHOLD=0.7
AUTO_TAGGING_MAX_TAGS_PER_DOC=20
AUTO_TAGGING_LLM_TIMEOUT_SECONDS=30
```

### 4.3 Health Check

```json
{
  "auto_tagging": {
    "enabled": true,
    "strategy": "insurance_agency",
    "strategy_name": "Insurance Agency",
    "strategy_version": "1.0",
    "strategy_status": "loaded",           // "loaded" | "fallback_generic" | "error"
    "path_enabled": true,
    "llm_enabled": true,
    "llm_model": "qwen3:8b",
    "require_approval": false,
    "namespaces": ["client", "year", "doctype", "stage", "entity", "topic"],
    "document_types": 14,
    "path_rules": 4,
    "guardrails": {
      "max_tags_per_doc": 20,
      "max_client_tags": 500,
      "current_client_tags": 12
    }
  }
}
```
```

### 4.4 Admin Strategy Management Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/api/admin/auto-tagging/strategies` | Authenticated (any role) | List available strategies |
| `GET` | `/api/admin/auto-tagging/strategies/{id}` | Authenticated (any role) | Get strategy details |
| `PUT` | `/api/admin/auto-tagging/strategies/{id}` | **Admin only** | Update custom strategy |
| `POST` | `/api/admin/auto-tagging/strategies` | **Admin only** | Create custom strategy |
| `DELETE` | `/api/admin/auto-tagging/strategies/{id}` | **Admin only** | Delete custom strategy (not built-in) |
| `GET` | `/api/admin/auto-tagging/active` | Authenticated (any role) | Get active strategy config |
| `PUT` | `/api/admin/auto-tagging/active` | **Admin only** | Switch active strategy |

**Audit requirements:** All mutating operations (`POST`, `PUT`, `DELETE` on strategies; `PUT` on active) **must** write an audit log entry:

```python
class StrategyAuditEntry(BaseModel):
    timestamp: datetime
    actor_id: str               # User ID performing the action
    actor_email: str
    action: Literal["create", "update", "delete", "switch"]
    strategy_id: str
    strategy_version: str | None
    before: dict | None         # Previous state (for update/switch)
    after: dict | None          # New state (for create/update/switch)
```

Audit entries are written to the `audit_log` table and included in admin activity feeds. Denied-access attempts (non-admin calling mutate endpoints) return `403 Forbidden` and log the attempt.

---

## 5. Strategy Engine (Core)

### 5.1 Strategy Loader

```python
class AutoTagStrategy:
    """Loaded from YAML, drives both path-based and LLM-based tagging."""

    id: str
    name: str
    description: str
    namespaces: dict[str, NamespaceConfig]
    path_rules: list[PathRule]
    document_types: dict[str, DocumentTypeConfig]
    entity_extraction: EntityExtractionConfig | None
    topic_extraction: TopicExtractionConfig | None
    llm_prompt_template: str
    email_patterns: list[EmailPattern]

    @classmethod
    def load(cls, path: str) -> "AutoTagStrategy":
        """Load strategy from YAML file."""

    def parse_path(self, source_path: str) -> list[AutoTag]:
        """Apply path rules to extract tags from folder structure."""

    def build_llm_prompt(self, filename: str, source_path: str, content_preview: str) -> str:
        """Render the LLM prompt with strategy-specific types, entities, topics."""

    def parse_llm_response(self, response: dict) -> list[AutoTag]:
        """Convert LLM JSON response into AutoTag list, applying aliases."""

    def parse_email_subject(self, subject: str) -> list[AutoTag]:
        """Apply email subject patterns to extract tags."""
```

### 5.2 Path Rule Processing

```python
@dataclass
class PathRule:
    namespace: str
    level: int                        # Folder depth (0 = root)
    pattern: str | None = None        # Regex to match folder name
    capture_group: int = 1
    transform: str | None = None      # "slugify" | "year_range" | None
    mapping: dict[str, str] | None = None  # Static name → value mapping
    parent_match: str | None = None   # Only apply if parent folder matches

def apply_path_rules(rules: list[PathRule], source_path: str) -> list[AutoTag]:
    """
    Split source_path into path components, apply each rule by level.

    Example:
      path = "Bethany Terrace (12-13)/24 NB/Quote/CNA/file.pdf"
      components = ["Bethany Terrace (12-13)", "24 NB", "Quote", "CNA"]

      Rule level=0, pattern="^(.+?)(?:\\s*\\(.*\\))?$", transform=slugify
        → match "Bethany Terrace" → client:bethany-terrace

      Rule level=1, pattern="^(\\d{2})\\s+(?:NB|Renewal)$", transform=year_range
        → match "24" → year:2024-2025

      Rule level=2, mapping={"Quote": "quote"}
        → match "Quote" → stage:quote

      Rule level=3, parent_match="^(Quote|Sub)$", transform=slugify
        → parent is "Quote" (matches) → entity:cna
    """
```

### 5.3 Transform Functions

| Transform | Input | Output |
|-----------|-------|--------|
| `slugify` | `"Bethany Terrace"` | `"bethany-terrace"` |
| `year_range` | `"24"` | `"2024-2025"` |
| `lowercase` | `"CNA"` | `"cna"` |
| `none` / null | `"Quote"` | `"Quote"` (identity) |

### 5.4 LLM Prompt Rendering

The `llm_prompt` template in the YAML uses placeholders that are filled at runtime:

| Placeholder | Filled With |
|-------------|-------------|
| `{filename}` | `document.original_filename` |
| `{source_path}` | Upload `source_path` (if provided) |
| `{content_preview}` | First 2000 chars of extracted text |
| `{document_type_ids}` | Comma-separated list from `document_types` keys |
| `{document_type_rules}` | Auto-generated rules from `document_types[*].description` |
| `{topic_ids}` | Comma-separated list from `topic_extraction.values[*].id` |
| `{entity_extraction.prompt_label}` | e.g., `"carrier"`, `"opposing party"` |
| `{entity_extraction.prompt_instruction}` | Strategy-specific entity instructions |
| `{topic_extraction.prompt_instruction}` | Strategy-specific topic instructions |

### 5.5 Entity Alias Resolution

```python
def resolve_entity(raw_name: str, aliases: dict[str, str]) -> str:
    """
    Normalize entity names using alias table.

    resolve_entity("Continental Casualty Company", aliases) → "cna"
    resolve_entity("Some New Carrier", aliases) → "some-new-carrier"  (slugified)
    """
    # Check exact match in aliases
    if raw_name in aliases:
        return aliases[raw_name]
    # Check case-insensitive match
    for alias, slug in aliases.items():
        if raw_name.lower() == alias.lower():
            return slug
    # Fallback: slugify the raw name
    return slugify(raw_name)
```

---

## 6. AutoTag Model

```python
class AutoTag(BaseModel):
    namespace: str              # "client", "year", "doctype", "stage", "entity", "topic"
    value: str                  # "bethany-terrace", "2024-2025", "policy", "cna"
    source: Literal["path", "llm", "email", "manual"]
    confidence: float = 1.0     # Path/email = 1.0, LLM = model confidence
    strategy_id: str = ""       # Which strategy produced this tag
    strategy_version: str = ""  # Strategy version at time of tagging

    @property
    def tag_name(self) -> str:
        """Full namespaced tag name for DB storage."""
        return f"{self.namespace}:{self.value}"

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        return self.value.replace("-", " ").title()
```

---

## 7. Upload API Changes

### 7.1 Single Upload

```python
@router.post("/upload")
async def upload_document(
    file: UploadFile,
    tag_ids: Annotated[list[str], Form(default=[])],   # Optional when auto-tagging enabled
    source_path: str | None = Form(None),               # NEW: Original folder path
    auto_tag: bool | None = Form(None),                 # NEW: Override auto-tagging for this upload
    ...
)
```

**Validation:** If `auto_tagging_enabled=True` AND (`source_path` provided OR `auto_tag=True`), then `tag_ids` may be empty. Otherwise, at least one `tag_id` is required.

### 7.2 Batch Upload

```python
@router.post("/upload/batch")
async def upload_batch(
    files: list[UploadFile],
    source_paths: Annotated[list[str], Form()],        # One path per file
    tag_ids: Annotated[list[str], Form(default=[])],   # Manual tags for ALL files
    ...
) -> BatchUploadResponse
```

**Idempotency:** Each file is identified by `SHA256(content_hash + source_path + strategy_id)`. Re-uploading the same file with the same path and strategy is a no-op (returns existing document with `status: "duplicate"`). Changing the strategy or path for the same content creates a new document.

### 7.3 Batch Upload Response

```python
class BatchUploadResponse(BaseModel):
    total: int
    uploaded: int
    duplicates: int
    failed: int
    auto_tags_applied: int        # Count of auto-tags applied
    results: list[BatchFileResult]

class BatchFileResult(BaseModel):
    filename: str
    source_path: str | None
    status: Literal["uploaded", "duplicate", "failed"]
    error_code: str | None        # Only set when status="failed"
    error_message: str | None
    document_id: str | None       # Set when status="uploaded" or "duplicate"
    auto_tags: list[str]          # Tags applied to this file
```

**Error codes for batch files:**

| Code | Description |
|------|-------------|
| `UPLOADED` | Successfully uploaded and queued for processing |
| `DUPLICATE` | Content hash + source_path already exists |
| `FAILED_SIZE` | File exceeds max upload size |
| `FAILED_TYPE` | Unsupported file type |
| `FAILED_PARSE` | File could not be read (corrupted, encrypted) |
| `FAILED_TAG` | Tag creation/assignment failed |
| `FAILED_STORAGE` | Storage write failed |

### 7.4 CLI Resume Manifest

The bulk upload CLI writes a manifest file (`{upload_dir}/.ingest_manifest.json`) after each file:

```json
{
  "strategy_id": "insurance_agency",
  "strategy_version": "1.0",
  "started_at": "2026-02-16T10:00:00Z",
  "base_path": "/path/to/Customer_Projects/",
  "files": {
    "Bethany Terrace/24 NB/Quote/CNA/DO_Quote.pdf": {
      "status": "uploaded",
      "document_id": "abc123",
      "content_hash": "sha256:...",
      "uploaded_at": "2026-02-16T10:00:05Z"
    },
    "Bethany Terrace/24 NB/Quote/CNA/GL_Quote.pdf": {
      "status": "failed",
      "error_code": "FAILED_SIZE",
      "error_message": "File exceeds 50MB limit"
    }
  }
}
```

On resume (`--resume`), the CLI skips files with `status: "uploaded"` or `"duplicate"` and retries `"failed"` entries.

---

## 8. Processing Flow

### 8.1 Upload Stage (Synchronous)

```
1. Receive file + source_path + optional manual tag_ids
2. Load active AutoTagStrategy from YAML
3. Record strategy pinning: strategy_id + strategy_version
4. If path_enabled AND source_path provided:
   a. strategy.parse_path(source_path) → path_tags[]
   b. If .msg file: strategy.parse_email_subject() → more tags
   c. Enforce guardrails: truncate to max_tags_per_doc, validate tag name length
   d. ensure_tag_exists() for each auto-tag
   e. Add to document.tags alongside manual tags
5. Save file, create Document(status="pending", auto_tag_strategy=id, auto_tag_version=version)
6. db.commit() → enqueue processing
```

### 8.2 Processing Stage (Async Background Task)

```
1. Chunk document (Docling/Simple chunker)
2. If llm_enabled:
   a. content_preview = first 2000 chars of chunks
   b. Load strategy using pinned strategy_id + strategy_version
      - If pinned version unavailable, log warning, use current version
   c. prompt = strategy.build_llm_prompt(filename, path, text)
   d. Call LLM with timeout (auto_tagging_llm_timeout_seconds)
      - On timeout: retry up to auto_tagging_llm_max_retries with exponential backoff
      - On final failure: set auto_tag_status="partial", skip to step 3
   e. Parse LLM JSON response:
      - Attempt strict JSON parse
      - On failure: strip markdown fences (```json ... ```), extract first {...} block
      - On second failure: log error, set auto_tag_status="partial", skip to step 3
   f. llm_tags = strategy.parse_llm_response(response)
   g. Apply confidence/approval decision table (see Section 8.3)
   h. Apply namespace-specific conflict resolution (see Section 8.4)
   i. Enforce guardrails: total tags (manual + path + llm) ≤ max_tags_per_doc
3. Collect all final tag_names (manual + path + llm)
4. vector_service.add_document(tags=tag_names)
5. Write auto_tag_source provenance JSON (see Section 8.5)
6. Update document.auto_tag_status = "completed" | "partial" | "failed"
7. db.commit()
```

### 8.3 Confidence Decision Table

| Confidence Band | `require_approval=false` | `require_approval=true` |
|-----------------|--------------------------|-------------------------|
| `≥ threshold` (default 0.7) | **Auto-apply** | **Suggest** (admin reviews all LLM tags) |
| `≥ suggestion_threshold` and `< threshold` (0.4–0.7) | **Auto-apply** | **Suggest** |
| `< suggestion_threshold` (< 0.4) | **Discard** (not applied, not suggested) | **Discard** |

**Key rules:**
- Path-based tags always auto-apply (confidence = 1.0, deterministic).
- When `require_approval=true`, **all** LLM-derived tags become suggestions regardless of confidence. Admin must approve before they are applied to the document.
- When `require_approval=false`, the threshold bands determine auto-apply vs discard. There is no "suggest" state — tags are either applied or dropped.
- Discarded tags are still recorded in `auto_tag_source` provenance for audit.

### 8.4 Namespace-Specific Conflict Resolution

When both path-based and LLM-based tagging produce a tag in the same namespace for the same document:

| Namespace | Authoritative Source | Rationale |
|-----------|---------------------|-----------|
| `client:` | **Path** | Client name is definitively encoded in folder structure |
| `year:` | **Path** | Year/period is definitively encoded in folder structure |
| `stage:` | **Path** | Workflow stage is definitively encoded in folder structure |
| `doctype:` | **LLM** (if confidence ≥ threshold) | Content analysis is more accurate than path for document type |
| `entity:` | **LLM** (if confidence ≥ threshold), else **Path** | LLM can read the actual carrier/party name from content |
| `topic:` | **LLM** (if confidence ≥ threshold) | Topics are content-derived, not path-derived |

**Conflict resolution algorithm:**
1. Start with path_tags as baseline.
2. For each llm_tag, check if a path_tag exists in the same namespace.
3. If conflict: use the authoritative source per the table above. Record both candidates in provenance.
4. If no conflict: add the llm_tag.

### 8.5 Provenance Schema (`auto_tag_source`)

The `auto_tag_source` column stores a JSON document recording all tagging decisions:

```json
{
  "strategy_id": "insurance_agency",
  "strategy_version": "1.0",
  "path_candidates": [
    {
      "namespace": "client",
      "value": "bethany-terrace",
      "confidence": 1.0,
      "rule_level": 0,
      "raw_input": "Bethany Terrace (12-13)"
    }
  ],
  "llm_candidates": [
    {
      "namespace": "doctype",
      "value": "quote",
      "confidence": 0.92,
      "raw_response": "{\"document_type\": \"quote\", ...}"
    }
  ],
  "conflicts": [
    {
      "namespace": "entity",
      "path_value": "cna",
      "llm_value": "continental-casualty",
      "winner": "path",
      "reason": "entity: LLM confidence 0.65 below threshold 0.7, path wins"
    }
  ],
  "applied": ["client:bethany-terrace", "year:2024-2025", "stage:quote", "entity:cna", "doctype:quote"],
  "discarded": ["topic:umbrella"],
  "suggested": []
}
```

### 8.6 LLM Failure Handling

| Failure | Behavior | Status |
|---------|----------|--------|
| LLM timeout (after retries) | Skip LLM tags, keep path tags | `auto_tag_status = "partial"` |
| LLM returns invalid JSON (after repair attempt) | Skip LLM tags, keep path tags | `auto_tag_status = "partial"` |
| LLM returns valid JSON with unknown document_type | Map to `unknown`, log warning | `auto_tag_status = "completed"` |
| Strategy YAML not loadable | Fall back to `generic` strategy | `auto_tag_status = "completed"` (with generic) |
| Path parsing error (regex failure) | Skip that rule, continue others | `auto_tag_status = "completed"` |
| Tag count exceeds `max_tags_per_doc` | Truncate: keep path tags first, then LLM by confidence desc | `auto_tag_status = "completed"` |

**JSON repair policy:** On initial parse failure, attempt these repairs in order:
1. Strip leading/trailing markdown fences (` ```json `, ` ``` `)
2. Extract first `{...}` block from response
3. If still invalid, give up and log the raw response in provenance

---

## 9. Tag Lifecycle

### 9.1 Auto-Creation

When `auto_tagging_create_missing_tags=True`:

```python
async def ensure_tag_exists(
    db: Session,
    tag_name: str,
    display_name: str,
    namespace: str,
    strategy: AutoTagStrategy,
) -> Tag:
    existing = db.query(Tag).filter(Tag.name == tag_name).first()
    if existing:
        return existing

    ns_config = strategy.namespaces.get(namespace)
    color = ns_config.color if ns_config else "#6B7280"

    tag = Tag(
        name=tag_name,
        display_name=display_name,
        description=f"Auto-created by {strategy.name} strategy",
        color=color,
        is_system=False,
    )
    db.add(tag)
    db.flush()
    return tag
```

### 9.2 Tag Suggestions (Approval Mode)

When `auto_tagging_require_approval=True`, low-confidence tags are stored as suggestions:

```python
class TagSuggestion(Base):
    __tablename__ = "tag_suggestions"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"))
    tag_name = Column(String, nullable=False)           # e.g., "doctype:policy"
    display_name = Column(String, nullable=False)
    namespace = Column(String, nullable=False)
    source = Column(String, nullable=False)              # "path" | "llm" | "email"
    confidence = Column(Float, default=1.0)
    strategy_id = Column(String, nullable=False)
    status = Column(String, default="pending")           # pending | approved | rejected
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

**Approval endpoints:**

```python
GET    /api/documents/{id}/tag-suggestions
POST   /api/documents/{id}/tag-suggestions/approve    # Body: {"suggestion_ids": [...]}
POST   /api/documents/{id}/tag-suggestions/reject     # Body: {"suggestion_ids": [...]}
POST   /api/documents/tag-suggestions/approve-batch   # Body: {"suggestion_ids": [...]}
```

### 9.3 Document Model Extension

```python
# In Document model
auto_tag_status = Column(String, nullable=True)    # null | "pending" | "completed" | "partial" | "failed"
auto_tag_strategy = Column(String, nullable=True)  # Strategy ID used (pinned at upload)
auto_tag_version = Column(String, nullable=True)    # Strategy version used (pinned at upload)
auto_tag_source = Column(Text, nullable=True)       # JSON provenance (see Section 8.5 schema)
```

**Status transitions:**
- `null` → Document uploaded without auto-tagging
- `"pending"` → Path tags applied, awaiting LLM processing
- `"completed"` → Both path and LLM stages finished successfully
- `"partial"` → Path tags applied, LLM stage failed (timeout, parse error)
- `"failed"` → Both stages failed (strategy not loadable)

---

## 10. Access Control

### 10.1 Per-User Tag Access Toggle

The User model gains a configurable access mode:

```python
# In User model
tag_access_enabled = Column(Boolean, default=True, nullable=False)
```

| `tag_access_enabled` | Behavior |
|----------------------|----------|
| `True` (default) | User can only search/chat documents whose tags overlap with user's assigned tags |
| `False` | User can search/chat **all** documents regardless of tags |

**Admin UI:** The Users management view displays a toggle switch per user: **"Enable tag-based access"**. Admins can enable/disable per user.

**Admin users** always see all documents regardless of this flag.

### 10.2 Access Policy Enforcement

**Rule:** Auto-created tags are **metadata-only** until explicitly assigned to a user by an admin. Creating a tag via auto-tagging does NOT grant any user access to documents with that tag.

**Enforcement points:**

1. **Document list API** (`GET /api/documents`): If `user.tag_access_enabled`, filter by `user.assigned_tags ∩ document.tags`. Namespaced auto-tags (e.g., `client:bethany-terrace`) are treated identically to manual tags for filtering — they must be in the user's assigned set.

2. **Vector retrieval** (`_build_access_filter()`): Qdrant filter uses only the user's **assigned** tags. Auto-tags on documents that are not assigned to the user are invisible to retrieval.

3. **Chat API**: Same filter as vector retrieval. A user without `client:bethany-terrace` assigned cannot retrieve or chat about Bethany Terrace documents, even though the tag exists in the system.

**Invariant:** `auto-created tag ∉ user.assigned_tags → user cannot access documents with only that tag`

### 10.3 Workflow

1. Admin uploads customer folder → `client:bethany-terrace` tag auto-created on documents
2. Tag exists in system but is **not assigned to any user** — no access granted
3. Admin navigates to Users view → selects account manager → assigns `client:bethany-terrace`
4. Account manager can now see and chat about Bethany Terrace documents
5. Admin can also set `tag_access_enabled=False` for a user who needs unrestricted access

### 10.4 Bulk Tag Assignment Helper

```python
POST /api/users/{user_id}/tags/auto
Body: {
    "client_names": ["bethany-terrace", "cedar-ridge"],
    "include_doctypes": true,
    "include_entities": false
}
```

### 10.5 Required Integration Tests

| Test | Assertion |
|------|-----------|
| Upload with auto-tagging creates `client:x` tag | Tag exists in DB, not assigned to any non-admin user |
| User without `client:x` assigned queries documents | Zero results for `client:x` documents |
| Admin assigns `client:x` to user | User now sees `client:x` documents |
| User with `tag_access_enabled=False` | User sees all documents regardless of tags |
| Admin user | Always sees all documents regardless of assigned tags |
| Remove `client:x` from user | User no longer sees `client:x` documents |

---

## 11. Operational Guardrails

### 11.1 Per-Document Limits

| Guardrail | Default | Configurable |
|-----------|---------|--------------|
| Max auto-tags per document | 20 | `AUTO_TAGGING_MAX_TAGS_PER_DOC` |
| Max tag name length | 100 chars | `AUTO_TAGGING_MAX_TAG_NAME_LENGTH` |
| Tag value normalization | slugify (lowercase, hyphens, strip special chars) | Per path-rule transform |

When the tag count exceeds `max_tags_per_doc`, tags are prioritized: manual tags first, then path-based (by rule order), then LLM-based (by confidence descending). Excess tags are recorded in provenance as `"truncated"`.

### 11.2 Namespace Cardinality

| Guardrail | Default | Behavior |
|-----------|---------|----------|
| Max unique `client:` tags | 500 | Log warning at 80% (400), reject new at 100% |
| Max unique tags per namespace (other) | 1000 | Log warning at 80%, reject new at 100% |

When a namespace hits its cardinality cap, new tag creation is blocked and the file is still uploaded with existing tags only. The health endpoint reports `current_client_tags` count.

### 11.3 Tag Name Normalization

All auto-generated tag values are normalized:
1. Lowercase
2. Replace spaces and underscores with hyphens
3. Strip non-alphanumeric characters (except hyphens)
4. Collapse multiple hyphens to single
5. Trim leading/trailing hyphens
6. Truncate to fit within `max_tag_name_length` (including `namespace:` prefix)

Example: `"Continental Casualty Company (IL)"` → `"continental-casualty-company-il"`

### 11.4 Deduplication

Tags are deduplicated by `tag_name` (namespace:value). If path-based and LLM-based tagging produce the same tag_name, the conflict resolution rules in Section 8.4 apply. Duplicate manual tags are ignored.

---

## 12. Search, Filtering, and Facets

### 11.1 Namespace Filtering

```python
GET /api/documents?tag_namespace=client&tag_value=bethany-terrace
GET /api/documents?tag_prefix=client:
GET /api/documents?tag_prefix=doctype:policy
```

### 11.2 Tag Facets

```python
GET /api/tags/facets
Response: {
    "client": [
        {"name": "client:bethany-terrace", "display": "Bethany Terrace", "count": 112},
        {"name": "client:cedar-ridge", "display": "Cedar Ridge", "count": 87}
    ],
    "doctype": [
        {"name": "doctype:policy", "display": "Policy", "count": 45},
        {"name": "doctype:quote", "display": "Quote", "count": 38}
    ],
    ...
}
```

### 11.3 Chat Context

The existing tag-based Qdrant filtering works unchanged — auto-tags flow into the `tags` payload field and are filtered during vector retrieval.

```
User: "What is the D&O premium for Bethany Terrace?"
→ Filter: tags includes "client:bethany-terrace" (from user's assigned tags)
→ Retrieves coverage summaries and quotes
→ Answers with citations
```

---

## 13. Bulk Upload CLI

```bash
# Upload one customer folder
python -m ai_ready_rag.cli.bulk_upload \
    --path "/path/to/Bethany Terrace (12-13)/" \
    --strategy insurance_agency \
    --manual-tags hr \
    --api-url http://localhost:8502

# Upload all customers in a parent directory
python -m ai_ready_rag.cli.bulk_upload \
    --path "/path/to/Customer_Projects/" \
    --strategy insurance_agency \
    --recursive \
    --dry-run                    # Preview tags without uploading

# Upload with generic strategy
python -m ai_ready_rag.cli.bulk_upload \
    --path "/path/to/Mixed_Documents/" \
    --strategy generic \
    --skip-images
```

**Features:**
- Walks directory tree, preserves relative paths as `source_path`
- Applies strategy path rules per file
- Skips duplicates (content hash)
- Dry-run mode: prints tags without uploading
- Resume support via manifest file
- File type filters: `--skip-images`, `--skip-emails`, `--only-pdf`, `--only-docx`
- Progress bar with ETA
- Concurrent uploads with configurable parallelism

---

## 14. .msg Email Support (v1 Scope)

Strategies can process `.msg` files using `extract-msg` (optional dependency).

### 13.1 v1 Capabilities

1. Extract metadata: `from`, `to`, `cc`, `subject`, `date`
2. Extract body text (HTML → plain text)
3. Apply strategy's `email_patterns` to subject line
4. Auto-tag with `doctype:correspondence` (unless a more specific match)
5. Include email metadata in chunk metadata for citation

### 13.2 v1 Constraints

| Constraint | Limit |
|------------|-------|
| Max .msg file size | 10 MB |
| Attachment extraction | **Not supported in v1** — attachments are ignored |
| Malformed .msg handling | Log warning, skip file, tag `doctype:correspondence` from filename only |
| `extract-msg` dependency | Optional — if not installed, .msg files are skipped with logged warning |

### 13.3 Content Sanitization

- HTML body is converted to plain text (strip all tags)
- Embedded images and OLE objects are stripped
- Only plain text content is passed to LLM classification
- No executable content is extracted or stored

### 13.4 Future (v2)

- Attachment extraction as child documents
- Attachment size/count limits (configurable)
- Thread detection and conversation grouping

---

## 15. Implementation Plan

### Phase 1: Strategy Engine + Path-Based Tagging
- [ ] Define `AutoTagStrategy` loader with strict YAML schema validation (Section 3.3)
- [ ] Implement `PathRule` processing with transforms
- [ ] Create `generic.yaml` and `insurance_agency.yaml` built-in strategies
- [ ] Add `auto_tagging_*` settings to `config.py` (including guardrails)
- [ ] Add `source_path` and `auto_tag` fields to upload endpoint
- [ ] Modify `document_service.upload()` to apply path-based tags with strategy pinning
- [ ] Add `ensure_tag_exists()` helper with tag name length validation
- [ ] Relax `tag_ids` requirement when auto-tagging enabled
- [ ] Add auto-tagging status to health endpoint (including strategy_status, guardrails)
- [ ] Unit tests: path parser, strategy loader, transform functions, YAML validation, guardrail enforcement

### Phase 2: Access Control + User Model
- [ ] Add `tag_access_enabled` boolean to User model
- [ ] Update `_build_access_filter()` to check `tag_access_enabled` flag
- [ ] Update document list API to respect the flag
- [ ] Frontend: add "Enable tag-based access" toggle to Users admin view
- [ ] Integration tests: all 6 access control scenarios (Section 10.5)

### Phase 3: LLM-Based Tagging
- [ ] Implement `DocumentClassifier` service using strategy prompts
- [ ] Integrate into `processing_service.py` post-chunking
- [ ] Add namespace-specific conflict resolution (Section 8.4)
- [ ] Add confidence decision table logic (Section 8.3)
- [ ] Add LLM failure handling: timeout, retry, JSON repair (Section 8.6)
- [ ] Write provenance JSON to `auto_tag_source` (Section 8.5 schema)
- [ ] Add `auto_tag_status`, `auto_tag_strategy`, `auto_tag_version` to Document model
- [ ] Create `law_firm.yaml` and `construction.yaml` strategies
- [ ] Unit tests: LLM classifier (mocked), prompt rendering, response parsing, conflict resolution, failure handling

### Phase 4: Tag Suggestions & Approval
- [ ] Create `TagSuggestion` DB model
- [ ] Add suggestion CRUD + bulk approval endpoints
- [ ] Implement decision table for `require_approval=true` mode
- [ ] Frontend: suggestion badges on document cards
- [ ] Frontend: approval dialog with confidence display

### Phase 5: Batch Upload
- [ ] Create `/upload/batch` endpoint with per-file error codes (Section 7.3)
- [ ] Implement idempotency via content_hash + source_path + strategy_id
- [ ] Create `ai_ready_rag.cli.bulk_upload` CLI tool
- [ ] Resume manifest support (Section 7.4)
- [ ] Dry-run mode, progress reporting

### Phase 6: Admin Strategy Management
- [ ] Strategy list/detail/switch endpoints with route-level auth (Section 4.4)
- [ ] Audit logging for all mutating operations
- [ ] Custom strategy creation endpoint with schema validation
- [ ] Frontend: strategy selector in settings
- [ ] Frontend: strategy preview (show namespaces, rules, doc types)

### Phase 7: Search & Filtering
- [ ] Namespace filtering on document list API
- [ ] `/tags/facets` endpoint
- [ ] Frontend: namespace filter chips
- [ ] Frontend: client selector dropdown

### Phase 8: Email Processing (v1 Scope)
- [ ] `.msg` metadata extraction and email pattern matching (10MB limit)
- [ ] Email-specific auto-tags from subject line
- [ ] Content sanitization (HTML → plain text, strip attachments)
- [ ] Malformed .msg graceful handling

---

## 16. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM misclassifies document type | Wrong tags | Confidence thresholds, approval mode, path-based tags as baseline, provenance audit trail |
| Tag explosion (too many auto-tags) | DB/UI degradation | Hard limits: max 20 tags/doc, 100-char tag names, 500 client: cap, slugify normalization |
| Strategy doesn't match folder structure | Path tags missing | `generic` fallback always works; custom strategies via YAML; dry-run testing |
| Entity name normalization | Duplicate entities | Alias tables in strategy YAML; slugify fallback; admin merge endpoint (future) |
| Large batch upload overloads system | Queue backed up | Rate limiting in CLI, existing processing semaphore, resume manifest |
| Auto-tags bypass access control | Data leakage | Per-user `tag_access_enabled` toggle; auto-tags metadata-only until admin assignment; integration tests |
| YAML parsing errors | Strategy fails to load | Strict schema validation on load; reject unknown fields; fallback to `generic`; health warning |
| Strategy YAML drift across environments | Inconsistent tagging | Strategy version pinned per document; built-in strategies bundled in package |
| LLM timeout or failure | Missing content tags | 30s timeout, 1 retry, JSON repair, fallback to path-only with `partial` status |
| .msg file security | Malicious content | 10MB limit, no attachment extraction in v1, HTML sanitization, optional dependency |
| Batch upload partial failure | Inconsistent state | Per-file error codes, idempotency keys, resume manifest, deterministic final counts |

---

## 17. Success Metrics

| Metric | Target |
|--------|--------|
| Path-based tagging accuracy | > 95% (deterministic from rules) |
| LLM document type accuracy | > 85% (validated on 50-doc sample per strategy) |
| LLM entity identification accuracy | > 80% |
| Time to upload 100-file customer folder | < 5 min (batch) vs 60+ min (manual) |
| Manual tag assignments reduced | > 80% auto-applied |
| User approval rate (approval mode) | > 90% accepted without changes |
| Strategy switch time | < 1 minute (config change + restart) |
| Custom strategy creation time | < 30 minutes (copy + edit YAML) |
