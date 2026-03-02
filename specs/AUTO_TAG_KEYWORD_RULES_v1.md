---
title: Auto-Tag Keyword Rules — Content-Based Tag Override
status: FINAL - Ready for Implementation
created: 2026-03-02
revised: 2026-03-02
author: —
type: Enhancement
complexity: MODERATE
version: v1.0
review_status: Engineering review incorporated
issues: "#493, #494, #495, #496, #497, #498"
---

# Auto-Tag Keyword Rules — Content-Based Tag Override

## Summary

Add a `keyword_rules` section to strategy YAML files that allows **extracted
document text** to override path-derived tags for specific namespaces.
Path/directory structure remains the default source of truth; keyword rules act
as a targeted correction layer when folder placement does not reflect actual
document type.

**Primary use case:** A COI (Certificate of Insurance) stored in a `Policy`
folder should be tagged `doctype:coi`, not `doctype:policy`, because the
extracted header text "CERTIFICATE OF PROPERTY INSURANCE" is authoritative over
the folder name.

---

## Goals

- Path rules remain the default and run first at upload time (no behaviour change)
- Keyword rules scan the first 1500 normalized chars of **Docling-extracted text**
  (not raw file bytes) and can override path tags in the same namespace
- Keyword rules run in `processing_service.py` after Docling parsing, where
  extracted text is available — not at upload time
- Fully configurable in YAML — no code changes needed to add or update rules
- Manual tags are immutable to keyword rules on every processing attempt
- Provenance records which rule won and why
- All four strategy files receive keyword_rules examples

---

## Scope

### In Scope
- New `keyword_rules` YAML section in all strategy files (generic, insurance_agency,
  law_firm, construction)
- `keywords_any` matching (OR logic — match if ANY keyword found in content)
- `keywords_all` matching (AND logic — match if ALL keywords found)
- When both specified in one rule: AND between groups (both must satisfy)
- `priority` field (integer ≥ 1) for resolving multiple matching rules in the same
  namespace; YAML loader rejects `priority` < 1
- Keyword rules can override `doctype`, `topic`, `entity`, and `stage` namespaces
- YAML loader rejects keyword rules targeting `client` or `year` (hard fail, not
  runtime skip)
- Keyword rules never override manually assigned tags (enforced by provenance check)
- Case-insensitive matching by default (`case_sensitive: false`)
- Content scan window: first **1500 normalized characters of extracted text**
  (post-Docling, post-normalization)
- YAML loader rejects keyword rules referencing undeclared namespaces (fail fast)
- Text normalization contract for matching
- Transaction semantics: keyword swap + flush before vector indexing
- Reprocess idempotency guarantees
- Provenance updated: `keyword_candidates` key added
- New `source: "keyword"` value on `AutoTag`
- Rename `ConflictRecord.llm_value` to `override_value` with `override_source` field
  and backward-compatibility handling for existing provenance JSON

### Out of Scope
- Regex pattern matching in content (keywords only — regex is for `path_rules`)
- UI for managing keyword rules (YAML-only for now)
- Wiring `DocumentClassifier` (LLM Layer 3) into the upload/processing flow
  (separate issue — this spec is keyword rules only)
- Structured telemetry/metrics (deferred — provenance JSON is sufficient for v1)

---

## Architecture: Why Processing Time, Not Upload Time

**Problem:** At upload time, `content` is raw file bytes. For a PDF, the first
1500 bytes are `%PDF-1.4...` binary headers — not "CERTIFICATE OF INSURANCE".
DOCX, XLSX, and images are equally opaque as raw bytes.

**Solution:** Keyword rules must run **after Docling parses the file** and
produces extracted text chunks. This happens in `processing_service.py`, not
`document_service.py`.

```
Upload Time (document_service.py):
  ├── Layer 1: path_rules → [doctype:policy, client:cervantes, year:2025-2026]
  ├── Save to DB with status="pending"
  └── Enqueue to ARQ worker

Processing Time (processing_service.py):
  ├── Docling parses file → extracted text chunks available
  ├── Layer 2: keyword_rules on first 1500 normalized chars of extracted text
  │   → "CERTIFICATE OF PROPERTY INSURANCE" → doctype:coi overrides doctype:policy
  ├── Tag swap in DB + db.flush() (corrected tags visible)
  ├── Index chunks to pgvector (with corrected tags)
  └── Outer transaction commit; update document status → "ready"
```

Path tags are applied at upload (existing behaviour, unchanged). Keyword rules
correct them during processing once real text is available. This is a
**two-phase approach**: tags start as path-derived estimates, then get refined.

---

## Invariants

These rules hold on every processing attempt, including retries and reprocessing:

1. **Manual tags are immutable to Layer 2.** Keyword swap removes only tags
   whose `tag.name` appears in `provenance.path_candidates`. Tags not in that
   set (manual, email, or previously applied keyword tags) are never removed.

2. **Keyword evaluation is deterministic.** Same extracted text + same YAML =
   same keyword tags. No randomness, no LLM involvement.

3. **Reprocess idempotency.** On reprocess, keyword rules are re-evaluated from
   scratch against freshly extracted text. Previous keyword-derived tags in
   overrideable namespaces are cleared first. Provenance is overwritten (not
   appended). No duplicate tags per namespace/value/source.

4. **Transaction atomicity.** Keyword tag swap calls `db.flush()` before vector
   indexing. The outer `process_document()` transaction commits once at the end.
   If any step fails, the entire transaction rolls back — document stays in
   `processing` status with original path tags.

---

## Current Behaviour (Baseline)

```
upload(source_path, content) →
  path_rules  →  [doctype:policy, client:cervantes, year:2025-2026]
                        ↑
              "Policy" folder wins → WRONG for COI

process_document() →
  Docling extracts text → chunks available
  LLM classification → (exists but not wired in)
  No keyword matching → MISSING
```

`conflict.py` today recognises three sources: `path`, `llm`, `email`.

---

## Proposed Behaviour

```
upload(source_path) →
  Layer 1 — path_rules (unchanged)
    → [doctype:policy, client:cervantes, year:2025-2026]
    → document.tags set, status="pending", enqueue to ARQ

process_document() →
  Docling extracts text → chunks available

  Layer 2 — keyword_rules on first 1500 normalized chars of extracted text
    → "CERTIFICATE OF PROPERTY INSURANCE" matches {doctype:coi, priority:10}
    → conflict: keyword(10) beats path(0) for doctype namespace
    → DB tag swap: remove doctype:policy (only because it's in path_candidates)
    → DB tag add: doctype:coi
    → db.flush() — corrected tags visible to subsequent steps
    → Update document.auto_tag_source provenance

  Index chunks to pgvector with corrected tag list
  → outer commit; document.status = "ready"

Final tags → [doctype:coi, client:cervantes, year:2025-2026]  ✓
```

---

## Text Normalization Contract

Both the content haystack and keyword needles are normalized identically before
matching. The normalization function:

```python
import re
import unicodedata

def normalize_for_keyword_match(text: str, case_sensitive: bool = False) -> str:
    """Normalize text for keyword matching.

    1. Unicode NFC normalization (canonical decomposition + composition)
    2. Collapse all whitespace sequences to a single space
    3. Strip leading/trailing whitespace
    4. Case fold to uppercase (unless case_sensitive=True)
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not case_sensitive:
        text = text.upper()
    return text
```

Applied to:
- **Haystack:** The assembled content preview (1500 chars post-normalization)
- **Each keyword:** Every string in `keywords_any` and `keywords_all`

### Matching semantics

Matching is **exact normalized substring** (`needle in haystack`). Token-boundary
matching is not implemented. The risk of false positives is accepted for v1:
keyword phrases are typically 3+ words ("CERTIFICATE OF PROPERTY INSURANCE",
"THIS ENDORSEMENT CHANGES THE POLICY") making accidental substring collisions
near-zero in practice.

OCR artifacts that break keyword text will result in no match — the path-derived
tag survives, which is the safe fallback. OCR-resilient matching is deferred to
the future LLM Layer 3.

---

## Content Preview Assembly

The content preview fed to keyword matching is assembled deterministically:

1. **Source:** Post-filter chunks (chunks that passed the minimum-word-count
   filter in `processing_service.py`)
2. **Ordering:** Chunks in `chunk_index` order (ascending, stable — set by
   Docling/HybridChunker)
3. **Separator:** Single space between chunk texts
4. **Normalization:** `normalize_for_keyword_match()` applied to the assembled
   string
5. **Truncation:** At exactly 1500 characters after normalization

```python
raw_preview = " ".join(chunk.text for chunk in chunks)
content_preview = normalize_for_keyword_match(raw_preview, case_sensitive=False)
content_preview = content_preview[:1500]
```

If total normalized text is shorter than 1500 chars, the entire text is used.

---

## YAML Schema Addition

### New `keyword_rules` section

```yaml
keyword_rules:                        # Optional — omit if not needed
  - namespace: "doctype"              # REQUIRED: must be declared + overrideable
    value: "coi"                      # REQUIRED: tag value assigned on match
    priority: 10                      # REQUIRED: integer >= 1 (path = implicit 0)
    case_sensitive: false             # Optional, default: false
    keywords_any:                     # Match if ANY of these strings present
      - "CERTIFICATE OF INSURANCE"
      - "CERTIFICATE OF PROPERTY INSURANCE"
      - "THIS CERTIFICATE IS ISSUED AS A MATTER OF INFORMATION"
      - "ACORD 25"
      - "ACORD 27"
      - "ACORD 28"

  - namespace: "doctype"
    value: "loss_run"
    priority: 10
    keywords_any:
      - "LOSS RUN"
      - "LOSS HISTORY"
      - "CLAIMS EXPERIENCE"

  - namespace: "doctype"
    value: "endorsement"
    priority: 8
    keywords_any:
      - "POLICY CHANGE"
      - "ENDORSEMENT NUMBER"
      - "THIS ENDORSEMENT CHANGES THE POLICY"

  - namespace: "doctype"
    value: "invoice"
    priority: 8
    keywords_any:
      - "PREMIUM INVOICE"
      - "AMOUNT DUE"
      - "PREMIUM DUE"

  - namespace: "doctype"
    value: "application"
    priority: 7
    keywords_all:                     # ALL must be present
      - "APPLICANT"
      - "SIGNATURE"
      - "DATE OF APPLICATION"

  - namespace: "stage"
    value: "bind"
    priority: 9
    keywords_any:                     # stage override example
      - "BIND CONFIRMATION"
      - "BOUND AS OF"
      - "COVERAGE IS BOUND"
```

### Field reference

| Field | Type | Required | Default | Validation |
|-------|------|----------|---------|------------|
| `namespace` | string | Yes | — | Must be declared in `namespaces`; must be in `KEYWORD_OVERRIDEABLE` (`doctype`, `topic`, `entity`, `stage`). Loader rejects `client`, `year`, or undeclared namespaces. |
| `value` | string | Yes | — | Non-empty string |
| `priority` | int | Yes | — | Must be ≥ 1. Loader rejects 0 or negative. Path tags have implicit priority 0. |
| `keywords_any` | list[str] | Cond. | `[]` | Required if `keywords_all` absent. Non-empty strings. |
| `keywords_all` | list[str] | Cond. | `[]` | Required if `keywords_any` absent. Non-empty strings. |
| `case_sensitive` | bool | No | `false` | — |

**When both `keywords_any` and `keywords_all` are specified:** AND between groups —
at least one from `keywords_any` must be present AND all of `keywords_all` must be
present.

---

## Conflict Resolution Rules (updated)

Priority order within a namespace (highest wins):

| Priority | Source | Notes |
|----------|--------|-------|
| Always wins | manual | Human-assigned tags; never removed by Layer 2 |
| `rule.priority` (≥ 1) | keyword | Content-based; explicit in YAML |
| 0 (implicit) | path | Folder-derived default |

When two keyword rules match the same namespace, higher `priority` wins.
On a tie, first rule by YAML document order wins. This is deterministic across
retries.

**Manual-tag protection:** Keyword swap identifies removable tags by checking
`provenance.path_candidates`. Only tags whose `namespace:value` appears in that
list are eligible for removal. Tags not in the path provenance set — including
manual tags, email-derived tags, and tags from prior keyword runs — are never
touched.

`PATH_AUTHORITATIVE` and `LLM_AUTHORITATIVE` sets in `conflict.py` are unchanged.

---

## Technical Specification

### 1. `models.py` — New `KeywordRule` model

```python
class KeywordRule(BaseModel):
    """A content keyword rule that can override a path-derived tag."""

    model_config = ConfigDict(extra="forbid")

    namespace: str
    value: str
    priority: int
    keywords_any: list[str] = []
    keywords_all: list[str] = []
    case_sensitive: bool = False

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: int) -> int:
        if v < 1:
            raise ValueError("priority must be >= 1")
        return v

    @model_validator(mode="after")
    def require_keywords(self) -> KeywordRule:
        if not self.keywords_any and not self.keywords_all:
            raise ValueError("At least one of keywords_any or keywords_all is required")
        return self
```

Add to `StrategyYAML`:

```python
keyword_rules: list[KeywordRule] = []   # default empty — backward-compatible
```

Add namespace + overrideability validation to `StrategyYAML` (extend existing
`model_validator`):

```python
KEYWORD_OVERRIDEABLE = {"doctype", "topic", "entity", "stage"}

@model_validator(mode="after")
def validate_keyword_rule_namespaces(self) -> StrategyYAML:
    declared = set(self.namespaces.keys())
    for i, rule in enumerate(self.keyword_rules):
        if rule.namespace not in declared:
            raise ValueError(
                f"keyword_rules[{i}] references undeclared namespace '{rule.namespace}'"
            )
        if rule.namespace not in KEYWORD_OVERRIDEABLE:
            raise ValueError(
                f"keyword_rules[{i}] targets non-overrideable namespace '{rule.namespace}'. "
                f"Allowed: {sorted(KEYWORD_OVERRIDEABLE)}"
            )
    return self
```

Add to `AutoTag.source` Literal:

```python
source: Literal["path", "llm", "email", "manual", "keyword"]
```

### 2. `conflict.py` — Updated `ConflictRecord` and keyword resolution

Rename `llm_value` to `override_value` and add `override_source`:

```python
class ConflictRecord(TypedDict):
    """Record of a single namespace conflict."""

    namespace: str
    path_value: str
    override_value: str             # was: llm_value
    override_source: str            # "keyword" | "llm"
    winner: str                     # "path" | "keyword" | "llm"
    reason: str
```

**Backward compatibility:** Existing provenance JSON records contain `llm_value`.
New writes use `override_value` / `override_source`. Provenance readers must
handle both field names (check for `override_value` first, fall back to
`llm_value`). Existing records are not backfilled — they remain valid as-is.
There are no external API consumers or analytics readers of provenance JSON
today; all access is internal to `conflict.py` and `processing_service.py`.

Update existing `resolve_conflicts()` callers to use new field names. These are:
- `ai_ready_rag/services/auto_tagging/conflict.py` (internal)
- `ai_ready_rag/services/processing_service.py` (internal)

Add constant:

```python
KEYWORD_AUTHORITATIVE = {"doctype", "topic", "entity", "stage"}
```

New function for keyword-vs-path resolution (separate function — does not modify
existing `resolve_conflicts` signature):

```python
def resolve_keyword_conflicts(
    path_tags: list[AutoTag],
    keyword_tags: list[AutoTag],
) -> tuple[list[AutoTag], list[AutoTag], list[ConflictRecord]]:
    """Resolve keyword-vs-path conflicts.

    Only removes path tags — never manual, email, or other source tags.

    Returns:
        (winning_keyword_tags, losing_path_tags, conflict_records)
    """
    path_by_ns: dict[str, list[AutoTag]] = {}
    for pt in path_tags:
        if pt.source == "path":  # Only path-derived tags are removable
            path_by_ns.setdefault(pt.namespace, []).append(pt)

    winning: list[AutoTag] = []
    losing_path: list[AutoTag] = []
    conflicts: list[ConflictRecord] = []

    for kw_tag in keyword_tags:
        ns = kw_tag.namespace
        path_in_ns = path_by_ns.get(ns)
        if path_in_ns and ns in KEYWORD_AUTHORITATIVE:
            for pt in path_in_ns:
                if pt not in losing_path:
                    losing_path.append(pt)
                conflicts.append(ConflictRecord(
                    namespace=ns,
                    path_value=pt.value,
                    override_value=kw_tag.value,
                    override_source="keyword",
                    winner="keyword",
                    reason=f"{ns}: keyword rule (priority>0) overrides path",
                ))
            winning.append(kw_tag)
        else:
            # No path conflict, or namespace not overrideable
            winning.append(kw_tag)

    return winning, losing_path, conflicts
```

### 3. `strategy.py` — `parse_keywords()` method

Add `keyword_rules: list[KeywordRule]` to `AutoTagStrategy.__init__` and load
from `validated.keyword_rules` in `AutoTagStrategy.load()`.

New method:

```python
def parse_keywords(self, content_preview: str) -> list[AutoTag]:
    """Scan normalized extracted text against keyword_rules.

    Args:
        content_preview: First 1500 chars of normalized Docling-extracted text.
            Caller is responsible for normalization and truncation.

    Returns:
        At most one AutoTag per namespace (highest-priority matching rule).
    """
    if not self.keyword_rules or not content_preview:
        return []

    by_namespace: dict[str, list[KeywordRule]] = {}
    for rule in self.keyword_rules:
        by_namespace.setdefault(rule.namespace, []).append(rule)

    tags: list[AutoTag] = []

    for namespace, rules in by_namespace.items():
        for rule in sorted(rules, key=lambda r: r.priority, reverse=True):
            # Normalize keywords with same function as haystack
            haystack = content_preview  # already normalized by caller

            any_match = (
                any(normalize_for_keyword_match(kw, rule.case_sensitive) in haystack
                    for kw in rule.keywords_any)
                if rule.keywords_any else True
            )
            all_match = (
                all(normalize_for_keyword_match(kw, rule.case_sensitive) in haystack
                    for kw in rule.keywords_all)
                if rule.keywords_all else True
            )

            if any_match and all_match:
                tags.append(AutoTag(
                    namespace=namespace,
                    value=rule.value,
                    source="keyword",
                    confidence=1.0,
                    strategy_id=self.id,
                    strategy_version=self.version,
                ))
                break  # highest-priority match wins for this namespace

    return tags
```

### 4. `processing_service.py` — Wire keyword rules after Docling extraction

Insert keyword rule evaluation after chunks are produced and filtered (around
line 389) and before vector indexing (line 447). This is in `process_document()`:

```python
# After chunks are created and filtered (existing code ~line 389):
chunks: list[ChunkInfo] = [...]

# --- NEW: Layer 2 keyword rules ---
keyword_conflicts: list[ConflictRecord] = []
if (
    document.auto_tag_strategy
    and document.source_path
    and strategy is not None
    and strategy.keyword_rules
):
    # Assemble content preview: post-filter chunks, chunk_index order,
    # space-separated, normalized, truncated at 1500 chars
    raw_preview = " ".join(chunk.text for chunk in chunks)
    content_preview = normalize_for_keyword_match(raw_preview, case_sensitive=False)
    content_preview = content_preview[:1500]

    if content_preview:
        keyword_auto_tags = strategy.parse_keywords(content_preview)
        if keyword_auto_tags:
            # Reconstruct path-derived AutoTags from provenance
            path_auto_tags = self._get_path_auto_tags_from_provenance(document)

            winning_kw, losing_path, keyword_conflicts = resolve_keyword_conflicts(
                path_tags=path_auto_tags,
                keyword_tags=keyword_auto_tags,
            )
            # Swap tags in DB: remove path losers, add keyword winners
            self._apply_keyword_tag_swaps(
                document, winning_kw, losing_path, strategy, db
            )
            # db.flush() inside _apply_keyword_tag_swaps ensures corrected
            # tags are visible before vector indexing

    # Update provenance with keyword results
    self._update_provenance_with_keywords(document, keyword_auto_tags, keyword_conflicts)

# Continue with existing flow: vector indexing reads document.tags
# which now reflects keyword corrections
```

New helpers in `processing_service.py`:

```python
def _get_path_auto_tags_from_provenance(self, document: Document) -> list[AutoTag]:
    """Reconstruct path-derived AutoTag objects from provenance JSON.

    Uses provenance.path_candidates to identify which tags are path-derived
    and therefore eligible for keyword override. Tags not in provenance
    (manual, email) are excluded — they are immutable to Layer 2.
    """
    provenance = json.loads(document.auto_tag_source or "{}")
    path_candidates = provenance.get("path_candidates", [])

    tags = []
    for candidate in path_candidates:
        tags.append(AutoTag(
            namespace=candidate["namespace"],
            value=candidate["value"],
            source="path",
            confidence=candidate.get("confidence", 1.0),
        ))
    return tags

def _apply_keyword_tag_swaps(
    self,
    document: Document,
    winning_kw: list[AutoTag],
    losing_path: list[AutoTag],
    strategy: AutoTagStrategy,
    db: Session,
) -> None:
    """Remove losing path tags from document, add winning keyword tags.

    Only removes tags identified as path-derived via provenance.
    Manual and other-source tags are never touched.
    Calls db.flush() to ensure corrected tags are visible before vector indexing.
    """
    losing_names = {f"{t.namespace}:{t.value}" for t in losing_path}

    # Remove only path-derived losing tags
    document.tags = [t for t in document.tags if t.name not in losing_names]

    # Add winning keyword tags
    for kw in winning_kw:
        tag_obj = self._ensure_tag_exists(kw, strategy, db)
        if tag_obj and tag_obj not in document.tags:
            document.tags.append(tag_obj)

    db.flush()  # Corrected tags visible to subsequent vector indexing
```

### 5. Strategy YAML updates

Add `keyword_rules` sections to all four strategy files:

| File | Rules to add |
|------|-------------|
| `insurance_agency.yaml` | coi, loss_run, endorsement, invoice, application, stage:bind |
| `generic.yaml` | invoice, correspondence (basic document type hints) |
| `law_firm.yaml` | contract, pleading, discovery, motion, correspondence |
| `construction.yaml` | submittal, rfi, lien_waiver, change_order, contract |

---

## Provenance Update

`build_provenance()` gains a `keyword_candidates` key:

```python
provenance = {
    "strategy_id": ...,
    "strategy_version": ...,
    "path_candidates": [...],
    "keyword_candidates": [          # NEW
        {"namespace": t.namespace, "value": t.value}
        for t in keyword_tags
    ],
    "conflicts": [...],             # Now includes keyword conflicts
    "applied": [...],
    "discarded": [...],
    "suggested": [...],
}
```

**On reprocess:** Provenance is overwritten entirely (not appended). The new
provenance reflects the current run's path_candidates, keyword_candidates,
conflicts, and applied tags.

---

## Schema Compatibility Checklist

| Change | Storage Location | Migration Needed | Consumers | Risk |
|--------|-----------------|-----------------|-----------|------|
| `AutoTag.source` adds `"keyword"` | Runtime Pydantic — never persisted | No | Internal only | None |
| `ConflictRecord.llm_value` → `override_value` | `document.auto_tag_source` JSON | No — new writes use new key; old records keep `llm_value` | `conflict.py`, `processing_service.py` (internal) | Low — readers check both keys |
| `ConflictRecord` adds `override_source` | `document.auto_tag_source` JSON | No — additive field | Same as above | None |
| `keyword_candidates` added to provenance | `document.auto_tag_source` JSON | No — additive field | Same as above | None |
| `KeywordRule` model added | Not persisted — YAML only | No | `strategy.py` loader | None |
| `StrategyYAML.keyword_rules` field | YAML files | No — default `[]` | `strategy.py` loader | None — backward-compatible |

---

## Example: Full Path Through the System

**File:**
```
C:\Users\jjob\OneDrive\Customer_Projects\Test_Sara_2\
  Cervantes (12-01)\25 Renewal\Policy\25-26 Certificate of Insurance.pdf
```

**Upload time — Layer 1 (path_rules):**
```
client:cervantes   level 0 — regex strips "(12-01)", slugify
year:2025-2026     level 1 — "25 Renewal" → year_range
doctype:policy     level 2 — mapping "Policy" → "policy"

→ document.tags = [client:cervantes, year:2025-2026, doctype:policy]
→ provenance.path_candidates = [{ns:client, val:cervantes}, {ns:year, val:2025-2026}, {ns:doctype, val:policy}]
→ document.status = "pending"
→ Enqueue to ARQ
```

**Processing time — Docling extraction:**
```
Docling parses PDF → extracts text chunks
First chunk text: "CERTIFICATE OF PROPERTY INSURANCE\nACORD 25 (2016/03)..."
```

**Processing time — Layer 2 (keyword_rules):**
```
content_preview = normalize("CERTIFICATE OF PROPERTY INSURANCE ACORD 25...")[:1500]
→ "CERTIFICATE OF PROPERTY INSURANCE" matches {doctype:coi, priority:10}
→ Check provenance: doctype:policy IS in path_candidates → eligible for removal
→ DB swap: remove doctype:policy, add doctype:coi
→ db.flush()
```

**Processing time — Vector indexing:**
```
Index chunks to pgvector with tags: [client:cervantes, year:2025-2026, doctype:coi]
Outer commit; document.status = "ready"
```

**Final tags:**
```
client:cervantes    (path — unchanged)
year:2025-2026      (path — unchanged)
doctype:coi         (keyword override)
```

**Manual-tag protection example:** If admin had manually set `doctype:policy`
(not from path), it would NOT appear in `provenance.path_candidates`, and keyword
rules would not remove it.

---

## Files to Modify

| File | Change |
|------|--------|
| `ai_ready_rag/services/auto_tagging/models.py` | Add `KeywordRule` with priority ≥ 1 validation; add `keyword_rules` to `StrategyYAML` with namespace + overrideability validation; add `"keyword"` to `AutoTag.source` |
| `ai_ready_rag/services/auto_tagging/strategy.py` | Add `keyword_rules` to `__init__` + `load()`; add `parse_keywords()`; add `normalize_for_keyword_match()` |
| `ai_ready_rag/services/auto_tagging/conflict.py` | Rename `ConflictRecord.llm_value` → `override_value`; add `override_source`; add `KEYWORD_AUTHORITATIVE`; add `resolve_keyword_conflicts()`; update existing callers |
| `ai_ready_rag/services/processing_service.py` | After Docling chunks: assemble preview, normalize, call `parse_keywords()`, resolve conflicts, swap tags + flush before vector indexing; add `_get_path_auto_tags_from_provenance()`, `_apply_keyword_tag_swaps()` |
| `data/auto_tag_strategies/insurance_agency.yaml` | Add `keyword_rules` for coi, loss_run, endorsement, invoice, application, stage:bind |
| `data/auto_tag_strategies/generic.yaml` | Add basic `keyword_rules` for invoice, correspondence |
| `data/auto_tag_strategies/law_firm.yaml` | Add `keyword_rules` for contract, pleading, discovery, motion |
| `data/auto_tag_strategies/construction.yaml` | Add `keyword_rules` for submittal, rfi, lien_waiver, change_order |

---

## Rollout

Existing documents with incorrect path-derived tags (e.g., COIs tagged
`doctype:policy`) are corrected by reprocessing via the existing
`POST /api/admin/documents/reprocess` endpoint. Once keyword rules ship, admin
can reprocess affected documents to trigger the full pipeline including Layer 2
keyword evaluation. No one-time migration job is required.

**Recommended rollout steps:**

1. Deploy code + updated YAML strategies
2. Admin identifies documents with suspected mis-tags (e.g., filter by
   `doctype:policy` in the Policy folder)
3. Reprocess affected documents via admin API
4. Verify corrected tags in UI and pgvector

---

## Acceptance Criteria

- [ ] A document with "CERTIFICATE OF PROPERTY INSURANCE" in its extracted text,
      stored in a `Policy` folder, is tagged `doctype:coi` not `doctype:policy`
- [ ] A document with no keyword match retains its path-derived tags unchanged
- [ ] Keyword rules run on Docling-extracted text, NOT raw file bytes
- [ ] `keywords_any` matches when ANY listed keyword is present (case-insensitive)
- [ ] `keywords_all` matches only when ALL listed keywords are present
- [ ] When both `keywords_any` and `keywords_all` specified: AND between groups
- [ ] When two keyword rules match the same namespace, higher `priority` wins
- [ ] Priority tie broken by YAML document order (first rule wins)
- [ ] Keyword rules can override `doctype`, `topic`, `entity`, `stage`
- [ ] YAML loader rejects keyword rules targeting `client` or `year`
- [ ] YAML loader rejects `priority` < 1
- [ ] Keyword rules never override manually assigned tags (provenance-based check)
- [ ] `case_sensitive: false` (default) matches regardless of case
- [ ] Text normalization: Unicode NFC, collapse whitespace, case fold
- [ ] YAML loader rejects a keyword rule referencing an undeclared namespace
- [ ] YAML loader rejects a keyword rule with neither `keywords_any` nor `keywords_all`
- [ ] Provenance JSON includes `keyword_candidates` list
- [ ] `ConflictRecord` uses `override_value` and `override_source`; readers handle
      legacy `llm_value` field
- [ ] Tag corrections reflected in pgvector payload (chunks indexed with corrected
      tags after `db.flush()`)
- [ ] Reprocess produces identical tags for identical content + YAML (idempotent)
- [ ] Reprocess overwrites provenance (no duplicate entries)
- [ ] All existing tests pass (backward-compatible; `keyword_rules: []` is default)

---

## Test Matrix

| # | Test Case | Validates | Priority |
|---|-----------|-----------|----------|
| 1 | COI in Policy folder → keyword match | Core override | P0 |
| 2 | No keyword match → path tags survive | Fallback safety | P0 |
| 3 | Manual tag + keyword rule in same namespace | Manual immutability (Invariant 1) | P0 |
| 4 | Reprocess same document twice | Idempotency (Invariant 3) | P0 |
| 5 | Transaction failure mid-swap | Rollback (Invariant 4) | P0 |
| 6 | `keywords_any` with multiple keywords | OR logic | P1 |
| 7 | `keywords_all` with multiple keywords | AND logic | P1 |
| 8 | Both `keywords_any` + `keywords_all` | AND between groups | P1 |
| 9 | Two rules same namespace, different priority | Higher priority wins | P1 |
| 10 | Two rules same namespace, same priority | YAML order tiebreak | P1 |
| 11 | `priority: 0` in YAML | Load-time rejection | P1 |
| 12 | `namespace: "client"` in keyword rule | Load-time rejection | P1 |
| 13 | Undeclared namespace in keyword rule | Load-time rejection | P1 |
| 14 | `case_sensitive: true` matching | Case-sensitive mode | P1 |
| 15 | `case_sensitive: false` with mixed-case content | Case-insensitive mode | P1 |
| 16 | OCR-degraded text (keyword broken by artifacts) | No match → path survives | P2 |
| 17 | Provenance JSON with legacy `llm_value` key | Schema backward compat | P2 |
| 18 | Provenance after keyword override | `keyword_candidates` present | P2 |
| 19 | Empty `keyword_rules: []` in YAML | Backward compat — no behavior change | P2 |
| 20 | Content shorter than 1500 chars | Full text used | P2 |

---

## Future Work (Separate Issues)

- **Wire `DocumentClassifier` (LLM) into processing pipeline** — Layer 3 for
  ambiguous cases not covered by keyword rules. The code exists in
  `classifier.py`; needs integration into `processing_service.py` and conflict
  resolution with keyword results.
- **Keyword rules UI** — Admin interface to manage keyword rules without editing YAML.
- **Regex content matching** — Allow regex patterns in keyword rules for more
  flexible matching (e.g., `ACORD \d{2}` instead of listing every form number).
- **Telemetry** — Structured metrics for keyword rule hit rates, override counts
  by namespace, skip reasons, and false-positive monitoring.

---

## Related Files

- `ai_ready_rag/services/auto_tagging/strategy.py`
- `ai_ready_rag/services/auto_tagging/models.py`
- `ai_ready_rag/services/auto_tagging/conflict.py`
- `ai_ready_rag/services/auto_tagging/classifier.py`
- `ai_ready_rag/services/processing_service.py`
- `ai_ready_rag/services/document_service.py`
- `data/auto_tag_strategies/insurance_agency.yaml`
- `data/auto_tag_strategies/generic.yaml`
- `data/auto_tag_strategies/law_firm.yaml`
- `data/auto_tag_strategies/construction.yaml`
