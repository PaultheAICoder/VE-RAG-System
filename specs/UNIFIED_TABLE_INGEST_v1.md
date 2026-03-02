# UNIFIED_TABLE_INGEST_v1 — Format-Agnostic Table Extraction, Storage, and SQL Routing

**Version:** 1.1
**Date:** 2026-03-01
**Status:** Draft — Engineering Review Incorporated
**Type:** Enhancement
**Complexity:** COMPLEX
**Depends on:** ProcessingService, DoclingChunker, ExcelTablesService, QueryRouter, ModuleRegistry
**Changelog:** v1.1 — Address engineering review: tenant-scoped registry, atomic blue/green re-upload,
SQL execution-time access control, CSV parser contract, prompt token caps, startup scaling SLOs,
signal canonicalization, schema lifecycle controls, dual-registry precedence, naming contract,
observability metrics, executable acceptance criteria.

---

## 1. Problem Statement

Structured table data in uploaded documents is only SQL-queryable if the source format is Excel.
All other formats (PDF, images, DOCX, CSV) store tables as unstructured text chunks in the vector
store, making precise data queries unreliable (wrong values, hallucinated totals, no citations).

Three compounding gaps:

### 1.1 Format Gap — Only Excel reaches the SQL routing path

```
Current ingest pipeline
├── Excel (.xlsx)
│   └── ExcelRouter → excel_tables schema → excel_table_registry
│       → ExcelTablesService → SQL templates registered  ✓
│
├── PDF / Image / DOCX / CSV
│   └── DoclingChunker → text chunks → chunk_vectors only
│       → QueryRouter never routes to SQL               ✗
│       → Tables answered by RAG: slow, imprecise, no citation
```

Docling **does** extract tables from PDFs and images into structured `TableItem` objects with
typed rows and column headers. This structured data is discarded before reaching the vector store
— only the markdown serialization is chunked and indexed.

### 1.2 Signal Gap — column_signals built from headers only, not content

`_compute_column_signals()` generates routing signals from column header names only:

```python
signals["item"] = ["item"]                   # header only
# Missing:
signals["item"] = ["item", "revenue", "cogs", "income tax expense", ...]  # actual row values
```

For row-labeled tables (P&L, product catalogs, HR data), users query by **data values**, not
column names. The current design misses all of them.

### 1.3 Static Vocabulary Gap — P&L trigger phrases require manual maintenance

`_PL_TRIGGER_PHRASES` is a hardcoded list. When a P&L spreadsheet contains line items not on
the list ("Income Tax Expense", "Depreciation & Amortization"), those queries fall through to RAG.
No automation catches the gap at ingest time.

---

## 2. Goals

1. **Universal SQL routing** — any table from any supported format becomes SQL-queryable after ingest
2. **Content-aware signals** — routing vocabulary built from actual cell values, not just headers
3. **Zero manual maintenance** — no hardcoded trigger phrase lists; all signals derived from data
4. **Backward compatible** — existing Excel tables continue to work without re-ingestion
5. **Tenant-safe** — all registry operations scoped by tenant_id; no cross-tenant bleed
6. **Access control at execution time** — tag predicates enforced at SQL execution, not just routing

---

## 3. Scope

### In Scope
- Tables extracted by Docling from PDF, DOCX, and image files
- Excel tables (current path extended with row value sampling)
- CSV files ingested as single tables (both structured + vector paths)
- Startup registration for all formats via a unified `TableRegistrationService`
- `document_table_registry` — generalized, tenant-scoped replacement for `excel_table_registry`
- Row value sampling at ingest time stored in the registry
- `_compute_column_signals()` enhanced with sampled row values and canonicalization
- Atomic blue/green table replacement on re-upload
- SQL execution-time access control predicate injection

### Out of Scope
- HTML tables in web pages
- Tables inside email attachments (separate pipeline)
- Real-time/streaming table updates
- Table JOIN queries across multiple documents (future)
- Schema inference for nested/multi-header tables (future)
- CSV multi-sheet (CSVs are single-table by definition)

---

## 4. Current Architecture (to be changed)

```
excel_table_registry
  └── UNIQUE(schema_name, table_name)  ← NOT tenant-scoped
  └── table_name, schema_name, columns (JSON), column_types (JSON),
      document_id, row_count, table_metadata (JSON: access_tags)

ExcelTablesService (reads excel_table_registry at startup)
  ├── P&L tables → _register_pl_template()
  │   └── Hardcoded _PL_TRIGGER_PHRASES list
  └── Non-P&L tables → _register_table_template()
      └── _compute_column_signals(table_name, columns, column_types)
          └── column headers + table name tokens only (no row values)

Access control: template selection only — not enforced at SQL execution time
```

---

## 5. Proposed Architecture

```
document_table_registry  (new — supersedes excel_table_registry)
  └── UNIQUE(tenant_id, schema_name, table_name)  ← tenant-scoped
  └── tenant_id, table_name, schema_name, source_format, columns (JSON),
      column_types (JSON), row_value_samples (JSON),
      document_id, row_count, table_metadata (JSON: access_tags)

Ingest time (all formats):
  ├── Excel → VERagPostgresStructuredDB (existing, updated)
  │   └── Blue/green table swap on re-upload                    ← NEW
  │   └── Row value sampling → row_value_samples in registry    ← NEW
  │   └── writes to document_table_registry                     ← updated
  │
  ├── PDF / DOCX / Image → TableExtractionAdapter (new)
  │   └── Docling TableItem → DataFrame
  │   └── Blue/green table swap
  │   └── writes to document_tables schema
  │   └── registers in document_table_registry
  │
  └── CSV → CsvProcessingService (new)
      └── Both paths: document_tables + chunk_vectors
      └── Parser contract (see §6.5)

Startup registration:
  TableRegistrationService(db_url).discover_and_register_all()
    └── reads document_table_registry (tenant-scoped)
    └── dual-registry compat: document_table_registry wins on conflict
    └── all tables → _register_table_template() with column_signals
        └── _compute_column_signals(columns, column_types, row_value_samples)
            ├── column headers + synonyms
            ├── canonicalized row value samples (see §6.4)
            └── __table__ tokens + __quantitative__ signals
    └── startup SLO gates (see §6.6)

SQL execution-time access control:
  _execute_sql_route() / _execute_nl2sql_route()
    └── inject tenant_id + tag predicates into query
    └── validate predicates against allowlist before execution  ← NEW
```

---

## 6. Detailed Specifications

### 6.1 Tenant Safety

All registry operations are scoped by `tenant_id`. The default tenant is `'default'` for
single-tenant deployments. Multi-tenant deployments pass tenant_id from the authenticated user.

**Invariants:**
- Every INSERT/SELECT/UPDATE on `document_table_registry` includes `WHERE tenant_id = :tenant_id`
- `TableRegistrationService` receives `tenant_id` at construction time, not as a per-call param
- SQL templates registered in `ModuleRegistry` are namespaced: `"{tenant_id}.{template_name}"`
- No cross-tenant registry read is permitted at any layer

### 6.2 Atomic Blue/Green Table Replacement

On document re-upload, tables are replaced atomically to prevent query-time unavailability:

```
1. Create temp table:  document_tables."<name>_new"
   └── Load and validate all rows (row count, column schema check)
   └── If validation fails: drop _new, abort, raise error

2. Within a single transaction:
   a. UPDATE document_table_registry SET table_name = '<name>_old' WHERE ...
   b. UPDATE document_table_registry SET table_name = '<name>' (for _new row)
   c. COMMIT

3. Post-commit (best-effort):
   DROP TABLE document_tables."<name>_old"
   └── If DROP fails: log warning, enqueue orphan cleanup job
```

**Race condition handling:**
- Registry swap is a single atomic transaction; in-flight queries against `<name>` complete
- New queries after commit see the new table immediately
- Orphan cleanup job (§6.7) handles any `_old` tables left behind by failed drops

### 6.3 SQL Execution-Time Access Control

Template selection (routing) is a performance optimization, not a security gate. Tag and tenant
predicates are enforced at execution time:

**Predicate injection (NL2SQL path):**
```sql
-- Claude generates:
SELECT "item", "value" FROM excel_tables."PL_2019" WHERE "item" = 'Revenue'

-- Execution layer wraps with tenant + tag guard:
SELECT * FROM (
    SELECT "item", "value" FROM excel_tables."PL_2019" WHERE "item" = 'Revenue'
) AS __inner__
WHERE 1=1
-- tenant predicate injected if table has tenant column
-- tag predicate: checked against template.access_tags vs user.tags BEFORE execution
```

**Pre-execution validation (both SQL and NL2SQL paths):**
1. `SqlInjectionGuard.validate()` — existing, checks SELECT-only
2. **NEW** `TagPredicateValidator.validate(template, user_tags)`:
   - If `template.access_tags` is non-empty: user must have ALL required tags
   - Raises `AccessDeniedError` (→ HTTP 403) before any SQL executes
   - Does not rely on routing to have filtered correctly

**Allowlisted predicate forms:**
Only the following WHERE clause patterns may be injected:
- `WHERE tenant_id = :tenant_id`
- `WHERE tag IN (:tags)`
- Compound AND of the above

Any other injected predicate form is rejected with a security error.

### 6.4 Signal Canonicalization Pipeline

Applied to every value before storing in `row_value_samples` and before routing comparison:

```
Raw value: "  Income-Tax Expense\u00a0"
Step 1 — Unicode NFKC normalize:  "  Income-Tax Expense "
Step 2 — Strip leading/trailing whitespace: "Income-Tax Expense"
Step 3 — Replace hyphens/underscores with space: "Income Tax Expense"
Step 4 — Collapse multiple spaces: "Income Tax Expense"
Step 5 — Lowercase: "income tax expense"
Step 6 — Stopword filter (remove): the, a, an, of, in, for, with, and, or, to, by, at
          → "income tax expense"  (none removed here)
Step 7 — Length gate: drop values < 3 chars or > 80 chars
Step 8 — Deduplicate within column
```

**Stopword list** (exhaustive, not regex-based):
`{"the", "a", "an", "of", "in", "for", "with", "and", "or", "to", "by", "at", "is", "are",
"was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
"would", "could", "should", "may", "might", "shall", "can", "not", "no", "nor"}`

Stopwords are only removed when the resulting string has ≥ 3 chars. "A" alone is not stored.

**No stemming in v1.0.** Synonym expansion is out of scope.

### 6.5 CSV Parser Contract

CSV ingestion uses `pandas.read_csv()` with the following normative settings:

| Concern | Behavior |
|---------|----------|
| Delimiter detection | Try `,` first; if single column > 90% of rows, retry with `;` then `\t` |
| BOM handling | `encoding='utf-8-sig'` (strips UTF-8 BOM silently) |
| Quoted multiline fields | `quoting=csv.QUOTE_MINIMAL`, `engine='python'` (handles embedded newlines) |
| Header inference | First row treated as header if all values are strings; otherwise no-header + promote via §7.1 heuristic |
| Encoding fallback | Try UTF-8 → UTF-8-SIG → latin-1; log encoding used |
| Bad rows | `error_bad_lines=False` (pandas ≥1.3: `on_bad_lines='skip'`); increment `bad_row_count` counter |
| Bad row threshold | If `bad_row_count / total_rows > 0.05`: abort ingest, set document status `failed`, log reason |
| Empty file | Abort; document status `failed` with reason `csv_empty` |
| Single-column result | After delimiter detection, if still single column: abort with reason `csv_delimiter_undetected` |
| Max file size | 100MB (configurable via `csv_max_file_size_mb`); larger files rejected at upload |

All counters (`bad_row_count`, `encoding_used`, `delimiter_detected`) stored in `table_metadata` JSON.

### 6.6 Prompt and Context Size Controls

Hard limits applied before constructing the NL2SQL prompt:

| Limit | Value | Config key |
|-------|-------|------------|
| Max columns included in prompt | 20 | `nl2sql_max_prompt_columns` |
| Max sample values per column in prompt | 20 | `nl2sql_max_samples_per_column` |
| Total sample token budget | 2000 tokens (≈8000 chars) | `nl2sql_sample_token_budget` |
| Max total prompt size | 8000 tokens | `nl2sql_max_prompt_tokens` |

**Truncation order** when budget is exceeded:
1. Reduce samples per column (from 20 down to 5 minimum)
2. Drop columns with zero string samples first
3. Drop columns whose header doesn't appear in the query (lowest relevance)
4. If still over budget: include only columns that overlap with query terms

Truncation is logged at DEBUG level with column names dropped.

**Row sampling storage limits** (registry, not prompt):

| Limit | Value | Config key |
|-------|-------|------------|
| Max sample values per column stored | 50 | `table_row_sample_max_per_col` |
| Max cardinality before column skipped | 200 | `table_row_sample_cardinality_limit` |
| Max total signals per table (all columns) | 500 | `table_max_signals_total` |

### 6.7 Schema Lifecycle and Orphan Cleanup

**Orphan definition:** A table in `document_tables` or `excel_tables` schema with no
corresponding row in `document_table_registry` or `excel_table_registry`.

**Causes:** failed blue/green drop, manual DB operations, document hard-delete without cleanup.

**Orphan cleanup job** runs:
- At startup (before registration), logs orphans found
- On-demand via admin endpoint `POST /api/admin/tables/reconcile`
- Does NOT auto-drop: logs orphans, updates metric `orphaned_tables_count`
- Admin must confirm drop via `POST /api/admin/tables/orphans/{table_name}/drop`

**Retention policy:**
- When a document is deleted, its registry rows are deleted and tables are dropped immediately
- `_old` tables from interrupted blue/green swaps are dropped at next startup reconciliation
  if the swap completed (i.e., no corresponding active registry row)

### 6.8 Startup Registration SLOs and Scaling

| Registry size | Startup SLO | Behavior |
|---------------|-------------|----------|
| 0–100 tables | < 5 seconds | Full scan, synchronous |
| 100–500 tables | < 30 seconds | Full scan, log warning if > 15s |
| 500+ tables | No SLO guarantee | Log warning, enable `--lazy-register` flag |

**`--lazy-register` mode** (env: `TABLE_LAZY_REGISTER=true`):
- At startup: register only tables modified in the last 7 days
- On first query hit for an unregistered table: register on demand, cache result
- Avoids full scan; trades cold-start latency for startup speed
- Default: disabled. Recommended for deployments with > 500 registered tables.

### 6.9 Dual-Registry Transition

During the v1.x transition, `TableRegistrationService` reads both registries at startup.

**Precedence rule:** `document_table_registry` wins unconditionally over `excel_table_registry`
for the same logical key `(tenant_id, schema_name, table_name)`.

**Conflict handling:**
```
If same (tenant_id, schema_name, table_name) exists in both registries:
  - Use document_table_registry row
  - Log WARNING: "dual_registry_conflict: table=<name> using=document_table_registry
                  divergent_fields=<list of fields that differ>"
  - Do NOT fail startup
```

**Sunset:** `excel_table_registry` reads are removed in v2.0. Admin tooling will migrate
remaining rows to `document_table_registry` as part of the v2.0 upgrade script.

### 6.10 Table Naming Contract

**Identifier construction:**
```
base = slugify(document_name, separator="_")[:40]
       where slugify: Unicode NFKC → lowercase → replace [^a-z0-9] with _ → collapse __
suffix = sha256(f"{document_id}:{table_index}")[:8]
table_name = f"{base}_{suffix}"
```

**Examples:**
- `Benefits_Guide_2025.pdf`, table 0 → `benefits_guide_2025_a3f92b1c`
- `Q4 Financial Report (FINAL).pdf`, table 2 → `q4_financial_report_final_8e4d1a7b`

**Guarantees:**
- Always ≤ 63 bytes (PostgreSQL identifier limit): `40 + 1 + 8 = 49` chars max
- Deterministic: same document + same table index always produces same name
- No reserved word conflicts: all names begin with lowercase alpha after slugify
- All identifiers double-quoted at every SQL construction site

**Collision handling:**
If the generated name already exists in the registry under a different document_id (hash
collision — astronomically unlikely but handled): append `_2`, `_3`, etc.

---

## 7. New and Modified Components

### 7.1 NEW: `TableExtractionAdapter`

**File:** `ai_ready_rag/services/table_extraction_adapter.py`

Extracts tables from Docling output and persists them via blue/green swap.

```python
class TableExtractionAdapter:
    def extract_and_persist(
        self,
        docling_document: DoclingDocument,
        document_id: str,
        document_name: str,
        source_format: str,
        access_tags: list[str],
        tenant_id: str,
        db_url: str,
    ) -> list[str]:
        """Returns list of table_names registered in document_table_registry."""
```

**Header promotion heuristic** (§6.5 decision):
```python
def _promote_first_row_as_headers(df: pd.DataFrame) -> pd.DataFrame:
    if list(df.columns) == list(range(len(df.columns))):
        df.columns = df.iloc[0].astype(str).str.strip()
        return df.iloc[1:].reset_index(drop=True)
    return df
```

**Minimum table gates:**
- `< 2 columns` → skip (likely parsing artifact)
- `< 2 rows` (after header promotion) → skip
- `> 10,000 rows` → truncate to 10,000, log warning `table_truncated`

### 7.2 NEW: `TableRegistrationService`

**File:** `ai_ready_rag/services/table_registration_service.py`

```python
class TableRegistrationService:
    def __init__(self, database_url: str, tenant_id: str = "default") -> None: ...

    def discover_and_register_all(self) -> int:
        """Reads document_table_registry (+ excel_table_registry compat).
        Returns total SQL templates registered. Respects startup SLOs (§6.8)."""

    def _compute_column_signals(
        self,
        table_name: str,
        columns: list[str],
        column_types: dict,
        row_value_samples: dict,
    ) -> dict:
        """Builds signals from headers + canonicalized row values + table tokens.
        Enforces max_signals_total cap (§6.6)."""
```

**No P&L special path.** All tables use `_register_table_template()`.

### 7.3 NEW: `TagPredicateValidator`

**File:** `ai_ready_rag/services/sql_access_control.py`

```python
class TagPredicateValidator:
    def validate(self, template: SQLTemplate, user_tags: list[str]) -> None:
        """Raise AccessDeniedError if user lacks required tags.
        Called at execution time, before SqlInjectionGuard."""
        if template.access_tags and not set(template.access_tags).issubset(set(user_tags)):
            raise AccessDeniedError(
                f"Insufficient tags for table {template.name}. "
                f"Required: {template.access_tags}"
            )
```

### 7.4 MODIFIED: `VERagPostgresStructuredDB`

- Blue/green swap on re-upload (§6.2)
- Row value sampling after table creation → `row_value_samples` in registry
- Writes to `document_table_registry` with `source_format='excel'` and `tenant_id`

### 7.5 MODIFIED: `ProcessingService`

After `DoclingChunker` completes for PDF/DOCX/image:
- Instantiate `TableExtractionAdapter` if `structured_table_extraction_enabled`
- Pass `tenant_id` from authenticated user context
- Update `document.doc_tables_created` and `document.doc_table_names`

### 7.6 MODIFIED: `_execute_nl2sql_route()` in `RAGService`

- Load `row_value_samples` from registry for the matched template
- Apply token budget truncation (§6.6) before building prompt
- Call `TagPredicateValidator.validate()` before SQL execution
- Include sample values in prompt schema block:

```
Table: excel_tables."PL_2019"
Columns: item (text), value (numeric)
Known values for "item": ["Revenue", "COGS", "Gross Profit", "Income Tax Expense", "Net Income"]

Question: What was the income tax expense in 2019?
Generate a single SELECT statement using ONLY the exact values shown above.
```

### 7.7 DEPRECATED: `ExcelTablesService`

Thin wrapper delegating to `TableRegistrationService`. Removed in v2.0.

---

## 8. Database Schema

### 8.1 New table: `document_table_registry`

```sql
CREATE TABLE document_table_registry (
    id              VARCHAR PRIMARY KEY,
    tenant_id       VARCHAR NOT NULL DEFAULT 'default',
    table_name      VARCHAR NOT NULL,
    schema_name     VARCHAR NOT NULL DEFAULT 'document_tables',
    source_format   VARCHAR NOT NULL,          -- 'excel', 'pdf', 'image', 'docx', 'csv'
    source_page     INTEGER,
    table_index     INTEGER DEFAULT 0,
    columns         TEXT NOT NULL,             -- JSON list[str]
    column_types    TEXT,                      -- JSON dict[str, str]
    row_value_samples TEXT,                    -- JSON dict[str, list[str]]
    document_id     VARCHAR NOT NULL,
    document_name   VARCHAR,
    row_count       INTEGER,
    table_metadata  TEXT,                      -- JSON {access_tags, csv_stats, ...}
    created_at      DATETIME,
    updated_at      DATETIME,

    UNIQUE (tenant_id, schema_name, table_name)   -- tenant-scoped
);

CREATE INDEX ix_dtr_tenant_id    ON document_table_registry(tenant_id);
CREATE INDEX ix_dtr_document_id  ON document_table_registry(document_id);
CREATE INDEX ix_dtr_source_format ON document_table_registry(source_format);
```

### 8.2 New schema: `document_tables`

```sql
CREATE SCHEMA document_tables;
-- Tables named per §6.10 naming contract
```

### 8.3 Alembic migration: `009_document_table_registry.py`

- Creates `document_table_registry`
- Creates `document_tables` schema
- Does NOT drop `excel_table_registry` (dual-registry compat until v2.0)
- Adds `doc_tables_created` (Integer), `doc_table_names` (Text/JSON) to `documents` table

---

## 9. Configuration

New settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `structured_table_extraction_enabled` | `True` | Extract tables from PDFs/images/DOCX |
| `table_extraction_min_rows` | `2` | Skip tables with fewer rows |
| `table_extraction_min_cols` | `2` | Skip tables with fewer columns |
| `table_extraction_max_rows` | `10000` | Truncate tables larger than this |
| `table_row_sample_max_per_col` | `50` | Max unique values sampled per column (registry) |
| `table_row_sample_cardinality_limit` | `200` | Skip columns with more distinct values |
| `table_max_signals_total` | `500` | Max total routing signals per table |
| `table_schema_name` | `document_tables` | PostgreSQL schema for non-Excel tables |
| `nl2sql_max_prompt_columns` | `20` | Max columns in NL2SQL prompt |
| `nl2sql_max_samples_per_column` | `20` | Max sample values per column in prompt |
| `nl2sql_sample_token_budget` | `2000` | Total token budget for samples in prompt |
| `nl2sql_max_prompt_tokens` | `8000` | Hard cap on total NL2SQL prompt size |
| `csv_max_file_size_mb` | `100` | Max CSV upload size |
| `table_lazy_register` | `False` | Defer startup registration for large registries |
| `table_lazy_register_lookback_days` | `7` | Days lookback for lazy registration |

---

## 10. Backward Compatibility

| Concern | Approach |
|---------|----------|
| Existing `excel_table_registry` rows | Read at startup with lower precedence; `document_table_registry` wins on conflict |
| Existing `excel_tables` schema | Unchanged; SQL templates continue to reference `excel_tables."<table>"` |
| `ExcelTablesService` callers | Thin wrapper delegates to `TableRegistrationService` |
| Hardcoded `_PL_TRIGGER_PHRASES` | Removed from new service; P&L routing via row value column signals |
| Re-ingestion for row samples | Not required; existing tables get empty `row_value_samples` until re-uploaded |
| `excel_table_registry` sunset | v2.0 upgrade script migrates remaining rows; reads removed in v2.0 |

---

## 11. Implementation Phases

### Phase 1 — Registry, Schema, Access Control (prerequisite)
- Alembic migration `009_document_table_registry.py`
- `TagPredicateValidator` + wire into `_execute_sql_route()` and `_execute_nl2sql_route()`
- Document model additions

### Phase 2 — Excel row sampling + signal canonicalization (quick win)
- `VERagPostgresStructuredDB`: row sampling + write to `document_table_registry`
- `_compute_column_signals()`: canonicalization pipeline + row value signals
- `TableRegistrationService` (reads both registries, dual-registry precedence)
- Update `main.py` startup
- Verify: re-ingest `PL_Statements_2015-2024.xlsx` → "income tax expense" routes to SQL

### Phase 3 — NL2SQL prompt enrichment
- `_execute_nl2sql_route()`: load `row_value_samples` from registry, token budget truncation
- Verify: NL2SQL generates `WHERE item = 'Income Tax Expense'` (exact match)

### Phase 4 — PDF/DOCX/Image table extraction
- `TableExtractionAdapter` (with blue/green swap, header heuristic, naming contract)
- `ProcessingService` integration
- Verify: ingest `Benefits_Guide_2025.pdf` → tables SQL-queryable

### Phase 5 — CSV support
- `CsvProcessingService` (parser contract, both structured + vector paths)
- Verify: ingest CSV → SQL-queryable + discoverable via RAG

### Phase 6 — Observability and operational tooling
- Metrics instrumentation (§12)
- Admin orphan cleanup endpoints
- Lazy registration mode

---

## 12. Observability

### Required Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `table_extraction_total` | Counter | Tables extracted, labels: `source_format`, `status` (success/skip/fail) |
| `table_extraction_failure_reasons` | Counter | Labels: `reason` (min_rows, min_cols, header_fail, db_error, ...) |
| `table_registration_total` | Counter | Templates registered at startup, labels: `source_format` |
| `table_registration_duration_seconds` | Histogram | Startup registration time |
| `sql_route_hit_total` | Counter | SQL route hits, labels: `template_name`, `source_format` |
| `rag_fallback_total` | Counter | Queries that fell back to RAG after SQL attempt |
| `nl2sql_generation_duration_seconds` | Histogram | p50/p95/p99 NL2SQL prompt → SQL latency |
| `tag_predicate_denied_total` | Counter | Access denied at execution time, labels: `template_name` |
| `orphaned_tables_count` | Gauge | Tables in schema with no registry row |
| `dual_registry_conflicts_total` | Counter | Conflicts resolved during startup |
| `csv_bad_row_rate` | Histogram | `bad_rows / total_rows` per CSV ingest |

### SLOs

| SLO | Target |
|-----|--------|
| Startup registration (≤ 100 tables) | < 5 seconds |
| Row sampling per table at ingest | < 2 seconds |
| Table extraction per PDF table | < 5 seconds |
| NL2SQL generation p95 | < 3 seconds |
| SQL route execution p95 | < 500ms |

---

## 13. Acceptance Criteria

All criteria are bound to named fixtures from `test_data/` with expected outputs and timing budgets.

### Routing Correctness

- [ ] Fixture: `PL_Statements_2015-2024.xlsx` — Query: "What was the income tax expense in 2019?" → route=SQL, template=`excel_pl_statements_*`, confidence ≥ 0.70, response time ≤ 200ms
- [ ] Fixture: `AR_Aging_Report_Dec2024.xlsx` — Query: "How much does [customer in file] owe?" → route=SQL, template contains `ar_aging`, confidence ≥ 0.70
- [ ] Fixture: `Budget_2025.xlsx` — Query: "What is the marketing budget?" → route=SQL, not RAG
- [ ] Fixture: `Benefits_Guide_2025.pdf` (contains benefit tables) — any benefit table row label query → route=SQL after ingest

### Signal Quality

- [ ] After ingesting `PL_Statements_2015-2024.xlsx`, `row_value_samples["item"]` contains `"income tax expense"` (canonicalized)
- [ ] High-cardinality column (> 200 unique values) is absent from `column_signals`
- [ ] Total signals per table ≤ 500 (verified by unit test against large fixture)
- [ ] Canonicalization: `"  Income-Tax Expense\u00a0"` → `"income tax expense"` (unit test)

### Tenant Safety

- [ ] Two tenants uploading same-named file produce non-colliding registry rows (integration test)
- [ ] Tenant A cannot SQL-query Tenant B's table (returns 403, not data)
- [ ] `TagPredicateValidator` blocks execution for user missing required tag (unit test)

### Blue/Green Re-upload

- [ ] Re-upload of `PL_Statements_2015-2024.xlsx` completes with no query errors during swap (concurrent query test)
- [ ] Registry row reflects updated `row_count` and `updated_at` after re-upload
- [ ] No `_old` tables remain in schema after successful swap

### CSV Parser

- [ ] UTF-8 BOM CSV ingests without error; BOM absent from column names
- [ ] CSV with semicolon delimiter detected and parsed correctly
- [ ] CSV with > 5% bad rows: document status = `failed`, reason logged
- [ ] Quoted multiline field preserved as single value (unit test)

### Backward Compatibility

- [ ] All existing Excel SQL routing tests pass without re-ingestion
- [ ] `excel_table_registry` rows load and register at startup
- [ ] `ExcelTablesService` import resolves without error (delegation check)

### Performance

- [ ] Startup with 50 registered tables completes in < 5 seconds
- [ ] Row sampling for a 1000-row, 10-column table adds < 2 seconds to ingest
- [ ] NL2SQL prompt construction with max settings (20 cols × 20 samples) < 100ms

---

## 14. Risk Flags

### P0: TENANT_COLLISION
Registry uniqueness was not tenant-scoped in v1.0.
**Resolution:** `UNIQUE(tenant_id, schema_name, table_name)` + tenant-scoped queries everywhere (§6.1).

### P0: ATOMIC_REUPLOAD
"Drop and recreate" creates transient unavailability.
**Resolution:** Blue/green swap with transactional registry pointer update (§6.2).

### P0: SQL_ACCESS_CONTROL
Template gating is not a security boundary.
**Resolution:** `TagPredicateValidator` at execution time (§6.3, §7.3).

### P0: CSV_RELIABILITY
CSV parser behavior was undefined.
**Resolution:** Normative parser contract with delimiter detection, BOM, bad-row policy (§6.5).

### P1: PROMPT_OVERFLOW
Unbounded row samples in NL2SQL prompt.
**Resolution:** Hard token budget with deterministic truncation order (§6.6).

### P1: STARTUP_SCALE
Full-scan startup unbounded for large registries.
**Resolution:** Tier-based SLOs + `--lazy-register` escape hatch (§6.8).

### P1: SIGNAL_NOISE
Noisy/un-normalized signals degrade routing.
**Resolution:** Canonicalization pipeline with stopword filter (§6.4).

### P1: SCHEMA_GROWTH
Table-count in `document_tables` grows unbounded.
**Resolution:** Orphan cleanup job, retention policy on delete (§6.7).

### P1: DUAL_REGISTRY_CONFLICT
Same table in both registries with divergent metadata.
**Resolution:** `document_table_registry` wins; conflict logged (§6.9).

### P2: NAMING_COLLISION
Truncation can produce non-unique identifiers.
**Resolution:** Deterministic `slug[:40] + sha256[:8]` contract (§6.10).

### P2: TABLE_NAMING_UNICODE
Non-ASCII filenames produce unsafe identifiers.
**Resolution:** NFKC normalize + `[^a-z0-9]` → `_` in slugify (§6.10).

### P2: DOCLING_TABLE_QUALITY
Low-quality scans produce malformed tables.
**Resolution:** Min row/col gates + header heuristic (§7.1). Log quality metrics per extraction.

---

## 15. Open Questions

All questions resolved. See decision log in earlier spec versions.

---

## 16. Files Affected

| File | Change |
|------|--------|
| `alembic/versions/009_document_table_registry.py` | **NEW** — migration |
| `ai_ready_rag/db/models/document.py` | ADD `doc_tables_created`, `doc_table_names` |
| `ai_ready_rag/services/table_extraction_adapter.py` | **NEW** — Docling table → PostgreSQL |
| `ai_ready_rag/services/table_registration_service.py` | **NEW** — replaces ExcelTablesService |
| `ai_ready_rag/services/sql_access_control.py` | **NEW** — TagPredicateValidator |
| `ai_ready_rag/services/excel_tables_service.py` | DEPRECATE — thin wrapper |
| `ai_ready_rag/services/ingestkit_adapters.py` | ADD row sampling + blue/green + new registry |
| `ai_ready_rag/services/processing_service.py` | CALL TableExtractionAdapter after Docling |
| `ai_ready_rag/services/csv_processing_service.py` | **NEW** — CSV both-path ingest |
| `ai_ready_rag/services/rag_service.py` | ADD TagPredicateValidator + row_value_samples in prompt |
| `ai_ready_rag/config.py` | ADD 15 new settings |
| `ai_ready_rag/main.py` | SWITCH startup to TableRegistrationService |

---

*Next: Run `/spec-review specs/UNIFIED_TABLE_INGEST_v1.md` to generate GitHub issues.*
