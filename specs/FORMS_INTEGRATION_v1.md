# Forms Processing Integration (ingestkit-forms)

| Field | Value |
|-------|-------|
| **Status** | DRAFT |
| **Version** | 0.1 |
| **Created** | 2026-02-16 |
| **Type** | Backend Service + Integration |
| **Owner** | -- |
| **Complexity** | MODERATE |
| **Related Specs** | DOCUMENT_MANAGEMENT.md, ENV_PROFILE_PIPELINE.md |

## Summary

Integrate the `ingestkit-forms` package into VE-RAG-System to enable template-based extraction of structured data from standardized PDF forms (e.g., ACORD insurance certificates, government forms, HR onboarding packets). Documents matching a registered template are routed through the forms pipeline for **dual-write** -- structured fields to SQLite and semantic chunks to Qdrant -- enabling both exact lookups and semantic search over form data.

---

## Goals

- Route PDF uploads through `ingestkit-forms` when a registered template matches the document layout
- Extract structured field values (policy numbers, coverage amounts, dates, names) into a queryable SQLite database
- Dual-write: structured fields to SQL + embedded chunks to Qdrant for semantic search
- Graceful fallback to the standard Docling/SimpleChunker pipeline when no template matches
- Follow the established Excel integration pattern (`ExcelProcessingService` + adapters) for consistency
- Support template registration via CLI script and admin API endpoint
- Track form extraction metadata on the Document model for audit and lifecycle management

## Non-Goals

- Building a visual template editor UI (future work)
- Auto-generating templates from unregistered forms (requires ML, future work)
- Real-time form submission processing (this is batch/upload-driven)
- OCR backend implementation inside VE-RAG (OCR backends live in ingestkit-forms)
- Supporting non-PDF form formats in v1 (Excel forms already handled by ingestkit-excel)

---

## Background

### What is Template-Based Form Extraction?

Standardized forms (ACORD certificates, W-9s, I-9s, building permits) share a fixed layout. The same fields appear at the same positions on every copy of the form. Template-based extraction exploits this:

1. **Register once**: Admin provides a sample form, defines field positions and names
2. **Auto-detect**: When a PDF is uploaded, the system computes a layout fingerprint and compares it against registered templates
3. **Extract**: If a template matches (confidence >= threshold), the system extracts field values from known positions
4. **Dual-write**: Extracted fields go to both SQL (exact queries) and Qdrant (semantic search)

### Why Not Just Use Docling?

Docling produces generic text chunks optimized for semantic search. For standardized forms, this means:

| Capability | Docling (current) | Forms Pipeline (proposed) |
|------------|-------------------|---------------------------|
| "What is the policy number?" | Semantic search across chunks; hopes the right chunk is retrieved | SQL query: `SELECT policy_number FROM form_extractions WHERE document_id = ?` |
| "Show all policies expiring before March" | Not possible (no structured fields) | `SELECT * FROM ... WHERE expiration_date < '2026-03-01'` |
| "Does this policy include earthquake coverage?" | Works (semantic search) | Works (semantic search) + SQL confirmation |
| Accuracy for known fields | Variable (depends on chunk boundaries) | High (field position is known from template) |

The forms pipeline complements Docling -- it handles the structured extraction that Docling cannot do, while Docling handles free-text documents that don't match any template.

---

## Architecture

### Data Flow

```
PDF Upload
  |
  v
ProcessingService.process_document()
  |
  +-- Is PDF AND use_ingestkit_forms=True?
  |     |
  |     v
  |   FormsProcessingService.process_form()
  |     |
  |     +-- FormRouter.try_match(file_path)
  |     |     |
  |     |     +-- Match found (confidence >= 0.8)
  |     |     |     |
  |     |     |     v
  |     |     |   FormRouter.extract_form(request)
  |     |     |     |
  |     |     |     +-- Extract fields (native/OCR/VLM)
  |     |     |     +-- Dual-write: SQL + Qdrant
  |     |     |     +-- Return FormProcessingResult
  |     |     |     |
  |     |     |     v
  |     |     |   Update Document model (forms_* fields)
  |     |     |   Return ProcessingResult(success=True)
  |     |     |
  |     |     +-- No match found
  |     |           |
  |     |           v
  |     |         Return (None, should_fallback=True)
  |     |
  |     +-- Fallback or hard error handling
  |
  +-- Is XLSX AND use_ingestkit_excel=True?  (existing)
  |     ...
  |
  +-- Standard chunker pipeline (Docling/Simple)
        ...
```

### Component Diagram

```
VE-RAG-System                          ingestkit-forms
+-----------------------------------+  +--------------------------------+
| ProcessingService                 |  | FormRouter                     |
|   +-- FormsProcessingService      |->|   +-- FormMatcher              |
|       +-- VERagFormVectorAdapter  |  |   +-- NativePDFExtractor       |
|       +-- VERagFormDBAdapter      |  |   +-- OCROverlayExtractor      |
|       +-- OllamaEmbedding (reuse)|  |   +-- VLMFallbackExtractor     |
|       +-- OllamaVLM (optional)   |  |   +-- DualWriter               |
+-----------------------------------+  +--------------------------------+
                                       |
+-----------------------------------+  | FileSystemTemplateStore
| Template Registration             |  |   +-- /data/form_templates/
|   +-- CLI script (deployment)     |->|       +-- tmpl_acord24/
|   +-- Admin API (runtime)         |  |       +-- tmpl_acord25/
+-----------------------------------+  +--------------------------------+
```

---

## Implementation

### 1. Document Model Extensions

Add form-specific columns to the `Document` model, following the Excel pattern (`excel_*` columns):

```python
# db/models/document.py -- new nullable columns
forms_template_id = Column(String, nullable=True)
forms_template_name = Column(String, nullable=True)
forms_template_version = Column(Integer, nullable=True)
forms_extraction_method = Column(String, nullable=True)   # "native_fields" | "ocr_overlay" | "vlm_fallback"
forms_ingest_key = Column(String, nullable=True)           # SHA-256 idempotency key
forms_match_confidence = Column(Float, nullable=True)      # Template match score (0.0-1.0)
forms_overall_confidence = Column(Float, nullable=True)    # Extraction confidence (0.0-1.0)
forms_tables_created = Column(Integer, nullable=True)      # Count of DB tables written
forms_db_table_names = Column(Text, nullable=True)         # JSON list of table names
forms_field_count = Column(Integer, nullable=True)         # Total fields extracted
forms_failed_field_count = Column(Integer, nullable=True)  # Fields with None values
forms_pages_processed = Column(Integer, nullable=True)
forms_match_method = Column(String, nullable=True)         # "auto_detect" | "manual_override"
```

**Migration**: 13 `ALTER TABLE documents ADD COLUMN` statements. All nullable, no default values, no index requirements.

### 2. Configuration

Add to `Settings` class in `config.py`:

```python
# Feature flag
use_ingestkit_forms: bool = False

# Template matching
forms_match_confidence_threshold: float = 0.8

# OCR settings
forms_ocr_engine: str = "tesseract"     # "tesseract" or "paddleocr"
forms_ocr_language: str = "en"

# VLM fallback (optional, requires GPU)
forms_vlm_enabled: bool = False
forms_vlm_model: str = "qwen2.5-vl:7b"
forms_vlm_fallback_threshold: float = 0.4

# Extraction confidence
forms_extraction_min_field_confidence: float = 0.5
forms_extraction_min_overall_confidence: float = 0.3

# Storage
forms_template_storage_path: str = "./data/form_templates"
forms_db_path: str = "./data/forms_data.db"
forms_db_table_prefix: str = "form_"
forms_enable_dual_write: bool = True
```

Profile defaults:

| Setting | Laptop | Spark |
|---------|--------|-------|
| `use_ingestkit_forms` | `False` | `False` (opt-in) |
| `forms_ocr_engine` | `"tesseract"` | `"paddleocr"` |
| `forms_vlm_enabled` | `False` | `True` |

### 3. Adapter Layer

Create `services/forms_adapters.py` with factory functions that bridge ingestkit-forms protocols to VE-RAG backends:

| Adapter | ingestkit Protocol | VE-RAG Backend |
|---------|-------------------|----------------|
| `VERagFormVectorAdapter` | `VectorStoreBackend` | Qdrant (sync client) |
| `VERagFormDBAdapter` | `FormDBBackend` | SQLite (`forms_data.db`) |
| `OllamaEmbedding` | `EmbeddingBackend` | Reuse from `ingestkit_adapters.py` |
| `OllamaVLM` | `VLMBackend` | Ollama (optional) |
| `create_form_template_store()` | `FormTemplateStore` | `FileSystemTemplateStore` |
| `create_form_fingerprinter()` | `LayoutFingerprinter` | PyMuPDF-based rendering |
| `create_form_pdf_widget_backend()` | `PDFWidgetBackend` | PyMuPDF widget extraction |
| `create_form_ocr_backend()` | `OCRBackend` | Tesseract CLI or PaddleOCR |

**Vector payload schema** (Qdrant point payload):

```python
{
    # Standard VE-RAG fields
    "chunk_id": "doc-uuid:0",
    "document_id": "doc-uuid",
    "document_name": "certificate.pdf",
    "chunk_index": 0,
    "chunk_text": "Policy Number is ACPBP023038243547. ...",
    "tags": ["insurance", "compliance"],
    "tenant_id": "default",
    "uploaded_by": "user-uuid",
    "uploaded_at": "2026-02-16T...",
    "page_number": 1,

    # Form provenance fields
    "ingestkit_forms_template_id": "tmpl_acord24",
    "ingestkit_forms_template_name": "ACORD 24 - Certificate of Property Insurance",
    "ingestkit_forms_extraction_method": "native_fields",
    "ingestkit_forms_overall_confidence": 0.95,
    "ingestkit_forms_field_names": ["policy_number", "building_coverage", "insured_name"],
    "ingestkit_forms_match_method": "auto_detect",
    "ingestkit_forms_ingest_key": "sha256...",
}
```

### 4. Forms Processing Service

Create `services/forms_processing_service.py` following the `ExcelProcessingService` pattern:

```python
class FormsProcessingService:
    """Processes PDF forms through ingestkit-forms pipeline.

    Parallel structure to ExcelProcessingService:
    - Builds FormProcessorConfig from VE-RAG Settings
    - Creates all backends via dependency injection
    - Delegates to FormRouter.extract_form()
    - Updates Document model with forms_* metadata
    - Returns (ProcessingResult, should_fallback) tuple
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def process_form(
        self,
        document: Document,
        db: Session,
        template_id: str | None = None,
    ) -> tuple[ProcessingResult | None, bool]:
        ...
```

**Fallback logic** (mirrors Excel):

| Condition | Behavior |
|-----------|----------|
| No template match (`extract_form()` returns `None`) | `should_fallback=True` -- fall through to Docling |
| Low confidence error in `result.errors` | `should_fallback=True` -- fall through to Docling |
| Hard error (corrupt PDF, template invalid) | Return `ProcessingResult(success=False)`, no fallback |
| Success | Return `ProcessingResult(success=True)`, update Document |

### 5. Processing Service Routing

Modify `ProcessingService.process_document()` to add forms routing before the existing Excel check:

```python
# In processing_service.py

# 1. Try forms pipeline first (PDF only)
if file_ext == ".pdf" and self._should_use_ingestkit_forms():
    result, should_fallback = await self._process_with_ingestkit_forms(document, db)
    if not should_fallback and result is not None:
        return result
    logger.info("Forms pipeline: no match or fallback for %s", document.id)

# 2. Try Excel pipeline (existing, unchanged)
if file_ext == ".xlsx" and self._should_use_ingestkit():
    ...

# 3. Standard chunker pipeline (existing, unchanged)
...
```

### 6. Document Lifecycle

Extend `DocumentService.delete_document()` to clean up forms tables:

```python
# When deleting a document with forms_db_table_names:
# 1. Parse JSON list of table names
# 2. DROP TABLE each from forms_data.db
# 3. Delete vectors from Qdrant (already handled by existing code)
```

---

## Template Registration

### CLI Script (Deployment-Time)

Create `scripts/register_form_template.py` for one-time template setup:

```python
"""Register a form template from a sample PDF.

Usage:
    python -m scripts.register_form_template \
        --sample-pdf samples/acord24_sample.pdf \
        --template-id tmpl_acord24 \
        --name "ACORD 24 - Certificate of Property Insurance" \
        --fields-json templates/acord24_fields.json
"""
```

**Field definition file** (`templates/acord24_fields.json`):

```json
[
    {
        "field_name": "policy_number",
        "field_label": "Policy Number",
        "field_type": "text",
        "page_number": 0,
        "region": {"x": 0.35, "y": 0.42, "width": 0.30, "height": 0.02}
    },
    {
        "field_name": "insured_name",
        "field_label": "Named Insured",
        "field_type": "text",
        "page_number": 0,
        "region": {"x": 0.50, "y": 0.15, "width": 0.45, "height": 0.04},
        "required": true
    },
    {
        "field_name": "effective_date",
        "field_label": "Effective Date",
        "field_type": "date",
        "page_number": 0,
        "region": {"x": 0.35, "y": 0.48, "width": 0.12, "height": 0.02},
        "validation_pattern": "^\\d{2}/\\d{2}/\\d{4}$"
    },
    {
        "field_name": "building_coverage",
        "field_label": "Building Coverage Amount",
        "field_type": "number",
        "page_number": 0,
        "region": {"x": 0.60, "y": 0.55, "width": 0.15, "height": 0.02}
    }
]
```

The script:

1. Renders the sample PDF to images (PyMuPDF, 150 DPI)
2. Computes a layout fingerprint (20x16 grid, Otsu thresholding)
3. Loads field definitions from JSON
4. Creates a `FormTemplate` model
5. Saves to `FileSystemTemplateStore` at `forms_template_storage_path`

### Admin API Endpoint (Runtime)

Add to `api/admin.py`:

```
POST /api/admin/form-templates
    Body: multipart/form-data
    - sample_pdf: file (required)
    - name: str (required)
    - template_id: str (optional, auto-generated if omitted)
    - fields: JSON string (required)
    Response: { template_id, name, field_count, fingerprint_size }

GET /api/admin/form-templates
    Response: [ { template_id, name, version, field_count, created_at } ]

GET /api/admin/form-templates/{template_id}
    Response: { template_id, name, version, fields, fingerprint_hex, ... }

DELETE /api/admin/form-templates/{template_id}
    Response: { deleted: true }
```

**Access control**: Admin-only (existing `require_admin` dependency).

---

## Storage Layout

```
data/
  form_templates/                    # FileSystemTemplateStore root
    tmpl_acord24/
      v1.json                        # Template definition + fingerprint
      _meta.json                     # Version/deletion metadata
    tmpl_acord25/
      v1.json
      _meta.json
  forms_data.db                      # SQLite: extracted form field values
  excel_tables.db                    # Existing: Excel structured data
  uploads/                           # Existing: uploaded files
```

**forms_data.db schema** (auto-created by ingestkit-forms):

```sql
-- One table per template, prefixed with forms_db_table_prefix
CREATE TABLE form_tmpl_acord24 (
    id TEXT PRIMARY KEY,              -- form extraction ID
    document_id TEXT,                  -- VE-RAG document reference
    ingest_key TEXT,                   -- idempotency key
    policy_number TEXT,
    insured_name TEXT,
    effective_date TEXT,
    expiration_date TEXT,
    building_coverage REAL,
    ...
    extracted_at TIMESTAMP,
    overall_confidence REAL
);
```

---

## Query Capabilities

Once forms are extracted, the RAG system gains two query paths:

### Semantic Search (Qdrant -- existing)

Unchanged. Form chunks are embedded and searchable like any other document:

> User: "Does the Cervantes policy include earthquake coverage?"
> RAG: Searches Qdrant, finds chunk with earthquake deductible info, generates answer with citation.

### Structured Query (SQL -- new)

The `forms_data.db` enables exact-match queries that semantic search cannot reliably answer:

> "What is the policy number for Cervantes Villas?"
> `SELECT policy_number FROM form_tmpl_acord24 WHERE insured_name LIKE '%Cervantes%'`

> "Show all policies expiring before March 2026"
> `SELECT insured_name, policy_number, expiration_date FROM form_tmpl_acord24 WHERE expiration_date < '2026-03-01'`

> "Total building coverage across all certificates"
> `SELECT SUM(building_coverage) FROM form_tmpl_acord24`

**Integration with RAG**: The RAG service can be extended (future spec) to detect structured queries and route them to SQL before falling back to vector search.

---

## Error Handling

| Error | Source | Behavior |
|-------|--------|----------|
| No template match | `FormRouter.extract_form()` returns `None` | Fallback to Docling chunker |
| Low extraction confidence | `result.errors` contains confidence error | Fallback to Docling chunker |
| Template not found (manual override) | `FormTemplateStore.get_template()` returns `None` | Return `ProcessingResult(success=False)` |
| Corrupt/encrypted PDF | PyMuPDF or OCR failure | Return `ProcessingResult(success=False)` |
| OCR backend unavailable | `OCRBackend` raises connection error | Log warning, attempt native-only extraction |
| VLM backend unavailable | `VLMBackend` raises connection error | Log warning, skip VLM fallback |
| Forms DB write failure | SQLite error | Log error; if `dual_write_mode="best_effort"`, continue with vector-only |

---

## Dependencies

### Required (already installed in VE-RAG)

| Package | Version | Purpose |
|---------|---------|---------|
| `ingestkit-forms` | Latest (editable) | Form router, matcher, extractors |
| `ingestkit-core` | Latest (editable) | Shared models and protocols |
| `PyMuPDF` (fitz) | >= 1.24 | PDF rendering, widget extraction |
| `Pillow` | >= 10.0 | Image processing for fingerprinting |

### Optional

| Package | Version | Purpose |
|---------|---------|---------|
| `paddleocr` | >= 2.7 | OCR backend (Spark profile) |
| `tesseract` | System pkg | OCR backend (Laptop profile) |

### Not Required

- No new Python packages to install
- No new external services (uses existing Qdrant, Ollama, SQLite)

---

## Phased Rollout

### Phase 1: Core Pipeline (MVP)

- [ ] Document model migration (13 columns)
- [ ] Configuration settings (`use_ingestkit_forms` + related)
- [ ] `FormsProcessingService` with native PDF extraction only
- [ ] `VERagFormVectorAdapter` and `VERagFormDBAdapter`
- [ ] Processing service routing (PDF check before Excel check)
- [ ] Document lifecycle cleanup (delete form tables)
- [ ] CLI registration script
- [ ] Unit tests (mock FormRouter, test routing/fallback/cleanup)

### Phase 2: Template Management

- [ ] Admin API endpoints (CRUD for templates)
- [ ] Template list view in admin dashboard
- [ ] Manual template_id override on upload (optional query param)

### Phase 3: OCR + VLM Backends

- [ ] OCR adapter (Tesseract CLI for laptop, PaddleOCR for Spark)
- [ ] VLM adapter (Ollama with vision model)
- [ ] Integration tests with real ACORD certificates
- [ ] Confidence threshold tuning

### Phase 4: RAG Integration

- [ ] Structured query detection in RAG service (SQL vs vector routing)
- [ ] Combined results (SQL + semantic for form documents)
- [ ] Citation format for SQL-sourced answers

---

## Acceptance Criteria

- [ ] PDF uploads are routed through forms pipeline when `use_ingestkit_forms=True` and a template matches
- [ ] Extracted fields are written to `forms_data.db` with correct values
- [ ] Embedded chunks are written to Qdrant with `ingestkit_forms_*` payload fields
- [ ] No template match gracefully falls back to Docling/SimpleChunker pipeline
- [ ] Document model tracks all `forms_*` metadata fields
- [ ] Document deletion cleans up form tables from `forms_data.db`
- [ ] CLI script successfully registers an ACORD 24 template from a sample PDF
- [ ] Re-uploading the same PDF produces identical `forms_ingest_key` (idempotency)
- [ ] Feature is fully disabled when `use_ingestkit_forms=False` (no import errors, no side effects)
- [ ] All new code has unit tests with mock FormRouter (no external services required)

---

## Open Questions

1. **SQL query routing in RAG**: Should the RAG service auto-detect structured queries and route to SQL, or should this be a separate endpoint? (Deferred to Phase 4 spec)
2. **Multi-tenant template isolation**: Should templates be tenant-scoped or global? Current design uses `tenant_id` on templates but `FileSystemTemplateStore` stores globally.
3. **Template versioning UX**: When a form layout changes (e.g., ACORD updates their template), how should the admin register a v2? Auto-detect layout drift vs manual re-registration?
4. **Batch registration**: For customers with 50+ form types, should there be a bulk import from a directory of samples + field definitions?

---

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-16 | -- | Initial draft |
