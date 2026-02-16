# INGESTKIT_PDF_FORMS_INTEGRATION_v1.md

## Meta
- Status: DRAFT
- Version: 0.3
- Date: 2026-02-16
- Replaces: `FORMS_INTEGRATION_v1.md` (superseded — too complex)
- Companion spec: `ingestkit/specs/PLUGIN_API_SURFACE_v1.md`
- Changelog:
  - v0.3: Align with ingestkit-forms issues #101-#107. Fix FormIngestRequest construction
    (don't pass template_id from auto-match — triggers manual override path). Wire OCR/PDF/VLM
    backend adapters into router. Use ingestkit-forms' validate_table_name() instead of duplicating.
    Add tenant_id to try_match call. Update log event names to match library. Document new
    TemplateMatch fields (source_format, page_alignment).
  - v0.2: Address eng review — add data security model, consistency contract, template lifecycle,
    identifier hardening, error taxonomy, routing clarity, idempotency, authz matrix, observability,
    test plan, path safety. Fix config ordering bug. Remove PDF pipeline flags (out of scope).

## 1. Overview

Integrate ingestkit-forms into VE-RAG-System following the proven ingestkit-excel pattern. The integration is optional, feature-flagged, and falls back to the existing Docling/SimpleChunker pipeline when disabled or when processing fails.

**Design principles:**
- Mirror the Excel integration pattern (adapter, fallback tuple, config, document columns).
- Fail-closed: inconclusive results return to the standard pipeline, never guess.
- Non-sensitive MVP: extracted field *values* are stored in `forms_data.db` and vector payloads. High-risk fields are redacted per the data handling policy (Section 14).

**Scope for v1:** ingestkit-forms only. ingestkit-pdf integration is deferred — no `use_ingestkit_pdf` flag, no PDF routing, no PDF document columns. Those will be specified in a separate `INGESTKIT_PDF_INTEGRATION_v1.md` when the package is ready.

## 2. Feature Flags

Add to `ai_ready_rag/config.py` Settings class (after existing Excel block, lines ~235):

```python
# ingestkit-forms integration
use_ingestkit_forms: bool = False
forms_match_confidence_threshold: float = 0.8
forms_ocr_engine: str = "tesseract"
forms_vlm_enabled: bool = False
forms_vlm_model: str = "qwen2.5-vl:7b"
forms_db_path: str = "./data/forms_data.db"
forms_template_storage_path: str = "./data/form_templates"
forms_redact_high_risk_fields: bool = True   # Redact SSN, tax ID, account numbers
forms_template_require_approval: bool = True # Templates must be approved before matching
```

**Profile defaults:**
- Laptop: `use_ingestkit_forms: False`
- Spark: `use_ingestkit_forms: True`, `forms_ocr_engine: "paddleocr"`, `forms_vlm_enabled: True`

## 3. Processing Route Order

Update `processing_service.py` routing (currently: Excel check -> standard chunker). New order:

```
PDF upload (.pdf)
  |-- 1. Forms check (use_ingestkit_forms=True)
  |     +-- FormsProcessingService.process_form(document, db)
  |         |-- Internally calls router.try_match()
  |         |-- Match found -> extract -> Return (ProcessingResult, should_fallback=False)
  |         +-- No match   -> Return (None, should_fallback=True)
  |
  +-- 2. Standard chunker (Docling/SimpleChunker)
         +-- Existing pipeline

XLSX upload (.xlsx)
  |-- 1. Excel check (use_ingestkit_excel=True) -- existing
  +-- 2. Standard chunker -- existing

Other formats -> Standard chunker directly
```

**Routing ownership (fixes #6):** `ProcessingService` dispatches by file type + feature flag only. All matching logic (`try_match`, confidence gating) lives inside `FormsProcessingService.process_form()`. `ProcessingService` never calls `try_match` directly.

**Implementation in ProcessingService.process_document():**

```python
# After existing Excel routing block:
if file_ext == ".pdf" and self._should_use_ingestkit_forms():
    result, should_fallback = await self._process_with_ingestkit_forms(document, db)
    if not should_fallback and result is not None:
        return result
    logger.info("forms.routing.fallback", document_id=document.id)

# Existing standard chunker pipeline continues below
```

## 4. Adapter Reuse

### 4.1 Existing Adapters (reuse as-is from `ingestkit_adapters.py`)

| Adapter | Protocol | Used By |
|---------|----------|---------|
| `VERagVectorStoreAdapter` | `VectorStoreBackend` | Excel, Forms |
| `create_embedding_adapter()` | Returns `OllamaEmbedding` | Excel, Forms |
| `create_llm_adapter()` | Returns `OllamaLLM` | Excel |
| `create_structured_db()` | Returns `SQLiteStructuredDB` | Excel |

**Key:** VERagVectorStoreAdapter already handles merging VE-RAG payload fields (document_id, tags, tenant_id, uploaded_by) with ingestkit provenance fields (ingestkit_*). It works for forms too — forms just add additional metadata fields.

### 4.2 New Adapter: VERagFormDBAdapter

One new adapter needed for the `FormDBBackend` protocol:

```python
from ingestkit_forms import validate_table_name

class VERagFormDBAdapter:
    """Thin SQLite wrapper implementing ingestkit_forms.FormDBBackend protocol."""

    def __init__(self, db_path: str) -> None:
        self._db_path = self._validate_path(db_path)
        self._ensure_db()

    @staticmethod
    def _validate_path(db_path: str) -> str:
        """Normalize and validate DB path is under data/ directory."""
        resolved = Path(db_path).resolve()
        allowed_parent = Path("./data").resolve()
        if not str(resolved).startswith(str(allowed_parent)):
            raise ValueError(f"forms_db_path must be under data/: {db_path}")
        return str(resolved)

    @staticmethod
    def check_table_name(name: str) -> None:
        """Validate table name using ingestkit-forms' validate_table_name.

        Raises ValueError if the name is not a safe SQL identifier.
        Uses ingestkit-forms' canonical regex (^[a-zA-Z_][a-zA-Z0-9_]{0,127}$)
        to avoid duplication and drift.
        """
        error = validate_table_name(name)
        if error is not None:
            raise ValueError(f"Unsafe table identifier rejected: {name!r} — {error}")

    def execute_sql(self, sql: str, params: tuple | None = None) -> None: ...
    def get_table_columns(self, table_name: str) -> list[str]: ...
    def delete_rows(self, table_name: str, column: str, values: list[str]) -> int: ...
    def table_exists(self, table_name: str) -> bool: ...
    def get_connection_uri(self) -> str: ...
```

**Location:** Add to `ai_ready_rag/services/ingestkit_adapters.py`

**DB file:** `settings.forms_db_path` (default `./data/forms_data.db`). Separate from main app DB and Excel tables DB. Path is validated at startup (Section 16).

### 4.3 New Adapter: VERagFormTemplateStore

Implements `FormTemplateStore` protocol. Two options:

**Option A (Recommended for MVP):** Use the bundled `FileSystemTemplateStore` from ingestkit-forms:
```python
from ingestkit_forms import FileSystemTemplateStore
store = FileSystemTemplateStore(base_path=settings.forms_template_storage_path)
```

**Option B (Future):** Custom SQLite-backed store in VE-RAG for better querying.

### 4.4 LayoutFingerprinter Adapter

Use ingestkit-forms' built-in fingerprinting utilities:
```python
from ingestkit_forms import compute_layout_fingerprint_from_file

class VERagLayoutFingerprinter:
    def __init__(self, config: FormProcessorConfig) -> None:
        self._config = config

    def compute_fingerprint(self, file_path: str) -> list[bytes]:
        return compute_layout_fingerprint_from_file(file_path, self._config)
```

### 4.5 OCR Backend Adapter

Wraps Tesseract or PaddleOCR to implement the `OCRBackend` protocol:

```python
class VERagOCRAdapter:
    """OCR adapter implementing ingestkit_forms.OCRBackend protocol.

    Delegates to Tesseract (default) or PaddleOCR based on settings.
    """

    def __init__(self, engine: str = "tesseract") -> None:
        self._engine = engine

    def ocr_region(
        self,
        image_bytes: bytes,
        language: str = "en",
        config: str | None = None,
        timeout: float | None = None,
    ) -> OCRRegionResult:
        """Run OCR on a cropped image region."""
        if self._engine == "tesseract":
            return self._tesseract_ocr(image_bytes, language, config)
        elif self._engine == "paddleocr":
            return self._paddleocr_ocr(image_bytes, language)
        raise ValueError(f"Unknown OCR engine: {self._engine}")

    def engine_name(self) -> str:
        return self._engine

    def _tesseract_ocr(self, image_bytes, language, config) -> OCRRegionResult: ...
    def _paddleocr_ocr(self, image_bytes, language) -> OCRRegionResult: ...
```

### 4.6 VLM Backend Adapter (Optional)

Wraps Ollama VLM endpoint for vision-language model fallback:

```python
class VERagVLMAdapter:
    """VLM adapter implementing ingestkit_forms.VLMBackend protocol.

    Calls Ollama API with a multimodal model (e.g., qwen2.5-vl:7b).
    Only created when forms_vlm_enabled=True.
    """

    def __init__(self, ollama_url: str, model: str = "qwen2.5-vl:7b") -> None:
        self._ollama_url = ollama_url
        self._model = model

    def extract_field(
        self,
        image_bytes: bytes,
        field_type: str,
        field_name: str,
        extraction_hint: str | None = None,
        timeout: float | None = None,
    ) -> VLMFieldResult: ...

    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        """Ping Ollama API to check VLM model is loaded."""
        ...
```

**Note:** `PDFWidgetBackend` adapter is deferred to v2 — requires either PyMuPDF (AGPL) or pdfplumber+pypdf. For v1, PDF forms use OCR extraction only. The router will log `W_FORM_NATIVE_FIELDS_UNAVAILABLE` at startup, which is expected.

## 5. FormsProcessingService

New file: `ai_ready_rag/services/forms_processing_service.py`

Mirror the `ExcelProcessingService` pattern with ordered writes and compensation (fixes #2, #7, #10):

```python
class FormsProcessingService:
    """Processes PDF/XLSX forms through ingestkit-forms extraction pipeline."""

    _FALLBACK_ERROR_CODES = {"E_FORM_NO_MATCH", "E_FORM_UNSUPPORTED_FORMAT"}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def process_form(
        self,
        document: Document,
        db: Session,
    ) -> tuple[ProcessingResult | None, bool]:
        """
        Returns:
            (ProcessingResult or None, should_fallback)

        Owns the full forms lifecycle: match -> extract -> write -> update document.
        ProcessingService only calls this method; never calls try_match directly.
        """
        file_path = Path(self.settings.storage_path) / document.id / document.filename

        # 1. Build config FIRST (fixes #7 — config before adapters)
        config = FormProcessorConfig(
            form_match_confidence_threshold=self.settings.forms_match_confidence_threshold,
            form_ocr_engine=self.settings.forms_ocr_engine,
            form_vlm_enabled=self.settings.forms_vlm_enabled,
            form_vlm_model=self.settings.forms_vlm_model,
            embedding_model=self.settings.embedding_model or "nomic-embed-text",
            embedding_dimension=self.settings.embedding_dimension,
            default_collection=self.settings.qdrant_collection,
            tenant_id=self.settings.default_tenant_id,
            form_template_storage_path=self.settings.forms_template_storage_path,
            redact_patterns=_HIGH_RISK_PATTERNS if self.settings.forms_redact_high_risk_fields else [],
        )

        # 2. Create adapters (config is available)
        vector_store = VERagVectorStoreAdapter(
            qdrant_url=self.settings.qdrant_url,
            collection_name=self.settings.qdrant_collection,
            embedding_dimension=self.settings.embedding_dimension,
            document_id=document.id,
            document_name=document.original_filename,
            tags=[t.name for t in document.tags],
            uploaded_by=document.uploaded_by or "system",
            tenant_id=self.settings.default_tenant_id,
        )
        embedder = create_embedding_adapter(
            ollama_url=self.settings.ollama_base_url,
            embedding_model=self.settings.embedding_model or "nomic-embed-text",
            embedding_dimension=self.settings.embedding_dimension,
        )
        form_db = VERagFormDBAdapter(db_path=self.settings.forms_db_path)
        template_store = FileSystemTemplateStore(
            base_path=self.settings.forms_template_storage_path,
        )
        fingerprinter = VERagLayoutFingerprinter(config)

        # OCR backend (required for scanned/flattened PDFs)
        ocr_backend = VERagOCRAdapter(engine=self.settings.forms_ocr_engine)

        # VLM backend (optional, for low-confidence field fallback)
        vlm_backend = None
        if self.settings.forms_vlm_enabled:
            vlm_backend = VERagVLMAdapter(
                ollama_url=self.settings.ollama_base_url,
                model=self.settings.forms_vlm_model,
            )

        # 3. Create router (with OCR + optional VLM backends)
        # Note: pdf_widget_backend deferred to v2 (requires PyMuPDF or pdfplumber).
        # Router logs W_FORM_NATIVE_FIELDS_UNAVAILABLE at startup — expected for v1.
        router = create_default_router(
            template_store=template_store,
            form_db=form_db,
            vector_store=vector_store,
            embedder=embedder,
            fingerprinter=fingerprinter,
            ocr_backend=ocr_backend,
            vlm_backend=vlm_backend,
            config=config,
        )

        # 4. Match (FormsProcessingService owns this decision)
        try:
            match = await asyncio.to_thread(
                router.try_match,
                str(file_path),
                tenant_id=self.settings.default_tenant_id,
            )
        except Exception as e:
            logger.warning("forms.match.error", document_id=document.id, error=str(e))
            return (None, True)  # Fallback on match failure

        if match is None:
            return (None, True)  # No match, fallback

        # 5. Extract + dual-write (atomic-ish via ingestkit's FormDualWriter)
        # IMPORTANT: Do NOT pass template_id from the auto-match result here.
        # extract_form does its own auto-matching internally. Passing template_id
        # would trigger the manual override path (router.py line 273), incorrectly
        # recording match_method="manual_override" instead of "auto_detect".
        request = FormIngestRequest(
            file_path=str(file_path),
            tenant_id=self.settings.default_tenant_id,
        )
        try:
            result = await asyncio.to_thread(router.extract_form, request)
        except Exception as e:
            logger.error("forms.extract.error", document_id=document.id, error=str(e))
            return (self._error_result(e), False)

        if result is None:
            return (None, True)

        # 6. Check for fallback error codes
        for err in result.error_details:
            if err.code.value in self._FALLBACK_ERROR_CODES:
                # Compensate: clean up any partial writes
                await self._compensate(result, form_db, vector_store)
                return (None, True)

        # 7. Update document columns (last step — after all writes succeeded)
        document.forms_template_id = result.extraction_result.template_id
        document.forms_template_name = result.extraction_result.template_name
        document.forms_template_version = result.extraction_result.template_version
        document.forms_overall_confidence = result.extraction_result.overall_confidence
        document.forms_extraction_method = result.extraction_result.extraction_method
        document.forms_match_method = result.extraction_result.match_method
        document.forms_ingest_key = result.ingest_key
        document.forms_db_table_names = json.dumps(result.tables)
        db.commit()

        return (ProcessingResult(
            success=len(result.errors) == 0,
            chunk_count=result.chunks_created,
            page_count=result.extraction_result.pages_processed,
            word_count=0,
            processing_time_ms=int(result.processing_time_seconds * 1000),
            error_message="; ".join(result.errors) if result.errors else None,
        ), False)

    async def _compensate(
        self,
        result: FormProcessingResult,
        form_db: VERagFormDBAdapter,
        vector_store: VERagVectorStoreAdapter,
    ) -> None:
        """Best-effort compensation: remove partial writes on fallback/failure."""
        # Delete vectors by ingest_key
        if result.written and result.written.vector_point_ids:
            try:
                vector_store.delete_by_filter(
                    self.settings.qdrant_collection,
                    "ingestkit_ingest_key",
                    result.ingest_key,
                )
            except Exception as e:
                logger.warning("forms.compensate.vectors_failed", error=str(e))

        # Drop created tables
        for table_name in result.tables:
            try:
                form_db.check_table_name(table_name)
                form_db.execute_sql(f"DROP TABLE IF EXISTS [{table_name}]")
            except Exception as e:
                logger.warning("forms.compensate.table_failed", table=table_name, error=str(e))
```

### 5.1 Idempotency Contract (fixes #10)

ingestkit-forms provides three-tier deterministic keying:

1. **Global ingest key:** `SHA256(content_hash | source_uri | parser_version | tenant_id)` — same file + same config = same key.
2. **Form extraction key:** `SHA256(ingest_key | template_id | template_version)` — changes when template is updated.
3. **Vector point ID:** `UUID5(NAMESPACE_URL, extraction_key:chunk_index)` — deterministic, enables upsert.

**Reprocessing behavior:** If the same PDF is uploaded again:
- `ingest_key` will match → `FormDualWriter` uses `INSERT OR REPLACE` for DB rows and `upsert` for vectors.
- No duplicate vectors or DB rows are created.
- Document columns are overwritten with latest extraction results.

**Constraint:** `forms_ingest_key` column has a non-unique index for efficient lookup during dedup checks.

## 6. Document Model Extensions

Add to `ai_ready_rag/db/models/document.py` (optional columns, all nullable):

```python
# ingestkit-forms fields
forms_template_id = Column(String, nullable=True)
forms_template_name = Column(String, nullable=True)
forms_template_version = Column(Integer, nullable=True)
forms_overall_confidence = Column(Float, nullable=True)
forms_extraction_method = Column(String, nullable=True)  # native_fields|ocr_overlay|cell_mapping
forms_match_method = Column(String, nullable=True)        # auto_detect|manual_override
forms_ingest_key = Column(String, nullable=True, index=True)
forms_db_table_names = Column(Text, nullable=True)        # JSON array of table names
```

### 6.1 Schema Migration (fixes #8)

Replace ad-hoc `ALTER TABLE ... except pass` with a tracked migration approach:

```python
# In db/database.py — schema_version table tracks applied migrations
_FORMS_MIGRATIONS = [
    (
        "forms_v1_columns",
        [
            "ALTER TABLE documents ADD COLUMN forms_template_id VARCHAR",
            "ALTER TABLE documents ADD COLUMN forms_template_name VARCHAR",
            "ALTER TABLE documents ADD COLUMN forms_template_version INTEGER",
            "ALTER TABLE documents ADD COLUMN forms_overall_confidence REAL",
            "ALTER TABLE documents ADD COLUMN forms_extraction_method VARCHAR",
            "ALTER TABLE documents ADD COLUMN forms_match_method VARCHAR",
            "ALTER TABLE documents ADD COLUMN forms_ingest_key VARCHAR",
            "ALTER TABLE documents ADD COLUMN forms_db_table_names TEXT",
            "CREATE INDEX IF NOT EXISTS ix_documents_forms_ingest_key ON documents(forms_ingest_key)",
        ],
    ),
]

def apply_forms_migrations(conn: Connection) -> None:
    """Apply forms migrations that haven't been applied yet. Fail-fast on error."""
    conn.execute(text(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(name VARCHAR PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    ))
    conn.commit()

    for name, statements in _FORMS_MIGRATIONS:
        row = conn.execute(
            text("SELECT 1 FROM schema_migrations WHERE name = :name"),
            {"name": name},
        ).fetchone()
        if row:
            continue  # Already applied

        for stmt in statements:
            conn.execute(text(stmt))
        conn.execute(
            text("INSERT INTO schema_migrations (name) VALUES (:name)"),
            {"name": name},
        )
        conn.commit()
        logger.info("forms.migration.applied", migration=name)
```

**Behavior on failure:** Migration raises, application refuses to start. This is intentional — silent column-missing errors are worse than a blocked deploy.

## 7. Template Lifecycle (fixes #3, #9)

### 7.1 Template States

Templates follow a `draft -> approved -> active -> archived` lifecycle:

```python
class TemplateStatus(str, Enum):
    DRAFT = "draft"           # Created, not yet approved for matching
    APPROVED = "approved"     # Reviewed and approved, eligible for matching
    ARCHIVED = "archived"     # Soft-deleted, not eligible for matching
```

**State transitions:**
- `POST /templates` -> creates in `draft` status
- `POST /templates/{id}/approve` -> requires admin, sets `approved` with `approved_by` + `approved_at`
- `DELETE /templates/{id}` -> sets `archived` (soft delete)
- Only `approved` templates are returned by `FormMatcher.match_document()`

**When `forms_template_require_approval = False`:** Templates are created directly in `approved` status (dev/testing shortcut).

### 7.2 Template API Endpoints

**New file: `ai_ready_rag/api/forms_templates.py`**

```python
router = APIRouter()

@router.post("/templates", status_code=201)
async def create_template(
    request: FormTemplateCreateRequest,
    current_user=Depends(get_admin_user),
):
    """Register a new form template (status: draft or approved per config)."""

@router.get("/templates")
async def list_templates(
    tenant_id: str | None = None,
    source_format: str | None = None,
    status: str | None = None,
    current_user=Depends(get_current_user),
):
    """List templates. Non-admin users see approved/active only."""

@router.get("/templates/{template_id}")
async def get_template(
    template_id: str,
    version: int | None = None,
    current_user=Depends(get_current_user),
):
    """Get a specific template. Non-admin users cannot see draft templates."""

@router.post("/templates/{template_id}/approve", status_code=200)
async def approve_template(
    template_id: str,
    current_user=Depends(get_admin_user),
):
    """Approve a draft template for active matching. Requires admin role."""

@router.delete("/templates/{template_id}", status_code=204)
async def delete_template(
    template_id: str,
    current_user=Depends(get_admin_user),
):
    """Archive a template (soft delete). Requires admin role."""

@router.post("/templates/{template_id}/preview")
async def preview_extraction(
    template_id: str,
    file: UploadFile,
    current_user=Depends(get_admin_user),
):
    """Dry-run extraction against a sample document. Admin only (accesses raw field values)."""
```

### 7.3 Authorization Matrix (fixes #11)

| Endpoint | Method | Role | Notes |
|----------|--------|------|-------|
| `/templates` | POST | admin | Creates template |
| `/templates` | GET | user+ | Non-admin: approved only |
| `/templates/{id}` | GET | user+ | Non-admin: approved only, no field definitions |
| `/templates/{id}/approve` | POST | admin | Promotes draft -> approved |
| `/templates/{id}` | DELETE | admin | Archives template |
| `/templates/{id}/preview` | POST | admin | Accesses raw extracted values |

### 7.4 Template Deletion / Versioning Impact (fixes #9)

- **Archiving a template:** Existing documents that reference `forms_template_id` are unaffected. Their extracted data persists. The template simply stops matching new uploads.
- **New template version:** Creates a new version entry. Old documents keep their `forms_template_version` reference. New uploads match against the latest approved version.
- **Reprocessing:** If a document is reprocessed after a template version bump, the new `form_extraction_key` (which includes `template_version`) triggers fresh extraction. Old vectors are overwritten via deterministic point IDs.

Mount the forms template management API in `main.py`:

```python
# In create_app() or startup, after existing router includes:
if settings.use_ingestkit_forms:
    try:
        from ai_ready_rag.api.forms_templates import router as forms_router
        app.include_router(
            forms_router,
            prefix="/api/forms",
            tags=["Form Templates"],
        )
        logger.info("forms.router.mounted")
    except ImportError:
        logger.warning("forms.router.import_failed")
```

## 8. Deletion Cleanup (fixes #4)

Add to `document_service.py` `delete_document()` method (after existing Excel cleanup):

```python
# Clean up ingestkit-forms tables if present
if document.forms_db_table_names:
    self._cleanup_forms_tables(document.forms_db_table_names)

def _cleanup_forms_tables(self, table_names_json: str) -> None:
    """Drop ingestkit-forms tables from the forms data DB."""
    try:
        table_names = json.loads(table_names_json)
    except (json.JSONDecodeError, TypeError):
        logger.warning("forms.cleanup.invalid_json", raw=table_names_json)
        return

    if not isinstance(table_names, list):
        logger.warning("forms.cleanup.not_list", raw=table_names_json)
        return

    db_path = self.settings.forms_db_path
    if not Path(db_path).exists():
        return

    try:
        from ai_ready_rag.services.ingestkit_adapters import VERagFormDBAdapter
        form_db = VERagFormDBAdapter(db_path=db_path)
        for table_name in table_names:
            try:
                # Validate identifier BEFORE using in SQL (fixes #4)
                form_db.check_table_name(table_name)
                form_db.execute_sql(f"DROP TABLE IF EXISTS [{table_name}]")
                logger.info("forms.cleanup.table_dropped", table=table_name)
            except ValueError:
                logger.error("forms.cleanup.unsafe_identifier", table=table_name)
            except Exception as e:
                logger.warning("forms.cleanup.table_failed", table=table_name, error=str(e))
    except ImportError:
        logger.warning("forms.cleanup.import_failed")
```

**Identifier validation rule:** Table names must match `^[a-zA-Z_][a-zA-Z0-9_]{0,127}$`. Validation uses `ingestkit_forms.validate_table_name()` (the canonical implementation) via `VERagFormDBAdapter.check_table_name()`. ingestkit-forms' `FormDBWriter` also validates internally when generating names from `form_db_table_prefix` + sanitized template name. Any name not matching the pattern is rejected and logged as an error.

## 9. Vector Payload Extensions

The existing `VERagVectorStoreAdapter` merges VE-RAG standard fields with ingestkit provenance fields. For forms, the ingestkit-forms `FormChunkWriter` produces chunks with `FormChunkMetadata` that includes:

```python
# These map to Qdrant payload fields (ingestkit_ prefix)
"ingestkit_source_format": "pdf",
"ingestkit_ingestion_method": "form_extraction",
"ingestkit_parser_version": "ingestkit_forms:1.0.0",
"ingestkit_ingest_key": "sha256...",
"ingestkit_chunk_hash": "sha256...",
"ingestkit_source_uri": "file://...",
"ingestkit_ingest_run_id": "uuid",

# Form-specific fields
"ingestkit_forms_template_id": "tmpl-uuid",
"ingestkit_forms_template_name": "W-4 2026",
"ingestkit_forms_template_version": 1,
"ingestkit_forms_extraction_method": "native_fields",
"ingestkit_forms_overall_confidence": 0.95,
"ingestkit_forms_field_names": ["employee_name", ...],
"ingestkit_forms_match_method": "auto_detect",
"ingestkit_forms_form_id": "form-uuid",
```

**No adapter changes needed** — the VERagVectorStoreAdapter already handles arbitrary metadata passthrough.

**Data handling note:** Field *names* appear in vector payloads (for search/filtering). Field *values* appear in `chunk_text` only after redaction (Section 14). High-risk field values are replaced with `[REDACTED]` in chunk text when `forms_redact_high_risk_fields = True`.

## 10. File Layout

```
data/
  ai_ready_rag.db           # Main app DB (existing)
  excel_tables.db            # ingestkit-excel structured data (existing)
  forms_data.db              # ingestkit-forms structured data (NEW)
  form_templates/            # Template JSON files (NEW)
    {template_id}/
      v1.json
      v2.json
  chroma_db/                 # ChromaDB persist (existing)
```

## 11. Dependencies

Add to `pyproject.toml` optional dependencies:

```toml
[project.optional-dependencies]
ingestkit-forms = ["ingestkit-forms>=0.1"]
```

Import guards everywhere (same as Excel pattern):
```python
def _should_use_ingestkit_forms(self) -> bool:
    if not self.settings.use_ingestkit_forms:
        return False
    try:
        import ingestkit_forms
        return True
    except ImportError:
        logger.warning("forms.dependency.missing")
        return False
```

## 12. Error Taxonomy & Failure Handling (fixes #5)

### 12.1 Error Classification

| Error Code | Category | User-Visible Status | Retry? | Fallback? |
|------------|----------|-------------------|--------|-----------|
| `E_FORM_NO_MATCH` | Match | N/A (transparent) | No | Yes — standard chunker |
| `E_FORM_UNSUPPORTED_FORMAT` | Validation | N/A (transparent) | No | Yes — standard chunker |
| `E_FORM_FILE_CORRUPT` | Validation | `failed` | No | No — reject file |
| `E_FORM_FILE_TOO_LARGE` | Validation | `failed` | No | No — reject file |
| `E_FORM_TEMPLATE_NOT_FOUND` | Template | `failed` | No | No |
| `E_FORM_EXTRACTION_LOW_CONFIDENCE` | Extraction | `failed` + warning | No | No — fail-closed |
| `E_FORM_EXTRACTION_FAILED` | Extraction | `failed` | No | No |
| `E_FORM_OCR_FAILED` | Dependency | `failed` + log | No | Yes — if OCR optional |
| `E_FORM_VLM_UNAVAILABLE` | Dependency | Continue w/o VLM | No | Graceful degrade |
| `E_BACKEND_VECTOR_TIMEOUT` | Backend | `failed` | Yes (2x) | No |
| `E_BACKEND_VECTOR_CONNECT` | Backend | `failed` | Yes (2x) | No |
| `E_BACKEND_DB_TIMEOUT` | Backend | `failed` | Yes (2x) | No |
| `E_BACKEND_DB_CONNECT` | Backend | `failed` | Yes (2x) | No |
| `E_BACKEND_EMBED_TIMEOUT` | Backend | `failed` | Yes (2x) | No |
| `E_BACKEND_EMBED_CONNECT` | Backend | `failed` | Yes (2x) | No |
| `E_FORM_DUAL_WRITE_PARTIAL` | Output | `failed` + compensate | No | No |

### 12.2 Fallback Behavior

Codes in `_FALLBACK_ERROR_CODES` trigger:
1. Best-effort compensation of partial writes (Section 5, `_compensate`)
2. Return `(None, True)` to ProcessingService
3. Standard chunker processes the document instead
4. No user-visible error — fallback is transparent

All other errors:
1. Set `document.status = "failed"` with `error_message`
2. Return `(ProcessingResult(success=False, ...), False)`
3. User sees "Processing failed" with error summary

### 12.3 Runtime Dependency Handling

| Dependency | Required? | Missing Behavior |
|------------|-----------|-----------------|
| ingestkit-forms package | Yes | `_should_use_ingestkit_forms()` returns False, log warning |
| Qdrant | Yes | Backend error, document fails, retry available |
| Ollama (embedding) | Yes | Backend error, document fails, retry available |
| OCR engine | No | Extraction uses native_fields only. If PDF is flattened, extraction fails with low confidence |
| VLM model | No | Low-confidence fields keep OCR values. Warning logged |

## 13. Summary of New Files

| File | Purpose |
|------|---------|
| `services/forms_processing_service.py` | FormsProcessingService (mirrors ExcelProcessingService) |
| `api/forms_templates.py` | Template CRUD + approval REST endpoints |
| `services/ingestkit_adapters.py` (edit) | Add VERagFormDBAdapter, VERagLayoutFingerprinter, VERagOCRAdapter, VERagVLMAdapter |
| `config.py` (edit) | Add feature flags and forms settings |
| `db/models/document.py` (edit) | Add forms_* columns |
| `db/database.py` (edit) | Add tracked migration with schema_migrations table |
| `services/processing_service.py` (edit) | Add forms routing before standard chunker |
| `services/document_service.py` (edit) | Add _cleanup_forms_tables() with identifier validation |
| `main.py` (edit) | Conditional mount of forms router |

## 14. Data Handling & Security (fixes #1)

### 14.1 Data Classification

| Data Type | Storage Location | Sensitivity | Handling |
|-----------|-----------------|-------------|----------|
| Template definitions (field names, regions) | `form_templates/` JSON | Low | No special handling |
| Extracted field *names* | Vector payloads, DB columns | Low | Stored as-is |
| Extracted field *values* | `forms_data.db` rows, `chunk_text` | **Variable** | Redaction applied per config |
| Layout fingerprints | `form_templates/` | None | Binary structural data, no content |
| Match confidence scores | Document model, vector payloads | None | Numeric metadata |

### 14.2 High-Risk Field Redaction

When `forms_redact_high_risk_fields = True` (default):

- ingestkit-forms' `redact_patterns` config receives patterns for SSN, tax IDs, account numbers:
  ```python
  _HIGH_RISK_PATTERNS = [
      r"\b\d{3}-\d{2}-\d{4}\b",       # SSN
      r"\b\d{2}-\d{7}\b",              # EIN
      r"\b\d{9,18}\b",                 # Account numbers (9-18 digits)
  ]
  ```
- Redaction applies to `chunk_text` (vector payloads) via ingestkit-forms' built-in redaction.
- `forms_data.db` stores **original values** for operational use (e.g., form re-export). Access to `forms_data.db` is restricted to the application process.

### 14.3 Retention & Deletion

- **Document deletion:** Triggers `_cleanup_forms_tables()` (drops DB rows) and `vector_service.delete_document()` (removes vectors). Both are best-effort.
- **No automatic retention policy in v1.** Data persists until document is deleted by a user/admin.
- **Future:** Add `forms_retention_days` config for automatic cleanup.

### 14.4 Access Controls

- `forms_data.db` file permissions: `0600` (owner-only), set at creation time by `VERagFormDBAdapter._ensure_db()`.
- Template preview endpoint (which shows raw extracted values) is admin-only.
- Vector payloads containing redacted chunk_text are accessible via normal RAG search (respects existing tag-based access control).

## 15. Observability (fixes #12)

### 15.1 Structured Log Events

All forms-related logs use the `ingestkit_forms` logger with structured `extra={}` dicts.

**Library-emitted events** (from ingestkit-forms internals, logger `ingestkit_forms`):

| Event | Level | Extra Keys | When |
|-------|-------|------------|------|
| `forms.match.auto` | INFO | `template_candidates, confidence, match_duration_ms, template_id` | Auto-match found templates |
| `forms.match.fallthrough` | INFO | `template_candidates, confidence, match_duration_ms, template_id` | Auto-match found no templates above threshold |
| `forms.match.manual` | INFO | `template_id, template_version, confidence, match_duration_ms` | Manual override path used |
| `forms.match.disabled` | INFO | `template_candidates, confidence` | Matching disabled via config |
| `forms.match.template_not_found` | WARNING | `template_id, template_version` | Matched template not in store |
| `forms.match.invalid_fingerprint` | WARNING | `template_id, template_version` | Template has corrupt fingerprint |
| `forms.extract.completed` | INFO | `template_id, template_version, extraction_method, duration_s, field_count, fields_extracted, fields_failed` | Extraction finished |
| `forms.extract.low_confidence` | WARNING | `template_id, overall_confidence, threshold` | Overall confidence below threshold (fail-closed) |
| `forms.write.table_created` | INFO | `table_name, template_id, template_name` | DB table created for template |
| `forms.write.schema_evolved` | WARNING | `table_name, columns_added, column_count` | New columns added to existing table |
| `forms.write.db_retry` | WARNING | `table_name, attempt, max_attempts, retry_delay_s, error` | DB write retry |
| `forms.write.embed_retry` | WARNING | `attempt, max_attempts, retry_delay_s, error` | Embedding retry |
| `forms.write.upsert_retry` | WARNING | `attempt, max_attempts, retry_delay_s, error` | Vector upsert retry |

**Host-emitted events** (from VE-RAG-System, logger `ai_ready_rag`):

| Event | Level | Keys | When |
|-------|-------|------|------|
| `forms.routing.fallback` | INFO | `document_id` | No match, falling back to standard pipeline |
| `forms.match.error` | WARNING | `document_id, error` | Match threw exception (host catch) |
| `forms.extract.error` | ERROR | `document_id, error_code, error` | Extraction failed (host catch) |
| `forms.compensate.vectors_failed` | WARNING | `error` | Compensation couldn't delete vectors |
| `forms.compensate.table_failed` | WARNING | `table, error` | Compensation couldn't drop table |
| `forms.cleanup.unsafe_identifier` | ERROR | `table` | Identifier validation rejected table name |
| `forms.migration.applied` | INFO | `migration` | Schema migration applied |
| `forms.dependency.missing` | WARNING | — | Package not installed but flag is True |

### 15.2 Metrics (Prometheus-compatible, via existing metrics middleware)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `forms_documents_processed_total` | Counter | `status={success,failed,fallback}` | Total documents routed through forms pipeline |
| `forms_match_confidence` | Histogram | `template_id` | Distribution of match confidence scores |
| `forms_extraction_duration_seconds` | Histogram | `extraction_method` | Extraction latency |
| `forms_fallback_total` | Counter | `reason={no_match,error,unsupported}` | Why documents fell back to standard pipeline |
| `forms_compensation_total` | Counter | `target={vectors,tables}, status={success,failed}` | Compensation action outcomes |

### 15.3 Health Checks

Add to existing `/api/health` response when `use_ingestkit_forms = True`:

```json
{
  "forms": {
    "enabled": true,
    "package_installed": true,
    "forms_db_writable": true,
    "template_store_readable": true,
    "templates_approved": 3,
    "templates_draft": 1
  }
}
```

## 16. Startup Validation (fixes #14)

When `use_ingestkit_forms = True`, validate at application startup:

```python
def _validate_forms_config(settings: Settings) -> None:
    """Fail-fast if forms config is invalid. Called during app startup."""
    # 1. Package importable
    try:
        import ingestkit_forms
    except ImportError:
        raise RuntimeError("use_ingestkit_forms=True but ingestkit-forms not installed")

    # 2. DB path is under data/ and parent directory is writable
    db_path = Path(settings.forms_db_path).resolve()
    data_dir = Path("./data").resolve()
    if not str(db_path).startswith(str(data_dir)):
        raise RuntimeError(f"forms_db_path must be under data/: {settings.forms_db_path}")
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(db_path.parent, os.W_OK):
        raise RuntimeError(f"forms_db_path parent not writable: {db_path.parent}")

    # 3. Template storage path is writable
    tmpl_path = Path(settings.forms_template_storage_path).resolve()
    if not tmpl_path.exists():
        tmpl_path.mkdir(parents=True, exist_ok=True)
    if not os.access(tmpl_path, os.W_OK):
        raise RuntimeError(f"forms_template_storage_path not writable: {tmpl_path}")

    # 4. Apply schema migrations (fail-fast)
    apply_forms_migrations(engine.connect())
```

## 17. Acceptance Criteria & Test Plan (fixes #13)

### 17.1 Required Tests

| Category | Test | Type | Description |
|----------|------|------|-------------|
| **Happy path** | `test_forms_process_matching_pdf` | Integration | Upload PDF that matches a template -> extraction succeeds, document columns populated, vectors written, DB rows written |
| **Happy path** | `test_forms_template_crud_lifecycle` | Integration | Create draft -> approve -> list (appears) -> archive -> list (gone) |
| **Fallback** | `test_forms_no_match_falls_back` | Unit | PDF with no matching template -> `(None, True)`, standard chunker processes it |
| **Fallback** | `test_forms_disabled_flag` | Unit | `use_ingestkit_forms=False` -> forms pipeline skipped entirely |
| **Fallback** | `test_forms_package_missing` | Unit | Mock ImportError -> `_should_use_ingestkit_forms()` returns False |
| **Compensation** | `test_forms_partial_write_compensation` | Unit | Simulate vector write success + DB write failure -> vectors cleaned up |
| **Compensation** | `test_forms_fallback_cleans_partial` | Unit | Fallback error code after partial write -> compensation runs |
| **Idempotency** | `test_forms_reprocess_same_pdf` | Integration | Process same PDF twice -> same ingest_key, no duplicate vectors/rows |
| **Security** | `test_forms_identifier_validation` | Unit | Malformed table names rejected by `validate_table_name()` |
| **Security** | `test_forms_path_traversal_rejected` | Unit | `forms_db_path` outside `data/` -> startup validation fails |
| **Security** | `test_forms_redaction_applied` | Unit | High-risk patterns redacted from chunk_text when config enabled |
| **Migration** | `test_forms_migration_idempotent` | Unit | Run `apply_forms_migrations()` twice -> no error, no duplicate columns |
| **Migration** | `test_forms_migration_failure_blocks_startup` | Unit | Invalid SQL in migration -> RuntimeError, app doesn't start |
| **Authz** | `test_forms_template_create_requires_admin` | Unit | Non-admin user -> 403 |
| **Authz** | `test_forms_template_list_filters_drafts` | Unit | Non-admin sees only approved templates |
| **Authz** | `test_forms_preview_requires_admin` | Unit | Non-admin -> 403 (preview shows raw values) |
| **Lifecycle** | `test_forms_template_archive_preserves_documents` | Unit | Archive template -> existing documents unaffected |
| **Observability** | `test_forms_structured_log_events` | Unit | Match/extract/fallback emit expected log events with required keys |

### 17.2 Out of Scope for v1

- ingestkit-pdf integration (no `use_ingestkit_pdf` flag, no PDF-specific routing)
- Automatic retention/expiry of form data
- `strict_atomic` dual-write mode (only `best_effort` in v1)
- Custom SQLite-backed template store (using FileSystemTemplateStore)
- Multi-file batch form processing

## 18. Implementation Order

1. **Phase 1 — Config + Model + Startup Validation** (no functional changes)
   - Add feature flags to config.py
   - Add document columns + tracked migration (Section 6.1)
   - Add VERagFormDBAdapter, VERagLayoutFingerprinter, VERagOCRAdapter, VERagVLMAdapter to adapters
   - Add startup validation (Section 16)

2. **Phase 2 — Processing Pipeline + Compensation**
   - Create FormsProcessingService with ordered writes + `_compensate`
   - Wire into ProcessingService routing (file type + flag dispatch only)
   - Add deletion cleanup with identifier validation

3. **Phase 3 — Template API + Lifecycle**
   - Create forms_templates.py endpoints with draft/approved/archived workflow
   - Add authz per endpoint matrix
   - Mount in main.py

4. **Phase 4 — Observability + Tests**
   - Add structured log events
   - Add metrics
   - Add health check fields
   - Run full test matrix (Section 17.1)
