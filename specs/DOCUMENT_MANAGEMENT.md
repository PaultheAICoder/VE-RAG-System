# Document Management Specification

| Field | Value |
|-------|-------|
| **Status** | DRAFT |
| **Version** | 1.2 |
| **Created** | 2026-01-28 |
| **Updated** | 2026-01-28 |
| **Type** | Backend Service + API |
| **Complexity** | COMPLEX |
| **Depends On** | Vector Service, Tag System, Auth System, ENV_PROFILE_PIPELINE |

---

## Summary

Document Management handles file upload, processing, indexing, and lifecycle management. Admin users upload documents with tag assignments; documents are processed, chunked, embedded, and indexed to vectors. The system tracks processing status and enables document listing, search, and deletion.

**Backend Selection:** The chunker and vector backends are determined by `ENV_PROFILE` (see `specs/ENV_PROFILE_PIPELINE.md`):
- **Laptop profile:** SimpleChunker + Chroma (fast local development)
- **Spark profile:** Docling + HybridChunker + Qdrant (production quality)

---

## Goals

- Admin-only document upload with required tag assignment
- Background processing with status tracking
- Docling integration with OCR, table extraction, and chunking
- Vector indexing via existing VectorService
- Document listing with filtering and pagination
- Document deletion (file + metadata + vectors)
- Processing error recovery and retry

---

## Scope

### In Scope (Phase 1)

- File upload API (admin only)
- Tag assignment at upload time
- Background processing queue
- Docling parsing with configurable options
- HybridChunker for semantic chunking
- Integration with VectorService for indexing
- Document status tracking (pending, processing, ready, failed)
- Document list API with filtering
- Document deletion (cascade to vectors)
- Admin UI for upload and management

### Out of Scope (Future)

- Bulk upload (multiple files at once)
- Scheduled re-indexing
- Document versioning
- Document viewer with highlighting
- Processing progress websocket
- Document OCR language auto-detection

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Document Management                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │   API Layer  │────▶│  Service     │────▶│  Processing  │            │
│  │  /api/docs/* │     │  Layer       │     │  Queue       │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│         │                    │                    │                     │
│         ▼                    ▼                    ▼                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │   File       │     │   SQLite     │     │   Docling    │            │
│  │   Storage    │     │   (metadata) │     │   Processor  │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│                                                   │                     │
│                                                   ▼                     │
│                                            ┌──────────────┐            │
│                                            │   Vector     │            │
│                                            │   Service    │            │
│                                            └──────────────┘            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### File Structure

```
ai_ready_rag/
├── api/
│   └── documents.py           # REST API endpoints
├── services/
│   ├── document_service.py    # Business logic
│   └── processing_service.py  # Docling processing
├── db/
│   └── models.py              # Document model (exists)
└── utils/
    └── file_utils.py          # File handling helpers
```

---

## Backend Selection

Document processing uses **factory functions** to obtain the correct backend based on `ENV_PROFILE`. This allows the same API to work on laptop (development) and Spark (production) without code changes.

### Service Factories

```python
from ai_ready_rag.services.factory import get_vector_service, get_chunker
from ai_ready_rag.config import get_settings

settings = get_settings()
vector_service = get_vector_service(settings)  # Returns Chroma or Qdrant
chunker = get_chunker(settings)                # Returns Simple or Docling
```

### Profile Comparison

| Component | Laptop Profile | Spark Profile |
|-----------|---------------|---------------|
| **Vector Backend** | Chroma (local) | Qdrant (server) |
| **Chunker** | SimpleChunker (RecursiveCharacterTextSplitter) | DoclingChunker (HybridChunker) |
| **OCR** | Disabled | Enabled (Tesseract) |
| **Text Extraction** | PyPDF2, python-docx | Docling (full parsing) |
| **Metadata** | Basic (no page numbers) | Full (page, section, headings) |

**Note:** All API endpoints and service interfaces remain identical regardless of profile. Only the underlying implementations differ.

See `specs/ENV_PROFILE_PIPELINE.md` for full profile configuration details.

---

## Data Model

### Existing Document Model

The Document model already exists in `ai_ready_rag/db/models.py`:

```python
class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String, nullable=False)           # Stored filename (UUID-based)
    original_filename = Column(String, nullable=False)  # User's original filename
    file_path = Column(String, nullable=False)          # Full path to file
    file_type = Column(String, nullable=False)          # MIME type or extension
    file_size = Column(Integer, nullable=False)         # Size in bytes
    status = Column(String, default="pending")          # Processing status
    error_message = Column(Text, nullable=True)         # Error details if failed
    chunk_count = Column(Integer, nullable=True)        # Number of chunks created
    uploaded_by = Column(String, nullable=False)        # User ID who uploaded
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)      # When processing completed

    tags = relationship("Tag", secondary=document_tags, back_populates="documents")
```

### Document Status Enum

```python
class DocumentStatus(str, Enum):
    PENDING = "pending"         # Uploaded, awaiting processing
    PROCESSING = "processing"   # Currently being processed
    READY = "ready"            # Successfully processed and indexed
    FAILED = "failed"          # Processing failed (see error_message)
```

### Extended Fields (Add to Model)

```python
# Additional columns to add to Document model
title = Column(String, nullable=True)               # Extracted or user-provided title
description = Column(Text, nullable=True)           # Optional description
page_count = Column(Integer, nullable=True)         # Number of pages (PDF)
word_count = Column(Integer, nullable=True)         # Approximate word count
processing_time_ms = Column(Integer, nullable=True) # Processing duration
content_hash = Column(String, nullable=True)        # SHA-256 of file content
```

---

## File Storage

### Storage Strategy

```
data/
└── uploads/
    └── {document_id}/
        ├── original.{ext}      # Original uploaded file
        └── metadata.json       # Processing metadata (optional)
```

### File Naming

- Store files in document-specific subdirectory
- Keep original extension for proper MIME detection
- Use `original.{ext}` to preserve extension while avoiding filename conflicts

### Storage Limits

| Limit | Value | Configurable |
|-------|-------|--------------|
| Max file size | 100 MB | Yes (`MAX_UPLOAD_SIZE_MB`) |
| Max total storage | 10 GB | Yes (`MAX_STORAGE_GB`) |
| Allowed extensions | pdf, docx, xlsx, pptx, txt, md, html, csv | Yes |

---

## API Endpoints

### POST /api/documents/upload

Upload a new document with tag assignment.

**Authentication:** Admin only

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | File | Yes | Document file |
| tag_ids | string[] | Yes | Array of tag IDs (at least one) |
| title | string | No | Optional title (extracted if not provided) |
| description | string | No | Optional description |

**Response:** `201 Created`

```json
{
  "id": "doc-uuid",
  "original_filename": "Employee Handbook.pdf",
  "file_type": "pdf",
  "file_size": 2456789,
  "status": "pending",
  "tags": [
    {"id": "tag-uuid", "name": "hr", "display_name": "HR"}
  ],
  "uploaded_by": "user-uuid",
  "uploaded_at": "2026-01-28T10:00:00Z"
}
```

**Errors:**

| Status | Condition |
|--------|-----------|
| 400 | No file provided |
| 400 | No tags provided |
| 400 | File too large |
| 400 | Invalid file type |
| 403 | Not admin |
| 409 | Duplicate file (same content hash) |
| 507 | Storage quota exceeded |

---

### GET /api/documents

List documents with filtering and pagination.

**Authentication:** Admin sees all; users see accessible documents

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| limit | int | 20 | Max results (1-100) |
| offset | int | 0 | Pagination offset |
| status | string | all | Filter by status |
| tag_id | string | - | Filter by tag |
| search | string | - | Search filename/title |
| sort_by | string | uploaded_at | Sort field |
| sort_order | string | desc | asc or desc |

**Response:** `200 OK`

```json
{
  "documents": [
    {
      "id": "doc-uuid",
      "original_filename": "Employee Handbook.pdf",
      "title": "Employee Handbook 2026",
      "file_type": "pdf",
      "file_size": 2456789,
      "status": "ready",
      "chunk_count": 42,
      "tags": [
        {"id": "tag-uuid", "name": "hr", "display_name": "HR"}
      ],
      "uploaded_by": "user-uuid",
      "uploaded_at": "2026-01-28T10:00:00Z",
      "processed_at": "2026-01-28T10:02:30Z"
    }
  ],
  "total": 156,
  "limit": 20,
  "offset": 0
}
```

---

### GET /api/documents/{id}

Get document details.

**Authentication:** Admin or user with matching tags

**Response:** `200 OK`

```json
{
  "id": "doc-uuid",
  "original_filename": "Employee Handbook.pdf",
  "title": "Employee Handbook 2026",
  "description": "Company policies and procedures",
  "file_type": "pdf",
  "file_size": 2456789,
  "status": "ready",
  "chunk_count": 42,
  "page_count": 85,
  "word_count": 32450,
  "processing_time_ms": 15230,
  "tags": [
    {"id": "tag-uuid", "name": "hr", "display_name": "HR"}
  ],
  "uploaded_by": "user-uuid",
  "uploaded_at": "2026-01-28T10:00:00Z",
  "processed_at": "2026-01-28T10:02:30Z"
}
```

---

### DELETE /api/documents/{id}

Delete a document (file, metadata, and vectors).

**Authentication:** Admin only

**Response:** `204 No Content`

**Side Effects:**
1. Delete file from storage
2. Delete all vectors from Qdrant (by document_id filter)
3. Delete document record and tag associations from SQLite

---

### POST /api/documents/{id}/reprocess

Reprocess a failed or existing document.

**Authentication:** Admin only

**Request Body:**

```json
{
  "enable_ocr": true,
  "force_ocr": false,
  "ocr_language": "eng"
}
```

**Response:** `202 Accepted`

```json
{
  "id": "doc-uuid",
  "status": "pending",
  "message": "Document queued for reprocessing"
}
```

---

### PATCH /api/documents/{id}/tags

Update document tags.

**Authentication:** Admin only

**Request Body:**

```json
{
  "tag_ids": ["tag-uuid-1", "tag-uuid-2"]
}
```

**Response:** `200 OK`

**Side Effects:**
1. Update document_tags in SQLite
2. Update tag payload in Qdrant vectors for this document

---

## Processing Pipeline

### Upload Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           UPLOAD FLOW                                    │
└─────────────────────────────────────────────────────────────────────────┘

  Admin Upload
       │
       ▼
  ┌─────────────────┐
  │ 1. Validate     │  - File size ≤ 100MB
  │    Request      │  - Allowed file type
  │                 │  - At least one tag
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ 2. Check        │  - Compute SHA-256 hash
  │    Duplicate    │  - Query by content_hash
  │                 │  - Reject if exists
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ 3. Store File   │  - Create /data/uploads/{doc_id}/
  │                 │  - Save as original.{ext}
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ 4. Create       │  - Insert Document row
  │    DB Record    │  - Status = "pending"
  │                 │  - Link tags
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ 5. Queue        │  - Add to processing queue
  │    Processing   │  - Return 201 to client
  └─────────────────┘
```

### Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROCESSING FLOW                                  │
└─────────────────────────────────────────────────────────────────────────┘

  Queue picks up document
       │
       ▼
  ┌─────────────────┐
  │ 1. Update       │  - Status = "processing"
  │    Status       │  - Record start time
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ 2. Docling      │  - Create DocumentConverter
  │    Parse        │  - Configure OCR options
  │                 │  - Parse document
  └────────┬────────┘
           │
           ├──── Error ────▶ Set status="failed", error_message
           │
           ▼
  ┌─────────────────┐
  │ 3. Extract      │  - Title from metadata/first heading
  │    Metadata     │  - Page count
  │                 │  - Word count
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ 4. Chunk        │  - HybridChunker (512 tokens)
  │    Document     │  - Preserve section info
  │                 │  - Track chunk indices
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ 5. Index to     │  - Call VectorService.index_document()
  │    Qdrant       │  - Pass tags for filtering
  │                 │  - Pass metadata (page, section)
  └────────┬────────┘
           │
           ├──── Error ────▶ Set status="failed", error_message
           │
           ▼
  ┌─────────────────┐
  │ 6. Update       │  - Status = "ready"
  │    Document     │  - chunk_count = N
  │                 │  - processed_at = now()
  │                 │  - processing_time_ms = duration
  └─────────────────┘
```

---

## Docling Integration

> **Note:** This section describes the **Spark profile** chunker implementation. For laptop profile, see SimpleChunker in `specs/ENV_PROFILE_PIPELINE.md`. The ProcessingService uses `get_chunker(settings)` to obtain the appropriate implementation.

### Document Converter Configuration

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TesseractOcrOptions,
)

def create_document_converter(
    enable_ocr: bool = True,
    force_ocr: bool = False,
    ocr_language: str = "eng",
    table_mode: str = "accurate",
) -> DocumentConverter:
    """Create configured Docling converter."""

    table_options = TableStructureOptions(
        do_cell_matching=True,
        mode=table_mode,  # "accurate" or "fast"
    )

    ocr_options = None
    if enable_ocr:
        ocr_options = TesseractOcrOptions(
            lang=[ocr_language],
            force_full_page_ocr=force_ocr,
        )

    pipeline_options = PdfPipelineOptions(
        do_ocr=enable_ocr,
        do_table_structure=True,
        table_structure_options=table_options,
    )

    if ocr_options:
        pipeline_options.ocr_options = ocr_options

    return DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
```

### Chunking Configuration

```python
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

# Use same tokenizer as embedding model
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1")

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=512,
    merge_peers=True,           # Merge small adjacent chunks
    include_metadata=True,      # Include section/page info
)

def chunk_document(docling_result) -> list[dict]:
    """Chunk document and return list of chunk dicts."""
    chunks = []
    for i, chunk in enumerate(chunker.chunk(docling_result.document)):
        chunks.append({
            "chunk_index": i,
            "text": chunk.text,
            "page_number": chunk.meta.get("page_number"),
            "section": chunk.meta.get("headings", [None])[-1],
            "token_count": len(tokenizer.encode(chunk.text)),
        })
    return chunks
```

### Supported File Types

| Extension | MIME Type | Docling Support | Notes |
|-----------|-----------|-----------------|-------|
| pdf | application/pdf | Full | OCR, tables, images |
| docx | application/vnd.openxmlformats-officedocument.wordprocessingml.document | Full | Native parsing |
| xlsx | application/vnd.openxmlformats-officedocument.spreadsheetml.sheet | Full | Sheet-aware |
| pptx | application/vnd.openxmlformats-officedocument.presentationml.presentation | Full | Slide-aware |
| txt | text/plain | Basic | Direct text |
| md | text/markdown | Basic | Markdown preserved |
| html | text/html | Full | HTML parsing |
| csv | text/csv | Basic | Tabular data |

---

## Service Layer

### DocumentService

```python
class DocumentService:
    """Document management business logic."""

    def __init__(
        self,
        db: Session,
        vector_service: VectorService,
        storage_path: Path,
    ):
        self.db = db
        self.vector_service = vector_service
        self.storage_path = storage_path

    async def upload(
        self,
        file: UploadFile,
        tag_ids: list[str],
        uploaded_by: str,
        title: str | None = None,
        description: str | None = None,
    ) -> Document:
        """Upload and queue document for processing."""
        ...

    async def list_documents(
        self,
        user_id: str,
        user_tags: list[str],
        is_admin: bool,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
        tag_id: str | None = None,
        search: str | None = None,
    ) -> tuple[list[Document], int]:
        """List documents with filtering."""
        ...

    async def get_document(
        self,
        document_id: str,
        user_id: str,
        user_tags: list[str],
        is_admin: bool,
    ) -> Document | None:
        """Get document if user has access."""
        ...

    async def delete_document(
        self,
        document_id: str,
    ) -> bool:
        """Delete document, file, and vectors."""
        ...

    async def update_tags(
        self,
        document_id: str,
        tag_ids: list[str],
    ) -> Document:
        """Update document tags (also updates vectors)."""
        ...
```

### ProcessingService

```python
class ProcessingService:
    """Document processing with Docling."""

    def __init__(
        self,
        vector_service: VectorService,
        enable_ocr: bool = True,
        ocr_language: str = "eng",
    ):
        self.vector_service = vector_service
        self.converter = create_document_converter(
            enable_ocr=enable_ocr,
            ocr_language=ocr_language,
        )
        self.chunker = self._create_chunker()

    async def process_document(
        self,
        document: Document,
        file_path: Path,
    ) -> ProcessingResult:
        """Process document and index to vectors."""
        ...

    def _extract_metadata(
        self,
        docling_result,
    ) -> dict:
        """Extract title, page count, word count."""
        ...

    def _chunk_document(
        self,
        docling_result,
    ) -> list[ChunkInfo]:
        """Chunk document with metadata."""
        ...
```

---

## Background Processing

### Queue Strategy

For Phase 1, use simple in-process background tasks via FastAPI's `BackgroundTasks`:

```python
from fastapi import BackgroundTasks

@router.post("/upload")
async def upload_document(
    ...,
    background_tasks: BackgroundTasks,
):
    # ... validation and storage ...

    # Queue processing
    background_tasks.add_task(
        process_document_task,
        document_id=document.id,
    )

    return document
```

### Future: Task Queue

For production scale, migrate to Celery or ARQ:

```python
# Future pattern with Celery
@celery_app.task(bind=True, max_retries=3)
def process_document_task(self, document_id: str):
    try:
        # ... processing logic ...
    except Exception as e:
        self.retry(countdown=60 * (self.request.retries + 1))
```

---

## Error Handling

### Processing Errors

| Error Type | Handling | User Message |
|------------|----------|--------------|
| File not found | Set status=failed | "File could not be read" |
| Docling parse error | Set status=failed, log details | "Document parsing failed" |
| OCR timeout | Retry without OCR | "Processing without OCR" |
| Embedding error | Retry 3x, then fail | "Vector indexing failed" |
| Qdrant connection | Retry 3x, then fail | "Vector service unavailable" |

### Error Recovery

```python
async def process_with_retry(
    document_id: str,
    max_retries: int = 3,
) -> bool:
    """Process document with retry logic."""
    for attempt in range(max_retries):
        try:
            await process_document(document_id)
            return True
        except TransientError as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
    return False
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UPLOAD_DIR` | `./data/uploads` | File storage directory |
| `MAX_UPLOAD_SIZE_MB` | `100` | Maximum file size in MB |
| `MAX_STORAGE_GB` | `10` | Maximum total storage in GB |
| `ENABLE_OCR` | `true` | Enable OCR by default |
| `OCR_LANGUAGE` | `eng` | Default OCR language |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Token overlap between chunks |

### Settings Class

```python
class DocumentSettings(BaseSettings):
    upload_dir: Path = Path("./data/uploads")
    max_upload_size_mb: int = 100
    max_storage_gb: int = 10
    enable_ocr: bool = True
    ocr_language: str = "eng"
    chunk_size: int = 512
    chunk_overlap: int = 50
    allowed_extensions: list[str] = [
        "pdf", "docx", "xlsx", "pptx", "txt", "md", "html", "csv"
    ]

    class Config:
        env_prefix = "DOCUMENT_"
```

---

## Implementation Issues

### Issue 031: Document Upload API (SIMPLE)

**Scope:** Create `/api/documents/upload` endpoint

**Files:**
- Create: `ai_ready_rag/api/documents.py`
- Create: `ai_ready_rag/services/document_service.py`
- Create: `ai_ready_rag/utils/file_utils.py`

**Acceptance Criteria:**
- [ ] POST /api/documents/upload accepts multipart file + tag_ids
- [ ] Validates file size, type, and tag requirements
- [ ] Stores file in /data/uploads/{doc_id}/
- [ ] Creates Document record with status="pending"
- [ ] Returns 201 with document details
- [ ] Admin-only access enforced

---

### Issue 032: Document Processing Service (MODERATE)

**Scope:** Docling integration and chunking

**Files:**
- Create: `ai_ready_rag/services/processing_service.py`

**Acceptance Criteria:**
- [ ] Configurable DocumentConverter with OCR options
- [ ] HybridChunker with 512-token chunks
- [ ] Metadata extraction (title, page count, word count)
- [ ] Integration with VectorService.index_document()
- [ ] Status updates during processing
- [ ] Error handling with meaningful messages

---

### Issue 033: Background Processing Queue (SIMPLE)

**Scope:** Async processing with status tracking

**Files:**
- Modify: `ai_ready_rag/api/documents.py`
- Modify: `ai_ready_rag/services/document_service.py`

**Acceptance Criteria:**
- [ ] Processing runs in background (FastAPI BackgroundTasks)
- [ ] Document status updates: pending → processing → ready/failed
- [ ] processing_time_ms recorded
- [ ] Error messages captured on failure

---

### Issue 034: Document List and Get APIs (SIMPLE)

**Scope:** List and retrieve documents

**Files:**
- Modify: `ai_ready_rag/api/documents.py`

**Acceptance Criteria:**
- [ ] GET /api/documents with pagination
- [ ] Filter by status, tag_id, search term
- [ ] Sort by uploaded_at, filename, status
- [ ] GET /api/documents/{id} returns full details
- [ ] Access control: admins see all, users see accessible

---

### Issue 035: Document Deletion (SIMPLE)

**Scope:** Delete document with cascade

**Files:**
- Modify: `ai_ready_rag/api/documents.py`
- Modify: `ai_ready_rag/services/document_service.py`

**Acceptance Criteria:**
- [ ] DELETE /api/documents/{id}
- [ ] Deletes file from storage
- [ ] Deletes vectors from Qdrant (via VectorService)
- [ ] Deletes Document row and tag associations
- [ ] Admin-only access

---

### Issue 036: Document Tag Update (SIMPLE)

**Scope:** Update tags and sync to vectors

**Files:**
- Modify: `ai_ready_rag/api/documents.py`
- Modify: `ai_ready_rag/services/document_service.py`

**Acceptance Criteria:**
- [ ] PATCH /api/documents/{id}/tags
- [ ] Updates document_tags in SQLite
- [ ] Updates tag payload in Qdrant vectors
- [ ] Admin-only access

---

### Issue 037: Document Reprocessing (SIMPLE)

**Scope:** Reprocess failed or existing documents

**Files:**
- Modify: `ai_ready_rag/api/documents.py`

**Acceptance Criteria:**
- [ ] POST /api/documents/{id}/reprocess
- [ ] Accepts optional OCR settings
- [ ] Deletes existing vectors before reprocessing
- [ ] Queues document for processing
- [ ] Admin-only access

---

### Issue 038: Admin Document UI (MODERATE)

**Scope:** Gradio UI for document management

**Files:**
- Modify: `ai_ready_rag/ui/gradio_app.py`
- Create: `ai_ready_rag/ui/document_components.py`

**Acceptance Criteria:**
- [ ] Upload form with file picker and tag selector
- [ ] Document list table with status, tags, actions
- [ ] Delete confirmation dialog
- [ ] Processing status indicator
- [ ] Tag update modal
- [ ] Admin-only visibility

---

### Issue 039: Document Management Tests (MODERATE)

**Scope:** Integration tests for document APIs

**Files:**
- Create: `tests/test_documents.py`

**Acceptance Criteria:**
- [ ] Test upload with valid file and tags
- [ ] Test upload validation (size, type, tags)
- [ ] Test list with filtering and pagination
- [ ] Test delete cascade (file, vectors, db)
- [ ] Test access control (admin vs user)
- [ ] Mock Docling and VectorService for unit tests

---

## Acceptance Criteria

### Phase 1 Complete When:

- [ ] Admin can upload documents via API
- [ ] Documents are processed and indexed in background
- [ ] Document status is tracked (pending → processing → ready/failed)
- [ ] Documents can be listed with filtering
- [ ] Documents can be deleted (cascades to vectors)
- [ ] Document tags can be updated
- [ ] Failed documents can be reprocessed
- [ ] Admin UI enables all management operations
- [ ] All tests pass

---

## Security Considerations

| Concern | Mitigation |
|---------|------------|
| File upload attacks | Validate MIME type, use allowlist, scan with ClamAV (future) |
| Path traversal | Use UUID-based storage paths, never use user-provided filenames |
| DoS via large files | Enforce size limits, rate limiting on upload |
| Unauthorized access | Admin-only upload/delete, tag-based read access |
| Data leakage | Pre-retrieval filtering ensures users only see accessible documents |

---

## Future Considerations

| Feature | Notes |
|---------|-------|
| Bulk upload | ZIP archive with manifest for batch processing |
| Document versioning | Keep version history, diff between versions |
| Document viewer | In-app PDF viewer with search highlighting |
| Progress streaming | WebSocket for real-time processing progress |
| Auto-tagging | ML-based tag suggestions based on content |
| OCR language detection | Automatic language detection for OCR |
| Scheduled reindex | Periodic re-embedding with newer models |

---

## Next Steps

1. Review this spec for completeness
2. Run `/orchestrate 031 through 039` to implement
