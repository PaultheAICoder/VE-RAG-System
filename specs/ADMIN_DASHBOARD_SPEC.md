# Admin Dashboard Features Specification

**Version**: 1.1 DRAFT
**Status**: For Review
**Date**: 2026-01-29
**Source**: Spark server UI screenshot analysis

---

## Overview

This specification documents four admin dashboard features observed on the Spark server that need to be implemented in the main codebase:

1. **Processing Options** - Per-upload document processing configuration
2. **Knowledge Base Statistics** - Real-time vector store metrics
3. **Model Configuration** - Runtime LLM model switching
4. **Model Architecture Display** - System component visibility

---

## Implementation Status Summary

### 1. Processing Options

| Function | Status | Location | Notes |
|----------|--------|----------|-------|
| `enable_ocr` setting | ✅ COMPLETE | `config.py:117` | Profile-based default |
| `ocr_language` setting | ✅ COMPLETE | `config.py:118` | Default "eng" |
| `force_full_page_ocr` setting | ⚠️ HARDCODED | `chunker_docling.py:69` | Hardcoded to `False`, not configurable |
| `table_extraction_mode` setting | ❌ MISSING | - | Need accurate/fast toggle |
| `include_image_descriptions` setting | ❌ MISSING | - | Docling supports this, not exposed |
| OCR options passed to Docling | ✅ COMPLETE | `chunker_docling.py:67-71` | TesseractOcrOptions configured |
| Table structure extraction | ✅ COMPLETE | `chunker_docling.py:59` | `do_table_structure=True` |
| API endpoint for options | ❌ MISSING | - | No PATCH /api/admin/processing-options |
| Per-upload options override | ❌ MISSING | - | Upload doesn't accept processing params |

### 2. Knowledge Base Statistics

| Function | Status | Location | Notes |
|----------|--------|----------|-------|
| `get_stats()` method | ✅ COMPLETE | `vector_service.py:672` | Returns CollectionStats |
| Total chunks count | ✅ COMPLETE | `vector_service.py:687` | From `points_count` |
| Unique documents count | ✅ COMPLETE | `vector_service.py:703-728` | Scrolls and counts unique `document_id` |
| Collection size bytes | ✅ COMPLETE | `vector_service.py:690-700` | payload + vectors storage |
| File list with names | ❌ MISSING | - | Only counts IDs, doesn't return filenames |
| API endpoint GET stats | ❌ MISSING | - | No /api/admin/knowledge-base/stats |
| API endpoint DELETE (clear) | ❌ MISSING | - | No /api/admin/knowledge-base |
| `clear_collection()` method | ✅ COMPLETE | `vector_service.py` | Exists but no API endpoint |

### 3. Model Configuration

| Function | Status | Location | Notes |
|----------|--------|----------|-------|
| `chat_model` setting | ✅ COMPLETE | `config.py:89` | Profile-based default |
| List available Ollama models | ✅ COMPLETE | `rag_service.py:1077-1084` | In `check_health()` method |
| Check model availability | ✅ COMPLETE | `rag_service.py:1086-1091` | Validates default model exists |
| Runtime model switching | ❌ MISSING | - | No way to change model without restart |
| Persist model selection | ❌ MISSING | - | No admin_settings table |
| API endpoint GET models | ❌ MISSING | - | No /api/admin/models |
| API endpoint PATCH model | ❌ MISSING | - | No /api/admin/models/chat |

### 4. Model Architecture Display

| Function | Status | Location | Notes |
|----------|--------|----------|-------|
| Profile info in health | ✅ COMPLETE | `health.py:20` | Returns `settings.env_profile` |
| Vector backend info | ✅ COMPLETE | `health.py:22` | Returns backend type |
| Chunker backend info | ✅ COMPLETE | `health.py:23` | Returns backend type |
| OCR enabled status | ✅ COMPLETE | `health.py:24` | Returns boolean |
| Ollama health check | ✅ COMPLETE | `rag_service.py:1074-1081` | Latency + availability |
| Available models list | ✅ COMPLETE | `rag_service.py:1084` | From Ollama /api/tags |
| Embedding model info | ⚠️ PARTIAL | `config.py:83` | Setting exists, not in health response |
| Embedding dimensions | ⚠️ PARTIAL | `config.py:84` | Setting exists, not in health response |
| Vector store URL | ⚠️ PARTIAL | `config.py:79` | Setting exists, not in health response |
| Tesseract availability | ❌ MISSING | - | No runtime check |
| EasyOCR availability | ❌ MISSING | - | No runtime check |
| Combined architecture API | ❌ MISSING | - | No /api/admin/architecture |

---

## Development Required Summary

| Feature Area | Complete | Partial | Missing | Priority |
|--------------|----------|---------|---------|----------|
| Processing Options | 3 | 1 | 4 | Medium |
| Knowledge Base Stats | 4 | 0 | 3 | **High** |
| Model Configuration | 3 | 0 | 4 | Low |
| Architecture Display | 6 | 3 | 3 | Low |

### High Priority Items (Demo-ready)
1. **GET /api/admin/knowledge-base/stats** - Expose existing `get_stats()` via API
2. **Add filenames to stats** - Extend `get_stats()` to return source filenames
3. **DELETE /api/admin/knowledge-base** - Clear all vectors endpoint

### Medium Priority Items
4. **`force_full_page_ocr` setting** - Make configurable instead of hardcoded
5. **`table_extraction_mode` setting** - Add accurate/fast toggle
6. **Per-upload processing options** - Accept options in upload endpoint

### Low Priority Items (Post-demo)
7. **Runtime model switching** - Requires persistence layer
8. **Architecture info endpoint** - Consolidate health info
9. **OCR availability detection** - Runtime checks for Tesseract/EasyOCR

---

## 1. Processing Options

### 1.1 Purpose
Allow admins to configure document processing settings on a per-upload or global basis, controlling OCR behavior, table extraction quality, and image description generation.

### 1.2 UI Components

| Component | Type | Default | Description |
|-----------|------|---------|-------------|
| Enable OCR | Checkbox | ✓ Checked | Extract text from scanned documents and images |
| Force Full Page OCR | Checkbox | ☐ Unchecked | Use OCR on all pages (for fully scanned documents) |
| OCR Language | Dropdown | English | Language for OCR text recognition |
| Table Extraction Mode | Radio Group | Accurate (slower) | Trade-off between accuracy and speed |
| Include Image Descriptions | Checkbox | ✓ Checked | Add descriptions of images/figures found in documents |

### 1.3 OCR Language Options
Based on Tesseract language packs:
- English (eng)
- Spanish (spa)
- French (fra)
- German (deu)
- Chinese Simplified (chi_sim)
- Japanese (jpn)
- Korean (kor)
- Arabic (ara)
- Custom (allow entering Tesseract language code)

### 1.4 Table Extraction Mode Details

| Mode | Behavior | Use Case |
|------|----------|----------|
| Accurate (slower) | Full table structure analysis with cell merging | Financial documents, complex tables |
| Fast (less precise) | Basic table detection, may miss merged cells | Simple tables, high volume processing |

### 1.5 Backend Requirements

#### Settings to Add (config.py)
```
# Document Processing - OCR
enable_ocr: bool = True                    # Already exists
force_full_page_ocr: bool = False          # NEW
ocr_language: str = "eng"                  # Already exists
ocr_available_languages: list[str]         # NEW - populated from Tesseract

# Document Processing - Tables
table_extraction_mode: Literal["accurate", "fast"] = "accurate"  # NEW

# Document Processing - Images
include_image_descriptions: bool = True    # NEW
```

#### API Endpoint
```
PATCH /api/admin/processing-options
Request Body:
{
  "enable_ocr": bool,
  "force_full_page_ocr": bool,
  "ocr_language": str,
  "table_extraction_mode": "accurate" | "fast",
  "include_image_descriptions": bool
}
Response: Updated settings object
```

#### Processing Service Changes
- Pass options to Docling pipeline
- `force_full_page_ocr`: When true, run OCR on every page regardless of text layer detection
- `table_extraction_mode`: Configure Docling's TableFormerMode (ACCURATE vs FAST)
- `include_image_descriptions`: Enable/disable image captioning in Docling

### 1.6 Persistence Strategy

**Option A: Per-Upload (Recommended for MVP)**
- Options are passed with each upload request
- No persistence needed
- Each document can have different settings

**Option B: Global Settings**
- Store in database (admin_settings table)
- Apply to all uploads by default
- Override per-upload if specified

### 1.7 Dependencies
- Tesseract OCR installed and accessible
- EasyOCR as fallback
- Docling with table extraction support

---

## 2. Knowledge Base Statistics

### 2.1 Purpose
Provide real-time visibility into the vector store contents, including chunk counts, file counts, and file listings.

### 2.2 UI Components

| Component | Description |
|-----------|-------------|
| Total chunks | Number of vector chunks in the collection |
| Unique files | Number of distinct documents processed |
| Files list | Comma-separated filenames with "+N more" truncation |
| Refresh Stats | Button to reload statistics |
| Clear Knowledge Base | Destructive action to delete all vectors |

### 2.3 Display Format (from screenshot)
```
Knowledge Base Statistics:
  Total chunks: 230
  Unique files: 22
  Files: Company_Overview.docx, IT_Security_Policy.docx, Product_Catalog_2025.pdf,
         Leadership_Meeting_Minutes_Dec2024.docx, Cash_Flow_Statement_2024.xlsx (+17 more)
```

### 2.4 API Endpoints

#### Get Statistics
```
GET /api/admin/knowledge-base/stats
Response:
{
  "total_chunks": int,
  "unique_files": int,
  "total_vectors": int,           // May differ from chunks if duplicates
  "collection_name": str,
  "files": [
    {
      "document_id": str,
      "filename": str,
      "chunk_count": int,
      "status": str
    }
  ],
  "storage_size_bytes": int | null,  // If available from vector DB
  "last_updated": datetime
}
```

#### Clear Knowledge Base
```
DELETE /api/admin/knowledge-base
Request Body:
{
  "confirm": true,               // Required confirmation flag
  "delete_source_files": bool    // Also delete uploaded files?
}
Response:
{
  "deleted_chunks": int,
  "deleted_files": int,
  "success": bool
}
```

### 2.5 Backend Implementation

#### Vector Service Method
```python
async def get_collection_stats(self) -> CollectionStats:
    """Get statistics about the vector collection."""
    # Query Qdrant/Chroma for:
    # - Total point count
    # - Unique document_id values (scroll/aggregate)
    # - List of filenames from payloads
```

#### Aggregation Query (Qdrant)
```python
# Get unique document IDs
points = client.scroll(
    collection_name=collection,
    with_payload=["document_id", "source"],
    limit=10000
)
# Aggregate in Python
```

### 2.6 Clear Knowledge Base Behavior

1. **Confirmation Required**: UI must show confirmation dialog
2. **Cascade Options**:
   - Delete vectors only (keep files, allow reprocessing)
   - Delete vectors AND source files (full cleanup)
   - Delete vectors AND update document status to "pending"
3. **Audit Logging**: Log who cleared and when
4. **Progress Feedback**: Show deletion progress for large collections

### 2.7 UI Refresh Behavior
- Auto-refresh on document upload completion
- Auto-refresh on document deletion
- Manual refresh via button
- Consider polling interval (30s) for processing status

---

## 3. Model Configuration

### 3.1 Purpose
Allow runtime switching of the chat/reasoning LLM model without application restart.

### 3.2 UI Components

| Component | Type | Description |
|-----------|------|-------------|
| Chat/Reasoning Model | Dropdown | Select from available Ollama models |
| Model Description | Text | "Model for routing, RAG responses, and evaluation" |
| Apply Model Change | Button | Apply the selected model |

### 3.3 Model Options (from Spark)
Based on available Ollama models:
- Qwen3 8B (Recommended) - `qwen3:8b`
- Llama 3.2 - `llama3.2:latest`
- DeepSeek R1 32B - `deepseek-r1:32b`
- Mistral 7B - `mistral:7b`
- Custom (enter model name)

### 3.4 API Endpoints

#### Get Available Models
```
GET /api/admin/models
Response:
{
  "available_models": [
    {
      "name": "qwen3:8b",
      "display_name": "Qwen3 8B (Recommended)",
      "size_gb": 4.7,
      "parameters": "8B",
      "quantization": "Q4_K_M",
      "recommended": true
    }
  ],
  "current_chat_model": str,
  "current_embedding_model": str
}
```

#### Change Model
```
PATCH /api/admin/models/chat
Request Body:
{
  "model_name": "qwen3:8b"
}
Response:
{
  "previous_model": str,
  "current_model": str,
  "success": bool,
  "message": str
}
```

### 3.5 Backend Implementation

#### Ollama Model Discovery
```python
async def list_available_models() -> list[ModelInfo]:
    """Query Ollama for available models."""
    response = await httpx.get(f"{ollama_url}/api/tags")
    models = response.json()["models"]
    return [
        ModelInfo(
            name=m["name"],
            size=m["size"],
            modified_at=m["modified_at"],
            # Add display names from config mapping
        )
        for m in models
    ]
```

#### Runtime Model Switching
```python
# Option A: Update settings singleton (requires careful thread safety)
settings.chat_model = new_model

# Option B: Store in database, read on each request
db.admin_settings.update({"chat_model": new_model})

# Option C: Environment variable + restart signal (not recommended)
```

### 3.6 Model Change Behavior

1. **Validation**: Verify model exists in Ollama before accepting
2. **Warm-up**: Optionally send a test prompt to load model into memory
3. **Graceful Transition**: Don't interrupt in-flight requests
4. **Persistence**: Store selection in database for restart persistence
5. **Audit**: Log model changes with timestamp and user

### 3.7 Considerations

- **Memory Management**: Switching models may require Ollama to unload/load
- **Response Time**: First request after switch may be slower
- **Embedding Model**: Should NOT be switchable at runtime (would require re-indexing)

---

## 4. Model Architecture Display

### 4.1 Purpose
Provide transparency about the system's AI components, their capabilities, and current status.

### 4.2 Display Sections (from screenshot)

#### Document Parsing
```
Document Parsing: Docling (local ML)
  • Layout analysis & table structure
  • OCR via Tesseract/EasyOCR
  • No LLM - uses bundled ML models
```

#### Embeddings
```
Embeddings: nomic-embed-text
  • Creates 768-dim vectors
  • Stored in ChromaDB
```

#### Chat Model
```
Chat Model: qwen3:8b
  • Query routing decisions
  • RAG response generation
  • Response evaluation
```

#### Infrastructure
```
Ollama: http://localhost:11434
```

#### OCR Status
```
OCR Status
  Tesseract OCR: Available
  EasyOCR: Available
```

### 4.3 API Endpoint

```
GET /api/admin/architecture
Response:
{
  "document_parsing": {
    "engine": "Docling",
    "version": "2.68.0",
    "type": "local ML",
    "capabilities": [
      "Layout analysis & table structure",
      "OCR via Tesseract/EasyOCR",
      "No LLM - uses bundled ML models"
    ]
  },
  "embeddings": {
    "model": "nomic-embed-text",
    "dimensions": 768,
    "vector_store": "Qdrant",  // or "ChromaDB"
    "vector_store_url": "http://localhost:6333"
  },
  "chat_model": {
    "name": "qwen3:8b",
    "provider": "Ollama",
    "capabilities": [
      "Query routing decisions",
      "RAG response generation",
      "Response evaluation"
    ]
  },
  "infrastructure": {
    "ollama_url": "http://localhost:11434",
    "ollama_status": "healthy",
    "vector_db_status": "healthy"
  },
  "ocr_status": {
    "tesseract": {
      "available": true,
      "version": "5.3.0",
      "languages": ["eng", "spa", "fra"]
    },
    "easyocr": {
      "available": true,
      "version": "1.7.0"
    }
  },
  "profile": "spark"  // or "laptop"
}
```

### 4.4 Health Check Integration

The architecture endpoint should include real-time health status:

```python
async def get_architecture_info() -> ArchitectureInfo:
    # Check Ollama
    ollama_healthy = await check_ollama_health()

    # Check vector DB
    vector_healthy = await vector_service.health_check()

    # Check OCR availability
    tesseract_available = shutil.which("tesseract") is not None
    easyocr_available = check_easyocr_import()

    return ArchitectureInfo(...)
```

### 4.5 UI Presentation

- **Read-only display**: This is informational, not configurable
- **Status indicators**: Green checkmark for available, red X for unavailable
- **Collapsible sections**: Allow expanding for details
- **Copy button**: Allow copying configuration for support tickets

---

## 5. Implementation Phases

### Phase 1: Knowledge Base Statistics (Low effort, High value)
- Add vector service stats method
- Create API endpoint
- Add UI display in Document Management section
- Add Refresh Stats button

### Phase 2: Model Architecture Display (Low effort, Medium value)
- Create architecture info aggregation
- Add API endpoint
- Add read-only UI section
- Include health status indicators

### Phase 3: Processing Options (Medium effort, High value)
- Add new settings fields
- Create processing options API
- Modify upload flow to accept options
- Update Docling pipeline configuration
- Add UI accordion with controls

### Phase 4: Model Configuration (Medium effort, Medium value)
- Add Ollama model discovery
- Create model switching API
- Implement runtime model change
- Add persistence layer
- Add UI dropdown and apply button

---

## 6. Database Schema Changes

### New Table: admin_settings
```sql
CREATE TABLE admin_settings (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,  -- JSON encoded
    updated_by TEXT REFERENCES users(id),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Initial settings
INSERT INTO admin_settings (key, value) VALUES
    ('chat_model', '"qwen3:8b"'),
    ('enable_ocr', 'true'),
    ('force_full_page_ocr', 'false'),
    ('ocr_language', '"eng"'),
    ('table_extraction_mode', '"accurate"'),
    ('include_image_descriptions', 'true');
```

---

## 7. Open Questions

1. **Processing Options Scope**: Per-upload vs global default with override?
2. **Model Switch Behavior**: Immediate or queued for next request?
3. **Clear Knowledge Base**: Should it require 2FA/password confirmation?
4. **Statistics Refresh**: Auto-poll or manual only?
5. **Embedding Model**: Should it be displayed but marked as "not changeable"?

---

## 8. Acceptance Criteria

### Processing Options
- [ ] Admin can toggle OCR on/off before upload
- [ ] Admin can select OCR language from dropdown
- [ ] Admin can choose table extraction mode
- [ ] Settings are passed to processing pipeline
- [ ] Settings persist between sessions (if global mode)

### Knowledge Base Statistics
- [ ] Total chunks displays correctly
- [ ] Unique files count is accurate
- [ ] File list shows with truncation
- [ ] Refresh Stats updates display
- [ ] Clear Knowledge Base requires confirmation
- [ ] Clear removes all vectors

### Model Configuration
- [ ] Available models populated from Ollama
- [ ] Current model displayed
- [ ] Model change takes effect on next request
- [ ] Invalid model name shows error
- [ ] Model change is audited

### Model Architecture
- [ ] All components displayed with correct info
- [ ] Health status shown with visual indicator
- [ ] OCR availability detected correctly
- [ ] Profile (laptop/spark) displayed
