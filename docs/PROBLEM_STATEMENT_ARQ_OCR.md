# Problem Statement: PDF Processing Fails Through ARQ Worker with Stale tesserocr Error

**Date:** 2026-02-11
**Severity:** P1 — All PDF document processing blocked on Spark
**System:** AI Ready RAG v0.5.0 on DGX Spark
**Docling:** 2.71.0 | **Tesseract CLI:** 5.3.4 | **Python:** 3.12

---

## Summary

All PDF documents fail during processing with the error:

> tesserocr is not correctly configured. No language models have been detected.
> Please ensure that the TESSDATA_PREFIX envvar points to tesseract languages dir.

This error originates from Docling's **C-binding OCR model** (`TesseractOcrModel`), but our code explicitly configures the **CLI OCR model** (`TesseractCliOcrOptions`), and the `tesserocr` Python package is **not installed**.

The failure occurs **only** when documents are processed through the embedded ARQ worker. Direct Python calls to the same code path succeed every time.

---

## Environment

| Component | Value |
|-----------|-------|
| Server | DGX Spark (VTMP-ENT-SPK) |
| Docling | 2.71.0 |
| Tesseract CLI | 5.3.4 (`/usr/bin/tesseract`) |
| tessdata | `/usr/share/tesseract-ocr/5/tessdata/` (eng.traineddata present) |
| tesserocr Python package | **NOT installed** (confirmed: `ModuleNotFoundError`) |
| TESSDATA_PREFIX | `/usr/share/tesseract-ocr/5/tessdata` |
| ARQ worker | Embedded in FastAPI lifespan, `max_jobs=2` |
| Affected documents | 41 PDFs (100% failure rate); 5 non-PDF docs (xlsx/docx) succeed |

---

## Evidence

### 1. Our OCR configuration IS correct (confirmed by debug logs)

```
OCR config: mode=cli, TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata, lang=eng, force_ocr=False
Creating DocumentConverter: do_ocr=True, ocr_options_type=TesseractCliOcrOptions, ocr_options_kind=tesseract
Converter pipeline check: do_ocr=True, ocr_type=TesseractCliOcrOptions, ocr_kind=tesseract
```

At every checkpoint — converter creation and conversion time — the options are `TesseractCliOcrOptions` (kind=`tesseract`), not `TesseractOcrOptions` (kind=`tesserocr`).

### 2. tesserocr is NOT installed

```bash
$ pip list | grep tesserocr     # (no output)
$ python -c "import tesserocr"  # ModuleNotFoundError: No module named 'tesserocr'
$ find /srv/VE-RAG-System/.venv -name '*tesserocr*'  # (no output, except .pyc in docling)
```

The `tesserocr` package is completely absent. The error string only exists inside Docling's `TesseractOcrModel` class (line 59 of `tesseract_ocr_model.py`), which requires `import tesserocr` to succeed before reaching that line.

### 3. Tesseract CLI IS executing successfully

Server logs show the CLI model running `tesseract` subprocess commands:

```
command: tesseract --psm 0 -l osd /tmp/tmpXXXXXXXX.png stdout
command: tesseract -l eng --tessdata-dir /usr/share/tesseract-ocr/5/tessdata /tmp/tmpXXXXXXXX.png stdout tsv
```

Multiple pages are being processed via CLI subprocess. The `TesseractOcrCliModel` IS being instantiated and IS working.

### 4. The error message CANNOT originate from the current code path

| Model class | Error on failure | Requires tesserocr? |
|-------------|-----------------|---------------------|
| `TesseractOcrModel` (C-binding) | "tesserocr is not correctly configured" | Yes — `import tesserocr` at line 68 |
| `TesseractOcrCliModel` (subprocess) | "Tesseract is not available, aborting" | No — uses `subprocess.Popen` |

The error we see ("tesserocr is not correctly configured") can only originate from `TesseractOcrModel.__init__()`. But:
- The factory maps `TesseractCliOcrOptions` → `TesseractOcrCliModel` (deterministic type-based lookup)
- `import tesserocr` would raise `ImportError` before reaching the "not correctly configured" line
- Our debug logs confirm CLI options at every checkpoint

### 5. Documents fail in ~450ms

Processing time is 400-600ms. This is too fast for multi-page OCR (CLI tesseract takes seconds per page) but consistent with pipeline initialization overhead + early failure.

### 6. Direct calls succeed; ARQ worker fails

| Execution context | Result |
|-------------------|--------|
| Direct `ProcessingService.process_document()` in Python REPL | **Success** |
| `asyncio.to_thread(chunker.chunk_document, ...)` in Python REPL | **Success** |
| Concurrent test (2 parallel `asyncio.to_thread` calls) | **Success** |
| Same code called via ARQ embedded worker task | **Failure** |

---

## Architecture (How Processing Works)

```
User clicks "Reprocess"
  → API sets document.status = "pending"
  → Enqueues ARQ job to Redis
  → ARQ EmbeddedWorker picks up job
    → Creates new ProcessingService (no cached state)
    → ProcessingService.process_document()
      → get_chunker() → new DoclingChunker(enable_ocr=True)
      → DoclingChunker._create_converter()
        → Sets TesseractCliOcrOptions on PdfPipelineOptions
        → Creates DocumentConverter
      → asyncio.to_thread(chunker.chunk_document, file_path)
        → converter.convert(path)  ← Error happens here
          → StandardPdfPipeline.__init__()
            → _init_models() → _make_ocr_model()
              → factory.create_instance(options=TesseractCliOcrOptions)
              → TesseractOcrCliModel(...)  ← Should be this
    → On failure: sets document.status = "failed", error_message = str(exception)
```

### Key Code Locations

- **ARQ worker:** `ai_ready_rag/workers/arq_worker.py` — `EmbeddedArqWorker` wraps ARQ `Worker` as asyncio.Task
- **ARQ task:** `ai_ready_rag/workers/tasks/document.py` — `process_document()` creates fresh `ProcessingService` per job
- **Processing:** `ai_ready_rag/services/processing_service.py` — calls `chunker.chunk_document()` via `asyncio.to_thread`
- **Chunker:** `ai_ready_rag/services/chunker_docling.py` — lazy-creates `DocumentConverter` with `TesseractCliOcrOptions`
- **Docling factory:** `docling/models/factories/base_factory.py:56` — `self._classes[type(options)]` lookup
- **Error source:** `docling/models/stages/ocr/tesseract_ocr_model.py:59` — the only location of this error string

---

## Contradictions That Need Resolution

1. **Error requires `import tesserocr` to succeed, but tesserocr is not installed.** If `import tesserocr` fails, the error message would be "tesserocr is not correctly installed" (line 51), not "No language models have been detected" (line 59). Yet we see the line-59 message.

2. **Debug logs confirm CLI options, but error comes from C-binding model.** Every logging checkpoint shows `TesseractCliOcrOptions`. The factory lookup is deterministic (`type(options)` → class). There is no fallback.

3. **Tesseract CLI commands appear in server logs.** The `TesseractOcrCliModel` IS running `tesseract` subprocess calls successfully. Yet documents end up "failed" with the C-binding error.

4. **Direct Python calls succeed with identical code.** Same `ProcessingService`, same `DoclingChunker`, same `asyncio.to_thread` — only difference is execution context (ARQ task vs direct call).

---

## Theories to Investigate

### A. Stale error message from previous processing (most likely)
The document's `error_message` might not be properly reset, or a race condition between the reprocess endpoint and a previously-queued ARQ job could restore the old error. Check if a stale ARQ job from before the fix is re-running and overwriting the reset status.

### B. Duplicate ARQ jobs for the same document
Server logs show the same document enqueued multiple times (once from UI bulk reprocess, once from test script). If a stale job with old code/state processes first, it would set the error, and the second job might not run because the document is already in "failed" state.

### C. Docling pipeline caching
`get_ocr_factory()` is `@lru_cache`-decorated. If the factory was first created in a context where it resolved differently, the cached factory might return a stale model class mapping.

### D. Python `__pycache__` or `.pyc` for `chunker_docling.py`
If a stale bytecode file from before the `TesseractCliOcrOptions` fix (PR #230) is being loaded by the server process, the old code (using `TesseractOcrOptions`) would run despite the source being updated. `__pycache__` was cleared, but maybe not from all locations.

### E. Thread-local or subprocess environment difference in ARQ context
The embedded ARQ worker runs tasks as coroutines on FastAPI's event loop. `asyncio.to_thread` uses the default `ThreadPoolExecutor`. If the thread inherits a different environment or module state, Docling's pipeline might behave differently.

---

## Reproduction Steps

1. SSH to Spark: `ssh spark`
2. Server running at `/srv/VE-RAG-System` on port 8502
3. Login: `admin@test.com` / `npassword2002!`
4. Find any PDF with status "failed" or "pending"
5. POST `/api/documents/{id}/reprocess`
6. Wait 20 seconds, GET `/api/documents/{id}`
7. Observe: `status=failed`, `error_message` contains "tesserocr is not correctly configured"

**Direct call succeeds:**
```python
# On Spark, in Python REPL with venv activated
from ai_ready_rag.config import get_settings
from ai_ready_rag.services.factory import get_chunker
settings = get_settings()
chunker = get_chunker(settings)
result = chunker.chunk_document("/path/to/any/pdf")  # Succeeds
```

---

## Immediate Workaround Options

1. **Disable OCR for these documents:** Set `enable_ocr=False` to skip the OCR pipeline entirely. Native-text PDFs will still extract text; scanned PDFs will produce empty results.

2. **Bypass ARQ — use BackgroundTasks fallback:** Stop Redis so the system falls back to FastAPI's `BackgroundTasks` (synchronous processing). This removes the ARQ worker from the equation.

3. **Process via direct Python script:** Write a one-off script that calls `ProcessingService.process_document()` directly for each pending document (proven to work).
