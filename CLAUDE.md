# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Ready RAG is an enterprise RAG (Retrieval-Augmented Generation) system for NVIDIA DGX Spark. It processes documents using Docling, stores vectors in ChromaDB (migrating to Qdrant), and uses Ollama for LLM inference. All components run locally for air-gap deployment.

**Current Status:** v0.4.1 - Transitioning from Gradio standalone to FastAPI + Gradio backend architecture (target: Feb 13, 2026).

## Common Commands

```bash
# First-time setup (creates venv, installs deps, pulls Ollama models)
./setup.sh

# Start the application (serves at http://localhost:8501)
./start.sh

# Manual run (after activating venv)
source .venv/bin/activate
python app.py
```

**Environment Variables** (set in `start.sh` or export before running):
- `OLLAMA_BASE_URL` - Ollama server (default: http://localhost:11434)
- `EMBEDDING_MODEL` - Embedding model (default: nomic-embed-text)
- `CHAT_MODEL` - Chat model (default: qwen3:8b, options: llama3.2, deepseek-r1:32b)
- `CHROMA_PERSIST_DIR` - Vector storage path (default: ./chroma_db)

## Architecture

### Current State (v0.4.1)
Single monolithic `app.py` with Gradio UI on port 8501. Direct integration with Docling, ChromaDB, and Ollama.

### Planned Architecture (see DEVELOPMENT_PLANS.md)
FastAPI backend with Gradio mounted as sub-app:
```
FastAPI (:8000)
├── /api/auth/*      - JWT authentication
├── /api/chat/*      - Chat sessions & messages
├── /api/documents/* - Upload, list, delete, tag
├── /api/tags/*      - Tag CRUD
├── /api/users/*     - User management (admin)
├── /api/admin/*     - System settings, audit logs
├── /app/*           - Gradio UI (mounted)
└── Middleware: CORS → Auth → Access Control → Audit
```

**Key Components:**
- **SQLite** - Users, sessions, chat history, tags, audit logs (WAL mode)
- **Qdrant** - Vector storage replacing ChromaDB (tag-based filtering for access control)
- **Ollama** - LLM inference (localhost:11434)
- **Docling** - Document parsing with OCR, table extraction, semantic chunking

### Data Flow
1. Document upload → Docling parsing → HybridChunker → Embeddings → Vector store
2. User query → Query expansion → Vector search (filtered by user tags) → LLM response with citations
3. Confidence scoring determines CITE vs ROUTE to human

## Key Architectural Decisions (from docs/ARCHITECTURE.md)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backend | FastAPI + Gradio | Enterprise auth, REST API, middleware for access control |
| Vector DB | Qdrant (replacing ChromaDB) | Superior tag filtering, GPU acceleration |
| App DB | SQLite | Zero infrastructure, air-gap friendly |
| Access Control | Pre-retrieval filtering | User tags filter vectors BEFORE search; LLM never sees inaccessible docs |
| Audit | 3-level configurable | essential, comprehensive, full_debug |

## Technology Stack

- **Python 3.12+**
- **Document Processing:** Docling 2.68.0, Tesseract/EasyOCR
- **Vector:** ChromaDB 0.5.0 (current), Qdrant 1.13.x (planned)
- **LLM:** LangChain 0.3.0+, LangChain-Ollama
- **UI:** Gradio 5.0+
- **Backend:** FastAPI 0.115.x (planned)
- **Infrastructure:** Docker, NVIDIA Container Toolkit

## Planned Project Structure (for refactoring)

```
ai_ready_rag/
├── main.py              # FastAPI entry point
├── config.py            # Configuration management
├── api/                 # Route handlers
├── core/                # Security, dependencies, exceptions
├── middleware/          # Auth, audit, access control
├── db/                  # SQLite, models, migrations
├── services/            # Business logic (auth, chat, document, vector, rag, audit)
├── ui/                  # Gradio app and components
└── utils/
```

## Important Documentation

- **docs/ARCHITECTURE.md** - Architecture Decision Records (ADRs)
- **DEVELOPMENT_PLANS.md** - 17-day development timeline with detailed task breakdown
- **PRD_v0.80.md** - Current product requirements
- **versions.md** - Component version matrix for reproducibility

## File Maintenance Rules

### DEVELOPMENT_PLANS.md
When modifying this file, you **must** update the Change Log section:
1. Increment the version number (e.g., 0.4.2 → 0.4.3)
2. Update the "Version" field in the header metadata
3. Add a new row to the Change Log table with:
   - **Version**: New version number
   - **Date**: Current date (YYYY-MM-DD format)
   - **Author**: Leave as `—` unless specified by user
   - **Changes**: Brief summary of what was added/modified/removed
