# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Ready RAG is an enterprise RAG (Retrieval-Augmented Generation) system for NVIDIA DGX Spark. It processes documents using Docling, stores vectors in ChromaDB (migrating to Qdrant), and uses Ollama for LLM inference. All components run locally for air-gap deployment.

**Current Status:** v0.4.1 - Transitioning from Gradio standalone to FastAPI + Gradio backend architecture (target: Feb 13, 2026).

## Development Environment

**IMPORTANT: Use `requirements-wsl.txt` for all development work.**

| File | Purpose | When to Use |
|------|---------|-------------|
| `requirements-wsl.txt` | WSL2/Linux development with Ollama + Qdrant | **Always use this for development** |
| `requirements-api.txt` | API-only testing (auth, users, tags) | Minimal testing without RAG |
| `requirements-spark.txt` | DGX Spark production with Docling + Qdrant | **Production on DGX Spark** |
| `requirements.txt` | Legacy (references chromadb) | Deprecated - use requirements-spark.txt |

```bash
# Correct way to install dependencies
pip install -r requirements-wsl.txt

# DO NOT use requirements.txt for development - it has chromadb which requires onnxruntime
```

**Active Development Path:** `/home/jjob/projects/VE-RAG-System` (WSL2 Ubuntu 24.04)

## Common Commands

```bash
# Activate environment
cd ~/projects/VE-RAG-System
source .venv/bin/activate

# Run FastAPI backend (new architecture)
python -m uvicorn ai_ready_rag.main:app --reload --port 8000

# Run tests
pytest tests/ -v

# Lint and format
ruff check ai_ready_rag tests
ruff format ai_ready_rag tests

# Legacy Gradio app (old monolithic)
python app.py
```

**Environment Variables** (defaults work for local development):
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

## Git Workflow

**Main stays green at all times.** All development happens on feature branches.

### Branch Strategy
```
main (protected - always green)
├── feat/issue-XXX-description   # New features
├── fix/issue-XXX-description    # Bug fixes
├── chore/description            # Maintenance tasks
└── test/description             # Test additions
```

### Workflow
1. **Create branch** from main: `git checkout -b feat/issue-XXX-description`
2. **Develop** on branch with commits
3. **Run tests** before merge: `pytest tests/ -v`
4. **Merge to main** only when all tests pass
5. **Delete branch** after merge

### Rules
- **NEVER commit directly to main** (except emergency fixes)
- **NEVER push broken code to main**
- **Run tests before every merge to main**
- Branch names should include issue number when applicable

### Commands
```bash
# Create feature branch
git checkout main
git checkout -b feat/issue-XXX-description

# After development, merge to main
git checkout main
git merge feat/issue-XXX-description
git branch -d feat/issue-XXX-description
```

### Spec Workflow
Specs require review before committing. **Do not commit draft specs.**

1. **Create branch**: `git checkout -b docs/spec-name`
2. **Draft spec** in `specs/` directory
3. **Share for review** (do NOT commit yet)
4. **Revise based on feedback**
5. **Commit only finalized spec**
6. **Merge to main**

```bash
# Spec workflow
git checkout -b docs/rag-service-spec
# ... draft spec, get review, revise ...
git add specs/RAG_SERVICE.md
git commit -m "docs: Add RAG Service specification (finalized)"
git checkout main && git merge docs/rag-service-spec
```

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
