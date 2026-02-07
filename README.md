# AI Ready RAG

Enterprise Retrieval-Augmented Generation system for NVIDIA DGX Spark. Processes documents using Docling, stores vectors in Qdrant, and uses Ollama for local LLM inference. All components run locally for air-gap deployment.

## Architecture

```
React SPA (:5173 dev / :8502 prod)     FastAPI Backend (:8502)
├── /chat        - Chat interface       ├── /api/auth/*      - JWT auth
├── /admin       - Admin dashboard      ├── /api/chat/*      - Sessions + SSE
├── /login       - Authentication       ├── /api/documents/* - Upload + process
                                        ├── /api/tags/*      - Tag CRUD
                                        ├── /api/users/*     - User management
                                        ├── /api/admin/*     - Settings + cache
                                        └── /api/health      - Health check
```

**Key Components:**
- **SQLite** — Users, sessions, chat history, tags, cache, audit logs (WAL mode)
- **Qdrant** — Vector storage with tag-based filtering for access control
- **Ollama** — Local LLM inference (localhost:11434)
- **Docling** — Document parsing with OCR, table extraction, semantic chunking

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+ (for frontend)
- [Ollama](https://ollama.ai/) running locally
- [Qdrant](https://qdrant.tech/) running locally (Docker recommended)

### Backend Setup

```bash
cd ~/projects/VE-RAG-System
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-wsl.txt

# Run backend
python -m uvicorn ai_ready_rag.main:app --host 0.0.0.0 --port 8502
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev          # Development server (:5173)
npm run build        # Production build → dist/
```

### First-Time Setup

1. Start the backend
2. Navigate to `http://localhost:8502/api/setup` (or the frontend setup wizard)
3. Create an admin account
4. Upload documents and assign tags

## Development

```bash
# Activate environment
source .venv/bin/activate

# Run tests
pytest tests/ -v

# Lint and format
ruff check ai_ready_rag tests
ruff format ai_ready_rag tests

# Run with auto-reload
python -m uvicorn ai_ready_rag.main:app --host 0.0.0.0 --port 8502 --reload
```

**Requirements files:**

| File | Purpose |
|------|---------|
| `requirements-wsl.txt` | WSL2/Linux development (Ollama + Qdrant) |
| `requirements-spark.txt` | DGX Spark production (Docling + Qdrant) |
| `requirements-api.txt` | Minimal API testing (no RAG) |

## Documentation

- **[Architecture Decisions](docs/ARCHITECTURE.md)** — ADRs for key technical choices
- **[WSL2 Setup Guide](docs/WSL2_SETUP.md)** — Complete development environment setup
- **[CLAUDE.md](CLAUDE.md)** — AI assistant instructions for this codebase

## License

Proprietary - NVIDIA DGX Spark deployment.
