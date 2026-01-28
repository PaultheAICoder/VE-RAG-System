# Project Stack

## Overview

**Project Name**: AI Ready RAG (VE-RAG-System)
**Description**: Enterprise RAG system with authentication and access control for NVIDIA DGX Spark

## Technology Stack

### Backend
- Language: Python 3.12
- Framework: FastAPI 0.115.x
- Database: SQLite (WAL mode)
- ORM: SQLAlchemy 2.0
- Vector DB: Qdrant 1.13.x
- LLM: Ollama (llama3.2, nomic-embed-text)

### Frontend
- Framework: Gradio 5.0+ (mounted on FastAPI)
- Styling: Gradio built-in theming

### Testing
- Framework: pytest 9.0
- Async: pytest-asyncio
- Coverage: pytest-cov
- Linting: Ruff

### Infrastructure
- Container: Docker, Docker Compose
- GPU: NVIDIA Container Toolkit (DGX Spark)
- Dev Environment: WSL2 Ubuntu 24.04

## Directory Structure

```
ai_ready_rag/
├── main.py              # FastAPI entry point
├── config.py            # Configuration management
├── api/                 # Route handlers (auth, users, tags, health, chat)
├── core/                # Security, dependencies, exceptions
├── middleware/          # Auth, audit, access control
├── db/                  # SQLite, models, migrations
├── services/            # Business logic (auth, chat, document, vector, rag)
└── ui/                  # Gradio app and components
```

## Key Commands

```bash
# Activate environment
cd ~/projects/VE-RAG-System
source .venv/bin/activate

# Run tests
pytest

# Run with coverage
pytest --cov=ai_ready_rag --cov-report=term-missing

# Lint and format
ruff check ai_ready_rag tests
ruff format ai_ready_rag tests

# Start dev server
python -m uvicorn ai_ready_rag.main:app --reload --host 0.0.0.0 --port 8000

# Check services
curl http://localhost:6333/collections  # Qdrant
curl http://localhost:11434/api/tags    # Ollama
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DEBUG | true | Enable debug mode |
| DATABASE_URL | sqlite:///./data/ai_ready_rag.db | SQLite path |
| QDRANT_URL | http://localhost:6333 | Qdrant server |
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama server |
| EMBEDDING_MODEL | nomic-embed-text | Embedding model |
| CHAT_MODEL | llama3.2 | Chat model |
| JWT_SECRET_KEY | (required) | JWT signing key |
