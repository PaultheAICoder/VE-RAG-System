# AI Ready RAG - Component Versions

**Application Version:** 0.4.1
**Last Updated:** January 27, 2026

---

## Pinned Versions

All components are pinned to specific versions for reproducibility and air-gapped deployment.

### Core Application

| Component | Version | Purpose | Notes |
|-----------|---------|---------|-------|
| Python | 3.12.x | Runtime | Use 3.12.4 or later |
| FastAPI | 0.115.x | Backend framework | API + middleware |
| Gradio | 6.3.0 | Chat UI | Embedded in FastAPI |
| Pydantic | 2.10.x | Data validation | FastAPI dependency |
| SQLAlchemy | 2.0.x | Database ORM | SQLite interface |
| python-jose | 3.3.0 | JWT handling | Auth tokens |
| passlib[bcrypt] | 1.7.4 | Password hashing | Local auth |
| httpx | 0.28.x | HTTP client | Ollama communication |

### Document Processing

| Component | Version | Purpose | Notes |
|-----------|---------|---------|-------|
| Docling | 2.68.0 | Document parsing | PDF, DOCX, etc. |
| Tesseract | 5.x | OCR engine | System package |
| EasyOCR | 1.7.x | Backup OCR | GPU-accelerated |
| Pillow | 11.x | Image handling | Docling dependency |

### Vector Database

| Component | Version | Purpose | Notes |
|-----------|---------|---------|-------|
| Qdrant | 1.13.x | Vector storage | Docker image |
| qdrant-client | 1.13.x | Python client | Matches server |

### LLM Runtime

| Component | Version | Purpose | Notes |
|-----------|---------|---------|-------|
| Ollama | 0.5.x | LLM server | GPU inference |
| qwen3:8b | latest | Chat model | ~5GB VRAM |
| nomic-embed-text | latest | Embeddings | 768 dimensions |

### Infrastructure

| Component | Version | Purpose | Notes |
|-----------|---------|---------|-------|
| Docker | 27.x | Containerization | With compose v2 |
| Docker Compose | 2.32.x | Orchestration | GPU support |
| Ubuntu | 22.04 LTS | Base OS | DGX Spark default |
| NVIDIA Driver | 550.x+ | GPU support | CUDA 12.4+ |
| NVIDIA Container Toolkit | 1.17.x | GPU in Docker | Required |

---

## Docker Image Tags

For air-gapped deployment, pull and save these exact images:

```bash
# Application (built locally)
ai-ready-rag:0.4.1

# Dependencies
qdrant/qdrant:v1.13.2
ollama/ollama:0.5.7

# Base images
python:3.12-slim-bookworm
```

### Pre-staging for Air-Gap

```bash
# Save images for USB transfer
docker save -o ai-ready-rag-images.tar \
  ai-ready-rag:0.4.1 \
  qdrant/qdrant:v1.13.2 \
  ollama/ollama:0.5.7

# Save Ollama models
ollama pull qwen3:8b
ollama pull nomic-embed-text
# Models stored in ~/.ollama/models/
```

---

## Version Display

The application displays version information in the UI footer:

```
AI Ready RAG v0.4.1 | Last commit: 2026-01-27 14:32:05
```

This is generated from:
- `VERSION` file in repository root
- Git commit timestamp (or build timestamp for releases)

### Implementation

```python
# version.py
import subprocess
from datetime import datetime

VERSION = "0.4.1"

def get_commit_timestamp():
    try:
        timestamp = subprocess.check_output(
            ["git", "log", "-1", "--format=%ci"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return timestamp[:19]  # YYYY-MM-DD HH:MM:SS
    except:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_version_string():
    return f"AI Ready RAG v{VERSION} | Last commit: {get_commit_timestamp()}"
```

---

## Upgrade Policy

1. **Patch versions** (0.4.x): Bug fixes, no breaking changes
2. **Minor versions** (0.x.0): New features, backward compatible
3. **Major versions** (x.0.0): Breaking changes, migration required

### Air-Gap Update Process

1. Download new version package on internet-connected machine
2. Verify checksums
3. Transfer via USB
4. Run `apply-update.sh`

---

## Compatibility Matrix

| AI Ready RAG | Qdrant | Ollama | Python |
|--------------|--------|--------|--------|
| 0.4.x | 1.13.x | 0.5.x | 3.12 |
| 0.5.x (planned) | 1.14.x | 0.6.x | 3.12 |

---

*Update this document when changing any component version.*
