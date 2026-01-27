# AI Ready RAG - Architecture Decision Record

**Version:** 0.4.1
**Date:** January 27, 2026
**Status:** Approved

---

## Overview

This document captures the key architectural decisions for AI Ready RAG and the reasoning behind them.

---

## ADR-001: FastAPI + Gradio Backend Architecture

### Decision
Use **FastAPI as the primary backend framework** with **Gradio embedded as a sub-application** for the chat UI.

### Context
The initial prototype used Gradio standalone (port 8501) for rapid development. As we move to production with enterprise authentication, RBAC, REST APIs, and audit logging, we need a more robust backend architecture.

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **Gradio Standalone** | Simple, fast to build | No real session management, limited auth, no REST API support, hard to add middleware |
| **FastAPI + Gradio** | Full REST API, proper middleware, session/JWT support, Gradio for ML UI | More complex setup, two frameworks to maintain |
| **FastAPI + React/Next.js** | Maximum flexibility, modern frontend | Complete rewrite, longer timeline, more frontend expertise needed |

### Decision Rationale

1. **Authentication Requirements**: We need JWT-based sessions, secure cookies, RBAC middleware, and admin password reset flows. Gradio's built-in auth is insufficient for enterprise use.

2. **REST API Surface**: The PRD specifies REST endpoints for chat, documents, tags, users, and system health. FastAPI provides automatic OpenAPI docs and type safety.

3. **Middleware Needs**: Access control checks must run BEFORE retrieval/citation to prevent data leakage. FastAPI's dependency injection and middleware system handles this cleanly.

4. **Audit Trail**: Every action needs logging with configurable verbosity. FastAPI middleware can intercept all requests consistently.

5. **Timeline Preservation**: Embedding Gradio in FastAPI preserves our existing UI work while adding backend capabilities. A full React rewrite would miss our deadline.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        NGINX (optional)                          │
│                    (HTTPS termination, :443)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                         │
│                         (Port 8000)                              │
├─────────────────────────────────────────────────────────────────┤
│  Middleware Stack:                                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │   CORS      │ │   Auth      │ │   Audit     │               │
│  │  Middleware │ │  Middleware │ │   Logger    │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│  Routes:                                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ /api/auth/*     - Login, logout, password reset          │   │
│  │ /api/chat/*     - Chat sessions and messages             │   │
│  │ /api/documents/*- Upload, list, delete, tag              │   │
│  │ /api/tags/*     - Tag CRUD                               │   │
│  │ /api/users/*    - User management (admin)                │   │
│  │ /api/admin/*    - System settings, audit logs            │   │
│  │ /api/health     - Health check endpoint                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ /app/*          - Gradio UI (mounted sub-app)            │   │
│  │ /setup          - First-time setup wizard                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                │               │               │
        ┌───────┴───────┐       │       ┌───────┴───────┐
        ▼               ▼       ▼       ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   SQLite     │ │   Qdrant     │ │   Ollama     │ │   Docling    │
│  (Users,     │ │  (Vectors)   │ │   (LLM)      │ │  (Parsing)   │
│  Sessions,   │ │   :6333      │ │  :11434      │ │              │
│  Audit)      │ │              │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

### Implementation Notes

```python
# main.py structure
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

app = FastAPI(title="AI Ready RAG", version="0.4.1")

# Middleware
app.add_middleware(CORSMiddleware, ...)
app.add_middleware(AuditLogMiddleware, ...)

# API routes
app.include_router(auth_router, prefix="/api/auth")
app.include_router(chat_router, prefix="/api/chat")
app.include_router(documents_router, prefix="/api/documents")
# ... etc

# Mount Gradio
gradio_app = gr.Blocks(...)
app = gr.mount_gradio_app(app, gradio_app, path="/app")
```

### Consequences

**Positive:**
- Proper enterprise auth with JWT sessions
- Clean REST API with OpenAPI documentation
- Consistent middleware for access control and audit
- Gradio UI preserved, minimal rework

**Negative:**
- Slightly more complex deployment
- Two frameworks to understand
- Gradio state management needs care when embedded

### Status
**Accepted** - January 27, 2026

---

## ADR-002: Qdrant as Vector Database

### Decision
Use **Qdrant** instead of ChromaDB for vector storage.

### Context
The prototype uses ChromaDB, which is simple and embedded. However, production requirements include tag-based filtering, potential GPU acceleration, and scale to 50-200 users with potentially millions of document chunks.

### Options Considered

| Database | Scale | Filtering | GPU | Air-Gap | Effort |
|----------|-------|-----------|-----|---------|--------|
| ChromaDB | <10M vectors | Basic | No | Yes | Current |
| **Qdrant** | 100M+ vectors | Excellent | Yes | Yes | Medium |
| Milvus | Billions | Excellent | Yes | Yes | High |
| pgvector | 10M+ | SQL-based | No | Yes | Medium |

### Decision Rationale

1. **Tag-Based Filtering**: Qdrant's payload filtering is designed for exactly our use case - filtering vectors by metadata (tags) at query time. ChromaDB's where clause is limited.

2. **Performance**: Qdrant is Rust-based with excellent performance characteristics. GPU acceleration available for DGX Spark.

3. **Production Ready**: Qdrant has better clustering, persistence, and recovery compared to ChromaDB.

4. **Docker Native**: Official Docker image, easy to deploy alongside our application.

5. **Active Development**: Strong community, frequent releases, good documentation.

### Migration Strategy

Since we're changing DBs before production, we'll implement cleanly:

1. Create a `VectorStore` abstraction layer
2. Implement `QdrantVectorStore`
3. No migration needed - rebuild index from source documents

### Consequences

**Positive:**
- Native tag filtering for access control
- Scales to production workloads
- GPU acceleration potential
- Better operational characteristics

**Negative:**
- Additional container to manage
- Learning curve for Qdrant API
- Index rebuild required from prototype

### Status
**Accepted** - January 27, 2026

---

## ADR-003: SQLite for Application Data

### Decision
Use **SQLite** for users, sessions, chat history, tags, and audit logs.

### Context
We need persistent storage for application data that works in air-gapped environments without additional infrastructure.

### Decision Rationale

1. **Zero Infrastructure**: No separate database server needed
2. **Air-Gap Friendly**: Single file, easy to backup/restore
3. **Sufficient Scale**: 50-200 users with chat history is well within SQLite's capabilities
4. **Portable**: Database travels with the application
5. **WAL Mode**: Handles concurrent reads well

### Configuration

```python
# Enable WAL mode for better concurrency
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-64000;  # 64MB cache
```

### Future Consideration
If scale exceeds SQLite's write throughput (unlikely for this use case), PostgreSQL can be added as an option with the same schema.

### Status
**Accepted** - January 27, 2026

---

## ADR-004: Access Control Before Retrieval

### Decision
Run access control checks **before** vector retrieval and citation generation.

### Context
The cite-or-route system returns document names and locations in citations. If a user queries about a topic they don't have access to, we must not leak document names or content.

### Implementation

```
User Query
    │
    ▼
┌─────────────────────┐
│ 1. Get User's Tags  │  ← From session/JWT
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 2. Build Qdrant     │  ← Filter: tags IN user_tags OR "public"
│    Filter           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 3. Vector Search    │  ← Only searches accessible documents
│    (filtered)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 4. Generate Answer  │  ← LLM sees only accessible chunks
│    + Citations      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 5. Route if Needed  │  ← Low confidence or no results
└─────────────────────┘
```

### Key Principle
**The LLM never sees documents the user cannot access.** This is enforced at the vector search layer, not post-filtered.

### Status
**Accepted** - January 27, 2026

---

## ADR-005: Configurable Audit Logging

### Decision
Implement audit logging with **three configurable levels**: Essential, Comprehensive, and Full Debug.

### Context
During development and initial deployment, we need detailed logging to verify the system works correctly. In production, we may want to reduce log volume while maintaining compliance.

### Audit Levels

| Level | What's Logged |
|-------|---------------|
| **Essential** | Logins (success/fail), document access, admin actions (user/tag CRUD) |
| **Comprehensive** | Essential + all queries, all document views, failed access attempts |
| **Full Debug** | Comprehensive + retrieval details, confidence scores, routing decisions, chunk IDs |

### Configuration

```yaml
# config/audit.yml
audit:
  level: full_debug  # essential | comprehensive | full_debug
  retention_days: 90
  include_query_text: true  # Set false for privacy
```

### Storage

Audit logs stored in SQLite `audit_logs` table with structured JSON payload:

```sql
CREATE TABLE audit_logs (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level TEXT NOT NULL,  -- essential, comprehensive, full_debug
    user_id TEXT,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    details TEXT,  -- JSON payload
    ip_address TEXT,
    user_agent TEXT
);

CREATE INDEX idx_audit_timestamp ON audit_logs(timestamp);
CREATE INDEX idx_audit_user ON audit_logs(user_id);
CREATE INDEX idx_audit_action ON audit_logs(action);
```

### Status
**Accepted** - January 27, 2026

---

## Summary of Key Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Backend | FastAPI + Gradio | Enterprise auth, REST API, middleware support |
| Vector DB | Qdrant | Tag filtering, scale, performance |
| App DB | SQLite | Zero infrastructure, air-gap friendly |
| Access Control | Pre-retrieval filtering | Prevent data leakage |
| Audit | 3-level configurable | Debug now, reduce later |

---

*This document should be updated as architectural decisions evolve.*
