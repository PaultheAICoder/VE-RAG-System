# AI Ready RAG - Architecture Decision Record

**Version:** 0.5.0-rc
**Date:** February 6, 2026
**Status:** Approved

---

## Overview

This document captures the key architectural decisions for AI Ready RAG and the reasoning behind them.

---

## ADR-001: FastAPI + React Frontend Architecture

### Decision
Use **FastAPI as the backend** with **React as the frontend**. Gradio was used for the initial prototype but has been fully replaced.

### Context
The initial prototype used Gradio standalone for rapid development. As the system evolved with enterprise authentication, RBAC, REST APIs, and audit logging, we migrated to FastAPI backend with a React SPA frontend.

### Architecture Evolution

| Phase | Stack | Status |
|-------|-------|--------|
| Prototype | Gradio standalone (:8501) | Deprecated |
| v0.3 | FastAPI + embedded Gradio | Deprecated |
| **v0.4+** | **FastAPI + React SPA** | **Current** |

### Decision Rationale

1. **Authentication**: JWT-based sessions, secure cookies, RBAC middleware, admin password reset. React provides full control over auth UX.

2. **REST API Surface**: REST endpoints for chat, documents, tags, users, health. FastAPI provides OpenAPI docs and type safety.

3. **Middleware**: Access control checks run BEFORE retrieval/citation. FastAPI dependency injection handles this cleanly.

4. **Frontend Flexibility**: React provides full control over UI/UX, state management, and real-time features (SSE for streaming responses).

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      React SPA (:5173 dev)                       │
│               (built to frontend/dist for production)            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ /chat        - Chat interface with streaming             │   │
│  │ /admin       - Admin dashboard (settings, docs, users)   │   │
│  │ /login       - Authentication                            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                         │
│                         (Port 8502)                              │
├─────────────────────────────────────────────────────────────────┤
│  Middleware: CORS → Auth → Access Control → Audit               │
├─────────────────────────────────────────────────────────────────┤
│  Routes:                                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ /api/auth/*     - Login, logout, password reset          │   │
│  │ /api/chat/*     - Chat sessions and messages (SSE)       │   │
│  │ /api/documents/*- Upload, process, tag, search           │   │
│  │ /api/tags/*     - Tag CRUD                               │   │
│  │ /api/users/*    - User management (admin)                │   │
│  │ /api/admin/*    - Settings, cache, warming, reindex      │   │
│  │ /api/health     - Health check endpoint                  │   │
│  │ /api/setup      - First-time setup wizard                │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Service Layer (Layered Architecture):                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ BaseRepository[T] → Concrete Repos (User, Document, ...) │   │
│  │ BaseService[T, R]  → Domain services                      │   │
│  │ Depends() chain    → Service factories in dependencies.py │   │
│  │ Domain exceptions  → AppError hierarchy                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                │               │               │
        ┌───────┴───────┐       │       ┌───────┴───────┐
        ▼               ▼       ▼       ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   SQLite     │ │   Qdrant     │ │   Ollama     │ │   Docling    │
│  (Users,     │ │  (Vectors)   │ │   (LLM)      │ │  (Parsing)   │
│  Sessions,   │ │   :6333      │ │  :11434      │ │              │
│  Cache,      │ │              │ │              │ │              │
│  Audit)      │ │              │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

### Consequences

**Positive:**
- Full enterprise auth with JWT sessions
- Clean REST API with OpenAPI documentation
- Consistent middleware for access control and audit
- React SPA provides rich, responsive UI
- Layered architecture (repos, services, routes)

**Negative:**
- Two technology stacks (Python + TypeScript)
- Frontend build step required for production

### Status
**Accepted** - Updated February 6, 2026

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
| Backend | FastAPI + React | Enterprise auth, REST API, rich SPA frontend |
| Vector DB | Qdrant | Tag filtering, scale, performance |
| App DB | SQLite | Zero infrastructure, air-gap friendly |
| Access Control | Pre-retrieval filtering | Prevent data leakage |
| Audit | 3-level configurable | Debug now, reduce later |

---

*This document should be updated as architectural decisions evolve.*
