# VE-RAG-System Scaffold Alignment Specification

**Version:** 3.0
**Status:** FINAL - Ready for Sign-off
**Author:** Architecture Team
**Date:** 2026-02-06
**Target Release:** v0.5.0

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-05 | Architecture Team | Initial draft |
| 1.1 | 2026-02-05 | Architecture Team | Added Section 14: Configuration Migration to SQLite (49 settings) |
| 1.2 | 2026-02-05 | Architecture Team | Updated to Hybrid Layout: added schemas/, split models/, workers/ |
| 1.3 | 2026-02-05 | Architecture Team | Added Section 15: Unified Background Task System (ARQ + Redis) |
| 1.4 | 2026-02-05 | Architecture Team | Added Section 16: Observability & Logging Strategy |
| 1.5 | 2026-02-05 | Architecture Team | Resolved all 14 open questions (Q1-Q14) |
| 1.6 | 2026-02-05 | Architecture Team | Applied Codex review feedback: FK migration policy table (7.4), config precedence (14.1.1), query logging toggle (16.6), ARQ exception handling (15.4.3.1) |
| 2.0 | 2026-02-06 | Architecture Team | Spec review feedback: Drop ServiceContainer → FastAPI Depends() chain; global error handlers only; drop-and-recreate tables (pre-MVP); defer Alembic + SQLAlchemy 2.0 Mapped[] syntax to post-MVP; trim config migration to 23 settings (5.1+5.2 only); add PRAGMA foreign_keys=ON; add Redis to architecture invariants; reduced effort from 97h → 75h |
| 2.1 | 2026-02-06 | Architecture Team | Engineering team feedback: Import migration checklist with re-export pattern (6.2.7); transaction boundary rules (4.6.1); experimental endpoints excluded from scope (1.2); per-setting config acceptance criteria (12.5); debug override policy with 1h TTL (16.6.1); phase-level test checkpoints (9.4); data safety clause with auto-backup (7.3.1); Redis failure policy with degraded mode (15.3.4); release guardrail for v0.5.0-rc (11.2); password min default 8 for dev |
| 3.0 | 2026-02-06 | Architecture Team | Final consistency pass: reconciled Phase 2 task numbering (6.2 ↔ 11.1 Gantt); fixed Section 6.2 effort (16h → 22h); removed stale Alembic references (7.2, 8.1); corrected deferred settings count (26 → 48); marked FINAL for sign-off |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Business Context](#2-business-context)
3. [Current State Analysis](#3-current-state-analysis)
4. [Target State Architecture](#4-target-state-architecture)
5. [Gap Analysis](#5-gap-analysis)
6. [Detailed Implementation Plan](#6-detailed-implementation-plan)
7. [Migration Strategy](#7-migration-strategy)
8. [Risk Assessment](#8-risk-assessment)
9. [Testing Strategy](#9-testing-strategy)
10. [Rollback Plan](#10-rollback-plan)
11. [Timeline & Milestones](#11-timeline--milestones)
12. [Acceptance Criteria](#12-acceptance-criteria)
13. [Open Questions](#13-open-questions)
14. [Configuration Migration to SQLite](#14-configuration-migration-to-sqlite)
15. [Unified Background Task System (ARQ + Redis)](#15-unified-background-task-system-arq--redis)
16. [Observability & Logging Strategy](#16-observability--logging-strategy)
17. [Appendices](#17-appendices)

---

## 1. Executive Summary

### 1.1 Purpose

This specification defines the refactoring effort to align the VE-RAG-System codebase with the FastAPI Layered Architecture Pattern (the "scaffold pattern"). The goal is to improve maintainability, testability, and code quality while preserving all existing functionality.

### 1.2 Scope

**In Scope:**
- Backend architecture alignment (FastAPI, SQLAlchemy, Services)
- Database schema improvements (foreign keys, mixins, migrations)
- Removal of deprecated Gradio UI code
- Service layer refactoring (BaseService, BaseRepository patterns)
- Exception handling standardization

**Out of Scope:**
- React frontend changes (already well-architected)
- RAG pipeline algorithm changes
- New feature development
- Performance optimization (beyond removing dead code)
- Experimental features (`api/experimental.py`, `services/slide_generator.py`) — excluded from refactor; they continue to work as-is but are not migrated to BaseService/repositories until promoted to stable

### 1.3 Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Dead code (Gradio) | 366 KB / 6,500 lines | 0 |
| HTTPException in services | 10 instances | 0 |
| Missing foreign keys | 3 critical | 0 |
| Test coverage (services) | ~40% | 80% |
| Code duplication (timestamps) | 11 models | 0 (via mixin) |

### 1.4 Estimated Effort

| Phase | Effort | Risk |
|-------|--------|------|
| Phase 1: Critical Fixes | 7 hours | Low |
| Phase 2: Architecture (Hybrid Layout) | 22 hours | Medium |
| Phase 3: Modernization | 4 hours | Low |
| Phase 4: Quality | 8 hours | Low |
| Phase 5: Config Migration to SQLite (High-Priority Only) | 4 hours | Low |
| Phase 6-10: Unified Background Tasks (ARQ) | 24 hours | Medium |
| Phase 11: Observability & Logging | 6 hours | Low |
| **Total** | **75 hours** | **Medium** |

**Phase 2 Breakdown:**
- Schemas directory (8 files): 4 hours
- Split models.py into models/ package (7 files): 3 hours
- Repositories (7 files): 6 hours
- BaseService: 3 hours
- TimestampMixin + mixins.py: 2 hours
- Workers package restructure: 2 hours
- Update imports across codebase: 2 hours

**Phase 3 Breakdown (Reduced):**
- FK Indexes: 1 hour
- Replace lazy loading with constructor injection: 3 hours

**Deferred to post-MVP:**
- Alembic migrations (add when schema stabilizes before production)
- SQLAlchemy 2.0 `Mapped[]` syntax (cosmetic, no functional benefit)
- ServiceContainer (replaced by FastAPI `Depends()` chain)

**Phase 5 Breakdown (Trimmed):**
- Phase 5.1: Security/Policy Settings (6 settings): 2 hours
- Phase 5.2: Feature Flags (2 settings): 1 hour
- Migration script + testing: 1 hour
- Phases 5.3-5.7 deferred to future work (26 settings)

**Phase 6-10 Breakdown (ARQ + Redis):**
- Phase 6: ARQ Infrastructure: 4 hours
- Phase 7: Migrate Document Processing: 6 hours
- Phase 8: Migrate Cache Warming: 6 hours
- Phase 9: Migrate Reindexing: 4 hours
- Phase 10: Jobs API & Cleanup: 4 hours

**Phase 11 Breakdown (Observability & Logging):**
- Logging infrastructure setup (structlog): 1 hour
- Request/response middleware logging: 1 hour
- RAG pipeline instrumentation: 2 hours
- Background job logging: 1 hour
- Replace print statements + testing: 1 hour

---

## 2. Business Context

### 2.1 Why This Refactor?

The VE-RAG-System was developed rapidly over a 2-week sprint, prioritizing functionality over architectural purity. Now that the system is feature-complete and deployed on DGX Spark, technical debt must be addressed before:

1. **Scaling the team** - New developers need consistent patterns
2. **Adding features** - Phase 2 roadmap requires solid foundation
3. **Production hardening** - Data integrity issues must be resolved
4. **Compliance requirements** - Audit trails require referential integrity

### 2.2 Business Impact of NOT Refactoring

| Risk | Impact | Likelihood |
|------|--------|------------|
| Data integrity issues (missing FKs) | Orphaned records, audit failures | High |
| Maintenance burden (dead code) | Developer confusion, slower onboarding | High |
| Testing gaps (HTTPException in services) | Production bugs, regression risk | Medium |
| Technical debt accumulation | Exponential refactoring cost | High |

### 2.3 Stakeholders

| Role | Interest | Sign-off Required |
|------|----------|-------------------|
| Engineering Lead | Architecture quality | Yes |
| Product Manager | Timeline impact | Yes |
| DevOps | Deployment changes | No (minimal) |
| QA | Testing strategy | Yes |

---

## 3. Current State Analysis

### 3.1 Codebase Statistics

```
VE-RAG-System/
├── ai_ready_rag/           # Backend (Python)
│   ├── api/                # 8 routers, ~6,300 LOC
│   ├── services/           # 22 services, ~16,300 LOC
│   ├── db/                 # Models + database, ~500 LOC
│   ├── core/               # Auth, deps, exceptions, ~800 LOC
│   ├── ui/                 # DEPRECATED Gradio, ~5,500 LOC
│   └── main.py             # Entry point, ~200 LOC
├── frontend/               # React frontend
│   └── src/                # ~5,200 LOC TypeScript/TSX
├── tests/                  # Test suite, ~2,000 LOC
└── app.py                  # DEPRECATED Gradio entry, ~1,000 LOC

Total Backend: ~29,000 LOC (including 6,500 dead code)
Total Frontend: ~5,200 LOC
```

### 3.2 Current Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI App                              │
├─────────────────────────────────────────────────────────────────┤
│  Routers (api/*.py)                                             │
│  ├── auth.py      - JWT authentication                          │
│  ├── users.py     - User CRUD                                   │
│  ├── tags.py      - Tag management                              │
│  ├── chat.py      - Chat sessions/messages + RAG                │
│  ├── documents.py - Document upload/management                  │
│  ├── admin.py     - Settings, cache, reindex                    │
│  ├── health.py    - Health checks                               │
│  └── experimental.py - Slide generator (Phase 2)                │
├─────────────────────────────────────────────────────────────────┤
│  Services (services/*.py)                                       │
│  ├── RAGService        - Main RAG orchestration                 │
│  ├── VectorService     - Qdrant vector operations               │
│  ├── CacheService      - Multi-layer response caching           │
│  ├── DocumentService   - Document CRUD [HAS ISSUES]             │
│  ├── WarmingWorker     - Background cache warming               │
│  └── 17 more services...                                        │
├─────────────────────────────────────────────────────────────────┤
│  Database Layer (db/*.py)                                       │
│  ├── database.py  - SessionLocal, get_db, init_db               │
│  └── models.py    - 17 SQLAlchemy models [MISSING FKs]          │
├─────────────────────────────────────────────────────────────────┤
│  Core (core/*.py)                                               │
│  ├── dependencies.py - Auth dependencies                        │
│  ├── security.py     - JWT, password hashing                    │
│  └── exceptions.py   - Domain exceptions                        │
├─────────────────────────────────────────────────────────────────┤
│  DEPRECATED (to remove)                                         │
│  ├── ui/          - Gradio UI package (5,500 LOC)               │
│  └── app.py       - Legacy Gradio entry (1,000 LOC)             │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Database Schema (Current)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    User     │     │     Tag     │     │  Document   │
├─────────────┤     ├─────────────┤     ├─────────────┤
│ id (PK)     │◄────┤ owner_id    │     │ id (PK)     │
│ email       │     │ id (PK)     │     │ filename    │
│ password    │     │ name        │     │ uploaded_by │ ← NOT FK!
│ role        │     │ description │     │ status      │
│ created_at  │     └─────────────┘     │ created_at  │
└─────────────┘            │            └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  user_tags  │     │document_tags│     │ ChatSession │
├─────────────┤     ├─────────────┤     ├─────────────┤
│ user_id(FK) │     │ doc_id (FK) │     │ id (PK)     │
│ tag_id (FK) │     │ tag_id (FK) │     │ user_id     │ ← NOT FK!
└─────────────┘     └─────────────┘     │ created_at  │
                                        └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ ChatMessage │
                                        ├─────────────┤
                                        │ id (PK)     │
                                        │ session_id  │ ← FK ✓
                                        │ role        │
                                        │ content     │
                                        └─────────────┘
```

**Issues Identified:**
- `Document.uploaded_by` → No FK to `users.id`
- `ChatSession.user_id` → No FK to `users.id`
- `AuditLog.user_id` → No FK to `users.id`
- `document_tags.tag_id` → Missing `ondelete="CASCADE"`

### 3.4 Service Layer Issues

#### 3.4.1 HTTPException in DocumentService

**File:** `ai_ready_rag/services/document_service.py`

The DocumentService directly raises FastAPI HTTPException, violating separation of concerns:

```python
# CURRENT (Anti-pattern)
class DocumentService:
    async def upload(self, file, tag_ids, user_id, ...):
        if not file.filename:
            raise HTTPException(                    # ← Service knows HTTP!
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided",
            )
```

**Locations (10 instances):**
- Line 86: No file provided
- Line 94: Invalid file type
- Line 101: No tags provided
- Line 110: Tag not found
- Line 122: Duplicate document
- Line 131: Storage error
- Line 169: Hash calculation error
- Line 213: Duplicate on replace
- Line 245: Document not found
- Line 278: Delete error

#### 3.4.2 Missing Repository Pattern

Services query the database directly:

```python
# CURRENT (Direct ORM queries in services)
class SomeService:
    def get_user(self, user_id: str):
        return self.db.query(User).filter(User.id == user_id).first()
```

**Impact:**
- Code duplication across services
- Hard to unit test (requires full DB setup)
- ORM implementation details leak into business logic

#### 3.4.3 Inconsistent Dependency Injection

Some services use lazy loading via properties:

```python
# CURRENT (Lazy loading - hidden dependencies)
class RAGService:
    def __init__(self, settings, vector_service=None):
        self._vector_service = vector_service

    @property
    def vector_service(self):
        if self._vector_service is None:
            self._vector_service = get_vector_service(self.settings)
        return self._vector_service
```

**Issues:**
- Dependencies not obvious from constructor
- Lazy initialization can mask errors
- Hard to mock in tests

---

## 4. Target State Architecture

### 4.1 Scaffold Pattern Overview

The target architecture follows the **FastAPI Layered Architecture Pattern**:

```
Request → Router (HTTP) → Service (Business Logic) → Repository (Data Access) → Database
            ↑                    ↑                         ↑
          deps.py             schemas.py               models.py
       (access control)    (validation)             (ORM definitions)
```

**Key Principles:**
1. **Routers** handle HTTP concerns only (status codes, response models)
2. **Services** contain business logic, own transactions, raise domain exceptions
3. **Repositories** abstract data access, never commit
4. **Models** define ORM structure, no business logic
5. **Schemas** validate input/output, shape API contracts

### 4.2 Target Directory Structure (Hybrid Layout)

The target architecture uses a **Hybrid Layout** that balances layer-based organization with clear separation of concerns. This approach keeps shared infrastructure centralized while maintaining clear boundaries between layers.

```
ai_ready_rag/
├── main.py                     # FastAPI entry point
├── config.py                   # Settings (Pydantic) - bootstrap/infrastructure only
│
├── api/                        # HTTP Layer (Routers)
│   ├── __init__.py             # Router aggregation
│   ├── auth.py                 # Authentication endpoints
│   ├── users.py                # User CRUD endpoints
│   ├── tags.py                 # Tag management endpoints
│   ├── chat.py                 # Chat session/message endpoints
│   ├── documents.py            # Document upload/management endpoints
│   ├── admin.py                # Admin settings/cache endpoints
│   ├── health.py               # Health check endpoints
│   └── experimental.py         # Experimental features (slides)
│
├── schemas/                    # NEW: Pydantic Schemas (API Contracts)
│   ├── __init__.py             # Re-exports all schemas
│   ├── common.py               # Shared schemas (pagination, errors)
│   ├── user.py                 # UserCreate, UserResponse, UserUpdate
│   ├── document.py             # DocumentCreate, DocumentResponse
│   ├── chat.py                 # ChatSessionResponse, MessageCreate
│   ├── tag.py                  # TagCreate, TagResponse
│   ├── auth.py                 # LoginRequest, TokenResponse
│   └── admin.py                # SettingUpdate, CacheStats
│
├── services/                   # Business Logic Layer
│   ├── __init__.py             # Service exports
│   ├── base.py                 # NEW: BaseService with transaction management
│   ├── user_service.py         # User business logic
│   ├── document_service.py     # REFACTORED: No HTTPException
│   ├── chat_service.py         # Chat session/message logic
│   ├── tag_service.py          # Tag business logic
│   ├── auth_service.py         # Authentication logic
│   ├── settings_service.py     # Admin settings (SQLite config)
│   ├── rag_service.py          # RAG orchestration
│   ├── vector_service.py       # Vector store operations
│   ├── cache_service.py        # Multi-layer caching
│   ├── embedding_service.py    # Embedding generation
│   └── protocols.py            # Service protocols/interfaces
│
├── db/                         # Database Layer
│   ├── __init__.py             # Exports Base, SessionLocal, get_db
│   ├── database.py             # Engine, session factory, Base
│   ├── mixins.py               # NEW: TimestampMixin, other mixins
│   ├── models/                 # NEW: Split ORM models
│   │   ├── __init__.py         # from .user import User, etc.
│   │   ├── user.py             # User model
│   │   ├── document.py         # Document model
│   │   ├── chat.py             # ChatSession, ChatMessage models
│   │   ├── tag.py              # Tag model, association tables
│   │   ├── audit.py            # AuditLog model
│   │   ├── cache.py            # CachedResponse, CuratedQA models
│   │   └── admin.py            # AdminSetting model
│   └── repositories/           # NEW: Repository classes
│       ├── __init__.py         # Repository exports
│       ├── base.py             # BaseRepository[T] generic
│       ├── user.py             # UserRepository
│       ├── document.py         # DocumentRepository
│       ├── tag.py              # TagRepository
│       ├── chat.py             # ChatSessionRepository, ChatMessageRepository
│       ├── audit.py            # AuditLogRepository
│       └── cache.py            # CacheRepository, CuratedQARepository
│
├── core/                       # Cross-Cutting Concerns
│   ├── __init__.py
│   ├── dependencies.py         # FastAPI Depends (get_db, get_current_user, get_*_service)
│   ├── security.py             # JWT creation/validation, password hashing
│   ├── exceptions.py           # Domain exception hierarchy
│   └── error_handlers.py       # NEW: Global exception → HTTP response mapping
│
├── workers/                    # Background Tasks
│   ├── __init__.py
│   ├── warming_worker.py       # Cache warming worker
│   └── cleanup_worker.py       # Job cleanup worker
│
└── # NO ui/ directory (Gradio removed)
```

#### 4.2.1 Layer Responsibilities

| Layer | Directory | Responsibility |
|-------|-----------|----------------|
| **HTTP** | `api/` | Request parsing, response formatting, HTTP status codes |
| **Schema** | `schemas/` | Input validation, API contracts, serialization |
| **Service** | `services/` | Business logic, transaction boundaries, domain exceptions |
| **Repository** | `db/repositories/` | Data access, queries, no commits |
| **Model** | `db/models/` | ORM definitions, relationships, constraints |
| **Core** | `core/` | Security, shared dependencies, exceptions |
| **Workers** | `workers/` | Background processing, async tasks |

#### 4.2.2 Import Flow

```
api/ ──imports──> schemas/
  │                   │
  └──imports──> services/ ──imports──> db/repositories/
                    │                        │
                    └──imports──> core/      └──imports──> db/models/
                                                               │
                                                    db/database.py (Base, mixins)
```

**Rules:**
- Routers import schemas and services (never models directly)
- Services import repositories and exceptions (never HTTPException)
- Repositories import models (never services)
- Models import only from database.py and mixins.py

### 4.3 Target Database Schema

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    User     │     │     Tag     │     │  Document   │
├─────────────┤     ├─────────────┤     ├─────────────┤
│ id (PK)     │◄────┤ owner_id FK │     │ id (PK)     │
│ email       │     │ id (PK)     │     │ filename    │
│ password    │     │ name        │     │ uploaded_by │─► FK to User
│ role        │     │ description │     │ status      │
│ created_at  │◄┐   └─────────────┘     │ created_at  │
│ updated_at  │ │          │            │ updated_at  │
└─────────────┘ │          │            └─────────────┘
       │        │          │                   │
       ▼        │          ▼                   ▼
┌─────────────┐ │   ┌─────────────┐     ┌─────────────┐
│  user_tags  │ │   │document_tags│     │ ChatSession │
├─────────────┤ │   ├─────────────┤     ├─────────────┤
│ user_id(FK) │ │   │ doc_id (FK) │     │ id (PK)     │
│ tag_id (FK) │ │   │ tag_id (FK) │─►   │ user_id     │─► FK to User
└─────────────┘ │   │ ondelete=   │     │ created_at  │
                │   │   CASCADE   │     │ updated_at  │
                │   └─────────────┘     └─────────────┘
                │                              │
                │   ┌─────────────┐            ▼
                │   │  AuditLog   │     ┌─────────────┐
                │   ├─────────────┤     │ ChatMessage │
                └───┤ user_id FK  │     ├─────────────┤
                    │ action      │     │ id (PK)     │
                    │ created_at  │     │ session_id  │─► FK ✓
                    └─────────────┘     │ role        │
                                        │ content     │
                                        │ created_at  │
                                        └─────────────┘

Legend:
  ─► FK = Foreign Key constraint added
  ✓  = Already correct
```

**Critical: SQLite FK Enforcement**

SQLite does not enforce FK constraints by default. The following must be added to `database.py`:

```python
from sqlalchemy import event

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
```

**Pre-MVP Migration Strategy:** Since the application is pre-MVP, database tables are dropped and recreated on schema changes. Alembic migrations will be introduced when the schema stabilizes before production release.

### 4.4 Exception Hierarchy (Target)

```python
# core/exceptions.py

class AppError(Exception):
    """Base application exception."""
    default_message = "An unexpected error occurred"

    def __init__(self, message: str | None = None, context: dict | None = None):
        self.message = message or self.default_message
        self.context = context or {}
        super().__init__(self.message)

# Data layer exceptions
class EntityNotFound(AppError):
    default_message = "Requested entity was not found"

class ConflictError(AppError):
    default_message = "Resource conflict"

# Business logic exceptions
class BusinessValidationError(AppError):
    default_message = "Business rule validation failed"

class PermissionDenied(AppError):
    default_message = "Permission denied"

class AuthenticationError(AppError):
    default_message = "Authentication failed"

# Document-specific exceptions (NEW)
class DocumentError(AppError):
    """Base for document operations."""
    pass

class DocumentValidationError(DocumentError):
    """Document failed validation."""
    pass

class DuplicateDocumentError(DocumentError):
    """Document already exists."""
    pass

class DocumentStorageError(DocumentError):
    """Failed to store document."""
    pass

# Existing RAG exceptions (keep)
class VectorServiceError(AppError): ...
class RAGServiceError(AppError): ...
class WarmingError(AppError): ...
```

#### 4.4.1 Global Error Handlers (Target)

All domain exceptions are converted to HTTP responses via global handlers registered in `main.py`. Routers never catch domain exceptions with try/except.

```python
# core/error_handlers.py

from fastapi import Request
from fastapi.responses import JSONResponse
from ai_ready_rag.core.exceptions import (
    AppError, EntityNotFound, ConflictError,
    BusinessValidationError, PermissionDenied, AuthenticationError,
    DocumentValidationError, DuplicateDocumentError, DocumentStorageError,
)

EXCEPTION_STATUS_MAP = {
    EntityNotFound: 404,
    DocumentValidationError: 400,
    BusinessValidationError: 422,
    DuplicateDocumentError: 409,
    ConflictError: 409,
    PermissionDenied: 403,
    AuthenticationError: 401,
    DocumentStorageError: 500,
}

async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    status_code = EXCEPTION_STATUS_MAP.get(type(exc), 500)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": type(exc).__name__,
                "message": exc.message,
                "context": exc.context,
            }
        },
    )

def register_error_handlers(app):
    """Register in main.py during app startup."""
    app.add_exception_handler(AppError, app_error_handler)
```

**Rule:** Routers only raise `HTTPException(404)` for simple not-found cases where the service returns `None`. All other error handling flows through global handlers.

### 4.5 BaseRepository Pattern (Target)

```python
# db/repositories/base.py

from typing import Any, Generic, Iterable, List, Optional, Type, TypeVar
from sqlalchemy import exists as sa_exists, func, select
from sqlalchemy.orm import Session

T = TypeVar("T")

class BaseRepository(Generic[T]):
    """Generic base repository for SQLAlchemy models."""

    model: Type[T]

    def __init__(self, db: Session) -> None:
        self.db = db

    # --- Retrieval ---

    def get(self, id_: Any) -> Optional[T]:
        """Get entity by primary key."""
        return self.db.get(self.model, id_)

    def get_with_filter(self, id_: Any, **filters: Any) -> Optional[T]:
        """Get by PK with additional filters (e.g., tenant check)."""
        stmt = select(self.model).where(self.model.id == id_)
        for key, value in filters.items():
            stmt = stmt.where(getattr(self.model, key) == value)
        return self.db.scalar(stmt)

    def list_all(self) -> List[T]:
        """List all entities."""
        return list(self.db.scalars(select(self.model)).all())

    def list_by(self, **filters: Any) -> List[T]:
        """List entities matching filters."""
        stmt = select(self.model)
        for key, value in filters.items():
            stmt = stmt.where(getattr(self.model, key) == value)
        return list(self.db.scalars(stmt).all())

    def exists(self, **filters: Any) -> bool:
        """Check if entity exists matching filters."""
        conditions = [getattr(self.model, k) == v for k, v in filters.items()]
        stmt = select(sa_exists().where(*conditions))
        return self.db.scalar(stmt) or False

    def count(self, **filters: Any) -> int:
        """Count entities matching filters."""
        stmt = select(func.count()).select_from(self.model)
        for key, value in filters.items():
            if value is not None:
                stmt = stmt.where(getattr(self.model, key) == value)
        return self.db.scalar(stmt) or 0

    # --- Persistence (never commit) ---

    def add(self, obj: T) -> T:
        """Add entity to session."""
        self.db.add(obj)
        return obj

    def delete(self, obj: T) -> None:
        """Mark entity for deletion."""
        self.db.delete(obj)

    def flush(self) -> None:
        """Flush pending changes (get IDs without commit)."""
        self.db.flush()

    def refresh(self, obj: T) -> T:
        """Refresh entity from database."""
        self.db.refresh(obj)
        return obj

    # --- Update ---

    def partial_update(self, obj: T, skip_none: bool = True, **fields: Any) -> T:
        """Update entity fields."""
        for key, value in fields.items():
            if skip_none and value is None:
                continue
            setattr(obj, key, value)
        self.db.add(obj)
        return obj
```

### 4.6 BaseService Pattern (Target)

```python
# services/base.py

from typing import Generic, TypeVar, Optional, Any
from sqlalchemy.orm import Session
from ai_ready_rag.db.repositories.base import BaseRepository
from ai_ready_rag.core.exceptions import EntityNotFound

T = TypeVar("T")  # ORM model type
R = TypeVar("R", bound=BaseRepository)  # Repository type

class BaseService(Generic[T, R]):
    """Base service with standard CRUD and transaction management."""

    repository_class: type[R]

    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo: R = self.repository_class(db)

    def get(self, id_: Any) -> Optional[T]:
        """Get entity by ID."""
        return self.repo.get(id_)

    def get_or_raise(self, id_: Any) -> T:
        """Get entity or raise EntityNotFound."""
        obj = self.repo.get(id_)
        if obj is None:
            model_name = self.repo.model.__name__
            raise EntityNotFound(f"{model_name} not found", context={"id": str(id_)})
        return obj

    def list_all(self) -> list[T]:
        """List all entities."""
        return self.repo.list_all()

    def create(self, obj: T) -> T:
        """Create entity with commit."""
        self.repo.add(obj)
        self.commit()
        self.repo.refresh(obj)
        return obj

    def update(self, obj: T, **fields: Any) -> T:
        """Update entity with commit."""
        self.repo.partial_update(obj, **fields)
        self.commit()
        self.repo.refresh(obj)
        return obj

    def delete(self, obj: T) -> None:
        """Delete entity with commit."""
        self.repo.delete(obj)
        self.commit()

    def commit(self) -> None:
        """Commit transaction."""
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def flush(self) -> None:
        """Flush without commit."""
        self.repo.flush()
```

#### 4.6.1 Transaction Boundary Rules

All services in a single request share the same `db: Session` (injected via `Depends(get_db)`). This means flush/commit on any service affects all pending changes in that session. The following rules prevent partial commits and data inconsistency.

**Rule 1: Single-service operation — service commits.**

Standard CRUD operations (create, update, delete) commit within the service method. This is the default `BaseService` behavior shown above.

**Rule 2: Multi-service orchestration — orchestrator commits, inner services flush only.**

When one service coordinates work across multiple repositories or services, only the outermost (orchestrating) service calls `commit()`. Inner operations use `flush()` to get database-generated IDs without finalizing the transaction.

```python
# RAGService orchestrates chat + cache in one transaction
class RAGService:
    def __init__(self, db: Session, vector_service, cache_service):
        self.db = db
        self.chat_repo = ChatSessionRepository(db)
        self.cache_service = cache_service  # shares same db session

    async def generate(self, query: str, session_id: str, user) -> str:
        # Step 1: Save user message (no commit)
        msg = ChatMessage(session_id=session_id, role="user", content=query)
        self.chat_repo.add(msg)
        self.db.flush()  # get msg.id without committing

        # Step 2: Generate LLM response
        response = await self._call_llm(query, context)

        # Step 3: Save assistant message (no commit)
        reply = ChatMessage(session_id=session_id, role="assistant", content=response.text)
        self.chat_repo.add(reply)

        # Step 4: Cache response (no commit)
        self.cache_service.store_no_commit(query, response)

        # Step 5: Single commit — all-or-nothing
        self.db.commit()
        return response
```

**Rule 3: Never call a committing method from inside an orchestration.**

**Anti-pattern:**
```python
# BAD: inner service commits, then outer operation fails → partial commit
def orchestrate(self):
    self.user_service.create(user)     # commits internally!
    self.profile_service.create(profile)  # if this fails, user already committed
```

**Correct pattern:**
```python
# GOOD: orchestrator owns the single commit
def orchestrate(self):
    self.user_service.create_no_commit(user)    # flush only
    self.profile_service.create_no_commit(profile)  # flush only
    self.db.commit()  # all-or-nothing
```

**Summary:**

| Scenario | Who commits? | Inner services use |
|----------|-------------|-------------------|
| Simple CRUD (one entity) | The service method | `commit()` |
| Multi-step orchestration | Outermost service | `flush()` / `create_no_commit()` |
| Background job (ARQ task) | Task handler | `commit()` at end of task |

### 4.7 Dependency Injection Pattern (Target)

Services are injected via FastAPI's `Depends()` chain. No ServiceContainer.

```python
# core/dependencies.py (additions)

from fastapi import Depends, Request
from sqlalchemy.orm import Session
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.repositories.document import DocumentRepository
from ai_ready_rag.services.document_service import DocumentService

# --- DB-backed services: per-request (lightweight) ---

def get_document_repo(db: Session = Depends(get_db)) -> DocumentRepository:
    return DocumentRepository(db)

def get_document_service(db: Session = Depends(get_db)) -> DocumentService:
    return DocumentService(db)

# --- Expensive services: app-level singletons via lifespan ---

def get_vector_service(request: Request) -> VectorService:
    """Singleton initialized at app startup."""
    return request.app.state.vector_service

def get_rag_service(
    request: Request,
    db: Session = Depends(get_db),
) -> RAGService:
    """RAGService with singleton vector_service + per-request db."""
    return RAGService(
        settings=request.app.state.settings,
        vector_service=request.app.state.vector_service,
        db=db,
    )
```

```python
# main.py lifespan for expensive services
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize expensive services once
    settings = get_settings()
    vector_service = VectorService(settings)
    await vector_service.initialize()
    app.state.settings = settings
    app.state.vector_service = vector_service
    yield
    # Shutdown: cleanup
    await vector_service.close()

app = FastAPI(lifespan=lifespan)
```

**Rules:**
- DB-backed services (DocumentService, TagService, etc.): create per-request via `Depends(get_db)`
- Expensive services (VectorService): singleton via `app.state`, injected through `Depends()`
- Orchestrator services (RAGService): combine singleton deps + per-request db
- No ServiceContainer class

---

## 5. Gap Analysis

### 5.1 Gap Summary Table

| Category | Current State | Target State | Gap Severity |
|----------|---------------|--------------|--------------|
| **Gradio Code** | 366 KB present | 0 KB | Critical |
| **Foreign Keys** | 3 missing | All present | Critical |
| **HTTPException in Services** | 10 instances | 0 instances | High |
| **Repository Layer** | None | BaseRepository + 4 concrete | High |
| **BaseService Pattern** | None | BaseService + inheritance | Medium |
| **TimestampMixin** | Manual in 11 models | Mixin inheritance | Low |
| **SQLAlchemy Syntax** | Legacy Column() | Modern Mapped[] | Low (Deferred post-MVP) |
| **Alembic Migrations** | None | Full setup | Medium (Deferred post-MVP) |
| **Service DI Pattern** | Mixed (lazy + constructor) | FastAPI Depends() chain | Medium |

### 5.2 Detailed Gap Analysis

#### 5.2.1 Critical Gaps

**Gap C1: Gradio Dead Code**
- **Current:** 6,500 lines of unused Gradio code
- **Impact:** Developer confusion, larger deployments, maintenance burden
- **Remediation:** Delete `ui/` directory and `app.py`
- **Effort:** 1 hour
- **Risk:** Low (code is unused)

**Gap C2: Missing Foreign Keys**
- **Current:** `ChatSession.user_id`, `AuditLog.user_id`, `Document.uploaded_by` are plain strings
- **Impact:** Orphaned records, no cascade delete, audit failures
- **Remediation:** Add FK constraints with proper ondelete
- **Effort:** 2 hours
- **Risk:** Medium (requires data migration if orphans exist)

#### 5.2.2 High Gaps

**Gap H1: HTTPException in DocumentService**
- **Current:** Service layer raises HTTP 400/404/409 directly
- **Impact:** Untestable without FastAPI, violates separation of concerns
- **Remediation:** Create domain exceptions, move HTTP conversion to router
- **Effort:** 4 hours
- **Risk:** Low (behavior unchanged)

**Gap H2: Missing Repository Layer**
- **Current:** Services query ORM directly
- **Impact:** Code duplication, hard to test, ORM leakage
- **Remediation:** Create BaseRepository and concrete repositories
- **Effort:** 10 hours
- **Risk:** Medium (touches many files)

#### 5.2.3 Medium Gaps

**Gap M1: No BaseService Pattern**
- **Current:** Each service implements its own transaction management
- **Impact:** Inconsistent patterns, code duplication
- **Remediation:** Create BaseService, refactor services to inherit
- **Effort:** 3 hours
- **Risk:** Low

**Gap M2: No Alembic Migrations**
- **Current:** Schema created via `Base.metadata.create_all()`
- **Impact:** No version control of schema, hard to deploy changes
- **Remediation:** Initialize Alembic, create baseline migration
- **Effort:** 3 hours
- **Risk:** Low

**Gap M3: Mixed DI Patterns**
- **Current:** Some services use lazy loading via properties
- **Impact:** Hidden dependencies, harder testing
- **Remediation:** Convert to constructor injection
- **Effort:** 3 hours
- **Risk:** Low

#### 5.2.4 Low Gaps

**Gap L1: No TimestampMixin**
- **Current:** `created_at`/`updated_at` repeated in 11 models
- **Impact:** Code duplication
- **Remediation:** Create mixin, update model inheritance
- **Effort:** 2 hours
- **Risk:** Low

**Gap L2: Legacy SQLAlchemy Syntax**
- **Current:** Uses `Column()` instead of `Mapped[]`
- **Impact:** Less type safety, not "modern"
- **Remediation:** Convert to SQLAlchemy 2.0 syntax
- **Effort:** 4 hours
- **Risk:** Low (purely syntactic)

---

## 6. Detailed Implementation Plan

### 6.1 Phase 1: Critical Fixes (7 hours)

#### Task 1.1: Remove Gradio (1 hour)

**Objective:** Remove all deprecated Gradio code from codebase.

**Files to DELETE:**
```
ai_ready_rag/ui/__init__.py
ai_ready_rag/ui/gradio_app.py
ai_ready_rag/ui/api_client.py
ai_ready_rag/ui/components.py
ai_ready_rag/ui/document_components.py
ai_ready_rag/ui/theme.py
app.py
```

**Files to MODIFY:**

`ai_ready_rag/main.py`:
```python
# REMOVE lines 142-156 (Gradio mount block)
# REMOVE line 175: or path.startswith("app/")
# REMOVE line 195: "ui": "/app" if settings.enable_gradio else "disabled"
```

`ai_ready_rag/config.py`:
```python
# REMOVE: enable_gradio: bool = True (already removed - Gradio deprecated)
```

`ai_ready_rag/api/health.py`:
```python
# REMOVE: "gradio_enabled": settings.enable_gradio
```

`requirements-spark.txt`, `requirements-wsl.txt`, `requirements.txt`:
```
# REMOVE: gradio>=5.0.0
```

**Verification:**
```bash
grep -r "gradio" --include="*.py" ai_ready_rag/
# Should return empty
```

#### Task 1.2: Fix Missing Foreign Keys (2 hours)

**Objective:** Add FK constraints to ensure referential integrity.

**File:** `ai_ready_rag/db/models.py`

**Changes:**

```python
# ChatSession.user_id (line ~113)
# BEFORE:
user_id = Column(String, nullable=False)

# AFTER:
user_id = Column(
    String,
    ForeignKey("users.id", ondelete="CASCADE"),
    nullable=False,
    index=True,
)

# AuditLog.user_id (line ~150)
# BEFORE:
user_id = Column(String, nullable=True)

# AFTER:
user_id = Column(
    String,
    ForeignKey("users.id", ondelete="SET NULL"),
    nullable=True,
    index=True,
)

# Document.uploaded_by (line ~94)
# BEFORE:
uploaded_by = Column(String, nullable=False)

# AFTER:
uploaded_by = Column(
    String,
    ForeignKey("users.id", ondelete="SET NULL"),
    nullable=True,  # Changed to nullable for SET NULL
    index=True,
)

# document_tags association table (line ~78)
# BEFORE:
Column("tag_id", String, ForeignKey("tags.id"), primary_key=True),

# AFTER:
Column("tag_id", String, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
```

**Enable FK Enforcement (required):**

**File:** `ai_ready_rag/db/database.py`

```python
from sqlalchemy import event

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
```

**Pre-MVP Migration:** Delete the existing database file and let `init_db()` / `create_all()` recreate tables with the new FK constraints. No orphan migration script needed.

```bash
# Pre-MVP: drop and recreate
rm data/ai_ready_rag.db
python -m uvicorn ai_ready_rag.main:app  # Tables recreated on startup
```

**Verification:**
```sql
-- Verify FKs are enforced
PRAGMA foreign_keys;          -- Should return 1
PRAGMA foreign_key_list(chat_sessions);  -- Should show user_id → users.id
PRAGMA foreign_key_list(documents);       -- Should show uploaded_by → users.id
PRAGMA foreign_key_list(audit_logs);      -- Should show user_id → users.id
```

#### Task 1.3: Remove HTTPException from DocumentService (4 hours)

**Objective:** Move HTTP concerns to router layer.

**Step 1: Add Document Exceptions**

**File:** `ai_ready_rag/core/exceptions.py` (add at end)

```python
# Document exceptions
class DocumentError(AppError):
    """Base for document operations."""
    pass

class DocumentValidationError(DocumentError):
    """Document failed validation (file type, size, etc.)."""
    pass

class DuplicateDocumentError(DocumentError):
    """Document already exists (by hash)."""
    pass

class DocumentStorageError(DocumentError):
    """Failed to store document on disk."""
    pass

class DocumentNotFoundError(DocumentError):
    """Document not found in database."""
    pass

class InvalidTagError(DocumentError):
    """Tag does not exist or user lacks access."""
    pass
```

**Step 2: Refactor DocumentService**

**File:** `ai_ready_rag/services/document_service.py`

```python
# REMOVE this import:
from fastapi import HTTPException, status

# ADD this import:
from ai_ready_rag.core.exceptions import (
    DocumentValidationError,
    DuplicateDocumentError,
    DocumentStorageError,
    DocumentNotFoundError,
    InvalidTagError,
)

# CHANGE each HTTPException to domain exception:

# Line 86:
# BEFORE:
raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided")
# AFTER:
raise DocumentValidationError("No file provided")

# Line 94:
# BEFORE:
raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid file type...")
# AFTER:
raise DocumentValidationError(f"Invalid file type: {ext}")

# Line 101:
# BEFORE:
raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one tag...")
# AFTER:
raise DocumentValidationError("At least one tag is required")

# Line 110:
# BEFORE:
raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tag {tag_id} not found")
# AFTER:
raise InvalidTagError(f"Tag not found", context={"tag_id": tag_id})

# Line 122:
# BEFORE:
raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Document already exists...")
# AFTER:
raise DuplicateDocumentError("Document already exists", context={"existing_id": existing.id, "hash": content_hash})

# Line 131:
# BEFORE:
raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=...)
# AFTER:
raise DocumentStorageError(f"Failed to store file: {e}", context={"filename": file.filename})

# Continue for all remaining HTTPException instances...
```

**Step 3: Register Global Error Handlers + Simplify Router**

**File:** `ai_ready_rag/core/error_handlers.py` (NEW - see Section 4.4.1 for full implementation)

**File:** `ai_ready_rag/main.py` (add during app setup)
```python
from ai_ready_rag.core.error_handlers import register_error_handlers

app = FastAPI(...)
register_error_handlers(app)
```

**File:** `ai_ready_rag/api/documents.py` (simplified - no try/except needed)

```python
# NO exception imports needed in router - global handlers catch them

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile,
    tag_ids: list[str] = Form(...),
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
):
    # Domain exceptions raised by service are caught by global error handlers
    document = await service.upload(
        file=file,
        tag_ids=tag_ids,
        user_id=current_user.id,
    )
    return DocumentResponse.from_orm(document)
```

**Note:** Routers only raise `HTTPException(404)` for simple not-found cases where a service returns `None`. All domain exceptions flow through global handlers automatically.

**Verification:**
```bash
# Should return 0 results
grep -n "HTTPException" ai_ready_rag/services/document_service.py
```

### 6.2 Phase 2: Architecture Alignment (22 hours)

#### Task 2.1: Create schemas/ Directory (4 hours)

Extract inline Pydantic schemas from route files into dedicated schema modules. See Section 4.2 for target structure.

**Files to create:** `schemas/__init__.py`, `schemas/common.py`, `schemas/user.py`, `schemas/document.py`, `schemas/chat.py`, `schemas/tag.py`, `schemas/auth.py`, `schemas/admin.py`

**Approach:** For each route file in `api/`, identify inline Pydantic `BaseModel` classes (request/response schemas) and move them to the corresponding `schemas/*.py` file. Follow the Create/Update/Read pattern from the FastAPI Layered Pattern reference.

#### Task 2.2: Split models.py into models/ Package (3 hours)

Split the monolithic `db/models.py` (19 models, 399 lines) into per-entity modules. See Section 4.2 for target structure.

**Files to create:** `db/models/__init__.py`, `db/models/user.py`, `db/models/document.py`, `db/models/chat.py`, `db/models/tag.py`, `db/models/audit.py`, `db/models/cache.py`, `db/models/admin.py`

**File to delete:** `db/models.py` (replaced by package)

**Critical:** `db/models/__init__.py` must re-export all models (see Task 2.7) so existing imports are unaffected.

#### Task 2.3: Create TimestampMixin (2 hours)

**File:** `ai_ready_rag/db/database.py` (add to existing)

```python
from datetime import datetime
from sqlalchemy import Column, DateTime

class TimestampMixin:
    """Mixin adding created_at and updated_at columns."""

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
```

**File:** `ai_ready_rag/db/models.py` (update models)

```python
from ai_ready_rag.db.database import Base, TimestampMixin

# BEFORE:
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=generate_uuid)
    created_at = Column(DateTime, default=datetime.utcnow)
    # ... no updated_at in some models

# AFTER:
class User(TimestampMixin, Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=generate_uuid)
    # created_at and updated_at inherited from mixin
```

#### Task 2.4: Create workers/ Package (2 hours)

Move background worker files from `services/` into a dedicated `workers/` package.

**Files to create:** `workers/__init__.py`
**Files to move:** `services/warming_worker.py` → `workers/warming_worker.py`, `services/cleanup_worker.py` → `workers/cleanup_worker.py`

Update all imports referencing the moved files.

#### Task 2.5: Create BaseRepository + Concrete Repositories (6 hours)

**File:** `ai_ready_rag/db/repositories/__init__.py`
```python
from .base import BaseRepository
from .user import UserRepository
from .document import DocumentRepository
from .tag import TagRepository
from .chat import ChatSessionRepository

__all__ = [
    "BaseRepository",
    "UserRepository",
    "DocumentRepository",
    "TagRepository",
    "ChatSessionRepository",
]
```

**File:** `ai_ready_rag/db/repositories/base.py`
(See Section 4.5 for full implementation)

**File:** `ai_ready_rag/db/repositories/user.py`
```python
from typing import Optional
from sqlalchemy.orm import Session

from .base import BaseRepository
from ai_ready_rag.db.models import User

class UserRepository(BaseRepository[User]):
    model = User

    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        return self.list_by(email=email)[0] if self.exists(email=email) else None

    def exists_by_email(self, email: str) -> bool:
        """Check if user with email exists."""
        return self.exists(email=email)
```

**File:** `ai_ready_rag/db/repositories/document.py`
```python
from typing import Optional
import uuid
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from .base import BaseRepository
from ai_ready_rag.db.models import Document, Tag, document_tags

class DocumentRepository(BaseRepository[Document]):
    model = Document

    def get_by_hash(self, content_hash: str) -> Optional[Document]:
        """Get document by content hash."""
        results = self.list_by(content_hash=content_hash)
        return results[0] if results else None

    def list_for_user(self, user_id: str, tag_ids: list[str]) -> list[Document]:
        """List documents accessible to user via tags."""
        stmt = (
            select(Document)
            .join(document_tags)
            .where(document_tags.c.tag_id.in_(tag_ids))
            .distinct()
        )
        return list(self.db.scalars(stmt).all())

    def get_with_tags(self, doc_id: str) -> Optional[Document]:
        """Get document with eager-loaded tags."""
        stmt = (
            select(Document)
            .where(Document.id == doc_id)
            .options(joinedload(Document.tags))
        )
        return self.db.scalar(stmt)
```

**File:** `ai_ready_rag/db/repositories/tag.py`
```python
from typing import Optional
from .base import BaseRepository
from ai_ready_rag.db.models import Tag

class TagRepository(BaseRepository[Tag]):
    model = Tag

    def get_by_name(self, name: str) -> Optional[Tag]:
        """Get tag by name."""
        results = self.list_by(name=name)
        return results[0] if results else None

    def get_by_ids(self, tag_ids: list[str]) -> list[Tag]:
        """Get multiple tags by ID."""
        return [tag for tag_id in tag_ids if (tag := self.get(tag_id))]
```

**File:** `ai_ready_rag/db/repositories/chat.py`
```python
from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from .base import BaseRepository
from ai_ready_rag.db.models import ChatSession, ChatMessage

class ChatSessionRepository(BaseRepository[ChatSession]):
    model = ChatSession

    def list_for_user(self, user_id: str) -> list[ChatSession]:
        """List sessions for a user, ordered by recency."""
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def get_with_messages(self, session_id: str) -> Optional[ChatSession]:
        """Get session with eager-loaded messages."""
        stmt = (
            select(ChatSession)
            .where(ChatSession.id == session_id)
            .options(joinedload(ChatSession.messages))
        )
        return self.db.scalar(stmt)


class ChatMessageRepository(BaseRepository[ChatMessage]):
    model = ChatMessage

    def list_for_session(self, session_id: str) -> list[ChatMessage]:
        """List messages in a session, ordered by creation."""
        return self.list_by(session_id=session_id)
```

#### Task 2.6: Create BaseService (3 hours)

**File:** `ai_ready_rag/services/base.py`
(See Section 4.6 for full implementation)

#### Task 2.7: Import Migration with Re-exports (2 hours)

**Objective:** Ensure the models/ and schemas/ splits are zero-churn for consuming code.

**Strategy: Re-export pattern**

All package `__init__.py` files re-export their contents so existing import paths continue to work unchanged.

**File:** `ai_ready_rag/db/models/__init__.py`
```python
# Re-export all models — existing imports remain valid
from .user import User
from .document import Document
from .chat import ChatSession, ChatMessage
from .tag import Tag, user_tags, document_tags
from .audit import AuditLog
from .cache import CachedResponse, CuratedQA
from .admin import AdminSetting

__all__ = [
    "User", "Document", "ChatSession", "ChatMessage",
    "Tag", "user_tags", "document_tags",
    "AuditLog", "CachedResponse", "CuratedQA", "AdminSetting",
]
```

**File:** `ai_ready_rag/schemas/__init__.py`
```python
# Re-export all schemas for convenience
from .user import *
from .document import *
from .chat import *
from .tag import *
from .auth import *
from .admin import *
from .common import *
```

**Import path mapping (before → after):**

| Before | After | Change Required? |
|--------|-------|-----------------|
| `from ai_ready_rag.db.models import User` | `from ai_ready_rag.db.models import User` | **No** (re-exported) |
| `from ai_ready_rag.db.models import Document, Tag` | `from ai_ready_rag.db.models import Document, Tag` | **No** (re-exported) |
| Inline Pydantic schemas in `api/users.py` | `from ai_ready_rag.schemas.user import UserCreate, UserRead` | **Yes** — extract to schemas/ |
| Inline Pydantic schemas in `api/documents.py` | `from ai_ready_rag.schemas.document import DocumentResponse` | **Yes** — extract to schemas/ |
| `from ai_ready_rag.db.models import Base` | `from ai_ready_rag.db.database import Base` | **Yes** — if any file imports Base from models |

**Key rule:** Model imports require **zero changes** thanks to re-exports. Only schema extraction from route files requires import updates.

**Verification commands (run after each sub-task):**
```bash
# Confirm model re-exports work
python -c "from ai_ready_rag.db.models import User, Document, Tag, ChatSession, AuditLog, CachedResponse"

# Confirm all Python files compile
python -m compileall ai_ready_rag/ -q

# Confirm no broken imports at runtime
pytest tests/ -x --tb=short -q
```

### 6.3 Phase 3: Modernization (4 hours)

> **Deferred to post-MVP:** Alembic migrations, SQLAlchemy 2.0 `Mapped[]` syntax, ServiceContainer.
> See Section 13 for rationale.

#### Task 3.1: Add FK Indexes (1 hour)

**File:** `ai_ready_rag/db/models.py`

```python
class ChatSession(TimestampMixin, Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (
        Index("idx_chat_sessions_user_id", "user_id"),
    )
    # ...

class ChatMessage(TimestampMixin, Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("idx_chat_messages_session_id", "session_id"),
    )
    # ...
```

#### Task 3.2: Replace Lazy Loading with Constructor Injection + Depends() Chain (3 hours)

**File:** `ai_ready_rag/services/rag_service.py`

```python
# BEFORE:
class RAGService:
    def __init__(self, settings, vector_service=None, cache_service=None, ...):
        self._vector_service = vector_service
        self._cache_service = cache_service

    @property
    def vector_service(self):
        if self._vector_service is None:
            self._vector_service = get_vector_service(self.settings)
        return self._vector_service

# AFTER:
class RAGService:
    def __init__(
        self,
        settings: Settings,
        vector_service: VectorServiceProtocol,
        cache_service: CacheService,
        db: Session,
    ):
        self.settings = settings
        self.vector_service = vector_service  # Required, explicit
        self.cache_service = cache_service    # Required, explicit
        self.db = db
```

**Wire up via FastAPI Depends() chain (see Section 4.7):**

```python
# core/dependencies.py additions

def get_document_service(db: Session = Depends(get_db)) -> DocumentService:
    return DocumentService(db)

def get_vector_service(request: Request) -> VectorService:
    return request.app.state.vector_service

def get_rag_service(
    request: Request,
    db: Session = Depends(get_db),
) -> RAGService:
    return RAGService(
        settings=request.app.state.settings,
        vector_service=request.app.state.vector_service,
        db=db,
    )
```

**Initialize expensive services once in `main.py` lifespan:**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    vector_service = VectorService(settings)
    await vector_service.initialize()
    app.state.settings = settings
    app.state.vector_service = vector_service
    yield
    await vector_service.close()

app = FastAPI(lifespan=lifespan)
```

**Update routes to use Depends:**

```python
# In api/chat.py
@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    payload: MessageCreate,
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service),
):
    # rag_service injected with all dependencies resolved
    response = await rag_service.generate(...)
```

### 6.4 Phase 4: Quality (8 hours)

#### Task 4.1: Add Service Unit Tests (4 hours)

**File:** `tests/test_document_service.py` (new)

```python
import pytest
from unittest.mock import Mock, patch
from ai_ready_rag.services.document_service import DocumentService
from ai_ready_rag.core.exceptions import (
    DocumentValidationError,
    DuplicateDocumentError,
)

class TestDocumentService:
    @pytest.fixture
    def service(self, db_session):
        settings = Mock()
        settings.upload_dir = "/tmp/test_uploads"
        return DocumentService(db_session, settings)

    def test_upload_raises_validation_error_no_filename(self, service):
        file = Mock()
        file.filename = None

        with pytest.raises(DocumentValidationError) as exc:
            await service.upload(file, ["tag1"], "user1")

        assert "No file provided" in str(exc.value)

    def test_upload_raises_validation_error_invalid_type(self, service):
        file = Mock()
        file.filename = "test.exe"

        with pytest.raises(DocumentValidationError) as exc:
            await service.upload(file, ["tag1"], "user1")

        assert "Invalid file type" in str(exc.value)

    def test_upload_raises_duplicate_error(self, service, db_session):
        # Create existing document with same hash
        # ...
        with pytest.raises(DuplicateDocumentError):
            await service.upload(duplicate_file, ["tag1"], "user1")
```

#### Task 4.2: Remove Print Debugging (1 hour)

**File:** `ai_ready_rag/services/cache_service.py`

```python
# BEFORE:
print(f"[CACHE] Checking cache for: {query[:50]}...", flush=True)

# AFTER:
logger.debug(f"Checking cache for: {query[:50]}...")
```

Search and replace all `print(` with appropriate `logger.` calls.

#### Task 4.3: Add Semaphore to WarmingWorker (1 hour)

**File:** `ai_ready_rag/services/warming_worker.py`

```python
class WarmingWorker:
    def __init__(self, rag_service, settings):
        # ... existing init ...
        self._query_semaphore = asyncio.Semaphore(
            settings.max_concurrent_warming_queries  # Default: 2
        )

    async def _warm_query_with_retry(self, query, ...):
        async with self._query_semaphore:  # Limit concurrent Ollama calls
            # ... existing logic ...
```

**File:** `ai_ready_rag/config.py`
```python
max_concurrent_warming_queries: int = 2
```

#### Task 4.4: Update Documentation (2 hours)

**Files to update:**
- `CLAUDE.md` - Remove Gradio references, update architecture
- `README.md` - Update setup instructions
- `docs/ARCHITECTURE.md` - Document new patterns
- `docs/WSL2_SETUP.md` - Remove Gradio setup steps

---

## 7. Migration Strategy

### 7.1 Deployment Approach

**Recommended: Big Bang with Feature Flag**

Given the scope of changes, deploy all changes at once with feature flags for gradual rollout:

```python
# config.py
use_new_repository_layer: bool = False  # Toggle new repos
use_new_document_exceptions: bool = False  # Toggle new exceptions
```

### 7.2 Migration Steps

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Prepare (Day 1)                                         │
├─────────────────────────────────────────────────────────────────┤
│ - Create feature branch: refactor/scaffold-alignment            │
│ - Run orphan record fix script on production DB                 │
│ - Backup production database                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Phase 1 Implementation (Day 1-2)                        │
├─────────────────────────────────────────────────────────────────┤
│ - Remove Gradio code                                            │
│ - Add FK constraints                                            │
│ - Refactor DocumentService exceptions                           │
│ - Run full test suite                                           │
│ - Code review                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Phase 1 Deployment (Day 3)                              │
├─────────────────────────────────────────────────────────────────┤
│ - Deploy to staging environment                                 │
│ - Run integration tests                                         │
│ - Deploy to production (off-hours)                              │
│ - Monitor for 24 hours                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Phase 2-3 Implementation (Day 4-8)                      │
├─────────────────────────────────────────────────────────────────┤
│ - Implement repository layer                                    │
│ - Implement BaseService                                         │
│ - Add TimestampMixin                                            │
│ - FK indexes + Depends() chain                                  │
│ - Code review each component                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: Phase 2-3 Deployment (Day 9)                            │
├─────────────────────────────────────────────────────────────────┤
│ - Deploy to staging                                             │
│ - Run full regression                                           │
│ - Deploy to production                                          │
│ - Monitor for 48 hours                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 6: Phase 4 Implementation (Day 10-11)                      │
├─────────────────────────────────────────────────────────────────┤
│ - Add service unit tests                                        │
│ - Remove print debugging                                        │
│ - Update documentation                                          │
│ - Final code review                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 7: Final Deployment (Day 12)                               │
├─────────────────────────────────────────────────────────────────┤
│ - Deploy final changes                                          │
│ - Merge to main                                                 │
│ - Tag release v0.5.0                                            │
│ - Close refactor issues                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Data Migration

**Pre-MVP approach:** Drop and recreate all tables. No migration scripts needed.

```bash
rm data/ai_ready_rag.db
# Tables recreated on next startup via init_db() / create_all()
```

This is acceptable because:
- Application is pre-MVP with no production data to preserve
- Test data can be re-seeded via the setup wizard
- FK constraints require table recreation in SQLite anyway

**Post-MVP:** Alembic migrations will be introduced when schema stabilizes.

#### 7.3.1 Data Safety Clause

Even pre-MVP, destructive migrations require explicit authorization and automatic backup.

**Behavior:**

| `ALLOW_DESTRUCTIVE_MIGRATION` env var | Action |
|---------------------------------------|--------|
| Not set / `false` | Refuse to start. Log: `"Schema change detected. Set ALLOW_DESTRUCTIVE_MIGRATION=true to drop/recreate."` |
| `true` | Auto-backup → drop all tables → recreate via `create_all()` |

**Auto-backup:** Before any destructive operation, the database file is copied automatically:
```bash
# Performed automatically by init_db() when ALLOW_DESTRUCTIVE_MIGRATION=true
cp data/ai_ready_rag.db data/ai_ready_rag.db.bak.$(date +%Y%m%d%H%M%S)
```

**Restore if needed:**
```bash
cp data/ai_ready_rag.db.bak.TIMESTAMP data/ai_ready_rag.db
```

**Implementation in `database.py`:**
```python
import os
import shutil
from datetime import datetime

def init_db():
    """Initialize database, handling schema changes."""
    db_path = "data/ai_ready_rag.db"

    if os.path.exists(db_path) and _schema_changed():
        if not os.getenv("ALLOW_DESTRUCTIVE_MIGRATION", "").lower() == "true":
            raise RuntimeError(
                "Schema change detected. Set ALLOW_DESTRUCTIVE_MIGRATION=true "
                "to drop and recreate tables."
            )

        # Auto-backup before drop
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        backup_path = f"{db_path}.bak.{timestamp}"
        shutil.copy2(db_path, backup_path)
        print(f"Database backed up to {backup_path}")

        # Drop and recreate
        Base.metadata.drop_all(bind=engine)

    Base.metadata.create_all(bind=engine)
```

**Notes:**
- Backup files accumulate in `data/` — clean up periodically
- The `_schema_changed()` check compares current table definitions against the database; implementation deferred to Phase 1
- In CI/test environments, set `ALLOW_DESTRUCTIVE_MIGRATION=true` in the test runner

### 7.4 FK Policy Table

| Model | FK Column | Nullable | On Delete | Rationale |
|-------|-----------|----------|-----------|-----------|
| Document | uploaded_by | Yes | SET NULL | Preserve documents if user deleted |
| ChatSession | user_id | No | CASCADE | Sessions are ephemeral; clean delete |
| AuditLog | user_id | **Yes** | SET NULL | System/ARQ jobs have no user context |

**Implementation notes:**
- `AuditLog.user_id` must be nullable to support system-initiated events (background jobs, startup tasks)
- Use convention: `user_id=NULL` with `action` prefix "SYSTEM:" for automated events
- `PRAGMA foreign_keys=ON` must be set on every connection (see Task 1.2)

---

## 8. Risk Assessment

### 8.1 Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **FK constraint breaks existing data** | Medium | High | Run orphan fix script before deploy |
| **Service refactor breaks functionality** | Low | High | Comprehensive test suite, gradual rollout |
| **Gradio removal breaks something** | Very Low | Medium | Gradio is unused (feature flag off) |
| **Repository layer introduces bugs** | Medium | Medium | Keep existing code alongside, feature flag |
| **Drop-and-recreate loses data** | Low | Medium | Auto-backup before drop (Section 7.3.1); `ALLOW_DESTRUCTIVE_MIGRATION` flag required |
| **Performance regression** | Low | Medium | Benchmark before/after |

### 8.2 Risk Mitigation Details

**Risk R1: FK Constraint Breaks Existing Data**
- **Mitigation:** Run `migrate_add_fks.py` script before deploy
- **Rollback:** Script logs all deleted/modified records for manual recovery
- **Owner:** Database Admin

**Risk R2: Service Refactor Breaks Functionality**
- **Mitigation:** Keep existing code paths, use feature flags
- **Rollback:** Disable feature flag, revert to old code path
- **Owner:** Backend Lead

**Risk R3: Performance Regression**
- **Mitigation:** Benchmark critical paths before/after
- **Metrics:** Response time p95, p99 for /api/chat/send
- **Owner:** DevOps

---

## 9. Testing Strategy

### 9.1 Test Coverage Requirements

| Component | Current Coverage | Target Coverage |
|-----------|-----------------|-----------------|
| DocumentService | ~30% | 80% |
| Repository Layer | 0% (new) | 90% |
| API Routes | ~60% | 80% |
| RAG Pipeline | ~40% | 60% |

### 9.2 Test Types

**Unit Tests (New):**
- `test_document_service.py` - Service layer without HTTP
- `test_repositories.py` - Repository CRUD operations
- `test_exceptions.py` - Exception hierarchy

**Integration Tests (Existing, Update):**
- `test_documents.py` - Full upload/delete flow
- `test_chat_api.py` - Chat session management
- `test_auth.py` - Authentication flows

**Regression Tests:**
- Run full existing test suite after each phase
- Manual smoke test on staging before production

### 9.3 Test Execution Plan

```bash
# After each task
pytest tests/ -v --tb=short

# Before staging deploy
pytest tests/ -v --cov=ai_ready_rag --cov-report=html

# Performance benchmark
locust -f tests/load/locustfile.py --headless -u 10 -r 1 -t 60s
```

### 9.4 Phase Test Checkpoints

Each phase gate requires all prior tests plus the new phase-specific tests to pass before proceeding.

| Phase | Minimum Required Tests |
|-------|----------------------|
| **1: Critical Fixes** | FK constraint rejection (insert orphan row → expect `IntegrityError`). `PRAGMA foreign_keys` returns 1. Global error handler maps all 8 exception types to correct HTTP status codes. `grep HTTPException ai_ready_rag/services/` returns 0 results. Existing test suite passes. |
| **2: Architecture** | `BaseRepository` CRUD (add/get/list/delete) for at least 2 entities. Concrete repo custom queries (e.g., `UserRepository.get_by_email`). Import smoke test: `python -m compileall ai_ready_rag/ -q` passes. Re-export verification: `from ai_ready_rag.db.models import User, Document, Tag` works. Experimental endpoints respond 200. |
| **3: Modernization** | `Depends()` chain resolves service in route (integration test). Lifespan singleton created once, not per-request (assert `app.state.vector_service` is same object across requests). FK indexes exist (`PRAGMA index_list`). |
| **4: Quality** | Service test coverage >= 80% (measured via `pytest --cov`). `grep -r "print(" ai_ready_rag/services/` returns 0 results. |
| **5: Config** | Each of 8 settings: set in SQLite → verify runtime effect (see Section 12.5 table). Precedence test: SQLite override > profile default > env variable for at least one setting. |
| **6: ARQ Infra** | ARQ worker starts without error. Enqueue/dequeue round-trip: enqueue test job → worker executes → result stored in Redis. Redis health check endpoint returns worker status. |
| **7-9: Task Migration** | Job survives server restart (enqueue → restart server → job completes). Progress tracking: Redis hash updated during execution. Cancellation: set cancel flag → job stops within 10s. Per-task: document processes, cache warms, reindex completes. |
| **11: Observability** | All 10 log events fire correctly (unit test per event). `request_id` propagates: middleware → service log → `X-Request-ID` response header. No passwords, tokens, or JWT secrets in log output (grep verification). |

---

## 10. Rollback Plan

### 10.1 Phase 1 Rollback

**Trigger:** Critical bug in production after Phase 1 deploy

**Steps:**
1. Revert commit: `git revert HEAD~3..HEAD`
2. Redeploy previous version
3. FK constraints will remain (no harm) - data already migrated

**RTO:** 15 minutes

### 10.2 Phase 2-3 Rollback

**Trigger:** Repository layer causes bugs

**Steps:**
1. Set feature flag: `use_new_repository_layer = False`
2. Old code path activates immediately
3. No redeploy needed if feature flags implemented

**RTO:** 1 minute (feature flag) or 15 minutes (full revert)

### 10.3 Full Rollback

**Trigger:** Catastrophic failure, multiple components broken

**Steps:**
1. Restore database from pre-migration backup
2. Deploy previous release tag (v0.4.x)
3. Notify stakeholders

**RTO:** 30 minutes

---

## 11. Timeline & Milestones

### 11.1 Gantt Chart

```
Week 1:
├── Day 1-2: Phase 1 Implementation (7h)
│   ├── Task 1.1: Remove Gradio (1h)
│   ├── Task 1.2: Fix FKs + PRAGMA (2h)
│   └── Task 1.3: DocumentService exceptions + global error handlers (4h)
├── Day 3: Phase 1 Code Review + Deploy
├── Day 4-5: Phase 2 Implementation - Hybrid Layout (11h)
│   ├── Task 2.1: Create schemas/ directory (4h)
│   ├── Task 2.2: Split models.py into models/ package (3h)
│   ├── Task 2.3: Create db/mixins.py with TimestampMixin (2h)
│   └── Task 2.4: Create workers/ package, move workers (2h)

Week 2:
├── Day 6-7: Phase 2 Continued (11h)
│   ├── Task 2.5: BaseRepository + concrete repositories (6h)
│   ├── Task 2.6: BaseService (3h)
│   └── Task 2.7: Import migration with re-exports (2h)
├── Day 8: Phase 2 Code Review
├── Day 9: Phase 3 Implementation (4h)
│   ├── Task 3.1: FK Indexes (1h)
│   └── Task 3.2: Constructor Injection + Depends() chain (3h)
├── Day 10: Phase 3 Code Review

Week 3:
├── Day 11-12: Phase 4 Implementation (8h)
│   ├── Task 4.1: Service Tests (4h)
│   ├── Task 4.2: Remove Print Debugging (1h)
│   ├── Task 4.3: WarmingWorker Semaphore (1h)
│   └── Task 4.4: Documentation (2h)
├── Day 13: Phase 4 Code Review + Deploy
├── Day 14: Phase 5 Implementation - Config Migration (4h)
│   ├── Task 5.1: Security/Policy Settings (2h)
│   ├── Task 5.2: Feature Flags (1h)
│   └── Task 5.3: Migration script + testing (1h)
├── Day 15: Phase 5 Code Review + Deploy

Week 4:
├── Day 16-17: Phase 6 - ARQ Infrastructure (4h)
│   ├── Task 6.1: Install Redis (Docker) (0.5h)
│   ├── Task 6.2: Add ARQ dependencies (0.5h)
│   ├── Task 6.3: Create workers/ package structure (1h)
│   ├── Task 6.4: Create WorkerSettings configuration (1h)
│   └── Task 6.5: Add Redis connection pool (1h)
├── Day 18-19: Phase 7 - Migrate Document Processing (6h)
│   ├── Task 7.1: Create process_document task handler (2h)
│   ├── Task 7.2: Update upload endpoint to enqueue job (1h)
│   ├── Task 7.3: Add progress tracking to handler (1h)
│   ├── Task 7.4: Update document status from job result (1h)
│   └── Task 7.5: Remove old BackgroundTasks code (1h)

Week 5:
├── Day 20-21: Phase 8 - Migrate Cache Warming (6h)
│   ├── Task 8.1: Create warm_cache task handler (2h)
│   ├── Task 8.2: Migrate warming endpoints to ARQ (2h)
│   ├── Task 8.3: Remove warming_worker.py, warming_queue.py (1h)
│   └── Task 8.4: Update SSE to use Redis pub/sub (1h)
├── Day 22: Phase 9 - Migrate Reindexing (4h)
│   ├── Task 9.1: Create reindex_knowledge_base handler (2h)
│   ├── Task 9.2: Update reindex endpoints to use ARQ (1h)
│   └── Task 9.3: Remove reindex_worker.py, reindex_service.py (1h)
├── Day 23: Phase 10 - Jobs API & Cleanup (4h)
│   ├── Task 10.1: Create /api/jobs endpoints (2h)
│   ├── Task 10.2: Add SSE progress streaming (1h)
│   ├── Task 10.3: Remove legacy tables (0.5h)
│   └── Task 10.4: Update frontend to use job API (0.5h)
├── Day 24: Phase 6-10 Code Review + Deploy

Week 5 (continued):
├── Day 25: Phase 11 - Observability & Logging (6h)
│   ├── Task 11.1: Logging infrastructure (structlog) (1h)
│   ├── Task 11.2: Request logging middleware (1h)
│   ├── Task 11.3: RAG pipeline instrumentation (2h)
│   ├── Task 11.4: Background job logging (1h)
│   └── Task 11.5: Replace print statements + testing (1h)
├── Day 26: Phase 11 Code Review + Deploy
├── Day 27: Final Integration Testing
├── Day 28: Release v0.5.0
```

### 11.2 Milestones

| Milestone | Date | Deliverable |
|-----------|------|-------------|
| M1: Phase 1 Complete | Day 3 | Gradio removed, FKs added, global error handlers, PRAGMA FK ON |
| M2: Phase 2 Complete | Day 8 | Hybrid layout: schemas/, models/, repositories/, workers/ |
| M3: Phase 3 Complete | Day 10 | FK indexes, constructor injection, Depends() chain |
| M4: Phase 4 Complete | Day 13 | Tests, documentation, quality fixes |
| M5: Phase 5 Complete | Day 15 | High-priority config migrated to SQLite (8 settings) |
| M6: Phase 6 Complete | Day 17 | Redis + ARQ infrastructure ready |
| M7: Phase 7-9 Complete | Day 23 | All background tasks migrated to ARQ |
| M8: Phase 10 Complete | Day 24 | Jobs API, legacy code removed |
| M9: Phase 11 Complete | Day 26 | Structured logging (10 key events) |
| M10: Release v0.5.0 | Day 28 | All changes merged, deployed, tagged |

**Release guardrail:** Phases 1-4 (39h) constitute a self-contained, releasable **v0.5.0-rc** that delivers the core architecture improvements (Gradio removal, FKs, repositories, BaseService, error handlers, Depends() chain, tests). If Phases 6-11 (ARQ + Logging, 30h) cannot be completed in the target window, they can be deferred to **v0.6.0** without blocking the architecture refactor. Phase 5 (Config, 4h) can ship with either release.

---

## 12. Acceptance Criteria

### 12.1 Phase 1 Acceptance

- [ ] `grep -r "gradio" ai_ready_rag/` returns 0 results
- [ ] All FK constraints exist (verified via `PRAGMA foreign_key_list`)
- [ ] `PRAGMA foreign_keys` returns 1 on every connection
- [ ] `grep "HTTPException" ai_ready_rag/services/` returns 0 results
- [ ] Global error handlers registered in `core/error_handlers.py`
- [ ] All existing tests pass
- [ ] Manual smoke test passes on staging

### 12.2 Phase 2 Acceptance (Hybrid Layout)

- [ ] `schemas/` directory exists with 8 schema files
- [ ] `db/models/` directory exists with 7 model files (models.py deleted)
- [ ] `db/mixins.py` exists with `TimestampMixin`
- [ ] `db/repositories/` directory exists with `BaseRepository` and 7 concrete repositories
- [ ] `services/base.py` exists with `BaseService` class
- [ ] `workers/` directory exists with warming_worker.py and cleanup_worker.py
- [ ] All imports updated across codebase (no broken imports)
- [ ] `models/__init__.py` re-exports all models (existing import paths unchanged)
- [ ] `schemas/__init__.py` re-exports all schemas
- [ ] `python -m compileall ai_ready_rag/ -q` passes with 0 errors
- [ ] `TimestampMixin` used by all models with timestamps
- [ ] Experimental endpoints (`/api/experimental/*`) remain functional
- [ ] All existing tests pass

### 12.3 Phase 3 Acceptance

- [ ] FK columns have explicit indexes (`idx_*` naming)
- [ ] `RAGService` uses constructor injection (no lazy `@property` loading)
- [ ] Expensive services (VectorService) initialized in `lifespan`, stored in `app.state`
- [ ] All services injected via FastAPI `Depends()` chain in `core/dependencies.py`
- [ ] Routes use `Depends(get_X_service)` instead of manual instantiation
- [ ] All existing tests pass

**Deferred to post-MVP:**
- Alembic migrations (add when schema stabilizes)
- SQLAlchemy 2.0 `Mapped[]` syntax (cosmetic change)

### 12.4 Phase 4 Acceptance

- [ ] Service test coverage >= 80%
- [ ] No `print()` statements in services (only `logger`)
- [ ] `WarmingWorker` has semaphore limiting concurrent queries
- [ ] `CLAUDE.md` updated with new patterns
- [ ] `README.md` reflects current architecture

### 12.5 Phase 5 Acceptance (High-Priority Config Only)

- [ ] `get_effective_setting()` function implemented in settings_service.py
- [ ] 8 high-priority settings migrated to SQLite (6 security + 2 feature flags)
- [ ] Services use `get_effective_setting()` for migrated values
- [ ] All existing tests pass with new config pattern
- [ ] Manual verification: changing SQLite setting affects runtime behavior
- [ ] Precedence verified: SQLite override takes effect over profile default and environment variable for at least one setting

**Per-setting verification:**

| Setting Key | Default | Verification Test |
|-------------|---------|-------------------|
| `security_jwt_expiration_hours` | 24 | Set to 1 → new tokens expire in 1h |
| `security_password_min_length` | 8 | Set to 20 → registration rejects 15-char password |
| `security_lockout_attempts` | 5 | Set to 2 → 3rd failed login triggers lockout |
| `security_lockout_minutes` | 15 | Set to 1 → lockout clears after 1 minute |
| `security_bcrypt_rounds` | 12 | Set to 10 → new password hashes use 10 rounds |
| `audit_level` | "full_debug" | Set to "essential" → verbose audit entries stop |
| `feature_enable_rag` | true | Set to false → chat returns "RAG disabled" |
| `setup_wizard_completed` | false | Set to true → setup wizard skipped on login |

> **Note:** `security_password_min_length` defaults to 8 for development. Production deployments should increase to 12+ via the admin UI. This setting is runtime-configurable — no code change required.

**Deferred to future work:** Phases 5.3-5.7 (48 additional settings: document processing, pipeline, RAG, cache warming, SCTP/SSE)

### 12.6 Phase 6 Acceptance (ARQ Infrastructure)

- [ ] Redis running and accessible on localhost:6379
- [ ] `arq` and `redis` packages added to requirements
- [ ] `ai_ready_rag/workers/` package created with proper structure
- [ ] `WorkerSettings` class configured with all task functions
- [ ] Redis connection pool implemented in `core/redis.py`
- [ ] ARQ worker starts successfully: `arq ai_ready_rag.workers.settings.WorkerSettings`
- [ ] Worker health check endpoint returns worker status

### 12.7 Phase 7-9 Acceptance (Task Migration)

- [ ] `process_document` task processes documents via ARQ queue
- [ ] `warm_cache` task processes warming jobs via ARQ queue
- [ ] `reindex_knowledge_base` task processes reindex via ARQ queue
- [ ] All tasks report progress to Redis hash
- [ ] All tasks publish progress events for SSE streaming
- [ ] All tasks check for cancellation flag
- [ ] Jobs survive server restart (test: enqueue job, restart server, job completes)
- [ ] Legacy files deleted: warming_worker.py, warming_queue.py, reindex_worker.py

### 12.8 Phase 10 Acceptance (Jobs API)

- [ ] `GET /api/jobs/{id}/status` returns job status and progress
- [ ] `GET /api/jobs/{id}/stream` delivers real-time SSE updates
- [ ] `POST /api/jobs/{id}/cancel` sets cancellation flag
- [ ] Frontend updated to use new job polling/streaming
- [ ] Legacy warming/reindex tables removed from models
- [ ] No `BackgroundTasks.add_task()` calls remain in codebase
- [ ] `data/warming_queue/` directory removed

### 12.9 Phase 11 Acceptance (Observability & Logging)

- [ ] 10 essential log events implemented (see Section 16.2)
- [ ] All HTTP requests logged with request_id, latency, status
- [ ] RAG queries logged with timing breakdown and confidence
- [ ] Background jobs logged with start/complete/fail events
- [ ] Low confidence responses trigger WARN log
- [ ] All `print()` statements replaced with structured logging
- [ ] `grep {request_id}` returns full request trace
- [ ] No sensitive data (passwords, tokens, full query text) in logs
- [ ] `structlog` added to requirements files

### 12.10 Final Acceptance

- [ ] All tests pass (unit + integration)
- [ ] Performance benchmarks within 10% of baseline
- [ ] Background jobs complete successfully under load (10 concurrent jobs)
- [ ] No critical/high bugs in staging for 48 hours
- [ ] Engineering team sign-off
- [ ] Product sign-off on timeline impact

---

## 13. Open Questions

**Status: All questions resolved (2026-02-05)**

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| Q1 | SQLAlchemy 2.0 syntax upgrade timing | **Deferred to post-MVP** | Cosmetic change with no functional benefit; risk outweighs reward during refactor |
| Q2 | 80% service test coverage achievable? | **Yes** | Achievable with focused effort |
| Q3 | ServiceContainer vs Depends() chain | **FastAPI Depends() chain** | ServiceContainer re-introduces lazy loading anti-pattern; Depends() is idiomatic FastAPI |
| Q4 | API backward compatibility | **No** | Optimize API freely, no legacy clients |
| Q5 | Orphan ChatSessions handling | **Drop and recreate tables** | Pre-MVP, no production data to preserve |
| Q6 | Phase 4/5 execution order | **Sequential** | Reduces risk, clearer progress |
| Q7 | Admin UI settings scope | **High-priority only (8 settings)** | Migrate security/policy + feature flags only; defer remaining 26 settings |
| Q8 | Profile-based defaults | **Option 3: Profiles in config.py, SQLite for overrides** | Best balance of simplicity and flexibility |
| Q9 | Redis deployment on DGX Spark | **Docker container** | Docker on Spark is still air-gapped; current workers have been unreliable |
| Q10 | ARQ worker count | **2 workers** | Good balance with LLM workload |
| Q11 | Async chat endpoint | **Sync only** | Caching handles latency; avoid complexity |
| Q12 | Redis persistence | **RDB snapshots** | Simple, adequate for job queue |
| Q13 | Job result TTL | **7-day TTL** | Bounded memory, recent history available |
| Q14 | Query text logging | **Full query text** | Best debugging value for enterprise use |
| Q15 | Error handling strategy | **Global error handlers only** | Routers never catch domain exceptions; `core/error_handlers.py` maps to HTTP |
| Q16 | Alembic timing | **Deferred to post-MVP** | Pre-MVP uses drop-and-recreate; add Alembic when schema stabilizes |
| Q17 | PRAGMA foreign_keys | **ON for every connection** | SQLite ignores FKs by default; must enable via connect event |
| Q18 | Redis as architecture component | **Accepted** | Docker on DGX Spark preserves air-gap; current workers are operationally painful |

---

## 14. Configuration Migration to SQLite

### 14.1 Overview

To establish SQLite as the **single source of truth** for runtime configuration, this section defines the migration of configuration items from `config.py` (Pydantic Settings) to the `admin_settings` SQLite table.

**Principle:** Environment variables and `config.py` should only contain:
- Infrastructure settings (URLs, paths, credentials)
- Bootstrap settings (needed before SQLite is accessible)
- Security secrets (should never be stored in database)

All other settings should be runtime-configurable via the admin UI and stored in SQLite.

### 14.1.1 Configuration Precedence Order

Per Codex review feedback, explicit precedence order (highest to lowest priority):

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SQLite admin_settings table (runtime overrides)          │  ← Highest priority
│    - Set via Admin UI                                       │
│    - Persists across restarts                               │
│    - NULL = use lower priority source                       │
├─────────────────────────────────────────────────────────────┤
│ 2. config.py profile defaults (env_profile: laptop|spark)   │
│    - Deployment-appropriate defaults                        │
│    - laptop: conservative (smaller models, no OCR)          │
│    - spark: full capability (larger models, OCR enabled)    │
├─────────────────────────────────────────────────────────────┤
│ 3. Environment variables (infrastructure only)              │  ← Lowest priority
│    - URLs, paths, secrets                                   │
│    - Set at deployment time                                 │
└─────────────────────────────────────────────────────────────┘
```

**Resolution logic in `SettingsService.get()`:**
```python
def get(self, key: str) -> Any:
    # 1. Check SQLite override
    override = self.db.query(AdminSetting).filter_by(key=key).first()
    if override and override.value is not None:
        return override.value

    # 2. Check profile default
    profile = self.settings.env_profile  # "laptop" or "spark"
    if key in PROFILE_DEFAULTS.get(profile, {}):
        return PROFILE_DEFAULTS[profile][key]

    # 3. Fall back to config.py/environment
    return getattr(self.settings, key, None)
```

### 14.2 Current State

**Already in SQLite (17 settings):**

| SQLite Key | Current Value | Category |
|------------|---------------|----------|
| `chat_model` | "qwen3:8b" | LLM |
| `embedding_model` | "nomic-embed-text:latest" | Embedding |
| `retrieval_top_k` | 12 | Retrieval |
| `retrieval_min_score` | 0.3 | Retrieval |
| `retrieval_enable_expansion` | true | Retrieval |
| `chunk_size` | 368 | Processing |
| `chunk_overlap` | 100 | Processing |
| `cache_enabled` | true | Cache |
| `cache_min_confidence` | 40 | Cache |
| `cache_ttl_hours` | 168 | Cache |
| `cache_max_entries` | 1600 | Cache |
| `cache_semantic_threshold` | 0.95 | Cache |
| `cache_auto_warm_enabled` | true | Cache |
| `cache_auto_warm_count` | 25 | Cache |
| `llm_temperature` | 0.1 | LLM |
| `llm_max_response_tokens` | 2048 | LLM |
| `llm_confidence_threshold` | 40 | LLM |

### 14.3 Settings That MUST Stay in config.py/Environment

These settings are required **before the application starts** or contain sensitive information:

| Setting | Reason | Category |
|---------|--------|----------|
| `database_url` | Bootstrap - needed to connect to SQLite | Infrastructure |
| `jwt_secret_key` | Security - must not be in database | Security |
| `jwt_algorithm` | Security - tightly coupled with secret | Security |
| `admin_email` | Seed data - needed before users exist | Bootstrap |
| `admin_password` | Seed data - needed before users exist | Bootstrap |
| `admin_display_name` | Seed data - needed before users exist | Bootstrap |
| `host` | Server bootstrap - uvicorn needs before app loads | Infrastructure |
| `port` | Server bootstrap - uvicorn needs before app loads | Infrastructure |
| `ollama_base_url` | Infrastructure - external service URL | Infrastructure |
| `qdrant_url` | Infrastructure - external service URL | Infrastructure |
| `api_base_url` | Infrastructure - external reference | Infrastructure |
| `chroma_persist_dir` | File path - OS-level | Infrastructure |
| `upload_dir` | File path - OS-level | Infrastructure |
| `warming_queue_dir` | File path - OS-level | Infrastructure |
| `env_profile` | Bootstrap - determines defaults for other settings | Bootstrap |
| `sctp_tls_cert` | Security - file paths for TLS | Security |
| `sctp_tls_key` | Security - file paths for TLS | Security |
| `sctp_tls_ca` | Security - file paths for TLS | Security |
| `sctp_shared_secret` | Security - sensitive credential | Security |
| `debug` | Bootstrap - affects logging initialization | Bootstrap |
| `app_name` | Static metadata | Metadata |
| `app_version` | Static metadata | Metadata |

**Total: 22 settings (must remain in config.py)**

### 14.4 Settings to Migrate to SQLite

The following 56 settings should be migrated to SQLite for runtime configuration (8 in v0.5.0, 48 deferred):

#### 14.4.1 Phase 5.1: Security/Policy Settings (High Priority)

| config.py Setting | SQLite Key | Default | Notes |
|-------------------|------------|---------|-------|
| `jwt_expiration_hours` | `security_jwt_expiration_hours` | 24 | Token lifetime |
| `password_min_length` | `security_password_min_length` | 8 | Password policy (increase to 12+ for production) |
| `lockout_attempts` | `security_lockout_attempts` | 5 | Brute force protection |
| `lockout_minutes` | `security_lockout_minutes` | 15 | Lockout duration |
| `bcrypt_rounds` | `security_bcrypt_rounds` | 12 | Hashing cost |
| `audit_level` | `audit_level` | "full_debug" | Logging verbosity |

#### 14.4.2 Phase 5.2: Feature Flags (High Priority)

| config.py Setting | SQLite Key | Default | Notes |
|-------------------|------------|---------|-------|
| `enable_rag` | `feature_enable_rag` | true | RAG functionality toggle |
| `skip_setup_wizard` | `setup_wizard_completed` | false | One-time setup flag |

#### 14.4.3 — 14.4.7: DEFERRED TO FUTURE WORK

The following 48 settings (Phases 5.3-5.7) are deferred until an admin needs to change them at runtime. They remain in `config.py` where they work correctly today.

<details>
<summary>Click to expand deferred settings (48 total)</summary>

#### Phase 5.3: Document Processing (Medium Priority)

| config.py Setting | SQLite Key | Default | Notes |
|-------------------|------------|---------|-------|
| `max_upload_size_mb` | `upload_max_size_mb` | 100 | Upload limit |
| `max_storage_gb` | `upload_max_storage_gb` | 10 | Storage limit |
| `allowed_extensions` | `upload_allowed_extensions` | JSON array | File filter |
| `enable_ocr` | `ocr_enabled` | profile-based | OCR toggle |
| `ocr_language` | `ocr_language` | "eng" | OCR language |
| `force_full_page_ocr` | `ocr_force_full_page` | false | OCR mode |
| `table_extraction_mode` | `document_table_extraction_mode` | "accurate" | Processing quality |
| `include_image_descriptions` | `document_include_image_descriptions` | true | Image processing |

#### 14.4.4 Phase 5.4: Pipeline Settings (Medium Priority)

| config.py Setting | SQLite Key | Default | Notes |
|-------------------|------------|---------|-------|
| `vector_backend` | `vector_backend` | profile-based | "chroma" or "qdrant" |
| `chunker_backend` | `chunker_backend` | profile-based | "simple" or "docling" |
| `qdrant_collection` | `qdrant_collection` | "documents" | Collection name |
| `embedding_dimension` | `embedding_dimension` | 768 | Model config |
| `embedding_max_tokens` | `embedding_max_tokens` | 8192 | Model config |
| `default_tenant_id` | `default_tenant_id` | "default" | Multi-tenancy |

#### 14.4.5 Phase 5.5: RAG Service Settings (Medium Priority)

| config.py Setting | SQLite Key | Default | Notes |
|-------------------|------------|---------|-------|
| `rag_timeout_seconds` | `rag_timeout_seconds` | 30 | LLM timeout |
| `rag_admin_email` | `rag_admin_email` | "admin@company.com" | Route-to-human |
| `rag_max_context_tokens` | `rag_max_context_tokens` | profile-based | Token budget |
| `rag_max_history_tokens` | `rag_max_history_tokens` | profile-based | Token budget |
| `rag_system_prompt_tokens` | `rag_system_prompt_tokens` | 500 | Token budget |
| `rag_max_chunks_per_doc` | `retrieval_max_chunks_per_doc` | 3 | Retrieval tuning |
| `rag_total_context_chunks` | `retrieval_total_context_chunks` | 8 | Retrieval tuning |
| `rag_dedup_candidates_cap` | `retrieval_dedup_candidates_cap` | 15 | Retrieval tuning |
| `rag_chunk_overlap_threshold` | `retrieval_chunk_overlap_threshold` | 0.9 | Retrieval tuning |
| `rag_enable_hallucination_check` | `rag_enable_hallucination_check` | profile-based | Quality flag |

#### 14.4.6 Phase 5.6: Cache Warming Settings (Lower Priority)

| config.py Setting | SQLite Key | Default | Notes |
|-------------------|------------|---------|-------|
| `warming_delay_seconds` | `warming_delay_seconds` | 2.0 | Worker tuning |
| `warming_scan_interval_seconds` | `warming_scan_interval_seconds` | 60 | Worker tuning |
| `warming_lock_timeout_minutes` | `warming_lock_timeout_minutes` | 30 | Worker tuning |
| `warming_max_file_size_mb` | `warming_max_file_size_mb` | 10.0 | Upload limit |
| `warming_allowed_extensions` | `warming_allowed_extensions` | JSON array | Upload filter |
| `warming_archive_completed` | `warming_archive_completed` | false | Audit flag |
| `warming_checkpoint_interval` | `warming_checkpoint_interval` | 10 | Worker tuning |
| `warming_checkpoint_time_seconds` | `warming_checkpoint_time_seconds` | 5 | Worker tuning |
| `warming_lease_duration_minutes` | `warming_lease_duration_minutes` | 10 | Worker tuning |
| `warming_lease_renewal_seconds` | `warming_lease_renewal_seconds` | 60 | Worker tuning |
| `warming_max_retries` | `warming_max_retries` | 3 | Retry config |
| `warming_retry_delays` | `warming_retry_delays` | "5,30,120" | Retry config |
| `warming_cancel_timeout_seconds` | `warming_cancel_timeout_seconds` | 5 | Timeout |
| `warming_completed_retention_days` | `warming_completed_retention_days` | 7 | Cleanup policy |
| `warming_failed_retention_days` | `warming_failed_retention_days` | 30 | Cleanup policy |
| `warming_cleanup_interval_hours` | `warming_cleanup_interval_hours` | 6 | Cleanup schedule |

#### 14.4.7 Phase 5.7: SCTP & SSE Settings (Lower Priority)

| config.py Setting | SQLite Key | Default | Notes |
|-------------------|------------|---------|-------|
| `sctp_enabled` | `sctp_enabled` | false | Feature flag |
| `sctp_host` | `sctp_host` | "0.0.0.0" | Service config |
| `sctp_port` | `sctp_port` | 9900 | Service config |
| `sctp_max_file_size_mb` | `sctp_max_file_size_mb` | 10 | Upload limit |
| `sctp_max_queries_per_file` | `sctp_max_queries_per_file` | 10000 | Limit |
| `sctp_allowed_ips` | `sctp_allowed_ips` | null | Security filter |
| `sse_event_buffer_size` | `sse_event_buffer_size` | 1000 | Performance |
| `sse_heartbeat_seconds` | `sse_heartbeat_seconds` | 30 | Performance |

</details>

### 14.5 Implementation Pattern

#### 14.5.1 Fallback Strategy

Settings should fall back to config.py defaults when not present in SQLite:

```python
# ai_ready_rag/services/settings_service.py

def get_effective_setting(
    db: Session,
    key: str,
    config_default: Any,
    cast_type: type = str
) -> Any:
    """Get setting from SQLite, falling back to config.py default."""
    db_value = get_admin_setting(db, key)
    if db_value is not None:
        if cast_type == bool:
            return db_value.lower() in ("true", "1", "yes")
        elif cast_type == int:
            return int(db_value)
        elif cast_type == float:
            return float(db_value)
        elif cast_type == list:
            return json.loads(db_value)
        return db_value
    return config_default
```

#### 14.5.2 Service Usage Pattern

```python
# In service code, replace:
settings = get_settings()
timeout = settings.rag_timeout_seconds

# With:
from ai_ready_rag.services.settings_service import get_effective_setting

timeout = get_effective_setting(
    db,
    "rag_timeout_seconds",
    config_default=get_settings().rag_timeout_seconds,
    cast_type=int
)
```

#### 14.5.3 Migration Script

```python
# scripts/migrate_config_to_sqlite.py
"""
Migrate config.py defaults to SQLite for runtime configuration.
Run once per environment after deployment.
"""
from ai_ready_rag.db.database import SessionLocal
from ai_ready_rag.services.settings_service import set_admin_setting
from ai_ready_rag.config import get_settings

SETTINGS_TO_MIGRATE = {
    # Phase 5.1: Security/Policy
    "security_jwt_expiration_hours": ("jwt_expiration_hours", int),
    "security_password_min_length": ("password_min_length", int),
    "security_lockout_attempts": ("lockout_attempts", int),
    "security_lockout_minutes": ("lockout_minutes", int),
    "security_bcrypt_rounds": ("bcrypt_rounds", int),
    "audit_level": ("audit_level", str),
    # ... additional settings
}

def migrate_settings():
    settings = get_settings()
    db = SessionLocal()
    try:
        for sqlite_key, (config_attr, cast_type) in SETTINGS_TO_MIGRATE.items():
            value = getattr(settings, config_attr)
            if value is not None:
                set_admin_setting(db, sqlite_key, str(value))
                print(f"  ✓ {sqlite_key} = {value}")
        db.commit()
        print("Migration complete")
    finally:
        db.close()

if __name__ == "__main__":
    migrate_settings()
```

### 14.6 Effort Estimate

| Sub-Phase | Settings Count | Effort | Status |
|-----------|---------------|--------|--------|
| 5.1 Security/Policy | 6 | 2 hours | **In scope** |
| 5.2 Feature Flags | 2 | 1 hour | **In scope** |
| Migration script + testing | — | 1 hour | **In scope** |
| 5.3 Document Processing | 8 | 2 hours | Deferred |
| 5.4 Pipeline Settings | 6 | 2 hours | Deferred |
| 5.5 RAG Service Settings | 10 | 3 hours | Deferred |
| 5.6 Cache Warming | 16 | 3 hours | Deferred |
| 5.7 SCTP & SSE | 8 | 2 hours | Deferred |
| **v0.5.0 Total** | **8** | **4 hours** | |
| **Future Total** | **48** | **12 hours** | |

### 14.7 Admin UI Updates

The React admin dashboard will need UI updates to expose the 8 high-priority settings:

1. **Security Settings Panel** - JWT expiration, password policy, lockout settings

**Note:** Additional panels for deferred settings (RAG, processing, warming, SCTP) will be planned as follow-on work.

### 14.8 Settings Summary

| Category | Count | Location |
|----------|-------|----------|
| Must stay in config.py | 22 | Environment/config.py |
| Already in SQLite | 17 | admin_settings table |
| To migrate in v0.5.0 | 8 | admin_settings table |
| Deferred migration | 48 | config.py (migrate later) |
| **Total** | **95** | Mixed |

After v0.5.0 migration (high-priority only):
- **Environment/config.py:** 22 settings - Infrastructure, security, bootstrap
- **SQLite (existing):** 17 settings - Already runtime configurable
- **SQLite (new in v0.5.0):** 8 settings - Security/policy + feature flags
- **Remaining in config.py:** 48 settings - Deferred to future migration

---

## 15. Unified Background Task System (ARQ + Redis)

### 15.1 Overview

This section defines the migration from the current fragmented background task system to a unified **ARQ + Redis** architecture. ARQ (Async Redis Queue) is a production-grade, async-native job queue that provides robust, scalable background task processing.

#### 15.1.1 Current State Problems

| Problem | Impact | Current Workaround |
|---------|--------|-------------------|
| **4 different patterns** | Maintenance burden, inconsistent behavior | None - each feature evolved separately |
| **Tasks lost on restart** | Work lost, stuck documents | Startup recovery scripts |
| **No horizontal scaling** | Single-server bottleneck | None |
| **Polling-based progress** | 2-60s delays, wasted resources | Manual refresh in UI |
| **Stuck job detection only at startup** | Jobs hang indefinitely | Restart server |
| **Poor error visibility** | Hard to debug failures | Manual log inspection |

#### 15.1.2 Target State Benefits

| Benefit | Description |
|---------|-------------|
| **One unified pattern** | All background tasks use same infrastructure |
| **Crash recovery** | Jobs survive server/worker restarts |
| **Horizontal scaling** | Add workers to increase throughput |
| **Real-time progress** | Redis pub/sub for instant updates |
| **Built-in retry** | Automatic exponential backoff |
| **Dead worker detection** | Health checks identify stuck workers |
| **Battle-tested** | ARQ used in production by many companies |

### 15.2 Architecture

#### 15.2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DGX Spark Server                              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         FastAPI Application                      │   │
│  │                                                                  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │   │
│  │  │ /upload  │  │ /chat    │  │ /reindex │  │ /warm    │        │   │
│  │  │ endpoint │  │ endpoint │  │ endpoint │  │ endpoint │        │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │   │
│  │       │              │              │              │             │   │
│  │       └──────────────┴──────────────┴──────────────┘             │   │
│  │                              │                                   │   │
│  │                              ▼                                   │   │
│  │                    ┌─────────────────┐                          │   │
│  │                    │  ARQ Producer   │                          │   │
│  │                    │  (enqueue jobs) │                          │   │
│  │                    └────────┬────────┘                          │   │
│  └─────────────────────────────┼────────────────────────────────────┘   │
│                                │                                        │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Redis (localhost:6379)                      │   │
│  │                                                                  │   │
│  │   Queue: [doc_job, warm_job, reindex_job, ...]                  │   │
│  │   Results: {job_123: {...}, job_456: {...}}                     │   │
│  │   Progress: {progress:123: {current: 5, total: 10}}             │   │
│  │   Health: {worker_1: alive, worker_2: alive}                    │   │
│  │                                                                  │   │
│  └─────────────────────────────┬────────────────────────────────────┘   │
│                                │                                        │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      ARQ Worker Process(es)                      │   │
│  │                                                                  │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │                    Task Handlers                          │   │   │
│  │  │  ┌────────────────┐  ┌────────────────┐                 │   │   │
│  │  │  │ process_doc()  │  │ warm_cache()   │                 │   │   │
│  │  │  └────────────────┘  └────────────────┘                 │   │   │
│  │  │  ┌────────────────┐  ┌────────────────┐                 │   │   │
│  │  │  │ reindex_all()  │  │ async_chat()   │                 │   │   │
│  │  │  └────────────────┘  └────────────────┘                 │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│           ┌──────────────┼──────────────┐                              │
│           ▼              ▼              ▼                              │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐                          │
│     │  SQLite  │  │  Qdrant  │  │  Ollama  │                          │
│     │  (app)   │  │ (vectors)│  │  (LLM)   │                          │
│     └──────────┘  └──────────┘  └──────────┘                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 15.2.2 Job Lifecycle Flow

```
     User Request                Redis                    ARQ Worker
          │                        │                          │
    ┌─────▼─────┐                  │                          │
    │  POST     │                  │                          │
    │ /upload   │                  │                          │
    └─────┬─────┘                  │                          │
          │                        │                          │
          │  ① ENQUEUE JOB         │                          │
          │ ───────────────────────▶                          │
          │  LPUSH arq:queue       │                          │
          │                        │                          │
    ┌─────▼─────┐                  │                          │
    │  Return   │                  │  ② WORKER WAITING        │
    │  202 +    │                  │ ◀─────────────────────────
    │  job_id   │                  │  BRPOP (blocking pop)    │
    └───────────┘                  │                          │
                                   │  ③ JOB DELIVERED         │
                                   │ ─────────────────────────▶
                                   │                          │
                                   │                    ┌─────▼─────┐
                                   │                    │  Execute  │
                                   │                    │  Task     │
                                   │                    └─────┬─────┘
                                   │                          │
                                   │  ④ PROGRESS UPDATES      │
                                   │ ◀─────────────────────────
                                   │  HSET progress:{job_id}  │
                                   │  PUBLISH progress:{id}   │
                                   │                          │
                                   │  ⑤ STORE RESULT          │
                                   │ ◀─────────────────────────
                                   │  HSET arq:result:{id}    │
```

#### 15.2.3 Redis Data Structures

```
┌─────────────────────────────────────────────────────────────────────────┐
│  KEY: arq:queue:default                                                 │
│  TYPE: List (FIFO Queue)                                                │
│  PURPOSE: Pending jobs waiting to be processed                          │
│                                                                         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                               │
│  │Job 5│ │Job 4│ │Job 3│ │Job 2│ │Job 1│  ◀── Workers pop from right   │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘      (BRPOP - blocking)       │
│     ▲                                                                   │
│  New jobs pushed to left (LPUSH)                                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  KEY: arq:job:{job_id}                                                  │
│  TYPE: Hash (Dictionary)                                                │
│  PURPOSE: Job metadata and status                                       │
│                                                                         │
│  {                                                                      │
│    "function": "process_document",                                      │
│    "args": "[\"doc-123\", {\"ocr\": true}]",                           │
│    "status": "queued|in_progress|complete|failed",                     │
│    "enqueue_time": "1707134400.123",                                   │
│    "start_time": "1707134401.456",                                     │
│    "result": "{\"chunks\": 42}"                                        │
│  }                                                                      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  KEY: progress:{job_id}                                                 │
│  TYPE: Hash                                                             │
│  PURPOSE: Real-time progress tracking                                   │
│                                                                         │
│  {                                                                      │
│    "current": "42",                                                     │
│    "total": "100",                                                      │
│    "percent": "42",                                                     │
│    "message": "Embedding chunk 42 of 100"                              │
│  }                                                                      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  KEY: arq:health:{worker_id}                                            │
│  TYPE: String (with TTL)                                                │
│  PURPOSE: Worker heartbeat (auto-expires if worker dies)                │
│                                                                         │
│  Value: "1707134400.123"                                               │
│  TTL: 30 seconds (refreshed every 10 seconds)                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 15.2.4 Horizontal Scaling

```
                        Single Worker (Current)
    ════════════════════════════════════════════════════════════════════

                              ┌──────────┐
                              │  Redis   │
                              └────┬─────┘
                                   │
                                   ▼
                            ┌────────────┐
                            │  Worker 1  │
                            └────────────┘

                        Throughput: ~500 jobs/hour


    ════════════════════════════════════════════════════════════════════
                        Multiple Workers (Scaled)
    ════════════════════════════════════════════════════════════════════

                              ┌──────────┐
                              │  Redis   │
                              └────┬─────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
        ┌────────────┐      ┌────────────┐      ┌────────────┐
        │  Worker 1  │      │  Worker 2  │      │  Worker 3  │
        └────────────┘      └────────────┘      └────────────┘

                        Throughput: ~1500 jobs/hour

    To scale: Just start more workers!
    $ arq workers.tasks.WorkerSettings  # Start another worker
```

### 15.3 Infrastructure Requirements

#### 15.3.1 Redis Installation

**Option A: Docker (Recommended)**
```bash
# Run Redis as container (~6MB memory)
docker run -d \
  --name redis \
  --restart unless-stopped \
  -p 6379:6379 \
  -v redis-data:/data \
  redis:alpine \
  redis-server --appendonly yes
```

**Option B: Native Installation**
```bash
# Ubuntu/Debian
apt install redis-server
systemctl enable redis-server
systemctl start redis-server
```

#### 15.3.2 Python Dependencies

```
# requirements-wsl.txt / requirements-spark.txt
arq>=0.25.0
redis>=5.0.0
sse-starlette>=1.6.0  # For SSE progress streaming
```

#### 15.3.3 Configuration

```python
# ai_ready_rag/config.py

class Settings(BaseSettings):
    # ... existing settings ...

    # Redis connection
    redis_url: str = "redis://localhost:6379/0"

    # ARQ Worker settings
    arq_max_jobs: int = 100           # Max jobs before worker restarts
    arq_job_timeout: int = 1800       # 30 minutes max per job
    arq_max_tries: int = 3            # Retry attempts
    arq_retry_delay: int = 10         # Seconds between retries
    arq_health_check_interval: int = 10  # Health check frequency
    arq_result_ttl: int = 3600        # Keep results for 1 hour
```

#### 15.3.4 Redis Failure Policy

Redis is **required for background tasks** but **not required for core functionality** (chat, auth, document listing). The application must never refuse to start because Redis is down.

**Failure scenarios:**

| Scenario | Behavior | User Impact |
|----------|----------|------------|
| **Redis down at startup** | FastAPI starts in **degraded mode**. Background task endpoints return 503. Health endpoint reports `redis: "unavailable"`. Admin warned on login. | Document upload, warming, reindex disabled. Chat (synchronous) still works. |
| **Redis down mid-operation** | Enqueue calls raise `RedisConnectionError` → caught by global error handler → returns `503 Service Unavailable` with message "Background processing temporarily unavailable" | User sees error, can retry later |
| **Redis recovers** | Next enqueue call succeeds automatically (`redis-py` reconnects on next attempt). No manual intervention needed. | Transparent recovery |
| **Worker dies mid-job** | ARQ detects via heartbeat TTL (30s). Job returns to queue after `arq_job_timeout`. Next healthy worker picks it up. | Job delayed, not lost |

**Implementation in `core/redis.py`:**
```python
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

_redis_pool: Redis | None = None

async def get_redis_pool() -> Redis | None:
    """Get Redis connection pool. Returns None if Redis unavailable."""
    global _redis_pool
    if _redis_pool is None:
        try:
            _redis_pool = Redis.from_url(settings.redis_url)
            await _redis_pool.ping()
        except RedisConnectionError:
            _redis_pool = None
            return None
    return _redis_pool

async def is_redis_healthy() -> bool:
    """Check Redis connectivity for health endpoint."""
    pool = await get_redis_pool()
    if pool is None:
        return False
    try:
        await pool.ping()
        return True
    except RedisConnectionError:
        return False
```

**Health check integration:**
```python
# In /api/health endpoint
redis_status = "healthy" if await is_redis_healthy() else "unavailable"
return {
    ...
    "redis": redis_status,
    "background_tasks": "available" if redis_status == "healthy" else "degraded",
}
```

**Enqueue guard pattern:**
```python
# In any endpoint that enqueues background work
async def enqueue_or_fail(task_name: str, *args):
    """Enqueue job or raise 503 if Redis unavailable."""
    pool = await get_redis_pool()
    if pool is None:
        raise ExternalServiceError(
            "Background processing temporarily unavailable",
            context={"service": "redis"},
        )
    job = await pool.enqueue_job(task_name, *args)
    return job
```

### 15.4 Implementation

#### 15.4.1 Directory Structure

```
ai_ready_rag/
├── workers/                    # Background task package
│   ├── __init__.py
│   ├── settings.py             # ARQ WorkerSettings
│   ├── tasks/                  # Task handlers
│   │   ├── __init__.py
│   │   ├── document.py         # process_document task
│   │   ├── warming.py          # warm_cache task
│   │   ├── reindex.py          # reindex_knowledge_base task
│   │   └── chat.py             # async_chat task (optional)
│   └── utils.py                # Shared utilities
├── api/
│   └── jobs.py                 # Job status/progress endpoints
└── core/
    └── redis.py                # Redis connection pool
```

#### 15.4.2 Worker Settings

```python
# ai_ready_rag/workers/settings.py

from arq import cron
from arq.connections import RedisSettings
from datetime import timedelta

from ai_ready_rag.config import get_settings
from ai_ready_rag.workers.tasks import (
    process_document,
    warm_cache,
    reindex_knowledge_base,
    cleanup_old_jobs,
)

settings = get_settings()

class WorkerSettings:
    """ARQ worker configuration."""

    # Task functions to register
    functions = [
        process_document,
        warm_cache,
        reindex_knowledge_base,
    ]

    # Redis connection
    redis_settings = RedisSettings.from_dsn(settings.redis_url)

    # Retry configuration
    max_tries = settings.arq_max_tries
    retry_delay = timedelta(seconds=settings.arq_retry_delay)

    # Job timeout
    job_timeout = timedelta(seconds=settings.arq_job_timeout)

    # Health check
    health_check_interval = timedelta(seconds=settings.arq_health_check_interval)

    # Result TTL
    keep_result = timedelta(seconds=settings.arq_result_ttl)

    # Queue name
    queue_name = 'arq:queue'

    # Cron jobs (periodic tasks)
    cron_jobs = [
        cron(cleanup_old_jobs, hour=3, minute=0),  # Daily at 3am
    ]
```

#### 15.4.3 Task Handlers

```python
# ai_ready_rag/workers/tasks/document.py

import asyncio
from ai_ready_rag.db.database import SessionLocal
from ai_ready_rag.db.models import Document
from ai_ready_rag.services.processing_service import ProcessingService
from ai_ready_rag.services.vector_service import VectorService
from ai_ready_rag.config import get_settings

async def process_document(ctx, document_id: str, options: dict) -> dict:
    """
    Process uploaded document: parse → chunk → embed → index.

    Args:
        ctx: ARQ context (has redis connection, job_id)
        document_id: UUID of document to process
        options: Processing options (ocr, etc.)

    Returns:
        Result dict with chunk count and timing
    """
    redis = ctx['redis']
    job_id = ctx['job_id']
    settings = get_settings()

    db = SessionLocal()
    try:
        # Get document
        doc = db.get(Document, document_id)
        if not doc:
            raise ValueError(f"Document {document_id} not found")

        doc.status = "processing"
        db.commit()

        # Initialize services
        vector_service = VectorService(settings)
        await vector_service.initialize()
        processing_service = ProcessingService(settings, vector_service)

        # Parse and chunk
        chunks = await processing_service.chunk_document(doc.file_path, options)
        total = len(chunks)

        # Embed and index with progress
        for i, chunk in enumerate(chunks):
            # Check for cancellation
            if await redis.get(f"cancel:{job_id}"):
                doc.status = "cancelled"
                db.commit()
                return {"status": "cancelled", "processed": i}

            await vector_service.embed_and_index(chunk, doc.id)

            # Report progress
            await redis.hset(f"progress:{job_id}", mapping={
                "current": i + 1,
                "total": total,
                "percent": int((i + 1) / total * 100),
                "message": f"Embedding chunk {i+1} of {total}",
            })

            # Publish for real-time SSE
            await redis.publish(f"progress:{job_id}", f"{i+1}/{total}")

        # Mark complete
        doc.status = "ready"
        doc.chunk_count = total
        db.commit()

        return {"chunks_processed": total, "document_id": document_id}

    except Exception as e:
        doc.status = "failed"
        doc.error_message = str(e)
        db.commit()
        raise  # ARQ will handle retry

    finally:
        db.close()
```

```python
# ai_ready_rag/workers/tasks/warming.py

async def warm_cache(ctx, queries: list[str], options: dict) -> dict:
    """Warm cache with list of queries."""
    redis = ctx['redis']
    job_id = ctx['job_id']
    settings = get_settings()

    results = {"processed": 0, "failed": 0, "skipped": 0}
    total = len(queries)

    db = SessionLocal()
    try:
        vector_service = VectorService(settings)
        await vector_service.initialize()
        rag_service = RAGService(settings, vector_service)

        for i, query in enumerate(queries):
            # Check for cancellation
            if await redis.get(f"cancel:{job_id}"):
                results["cancelled"] = True
                break

            try:
                await rag_service.generate(
                    RAGRequest(query=query, user_tags=["*"]),
                    db
                )
                results["processed"] += 1
            except Exception as e:
                results["failed"] += 1
                logger.warning(f"Query {i} failed: {query[:50]}... - {e}")

            # Progress update
            await redis.hset(f"progress:{job_id}", mapping={
                "current": i + 1,
                "total": total,
                "percent": int((i + 1) / total * 100),
                "processed": results["processed"],
                "failed": results["failed"],
            })

            # Rate limiting (configurable)
            await asyncio.sleep(options.get("delay_seconds", 2))

        return results

    finally:
        db.close()
```

```python
# ai_ready_rag/workers/tasks/reindex.py

async def reindex_knowledge_base(ctx, options: dict) -> dict:
    """Full knowledge base reindex."""
    redis = ctx['redis']
    job_id = ctx['job_id']
    settings = get_settings()

    db = SessionLocal()
    try:
        docs = db.query(Document).filter(Document.status == "ready").all()
        total = len(docs)
        results = {"processed": 0, "failed": 0}

        vector_service = VectorService(settings)
        await vector_service.initialize()

        for i, doc in enumerate(docs):
            # Check for cancellation
            if await redis.get(f"cancel:{job_id}"):
                results["cancelled"] = True
                break

            try:
                # Delete old vectors
                await vector_service.delete_document(doc.id)

                # Reprocess
                await process_document(ctx, doc.id, options)
                results["processed"] += 1

            except Exception as e:
                results["failed"] += 1
                logger.error(f"Reindex failed for {doc.id}: {e}")

            await redis.hset(f"progress:{job_id}", mapping={
                "current": i + 1,
                "total": total,
                "percent": int((i + 1) / total * 100),
            })

        return results

    finally:
        db.close()
```

#### 15.4.3.1 Domain Exception Handling in Workers

Per Codex review: ARQ workers must handle domain exceptions (not just `HTTPException` in routers).

```python
# ai_ready_rag/workers/tasks/base.py

from ai_ready_rag.core.exceptions import AppError, NotFoundError, ValidationError
import structlog

logger = structlog.get_logger(__name__)

def handle_task_exception(ctx: dict, error: Exception) -> dict:
    """
    Convert domain exceptions to job failure results.
    Called in except block of task handlers.
    """
    job_id = ctx.get('job_id', 'unknown')

    if isinstance(error, NotFoundError):
        logger.warning("job_failed_not_found", job_id=job_id, error=str(error))
        return {"status": "failed", "error": "not_found", "message": str(error), "retry": False}

    if isinstance(error, ValidationError):
        logger.warning("job_failed_validation", job_id=job_id, error=str(error))
        return {"status": "failed", "error": "validation", "message": str(error), "retry": False}

    if isinstance(error, AppError):
        logger.error("job_failed_app_error", job_id=job_id, error=str(error))
        return {"status": "failed", "error": "app_error", "message": str(error), "retry": True}

    # Unknown exception - log full traceback, allow retry
    logger.exception("job_failed_unexpected", job_id=job_id, error=str(error))
    return {"status": "failed", "error": "unexpected", "message": str(error), "retry": True}
```

**Usage in task handlers:**
```python
async def process_document(ctx, document_id: str, options: dict) -> dict:
    try:
        # ... task logic ...
        return {"status": "success", ...}
    except Exception as e:
        return handle_task_exception(ctx, e)
```

**Key behaviors:**
- `NotFoundError`, `ValidationError`: No retry (user error)
- `AppError`: Retry (transient failure)
- Unknown exceptions: Retry with full traceback logging

#### 15.4.4 API Endpoints

```python
# ai_ready_rag/api/jobs.py

from fastapi import APIRouter, Depends, Request
from arq import create_pool
from arq.jobs import Job
from sse_starlette.sse import EventSourceResponse

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.dependencies import get_current_user

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

async def get_redis_pool():
    settings = get_settings()
    return await create_pool(RedisSettings.from_dsn(settings.redis_url))


@router.post("/documents/{doc_id}/process")
async def start_document_processing(
    doc_id: str,
    options: ProcessingOptions = ProcessingOptions(),
    current_user: User = Depends(get_current_user),
):
    """Start async document processing."""
    redis = await get_redis_pool()

    job = await redis.enqueue_job(
        'process_document',
        doc_id,
        options.dict(),
        _job_id=f"doc-{doc_id}",  # Idempotent
    )

    return {
        "job_id": job.job_id,
        "status": "queued",
        "poll_url": f"/api/jobs/{job.job_id}/status",
        "stream_url": f"/api/jobs/{job.job_id}/stream",
    }


@router.get("/{job_id}/status")
async def get_job_status(job_id: str):
    """Get current job status and progress."""
    redis = await get_redis_pool()

    job = Job(job_id, redis)
    info = await job.info()
    progress = await redis.hgetall(f"progress:{job_id}")

    return {
        "job_id": job_id,
        "status": info.status if info else "unknown",
        "progress": {
            "current": int(progress.get(b"current", 0)),
            "total": int(progress.get(b"total", 0)),
            "percent": int(progress.get(b"percent", 0)),
            "message": progress.get(b"message", b"").decode(),
        } if progress else None,
        "result": await job.result(timeout=0) if info and info.status == "complete" else None,
        "error": str(info.result) if info and info.status == "failed" else None,
    }


@router.get("/{job_id}/stream")
async def stream_job_progress(job_id: str, request: Request):
    """Stream job progress via Server-Sent Events."""
    redis = await get_redis_pool()

    async def event_generator():
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"progress:{job_id}")

        try:
            # Send initial state
            progress = await redis.hgetall(f"progress:{job_id}")
            if progress:
                yield {"event": "progress", "data": json.dumps(dict(progress))}

            # Stream updates
            async for message in pubsub.listen():
                if await request.is_disconnected():
                    break

                if message["type"] == "message":
                    progress = await redis.hgetall(f"progress:{job_id}")
                    yield {"event": "progress", "data": json.dumps(dict(progress))}

                    # Check completion
                    job = Job(job_id, redis)
                    info = await job.info()
                    if info and info.status in ("complete", "failed"):
                        yield {
                            "event": "complete",
                            "data": json.dumps({
                                "status": info.status,
                                "result": await job.result(timeout=0)
                            })
                        }
                        break
        finally:
            await pubsub.unsubscribe(f"progress:{job_id}")

    return EventSourceResponse(event_generator())


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str, current_user: User = Depends(get_admin_user)):
    """Cancel a running job."""
    redis = await get_redis_pool()
    await redis.set(f"cancel:{job_id}", "1", ex=3600)
    return {"status": "cancel_requested"}
```

### 15.5 Migration Plan

#### 15.5.1 Phase 6: ARQ Infrastructure (4 hours)

| Task | Description | Effort |
|------|-------------|--------|
| 6.1 | Install Redis (Docker or native) | 0.5h |
| 6.2 | Add ARQ dependencies to requirements | 0.5h |
| 6.3 | Create `workers/` package structure | 1h |
| 6.4 | Create `WorkerSettings` configuration | 1h |
| 6.5 | Add Redis connection pool to core | 1h |

#### 15.5.2 Phase 7: Migrate Document Processing (6 hours)

| Task | Description | Effort |
|------|-------------|--------|
| 7.1 | Create `process_document` task handler | 2h |
| 7.2 | Update `POST /documents/upload` to enqueue job | 1h |
| 7.3 | Add progress tracking to handler | 1h |
| 7.4 | Update document status from job result | 1h |
| 7.5 | Remove old `BackgroundTasks` code | 1h |

#### 15.5.3 Phase 8: Migrate Cache Warming (6 hours)

| Task | Description | Effort |
|------|-------------|--------|
| 8.1 | Create `warm_cache` task handler | 2h |
| 8.2 | Migrate warming endpoints to use ARQ | 2h |
| 8.3 | Remove `warming_worker.py` and `warming_queue.py` | 1h |
| 8.4 | Update SSE progress to use Redis pub/sub | 1h |

#### 15.5.4 Phase 9: Migrate Reindexing (4 hours)

| Task | Description | Effort |
|------|-------------|--------|
| 9.1 | Create `reindex_knowledge_base` task handler | 2h |
| 9.2 | Update reindex endpoints to use ARQ | 1h |
| 9.3 | Remove `reindex_worker.py` and `reindex_service.py` | 1h |

#### 15.5.5 Phase 10: Jobs API & Cleanup (4 hours)

| Task | Description | Effort |
|------|-------------|--------|
| 10.1 | Create `/api/jobs` endpoints | 2h |
| 10.2 | Add SSE progress streaming | 1h |
| 10.3 | Remove legacy warming/reindex tables | 0.5h |
| 10.4 | Update frontend to use new job API | 0.5h |

### 15.6 Deployment

#### 15.6.1 Service Configuration

**systemd service for ARQ worker:**
```ini
# /etc/systemd/system/arq-worker.service
[Unit]
Description=ARQ Background Worker
After=network.target redis.service

[Service]
Type=simple
User=ve-rag
WorkingDirectory=/srv/VE-RAG-System
Environment=PYTHONPATH=/srv/VE-RAG-System
ExecStart=/srv/VE-RAG-System/.venv/bin/arq ai_ready_rag.workers.settings.WorkerSettings
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Commands:**
```bash
# Enable and start
systemctl enable arq-worker
systemctl start arq-worker

# Scale to multiple workers
systemctl enable arq-worker@{1..3}
systemctl start arq-worker@{1..3}

# Check status
systemctl status arq-worker
journalctl -u arq-worker -f
```

#### 15.6.2 Monitoring

```python
# Health check endpoint
@router.get("/api/admin/workers/health")
async def get_worker_health():
    """Get ARQ worker health status."""
    redis = await get_redis_pool()

    # Find all worker health keys
    workers = []
    async for key in redis.scan_iter("arq:health:*"):
        worker_id = key.decode().split(":")[-1]
        last_heartbeat = await redis.get(key)
        workers.append({
            "worker_id": worker_id,
            "last_heartbeat": float(last_heartbeat) if last_heartbeat else None,
            "status": "healthy" if last_heartbeat else "dead",
        })

    # Queue stats
    queue_length = await redis.llen("arq:queue:default")

    return {
        "workers": workers,
        "queue_length": queue_length,
        "healthy_workers": sum(1 for w in workers if w["status"] == "healthy"),
    }
```

### 15.7 Effort Summary

| Phase | Description | Effort |
|-------|-------------|--------|
| Phase 6 | ARQ Infrastructure | 4 hours |
| Phase 7 | Migrate Document Processing | 6 hours |
| Phase 8 | Migrate Cache Warming | 6 hours |
| Phase 9 | Migrate Reindexing | 4 hours |
| Phase 10 | Jobs API & Cleanup | 4 hours |
| **Total** | **ARQ Migration** | **24 hours** |

### 15.8 Files to Delete After Migration

```
ai_ready_rag/services/warming_worker.py      # Replaced by workers/tasks/warming.py
ai_ready_rag/services/warming_queue.py       # File-based queue no longer needed
ai_ready_rag/services/warming_cleanup.py     # Cleanup handled by Redis TTL + cron
ai_ready_rag/services/reindex_worker.py      # Replaced by workers/tasks/reindex.py
ai_ready_rag/services/reindex_service.py     # Merged into reindex task
ai_ready_rag/services/sse_buffer_service.py  # Redis pub/sub replaces this
data/warming_queue/                          # File-based queue directory
```

### 15.9 Acceptance Criteria

- [ ] Redis running and accessible on localhost:6379
- [ ] ARQ worker starts successfully with `arq ai_ready_rag.workers.settings.WorkerSettings`
- [ ] Document upload returns job_id, processing happens in background
- [ ] Cache warming uses ARQ queue instead of custom worker
- [ ] Reindexing uses ARQ queue instead of custom worker
- [ ] `/api/jobs/{id}/status` returns progress information
- [ ] `/api/jobs/{id}/stream` delivers real-time SSE updates
- [ ] Jobs survive server restart (Redis persistence)
- [ ] Worker health check endpoint shows worker status
- [ ] All legacy warming/reindex code removed
- [ ] No `BackgroundTasks.add_task()` calls remain in codebase

---

## 16. Observability & Logging Strategy

### 16.1 Overview

A streamlined logging strategy focused on **actionable insights** for performance optimization and customer experience improvement. No DEBUG-level logging in production.

**Goals:**
- Track performance bottlenecks (LLM latency, vector search)
- Monitor customer experience (confidence scores, ROUTE rate)
- Enable troubleshooting (request tracing, job failures)

### 16.2 The 10 Essential Log Events

Log these 10 events well. Everything else is noise.

| # | Event | Level | Fields | Why It Matters |
|---|-------|-------|--------|----------------|
| 1 | **HTTP Request** | INFO | `request_id`, `method`, `path`, `status`, `latency_ms`, `user_id` | Trace any user complaint |
| 2 | **RAG Query Complete** | INFO | `request_id`, `latency_ms`, `llm_ms`, `vector_ms`, `cache_hit`, `confidence`, `action` | Performance + quality |
| 3 | **Job Started** | INFO | `job_id`, `type`, `document_id` | Track background work |
| 4 | **Job Completed** | INFO | `job_id`, `duration_ms`, `result_summary` | Success tracking |
| 5 | **Job Failed** | ERROR | `job_id`, `error`, `attempt`, `will_retry` | Find broken things |
| 6 | **Document Processed** | INFO | `doc_id`, `chunks`, `total_ms` | Ingestion performance |
| 7 | **Low Confidence Response** | WARN | `request_id`, `confidence`, `action=ROUTE` | Knowledge gaps |
| 8 | **Login Failed** | WARN | `email`, `reason`, `ip` | Security monitoring |
| 9 | **External Service Error** | ERROR | `service`, `error`, `retry_in` | Dependency health |
| 10 | **Cache Stats** (hourly) | INFO | `entries`, `hit_rate`, `avg_latency_ms` | Cache effectiveness |

### 16.3 Log Format

Simple JSON, one line per event. Grep-friendly.

```json
{"ts":"2026-02-05T14:30:45Z","level":"INFO","event":"rag_query_complete","request_id":"abc123","user_id":"usr-456","latency_ms":8500,"llm_ms":8200,"vector_ms":45,"cache_hit":false,"confidence":78,"action":"CITE"}
```

**No DEBUG level in production.** Turn it on temporarily if needed for specific troubleshooting.

### 16.4 Implementation

#### 16.4.1 Logging Library: structlog

```python
# ai_ready_rag/core/logging.py

import structlog
import logging

def configure_logging(log_level: str = "INFO") -> None:
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", key="ts"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
    )

def get_logger(name: str):
    return structlog.get_logger(name)
```

#### 16.4.2 Request Logging Middleware

```python
# ai_ready_rag/middleware/logging.py

import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        response = await call_next(request)

        latency_ms = int((time.perf_counter() - start) * 1000)

        # Log every request (Event #1)
        logger.info(
            "http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=latency_ms,
            user_id=getattr(request.state, "user_id", None),
        )

        response.headers["X-Request-ID"] = request_id
        return response
```

#### 16.4.3 RAG Pipeline Logging

```python
# In ai_ready_rag/services/rag_service.py

async def generate(self, request: RAGRequest, db: Session) -> RAGResponse:
    t_start = time.perf_counter()

    # Vector search
    t0 = time.perf_counter()
    chunks = await self.vector_service.search(...)
    vector_ms = int((time.perf_counter() - t0) * 1000)

    # Cache check
    cached = await self.cache_service.get(request.query)
    if cached:
        # Event #2 (cache hit)
        logger.info("rag_query_complete", cache_hit=True, ...)
        return cached

    # LLM call
    t0 = time.perf_counter()
    response = await self._call_llm(...)
    llm_ms = int((time.perf_counter() - t0) * 1000)

    # Event #2 (cache miss)
    logger.info(
        "rag_query_complete",
        request_id=request_id,
        latency_ms=int((time.perf_counter() - t_start) * 1000),
        llm_ms=llm_ms,
        vector_ms=vector_ms,
        cache_hit=False,
        confidence=response.confidence,
        action=response.action,
    )

    # Event #7 (low confidence)
    if response.confidence < 40:
        logger.warning(
            "low_confidence_response",
            request_id=request_id,
            confidence=response.confidence,
            action="ROUTE",
        )

    return response
```

#### 16.4.4 Background Job Logging

```python
# In ai_ready_rag/workers/tasks/document.py

async def process_document(ctx, document_id: str, options: dict) -> dict:
    job_id = ctx['job_id']

    # Event #3
    logger.info("job_started", job_id=job_id, type="process_document", document_id=document_id)

    try:
        t0 = time.perf_counter()
        # ... processing ...

        # Event #4
        logger.info(
            "job_completed",
            job_id=job_id,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            result_summary={"chunks": len(chunks)},
        )

        # Event #6
        logger.info("document_processed", doc_id=document_id, chunks=len(chunks), total_ms=...)

        return {"status": "success"}

    except Exception as e:
        # Event #5
        logger.error("job_failed", job_id=job_id, error=str(e), attempt=ctx.get("job_try", 1))
        raise
```

### 16.5 What NOT to Log

| Skip This | Why |
|-----------|-----|
| Request payload size | Not actionable |
| Client IP / User-Agent | Security theater for internal tool |
| Transaction commits | Too granular |
| Connection pool stats | Only at massive scale |
| Token refresh events | Noise |
| Chunk-by-chunk progress | Just log start/end |
| Worker heartbeats | Only for debugging |
| Semantic similarity scores | Niche optimization |

### 16.6 Configuration

```python
# ai_ready_rag/config.py
log_level: str = "INFO"  # INFO in production, DEBUG only for troubleshooting
log_file_path: str | None = None  # None = stdout only
log_query_text: bool = True  # Per Codex review: toggle for PII compliance
```

SQLite settings for runtime adjustment:

| Setting | Default | Description |
|---------|---------|-------------|
| `log_level` | "INFO" | Log level: INFO, DEBUG, WARN, ERROR |
| `log_query_text` | true | Log full query text in RAG events. **Set to false for PII compliance** if queries may contain sensitive data. When false, logs `query_length` instead of `query`. |

**Privacy note:** Full query logging is recommended for enterprise internal use (debugging value). Disable if:
- Queries may contain PII (employee names, SSNs)
- Regulatory requirements restrict data retention
- Logs are exported to external systems

#### 16.6.1 Debug Override Policy

DEBUG logging produces high-volume output and may expose sensitive data. The following operational policy governs its use:

1. **Who:** Only admin users can toggle `log_level` to DEBUG (via admin UI Security Panel)
2. **Time-boxed:** DEBUG mode auto-reverts to INFO after **1 hour**
   - The `admin_settings` table includes an `expires_at` column
   - `get_effective_setting()` ignores expired overrides and falls back to the config.py default
3. **Audited:** Every toggle is logged as an audit event:
   ```json
   {"event": "log_level_changed", "from": "INFO", "to": "DEBUG", "user_id": "usr-123", "expires_at": "2026-02-05T15:30:00Z"}
   ```
4. **Extension:** To extend beyond 1 hour, the admin must re-toggle (creates a fresh audit trail)
5. **Implementation:** Add `expires_at: DateTime | None` column to `admin_settings` table. In `get_effective_setting()`:
   ```python
   if override and override.expires_at and override.expires_at < datetime.utcnow():
       # Expired — ignore override, fall back to default
       return config_default
   ```

### 16.7 Querying Logs

```bash
# Find all logs for a specific request
grep "abc123" /var/log/ve-rag/app.log

# Find slow RAG queries (>10s)
grep "rag_query_complete" app.log | jq 'select(.latency_ms > 10000)'

# Find all job failures
grep "job_failed" app.log | jq '.'

# Find low confidence responses
grep "low_confidence_response" app.log | jq '.'

# Cache hit rate (last 100 queries)
grep "rag_query_complete" app.log | tail -100 | jq '.cache_hit' | sort | uniq -c
```

### 16.8 Implementation Plan

| Task | Description | Effort |
|------|-------------|--------|
| 11.1 | Install structlog, create `core/logging.py` | 1h |
| 11.2 | Add `RequestLoggingMiddleware` | 1h |
| 11.3 | Instrument RAG pipeline (Events #2, #7) | 2h |
| 11.4 | Instrument background jobs (Events #3-6) | 1h |
| 11.5 | Replace all `print()` with logger, test | 1h |
| **Total** | | **6h** |

### 16.9 Files to Create/Modify

| File | Action | Notes |
|------|--------|-------|
| `ai_ready_rag/core/logging.py` | CREATE | Logging config |
| `ai_ready_rag/middleware/logging.py` | CREATE | Request middleware |
| `ai_ready_rag/main.py` | MODIFY | Add middleware |
| `ai_ready_rag/services/rag_service.py` | MODIFY | Add Events #2, #7 |
| `ai_ready_rag/services/cache_service.py` | MODIFY | Add Event #10 |
| `ai_ready_rag/workers/tasks/*.py` | MODIFY | Add Events #3-6 |
| `requirements-*.txt` | MODIFY | Add structlog |

### 16.10 Acceptance Criteria

- [ ] All HTTP requests logged with request_id, latency, status
- [ ] RAG queries logged with timing breakdown and confidence
- [ ] Background jobs logged with start/complete/fail events
- [ ] Low confidence responses trigger WARN log
- [ ] All `print()` statements replaced with structured logging
- [ ] `grep {request_id}` returns full request trace
- [ ] No sensitive data (passwords, tokens, full query text) in logs
- [ ] `structlog` added to requirements files

---

## 17. Appendices

### Appendix A: File Change Summary

#### Phase 1: Critical Fixes
| File | Action | Notes |
|------|--------|-------|
| `ai_ready_rag/ui/*` | DELETE | Remove Gradio UI package |
| `app.py` | DELETE | Remove legacy Gradio entry point |
| `ai_ready_rag/main.py` | MODIFY | Remove Gradio mount |
| `ai_ready_rag/config.py` | MODIFY | Remove enable_gradio |
| `ai_ready_rag/api/health.py` | MODIFY | Remove gradio_enabled |
| `ai_ready_rag/db/models.py` | MODIFY | Add FK constraints |
| `ai_ready_rag/core/exceptions.py` | MODIFY | Add document exceptions |
| `ai_ready_rag/services/document_service.py` | MODIFY | Replace HTTPException with domain exceptions |
| `ai_ready_rag/core/error_handlers.py` | CREATE | Global exception → HTTP mapping |
| `ai_ready_rag/db/database.py` | MODIFY | Add PRAGMA foreign_keys=ON on connect |
| `ai_ready_rag/api/documents.py` | MODIFY | Remove try/except, rely on global handlers |
| `requirements-*.txt` | MODIFY | Remove gradio |

#### Phase 2: Architecture (Hybrid Layout)
| File | Action | Notes |
|------|--------|-------|
| `ai_ready_rag/schemas/__init__.py` | CREATE | Schema package init |
| `ai_ready_rag/schemas/common.py` | CREATE | Shared schemas (pagination, errors) |
| `ai_ready_rag/schemas/user.py` | CREATE | User schemas |
| `ai_ready_rag/schemas/document.py` | CREATE | Document schemas |
| `ai_ready_rag/schemas/chat.py` | CREATE | Chat schemas |
| `ai_ready_rag/schemas/tag.py` | CREATE | Tag schemas |
| `ai_ready_rag/schemas/auth.py` | CREATE | Auth schemas |
| `ai_ready_rag/schemas/admin.py` | CREATE | Admin schemas |
| `ai_ready_rag/db/mixins.py` | CREATE | TimestampMixin |
| `ai_ready_rag/db/models/__init__.py` | CREATE | Models package init |
| `ai_ready_rag/db/models/user.py` | CREATE | User model (split from models.py) |
| `ai_ready_rag/db/models/document.py` | CREATE | Document model |
| `ai_ready_rag/db/models/chat.py` | CREATE | ChatSession, ChatMessage models |
| `ai_ready_rag/db/models/tag.py` | CREATE | Tag model, associations |
| `ai_ready_rag/db/models/audit.py` | CREATE | AuditLog model |
| `ai_ready_rag/db/models/cache.py` | CREATE | CachedResponse, CuratedQA models |
| `ai_ready_rag/db/models/admin.py` | CREATE | AdminSetting model |
| `ai_ready_rag/db/models.py` | DELETE | Replaced by models/ package |
| `ai_ready_rag/db/repositories/__init__.py` | CREATE | Repository package init |
| `ai_ready_rag/db/repositories/base.py` | CREATE | BaseRepository[T] |
| `ai_ready_rag/db/repositories/user.py` | CREATE | UserRepository |
| `ai_ready_rag/db/repositories/document.py` | CREATE | DocumentRepository |
| `ai_ready_rag/db/repositories/tag.py` | CREATE | TagRepository |
| `ai_ready_rag/db/repositories/chat.py` | CREATE | ChatSessionRepository |
| `ai_ready_rag/db/repositories/audit.py` | CREATE | AuditLogRepository |
| `ai_ready_rag/db/repositories/cache.py` | CREATE | CacheRepository |
| `ai_ready_rag/services/base.py` | CREATE | BaseService |
| `ai_ready_rag/db/database.py` | MODIFY | Import mixins |
| `ai_ready_rag/workers/__init__.py` | CREATE | Workers package init |
| `ai_ready_rag/workers/warming_worker.py` | MOVE | From services/ |
| `ai_ready_rag/workers/cleanup_worker.py` | MOVE | From services/ |

#### Phase 3: Modernization (Reduced)
| File | Action | Notes |
|------|--------|-------|
| `ai_ready_rag/db/models/*.py` | MODIFY | Add FK indexes |
| `ai_ready_rag/services/rag_service.py` | MODIFY | Constructor injection |
| `ai_ready_rag/core/dependencies.py` | MODIFY | Add service Depends() factories |
| `ai_ready_rag/main.py` | MODIFY | Add lifespan for singleton services |

**Deferred:** alembic.ini, alembic/, Mapped[] syntax, container.py

#### Phase 4: Quality
| File | Action | Notes |
|------|--------|-------|
| `ai_ready_rag/services/cache_service.py` | MODIFY | Replace print with logger |
| `ai_ready_rag/workers/warming_worker.py` | MODIFY | Add semaphore |
| `tests/test_document_service.py` | CREATE | Service unit tests |
| `tests/test_repositories.py` | CREATE | Repository unit tests |
| `CLAUDE.md` | MODIFY | Update architecture docs |
| `README.md` | MODIFY | Update setup instructions |

#### Phase 5: Config Migration to SQLite (High-Priority Only)
| File | Action | Notes |
|------|--------|-------|
| `ai_ready_rag/services/settings_service.py` | MODIFY | Add get_effective_setting() |
| `scripts/migrate_config_to_sqlite.py` | CREATE | Migration script (8 settings) |

#### Phase 6: ARQ Infrastructure
| File | Action | Notes |
|------|--------|-------|
| `ai_ready_rag/workers/__init__.py` | CREATE | Workers package init |
| `ai_ready_rag/workers/settings.py` | CREATE | ARQ WorkerSettings |
| `ai_ready_rag/workers/tasks/__init__.py` | CREATE | Tasks subpackage init |
| `ai_ready_rag/workers/utils.py` | CREATE | Shared utilities |
| `ai_ready_rag/core/redis.py` | CREATE | Redis connection pool |
| `ai_ready_rag/config.py` | MODIFY | Add Redis/ARQ settings |
| `requirements-wsl.txt` | MODIFY | Add arq, redis |
| `requirements-spark.txt` | MODIFY | Add arq, redis |

#### Phase 7: Migrate Document Processing
| File | Action | Notes |
|------|--------|-------|
| `ai_ready_rag/workers/tasks/document.py` | CREATE | process_document task |
| `ai_ready_rag/api/documents.py` | MODIFY | Use ARQ enqueue |

#### Phase 8: Migrate Cache Warming
| File | Action | Notes |
|------|--------|-------|
| `ai_ready_rag/workers/tasks/warming.py` | CREATE | warm_cache task |
| `ai_ready_rag/api/admin.py` | MODIFY | Use ARQ for warming |
| `ai_ready_rag/services/warming_worker.py` | DELETE | Replaced by ARQ task |
| `ai_ready_rag/services/warming_queue.py` | DELETE | File queue no longer needed |
| `ai_ready_rag/services/warming_cleanup.py` | DELETE | Redis TTL handles cleanup |
| `ai_ready_rag/services/sse_buffer_service.py` | DELETE | Redis pub/sub replaces |

#### Phase 9: Migrate Reindexing
| File | Action | Notes |
|------|--------|-------|
| `ai_ready_rag/workers/tasks/reindex.py` | CREATE | reindex task |
| `ai_ready_rag/api/admin.py` | MODIFY | Use ARQ for reindex |
| `ai_ready_rag/services/reindex_worker.py` | DELETE | Replaced by ARQ task |
| `ai_ready_rag/services/reindex_service.py` | DELETE | Merged into task |

#### Phase 10: Jobs API & Cleanup
| File | Action | Notes |
|------|--------|-------|
| `ai_ready_rag/api/jobs.py` | CREATE | Job status/progress API |
| `ai_ready_rag/db/models/warming.py` | DELETE | Legacy warming tables |
| `ai_ready_rag/db/models/reindex.py` | DELETE | Legacy reindex tables |
| `data/warming_queue/` | DELETE | File-based queue directory |

#### Phase 11: Observability & Logging (Streamlined)
| File | Action | Notes |
|------|--------|-------|
| `ai_ready_rag/core/logging.py` | CREATE | Logging configuration (structlog) |
| `ai_ready_rag/middleware/logging.py` | CREATE | Request logging middleware |
| `ai_ready_rag/main.py` | MODIFY | Add logging middleware |
| `ai_ready_rag/services/rag_service.py` | MODIFY | Add 10 essential log events |
| `ai_ready_rag/services/cache_service.py` | MODIFY | Add cache stats logging |
| `ai_ready_rag/workers/tasks/*.py` | MODIFY | Add job start/complete/fail logs |
| `requirements-wsl.txt` | MODIFY | Add structlog |
| `requirements-spark.txt` | MODIFY | Add structlog |

#### Summary
| Action | Count |
|--------|-------|
| CREATE | 42 |
| MODIFY | 31 |
| DELETE | 12 |
| MOVE | 2 |
| **Total** | **87** |

### Appendix B: Reference Documents

- FastAPI Layered Pattern: `~/.claude/rules/fastapi-layered-pattern.md`
- Current Architecture: `docs/ARCHITECTURE.md`
- Scaffold Commands: `/scaffold-project`, `/scaffold-module`

### Appendix C: Agent Analysis Reports

The following analysis reports informed this specification:

1. **Backend API Structure Review** - Router patterns, auth, Gradio status
2. **Database Layer Review** - Models, FKs, migrations
3. **Services Layer Review** - Transaction management, exceptions
4. **React Frontend Review** - Component structure, stores
5. **RAG Pipeline Review** - Vector search, caching, confidence
6. **Background Workers Review** - Lifecycle, error handling
7. **Gradio Removal Checklist** - Complete removal inventory

### Appendix D: Background Task System Analysis

The following expert analyses informed Section 15 (Unified Background Task System):

1. **Chat Processing Flow Analysis**
   - Synchronous request-response (5-30s blocking)
   - No streaming or progress feedback
   - Timeout handling via ChatOllama (30s)

2. **Document Upload Processing Analysis**
   - FastAPI BackgroundTasks + Semaphore pattern
   - No persistent queue (lost on restart)
   - Status tracking via SQLite Document.status

3. **Cache Warming System Analysis**
   - AsyncIO worker with lease-based concurrency
   - File-based queue + DB queue (hybrid)
   - Checkpoint/resume capability
   - Critical gap: stuck job detection only at startup

4. **Reindexing Process Analysis**
   - FastAPI BackgroundTasks pattern
   - Polling-based pause/resume (2s interval)
   - No distributed support
   - Single active job enforcement

5. **Background Task Patterns Inventory**
   - 4 different patterns identified
   - Inconsistent state machines across features
   - No horizontal scaling capability
   - Recommendation: Unified ARQ + Redis architecture

**Key Findings:**
- 4 different background task patterns causing maintenance burden
- Tasks lost on server restart (no persistence)
- Polling-based progress (2-60s delays)
- No horizontal scaling support
- Poor error visibility and stuck job detection

**Recommendation:** Migrate to ARQ + Redis for unified, robust background processing.

---

## Document Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Engineering Lead | | | |
| Product Manager | | | |
| QA Lead | | | |
| Architecture | | | |

---

*End of Specification*
