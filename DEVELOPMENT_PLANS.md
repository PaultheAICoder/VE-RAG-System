# AI Ready RAG - Development Plans

**Product Name:** AI Ready RAG
**Version:** 0.4.1
**Target Deadline:** Thursday, February 13, 2026
**Platform:** NVIDIA DGX Spark (Ubuntu, headless)
**Last Updated:** January 27, 2026

---

## Executive Summary

This document outlines the development plans for AI Ready RAG, incorporating feedback from the Codex review and stakeholder decisions. The timeline has been extended to February 13, 2026, providing 17 days for development.

### Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backend Framework | **FastAPI + Gradio** | Enterprise auth, REST API, middleware for access control |
| Vector Database | **Qdrant** | Superior tag filtering, GPU acceleration, production scale |
| Application Database | **SQLite** | Zero infrastructure, air-gap friendly, sufficient scale |
| Access Control | **Pre-retrieval filtering** | Prevents data leakage via citations |
| Authentication | **Local auth (admin-created)** | Air-gap compatible, admin password reset |
| Audit Logging | **3-level configurable** | Full debug during dev, essential in production |
| Citations | **Clickable links** | Links to source document viewer |
| Tag Matching | **ANY match** | User needs any matching tag (flexible for future) |
| Admin Bootstrap | **Setup wizard** | First-time web UI for creating initial admin |

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed rationale.

---

## Timeline Overview

```
Week 1: Jan 27 - Feb 2   │ Foundation (FastAPI, Auth, Database)
Week 2: Feb 3 - Feb 9    │ Core Features (Chat, Citations, History)
Week 3: Feb 10 - Feb 13  │ Polish & Testing (UI, Docker, QA)
```

### Milestone Checkpoints

| Date | Milestone | Deliverables |
|------|-----------|--------------|
| **Jan 31 (Fri)** | Foundation Complete | FastAPI running, SQLite schema, local auth working |
| **Feb 5 (Wed)** | Core Features | Chat with citations, history persistence, Qdrant integrated |
| **Feb 10 (Mon)** | Feature Complete | All Phase 1 features implemented, bug fixing begins |
| **Feb 13 (Thu)** | Release Ready | Docker deployment tested, documentation complete |

---

## Plan 1: Backend Architecture (FastAPI + Gradio)

### Objective
Replace Gradio standalone with FastAPI backend, mounting Gradio as embedded UI.

### Why FastAPI + Gradio?
> **Decision from Codex Review:** Gradio standalone lacks proper session management, middleware support, and REST API capabilities required for enterprise authentication and audit logging. FastAPI provides these while preserving our Gradio UI investment.

### Application Structure

```
ai_ready_rag/
├── main.py                 # FastAPI app entry point
├── config.py               # Configuration management
├── version.py              # Version info for footer
├── api/
│   ├── __init__.py
│   ├── auth.py             # /api/auth/* routes
│   ├── chat.py             # /api/chat/* routes
│   ├── documents.py        # /api/documents/* routes
│   ├── tags.py             # /api/tags/* routes
│   ├── users.py            # /api/users/* routes
│   ├── admin.py            # /api/admin/* routes
│   └── health.py           # /api/health endpoint
├── core/
│   ├── __init__.py
│   ├── security.py         # JWT, password hashing
│   ├── dependencies.py     # FastAPI dependencies
│   └── exceptions.py       # Custom exceptions
├── middleware/
│   ├── __init__.py
│   ├── auth.py             # Authentication middleware
│   ├── audit.py            # Audit logging middleware
│   └── access_control.py   # Pre-retrieval access checks
├── db/
│   ├── __init__.py
│   ├── database.py         # SQLite connection, WAL mode
│   ├── models.py           # SQLAlchemy models
│   └── migrations/         # Schema migrations
├── services/
│   ├── __init__.py
│   ├── auth_service.py     # Login, password reset
│   ├── chat_service.py     # Chat session management
│   ├── document_service.py # Document processing
│   ├── vector_service.py   # Qdrant operations
│   ├── rag_service.py      # RAG pipeline
│   └── audit_service.py    # Audit log writing
├── ui/
│   ├── __init__.py
│   ├── gradio_app.py       # Gradio Blocks definition
│   ├── components/         # Reusable UI components
│   └── setup_wizard.py     # First-time setup UI
└── utils/
    ├── __init__.py
    └── helpers.py
```

### Tasks

| Priority | Task | Effort | Dependencies |
|----------|------|--------|--------------|
| P0 | Create FastAPI project structure | 2h | None |
| P0 | Implement configuration management (.env, config.py) | 2h | Structure |
| P0 | Set up SQLite with WAL mode | 2h | Config |
| P0 | Create database models (SQLAlchemy) | 3h | SQLite |
| P0 | Implement JWT token handling | 3h | Models |
| P0 | Mount Gradio as sub-application | 2h | FastAPI |
| P1 | Create middleware stack (CORS, Auth, Audit) | 4h | JWT |
| P1 | Implement health check endpoint | 1h | FastAPI |
| P1 | Add version footer to UI | 1h | Gradio mount |

**Estimated Total: 20 hours**

### Technical Specifications

**JWT Configuration:**
```python
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")  # Generated on first run
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
JWT_REFRESH_ENABLED = True
JWT_REFRESH_EXPIRATION_DAYS = 7
```

**SQLite WAL Configuration:**
```python
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-64000;  # 64MB
PRAGMA busy_timeout=5000;  # 5 second timeout
```

---

## Plan 2: Authentication & User Management

### Objective
Implement secure local authentication with admin-managed users and password reset.

### Authentication Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     First-Time Setup                             │
│  (No users exist in database)                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Setup Wizard (/setup)                         │
│  1. Welcome screen                                               │
│  2. Create admin account (email, password)                       │
│  3. Configure system name                                        │
│  4. Complete → Redirect to login                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Normal Operation                              │
│  - Login page at /                                               │
│  - JWT token issued on success                                   │
│  - Token stored in httpOnly cookie                               │
│  - All /app/* and /api/* routes require valid token             │
└─────────────────────────────────────────────────────────────────┘
```

### User Management (Admin Only)

> **Decision:** Admin-created users only. No self-registration for security in enterprise deployments.

**Admin capabilities:**
- Create new users (email, display name, role, initial password)
- Reset user passwords (generates temporary password)
- Assign/revoke tags to users
- Deactivate/reactivate users
- View user activity (from audit logs)

### Password Reset Flow (Air-Gap Compatible)

> **Codex Review Finding:** No email available in air-gapped environments.

```
Admin Password Reset Flow:
1. Admin navigates to User Management
2. Selects user → "Reset Password"
3. System generates temporary password
4. Temporary password displayed ONCE to admin
5. Admin communicates to user out-of-band
6. User logs in with temporary password
7. User forced to set new password on first login
```

### Database Schema

```sql
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',  -- 'admin' or 'user'
    is_active BOOLEAN DEFAULT TRUE,
    must_reset_password BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT REFERENCES users(id),
    last_login TIMESTAMP,
    login_count INTEGER DEFAULT 0
);

CREATE TABLE user_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    token_hash TEXT NOT NULL,  -- Hash of JWT for revocation
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    revoked_at TIMESTAMP,
    ip_address TEXT,
    user_agent TEXT
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_sessions_user ON user_sessions(user_id);
CREATE INDEX idx_sessions_expires ON user_sessions(expires_at);
```

### Security Requirements

| Requirement | Implementation |
|-------------|----------------|
| Password hashing | bcrypt with cost factor 12 |
| Password policy | Min 12 chars, 1 upper, 1 lower, 1 number |
| Account lockout | 5 failed attempts → 15 min lockout |
| Session timeout | 24 hours, configurable |
| Secure cookies | httpOnly, secure (when HTTPS), sameSite=strict |

### Tasks

| Priority | Task | Effort | Dependencies |
|----------|------|--------|--------------|
| P0 | Create users table and model | 2h | Database |
| P0 | Implement password hashing with bcrypt | 1h | Users model |
| P0 | Create login API endpoint | 3h | Password hashing |
| P0 | Implement JWT token issuance | 2h | Login API |
| P0 | Create auth middleware | 3h | JWT |
| P0 | Build setup wizard UI | 4h | Auth middleware |
| P0 | Build login page UI | 3h | Setup wizard |
| P1 | Implement account lockout | 2h | Login API |
| P1 | Create user management API (CRUD) | 4h | Auth middleware |
| P1 | Build user management UI (admin) | 4h | User API |
| P1 | Implement password reset flow | 3h | User management |
| P2 | Add password change on first login | 2h | Password reset |

**Estimated Total: 33 hours**

---

## Plan 3: Vector Database (Qdrant)

### Objective
Implement Qdrant as the vector database with tag-based filtering for access control.

### Why Qdrant Over ChromaDB?

> **Decision from earlier research:**
> - Native payload filtering ideal for tag-based access
> - Rust-based, high performance
> - GPU acceleration available
> - Scales to 100M+ vectors
> - Better production characteristics

### Vector Store Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VectorStore Abstraction                       │
│                    (services/vector_service.py)                  │
├─────────────────────────────────────────────────────────────────┤
│  Methods:                                                        │
│  - add_documents(docs, tags, metadata)                          │
│  - search(query, user_tags, limit) → filtered results           │
│  - delete_document(doc_id) → removes all chunks                 │
│  - get_collection_stats() → counts, size                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QdrantVectorStore                             │
│                    (implements VectorStore)                      │
├─────────────────────────────────────────────────────────────────┤
│  Collection: "documents"                                         │
│  Vector size: 768 (nomic-embed-text)                            │
│  Distance: Cosine                                                │
│                                                                  │
│  Payload schema:                                                 │
│  {                                                               │
│    "document_id": "uuid",                                       │
│    "document_name": "Employee Handbook.pdf",                    │
│    "chunk_index": 0,                                            │
│    "chunk_text": "...",                                         │
│    "tags": ["hr", "policy"],                                    │
│    "page_number": 12,                                           │
│    "section": "4.2 PTO Policy",                                 │
│    "uploaded_by": "user_id",                                    │
│    "uploaded_at": "2026-01-27T10:00:00Z"                       │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

### Access-Controlled Search

> **Critical from Codex Review:** Access checks BEFORE retrieval to prevent data leakage.

```python
def search_with_access_control(
    query: str,
    user_tags: List[str],
    limit: int = 8
) -> List[SearchResult]:
    """
    Search vectors filtered by user's accessible tags.
    User NEVER sees documents they don't have access to.
    """
    # Build filter: tags overlap with user_tags OR contains "public"
    tag_filter = models.Filter(
        should=[
            models.FieldCondition(
                key="tags",
                match=models.MatchAny(any=user_tags)
            ),
            models.FieldCondition(
                key="tags",
                match=models.MatchValue(value="public")
            )
        ]
    )

    # Search with filter - only accessible docs returned
    results = qdrant_client.search(
        collection_name="documents",
        query_vector=embed(query),
        query_filter=tag_filter,
        limit=limit
    )

    return results
```

### Tasks

| Priority | Task | Effort | Dependencies |
|----------|------|--------|--------------|
| P0 | Set up Qdrant container (docker-compose) | 2h | None |
| P0 | Create VectorStore abstraction interface | 2h | None |
| P0 | Implement QdrantVectorStore | 4h | Abstraction |
| P0 | Implement tag-filtered search | 3h | QdrantVectorStore |
| P1 | Create document indexing pipeline | 3h | Vector store |
| P1 | Implement document deletion (with chunks) | 2h | Pipeline |
| P1 | Add collection statistics endpoint | 1h | Vector store |
| P2 | Performance benchmarking | 2h | All above |

**Estimated Total: 19 hours**

---

## Plan 4: Document Management & Processing

### Objective
Enable admin document upload with tagging, Docling processing, and Qdrant indexing.

### Upload Flow

```
Admin Upload
      │
      ▼
┌─────────────────┐
│ 1. File Upload  │ Max size: 100MB
│    Validation   │ Allowed types: PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, CSV
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Tag Assignment│ Admin selects tags from existing list
│    (required)   │ At least one tag required
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Store File   │ /data/uploads/{doc_id}/{filename}
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 4. Background Processing (async)        │
│    a. Docling parse (OCR if needed)     │
│    b. HybridChunker (512 tokens)        │
│    c. Generate embeddings               │
│    d. Index to Qdrant with tags         │
│    e. Update document status            │
└─────────────────────────────────────────┘
```

### Document Database Schema

```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    status TEXT DEFAULT 'pending',  -- pending, processing, ready, failed
    error_message TEXT,
    chunk_count INTEGER,
    uploaded_by TEXT NOT NULL REFERENCES users(id),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);

CREATE TABLE document_tags (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tag_id TEXT NOT NULL REFERENCES tags(id),
    PRIMARY KEY (document_id, tag_id)
);

CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_uploaded_by ON documents(uploaded_by);
```

### Processing Options

```python
class ProcessingOptions:
    enable_ocr: bool = True
    ocr_languages: List[str] = ["eng"]  # Tesseract language codes
    table_mode: str = "accurate"  # fast, accurate
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 50  # tokens
```

### Tasks

| Priority | Task | Effort | Dependencies |
|----------|------|--------|--------------|
| P0 | Create documents table and model | 2h | Database |
| P0 | Implement file upload API | 3h | Documents model |
| P0 | Create background processing task | 4h | Upload API |
| P0 | Integrate Docling processing | 3h | Background task |
| P1 | Build admin upload UI | 4h | Upload API |
| P1 | Show processing status/progress | 2h | Upload UI |
| P1 | Implement document deletion (file + vectors) | 3h | Vector store |
| P1 | Build document list UI (admin) | 3h | Documents API |
| P2 | Add document search/filter | 2h | Document list |

**Estimated Total: 26 hours**

---

## Plan 5: Tag Management

### Objective
Implement tag CRUD with ownership for routing and access management.

### Tag Model

> **Decision:** Admins own tags. Tags are used for both document access control and routing (when access denied, notify tag owner).

```sql
CREATE TABLE tags (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT,
    color TEXT DEFAULT '#6B7280',  -- For UI display
    owner_id TEXT REFERENCES users(id),  -- Admin who owns this tag
    is_system BOOLEAN DEFAULT FALSE,  -- System tags (e.g., "public")
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT REFERENCES users(id)
);

-- User tag assignments
CREATE TABLE user_tags (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tag_id TEXT NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    granted_by TEXT REFERENCES users(id),
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, tag_id)
);

CREATE INDEX idx_tags_owner ON tags(owner_id);
CREATE INDEX idx_user_tags_user ON user_tags(user_id);
```

### Default System Tags

Created during setup wizard:

| Tag | Description |
|-----|-------------|
| `public` | Documents visible to all authenticated users |
| `admin` | Admin-only documents |

### Tag Matching Logic

> **Decision:** ANY match for Phase 1. User with tag "hr" can access documents tagged ["hr", "finance"].

```python
def user_can_access_document(user_tags: List[str], doc_tags: List[str]) -> bool:
    """
    ANY match: user needs at least one matching tag.
    Public documents accessible to all.
    """
    if "public" in doc_tags:
        return True
    return bool(set(user_tags) & set(doc_tags))
```

### Future Flexibility

Design for future enhancements:
- Per-document ANY/ALL toggle
- Batch upload access mode settings
- Department/company hierarchy
- Tag inheritance

### Tasks

| Priority | Task | Effort | Dependencies |
|----------|------|--------|--------------|
| P0 | Create tags table and model | 2h | Database |
| P0 | Implement tag CRUD API | 3h | Tags model |
| P0 | Create user_tags junction table | 1h | Tags model |
| P1 | Build tag management UI (admin) | 4h | Tag API |
| P1 | Add tag assignment to user management | 2h | User management |
| P1 | Add tag selection to document upload | 2h | Document upload |
| P2 | Tag color picker in UI | 1h | Tag management |

**Estimated Total: 15 hours**

---

## Plan 6: Chat Interface & History

### Objective
Implement persistent chat with session management and history browsing.

### Chat Data Model

```sql
CREATE TABLE chat_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    title TEXT,  -- Auto-generated from first message
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_archived BOOLEAN DEFAULT FALSE
);

CREATE TABLE chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    sources TEXT,  -- JSON array of citations
    confidence REAL,  -- 0.0 to 1.0
    was_routed BOOLEAN DEFAULT FALSE,
    routed_to TEXT,  -- Tag owner email if routed
    route_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_user ON chat_sessions(user_id);
CREATE INDEX idx_sessions_updated ON chat_sessions(updated_at DESC);
CREATE INDEX idx_messages_session ON chat_messages(session_id);
```

### Chat UI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  [Logo] AI Ready RAG                    [User Menu ▼] [Logout]  │
├─────────────────┬───────────────────────────────────────────────┤
│                 │                                               │
│  History        │   Chat Area                                   │
│  ───────────    │   ──────────                                  │
│  [+ New Chat]   │                                               │
│                 │   ┌─────────────────────────────────────────┐ │
│  Today          │   │ User: What is the PTO policy?          │ │
│  └─ PTO Policy  │   └─────────────────────────────────────────┘ │
│  └─ Benefits    │                                               │
│                 │   ┌─────────────────────────────────────────┐ │
│  Yesterday      │   │ Assistant: Based on the Employee       │ │
│  └─ Q3 Budget   │   │ Handbook, employees receive 15 days... │ │
│                 │   │                                         │ │
│  Last 7 days    │   │ Sources:                                │ │
│  └─ ...         │   │ 📄 Employee Handbook, Page 12          │ │
│                 │   │ 📄 PTO Policy Update 2024              │ │
│                 │   │                                         │ │
│                 │   │ Confidence: 94%                         │ │
│                 │   └─────────────────────────────────────────┘ │
│                 │                                               │
│                 │   ┌─────────────────────────────────────────┐ │
│                 │   │ Type your question...            [Send] │ │
│                 │   └─────────────────────────────────────────┘ │
├─────────────────┴───────────────────────────────────────────────┤
│  AI Ready RAG v0.4.1 | Last commit: 2026-01-27 14:32:05        │
└─────────────────────────────────────────────────────────────────┘
```

### Session Title Generation

Auto-generate from first user message:

```python
def generate_session_title(first_message: str) -> str:
    """Generate concise title from first message."""
    # Truncate to first 50 chars, clean up
    title = first_message[:50].strip()
    if len(first_message) > 50:
        title = title.rsplit(' ', 1)[0] + "..."
    return title
```

### Tasks

| Priority | Task | Effort | Dependencies |
|----------|------|--------|--------------|
| P0 | Create chat_sessions and chat_messages tables | 2h | Database |
| P0 | Implement chat API (create session, send message) | 4h | Tables |
| P0 | Integrate RAG pipeline with chat API | 4h | Chat API, Vector store |
| P0 | Build main chat UI in Gradio | 4h | Chat API |
| P1 | Implement history sidebar component | 3h | Chat UI |
| P1 | Add session title auto-generation | 1h | Chat API |
| P1 | Enable continuing past conversations | 2h | History sidebar |
| P2 | Add chat search functionality | 2h | History |
| P2 | Implement session archiving/deletion | 2h | History |

**Estimated Total: 24 hours**

---

## Plan 7: Cite-or-Route System

### Objective
Implement intelligent response routing with clickable source citations.

### Response Flow

> **Critical from Codex Review:** Access checks happen BEFORE retrieval. LLM never sees inaccessible documents.

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Get User Tags from Session                                    │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Vector Search with Tag Filter                                 │
│    (Only accessible documents searched)                          │
└─────────────────────────────────────────────────────────────────┘
      │
      ├─── No results found ───────────────────┐
      │                                         │
      ▼                                         ▼
┌─────────────────┐                    ┌─────────────────┐
│ 3a. Results     │                    │ 3b. Route       │
│     Found       │                    │     (no docs)   │
└────────┬────────┘                    └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Generate RAG Response with Sources                            │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Evaluate Confidence (1-10 scale)                              │
└─────────────────────────────────────────────────────────────────┘
         │
         ├─── Confidence < 6 ──────────────────┐
         │                                      │
         ▼                                      ▼
┌─────────────────┐                    ┌─────────────────┐
│ 6a. CITE        │                    │ 6b. ROUTE       │
│ Return answer   │                    │ Low confidence  │
│ with citations  │                    │ or no answer    │
└─────────────────┘                    └─────────────────┘
```

### Citation Format

> **Decision:** Clickable links that open source viewer.

```python
class Citation:
    document_id: str
    document_name: str
    page_number: Optional[int]
    section: Optional[str]
    chunk_text: str  # For preview
    relevance_score: float

def format_citations(citations: List[Citation]) -> str:
    """Format citations as clickable markdown links."""
    lines = ["**Sources:**"]
    for i, c in enumerate(citations, 1):
        location = f"Page {c.page_number}" if c.page_number else c.section or ""
        # Link to document viewer with highlight
        link = f"/app/viewer?doc={c.document_id}&highlight={c.chunk_text[:50]}"
        lines.append(f"{i}. [{c.document_name}, {location}]({link})")
    return "\n".join(lines)
```

### Routing Logic

```python
CONFIDENCE_THRESHOLD = 6  # Out of 10

class RouteReason(Enum):
    NO_DOCUMENTS = "no_documents"
    LOW_CONFIDENCE = "low_confidence"
    OUT_OF_SCOPE = "out_of_scope"

def should_route(
    search_results: List[SearchResult],
    confidence_score: float
) -> Tuple[bool, Optional[RouteReason]]:
    """Determine if question should be routed to human."""
    if not search_results:
        return True, RouteReason.NO_DOCUMENTS
    if confidence_score < CONFIDENCE_THRESHOLD:
        return True, RouteReason.LOW_CONFIDENCE
    return False, None

def get_route_message(reason: RouteReason, tag_owners: List[str]) -> str:
    """Generate friendly routing message."""
    messages = {
        RouteReason.NO_DOCUMENTS: (
            "I don't have information about this topic in the documents "
            "I can access. Would you like me to connect you with someone "
            "who might be able to help?"
        ),
        RouteReason.LOW_CONFIDENCE: (
            "I found some related information, but I'm not confident I can "
            "answer this accurately. I've noted your question for follow-up "
            f"by {', '.join(tag_owners) or 'an administrator'}."
        ),
    }
    return messages.get(reason, "I'll route this question for you.")
```

### Routed Questions Table

```sql
CREATE TABLE routed_questions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id),
    message_id TEXT NOT NULL REFERENCES chat_messages(id),
    question TEXT NOT NULL,
    reason TEXT NOT NULL,
    routed_to_user_id TEXT REFERENCES users(id),
    routed_to_tag_id TEXT REFERENCES tags(id),
    status TEXT DEFAULT 'pending',  -- pending, acknowledged, resolved
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution_notes TEXT
);

CREATE INDEX idx_routed_status ON routed_questions(status);
CREATE INDEX idx_routed_to_user ON routed_questions(routed_to_user_id);
```

### Tasks

| Priority | Task | Effort | Dependencies |
|----------|------|--------|--------------|
| P0 | Implement confidence scoring | 3h | RAG pipeline |
| P0 | Create citation extraction and formatting | 3h | RAG pipeline |
| P0 | Build routing decision logic | 2h | Confidence scoring |
| P1 | Create routed_questions table | 1h | Database |
| P1 | Implement routing message generation | 2h | Routing logic |
| P1 | Display citations in chat UI | 3h | Chat UI |
| P1 | Show confidence score in UI | 1h | Chat UI |
| P2 | Admin view of routed questions | 3h | Routed questions |
| P2 | Document viewer with highlight | 4h | Citations |

**Estimated Total: 22 hours**

---

## Plan 8: Audit Logging

### Objective
Implement configurable audit logging for compliance and debugging.

### Audit Levels

> **Decision:** Three configurable levels. Start with Full Debug during development.

| Level | Events Logged |
|-------|---------------|
| **essential** | Login success/fail, document access, admin actions |
| **comprehensive** | Essential + all queries, document views, access denials |
| **full_debug** | Comprehensive + retrieval details, confidence scores, chunk IDs |

### Audit Schema

```sql
CREATE TABLE audit_logs (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level TEXT NOT NULL,  -- essential, comprehensive, full_debug
    event_type TEXT NOT NULL,
    user_id TEXT REFERENCES users(id),
    user_email TEXT,
    action TEXT NOT NULL,
    resource_type TEXT,  -- user, document, tag, session, message
    resource_id TEXT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    details TEXT,  -- JSON payload with event-specific data
    ip_address TEXT,
    user_agent TEXT,
    request_id TEXT  -- For correlating related events
);

CREATE INDEX idx_audit_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX idx_audit_user ON audit_logs(user_id);
CREATE INDEX idx_audit_event ON audit_logs(event_type);
CREATE INDEX idx_audit_resource ON audit_logs(resource_type, resource_id);
```

### Event Types

```python
class AuditEvent(Enum):
    # Essential events (always logged)
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    PASSWORD_RESET = "password_reset"
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_DELETED = "document_deleted"
    TAG_CREATED = "tag_created"
    TAG_UPDATED = "tag_updated"
    TAG_DELETED = "tag_deleted"
    TAG_ASSIGNED = "tag_assigned"
    TAG_REVOKED = "tag_revoked"

    # Comprehensive events
    QUERY_SUBMITTED = "query_submitted"
    DOCUMENT_ACCESSED = "document_accessed"
    ACCESS_DENIED = "access_denied"
    SESSION_CREATED = "session_created"

    # Full debug events
    VECTOR_SEARCH = "vector_search"
    RAG_RESPONSE = "rag_response"
    CONFIDENCE_SCORE = "confidence_score"
    ROUTING_DECISION = "routing_decision"
```

### Middleware Implementation

```python
class AuditLogMiddleware:
    def __init__(self, app, audit_level: str = "full_debug"):
        self.app = app
        self.audit_level = audit_level

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request_id = str(uuid.uuid4())
            # Add request_id to scope for correlation
            scope["request_id"] = request_id

            # Log based on configured level
            await self.log_request(scope, request_id)

        await self.app(scope, receive, send)
```

### Tasks

| Priority | Task | Effort | Dependencies |
|----------|------|--------|--------------|
| P0 | Create audit_logs table | 1h | Database |
| P0 | Implement AuditService | 3h | Table |
| P0 | Create AuditLogMiddleware | 3h | AuditService |
| P1 | Add audit logging to all API endpoints | 4h | Middleware |
| P1 | Implement audit level configuration | 2h | AuditService |
| P2 | Build audit log viewer (admin) | 4h | Audit logging |
| P2 | Add log export functionality | 2h | Viewer |

**Estimated Total: 19 hours**

---

## Plan 9: Infrastructure & Docker

### Objective
Containerize the application with GPU support for DGX Spark deployment.

### Docker Architecture

```yaml
# docker-compose.yml
version: "3.8"

services:
  ai-ready-rag:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ai-ready-rag
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///data/sqlite/ai_ready_rag.db
      - QDRANT_URL=http://qdrant:6333
      - OLLAMA_URL=http://ollama:11434
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - AUDIT_LEVEL=full_debug
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    depends_on:
      - qdrant
      - ollama
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  qdrant:
    image: qdrant/qdrant:v1.13.2
    container_name: qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant:/qdrant/storage
    restart: unless-stopped

  ollama:
    image: ollama/ollama:0.5.7
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ./data/ollama:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    # For non-Swarm mode, also add:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    restart: unless-stopped

volumes:
  data:
```

> **Codex Review Fix:** Added `runtime: nvidia` for non-Swarm Docker Compose GPU support.

### Dockerfile

```dockerfile
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY ai_ready_rag/ ./ai_ready_rag/
COPY VERSION .

# Create data directories
RUN mkdir -p /app/data/sqlite /app/data/uploads /app/logs

EXPOSE 8000

CMD ["uvicorn", "ai_ready_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Directory Structure (Deployed)

```
/opt/ai-ready-rag/
├── docker-compose.yml
├── .env                      # Secrets (JWT_SECRET_KEY)
├── VERSION                   # Current version
├── data/
│   ├── sqlite/              # SQLite database
│   │   └── ai_ready_rag.db
│   ├── qdrant/              # Qdrant storage
│   ├── ollama/              # Ollama models
│   └── uploads/             # Uploaded documents
├── config/
│   ├── audit.yml            # Audit configuration
│   └── settings.yml         # Application settings
└── logs/                    # Application logs
```

### First-Time Setup Script

```bash
#!/bin/bash
# setup.sh - First-time deployment script

set -e

echo "AI Ready RAG Setup"
echo "=================="

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker required"; exit 1; }
command -v docker compose >/dev/null 2>&1 || { echo "Docker Compose required"; exit 1; }

# Check NVIDIA runtime
if ! docker info | grep -q "nvidia"; then
    echo "WARNING: NVIDIA runtime not detected. GPU acceleration may not work."
fi

# Create directories
mkdir -p data/{sqlite,qdrant,ollama,uploads} config logs

# Generate JWT secret if not exists
if [ ! -f .env ]; then
    echo "Generating JWT secret..."
    JWT_SECRET=$(openssl rand -hex 32)
    echo "JWT_SECRET_KEY=$JWT_SECRET" > .env
fi

# Pull images
echo "Pulling Docker images..."
docker compose pull

# Start services
echo "Starting services..."
docker compose up -d

# Wait for services
echo "Waiting for services to be ready..."
sleep 10

# Pull Ollama models
echo "Pulling Ollama models (this may take a while)..."
docker exec ollama ollama pull qwen3:8b
docker exec ollama ollama pull nomic-embed-text

# Health check
echo "Checking health..."
curl -s http://localhost:8000/api/health || echo "Service starting..."

echo ""
echo "Setup complete!"
echo "Access the application at: http://localhost:8000"
echo "Complete the setup wizard to create your admin account."
```

### Tasks

| Priority | Task | Effort | Dependencies |
|----------|------|--------|--------------|
| P0 | Create Dockerfile | 2h | Application code |
| P0 | Create docker-compose.yml | 2h | Dockerfile |
| P0 | Test GPU passthrough | 2h | Docker compose |
| P1 | Create setup.sh script | 2h | Docker compose |
| P1 | Implement health check endpoint | 1h | FastAPI |
| P1 | Create .env.example with documentation | 1h | Setup script |
| P2 | Create USB update package format | 3h | Docker |
| P2 | Implement apply-update.sh | 3h | Update package |

**Estimated Total: 16 hours**

---

## Plan 10: UI Polish & Branding

### Objective
Apply consistent branding and polish to all UI components.

### Brand Style Guide

```css
/* AI Ready RAG Brand Tokens */
:root {
  /* Colors */
  --color-primary: #2563EB;
  --color-primary-dark: #1D4ED8;
  --color-secondary: #1E293B;
  --color-accent: #10B981;
  --color-background: #F8FAFC;
  --color-surface: #FFFFFF;
  --color-text: #334155;
  --color-text-muted: #64748B;
  --color-border: #E2E8F0;
  --color-error: #EF4444;
  --color-warning: #F59E0B;
  --color-success: #10B981;

  /* Typography */
  --font-family: 'Inter', system-ui, -apple-system, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;

  /* Spacing */
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;

  /* Borders */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
}
```

### Version Footer Component

```python
def create_version_footer():
    """Create footer with version and commit timestamp."""
    version_string = get_version_string()  # "AI Ready RAG v0.4.1 | Last commit: 2026-01-27 14:32:05"

    return gr.HTML(
        f"""
        <footer style="
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 8px 16px;
            background: var(--color-secondary);
            color: white;
            font-size: 12px;
            text-align: center;
            font-family: var(--font-mono);
        ">
            {version_string}
        </footer>
        """
    )
```

### Tasks

| Priority | Task | Effort | Dependencies |
|----------|------|--------|--------------|
| P0 | Implement version footer | 1h | Version module |
| P1 | Apply brand colors to Gradio theme | 2h | Chat UI |
| P1 | Style login page | 2h | Login UI |
| P1 | Style setup wizard | 2h | Setup UI |
| P1 | Add logo to header | 1h | Brand assets |
| P2 | Style admin pages | 3h | Admin UI |
| P2 | Add loading states/spinners | 2h | All UI |
| P2 | Accessibility review (WCAG 2.1 AA) | 3h | All UI |

**Estimated Total: 16 hours**

---

## Timeline Summary

### Total Estimated Hours: ~210 hours

| Plan | Hours | Week 1 | Week 2 | Week 3 |
|------|-------|--------|--------|--------|
| 1. Backend (FastAPI) | 20h | ████████ | | |
| 2. Authentication | 33h | ████████████ | ████ | |
| 3. Vector DB (Qdrant) | 19h | ████████ | ██ | |
| 4. Documents | 26h | | ████████████ | |
| 5. Tags | 15h | | ████████ | |
| 6. Chat & History | 24h | | ████████████ | |
| 7. Cite-or-Route | 22h | | ████ | ████████ |
| 8. Audit Logging | 19h | ██ | ████████ | |
| 9. Docker | 16h | ████ | | ████████ |
| 10. UI Polish | 16h | | | ████████████ |

### Week-by-Week Breakdown

**Week 1 (Jan 27 - Feb 2): Foundation**
- FastAPI project structure
- SQLite database with models
- JWT authentication
- Login page & setup wizard
- Qdrant container setup
- Basic Docker configuration

**Week 2 (Feb 3 - Feb 9): Core Features**
- Document upload & processing
- Tag management
- Chat UI with history
- Citation display
- Qdrant integration complete
- Audit logging framework

**Week 3 (Feb 10 - Feb 13): Polish & QA**
- Cite-or-route refinement
- UI branding complete
- Docker deployment tested
- Bug fixes
- Documentation
- Final QA

---

## Risk Mitigation

### High-Risk Items

| Risk | Mitigation |
|------|------------|
| Gradio + FastAPI integration issues | Prototype mount pattern early (Day 1) |
| Qdrant GPU config on DGX | Test container GPU access first |
| Document processing timeouts | Implement async/background processing |
| Auth middleware complexity | Follow FastAPI best practices, test thoroughly |

### Fallback Options

If timeline slips:
1. **Defer document viewer** - Show citations as text links (no highlight)
2. **Defer audit admin UI** - Log to files, build UI later
3. **Simplify history** - No search, just chronological list
4. **Skip mobile polish** - Desktop-only for v0.4.1

---

## Definition of Done (Feb 13)

- [ ] Fresh Docker deployment works on clean DGX Spark
- [ ] Setup wizard creates first admin
- [ ] Admin can create users and assign tags
- [ ] Admin can upload documents with tags
- [ ] Users can log in and query documents
- [ ] Responses include clickable citations
- [ ] Chat history persists and is viewable
- [ ] Confidence scores displayed
- [ ] Routing works when confidence low
- [ ] Audit logs capture all actions
- [ ] Version footer visible on all pages
- [ ] UI branded consistently
- [ ] No critical bugs

---

## Open Items (Deferred Post-v0.4.1)

- Azure AD integration
- Email notifications for routing
- Mobile-responsive design
- USB update system
- Quality testing framework
- Document viewer with highlighting
- Tag inheritance/hierarchy
- Per-document ANY/ALL access mode

---

## Next Steps

1. **Today (Jan 27):** Set up FastAPI project structure, create database models
2. **Tomorrow:** Implement JWT auth, create login API
3. **Day 3:** Setup wizard, login UI, Qdrant container
4. **Daily standups:** Review progress against timeline

---

*Document maintained as single source of truth for development. Update as decisions change.*
