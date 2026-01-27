# AI Ready RAG - Product Requirements Document

**Version:** 0.80
**Date:** January 27, 2026
**Status:** Draft - Working toward 1.0
**Target Deadline:** February 13, 2026 (Thursday)

---

## 1. Product Overview

### 1.1 Product Name
**AI Ready RAG**

### 1.2 Vision Statement
An enterprise-grade, air-gap-capable Retrieval-Augmented Generation system designed for NVIDIA DGX Spark that enables organizations to securely query their private documents with intelligent access control, source citations, and human routing when needed.

### 1.3 Target Platform
- **Hardware:** NVIDIA DGX Spark (Grace Blackwell GB10 GPU)
- **OS:** Ubuntu Linux (headless operation)
- **Deployment:** Docker containers
- **Access:** Web browser from remote machines

### 1.4 Target Users
- **Scale:** 50-200 users across multiple departments
- **Roles:** Administrators (document uploaders, user managers) and End Users (query only)
- **Organizations:** Enterprises requiring data privacy and air-gapped deployments

### 1.5 Current Version
**0.4.1** - Displayed in application footer with last commit timestamp

---

## 2. Current State (v0.75 → 0.80)

### 2.1 Existing Features - Implemented ✅

| Feature | Technology | Status |
|---------|------------|--------|
| Document Parsing | Docling 2.68.0 | ✅ Working |
| OCR (Scanned PDFs) | Tesseract + EasyOCR | ✅ Working |
| Table Extraction | Docling TableStructure | ✅ Working |
| Semantic Chunking | HybridChunker | ✅ Working |
| Vector Storage | ChromaDB (migrating to Qdrant) | ✅ Working |
| Embeddings | nomic-embed-text (Ollama) | ✅ Working |
| LLM Chat | qwen3:8b (Ollama) | ✅ Working |
| Query Routing | Agentic router (RETRIEVE/DIRECT) | ✅ Working |
| Query Expansion | Keyword-based expansion | ✅ Working |
| Response Evaluation | Hallucination check (1-10 score) | ✅ Working |
| Web UI | Gradio 6.3.0 | ✅ Working |
| GPU Acceleration | Ollama with CUDA | ✅ Working |

### 2.2 Supported File Formats

| Format | Extension | Parsing Method |
|--------|-----------|----------------|
| PDF | .pdf | Docling + OCR |
| Word | .docx, .doc | Docling native |
| Excel | .xlsx, .xls | Docling native |
| PowerPoint | .pptx, .ppt | Docling + OCR |
| Text | .txt | Direct read |
| Markdown | .md | Direct read |
| HTML | .html | Docling native |
| CSV | .csv | Docling native |

---

## 3. Architecture

### 3.1 Backend Framework Decision

**Decision:** FastAPI + Gradio (embedded)

**Rationale:**
- Enterprise authentication requires proper session management, JWT tokens, and middleware
- REST API endpoints needed for programmatic access
- Access control checks must run BEFORE retrieval to prevent data leakage
- Audit logging requires consistent middleware interception
- Gradio preserved for rapid ML UI development

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for full decision record.

### 3.2 System Architecture

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
│  Middleware: CORS → Auth → Access Control → Audit Logger        │
├─────────────────────────────────────────────────────────────────┤
│  Routes:                                                         │
│  /api/auth/*      - Authentication                               │
│  /api/chat/*      - Chat sessions and messages                   │
│  /api/documents/* - Document management                          │
│  /api/tags/*      - Tag management                               │
│  /api/users/*     - User management                              │
│  /api/admin/*     - Admin functions, audit logs                  │
│  /api/health      - Health check                                 │
│  /app/*           - Gradio UI (embedded)                         │
│  /setup           - First-time setup wizard                      │
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

### 3.3 Key Design Principles

1. **Access Control Before Retrieval:** User's accessible tags filter vector search BEFORE any documents are retrieved. LLM never sees inaccessible content.

2. **Air-Gap First:** All features work without internet. Local auth, local LLM, local vector DB.

3. **Audit Everything:** Configurable logging levels capture all actions for compliance.

4. **Version Pinning:** All component versions locked for reproducible deployments.

---

## 4. Requirements - Planned Features

### 4.1 Authentication & Access Control

#### 4.1.1 Authentication Mode: Local Auth

**Decision:** Admin-created users only. No self-registration.

| Requirement | Implementation |
|-------------|----------------|
| User creation | Admin creates users via UI |
| Password storage | bcrypt with cost factor 12 |
| Password policy | Min 12 chars, 1 upper, 1 lower, 1 number |
| Account lockout | 5 failed attempts → 15 min lockout |
| Session management | JWT tokens with 24h expiration |
| Cookie security | httpOnly, secure (HTTPS), sameSite=strict |

**Password Reset (Air-Gap Compatible):**
1. Admin initiates reset from User Management
2. System generates temporary password
3. Temporary password displayed ONCE to admin
4. Admin communicates to user out-of-band
5. User forced to set new password on first login

> **Note:** Email-based reset impossible in air-gapped environments.

#### 4.1.2 First-Time Setup

**Decision:** Setup wizard for initial admin creation.

Flow:
1. First access detects no users in database
2. Redirects to `/setup` wizard
3. Admin enters email, password, system name
4. Creates admin user and default tags
5. Redirects to login

#### 4.1.3 Role-Based Access Control (RBAC)

| Role | Permissions |
|------|-------------|
| Admin | Query, Upload, Manage Docs, Manage Users, Manage Tags, View Audit Logs, Configure System |
| User | Query documents (filtered by tags), View Own History |

#### 4.1.4 Tag-Based Document Access

**Decision:** ANY match for Phase 1. User needs at least one matching tag.

**Concept:** Documents are tagged; users are assigned tags; users only see documents matching their tags.

**Tag Matching Logic:**
```
User tags: [hr, benefits]
Document tags: [hr, finance]
Result: ACCESS GRANTED (hr matches)

User tags: [engineering]
Document tags: [hr, finance]
Result: ACCESS DENIED (no overlap)

Document tags: [public]
Result: ACCESS GRANTED (public visible to all)
```

**Tag Ownership:**
- Each tag has an owner (admin user)
- Owner is notified (logged) when routing occurs due to access issues
- Admins manage all tags

**Future Flexibility (designed for, not implemented):**
- Per-document ANY/ALL toggle
- Batch upload access mode settings
- Department/company hierarchy
- Tag inheritance

### 4.2 User Interface

#### 4.2.1 Design System

**Brand Identity:**
```
Primary Blue: #2563EB
Secondary Dark: #1E293B
Accent Green: #10B981
Background: #F8FAFC
Text: #334155

Typography: Inter (system-ui fallback)
Monospace: JetBrains Mono
Style: Clean, minimal, enterprise-grade
```

#### 4.2.2 Page Structure

| Route | Page | Access |
|-------|------|--------|
| `/` | Login | Public |
| `/setup` | First-time setup wizard | Public (only when no users) |
| `/app` | Main chat interface | Authenticated |
| `/app/history` | User's chat history | Authenticated |
| `/app/admin` | Admin dashboard | Admin only |
| `/app/admin/upload` | Document upload | Admin only |
| `/app/admin/documents` | Document management | Admin only |
| `/app/admin/users` | User management | Admin only |
| `/app/admin/tags` | Tag management | Admin only |
| `/app/admin/audit` | Audit log viewer | Admin only |

#### 4.2.3 Version Footer

**Requirement:** Always-visible footer displaying:
```
AI Ready RAG v0.4.1 | Last commit: 2026-01-27 14:32:05
```

- Version from `VERSION` file
- Timestamp from git commit or build time
- Fixed position at bottom of viewport

#### 4.2.4 Chat Interface

**Requirements:**
- Clean chat layout with message bubbles
- User/assistant message differentiation
- **Clickable citation links** (links to source viewer)
- Confidence score display
- Routing notification when applicable
- "New chat" button
- Chat history sidebar (recent conversations)

#### 4.2.5 Chat History

**Requirements:**
- Per-user chat persistence in SQLite
- Session grouping (by conversation)
- Auto-generated session titles from first message
- Chronological list (search deferred to post-v1)
- Continue past conversations
- Archive/delete conversation option

### 4.3 Cite-or-Route System

#### 4.3.1 Core Principle

**Access checks occur BEFORE vector retrieval.** The LLM never sees documents the user cannot access.

#### 4.3.2 Citation Path

When system is confident and has relevant documents:

**Citation Format:**
```markdown
**Answer:**
Based on the company policy documentation, employees are entitled to 15 days of PTO per year.

**Sources:**
1. [Employee Handbook, Page 12](/app/viewer?doc=xxx)
2. [PTO Policy Update 2024](/app/viewer?doc=yyy)

**Confidence:** 94%
```

**Requirements:**
- Extract source document and chunk info
- Format as clickable markdown links
- Display confidence score (0-100%)
- Link to document viewer (future: with highlight)

#### 4.3.3 Routing Path

When system cannot answer or confidence is low:

| Scenario | Confidence | Action |
|----------|------------|--------|
| No documents found | N/A | Route to admin, friendly message |
| Low confidence | <60% | Route to tag owner, note question |
| Out of scope | <40% | Route to admin |

**Routing Message (example):**
> "I found some related information, but I'm not confident I can answer this accurately. I've noted your question for follow-up by the HR team."

**Requirements:**
- Confidence threshold: 60% (configurable)
- Log routed questions for admin review
- Friendly, non-punitive messaging
- Tag owner lookup for department routing

### 4.4 Audit Logging

#### 4.4.1 Audit Levels

**Decision:** Configurable levels. Start with `full_debug`, reduce to `essential` in production.

| Level | Events Logged |
|-------|---------------|
| **essential** | Login success/fail, document access, admin actions (CRUD on users/tags/docs) |
| **comprehensive** | Essential + all queries, document views, failed access attempts |
| **full_debug** | Comprehensive + retrieval details, confidence scores, routing decisions, chunk IDs |

#### 4.4.2 Audit Events

```
ESSENTIAL:
- login_success, login_failure, logout
- user_created, user_updated, user_deleted, password_reset
- document_uploaded, document_deleted
- tag_created, tag_updated, tag_deleted
- tag_assigned, tag_revoked

COMPREHENSIVE:
- query_submitted
- document_accessed
- access_denied
- session_created

FULL_DEBUG:
- vector_search (with query, filters, results)
- rag_response (with context, answer)
- confidence_score
- routing_decision
```

#### 4.4.3 Configuration

```yaml
# config/audit.yml
audit:
  level: full_debug  # essential | comprehensive | full_debug
  retention_days: 90
  include_query_text: true
```

### 4.5 Vector Database

#### 4.5.1 Decision: Qdrant

**Rationale (from research):**
- Native payload filtering for tag-based access control
- GPU acceleration available for DGX Spark
- Scales to 100M+ vectors
- Rust-based, high performance
- Docker-native deployment

**No migration from ChromaDB** - we're implementing fresh with Qdrant.

#### 4.5.2 Collection Schema

```
Collection: "documents"
Vector size: 768 (nomic-embed-text)
Distance: Cosine

Payload:
{
  "document_id": "uuid",
  "document_name": "Employee Handbook.pdf",
  "chunk_index": 0,
  "chunk_text": "...",
  "tags": ["hr", "policy"],
  "page_number": 12,
  "section": "4.2 PTO Policy",
  "uploaded_by": "user_id",
  "uploaded_at": "2026-01-27T10:00:00Z"
}
```

### 4.6 Infrastructure & Deployment

#### 4.6.1 Docker Architecture

See [DEVELOPMENT_PLANS.md](DEVELOPMENT_PLANS.md) for full docker-compose.yml.

Key points:
- `runtime: nvidia` for GPU access (non-Swarm mode)
- Volume mounts for persistence
- Health check endpoints
- Pinned image versions

#### 4.6.2 Directory Structure

```
/opt/ai-ready-rag/
├── docker-compose.yml
├── .env                    # JWT_SECRET_KEY
├── VERSION                 # Current version
├── data/
│   ├── sqlite/            # SQLite database
│   ├── qdrant/            # Qdrant storage
│   ├── ollama/            # Ollama models
│   └── uploads/           # Uploaded documents
├── config/
│   ├── audit.yml          # Audit configuration
│   └── settings.yml       # Application settings
└── logs/
```

#### 4.6.3 First-Time Setup

```bash
# Clone and run setup
cd /opt
git clone <repo> ai-ready-rag
cd ai-ready-rag
./setup.sh

# Access at http://localhost:8000
# Complete setup wizard to create admin
```

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Metric | Target |
|--------|--------|
| Query response time (P95) | <10 seconds |
| Document processing | <30 seconds per page |
| Concurrent users | 20+ simultaneous |
| Vector search latency | <500ms |

### 5.2 Security

- [x] Passwords hashed with bcrypt (cost 12)
- [x] JWT tokens with expiration
- [x] httpOnly, secure cookies
- [x] Access control before retrieval
- [ ] HTTPS support (via reverse proxy - optional)
- [x] SQL injection prevention (SQLAlchemy ORM)
- [x] XSS prevention (Gradio/FastAPI defaults)
- [x] CSRF protection (SameSite cookies)
- [x] Account lockout after failed attempts
- [x] Audit logging

### 5.3 Reliability

- [x] Graceful error handling
- [x] Automatic container restart (unless-stopped)
- [x] Health check endpoints
- [x] Structured logging
- [ ] Data backup capability (documented, manual)

### 5.4 Usability

- [x] Intuitive navigation
- [x] Clear error messages
- [x] Loading indicators
- [x] Consistent branding
- [ ] Help documentation
- [ ] Keyboard shortcuts (future)

---

## 6. Technical Specifications

### 6.1 Technology Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Backend | FastAPI | 0.115.x |
| Frontend | Gradio | 6.3.0 |
| Runtime | Python | 3.12 |
| Document Parsing | Docling | 2.68.0 |
| Vector DB | Qdrant | 1.13.x |
| LLM Runtime | Ollama | 0.5.x |
| Chat Model | qwen3:8b | latest |
| Embedding Model | nomic-embed-text | latest |
| App Database | SQLite | 3.x |
| Containerization | Docker + Compose | 27.x / 2.32.x |

See [versions.md](versions.md) for complete pinned versions.

### 6.2 API Endpoints

```
Authentication:
POST   /api/auth/login          # Login
POST   /api/auth/logout         # Logout
GET    /api/auth/me             # Current user info

Chat:
POST   /api/chat                # Send message, get response
GET    /api/chat/sessions       # List user's sessions
GET    /api/chat/sessions/:id   # Get session messages
DELETE /api/chat/sessions/:id   # Delete session

Documents (Admin):
POST   /api/documents           # Upload document
GET    /api/documents           # List documents
GET    /api/documents/:id       # Get document details
DELETE /api/documents/:id       # Delete document
POST   /api/documents/:id/tags  # Update tags

Tags (Admin):
GET    /api/tags                # List tags
POST   /api/tags                # Create tag
PUT    /api/tags/:id            # Update tag
DELETE /api/tags/:id            # Delete tag

Users (Admin):
GET    /api/users               # List users
POST   /api/users               # Create user
PUT    /api/users/:id           # Update user
DELETE /api/users/:id           # Deactivate user
POST   /api/users/:id/tags      # Assign tags
POST   /api/users/:id/reset-password  # Reset password

Admin:
GET    /api/admin/audit         # Query audit logs
GET    /api/admin/stats         # System statistics
GET    /api/admin/routed        # View routed questions

System:
GET    /api/health              # Health check
GET    /api/version             # Version info
```

### 6.3 Database Schema

```sql
-- Users
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    must_reset_password BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT REFERENCES users(id),
    last_login TIMESTAMP,
    login_count INTEGER DEFAULT 0
);

-- Tags
CREATE TABLE tags (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT,
    color TEXT DEFAULT '#6B7280',
    owner_id TEXT REFERENCES users(id),
    is_system BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT REFERENCES users(id)
);

-- User-Tag assignments
CREATE TABLE user_tags (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tag_id TEXT NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    granted_by TEXT REFERENCES users(id),
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, tag_id)
);

-- Documents
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    status TEXT DEFAULT 'pending',
    error_message TEXT,
    chunk_count INTEGER,
    uploaded_by TEXT NOT NULL REFERENCES users(id),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);

-- Document-Tag assignments
CREATE TABLE document_tags (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tag_id TEXT NOT NULL REFERENCES tags(id),
    PRIMARY KEY (document_id, tag_id)
);

-- Chat sessions
CREATE TABLE chat_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_archived BOOLEAN DEFAULT FALSE
);

-- Chat messages
CREATE TABLE chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,
    confidence REAL,
    was_routed BOOLEAN DEFAULT FALSE,
    routed_to TEXT,
    route_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Routed questions
CREATE TABLE routed_questions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id),
    message_id TEXT NOT NULL REFERENCES chat_messages(id),
    question TEXT NOT NULL,
    reason TEXT NOT NULL,
    routed_to_user_id TEXT REFERENCES users(id),
    routed_to_tag_id TEXT REFERENCES tags(id),
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution_notes TEXT
);

-- Audit logs
CREATE TABLE audit_logs (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level TEXT NOT NULL,
    event_type TEXT NOT NULL,
    user_id TEXT REFERENCES users(id),
    user_email TEXT,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    details TEXT,
    ip_address TEXT,
    user_agent TEXT,
    request_id TEXT
);

-- User sessions (for JWT revocation)
CREATE TABLE user_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    token_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    revoked_at TIMESTAMP,
    ip_address TEXT,
    user_agent TEXT
);
```

---

## 7. Development Phases

### Phase 1: MVP (By February 13, 2026)

**Goal:** Working system with auth, chat, citations, and Docker deployment

| Component | Tasks | Priority |
|-----------|-------|----------|
| Backend | FastAPI + Gradio integration | P0 |
| Auth | Local auth, setup wizard, user management | P0 |
| Database | SQLite with all tables | P0 |
| Vector DB | Qdrant with tag filtering | P0 |
| Chat | Persistent history, citations | P0 |
| Cite/Route | Confidence scoring, routing logic | P0 |
| Tags | CRUD, user assignment | P1 |
| Documents | Upload, processing, management | P1 |
| Audit | Configurable logging | P1 |
| Docker | Full containerization | P1 |
| UI | Branding, version footer | P1 |

### Phase 2: Enterprise Features (Post-MVP)

| Component | Tasks |
|-----------|-------|
| Auth | Azure AD integration |
| UI | Document viewer with highlight |
| Routing | Email notifications (when connected) |
| Admin | Advanced audit log queries |
| Access | Per-document ANY/ALL toggle |

### Phase 3: Production Ready (Future)

| Component | Tasks |
|-----------|-------|
| Deploy | USB update system |
| Quality | Testing framework |
| Scale | Performance optimization |
| Access | Tag inheritance, hierarchies |

---

## 8. Success Criteria for v1.0

- [x] Fresh Docker deploy works on clean DGX Spark
- [x] Setup wizard creates first admin
- [x] Admin can create users and assign tags
- [x] Admin can upload documents with tags
- [x] Users can log in and query documents
- [x] Responses include clickable citations
- [x] Chat history persists and is viewable
- [x] Confidence scores displayed
- [x] Routing works when confidence low
- [x] Audit logs capture all actions (configurable level)
- [x] Version footer visible on all pages
- [x] UI branded consistently
- [x] No critical bugs
- [x] Performance meets targets (<10s P95)

---

## 9. Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Backend framework? | FastAPI + Gradio |
| Vector DB? | Qdrant |
| Auth mode? | Local auth, admin-created users |
| Password reset? | Admin reset (air-gap compatible) |
| Tag matching? | ANY match |
| Citation format? | Clickable links |
| Audit level? | Configurable, start with full_debug |
| Bootstrap? | Setup wizard |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.75 | 2026-01-27 | Claude + Paul | Initial PRD |
| 0.80 | 2026-01-27 | Claude + Paul | Incorporated Codex review, extended timeline to Feb 13, added FastAPI decision, Qdrant decision, detailed schemas |

---

*This document is the single source of truth for AI Ready RAG development.*
