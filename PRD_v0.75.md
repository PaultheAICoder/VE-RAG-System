# AI Ready RAG - Product Requirements Document

**Version:** 0.75
**Date:** January 27, 2026
**Status:** Draft - Working toward 1.0
**Target Deadline:** January 31, 2026 (Friday)

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
- **Roles:** Administrators (document uploaders) and End Users (query only)
- **Organizations:** Enterprises requiring data privacy and air-gapped deployments

---

## 2. Current State (v0.75)

### 2.1 Existing Features - Implemented ✅

| Feature | Technology | Status |
|---------|------------|--------|
| Document Parsing | Docling 2.68.0 | ✅ Working |
| OCR (Scanned PDFs) | Tesseract + EasyOCR | ✅ Working |
| Table Extraction | Docling TableStructure | ✅ Working |
| Semantic Chunking | HybridChunker | ✅ Working |
| Vector Storage | ChromaDB | ✅ Working |
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

### 2.3 Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User (Web Browser)                       │
│                    http://<ip>:8501                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Gradio Web UI (:8501)                     │
│  - Chat interface                                            │
│  - Document upload                                           │
│  - Processing options (OCR, table mode, language)           │
│  - Model selection                                           │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│     Docling      │ │    ChromaDB      │ │     Ollama       │
│  (Doc Parsing)   │ │  (Vector Store)  │ │    (:11434)      │
│  - OCR           │ │  - Embeddings    │ │  - qwen3:8b      │
│  - Tables        │ │  - Similarity    │ │  - nomic-embed   │
│  - Chunking      │ │    search        │ │                  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

### 2.4 Current Configuration

```bash
# Environment Variables
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
CHAT_MODEL=qwen3:8b
CHROMA_PERSIST_DIR=./chroma_db
UPLOAD_DIR=./uploads

# Ollama GPU Settings (via systemd)
OLLAMA_NUM_GPU=999
OLLAMA_FLASH_ATTENTION=1
CUDA_VISIBLE_DEVICES=0
OLLAMA_KEEP_ALIVE=5m
OLLAMA_MAX_LOADED_MODELS=4
```

### 2.5 Current Processing Pipeline

```
Document Upload
      │
      ▼
┌─────────────────┐
│ Format Detection│
│ (PDF/DOCX/etc)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Docling Convert │
│ - OCR if needed │
│ - Table extract │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ HybridChunker   │
│ - 512 tokens    │
│ - merge_peers   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ nomic-embed-text│
│ (768 dimensions)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ChromaDB Store  │
│ + Metadata      │
└─────────────────┘
```

### 2.6 Current Query Pipeline

```
User Question
      │
      ▼
┌─────────────────┐
│ Router Agent    │
│ RETRIEVE/DIRECT │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
RETRIEVE    DIRECT
    │         │
    ▼         │
┌──────────┐  │
│ Expand   │  │
│ Query    │  │
└────┬─────┘  │
     ▼        │
┌──────────┐  │
│ Vector   │  │
│ Search   │  │
│ (k=8)    │  │
└────┬─────┘  │
     ▼        │
┌──────────┐  │
│ RAG      │  │
│ Prompt   │  │
└────┬─────┘  │
     │        │
     └───┬────┘
         ▼
┌─────────────────┐
│ LLM Response    │
│ (qwen3:8b)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Evaluation      │
│ (1-10 score)    │
└─────────────────┘
```

---

## 3. Requirements - Planned Features

### 3.1 Authentication & Access Control

#### 3.1.1 Authentication Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| Azure AD (Entra ID) | OIDC/OAuth2 with Microsoft | Corporate environments with internet |
| Local Auth | Username/password in SQLite | Air-gapped deployments |

**Requirements:**
- [ ] Login page with branded UI
- [ ] Azure AD OIDC integration (optional, configurable)
- [ ] Local user registration and login
- [ ] Session management with JWT tokens
- [ ] Secure cookie handling
- [ ] Logout functionality
- [ ] Password reset (local auth)

#### 3.1.2 Role-Based Access Control (RBAC)

| Role | Permissions |
|------|-------------|
| Admin | Query, Upload, Manage Docs, Manage Users, Configure, View Logs |
| User | Query, View Own History |

**Requirements:**
- [ ] Role assignment per user
- [ ] Permission checking middleware
- [ ] Admin-only routes/pages
- [ ] Role displayed in UI

#### 3.1.3 Tag-Based Document Access

**Concept:** Documents are tagged; users are assigned tags; users only see documents matching their tags.

**Requirements:**
- [ ] Tag CRUD operations
- [ ] Assign tags to documents during upload
- [ ] Assign tags to users
- [ ] Filter vector search by user's tags
- [ ] "public" tag for documents everyone can see
- [ ] Tag owner assignment (for routing)

**Configurable Access Modes (per deployment):**
- Tag-based (default)
- Fully isolated (separate collections per department)
- Shared + Private (common docs + department-specific)
- Hierarchical (executives see all, managers see dept, etc.)

### 3.2 User Interface

#### 3.2.1 Design System

**Brand Identity (derived from aireadypdx.com):**
```
Primary Blue: #2563EB
Secondary Dark: #1E293B
Accent Green: #10B981
Background: #F8FAFC
Text: #334155

Typography: Inter (system-ui fallback)
Style: Clean, minimal, enterprise-grade
```

**Requirements:**
- [ ] Branded login page
- [ ] Consistent color scheme across all pages
- [ ] Professional typography
- [ ] Responsive design (desktop-first, mobile-friendly)
- [ ] Accessible (WCAG 2.1 AA)

#### 3.2.2 Page Structure

| Route | Page | Access |
|-------|------|--------|
| `/` | Login | Public |
| `/chat` | Main chat interface | Authenticated |
| `/history` | User's chat history | Authenticated |
| `/admin` | Admin dashboard | Admin only |
| `/admin/upload` | Document upload | Admin only |
| `/admin/documents` | Document management | Admin only |
| `/admin/users` | User management | Admin only |
| `/admin/tags` | Tag management | Admin only |
| `/admin/settings` | System settings | Admin only |
| `/admin/testing` | Quality testing | Admin only |

#### 3.2.3 Chat Interface

**Requirements:**
- [ ] Clean chat layout with message bubbles
- [ ] User/assistant message differentiation
- [ ] Citation display (clickable sources)
- [ ] Confidence score display
- [ ] Routing notification when applicable
- [ ] "New chat" button
- [ ] Chat history sidebar (recent conversations)

#### 3.2.4 Chat History

**Requirements:**
- [ ] Per-user chat persistence
- [ ] Session grouping (by conversation)
- [ ] Auto-generated session titles
- [ ] Search within history
- [ ] Continue past conversations
- [ ] Delete conversation option

### 3.3 Cite-or-Route System

#### 3.3.1 Citation Path

When system is confident and user has access:

**Requirements:**
- [ ] Extract source document and chunk info
- [ ] Format citations with document name and location
- [ ] Display confidence score
- [ ] Link to source (if accessible)

**Citation Format:**
```
Answer: [response text]

Sources:
1. [Document Name, Section/Page](link)
2. [Document Name, Section/Page](link)

Confidence: 92%
```

#### 3.3.2 Routing Path

When system cannot answer or user lacks access:

| Scenario | Action |
|----------|--------|
| No permission | Route to tag owner, explain access request process |
| Low confidence | Route to subject matter expert |
| No documents found | Offer to route to admin |
| Out of scope | Route to appropriate contact |

**Requirements:**
- [ ] Confidence threshold configuration
- [ ] Tag owner lookup for permission routing
- [ ] Friendly, non-punitive routing messages
- [ ] Log routed questions for follow-up
- [ ] Notification to routed-to person (future: email/webhook)

### 3.4 Quality Testing

#### 3.4.1 Test Categories

| Category | What It Tests |
|----------|---------------|
| OCR Accuracy | Text extraction from scanned documents |
| Retrieval Relevance | Correct chunks retrieved for queries |
| Answer Faithfulness | LLM response matches source context |
| Hallucination Detection | Responses not grounded in sources |

#### 3.4.2 Requirements

- [ ] Test document fixtures with known content
- [ ] Expected output files for comparison
- [ ] OCR quality scoring
- [ ] Retrieval relevance metrics
- [ ] Answer faithfulness evaluation
- [ ] Admin UI for running tests
- [ ] Automated test suite (pytest)

### 3.5 Infrastructure & Deployment

#### 3.5.1 Docker Architecture

```yaml
# docker-compose.yml structure
services:
  ai-ready-rag:
    build: .
    ports: ["8501:8501"]
    volumes: [./data:/app/data]
    depends_on: [qdrant, ollama]

  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]
    volumes: [./data/qdrant:/qdrant/storage]

  ollama:
    image: ollama/ollama
    ports: ["11434:11434"]
    volumes: [./data/ollama:/root/.ollama]
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

**Requirements:**
- [ ] Dockerfile for ai-ready-rag
- [ ] docker-compose.yml with all services
- [ ] Volume mounts for persistence
- [ ] GPU passthrough for Ollama
- [ ] Health check endpoints
- [ ] Logging configuration

#### 3.5.2 Directory Structure

```
/opt/ai-ready-rag/
├── docker-compose.yml
├── .env
├── data/
│   ├── qdrant/          # Vector database
│   ├── sqlite/          # Users, history, tags
│   ├── uploads/         # Uploaded documents
│   └── ollama/          # Model cache
├── config/
│   ├── auth.yml         # Auth configuration
│   └── access.yml       # Access control config
└── logs/
```

#### 3.5.3 USB Update System (Air-Gapped)

**Requirements:**
- [ ] Update package format (manifest, images, migrations)
- [ ] Checksum validation
- [ ] apply-update.sh script
- [ ] Database migration support
- [ ] Rollback capability (future)

#### 3.5.4 First-Time Setup

**Requirements:**
- [ ] setup.sh for initial deployment
- [ ] Create default admin user
- [ ] Initialize database schema
- [ ] Pull required Docker images
- [ ] Configure Ollama models
- [ ] Validate GPU access

### 3.6 Vector Database

#### 3.6.1 Migration from ChromaDB to Qdrant

**Rationale:**
- Better filtering for tag-based access
- GPU acceleration support
- Scales to production workloads
- Active development

**Requirements:**
- [ ] Qdrant container setup
- [ ] Vector store abstraction layer
- [ ] Qdrant adapter implementation
- [ ] Tag filtering in queries
- [ ] Migration script from ChromaDB
- [ ] Benchmark comparison

### 3.7 External Vectorization (Optional/Future)

**Concept:** Allow vectorization on a more powerful external system, then import to DGX Spark.

**Requirements (Future):**
- [ ] Export format specification
- [ ] Import mechanism
- [ ] Metadata preservation
- [ ] Incremental updates

---

## 4. Non-Functional Requirements

### 4.1 Performance

| Metric | Target |
|--------|--------|
| Query response time (P95) | <10 seconds |
| Document processing | <30 seconds per page |
| Concurrent users | 20+ simultaneous |
| Vector search latency | <500ms |

### 4.2 Security

- [ ] Passwords hashed with bcrypt
- [ ] JWT tokens with expiration
- [ ] HTTPS support (via reverse proxy)
- [ ] No sensitive data in logs
- [ ] SQL injection prevention
- [ ] XSS prevention
- [ ] CSRF protection

### 4.3 Reliability

- [ ] Graceful error handling
- [ ] Automatic service restart
- [ ] Health check endpoints
- [ ] Logging for debugging
- [ ] Data backup capability

### 4.4 Usability

- [ ] Intuitive navigation
- [ ] Clear error messages
- [ ] Loading indicators
- [ ] Keyboard shortcuts (future)
- [ ] Help documentation

---

## 5. Technical Specifications

### 5.1 Technology Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Frontend | Gradio | 6.3.0 |
| Backend | Python | 3.12 |
| Document Parsing | Docling | 2.68.0 |
| Vector DB | Qdrant (migrating from ChromaDB) | Latest |
| LLM Framework | LangChain | 0.3.x |
| LLM Runtime | Ollama | Latest |
| Chat Model | qwen3:8b | - |
| Embedding Model | nomic-embed-text | - |
| User Database | SQLite | 3.x |
| Containerization | Docker + Compose | Latest |

### 5.2 API Endpoints (Planned)

```
Authentication:
POST   /api/auth/login          # Login (local or Azure)
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
DELETE /api/documents/:id       # Delete document
POST   /api/documents/:id/tags  # Add tags

Tags (Admin):
GET    /api/tags                # List tags
POST   /api/tags                # Create tag
PUT    /api/tags/:id            # Update tag
DELETE /api/tags/:id            # Delete tag

Users (Admin):
GET    /api/users               # List users
POST   /api/users               # Create user (local auth)
PUT    /api/users/:id           # Update user
DELETE /api/users/:id           # Delete user
POST   /api/users/:id/tags      # Assign tags to user

System (Admin):
GET    /api/system/health       # Health check
GET    /api/system/stats        # System statistics
POST   /api/system/test         # Run quality tests
```

### 5.3 Database Schema

```sql
-- Users
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    display_name TEXT,
    role TEXT DEFAULT 'user',
    auth_provider TEXT NOT NULL,  -- 'azure_ad' or 'local'
    password_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Tags
CREATE TABLE tags (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    owner_email TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User-Tag assignments
CREATE TABLE user_tags (
    user_id TEXT REFERENCES users(id),
    tag_id TEXT REFERENCES tags(id),
    granted_by TEXT,
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, tag_id)
);

-- Document metadata
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    filepath TEXT,
    file_type TEXT,
    chunk_count INTEGER,
    uploaded_by TEXT REFERENCES users(id),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document-Tag assignments
CREATE TABLE document_tags (
    document_id TEXT REFERENCES documents(id),
    tag_id TEXT REFERENCES tags(id),
    PRIMARY KEY (document_id, tag_id)
);

-- Chat sessions
CREATE TABLE chat_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(id),
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

-- Chat messages
CREATE TABLE chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES chat_sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,  -- JSON
    routed_to TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Routed questions (for follow-up)
CREATE TABLE routed_questions (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES chat_sessions(id),
    question TEXT NOT NULL,
    reason TEXT,
    routed_to_email TEXT,
    status TEXT DEFAULT 'pending',  -- pending, resolved, expired
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);
```

---

## 6. Development Phases

### Phase 1: Core MVP (By Friday, Jan 31)

**Goal:** Working system with auth, chat, and basic access control

| Component | Tasks | Priority |
|-----------|-------|----------|
| Auth | Local auth, login page | P0 |
| UI | Branded chat page, basic styling | P0 |
| Chat | Persist history, show sources | P0 |
| Cite | Display citations in responses | P0 |
| Docker | Containerize application | P1 |

### Phase 2: Enterprise Features (Week 2)

| Component | Tasks | Priority |
|-----------|-------|----------|
| Auth | Azure AD integration | P1 |
| Access | Tag-based filtering | P1 |
| Vector | Qdrant migration | P1 |
| Admin | Upload page, user management | P1 |
| Route | Implement routing logic | P1 |

### Phase 3: Production Ready (Week 3+)

| Component | Tasks | Priority |
|-----------|-------|----------|
| Testing | Quality testing framework | P2 |
| Deploy | USB update system | P2 |
| Admin | Full admin dashboard | P2 |
| Polish | Mobile responsive, accessibility | P2 |

---

## 7. Open Questions

1. **Azure AD:** Do we have app registration details, or start with local auth only?
2. **Initial Tags:** What tag structure should we create by default?
3. **Notifications:** How to notify tag owners? (Email requires SMTP setup)
4. **Branding:** Can we get the actual logo file from aireadypdx.com?
5. **First Users:** Who are the initial test users and admins?

---

## 8. Success Criteria for v1.0

- [ ] Users can log in (local auth working)
- [ ] Users can query documents and get cited answers
- [ ] Chat history persists and is viewable
- [ ] Admins can upload documents (separate page)
- [ ] Documents can be tagged
- [ ] Users only see documents they have access to
- [ ] System routes to contact when it can't answer
- [ ] Deploys via Docker on fresh DGX Spark
- [ ] UI looks professional and branded
- [ ] Performance meets targets (<10s response)

---

## 9. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.75 | 2026-01-27 | Claude + Paul | Initial PRD consolidating existing features and new requirements |

---

*This document is the single source of truth for AI Ready RAG development.*
