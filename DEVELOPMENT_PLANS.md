# AI Ready RAG - Development Plans

**Product Name:** AI Ready RAG
**Target Deadline:** Friday, January 31, 2026
**Platform:** NVIDIA DGX Spark (Ubuntu, headless)

---

## Executive Summary

Based on our requirements gathering, here are the parallel development plans organized by topic. Each plan is independent and can be worked on concurrently.

### Key Requirements Captured

| Requirement | Decision |
|-------------|----------|
| Authentication | Azure AD (Entra ID) OR local auth for air-gapped |
| User Scale | 50-200 users, multi-department |
| Roles | Admin, User, with department-based access |
| Access Control | Tag-based (primary), configurable per customer |
| Network | Customer-configurable (Azure AD or pure air-gap) |
| Storage | Local disk on DGX Spark |
| Deployment | Docker containers + automation scripts |
| Updates | USB-based for air-gapped environments |
| Branding | Derive from aireadypdx.com, modern UI framework |

---

## Plan 1: UI/UX & Frontend Design

### Objective
Transform the current basic Gradio interface into a polished, professional web application.

### Brand Style Guide (Derived from aireadypdx.com)

```
BRAND: AI Ready RAG

PRIMARY COLORS:
- Primary Blue: #2563EB (trust, technology)
- Secondary Dark: #1E293B (professional, serious)
- Accent Green: #10B981 (success, action)
- Background: #F8FAFC (clean, light)
- Text: #334155 (readable, soft black)

TYPOGRAPHY:
- Headings: Inter (or system-ui), Bold
- Body: Inter (or system-ui), Regular
- Monospace: JetBrains Mono (for code/citations)

DESIGN PRINCIPLES:
- Clean, minimal interface
- Clear hierarchy
- Accessible (WCAG 2.1 AA)
- Mobile-responsive
- Trust-inspiring for enterprise use

LOGO:
- Use aireadypdx.com logo-nav.webp
- White version for dark headers
- Minimum padding: 16px
```

### UI Framework Decision

**Recommended: Gradio 6 with Custom CSS + Tailwind**

Rationale:
- Gradio provides rapid development for ML apps
- Custom CSS/Tailwind for branding
- Avoids complete rewrite (time constraint)
- Production-ready with proper styling

Alternative for future: **FastAPI + React/Next.js**

### Page Structure

```
/                   → Login page
/chat               → Main chat interface (home after login)
/history            → User's chat history
/admin              → Admin dashboard (upload, manage docs)
/admin/upload       → Document upload page (admin only)
/admin/users        → User management
/admin/settings     → System settings
```

### Tasks

| Priority | Task | Effort |
|----------|------|--------|
| P0 | Create login page with Azure AD / local auth | 4h |
| P0 | Redesign main chat page with brand styling | 4h |
| P1 | Create chat history sidebar/page | 3h |
| P1 | Separate admin upload page | 2h |
| P1 | Add navigation header with user menu | 2h |
| P2 | Admin dashboard with stats | 3h |
| P2 | User management page | 4h |
| P3 | Mobile responsive adjustments | 2h |

**Estimated Total: 24 hours**

---

## Plan 2: Authentication & RBAC

### Objective
Implement secure authentication with role-based access control supporting both Azure AD and local auth.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Authentication Layer                      │
├─────────────────┬───────────────────────────────────────────┤
│   Azure AD      │            Local Auth                      │
│   (OIDC/OAuth2) │      (Username/Password + SQLite)         │
├─────────────────┴───────────────────────────────────────────┤
│                    Session Management                        │
│                  (JWT tokens, secure cookies)                │
├─────────────────────────────────────────────────────────────┤
│                    RBAC Engine                               │
│         Roles: admin, user (extensible)                     │
│         Permissions: upload, query, manage_users, etc.      │
├─────────────────────────────────────────────────────────────┤
│                    Tag-Based Access Control                  │
│         Documents tagged → Users assigned to tags           │
│         Query filter: only return docs user can access      │
└─────────────────────────────────────────────────────────────┘
```

### Authentication Modes

**Mode 1: Azure AD (Entra ID)**
- OIDC/OAuth2 flow
- Requires: Client ID, Tenant ID, Client Secret
- Minimal internet access (only to login.microsoftonline.com)
- Groups synced to local roles

**Mode 2: Local Auth (Air-Gapped)**
- Username/password stored in local SQLite
- Passwords hashed with bcrypt
- Admin creates users manually
- No external dependencies

### RBAC Implementation

```python
ROLES = {
    "admin": {
        "permissions": ["query", "upload", "manage_docs", "manage_users", "view_logs", "configure"],
        "description": "Full system access"
    },
    "user": {
        "permissions": ["query", "view_history"],
        "description": "Query documents and view own history"
    }
}

TAG_ACCESS = {
    "user_id": ["tag1", "tag2", "tag3"],  # User can access docs with these tags
    # Special tags:
    # "*" = access all documents
    # "public" = documents everyone can see
}
```

### Database Schema (SQLite for portability)

```sql
-- Users table
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE,
    display_name TEXT,
    role TEXT DEFAULT 'user',
    auth_provider TEXT, -- 'azure_ad' or 'local'
    password_hash TEXT, -- NULL for Azure AD users
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- User tag access
CREATE TABLE user_tags (
    user_id TEXT,
    tag TEXT,
    granted_by TEXT,
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Document tags
CREATE TABLE document_tags (
    document_id TEXT,
    tag TEXT,
    PRIMARY KEY (document_id, tag)
);

-- Chat history
CREATE TABLE chat_history (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    question TEXT,
    answer TEXT,
    sources TEXT, -- JSON array of cited sources
    routed_to TEXT, -- NULL if answered, else contact info
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### Tasks

| Priority | Task | Effort |
|----------|------|--------|
| P0 | Create auth module with provider abstraction | 3h |
| P0 | Implement Azure AD OIDC flow | 4h |
| P0 | Implement local auth (register/login) | 3h |
| P0 | Session management with JWT | 2h |
| P1 | RBAC permission checking middleware | 2h |
| P1 | Tag-based document filtering | 3h |
| P1 | User management API endpoints | 3h |
| P2 | Group sync from Azure AD | 2h |

**Estimated Total: 22 hours**

---

## Plan 3: Vector Database Evaluation

### Objective
Evaluate alternatives to ChromaDB for production use.

### Current State: ChromaDB
- Simple, embedded, works well for prototyping
- **Limitation:** Not designed for 50M+ vectors
- **Limitation:** Limited filtering capabilities
- **Limitation:** Single-node only

### Evaluation Matrix

| Database | Scale | Air-Gap OK | GPU Accel | Filtering | Effort to Migrate |
|----------|-------|------------|-----------|-----------|-------------------|
| **ChromaDB** | <10M | ✅ | ❌ | Basic | N/A (current) |
| **Qdrant** | 100M+ | ✅ | ✅ | Excellent | Medium |
| **Milvus** | Billions | ✅ | ✅ | Excellent | High |
| **pgvector** | 10M+ | ✅ | ❌ | SQL-based | Medium |

### Recommendation: **Qdrant**

**Rationale:**
1. **Open source & self-hosted** - perfect for air-gapped
2. **Rust-based** - high performance, low memory
3. **GPU acceleration available** - leverages DGX Spark
4. **Excellent filtering** - supports tag-based access control natively
5. **Docker-friendly** - easy deployment
6. **Active development** - strong community

### Migration Path

```
Phase 1: Dual-write (ChromaDB + Qdrant)
Phase 2: Read from Qdrant, write to both
Phase 3: Full Qdrant (remove ChromaDB)
```

### Tasks

| Priority | Task | Effort |
|----------|------|--------|
| P1 | Set up Qdrant container for DGX Spark | 2h |
| P1 | Create abstraction layer for vector store | 3h |
| P1 | Implement Qdrant adapter | 3h |
| P2 | Add tag filtering to vector queries | 2h |
| P2 | Migration script from ChromaDB | 2h |
| P3 | Performance benchmarking | 2h |

**Estimated Total: 14 hours**

---

## Plan 4: Infrastructure & Deployment

### Objective
Create reproducible, portable deployment for any DGX Spark.

### Docker Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      docker-compose.yml                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  ai-ready-   │  │   qdrant     │  │   ollama     │      │
│  │     rag      │  │  (vectors)   │  │   (LLM)      │      │
│  │   :8501      │  │   :6333      │  │  :11434      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           │                                  │
│                    ┌──────┴──────┐                          │
│                    │   volumes   │                          │
│                    │  /data/rag  │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
/opt/ai-ready-rag/
├── docker-compose.yml
├── .env                    # Configuration
├── data/
│   ├── chroma/            # Vector DB (or qdrant/)
│   ├── sqlite/            # User DB, chat history
│   ├── uploads/           # Uploaded documents
│   └── models/            # Cached models
├── config/
│   ├── auth.yml           # Auth configuration
│   └── tags.yml           # Default tag structure
└── logs/
```

### Environment Configuration

```bash
# .env file
AUTH_MODE=azure_ad  # or 'local'
AZURE_CLIENT_ID=xxx
AZURE_TENANT_ID=xxx
AZURE_CLIENT_SECRET=xxx

OLLAMA_MODEL=qwen3:8b
EMBEDDING_MODEL=nomic-embed-text
VECTOR_DB=qdrant  # or 'chromadb'

DATA_DIR=/opt/ai-ready-rag/data
LOG_LEVEL=INFO
```

### USB Update Process (Air-Gapped)

```
update-package/
├── manifest.json          # Version, checksums
├── images/
│   └── ai-ready-rag.tar   # Docker image
├── migrations/
│   └── *.sql              # DB migrations
└── install.sh             # Update script
```

**Update Flow:**
1. Admin inserts USB
2. Mounts to `/media/update`
3. Runs: `sudo /opt/ai-ready-rag/apply-update.sh /media/update`
4. Script validates checksums, loads images, runs migrations
5. Restarts containers

### Tasks

| Priority | Task | Effort |
|----------|------|--------|
| P0 | Create Dockerfile for ai-ready-rag | 2h |
| P0 | Create docker-compose.yml with all services | 2h |
| P1 | Setup script for first-time deployment | 2h |
| P1 | Configuration management (.env, auth.yml) | 2h |
| P2 | USB update package structure | 3h |
| P2 | Update application script | 3h |
| P3 | Health check endpoints | 1h |
| P3 | Logging configuration | 1h |

**Estimated Total: 16 hours**

---

## Plan 5: Cite-or-Route Implementation

### Objective
Implement intelligent response routing that either answers with citations or routes to a human.

### Research Summary

The "cite or route" pattern is a RAG strategy where:
1. **Cite**: System provides answer with specific source citations
2. **Route**: System routes to human when:
   - Confidence is low
   - User lacks permission
   - Question outside knowledge base scope

Sources: [RAG Survey 2025](https://arxiv.org/html/2506.00054v1), [Enterprise RAG Guide](https://datanucleus.dev/rag-and-agentic-ai/what-is-rag-enterprise-guide-2025)

### Implementation Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│ Access Control  │ ─── User lacks permission? ──┐
│    Check        │                               │
└────────┬────────┘                               │
         │ Has access                             │
         ▼                                        │
┌─────────────────┐                               │
│   Retrieval     │ ─── No relevant docs? ───────┤
│   (filtered by  │                               │
│    user tags)   │                               │
└────────┬────────┘                               │
         │ Found docs                             │
         ▼                                        │
┌─────────────────┐                               │
│  Confidence     │ ─── Low confidence? ─────────┤
│   Scoring       │                               │
└────────┬────────┘                               │
         │ High confidence                        │
         ▼                                        ▼
┌─────────────────┐                    ┌─────────────────┐
│   CITE PATH     │                    │   ROUTE PATH    │
│ Answer with     │                    │ Identify owner  │
│ source citations│                    │ Send notification│
└─────────────────┘                    │ Log for follow-up│
                                       └─────────────────┘
```

### Routing Rules

```python
ROUTE_REASONS = {
    "no_permission": {
        "message": "This information requires additional access. I've notified {owner} about your request.",
        "action": "notify_tag_owner"
    },
    "low_confidence": {
        "message": "I'm not confident I can answer this accurately. I've routed your question to {expert}.",
        "action": "notify_subject_expert"
    },
    "no_documents": {
        "message": "I don't have information about this topic. Would you like me to route this to someone who might help?",
        "action": "offer_routing"
    },
    "out_of_scope": {
        "message": "This question is outside my knowledge area. Let me connect you with the right person.",
        "action": "route_to_admin"
    }
}
```

### Citation Format

```markdown
**Answer:**
Based on the company policy documentation, employees are entitled to 15 days of PTO per year.

**Sources:**
1. [Employee Handbook v2024, Section 4.2](doc://hr/handbook-2024.pdf#page=12)
2. [PTO Policy Update - March 2024](doc://hr/pto-update-2024.pdf)

**Confidence:** 92%
```

### Tasks

| Priority | Task | Effort |
|----------|------|--------|
| P0 | Confidence scoring for retrieval results | 3h |
| P0 | Citation extraction and formatting | 2h |
| P1 | Permission-based routing logic | 3h |
| P1 | Tag owner lookup and notification | 2h |
| P2 | Routing notification system (email/log) | 3h |
| P2 | Follow-up tracking for routed questions | 2h |

**Estimated Total: 15 hours**

---

## Plan 6: Quality Testing Framework

### Objective
Ensure OCR accuracy, retrieval relevance, and answer correctness.

### Testing Layers

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: End-to-End Testing                                  │
│ "Given this document set, does the system answer correctly?" │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Retrieval Testing                                   │
│ "Are the right chunks being retrieved for queries?"          │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Ingestion Testing                                   │
│ "Is the document being parsed correctly? OCR quality?"       │
└─────────────────────────────────────────────────────────────┘
```

### Test Suite Structure

```
tests/
├── fixtures/
│   ├── sample_docs/           # Known test documents
│   │   ├── simple_text.pdf
│   │   ├── scanned_ocr.pdf
│   │   ├── complex_table.xlsx
│   │   └── mixed_format.docx
│   └── expected_outputs/      # Expected extraction results
│       └── simple_text.json
├── test_ingestion.py          # OCR and parsing tests
├── test_retrieval.py          # Vector search accuracy
├── test_answers.py            # LLM response quality
└── test_e2e.py               # Full pipeline tests
```

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| OCR Accuracy | Character error rate on known docs | <2% |
| Chunk Relevance | % of retrieved chunks actually relevant | >80% |
| Answer Faithfulness | Answer supported by citations | >90% |
| Hallucination Rate | Answers not grounded in sources | <5% |
| Latency P95 | 95th percentile response time | <10s |

### Admin Testing UI

Add a testing page in admin that:
1. Upload a test document with known content
2. Run extraction and show parsed text
3. Compare to expected output
4. Show quality scores

### Tasks

| Priority | Task | Effort |
|----------|------|--------|
| P1 | Create test fixtures (sample docs) | 2h |
| P1 | Implement ingestion quality tests | 3h |
| P1 | Implement retrieval relevance tests | 3h |
| P2 | Implement answer quality evaluation | 3h |
| P2 | Admin testing UI page | 3h |
| P3 | Automated test runner | 2h |

**Estimated Total: 16 hours**

---

## Plan 7: Per-User Chat History

### Objective
Store and display conversation history for each user.

### Data Model

```sql
CREATE TABLE chat_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT,  -- Auto-generated from first question
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    sources TEXT,  -- JSON array of citations
    routed_to TEXT,  -- If routed
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
);
```

### UI Components

**Sidebar (on chat page):**
```
┌────────────────────┐
│ 🔍 Search history  │
├────────────────────┤
│ Today              │
│  └─ "Q3 budget..." │
│  └─ "PTO policy.." │
├────────────────────┤
│ Yesterday          │
│  └─ "Employee..."  │
├────────────────────┤
│ Last 7 days        │
│  └─ ...            │
└────────────────────┘
```

**History Page (`/history`):**
- Full list of past conversations
- Search/filter by date, keyword
- Click to view full conversation
- Option to continue conversation

### Tasks

| Priority | Task | Effort |
|----------|------|--------|
| P0 | Database schema for chat history | 1h |
| P0 | API to save chat messages | 2h |
| P0 | API to retrieve chat history | 2h |
| P1 | Chat history sidebar component | 3h |
| P1 | Full history page | 2h |
| P2 | Search within history | 2h |
| P2 | Continue past conversation | 2h |

**Estimated Total: 14 hours**

---

## Plan 8: Data Access Control (Tags & Permissions)

### Objective
Implement flexible, customer-configurable access control.

### Access Control Models (All Supported)

```yaml
# config/access_control.yml

mode: tag_based  # Options: tag_based, isolated, hierarchical, shared_private

# Tag-based configuration
tag_based:
  default_tags: ["public"]  # Tags all users get
  tag_owners:
    hr_documents: "hr-team@company.com"
    finance_data: "finance-lead@company.com"

# Isolated configuration (if mode: isolated)
isolated:
  department_collections:
    - name: "HR"
      collection: "hr_docs"
      users: ["hr_group"]
    - name: "Finance"
      collection: "finance_docs"
      users: ["finance_group"]

# Hierarchical configuration (if mode: hierarchical)
hierarchical:
  levels:
    - name: "executive"
      inherits: ["manager", "employee"]
    - name: "manager"
      inherits: ["employee"]
    - name: "employee"
      inherits: []
```

### Document Tagging Flow

```
Upload Document
      │
      ▼
┌─────────────────┐
│ Admin assigns   │
│ tags during     │
│ upload          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Tags stored in  │
│ vector metadata │
│ + SQLite        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query time:     │
│ Filter by user's│
│ accessible tags │
└─────────────────┘
```

### API Design

```python
# Tag management
POST /api/tags                    # Create tag
GET  /api/tags                    # List all tags
PUT  /api/tags/{tag_id}          # Update tag (owner, description)
DELETE /api/tags/{tag_id}        # Delete tag

# User-tag assignment
POST /api/users/{user_id}/tags   # Assign tags to user
DELETE /api/users/{user_id}/tags/{tag}  # Remove tag

# Document-tag assignment
POST /api/documents/{doc_id}/tags  # Tag a document
GET  /api/documents?tag={tag}      # List docs by tag
```

### Tasks

| Priority | Task | Effort |
|----------|------|--------|
| P0 | Tag management API | 2h |
| P0 | User-tag assignment API | 2h |
| P0 | Document-tag assignment during upload | 2h |
| P1 | Query filtering by tags | 3h |
| P1 | Access control mode configuration | 2h |
| P2 | Tag owner notification on access request | 2h |
| P2 | Admin UI for tag management | 3h |

**Estimated Total: 16 hours**

---

## Timeline Summary

### Friday Deadline Reality Check

**Total Estimated Hours:** ~137 hours

**With parallel work (2 developers + Claude):**
- Available: ~4 days × 10 hours × 3 = 120 hours
- **Conclusion:** Achievable but tight

### Recommended Phase 1 (By Friday)

Focus on core functionality:

| Plan | Priority Items | Hours |
|------|---------------|-------|
| Plan 1: UI | Login page, chat redesign, history sidebar | 13h |
| Plan 2: Auth | Local auth, basic RBAC | 10h |
| Plan 4: Infra | Dockerfile, docker-compose | 6h |
| Plan 5: Cite/Route | Basic cite with sources | 5h |
| Plan 7: History | Save/retrieve chat history | 5h |

**Phase 1 Total: ~39 hours** ✅ Achievable

### Phase 2 (Week 2)

- Azure AD integration
- Qdrant migration
- Full tag-based access
- Admin UI

### Phase 3 (Week 3+)

- Quality testing framework
- USB update system
- Advanced access control modes
- Performance optimization

---

## Open Questions for Discussion

1. **Azure AD Setup:** Do you have the Azure AD app registration details, or should we start with local auth only?

2. **Tag Structure:** What initial tags should we create? (e.g., by department, sensitivity level, topic?)

3. **Notification Method:** For routing, how should we notify tag owners? (Email, in-app, webhook?)

4. **Existing Documents:** Are there documents already in the system that need to be tagged?

5. **First Users:** Who will be the first test users and admins?

---

## Next Steps

1. **Review this plan** - Let me know what to adjust
2. **Prioritize** - Confirm Phase 1 scope
3. **Start building** - I'll begin with the highest priority items
4. **Daily check-ins** - Review progress each day

Ready to proceed when you are!
