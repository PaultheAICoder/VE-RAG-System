---
title: Admin RAG Quality Tools
status: FINAL
created: 2026-02-04
revised: 2026-02-04
author: jjob
type: Fullstack
complexity: COMPLEX
related_issues: "#148, #149"
---

# Admin RAG Quality Tools

## Summary

A unified suite of admin tools to improve RAG performance over time through:
1. **Query Synonym Management** - Admin-managed term mappings (vacation → PTO)
2. **Cache Seeding** - Inject correct answers for failed queries
3. **Curated Q&A Knowledge Base** - Persistent Q&A pairs that bypass RAG

These tools allow admins to correct RAG failures without code changes or deployments.

## Goals

- Enable admins to improve RAG accuracy iteratively
- Reduce reliance on code changes for semantic improvements
- Provide immediate fixes for queries the LLM handles poorly
- Build institutional knowledge over time
- Maintain compliance/auditability with source citations

## Scope

### In Scope

- Database tables for synonyms, curated Q&A
- CRUD API endpoints with pagination (admin-only)
- Integration with RAG pipeline
- Admin UI in new "RAG Quality" top-level view (admin-only)
- Seed data for common mappings
- Cache invalidation mechanisms

### Out of Scope

- Automated synonym discovery (future enhancement)
- LLM-suggested synonyms (future enhancement)
- User feedback collection (separate feature)
- Analytics dashboard (separate feature)
- Bulk import/export (v2)

---

## Technical Specification

### Database Schema

#### Table: `query_synonyms`

| Column | Type | Required | Default | Notes |
|--------|------|----------|---------|-------|
| id | String (UUID) | Yes | generate_uuid() | Primary key |
| term | String | Yes | - | Source term (e.g., "vacation") - single word/phrase |
| synonyms | Text | Yes | - | JSON array of synonyms (e.g., `["PTO", "paid time off"]`) |
| enabled | Boolean | Yes | True | Toggle without deletion |
| created_by | String | No | - | User ID who created |
| created_at | DateTime | Yes | utcnow() | Creation timestamp |
| updated_at | DateTime | Yes | utcnow() | Last update timestamp |

**Indexes:**
- `idx_query_synonyms_term` on `term` (exact match lookup)
- `idx_query_synonyms_enabled` on `enabled` (for filtering)

**Notes:**
- `synonyms` stored as JSON array for proper parsing (no comma ambiguity)
- `term` should be a single normalized word/phrase for word-boundary matching

#### Table: `curated_qa`

| Column | Type | Required | Default | Notes |
|--------|------|----------|---------|-------|
| id | String (UUID) | Yes | generate_uuid() | Primary key |
| keywords | Text | Yes | - | JSON array of keywords (e.g., `["401k", "retirement", "match"]`) |
| answer | Text | Yes | - | Admin-curated answer (sanitized HTML from WYSIWYG) |
| source_reference | String | **Yes** | - | **Required** - Document/section reference for compliance |
| confidence | Integer | Yes | 85 | Confidence to return (0-100) |
| priority | Integer | Yes | 0 | Higher = checked first |
| enabled | Boolean | Yes | True | Toggle without deletion |
| access_count | Integer | Yes | 0 | Usage tracking |
| last_accessed_at | DateTime | No | - | Last time matched |
| created_by | String | No | - | User ID who created |
| created_at | DateTime | Yes | utcnow() | Creation timestamp |
| updated_at | DateTime | Yes | utcnow() | Last update timestamp |

**Indexes:**
- `idx_curated_qa_priority` on `priority DESC, enabled` (for ordered lookup)

**Notes:**
- `keywords` stored as JSON array for proper parsing
- `source_reference` is **required** for compliance/auditability
- `answer` HTML is sanitized on write AND display

#### Table: `curated_qa_keywords` (Normalized for efficient matching)

| Column | Type | Required | Default | Notes |
|--------|------|----------|---------|-------|
| id | String (UUID) | Yes | generate_uuid() | Primary key |
| qa_id | String | Yes | - | FK to curated_qa.id (CASCADE DELETE) |
| keyword | String | Yes | - | Single normalized token (lowercase) |
| original_keyword | String | Yes | - | Original multi-word keyword this token came from |

**Indexes:**
- `idx_curated_qa_keywords_keyword` on `keyword` (for efficient lookup)
- `idx_curated_qa_keywords_qa_id` on `qa_id` (for joins)

**Notes:**
- Separate table enables efficient keyword lookup without loading all Q&A rows
- Multi-word keywords are tokenized: "paid time off" → 3 rows ("paid", "time", "off") all with same `original_keyword`
- Keywords are pre-normalized (lowercase, trimmed)
- Synced automatically on Q&A create/update/delete (see Service Layer)

---

### API Endpoints

#### Synonym Management

| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| GET | `/api/admin/synonyms` | List synonym mappings (paginated) | require_system_admin |
| POST | `/api/admin/synonyms` | Create new mapping | require_system_admin |
| PUT | `/api/admin/synonyms/{id}` | Update mapping | require_system_admin |
| DELETE | `/api/admin/synonyms/{id}` | Delete mapping | require_system_admin |
| POST | `/api/admin/synonyms/invalidate-cache` | Force cache refresh | require_system_admin |

**Request/Response Models:**

```python
class SynonymCreate(BaseModel):
    term: str  # Single word/phrase
    synonyms: list[str]  # Array of synonyms

class SynonymUpdate(BaseModel):
    term: str | None = None
    synonyms: list[str] | None = None
    enabled: bool | None = None

class SynonymResponse(BaseModel):
    id: str
    term: str
    synonyms: list[str]
    enabled: bool
    created_by: str | None
    created_at: datetime
    updated_at: datetime

class SynonymListResponse(BaseModel):
    synonyms: list[SynonymResponse]
    total: int
    page: int
    page_size: int

# List endpoint params
@router.get("/synonyms")
async def list_synonyms(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    enabled: bool | None = Query(None),
    search: str | None = Query(None),  # Search term field
    ...
)
```

#### Curated Q&A Management

| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| GET | `/api/admin/qa` | List Q&A pairs (paginated) | require_system_admin |
| POST | `/api/admin/qa` | Create new Q&A pair | require_system_admin |
| PUT | `/api/admin/qa/{id}` | Update Q&A pair | require_system_admin |
| DELETE | `/api/admin/qa/{id}` | Delete Q&A pair | require_system_admin |
| POST | `/api/admin/qa/invalidate-cache` | Force cache refresh | require_system_admin |

**Request/Response Models:**

```python
class CuratedQACreate(BaseModel):
    keywords: list[str]  # Array of keywords
    answer: str  # HTML from WYSIWYG editor (will be sanitized)
    source_reference: str  # Required for compliance
    confidence: int = 85
    priority: int = 0

    @validator('source_reference')
    def source_required(cls, v):
        if not v or not v.strip():
            raise ValueError('source_reference is required for compliance')
        return v.strip()

    @validator('answer')
    def sanitize_html(cls, v):
        return sanitize_html(v, ALLOWED_TAGS)

class CuratedQAUpdate(BaseModel):
    keywords: list[str] | None = None
    answer: str | None = None  # Will be sanitized if provided
    source_reference: str | None = None
    confidence: int | None = None
    priority: int | None = None
    enabled: bool | None = None

class CuratedQAResponse(BaseModel):
    id: str
    keywords: list[str]
    answer: str  # Sanitized HTML
    source_reference: str
    confidence: int
    priority: int
    enabled: bool
    access_count: int
    last_accessed_at: datetime | None
    created_by: str | None
    created_at: datetime
    updated_at: datetime

class CuratedQAListResponse(BaseModel):
    qa_pairs: list[CuratedQAResponse]
    total: int
    page: int
    page_size: int
```

#### Cache Seeding

| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| POST | `/api/admin/cache/seed` | Seed cache with correct answer | require_system_admin |

**Request/Response Models:**

```python
class CacheSeedRequest(BaseModel):
    query: str
    answer: str  # Will be sanitized
    confidence: int = 85
    source_reference: str  # Required for compliance
    document_name: str | None = None  # Optional: friendly name for citation display

    @validator('source_reference')
    def source_required(cls, v):
        if not v or not v.strip():
            raise ValueError('source_reference is required')
        return v.strip()

class CacheSeedResponse(BaseModel):
    success: bool
    query_hash: str
    message: str
```

**Cache Seeding Behavior:**
1. Generates query embedding via embedding service
2. Builds a full Citation object for consistent downstream rendering:
   ```python
   citation = Citation(
       source_id=f"seeded:{query_hash}",
       document_id=f"seeded:{query_hash}",
       document_name=request.document_name or request.source_reference,
       chunk_index=0,
       page_number=None,
       section=request.source_reference,
       relevance_score=1.0,
       snippet=request.source_reference,
       snippet_full=request.source_reference,
   )
   ```
3. Stores in `ResponseCache` table with:
   - `query_hash`: SHA256 of normalized query
   - `query_embedding`: JSON-encoded embedding vector
   - `answer`: Sanitized HTML
   - `confidence_overall`: Specified confidence
   - `sources`: JSON-encoded list of Citation objects (full structure)
   - `model_used`: "admin_seeded"
4. TTL follows existing cache settings (`cache_ttl_hours`)
5. Checked after curated Q&A, before full RAG pipeline

---

### Service Layer Changes

#### HTML Sanitization

```python
import bleach

ALLOWED_TAGS = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li', 'a']
ALLOWED_ATTRIBUTES = {'a': ['href', 'title']}

def sanitize_html(html: str, allowed_tags: list[str] = ALLOWED_TAGS) -> str:
    """Sanitize HTML on write to prevent XSS."""
    return bleach.clean(
        html,
        tags=allowed_tags,
        attributes=ALLOWED_ATTRIBUTES,
        strip=True
    )
```

#### Word Boundary Matching

```python
import re

def tokenize_query(query: str) -> set[str]:
    """Extract normalized word tokens from query."""
    # Split on non-alphanumeric, lowercase, filter short words
    words = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
    return {w for w in words if len(w) >= 2}

def matches_keyword(keyword: str, query_tokens: set[str]) -> bool:
    """Check if keyword matches any query token (word boundary safe).

    Handles multi-word keywords by checking all words are present.
    Example: "paid time off" matches query "what is paid time off policy"
    """
    keyword_tokens = tokenize_query(keyword)
    return keyword_tokens.issubset(query_tokens)
```

#### Curated Q&A Keyword Sync

```python
def sync_qa_keywords(db: Session, qa_id: str, keywords: list[str]) -> None:
    """Sync curated_qa_keywords table when Q&A is created/updated.

    Deletes existing keywords for qa_id and inserts new tokenized entries.
    Called by CRUD endpoints after Q&A create/update.
    """
    # Delete existing keywords for this Q&A
    db.query(CuratedQAKeyword).filter(
        CuratedQAKeyword.qa_id == qa_id
    ).delete()

    # Insert tokenized keywords
    for original_keyword in keywords:
        tokens = tokenize_query(original_keyword)
        for token in tokens:
            db.add(CuratedQAKeyword(
                qa_id=qa_id,
                keyword=token,
                original_keyword=original_keyword.lower().strip(),
            ))

    db.flush()

    # Invalidate in-memory cache
    invalidate_qa_cache()

def delete_qa_keywords(db: Session, qa_id: str) -> None:
    """Delete all keywords for a Q&A (called before Q&A deletion).

    Note: CASCADE DELETE on FK handles this automatically, but explicit
    delete ensures cache invalidation happens.
    """
    db.query(CuratedQAKeyword).filter(
        CuratedQAKeyword.qa_id == qa_id
    ).delete()

    invalidate_qa_cache()
```

**CRUD Integration:**

```python
# In POST /api/admin/qa (create)
@router.post("/qa")
async def create_qa(request: CuratedQACreate, db: Session = Depends(get_db)):
    qa = CuratedQA(
        keywords=json.dumps(request.keywords),
        answer=sanitize_html(request.answer),
        source_reference=request.source_reference,
        confidence=request.confidence,
        priority=request.priority,
    )
    db.add(qa)
    db.flush()  # Get qa.id

    # Sync keywords to normalized table
    sync_qa_keywords(db, qa.id, request.keywords)

    db.commit()
    return qa

# In PUT /api/admin/qa/{id} (update)
@router.put("/qa/{qa_id}")
async def update_qa(qa_id: str, request: CuratedQAUpdate, db: Session = Depends(get_db)):
    qa = db.query(CuratedQA).filter(CuratedQA.id == qa_id).first()
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A not found")

    if request.keywords is not None:
        qa.keywords = json.dumps(request.keywords)
        # Re-sync keywords to normalized table
        sync_qa_keywords(db, qa.id, request.keywords)

    # ... update other fields
    db.commit()
    return qa

# In DELETE /api/admin/qa/{id}
@router.delete("/qa/{qa_id}")
async def delete_qa(qa_id: str, db: Session = Depends(get_db)):
    qa = db.query(CuratedQA).filter(CuratedQA.id == qa_id).first()
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A not found")

    # Keywords deleted via CASCADE, but invalidate cache
    delete_qa_keywords(db, qa_id)

    db.delete(qa)
    db.commit()
    return {"success": True}
```

#### Updated `expand_query()` in `rag_service.py`

```python
# Module-level cache with version tracking
_synonym_cache: list[QuerySynonym] | None = None
_synonym_cache_time: datetime | None = None
_synonym_cache_version: int = 0  # Incremented on invalidation

def invalidate_synonym_cache() -> None:
    """Explicitly invalidate synonym cache (call on CRUD operations)."""
    global _synonym_cache, _synonym_cache_version
    _synonym_cache = None
    _synonym_cache_version += 1

def get_cached_synonyms(db: Session) -> list[QuerySynonym]:
    """Get synonyms with in-memory caching (60s TTL + explicit invalidation)."""
    global _synonym_cache, _synonym_cache_time

    now = datetime.utcnow()
    cache_expired = (
        _synonym_cache is None or
        _synonym_cache_time is None or
        (now - _synonym_cache_time).total_seconds() > 60
    )

    if cache_expired:
        _synonym_cache = db.query(QuerySynonym).filter(
            QuerySynonym.enabled == True
        ).all()
        _synonym_cache_time = now

    return _synonym_cache or []

def expand_query(question: str, db: Session | None = None) -> list[str]:
    """Expand query with synonyms using word-boundary matching.

    Returns list of query variants for multi-query retrieval.
    """
    queries = [question]
    query_tokens = tokenize_query(question)

    # 1. Check database synonyms (if db provided)
    if db is not None:
        synonyms = get_cached_synonyms(db)
        for syn in synonyms:
            # Word-boundary match: "vacation" matches "vacation policy"
            # but NOT "vacationing" or substring matches
            if matches_keyword(syn.term, query_tokens):
                # Parse JSON array of synonyms
                syn_list = json.loads(syn.synonyms)
                queries.extend(syn_list)

    # 2. Existing hardcoded patterns (fallback)
    if "customer" in query_tokens or "client" in query_tokens:
        queries.append("customer company revenue sales account")
    # ... rest of hardcoded patterns

    return queries
```

#### New `check_curated_qa()` in `rag_service.py`

```python
# Module-level cache for curated Q&A
_qa_token_index: dict[str, set[str]] | None = None  # token -> set of qa_ids (candidate matches)
_qa_cache: dict[str, CuratedQA] | None = None  # qa_id -> CuratedQA
_qa_keywords_by_id: dict[str, list[str]] | None = None  # qa_id -> original keywords list
_qa_cache_time: datetime | None = None

def invalidate_qa_cache() -> None:
    """Explicitly invalidate Q&A cache (call on CRUD operations)."""
    global _qa_token_index, _qa_cache, _qa_keywords_by_id
    _qa_token_index = None
    _qa_cache = None
    _qa_keywords_by_id = None

def get_cached_qa(db: Session) -> tuple[dict[str, set[str]], dict[str, CuratedQA], dict[str, list[str]]]:
    """Get Q&A with in-memory caching (60s TTL + explicit invalidation).

    Returns:
        token_index: Maps single token -> set of candidate qa_ids
        qa_lookup: Maps qa_id -> CuratedQA object (sorted by priority)
        keywords_by_id: Maps qa_id -> list of original keywords for verification
    """
    global _qa_token_index, _qa_cache, _qa_keywords_by_id, _qa_cache_time

    now = datetime.utcnow()
    cache_expired = (
        _qa_token_index is None or
        _qa_cache_time is None or
        (now - _qa_cache_time).total_seconds() > 60
    )

    if cache_expired:
        # Load all enabled Q&A pairs ordered by priority
        qa_pairs = db.query(CuratedQA).filter(
            CuratedQA.enabled == True
        ).order_by(CuratedQA.priority.desc()).all()

        # Build token index (for candidate filtering) and keyword lookup (for verification)
        _qa_token_index = {}
        _qa_cache = {}
        _qa_keywords_by_id = {}

        for qa in qa_pairs:
            _qa_cache[qa.id] = qa
            keywords = json.loads(qa.keywords)
            _qa_keywords_by_id[qa.id] = keywords

            # Index each token from each keyword for fast candidate lookup
            for kw in keywords:
                for token in tokenize_query(kw):
                    if token not in _qa_token_index:
                        _qa_token_index[token] = set()
                    _qa_token_index[token].add(qa.id)

        _qa_cache_time = now

    return _qa_token_index or {}, _qa_cache or {}, _qa_keywords_by_id or {}

def check_curated_qa(query: str, db: Session) -> CuratedQA | None:
    """Check if query matches any curated Q&A pair by keyword.

    Uses two-phase matching:
    1. Token index lookup for candidate Q&A pairs (fast filter)
    2. Full keyword verification using matches_keyword() (accurate match)

    This handles multi-word keywords like "paid time off" correctly.
    Returns highest-priority matching Q&A, or None to proceed with RAG.
    """
    token_index, qa_lookup, keywords_by_id = get_cached_qa(db)
    query_tokens = tokenize_query(query)

    # Phase 1: Find candidate Q&A IDs (any token match)
    candidate_qa_ids: set[str] = set()
    for token in query_tokens:
        if token in token_index:
            candidate_qa_ids.update(token_index[token])

    if not candidate_qa_ids:
        return None

    # Phase 2: Verify full keyword match for each candidate
    # Iterate in priority order (qa_lookup is ordered by priority desc)
    for qa_id, qa in qa_lookup.items():
        if qa_id not in candidate_qa_ids:
            continue

        # Check if ANY keyword fully matches the query
        keywords = keywords_by_id.get(qa_id, [])
        for keyword in keywords:
            if matches_keyword(keyword, query_tokens):
                return qa

    return None
```

#### RAG Pipeline Integration

In `RAGService.generate()`, add checks in order:

```python
async def generate(self, request: RAGRequest, db: Session) -> RAGResponse:
    start_time = time.perf_counter()

    # 0. Check curated Q&A first (highest priority - admin-approved answers)
    curated_qa = check_curated_qa(request.query, db)
    if curated_qa:
        # Update access tracking
        curated_qa.access_count += 1
        curated_qa.last_accessed_at = datetime.utcnow()
        db.commit()

        # Build citation from source_reference
        citation = Citation(
            source_id=f"curated:{curated_qa.id}",
            document_id=curated_qa.id,
            document_name=curated_qa.source_reference,
            chunk_index=0,
            page_number=None,
            section=curated_qa.source_reference,
            relevance_score=1.0,
            snippet=curated_qa.source_reference,
            snippet_full=curated_qa.source_reference,
        )

        return RAGResponse(
            answer=curated_qa.answer,
            confidence=ConfidenceScore(
                overall=curated_qa.confidence,
                retrieval_score=1.0,
                coverage_score=1.0,
                llm_score=curated_qa.confidence,
            ),
            citations=[citation],  # Include citation for compliance
            action="CITE",
            route_to=None,
            model_used="curated",
            context_chunks_used=0,
            context_tokens_used=0,
            generation_time_ms=(time.perf_counter() - start_time) * 1000,
            grounded=True,
            routing_decision=None,
        )

    # 1. Check seeded cache (admin-seeded correct answers)
    # Uses existing cache.get() which checks semantic similarity
    # Seeded entries have model_used="admin_seeded"

    # 2. Continue with normal RAG pipeline...
    # (existing code)
```

**Pipeline Order:**
1. **Curated Q&A** - Exact keyword match, instant response
2. **Seeded Cache** - Semantic similarity match (existing cache layer)
3. **Full RAG Pipeline** - Retrieval + LLM generation

---

### Frontend Components

#### Navigation Structure

New top-level "RAG Quality" tab (admin-only):

```
Navigation (Admin view):
├── Chat
├── Documents
├── RAG Quality  ← New (admin-only, hidden from regular users)
│   ├── Synonyms tab
│   └── Curated Q&A tab
└── Settings
```

**Location:** `frontend/src/views/RAGQualityView.tsx` (new view)

---

### UI Wireframes

#### Main RAG Quality View - Synonyms Tab

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🏠 AI Ready RAG                          [Admin User ▼]  [Settings]  [?]   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐ ┌────────────┐ ┌───────────────┐ ┌──────────┐                │
│  │   Chat   │ │  Documents │ │  RAG Quality  │ │ Settings │                │
│  └──────────┘ └────────────┘ └───────────────┘ └──────────┘                │
│                                    ▲ active                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐                                 │
│   │    Synonyms     │  │   Curated Q&A   │                                 │
│   └─────────────────┘  └─────────────────┘                                 │
│          ▲ active                                                           │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │  Query Synonyms                                    [+ Add Synonym]  │  │
│   │                                                                     │  │
│   │  ┌─────────────────────────────────────────────────────────────┐   │  │
│   │  │ Search synonyms...                                      🔍  │   │  │
│   │  └─────────────────────────────────────────────────────────────┘   │  │
│   │                                                                     │  │
│   │  ┌───────────────┬─────────────────────────────┬────────┬───────┐  │  │
│   │  │ Term          │ Synonyms                    │ Status │Actions│  │  │
│   │  ├───────────────┼─────────────────────────────┼────────┼───────┤  │  │
│   │  │ vacation      │ PTO, paid time off, leave   │ ✓ On   │ ✎ 🗑  │  │  │
│   │  │ 401k          │ 401(k), retirement plan     │ ✓ On   │ ✎ 🗑  │  │  │
│   │  │ health ins... │ medical, healthcare         │ ✓ On   │ ✎ 🗑  │  │  │
│   │  │ dental        │ dental insurance, dental... │ ○ Off  │ ✎ 🗑  │  │  │
│   │  └───────────────┴─────────────────────────────┴────────┴───────┘  │  │
│   │                                                                     │  │
│   │  Showing 1-4 of 8                              [< 1 2 >]            │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Curated Q&A Tab

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐                                 │
│   │    Synonyms     │  │   Curated Q&A   │                                 │
│   └─────────────────┘  └─────────────────┘                                 │
│                               ▲ active                                      │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │  Curated Q&A                                      [+ Add Q&A Pair]  │  │
│   │                                                                     │  │
│   │  ┌─────────────────────────────────────────────────────────────┐   │  │
│   │  │ Search keywords...                                      🔍  │   │  │
│   │  └─────────────────────────────────────────────────────────────┘   │  │
│   │                                                                     │  │
│   │  ┌────────────────────────────────────────────────────────────────┐│  │
│   │  │ ┌──────────────────────────────────────────────────────────┐  ││  │
│   │  │ │ Keywords: 401k, retirement, match                        │  ││  │
│   │  │ │ Source: Benefits Guide, Section 5.2                      │  ││  │
│   │  │ │ Confidence: 85%  │  Priority: 10  │  Hits: 47            │  ││  │
│   │  │ ├──────────────────────────────────────────────────────────┤  ││  │
│   │  │ │ Example Dental matches 100% of the first 3% and 50%      │  ││  │
│   │  │ │ of the next 2% of salary contributed (up to 4% total     │  ││  │
│   │  │ │ match). Vesting: immediate for employee...               │  ││  │
│   │  │ └──────────────────────────────────────────────────────────┘  ││  │
│   │  │                                      ✓ Enabled   [✎ Edit] [🗑] ││  │
│   │  └────────────────────────────────────────────────────────────────┘│  │
│   │                                                                     │  │
│   │  ┌────────────────────────────────────────────────────────────────┐│  │
│   │  │ ┌──────────────────────────────────────────────────────────┐  ││  │
│   │  │ │ Keywords: vacation, pto, time off                        │  ││  │
│   │  │ │ Source: PTO Policy, Page 3                               │  ││  │
│   │  │ │ Confidence: 80%  │  Priority: 5   │  Hits: 23            │  ││  │
│   │  │ ├──────────────────────────────────────────────────────────┤  ││  │
│   │  │ │ PTO accrues bi-weekly based on years of service...       │  ││  │
│   │  │ └──────────────────────────────────────────────────────────┘  ││  │
│   │  │                                      ✓ Enabled   [✎ Edit] [🗑] ││  │
│   │  └────────────────────────────────────────────────────────────────┘│  │
│   │                                                                     │  │
│   │  Showing 1-2 of 5                              [< 1 2 3 >]          │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Add/Edit Q&A Modal (with WYSIWYG Editor)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                              [✕]    │  │
│   │   Add Curated Q&A                                                   │  │
│   │   ─────────────────────────────────────────────────────────────     │  │
│   │                                                                     │  │
│   │   Keywords                                                          │  │
│   │   ┌─────────────────────────────────────────────────────────────┐  │  │
│   │   │ [401k ✕] [retirement ✕] [company match ✕]  [+ Add keyword]  │  │  │
│   │   └─────────────────────────────────────────────────────────────┘  │  │
│   │   ℹ️ Words/phrases that trigger this answer. Press Enter to add.    │  │
│   │                                                                     │  │
│   │   Source Reference *                                                │  │
│   │   ┌─────────────────────────────────────────────────────────────┐  │  │
│   │   │ Benefits Guide, Section 5.2                                 │  │  │
│   │   └─────────────────────────────────────────────────────────────┘  │  │
│   │   ℹ️ Required for compliance (e.g., "Employee Handbook, Page 12")   │  │
│   │                                                                     │  │
│   │   Answer *                                                          │  │
│   │   ┌─────────────────────────────────────────────────────────────┐  │  │
│   │   │ [B] [I] [• List] [1. List] [🔗 Link]                        │  │  │
│   │   ├─────────────────────────────────────────────────────────────┤  │  │
│   │   │                                                             │  │  │
│   │   │ Example Dental matches **100% of the first 3%** and         │  │  │
│   │   │ **50% of the next 2%** of salary contributed.               │  │  │
│   │   │                                                             │  │  │
│   │   │ • Up to 4% total match                                      │  │  │
│   │   │ • Immediate vesting for employee contributions              │  │  │
│   │   │ • 3-year cliff vesting for company match                    │  │  │
│   │   │                                                             │  │  │
│   │   └─────────────────────────────────────────────────────────────┘  │  │
│   │                                                                     │  │
│   │   ┌─────────────────────┐  ┌─────────────────────┐                 │  │
│   │   │ Confidence: [85 ▼]  │  │ Priority:   [10  ]  │                 │  │
│   │   └─────────────────────┘  └─────────────────────┘                 │  │
│   │                                                                     │  │
│   │                               [Cancel]  [Save Q&A]                  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Add/Edit Synonym Modal

```
┌───────────────────────────────────────────────────┐
│                                            [✕]    │
│   Add Synonym Mapping                             │
│   ───────────────────────────────────────────     │
│                                                   │
│   Term                                            │
│   ┌───────────────────────────────────────────┐  │
│   │ vacation                                  │  │
│   └───────────────────────────────────────────┘  │
│   ℹ️ The word users might search for              │
│                                                   │
│   Synonyms                                        │
│   ┌───────────────────────────────────────────┐  │
│   │ [PTO ✕] [paid time off ✕] [leave ✕]      │  │
│   │ [+ Add synonym]                           │  │
│   └───────────────────────────────────────────┘  │
│   ℹ️ Related terms to expand the search           │
│                                                   │
│                     [Cancel]  [Save Synonym]      │
│                                                   │
└───────────────────────────────────────────────────┘
```

---

#### Components to Create

| Component | Location | Purpose |
|-----------|----------|---------|
| `RAGQualityView.tsx` | `views/` | Main view with tabs |
| `SynonymManager.tsx` | `components/admin/` | CRUD table with pagination |
| `QAManager.tsx` | `components/admin/` | CRUD table with pagination (card layout) |
| `SynonymForm.tsx` | `components/admin/` | Create/edit synonym modal |
| `QAForm.tsx` | `components/admin/` | Create/edit Q&A modal |
| `RichTextEditor.tsx` | `components/common/` | TipTap WYSIWYG wrapper |
| `TagInput.tsx` | `components/common/` | Tag/chip input for keywords & synonyms |

**New Dependencies:**

```bash
npm install @tiptap/react @tiptap/starter-kit @tiptap/extension-link dompurify
```

**RichTextEditor Component:**
- Wraps TipTap for consistent WYSIWYG editing
- Toolbar: Bold, Italic, Bullet List, Numbered List, Link
- Outputs sanitized HTML string for storage
- Sanitizes on display with DOMPurify

**Frontend HTML Sanitization:**
```typescript
import DOMPurify from 'dompurify';

const ALLOWED_TAGS = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li', 'a'];

export function sanitizeHtml(html: string): string {
  return DOMPurify.sanitize(html, { ALLOWED_TAGS });
}
```

**TagInput Component:**
```typescript
interface TagInputProps {
  value: string[];  // Array of tags
  onChange: (tags: string[]) => void;
  placeholder?: string;
  maxTags?: number;
}

function TagInput({ value, onChange, placeholder, maxTags }: TagInputProps) {
  const [inputValue, setInputValue] = useState('');

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && inputValue.trim()) {
      e.preventDefault();
      if (!maxTags || value.length < maxTags) {
        onChange([...value, inputValue.trim()]);
        setInputValue('');
      }
    }
  };

  const removeTag = (index: number) => {
    onChange(value.filter((_, i) => i !== index));
  };

  return (
    <div className="tag-input-container">
      {value.map((tag, i) => (
        <span key={i} className="tag">
          {tag}
          <button onClick={() => removeTag(i)}>✕</button>
        </span>
      ))}
      <input
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
      />
    </div>
  );
}
```

**API Contract:** The TagInput outputs `string[]` directly, matching the API's `list[str]` expectation. No transformation needed.

---

### Seed Data

Initial synonyms to pre-populate:

```python
INITIAL_SYNONYMS = [
    ("vacation", ["PTO", "paid time off", "time off", "leave", "annual leave"]),
    ("401k", ["401(k)", "retirement", "retirement plan", "retirement savings"]),
    ("health insurance", ["medical", "medical insurance", "healthcare", "health coverage"]),
    ("dental", ["dental insurance", "dental coverage", "dental plan"]),
    ("vision", ["vision insurance", "vision coverage", "eye care"]),
    ("sick leave", ["sick time", "sick days", "illness leave"]),
    ("maternity", ["maternity leave", "parental leave", "paternity leave"]),
    ("pto", ["paid time off", "vacation", "time off", "leave"]),
]
```

---

## Risk Flags

### MULTI_MODEL Risk
- Curated Q&A check happens before RAG pipeline
- Must ensure clean separation (no side effects on RAG flow)
- **Mitigation:** Return early with complete RAGResponse

### CACHE_INVALIDATION Risk
- Synonym and Q&A caches are in-memory per process (60s TTL)
- **Mitigation:**
  - Explicit invalidation endpoints
  - Cache version tracking
  - Single-worker deployment (current architecture)

### HTML_SANITIZATION Risk
- WYSIWYG editor outputs HTML stored in database
- **Mitigation:**
  - Sanitize on write (backend validator)
  - Sanitize on display (frontend DOMPurify)
  - Allowed tags whitelist

### FALSE_POSITIVE Risk
- Keyword matching could match unintended queries
- **Mitigation:**
  - Word-boundary matching (tokenization)
  - "pto" won't match "photo"
  - Admin review of Q&A effectiveness

---

## Acceptance Criteria

### Synonym Management
- [ ] `query_synonyms` table created with migration
- [ ] CRUD API endpoints with pagination (admin-only)
- [ ] `expand_query()` uses word-boundary matching
- [ ] Explicit cache invalidation endpoint
- [ ] Admin UI to view/add/edit/delete synonyms
- [ ] Seed data loaded on first run

### Curated Q&A
- [ ] `curated_qa` table created with migration
- [ ] `source_reference` is required (enforced by validator)
- [ ] CRUD API endpoints with pagination (admin-only)
- [ ] RAG pipeline checks Q&A before generation
- [ ] Keyword matching uses word boundaries (no false positives)
- [ ] Returns citation from source_reference for compliance
- [ ] Explicit cache invalidation endpoint
- [ ] Admin UI to manage Q&A pairs with WYSIWYG editor
- [ ] Access tracking (count, last_accessed_at)

### Cache Seeding
- [ ] `POST /api/admin/cache/seed` endpoint working
- [ ] Requires source_reference
- [ ] Generates query embedding for semantic matching
- [ ] Stores in ResponseCache with model_used="admin_seeded"
- [ ] Checked after curated Q&A, before full RAG

### HTML Safety
- [ ] Backend sanitizes HTML on write
- [ ] Frontend sanitizes HTML on display
- [ ] Allowed tags whitelist defined

### Integration
- [ ] All admin endpoints require `require_system_admin`
- [ ] Audit logging for all changes
- [ ] RAG Quality view accessible as top-level nav (admin-only)

---

## Design Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Answer formatting | Full WYSIWYG editor (TipTap) | Admins don't know markdown; visual editing is intuitive |
| Keyword/synonym input | Tag/chip input component | Directly outputs `string[]` matching API; no CSV parsing needed |
| Bulk import/export | Manual entry only for v1 | Build list organically from failed queries; add export later |
| Pattern matching | Keyword only for v1 (word boundaries) | Simpler for admins, avoids regex complexity/security risks |
| Multi-word keywords | Tokenize and verify | "paid time off" tokenized for index, full match verified via `matches_keyword()` |
| Tenant scope | Global | Single-tenant deployment; shared for consistent experience |
| Keywords storage | JSON array | Avoids comma-in-value ambiguity |
| Source reference | Required | Compliance/auditability requirement |
| Admin view location | Top-level "RAG Quality" nav tab | Dedicated view for admin tools, not buried in Settings |

---

## Implementation Phases

### Phase 1: Database & API (Backend)
- Create models and migrations
- Implement CRUD endpoints with pagination
- Add seed data
- Implement HTML sanitization

### Phase 2: RAG Integration (Backend)
- Update `expand_query()` with word-boundary matching
- Add `check_curated_qa()` with caching
- Implement cache seeding endpoint
- Add cache invalidation endpoints

### Phase 3: Admin UI (Frontend)
- Create RAGQualityView with Synonyms and Curated Q&A tabs
- Create SynonymManager and QAManager components
- Add TipTap WYSIWYG editor for Q&A answers
- Add TagInput component for keywords/synonyms
- Test end-to-end

---

## References

- Current expand_query: `ai_ready_rag/services/rag_service.py:187`
- Cache service: `ai_ready_rag/services/cache_service.py`
- Admin API patterns: `ai_ready_rag/api/admin.py`
- Settings service: `ai_ready_rag/services/settings_service.py`
- Related issues: #148, #149
