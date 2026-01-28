# Chat API Specification

| Field | Value |
|-------|-------|
| **Status** | DRAFT |
| **Version** | 1.0 |
| **Created** | 2026-01-28 |
| **Type** | CRUD + Integration |
| **Complexity** | SIMPLE |

## Summary

REST API for chat session management and message handling. Integrates with RAGService to generate AI responses with citations, confidence scoring, and routing.

## Goals

- Enable users to create and manage chat sessions
- Send messages and receive RAG-powered responses
- Persist chat history with citations and routing info
- Support session listing and message retrieval

## Scope

### In Scope

- CRUD for chat sessions (create, list, archive)
- Message creation with RAGService integration
- Message history retrieval with pagination
- Session ownership enforcement (users see only their sessions)

### Out of Scope

- Real-time streaming (WebSocket) - future enhancement
- Message editing/deletion - immutable history
- Shared sessions between users
- Session export/import

---

## Endpoints

### 1. Create Session

```
POST /api/chat/sessions
```

**Auth**: Required (`get_current_user`)

**Request Body**:
```json
{
  "title": "Optional session title"
}
```

**Response** (201 Created):
```json
{
  "id": "uuid",
  "user_id": "uuid",
  "title": "Optional session title",
  "created_at": "2026-01-28T12:00:00Z",
  "updated_at": "2026-01-28T12:00:00Z",
  "is_archived": false,
  "message_count": 0
}
```

**Notes**:
- `title` defaults to null if not provided
- `user_id` set from authenticated user

---

### 2. List Sessions

```
GET /api/chat/sessions
```

**Auth**: Required (`get_current_user`)

**Query Parameters**:
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `archived` | bool | `false` | Include archived sessions |
| `limit` | int | `20` | Max results (1-100) |
| `offset` | int | `0` | Pagination offset |

**Response** (200 OK):
```json
{
  "sessions": [
    {
      "id": "uuid",
      "title": "Session title",
      "created_at": "2026-01-28T12:00:00Z",
      "updated_at": "2026-01-28T14:30:00Z",
      "is_archived": false,
      "message_count": 5,
      "last_message_preview": "What is the policy on..."
    }
  ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

**Notes**:
- Only returns sessions for authenticated user
- Ordered by `updated_at` DESC (most recent first)
- `last_message_preview` is first 100 chars of last user message

---

### 3. Get Session

```
GET /api/chat/sessions/{session_id}
```

**Auth**: Required (`get_current_user`)

**Response** (200 OK):
```json
{
  "id": "uuid",
  "user_id": "uuid",
  "title": "Session title",
  "created_at": "2026-01-28T12:00:00Z",
  "updated_at": "2026-01-28T14:30:00Z",
  "is_archived": false,
  "message_count": 5
}
```

**Errors**:
- `404` - Session not found or doesn't belong to user

---

### 4. Update Session

```
PATCH /api/chat/sessions/{session_id}
```

**Auth**: Required (`get_current_user`)

**Request Body**:
```json
{
  "title": "New title",
  "is_archived": true
}
```

**Response** (200 OK): Updated session object

**Notes**:
- Only `title` and `is_archived` are updatable
- Use this to archive/unarchive sessions

---

### 5. Send Message

```
POST /api/chat/sessions/{session_id}/messages
```

**Auth**: Required (`get_current_user`)

**Request Body**:
```json
{
  "content": "What is the company policy on remote work?",
  "model": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | string | Yes | User's question (1-4000 chars) |
| `model` | string | No | Override model (must be in allowlist) |

**Response** (201 Created):
```json
{
  "user_message": {
    "id": "uuid",
    "role": "user",
    "content": "What is the company policy on remote work?",
    "created_at": "2026-01-28T14:30:00Z"
  },
  "assistant_message": {
    "id": "uuid",
    "role": "assistant",
    "content": "Based on the documentation, the remote work policy states... [SourceId: abc123:0]",
    "sources": [
      {
        "source_id": "abc123:0",
        "document_id": "abc123",
        "chunk_index": 0,
        "title": "HR Policy Manual",
        "snippet": "Remote work is permitted for..."
      }
    ],
    "confidence": {
      "overall": 75,
      "retrieval": 80,
      "coverage": 70,
      "llm": 75
    },
    "action": "CITE",
    "was_routed": false,
    "routed_to": null,
    "route_reason": null,
    "created_at": "2026-01-28T14:30:01Z"
  },
  "generation_time_ms": 1250
}
```

**When Routed** (`confidence.overall < 60`):
```json
{
  "user_message": { ... },
  "assistant_message": {
    "id": "uuid",
    "role": "assistant",
    "content": "I don't have enough information to answer confidently. This has been routed to an expert.",
    "sources": [],
    "confidence": {
      "overall": 35,
      "retrieval": 40,
      "coverage": 30,
      "llm": 35
    },
    "action": "ROUTE",
    "was_routed": true,
    "routed_to": "hr-team@company.com",
    "route_reason": "Low confidence - insufficient context",
    "created_at": "2026-01-28T14:30:01Z"
  },
  "generation_time_ms": 850
}
```

**Errors**:
- `404` - Session not found or doesn't belong to user
- `400` - Invalid content (empty, too long)
- `503` - RAGService unavailable (Ollama down)

**Implementation Flow**:
1. Validate session ownership
2. Create user message in DB
3. Load chat history (last N messages for context)
4. Build RAGRequest with user's tags
5. Call `RAGService.generate()`
6. Create assistant message in DB with response data
7. Update session's `updated_at`
8. Return both messages

---

### 6. Get Messages

```
GET /api/chat/sessions/{session_id}/messages
```

**Auth**: Required (`get_current_user`)

**Query Parameters**:
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | `50` | Max messages (1-100) |
| `before` | string | null | Cursor: get messages before this ID |
| `after` | string | null | Cursor: get messages after this ID |

**Response** (200 OK):
```json
{
  "messages": [
    {
      "id": "uuid",
      "role": "user",
      "content": "What is the policy?",
      "sources": null,
      "confidence": null,
      "was_routed": false,
      "routed_to": null,
      "route_reason": null,
      "created_at": "2026-01-28T14:30:00Z"
    },
    {
      "id": "uuid",
      "role": "assistant",
      "content": "The policy states...",
      "sources": [...],
      "confidence": { "overall": 75, ... },
      "was_routed": false,
      "routed_to": null,
      "route_reason": null,
      "created_at": "2026-01-28T14:30:01Z"
    }
  ],
  "has_more": true,
  "total": 24
}
```

**Notes**:
- Ordered by `created_at` ASC (oldest first for chat display)
- Use `before`/`after` cursors for infinite scroll

---

## Data Models

### Pydantic Schemas (to create)

```python
# Request schemas
class SessionCreate(BaseModel):
    title: str | None = None

class SessionUpdate(BaseModel):
    title: str | None = None
    is_archived: bool | None = None

class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None

# Response schemas
class SessionResponse(BaseModel):
    id: str
    user_id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    is_archived: bool
    message_count: int

    class Config:
        from_attributes = True

class SessionListItem(BaseModel):
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    is_archived: bool
    message_count: int
    last_message_preview: str | None

class SessionListResponse(BaseModel):
    sessions: list[SessionListItem]
    total: int
    limit: int
    offset: int

class SourceInfo(BaseModel):
    source_id: str
    document_id: str
    chunk_index: int
    title: str | None
    snippet: str | None

class ConfidenceInfo(BaseModel):
    overall: int
    retrieval: int
    coverage: int
    llm: int

class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources: list[SourceInfo] | None
    confidence: ConfidenceInfo | None
    was_routed: bool
    routed_to: str | None
    route_reason: str | None
    created_at: datetime

    class Config:
        from_attributes = True

class SendMessageResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse
    generation_time_ms: float

class MessageListResponse(BaseModel):
    messages: list[MessageResponse]
    has_more: bool
    total: int
```

---

## Implementation Details

### Files to Create

| File | Purpose |
|------|---------|
| `ai_ready_rag/api/chat.py` | Chat router with all endpoints |

### Files to Modify

| File | Changes |
|------|---------|
| `ai_ready_rag/main.py` | Add `chat.router` to app |

### Service Integration

```python
# In POST /messages endpoint
from ai_ready_rag.services import RAGService, VectorService

# 1. Get user's tags
user_tags = [tag.name for tag in current_user.tags]

# 2. Load chat history
history = db.query(ChatMessage).filter(
    ChatMessage.session_id == session_id
).order_by(ChatMessage.created_at.desc()).limit(10).all()

# 3. Build RAGRequest
from ai_ready_rag.services.rag_service import RAGRequest, ChatMessage as RAGChatMessage
rag_request = RAGRequest(
    query=message.content,
    user_tags=user_tags,
    tenant_id=settings.default_tenant_id,
    chat_history=[RAGChatMessage(role=m.role, content=m.content) for m in reversed(history)],
    model=message.model,
)

# 4. Call RAGService
vector_service = VectorService(settings)
rag_service = RAGService(vector_service, settings)
response = await rag_service.generate(rag_request)

# 5. Store assistant message with response data
assistant_msg = ChatMessage(
    session_id=session_id,
    role="assistant",
    content=response.answer,
    sources=json.dumps([c.__dict__ for c in response.citations]),
    confidence=response.confidence.overall,
    was_routed=(response.action == "ROUTE"),
    routed_to=response.route_to.email if response.route_to else None,
    route_reason=response.route_to.reason if response.route_to else None,
)
```

---

## Access Control

| Endpoint | Auth | Ownership Check |
|----------|------|-----------------|
| POST /sessions | `get_current_user` | N/A (creates for user) |
| GET /sessions | `get_current_user` | Filter by user_id |
| GET /sessions/{id} | `get_current_user` | Verify user_id matches |
| PATCH /sessions/{id} | `get_current_user` | Verify user_id matches |
| POST /sessions/{id}/messages | `get_current_user` | Verify session.user_id matches |
| GET /sessions/{id}/messages | `get_current_user` | Verify session.user_id matches |

---

## Error Handling

| Status | Condition | Response |
|--------|-----------|----------|
| 400 | Empty/invalid content | `{"detail": "Message content required"}` |
| 400 | Content too long | `{"detail": "Message exceeds 4000 characters"}` |
| 401 | Not authenticated | `{"detail": "Not authenticated"}` |
| 404 | Session not found | `{"detail": "Session not found"}` |
| 503 | Ollama unavailable | `{"detail": "AI service temporarily unavailable"}` |

---

## Implementation Issues

### Issue 017: Chat Session Endpoints (TRIVIAL)

**Scope**: Create, list, get, update session endpoints

**Files**:
- Create: `ai_ready_rag/api/chat.py`
- Modify: `ai_ready_rag/main.py`

**Acceptance Criteria**:
- [ ] POST /api/chat/sessions creates session
- [ ] GET /api/chat/sessions lists user's sessions
- [ ] GET /api/chat/sessions/{id} returns session
- [ ] PATCH /api/chat/sessions/{id} updates title/archived
- [ ] All endpoints enforce user ownership
- [ ] Tests pass

---

### Issue 018: Send Message Endpoint (MODERATE)

**Scope**: POST /messages with RAGService integration

**Files**:
- Modify: `ai_ready_rag/api/chat.py`

**Acceptance Criteria**:
- [ ] POST /api/chat/sessions/{id}/messages works
- [ ] Creates user message in DB
- [ ] Calls RAGService.generate() with correct params
- [ ] Creates assistant message with citations, confidence, routing
- [ ] Updates session.updated_at
- [ ] Returns both messages
- [ ] Handles Ollama errors gracefully (503)
- [ ] Tests pass

---

### Issue 019: Get Messages Endpoint (TRIVIAL)

**Scope**: GET /messages with pagination

**Files**:
- Modify: `ai_ready_rag/api/chat.py`

**Acceptance Criteria**:
- [ ] GET /api/chat/sessions/{id}/messages works
- [ ] Supports limit, before, after pagination
- [ ] Returns messages in chronological order
- [ ] Enforces session ownership
- [ ] Tests pass

---

### Issue 020: Chat API Tests (SIMPLE)

**Scope**: Test suite for all chat endpoints

**Files**:
- Create: `tests/test_chat_api.py`

**Acceptance Criteria**:
- [ ] Test session CRUD (create, list, get, update)
- [ ] Test message send with mocked RAGService
- [ ] Test message list with pagination
- [ ] Test ownership enforcement (can't access other user's sessions)
- [ ] Test error cases (404, 400, 503)
- [ ] All tests pass

---

## Acceptance Criteria

- [ ] All 4 core endpoints implemented (create session, list sessions, send message, get messages)
- [ ] Session ownership enforced on all endpoints
- [ ] RAGService integration working with citations and confidence
- [ ] Routed messages show routing info
- [ ] Pagination working on list endpoints
- [ ] Error handling for invalid input and service failures
- [ ] 20+ tests covering happy path and error cases
- [ ] `ruff check` passes
- [ ] All existing tests still pass

---

## Open Questions

1. **Session title auto-generation**: Should we auto-generate title from first message if not provided?
2. **Message streaming**: Future enhancement for SSE/WebSocket streaming?
3. **Session deletion**: Hard delete or just archive?

---

## Next Steps

1. Review this spec
2. Run `/orchestrate 017` to start implementation
3. Issues: 017 → 018 → 019 → 020
