# Gradio UI Specification

| Field | Value |
|-------|-------|
| **Status** | DRAFT |
| **Version** | 1.0 |
| **Created** | 2026-01-28 |
| **Type** | UI Component |
| **Complexity** | MODERATE |

## Summary

Modern chat interface built with Gradio Blocks and Tailwind CSS, mounted on FastAPI at `/app`. Integrates with Chat API for authentication and messaging.

## Goals

- Clean, modern chat UI with Tailwind styling
- Login/logout flow within Gradio
- Session management (New Chat, session list)
- Display AI responses with citations and confidence
- Show routing indicator when confidence is low

## Scope

### In Scope

- Login form (email/password)
- Chat interface with message history
- New Chat button to create sessions
- Session sidebar showing previous chats
- Citation display in responses
- Confidence indicator
- Routing notification (when routed to expert)

### Out of Scope (Phase 2)

- Document upload
- Document management
- Model selection
- Admin features

---

## Wire Diagrams

### Layout Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HEADER BAR                                    │
│  [Logo] AI Ready RAG                              [User] ▼ [Logout] │
├─────────────────┬───────────────────────────────────────────────────┤
│                 │                                                    │
│   SIDEBAR       │              CHAT AREA                            │
│   (250px)       │              (flex-1)                             │
│                 │                                                    │
│ ┌─────────────┐ │  ┌──────────────────────────────────────────────┐ │
│ │ [+ New Chat]│ │  │                                              │ │
│ └─────────────┘ │  │     Welcome! Ask me anything about          │ │
│                 │  │     your documents.                          │ │
│ Recent Chats    │  │                                              │ │
│ ─────────────── │  │  ┌────────────────────────────────────────┐ │ │
│ ▸ Policy Q&A    │  │  │ USER: What is the remote work policy?  │ │ │
│ ▸ HR Questions  │  │  └────────────────────────────────────────┘ │ │
│ ▸ Tech Support  │  │                                              │ │
│                 │  │  ┌────────────────────────────────────────┐ │ │
│                 │  │  │ AI: Based on the documentation...      │ │ │
│                 │  │  │                                        │ │ │
│                 │  │  │ 📄 Sources:                            │ │ │
│                 │  │  │ • HR Policy Manual (p.12)              │ │ │
│                 │  │  │                                        │ │ │
│                 │  │  │ Confidence: ████████░░ 78%             │ │ │
│                 │  │  └────────────────────────────────────────┘ │ │
│                 │  │                                              │ │
│                 │  └──────────────────────────────────────────────┘ │
│                 │                                                    │
│                 │  ┌──────────────────────────────────────────────┐ │
│                 │  │ Type your message...                    [Send]│ │
│                 │  └──────────────────────────────────────────────┘ │
└─────────────────┴───────────────────────────────────────────────────┘
```

### Login Screen (Before Auth)

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│                                                                      │
│                    ┌────────────────────────┐                       │
│                    │                        │                       │
│                    │    🤖 AI Ready RAG     │                       │
│                    │                        │                       │
│                    │  ┌──────────────────┐  │                       │
│                    │  │ Email            │  │                       │
│                    │  └──────────────────┘  │                       │
│                    │                        │                       │
│                    │  ┌──────────────────┐  │                       │
│                    │  │ Password         │  │                       │
│                    │  └──────────────────┘  │                       │
│                    │                        │                       │
│                    │  [      Login       ]  │                       │
│                    │                        │                       │
│                    │  ❌ Invalid credentials│                       │
│                    │                        │                       │
│                    └────────────────────────┘                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Routed Response (Low Confidence)

```
┌────────────────────────────────────────────────────────────────┐
│ AI: I don't have enough information to answer this            │
│ confidently.                                                   │
│                                                                │
│ ⚠️ ROUTED TO EXPERT                                           │
│ This question has been forwarded to: hr-team@company.com      │
│ Reason: Low confidence - insufficient context                 │
│                                                                │
│ Confidence: ██░░░░░░░░ 35%                                    │
└────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Login Component

```
┌─────────────────────────────────┐
│        Login Form               │
├─────────────────────────────────┤
│ State:                          │
│   - is_authenticated: bool      │
│   - error_message: str          │
│   - user_info: dict             │
│                                 │
│ Actions:                        │
│   - Submit → POST /api/auth/    │
│              login              │
│   - On success → hide login,    │
│                  show chat      │
└─────────────────────────────────┘
```

### 2. Sidebar Component

```
┌─────────────────────────────────┐
│        Sidebar                  │
├─────────────────────────────────┤
│ State:                          │
│   - sessions: list[Session]     │
│   - active_session_id: str      │
│                                 │
│ Actions:                        │
│   - New Chat → POST /api/chat/  │
│                sessions         │
│   - Select session → load       │
│                      messages   │
│   - On mount → GET /api/chat/   │
│                sessions         │
└─────────────────────────────────┘
```

### 3. Chat Area Component

```
┌─────────────────────────────────┐
│        Chat Area                │
├─────────────────────────────────┤
│ State:                          │
│   - messages: list[Message]     │
│   - is_loading: bool            │
│   - input_text: str             │
│                                 │
│ Actions:                        │
│   - Send → POST /api/chat/      │
│            sessions/{id}/       │
│            messages             │
│   - Load → GET /api/chat/       │
│            sessions/{id}/       │
│            messages             │
└─────────────────────────────────┘
```

### 4. Message Component

```
┌─────────────────────────────────┐
│        Message Display          │
├─────────────────────────────────┤
│ Props:                          │
│   - role: "user" | "assistant"  │
│   - content: str                │
│   - sources: list[Source]       │
│   - confidence: ConfidenceInfo  │
│   - was_routed: bool            │
│   - routed_to: str              │
│                                 │
│ Renders:                        │
│   - Message bubble (styled by   │
│     role)                       │
│   - Citations section (if       │
│     sources)                    │
│   - Confidence bar (if          │
│     assistant)                  │
│   - Routing alert (if routed)   │
└─────────────────────────────────┘
```

---

## Data Flow

### Authentication Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Login   │     │  Gradio  │     │ FastAPI  │     │   DB     │
│  Form    │     │   App    │     │ /auth    │     │          │
└────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │                │
     │ Submit         │                │                │
     │───────────────>│                │                │
     │                │ POST /login    │                │
     │                │───────────────>│                │
     │                │                │ Query user     │
     │                │                │───────────────>│
     │                │                │<───────────────│
     │                │ JWT + cookie   │                │
     │                │<───────────────│                │
     │ Show chat UI   │                │                │
     │<───────────────│                │                │
     │                │                │                │
```

### Message Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Chat    │     │  Gradio  │     │ FastAPI  │     │   RAG    │     │  Ollama  │
│  Input   │     │   App    │     │ /chat    │     │ Service  │     │          │
└────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │                │                │
     │ Send msg       │                │                │                │
     │───────────────>│                │                │                │
     │                │ POST /messages │                │                │
     │                │───────────────>│                │                │
     │                │                │ generate()     │                │
     │                │                │───────────────>│                │
     │                │                │                │ LLM call       │
     │                │                │                │───────────────>│
     │                │                │                │<───────────────│
     │                │                │ RAGResponse    │                │
     │                │<───────────────│<───────────────│                │
     │ Display msg    │                │                │                │
     │<───────────────│                │                │                │
     │                │                │                │                │
```

---

## Tailwind Styling

### Color Palette

```css
/* Primary */
--primary-50:  #eff6ff;   /* Light blue bg */
--primary-500: #3b82f6;   /* Blue accent */
--primary-700: #1d4ed8;   /* Blue hover */

/* Neutral */
--gray-50:  #f9fafb;      /* Page bg */
--gray-100: #f3f4f6;      /* Card bg */
--gray-200: #e5e7eb;      /* Borders */
--gray-700: #374151;      /* Text */
--gray-900: #111827;      /* Headings */

/* Semantic */
--success: #10b981;       /* Green - high confidence */
--warning: #f59e0b;       /* Amber - medium confidence */
--error:   #ef4444;       /* Red - low confidence/routing */
```

### Key Classes

```python
# Sidebar
sidebar_classes = "w-64 bg-gray-100 border-r border-gray-200 p-4"

# Chat container
chat_classes = "flex-1 flex flex-col bg-white"

# Message bubbles
user_msg_classes = "bg-primary-500 text-white rounded-lg p-3 ml-auto max-w-[80%]"
ai_msg_classes = "bg-gray-100 text-gray-900 rounded-lg p-3 mr-auto max-w-[80%]"

# Confidence bar
confidence_bar = "h-2 rounded-full bg-gray-200"
confidence_fill_high = "bg-success"      # >= 70
confidence_fill_med = "bg-warning"       # 40-69
confidence_fill_low = "bg-error"         # < 40

# Routing alert
routing_alert = "bg-error/10 border border-error/20 text-error rounded-lg p-3 mt-2"
```

---

## API Integration

### Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/auth/login` | POST | Authenticate user |
| `/api/auth/logout` | POST | Clear session |
| `/api/auth/me` | GET | Get current user |
| `/api/chat/sessions` | GET | List user sessions |
| `/api/chat/sessions` | POST | Create new session |
| `/api/chat/sessions/{id}/messages` | GET | Load chat history |
| `/api/chat/sessions/{id}/messages` | POST | Send message |

### HTTP Client Setup

```python
import httpx

# Use httpx with cookie persistence
client = httpx.Client(
    base_url="http://localhost:8000",
    cookies=httpx.Cookies(),
    timeout=60.0,  # RAG can be slow
)

# After login, cookies are automatically sent
def login(email: str, password: str) -> dict:
    response = client.post("/api/auth/login", json={
        "email": email,
        "password": password
    })
    return response.json()
```

---

## Implementation Plan

### Files to Create

| File | Purpose |
|------|---------|
| `ai_ready_rag/ui/__init__.py` | Package init |
| `ai_ready_rag/ui/gradio_app.py` | Main Gradio app |
| `ai_ready_rag/ui/components.py` | Reusable UI components |
| `ai_ready_rag/ui/api_client.py` | HTTP client for Chat API |

### Files to Modify

| File | Changes |
|------|---------|
| `ai_ready_rag/main.py` | Mount Gradio at `/app` |
| `ai_ready_rag/config.py` | Set `enable_gradio: true` |

---

## Implementation Issues

### Issue 021: Gradio App Structure (TRIVIAL)

**Scope**: Create UI package with app skeleton

**Files**:
- Create: `ai_ready_rag/ui/__init__.py`
- Create: `ai_ready_rag/ui/gradio_app.py`
- Create: `ai_ready_rag/ui/api_client.py`

**Acceptance Criteria**:
- [ ] Gradio Blocks app created
- [ ] Login form renders
- [ ] API client can make requests

---

### Issue 022: Login and Session Management (SIMPLE)

**Scope**: Implement login flow and session sidebar

**Files**:
- Modify: `ai_ready_rag/ui/gradio_app.py`

**Acceptance Criteria**:
- [ ] Login form authenticates via API
- [ ] Session list loads after login
- [ ] New Chat creates session
- [ ] Clicking session loads messages

---

### Issue 023: Chat Interface (SIMPLE)

**Scope**: Implement chat message display and sending

**Files**:
- Modify: `ai_ready_rag/ui/gradio_app.py`

**Acceptance Criteria**:
- [ ] Messages display in chat area
- [ ] User can send messages
- [ ] AI responses show citations
- [ ] Confidence bar displays
- [ ] Routing alert shows when routed

---

### Issue 024: Mount on FastAPI (TRIVIAL)

**Scope**: Mount Gradio app on FastAPI

**Files**:
- Modify: `ai_ready_rag/main.py`

**Acceptance Criteria**:
- [ ] Gradio accessible at `/app`
- [ ] Feature flag controls mounting
- [ ] Static files served correctly

---

## Acceptance Criteria

- [ ] Login works with existing users
- [ ] Can create new chat sessions
- [ ] Can send messages and receive AI responses
- [ ] Citations display with source info
- [ ] Confidence indicator shows score
- [ ] Routed messages show alert
- [ ] Session history persists
- [ ] Tailwind styling applied
- [ ] Mounted at `/app` on FastAPI
- [ ] Works with `enable_gradio: true`

---

## Open Questions

1. **Session titles**: Auto-generate from first message, or manual?
2. **Message streaming**: Add later with SSE?
3. **Dark mode**: Support theme toggle?

---

## Next Steps

1. Review this spec
2. Run `/orchestrate 021 through 024`
