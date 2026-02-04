---
title: Chat Window Streaming & Reliability
status: draft
created: 2026-02-02
author: codex
type: Enhancement
complexity: MEDIUM
stack: frontend + backend
---

# Chat Window Streaming & Reliability

## Summary
Add real‑time WebSocket streaming, accurate connection status, resilient retry/backoff, and message pagination to the React chat view. This aligns the UI with `react-ui-v1.md` and improves reliability for long sessions.

## Goals
1. Stream assistant responses token‑by‑token over WebSocket.
2. Show real connection state (connected/reconnecting/offline).
3. Support resilient retries/backoff for API and WS failures.
4. Paginate messages and sessions to avoid large in‑memory payloads.
5. Preserve existing REST endpoints for backward compatibility.

## Non‑Goals
- Replace the DB schema for chat messages.
- Implement multi‑device sync beyond existing backend behavior.
- Change LLM/rag behavior or message content format.

---

## Current Gaps (from review)
- No WS streaming (HTTP only).
- Connection badge hard‑coded to “Connected”.
- No pagination/infinite scroll beyond limit=50.
- Pending recovery is heuristic; can mis‑clear with multi‑tab usage.

---

## Technical Specification

### 1) WebSocket Streaming

**Backend**
- Implement `/ws/chat` streaming endpoint (if not already present).
- Stream events:
  - `token`: `{ type: "token", content: "..." }`
  - `done`: `{ type: "done", message_id: "...", sources: [...], confidence: 0.94, generation_time_ms: 1234 }`
  - `error`: `{ type: "error", message: "..." }`

**Frontend**
- On send, open WS connection (or reuse an existing one) and stream tokens into the typing message.
- Keep REST `POST /api/chat/sessions/{id}/messages` as fallback when WS unavailable.

**Connection Status**
- Track WS state: `connecting`, `open`, `closed`, `error`.
- Show badge: Connected / Reconnecting / Offline.
- Retry with exponential backoff: 1s, 2s, 4s, 8s (cap 30s).

### 2) Message Pagination

**Backend**
- Ensure existing `/api/chat/sessions/{id}/messages?limit=50&before=<id>` works reliably.
- Add `next_cursor` to response if not present.

**Frontend**
- Add “Load older messages” at top of `MessageList`.
- Append older messages to the start of the list.
- Preserve scroll position when prepending.

### 3) Session Pagination

**Backend**
- Ensure `/api/chat/sessions?limit=50&offset=0` supports pagination.

**Frontend**
- Add “Load more sessions” button in `SessionSidebar` if `has_more`.
- Avoid re-fetching entire list on each load.

### 4) Pending Message Recovery

- Replace heuristic with explicit backend acknowledgement:
  - On send, backend returns a `pending_message_id`.
  - On resume, load messages and clear pending only if that id is resolved.

---

## Data Contract Changes

### WebSocket
- Event stream as above.
- Optional ping/pong or keepalive every 30s.

### REST Additions
- `GET /api/chat/sessions/{id}/messages` returns:
  - `messages: [...]`
  - `next_cursor: <message_id or timestamp>`

---

## UX Requirements
- Connection badge reflects actual status.
- Typing indicator persists while streaming.
- If streaming fails mid‑response, show partial output with error state and retry action.
- For large history, load older messages explicitly.

---

## Files to Modify

**Frontend**
- `frontend/src/views/ChatView.tsx`
- `frontend/src/components/features/chat/MessageList.tsx`
- `frontend/src/stores/chatStore.ts`
- `frontend/src/api/client.ts` (WS client or helper)

**Backend**
- `ai_ready_rag/api/chat.py` (WS endpoint, pagination cursor)
- `ai_ready_rag/services/rag_service.py` (stream tokens)

---

## Acceptance Criteria
- [ ] Assistant responses stream via WS in UI.
- [ ] Connection badge reflects real WS state.
- [ ] WS reconnects with exponential backoff.
- [ ] Messages can load older history without reloading page.
- [ ] Sessions can load more when limit reached.
- [ ] REST fallback works if WS unavailable.
- [ ] Pending recovery uses message id, not heuristics.

---

## Risks
- **Concurrency**: double‑send if WS + REST both fire; ensure single source of truth.
- **Memory**: long sessions; paginate and cap in‑memory list.
- **UX**: maintain scroll position when prepending older messages.

---

## Next Steps
1. Backend WS event stream implementation.
2. Frontend WS client and streaming UI.
3. Pagination API updates.
4. QA with long sessions + network interruption.
