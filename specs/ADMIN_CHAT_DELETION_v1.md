---
title: Admin Chat Deletion
status: draft
created: 2026-02-03
author: admin
type: CRUD
complexity: SIMPLE
stack: fullstack
---

# Admin Chat Deletion

## Summary

Add the ability for administrators to delete chat sessions from the system. This includes single session deletion and bulk deletion capabilities. Deleted sessions and their messages are permanently removed (hard delete).

## Goals

- Allow admins to manage and clean up chat history
- Support both single and bulk deletion
- Maintain audit trail of deletions
- Provide clear UI feedback for delete operations

## Scope

### In Scope

- DELETE endpoint for single chat session
- DELETE endpoint for bulk chat sessions
- Admin-only access control (system_admin or customer_admin)
- Hard delete (permanent removal from database)
- Cascade delete of associated messages
- Frontend UI for delete actions in admin view
- Confirmation dialog before deletion
- Audit logging of deletions

### Out of Scope

- Soft delete / restore functionality (use existing archive feature)
- User self-deletion of chats (separate feature)
- Export before delete
- Scheduled/automated deletion

---

## Technical Specification

### Backend

#### Endpoints

| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| DELETE | `/api/chat/sessions/{session_id}` | Delete single session | require_admin |
| DELETE | `/api/chat/sessions/bulk` | Delete multiple sessions | require_admin |

#### Request/Response Schemas

**DELETE /api/chat/sessions/{session_id}**

Request: None (session_id in path)

Response (200 OK):
```json
{
  "success": true,
  "deleted_session_id": "uuid-string",
  "deleted_messages_count": 15
}
```

Response (404 Not Found):
```json
{
  "detail": "Session not found"
}
```

**DELETE /api/chat/sessions/bulk**

Request:
```json
{
  "session_ids": ["uuid-1", "uuid-2", "uuid-3"]
}
```

Validation:
- Maximum 100 session IDs per request
- Returns 400 Bad Request if exceeded

Response (200 OK):
```json
{
  "success": true,
  "deleted_count": 3,
  "failed_ids": [],
  "total_messages_deleted": 45
}
```

#### Models Affected

| Model | Changes |
|-------|---------|
| ChatSession | No schema changes (existing CASCADE handles messages) |
| ChatMessage | No changes (deleted via CASCADE) |

#### Implementation Notes

1. Use existing `ChatSession` model with `cascade="all, delete-orphan"` on messages relationship
2. Count messages before deletion for response
3. Use `require_admin` dependency for access control
4. Add audit log entry for each deletion

### Frontend

#### Components to Create

| Component | Purpose | Location |
|-----------|---------|----------|
| DeleteSessionButton | Trash icon button for single delete | `components/features/chat/` |
| BulkDeleteButton | Button for multi-select delete | `components/features/admin/` |
| DeleteConfirmModal | Confirmation dialog | `components/ui/` or reuse existing Modal |

#### Components to Reuse

| Component | Location | Usage |
|-----------|----------|-------|
| Modal | `components/ui/Modal.tsx` | Confirmation dialog |
| Button | `components/ui/Button.tsx` | Delete buttons |
| Checkbox | (if exists) | Bulk selection |

#### State Management

| Data | Hook/Store | Cache Key |
|------|------------|-----------|
| Sessions list | Existing chat store | Invalidate on delete |
| Selected sessions | Local component state | N/A |
| Delete loading | Local component state | N/A |

#### UI Locations

1. **Chat Session List (Admin View)**
   - Add delete icon button per session row
   - Add checkbox for bulk selection
   - Add "Delete Selected" button when items selected

2. **Session Sidebar** (optional)
   - Add delete option in session context menu

---

## API Client

Add to `frontend/src/api/chat.ts`:

```typescript
/**
 * Delete a single chat session (admin only).
 */
export async function deleteSession(sessionId: string): Promise<{
  success: boolean;
  deleted_session_id: string;
  deleted_messages_count: number;
}> {
  return apiClient.delete(`/api/chat/sessions/${sessionId}`);
}

/**
 * Bulk delete chat sessions (admin only).
 */
export async function bulkDeleteSessions(sessionIds: string[]): Promise<{
  success: boolean;
  deleted_count: number;
  failed_ids: string[];
  total_messages_deleted: number;
}> {
  return apiClient.delete('/api/chat/sessions/bulk', {
    body: JSON.stringify({ session_ids: sessionIds }),
  });
}
```

---

## Risk Flags

### DATA_LOSS Risk
Hard delete permanently removes data:
- [x] Confirmation dialog required before delete
- [x] Audit log captures who deleted what and when
- [ ] Consider showing message count in confirmation

### ACCESS_CONTROL Risk
Admin-only operation:
- [x] Use `require_admin` dependency
- [x] Verify both system_admin and customer_admin can delete

---

## Acceptance Criteria

- [ ] Admin can delete a single chat session via API
- [ ] Admin can bulk delete multiple chat sessions via API
- [ ] Non-admin users receive 403 Forbidden
- [ ] Deleted sessions are permanently removed from database
- [ ] Associated messages are cascade deleted
- [ ] Delete action is logged in audit_logs table
- [ ] Frontend shows delete button for admins
- [ ] Confirmation dialog appears before deletion
- [ ] Session list refreshes after successful delete
- [ ] Error toast shown if delete fails

---

## Open Questions

1. Should there be a "Delete All" option for a specific user's chats?
2. Should deletion be logged with the deleted content or just metadata?
3. ~~Maximum number of sessions for bulk delete?~~ **Resolved: 100 sessions max**

---

## Implementation Order

1. **Backend** - Add DELETE endpoints to `chat.py`
2. **Backend** - Add audit logging
3. **Frontend** - Add delete button to session list
4. **Frontend** - Add confirmation modal
5. **Frontend** - Add bulk selection UI
6. **Testing** - Unit and integration tests

---

**Next Steps**:
1. Review this spec
2. Resolve open questions
3. Run `/spec-review specs/ADMIN_CHAT_DELETION_v1.md --create-issues`
