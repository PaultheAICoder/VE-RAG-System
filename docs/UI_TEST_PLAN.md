# UI Test Plan - AI Ready RAG

**Version**: 1.0
**Date**: 2026-01-28
**Status**: Draft

## Overview

This test plan covers the Gradio-based UI for AI Ready RAG, including authentication, chat functionality, session management, and document management (admin-only).

---

## Test Environment

### Prerequisites
- FastAPI server running at `http://localhost:8000`
- Ollama running at `http://localhost:11434` with `llama3.2:latest` or configured chat model
- At least one admin user created via `/api/auth/setup`
- At least one regular user created (optional, for role-based tests)
- At least one tag created via `/api/tags`
- Test documents available (.pdf, .docx, .txt, .md)

### Test Data Setup
```bash
# Start server
./start.sh

# Create initial admin (first-time only)
curl -X POST http://localhost:8000/api/auth/setup \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@test.com","password":"SecurePassword123!","display_name":"Test Admin"}'

# Create a tag
curl -X POST http://localhost:8000/api/tags \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"name":"test-tag","display_name":"Test Tag"}'
```

---

## Test Categories

### 1. Authentication Tests

| ID | Test Case | Steps | Expected Result | Priority |
|----|-----------|-------|-----------------|----------|
| AUTH-01 | Login with valid credentials | 1. Navigate to `/app` 2. Enter valid email/password 3. Click Login | Login succeeds, main UI shows | Critical |
| AUTH-02 | Login with invalid password | 1. Enter valid email 2. Enter wrong password 3. Click Login | Error shows "Invalid email or password" | Critical |
| AUTH-03 | Login with non-existent email | 1. Enter non-existent email 2. Enter any password 3. Click Login | Error shows "Invalid email or password" | High |
| AUTH-04 | Login with empty fields | 1. Leave email/password empty 2. Click Login | Error shows "Please enter email and password" | High |
| AUTH-05 | Login via Enter key | 1. Enter credentials 2. Press Enter in password field | Login submits | Medium |
| AUTH-06 | Logout | 1. Login 2. Click Logout button | Returns to login screen, session cleared | Critical |
| AUTH-07 | Session persistence | 1. Login 2. Refresh page | Session should be lost (stateless) | Medium |
| AUTH-08 | User display | 1. Login | Header shows user display name | Low |

### 2. Session Management Tests

| ID | Test Case | Steps | Expected Result | Priority |
|----|-----------|-------|-----------------|----------|
| SESS-01 | Create new session | 1. Login 2. Click "+ New Chat" | New session created, appears in sidebar | Critical |
| SESS-02 | Session list loads on login | 1. Login | Previous sessions listed in sidebar | High |
| SESS-03 | Select session | 1. Have multiple sessions 2. Click on session in list | Messages load for that session | Critical |
| SESS-04 | Session title display | 1. Create session 2. Check sidebar | Shows title or "Chat {id}..." truncated | Medium |
| SESS-05 | Load more sessions | 1. Have >20 sessions 2. Click "Load More" | Additional sessions load | Medium |
| SESS-06 | Empty session state | 1. Login with no sessions | Sidebar shows empty list | Low |
| SESS-07 | Session date display | 1. View sessions | Updated date shows in MM/DD HH:MM format | Low |

### 3. Chat Interface Tests

| ID | Test Case | Steps | Expected Result | Priority |
|----|-----------|-------|-----------------|----------|
| CHAT-01 | Send message | 1. Select session 2. Type message 3. Click Send | Message appears, AI response shows | Critical |
| CHAT-02 | Send via Enter | 1. Select session 2. Type message 3. Press Enter | Message sends | High |
| CHAT-03 | Empty message | 1. Click Send with empty input | Nothing happens, no error | Medium |
| CHAT-04 | No session selected | 1. Don't select session 2. Send message | Error "Please select or create a chat session" | High |
| CHAT-05 | Loading indicator | 1. Send message | "Thinking..." shows while waiting | Medium |
| CHAT-06 | Message history loads | 1. Select existing session | Previous messages display | Critical |
| CHAT-07 | Long message handling | 1. Send very long message (>1000 chars) | Message sends, displays properly | Medium |

### 4. AI Response Display Tests

| ID | Test Case | Steps | Expected Result | Priority |
|----|-----------|-------|-----------------|----------|
| RESP-01 | Basic response | 1. Ask question with context available | Answer displays | Critical |
| RESP-02 | Citations display | 1. Get response with sources | Sources section shows with expandable details | High |
| RESP-03 | Confidence bar - high | 1. Get response with confidence >= 70 | Green confidence bar shows | High |
| RESP-04 | Confidence bar - medium | 1. Get response with confidence 40-69 | Amber confidence bar shows | High |
| RESP-05 | Confidence bar - low | 1. Get response with confidence < 40 | Red confidence bar shows | High |
| RESP-06 | Routing alert | 1. Get response that was routed | Routing alert box shows with expert email | High |
| RESP-07 | No context response | 1. Ask question with no matching docs | "I don't have enough information..." | High |
| RESP-08 | Citation expand/collapse | 1. Click citation summary | Details expand, click again collapses | Medium |

### 5. Document Management Tests (Admin Only)

| ID | Test Case | Steps | Expected Result | Priority |
|----|-----------|-------|-----------------|----------|
| DOC-01 | Admin sees doc management | 1. Login as admin | Document Management accordion visible | High |
| DOC-02 | Non-admin hidden | 1. Login as regular user | Document Management accordion NOT visible | High |
| DOC-03 | Upload document | 1. Select file 2. Select tags 3. Click Upload | Document created, shows in list | Critical |
| DOC-04 | Upload requires tags | 1. Select file 2. Don't select tags 3. Upload | Error shows | High |
| DOC-05 | List documents | 1. Click Refresh List | Documents display in table | High |
| DOC-06 | Delete document | 1. Enter doc ID 2. Click Delete | Document removed from list | Critical |
| DOC-07 | Reprocess document | 1. Enter doc ID for ready/failed doc 2. Click Reprocess | Status changes to pending | Medium |
| DOC-08 | Reprocess pending fails | 1. Enter doc ID for pending doc 2. Click Reprocess | Error shows | Medium |
| DOC-09 | Invalid doc ID delete | 1. Enter non-existent ID 2. Click Delete | Error shows | Low |
| DOC-10 | Tag dropdown populates | 1. Login as admin | Tag dropdown has available tags | High |

### 6. Error Handling Tests

| ID | Test Case | Steps | Expected Result | Priority |
|----|-----------|-------|-----------------|----------|
| ERR-01 | Server unavailable | 1. Stop server 2. Try to login | Connection error displayed | High |
| ERR-02 | Session expired | 1. Login 2. Wait for token expiry 3. Send message | "Session expired" error, prompts re-login | High |
| ERR-03 | Network timeout | 1. Send message to slow model | Loading shows, eventually response or timeout | Medium |
| ERR-04 | API returns 500 | 1. Trigger server error | Generic error message shown | Medium |
| ERR-05 | 403 Forbidden | 1. Try admin action as regular user | "Permission denied" error | High |

### 7. UI/UX Tests

| ID | Test Case | Steps | Expected Result | Priority |
|----|-----------|-------|-----------------|----------|
| UX-01 | Login form focus | 1. Navigate to `/app` | Email field focused | Low |
| UX-02 | Responsive sidebar | 1. Resize window | Sidebar maintains width, chat area flexes | Medium |
| UX-03 | Chat scroll | 1. Send many messages | Chat scrolls to show new messages | Medium |
| UX-04 | Input clears after send | 1. Send message | Input field clears | High |
| UX-05 | Input preserved on error | 1. Trigger send error | Input retains message for retry | Medium |
| UX-06 | Theme consistency | 1. Check all UI elements | Consistent colors, fonts, spacing | Low |
| UX-07 | Loading states | 1. Observe login, session load, message send | All show loading indicators | Medium |

---

## Integration Tests

### End-to-End Flow Tests

| ID | Test Case | Steps | Expected Result |
|----|-----------|-------|-----------------|
| E2E-01 | Full chat flow | 1. Login 2. Create session 3. Ask question 4. View citations 5. Logout | Complete flow works |
| E2E-02 | Multi-session | 1. Login 2. Create 2 sessions 3. Switch between 4. Send messages in each | Sessions isolated correctly |
| E2E-03 | Admin document flow | 1. Login admin 2. Upload doc 3. Wait for ready 4. Ask question about doc | Document searchable in RAG |

---

## Automated Test Recommendations

### Unit Tests (pytest)

```python
# tests/test_ui_components.py
class TestFormatConfidenceBar:
    def test_high_confidence_green()
    def test_medium_confidence_amber()
    def test_low_confidence_red()
    def test_none_confidence_empty()

class TestFormatCitations:
    def test_formats_sources()
    def test_empty_sources_returns_empty()

class TestFormatRoutingAlert:
    def test_routed_shows_alert()
    def test_not_routed_returns_empty()

class TestParseAssistantResponse:
    def test_parses_full_response()
    def test_handles_missing_fields()
```

### API Client Tests (pytest)

```python
# tests/test_api_client.py
class TestGradioAPIClient:
    def test_login_returns_token()
    def test_login_wrong_password_raises()
    def test_get_sessions_paginates()
    def test_send_message_returns_response()
    def test_upload_document()
```

### Integration Tests (playwright/selenium)

```python
# tests/test_ui_integration.py
class TestLoginFlow:
    async def test_login_success(page):
        await page.goto("/app")
        await page.fill("#email", "admin@test.com")
        await page.fill("#password", "password")
        await page.click("text=Login")
        await expect(page.locator("text=Logout")).to_be_visible()

class TestChatFlow:
    async def test_send_message(page, logged_in):
        await page.click("text=+ New Chat")
        await page.fill("[placeholder='Type your message...']", "Hello")
        await page.click("text=Send")
        await expect(page.locator(".chat-messages")).to_contain_text("Hello")
```

---

## Test Execution Checklist

### Pre-Release Checklist

- [ ] All Critical tests pass
- [ ] All High priority tests pass
- [ ] Medium priority tests pass (or documented)
- [ ] No console errors in browser
- [ ] No unhandled API errors
- [ ] Performance acceptable (<3s for chat response)

### Browser Compatibility

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | Latest | Required |
| Firefox | Latest | Recommended |
| Safari | Latest | Optional |
| Edge | Latest | Optional |

---

## Known Issues / Limitations

1. **Page refresh loses session** - Gradio state is not persisted across refreshes (by design)
2. **No streaming responses** - Messages appear all at once (Phase 2 feature)
3. **Upload Document disabled in sidebar** - Phase 2 feature (use Document Management accordion)
4. **No dark mode** - Single theme only

---

## Appendix: Component Reference

### UI Files
- `ai_ready_rag/ui/gradio_app.py` - Main application
- `ai_ready_rag/ui/components.py` - Message formatting
- `ai_ready_rag/ui/api_client.py` - API calls
- `ai_ready_rag/ui/document_components.py` - Document management handlers
- `ai_ready_rag/ui/theme.py` - Styling

### API Endpoints Used
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/auth/login` | POST | Login |
| `/api/auth/logout` | POST | Logout |
| `/api/chat/sessions` | GET/POST | List/create sessions |
| `/api/chat/sessions/{id}/messages` | GET/POST | List/send messages |
| `/api/tags` | GET | List tags |
| `/api/documents` | GET | List documents |
| `/api/documents/upload` | POST | Upload document |
| `/api/documents/{id}` | DELETE | Delete document |
| `/api/documents/{id}/reprocess` | POST | Reprocess document |
