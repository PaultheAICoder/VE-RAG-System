# AI Ready RAG -- DGX Spark Manual Test Plan

**Application URL:** http://172.16.20.51:8502
**Profile:** spark
**Date:** 2026-02-18
**Default Credentials:** admin@test.com / npassword2002

---

## Table of Contents

1. [Pre-Flight / Health Check](#1-pre-flight--health-check)
2. [Setup Wizard (First Run)](#2-setup-wizard-first-run)
3. [Authentication](#3-authentication)
4. [User Management](#4-user-management)
5. [Tag Management](#5-tag-management)
6. [Document Upload -- Single File](#6-document-upload--single-file)
7. [Document Upload -- Batch / Folder](#7-document-upload--batch--folder)
8. [Auto-Tagging](#8-auto-tagging)
9. [Document Management](#9-document-management)
10. [Forms Templates](#10-forms-templates)
11. [Chat / Q&A](#11-chat--qa)
12. [Hybrid Search](#12-hybrid-search)
13. [Admin Settings](#13-admin-settings)
14. [Evaluation System](#14-evaluation-system)
15. [RAG Quality Dashboard](#15-rag-quality-dashboard)
16. [System Health Dashboard](#16-system-health-dashboard)
17. [Edge Cases & Negative Tests](#17-edge-cases--negative-tests)

---

## 1. Pre-Flight / Health Check

**Prerequisites:** DGX Spark powered on, services started (Ollama, Qdrant, FastAPI).

| # | Step | Expected Result |
|---|------|-----------------|
| 1.1 | Open browser, navigate to `http://172.16.20.51:8502/health` | JSON response with `"status": "healthy"`, `"profile": "spark"` |
| 1.2 | Verify `backends.vector` is `"qdrant"` | Qdrant is the vector backend |
| 1.3 | Verify `backends.chunker` is `"docling"` | Docling chunker active |
| 1.4 | Verify `forms.enabled` is `true`, `templates_approved` is 2 | ACORD 24 and ACORD 25 templates present |
| 1.5 | Verify `auto_tagging.enabled` is `true`, `strategy` is `"insurance_agency"` | Insurance agency auto-tag strategy loaded |
| 1.6 | Verify `auto_tagging.path_enabled` is `true`, `auto_tagging.llm_enabled` is `false` | Path-based tagging, no LLM tagging |
| 1.7 | Verify `hybrid_search.enabled` is `true`, `active_mode` is `"hybrid"` | Hybrid search (dense + BM25) active |
| 1.8 | Verify `evaluation.enabled` is `true` | Evaluation framework ready |
| 1.9 | Navigate to `http://172.16.20.51:8502/version` | Returns app name and version number |

---

## 2. Setup Wizard (First Run)

**Prerequisites:** Fresh install with no prior setup completion, or database reset.

| # | Step | Expected Result |
|---|------|-----------------|
| 2.1 | Navigate to `http://172.16.20.51:8502` | Redirected to `/login` page |
| 2.2 | **[DEMO]** Click "Demo Login (Admin)" button | Login succeeds; if setup wizard is not yet complete, the UI redirects to the setup wizard |
| 2.3 | On setup wizard, verify "Change default password" prompt appears | Current password field, new password field, confirm field shown |
| 2.4 | Enter wrong current password, submit | Error: "Current password is incorrect" |
| 2.5 | Enter correct current password but same password as new | Error: "New password must be different from current password" |
| 2.6 | **[DEMO]** Enter correct current password and a new strong password, submit | Success: "Setup completed successfully. Default password has been changed." |
| 2.7 | Verify `GET /api/setup/status` returns `setup_complete: true` | Setup marked complete, wizard will not appear again |

> **Note:** If setup was previously completed, skip to Section 3. The setup wizard only appears once.

---

## 3. Authentication

**Prerequisites:** Setup complete, admin user exists.

| # | Step | Expected Result |
|---|------|-----------------|
| 3.1 | Navigate to `http://172.16.20.51:8502/login` | Login page renders with "AI Ready RAG" heading |
| 3.2 | **[DEMO]** Click "Demo Login (Admin)" or enter `admin@test.com` / your password | Login succeeds, redirect to `/chat` |
| 3.3 | Verify sidebar navigation shows: Chat, Documents, Tags, Users, RAG Quality, Settings, Health | All navigation items visible for admin |
| 3.4 | Open browser DevTools > Application > Cookies | `access_token` cookie present, httpOnly |
| 3.5 | Call `GET /api/auth/me` (with auth header) | Returns user object with `role: "admin"`, email, display_name |
| 3.6 | Click Logout (or call `POST /api/auth/logout`) | Cookie cleared, redirect to login page |
| 3.7 | Try to access `/chat` without logging in | Redirected to `/login` |
| 3.8 | **Negative:** Login with wrong password | Error: "Invalid email or password" (HTTP 401) |
| 3.9 | **Negative:** Login with non-existent email | Error: "Invalid email or password" (HTTP 401) |
| 3.10 | **Negative:** Login with deactivated user account | Error: "User account is deactivated" (HTTP 403) |

---

## 4. User Management

**Prerequisites:** Logged in as admin.

| # | Step | Expected Result |
|---|------|-----------------|
| 4.1 | Navigate to `/users` | Users list displays; admin@test.com visible |
| 4.2 | **[DEMO]** Create new user: email=`sarah@agency.com`, display_name="Sarah Johnson", role="user", password="TempPass123!" | User created (HTTP 201), appears in list |
| 4.3 | Create second user: email=`mike@agency.com`, display_name="Mike Williams", role="user", password="TempPass456!" | User created |
| 4.4 | Verify user list shows 3 users (admin + 2 new) | Count is 3 |
| 4.5 | Click on Sarah's user entry, verify detail view | Shows email, display_name, role, created_at, tags=[] |
| 4.6 | Update Sarah's display_name to "Sarah J. Johnson" | Update succeeds, name updated |
| 4.7 | **[DEMO]** Assign tags to Sarah (see Tag Management first) | Tags assigned, shown in user detail |
| 4.8 | Reset Mike's password | Returns `temporary_password`, `must_reset_password: true` |
| 4.9 | Deactivate Mike's account | Message: "User deactivated" |
| 4.10 | Attempt to login as Mike | Error: "User account is deactivated" (HTTP 403) |
| 4.11 | **Negative:** Try to deactivate yourself (admin) | Error: "Cannot deactivate yourself" (HTTP 400) |
| 4.12 | **Negative:** Create user with duplicate email | Error: "Email already registered" (HTTP 400) |

---

## 5. Tag Management

**Prerequisites:** Logged in as admin.

| # | Step | Expected Result |
|---|------|-----------------|
| 5.1 | Navigate to `/tags` | Tag list displays (may be empty on fresh install, or auto-tags may exist) |
| 5.2 | **[DEMO]** Create tag: name=`client:acme-corp`, display_name="ACME Corporation", description="ACME Corp client files" | Tag created (HTTP 201) |
| 5.3 | Create tag: name=`client:beta-inc`, display_name="Beta Inc", description="Beta Inc client files" | Tag created |
| 5.4 | Create tag: name=`doctype:policy`, display_name="Insurance Policy" | Tag created |
| 5.5 | Create tag: name=`doctype:certificate`, display_name="Certificate of Insurance" | Tag created |
| 5.6 | Verify tag list shows all created tags | 4+ tags visible |
| 5.7 | Get tag facets via `/tags` view or `GET /api/tags/facets` | Tags grouped by namespace (client, doctype) with document counts |
| 5.8 | Update ACME tag description to "ACME Corporation - Priority client" | Update succeeds |
| 5.9 | **Negative:** Create tag with duplicate name `client:acme-corp` | Error: "Tag name already exists" (HTTP 400) |
| 5.10 | **Negative:** Delete a system tag (if any exist with `is_system=true`) | Error: "Cannot delete system tags" (HTTP 400) |
| 5.11 | Delete `client:beta-inc` tag (if not system) | Tag deleted successfully |

---

## 6. Document Upload -- Single File

**Prerequisites:** Logged in as admin, tags exist from Section 5.

| # | Step | Expected Result |
|---|------|-----------------|
| 6.1 | Navigate to `/documents` | Document list page displays (empty on fresh install) |
| 6.2 | **[DEMO]** Upload a PDF file (e.g., `Benefits_Guide_2025.pdf`), assign tag `client:acme-corp` | Document created with `status: "pending"`, returns immediately |
| 6.3 | Observe document status transition | Status changes: `pending` -> `processing` -> `ready` |
| 6.4 | **[DEMO]** Once processing completes, verify document detail shows chunk_count > 0 | Chunks created and stored in Qdrant |
| 6.5 | Upload a DOCX file (e.g., `Employee_Handbook_2025.docx`), assign tag `client:acme-corp` | Document uploaded and processed |
| 6.6 | Upload an XLSX file (e.g., `Budget_2025.xlsx`) | Document uploaded and processed (spreadsheet parsing) |
| 6.7 | Upload a scanned PDF with OCR toggle enabled | OCR processes the document, chunks created |
| 6.8 | Verify uploaded documents list shows all uploads | Documents listed with filenames, status, chunk counts |
| 6.9 | **Negative:** Upload an unsupported file type (e.g., .exe) | Error or rejection |
| 6.10 | **Duplicate check:** Upload the same PDF again without replace flag | Duplicate detected, document not created (or deduplicated) |
| 6.11 | Upload same PDF with `replace=true` query parameter | Existing document replaced |

---

## 7. Document Upload -- Batch / Folder

**Prerequisites:** Logged in as admin, multiple test files available.

| # | Step | Expected Result |
|---|------|-----------------|
| 7.1 | **[DEMO]** Use batch upload with 3-5 files simultaneously | Batch upload response shows: `total`, `uploaded`, `duplicates`, `failed` counts |
| 7.2 | Verify each file result has `status: "uploaded"` or duplicate/failed | Per-file results in response |
| 7.3 | Verify background processing starts for all uploaded files | All documents transition from pending to processing to ready |
| 7.4 | Upload batch with `source_paths` for each file (simulating folder structure) | Documents created with source_path metadata preserved |
| 7.5 | Upload batch with `auto_tag=true` | Auto-tagging suggestions generated (see Section 8) |
| 7.6 | **Negative:** Upload batch with mismatched `source_paths` length | Error: "source_paths length must match files length" (HTTP 422) |
| 7.7 | Upload batch containing a mix of valid and duplicate files | Partial success: valid files uploaded, duplicates reported |

---

## 8. Auto-Tagging

**Prerequisites:** Logged in as admin, documents uploaded with source paths or auto_tag=true.

| # | Step | Expected Result |
|---|------|-----------------|
| 8.1 | Upload a document with `auto_tag=true` and a source_path like `/clients/acme-corp/policies/policy.pdf` | Document uploaded, tag suggestions generated based on path rules |
| 8.2 | **[DEMO]** View tag suggestions for the document: `GET /api/documents/{doc_id}/tag-suggestions` | Returns suggestions with `status: "pending"`, tag_name like `client:acme-corp` |
| 8.3 | Verify suggestion includes namespace, display_name, strategy_id | Auto-tag metadata present |
| 8.4 | **[DEMO]** Approve selected suggestions: `POST /api/documents/{doc_id}/tag-suggestions/approve` | Suggestions approved, tags created and assigned to document |
| 8.5 | Verify approved tags appear on the document | Document tags include the approved auto-tags |
| 8.6 | Verify vector store tags updated (no re-embedding needed) | Vector payloads updated via set_payload |
| 8.7 | Reject a suggestion: `POST /api/documents/{doc_id}/tag-suggestions/reject` | Suggestion status changed to "rejected" |
| 8.8 | Batch approve suggestions across multiple documents | `POST /api/documents/tag-suggestions/approve-batch` succeeds |
| 8.9 | **Negative:** Approve an already-approved suggestion | Returns `status: "already_processed"` |
| 8.10 | View active strategy: `GET /api/admin/auto-tagging/active` | Returns `insurance_agency` strategy details |
| 8.11 | View strategy list: `GET /api/admin/auto-tagging/strategies` | Lists available strategies |

---

## 9. Document Management

**Prerequisites:** Logged in as admin, documents uploaded from Sections 6-7.

| # | Step | Expected Result |
|---|------|-----------------|
| 9.1 | List documents with pagination: `GET /api/documents?limit=5&offset=0` | Returns first 5 documents with total count |
| 9.2 | Filter by status: `GET /api/documents?status=ready` | Only "ready" documents returned |
| 9.3 | Filter by tag: `GET /api/documents?tag_prefix=client:` | Only documents with client tags returned |
| 9.4 | Search by filename: `GET /api/documents?search=budget` | Only matching documents returned |
| 9.5 | **[DEMO]** Update document tags (add/change tags) | Tags updated in both SQLite and Qdrant |
| 9.6 | Reprocess a document: `POST /api/documents/{doc_id}/reprocess` | Status resets to "pending", reprocessing queued (HTTP 202) |
| 9.7 | Bulk reprocess multiple documents | Returns `queued` count and `skipped` IDs |
| 9.8 | Delete a document | Document, vectors, and file all removed (HTTP 204) |
| 9.9 | Bulk delete documents | Returns per-document results with `deleted_count` |
| 9.10 | Check duplicates before upload: `POST /api/documents/check-duplicates` | Returns lists of duplicate and unique filenames |
| 9.11 | **Negative:** Reprocess a document in "processing" status | Skipped (status must be "ready" or "failed") |
| 9.12 | **Negative:** Delete a non-existent document ID | Error: "Document not found" (HTTP 404) |
| 9.13 | **Negative:** Update tags with empty tag_ids list | Error: "At least one tag is required" (HTTP 400) |

---

## 10. Forms Templates

**Prerequisites:** Logged in as admin. Forms feature enabled, 2 approved templates exist (ACORD 24, ACORD 25).

| # | Step | Expected Result |
|---|------|-----------------|
| 10.1 | List templates: `GET /api/forms/templates` | Returns 2 approved templates (ACORD 24, ACORD 25) |
| 10.2 | **[DEMO]** Get ACORD 25 template details | Shows template_id, name, description, fields list with field_name, field_type, field_label |
| 10.3 | Verify fields include standard ACORD 25 fields | Fields like producer name, insured name, policy number, etc. |
| 10.4 | **[DEMO]** Preview extraction: Upload a filled ACORD 25 PDF against the ACORD 25 template | Returns extracted field values with confidence scores per field |
| 10.5 | Verify extraction results include `overall_confidence` and `extraction_method` | Confidence percentage and method (e.g., "pdf_widget", "ocr") returned |
| 10.6 | Preview extraction with ACORD 24 template and matching PDF | Extracted fields for Certificate of Property Insurance |
| 10.7 | Create a new template (draft): provide name, description, source_format="pdf", fields | Template created with `status: "draft"` |
| 10.8 | Approve the draft template | Status changes to `"approved"` |
| 10.9 | Delete (archive) a template | Template soft-deleted (HTTP 204) |
| 10.10 | **Negative:** Non-admin user lists templates | Only approved templates returned, fields stripped |
| 10.11 | **Negative:** Non-admin accesses a draft template by ID | Error: "Template not found" (HTTP 404) |

---

## 11. Chat / Q&A

**Prerequisites:** Logged in as admin (or user with tags), documents uploaded and processed (status=ready).

| # | Step | Expected Result |
|---|------|-----------------|
| 11.1 | Navigate to `/chat` | Chat interface displays, empty or with previous sessions listed |
| 11.2 | **[DEMO]** Create a new chat session | Session created, blank conversation view |
| 11.3 | **[DEMO]** Send a question about uploaded content: e.g., "What is the PTO policy?" | AI responds with answer, citations (source documents), and confidence score |
| 11.4 | **[DEMO]** Verify response includes `sources` array with document names and snippets | Citations link back to source documents |
| 11.5 | **[DEMO]** Verify confidence breakdown: overall, retrieval, coverage, llm scores | Confidence info displayed (e.g., overall: 85, retrieval: 90, coverage: 80, llm: 85) |
| 11.6 | Verify `generation_time_ms` is returned | Response time in milliseconds |
| 11.7 | Send a follow-up question in the same session | Context from previous messages used (chat history loaded) |
| 11.8 | Verify session title and `updated_at` are refreshed | Session metadata updated |
| 11.9 | List all sessions | Sessions listed with message counts and last message preview |
| 11.10 | Rename a session | Title updated |
| 11.11 | Archive a session | Session marked as archived, hidden from default list |
| 11.12 | List sessions with `archived=true` | Archived sessions included |
| 11.13 | View messages with pagination: `GET /api/chat/sessions/{id}/messages?limit=10` | Messages returned in chronological order with `has_more` flag |
| 11.14 | **[DEMO]** Ask a question that triggers ROUTE (low confidence) | Response includes `was_routed: true`, `routed_to` with owner email, `route_reason` |
| 11.15 | **[DEMO]** Ask about something not in any uploaded documents | Low confidence response, possibly routed to human |
| 11.16 | Delete a session (admin only) | Session and all messages deleted, audit log created |
| 11.17 | Bulk delete sessions | Multiple sessions deleted, returns count |
| 11.18 | **Negative:** Access another user's session | Error: "Session not found" (HTTP 404) |
| 11.19 | **Negative:** Send message to non-existent session | Error: "Session not found" (HTTP 404) |

---

## 12. Hybrid Search

**Prerequisites:** Logged in as admin, documents uploaded and processed, hybrid search enabled.

| # | Step | Expected Result |
|---|------|-----------------|
| 12.1 | Verify hybrid search active: check `/health` endpoint `hybrid_search.active_mode` is `"hybrid"` | Mode is "hybrid" (dense + BM25 sparse) |
| 12.2 | **[DEMO]** In chat, ask a specific keyword-heavy question: e.g., "ACORD 25 certificate number for ACME policy" | Hybrid search returns results combining semantic and keyword matching |
| 12.3 | Compare response quality: ask same question about a concept vs. specific term | Keyword queries benefit from BM25, conceptual queries from dense |
| 12.4 | Verify retrieval settings via `GET /api/admin/settings/retrieval` | Shows hybrid_enabled, min_score_dense, fusion parameters |
| 12.5 | **[DEMO]** Toggle hybrid search off via `PUT /api/admin/settings/retrieval` with `hybrid_enabled: false` | Setting saved |
| 12.6 | Re-check `/health` endpoint | `hybrid_search.active_mode` changes to `"dense_only"` |
| 12.7 | Re-enable hybrid search | Mode returns to "hybrid" |

---

## 13. Admin Settings

**Prerequisites:** Logged in as admin.

### 13a. Chat Model Settings

| # | Step | Expected Result |
|---|------|-----------------|
| 13a.1 | View available models: `GET /api/admin/models` | Lists available Ollama models (e.g., qwen3:8b, llama3.2, deepseek-r1:32b) |
| 13a.2 | View current chat model setting | Shows active chat model |
| 13a.3 | **[DEMO]** Change chat model: `PATCH /api/admin/models/chat` | Model changed, subsequent chat uses new model |
| 13a.4 | View model limits: `GET /api/admin/model-limits` | Returns context_length, max_tokens for current model |

### 13b. LLM Settings

| # | Step | Expected Result |
|---|------|-----------------|
| 13b.1 | View LLM settings: `GET /api/admin/settings/llm` | Shows temperature, top_p, system prompt, etc. |
| 13b.2 | Update LLM settings (e.g., change temperature) | Settings saved |

### 13c. Retrieval Settings

| # | Step | Expected Result |
|---|------|-----------------|
| 13c.1 | View retrieval settings: `GET /api/admin/settings/retrieval` | Shows top_k, min_score, hybrid_enabled, etc. |
| 13c.2 | **[DEMO]** Modify top_k (e.g., change from 5 to 10) | Setting saved, subsequent queries use new value |

### 13d. Processing Options

| # | Step | Expected Result |
|---|------|-----------------|
| 13d.1 | View processing options: `GET /api/admin/processing-options` | Shows OCR settings, table extraction mode, etc. |
| 13d.2 | Toggle OCR or change table extraction mode | Settings updated |

### 13e. Security Settings

| # | Step | Expected Result |
|---|------|-----------------|
| 13e.1 | View security settings: `GET /api/admin/settings/security` | Shows JWT expiration hours, etc. |
| 13e.2 | Modify JWT expiration hours | Setting saved |

### 13f. Feature Flags

| # | Step | Expected Result |
|---|------|-----------------|
| 13f.1 | View feature flags: `GET /api/admin/settings/feature-flags` | Shows enabled/disabled features |
| 13f.2 | Toggle a feature flag | Flag updated |

### 13g. Knowledge Base Management

| # | Step | Expected Result |
|---|------|-----------------|
| 13g.1 | View KB stats: `GET /api/admin/knowledge-base/stats` | Shows total documents, chunks, file sizes |
| 13g.2 | **[DEMO]** View architecture: `GET /api/admin/architecture` | Returns full architecture info: backends, models, pipeline status |
| 13g.3 | Detailed health: `GET /api/admin/health/detailed` | Component-level health with infrastructure status |

### 13h. Reindex

| # | Step | Expected Result |
|---|------|-----------------|
| 13h.1 | Get reindex estimate: `GET /api/admin/reindex/estimate` | Shows estimated time and document count |
| 13h.2 | Start reindex job | Reindex begins in background |
| 13h.3 | Check reindex status | Shows progress, completed/total counts |
| 13h.4 | View reindex history | Lists past reindex jobs |

### 13i. Cache Management

| # | Step | Expected Result |
|---|------|-----------------|
| 13i.1 | View cache stats: `GET /api/admin/cache/stats` | Shows hit/miss rates, size |
| 13i.2 | View cache settings | Shows TTL, max size |
| 13i.3 | Clear cache | Cache cleared |
| 13i.4 | Seed cache with queries | Cache seeded with provided queries |

### 13j. Synonyms & Curated Q&A

| # | Step | Expected Result |
|---|------|-----------------|
| 13j.1 | List synonyms: `GET /api/admin/synonyms` | Empty or existing synonyms |
| 13j.2 | Create synonym: e.g., "COI" -> "Certificate of Insurance" | Synonym created |
| 13j.3 | Create curated Q&A: question + answer pair | Q&A created with keywords |
| 13j.4 | Invalidate synonym/QA cache | Cache invalidated |

### 13k. Warming Queue

| # | Step | Expected Result |
|---|------|-----------------|
| 13k.1 | View warming queue: `GET /api/admin/warming/queue` | Lists warming jobs |
| 13k.2 | Submit manual warming job | Job queued |
| 13k.3 | View warming progress | Progress reported |

---

## 14. Evaluation System

**Prerequisites:** Logged in as admin, documents uploaded and processed.

| # | Step | Expected Result |
|---|------|-----------------|
| 14.1 | Navigate to RAG Quality view or use API | Evaluation dashboard |
| 14.2 | **[DEMO]** Create a manual evaluation dataset with Q&A pairs | Dataset created with sample_count |
| 14.3 | Add samples with `question` and `ground_truth` fields | Samples stored |
| 14.4 | View dataset samples | Lists questions and ground truth answers |
| 14.5 | **[DEMO]** Create an evaluation run from the dataset | Run created with `status: "pending"` (HTTP 201) |
| 14.6 | Monitor run progress: `GET /api/evaluations/runs/{run_id}` | Shows status, completed_samples, total_samples, ETA |
| 14.7 | Wait for run completion | Status changes to "completed" (or "completed_with_errors") |
| 14.8 | **[DEMO]** View run results: sample-level scores | Shows per-sample faithfulness, answer_relevancy, retrieval scores |
| 14.9 | View evaluation summary: `GET /api/evaluations/summary` | Dashboard summary with avg_scores, score_trend, totals |
| 14.10 | View live monitoring stats: `GET /api/evaluations/live/stats` | Shows total_scores, avg_faithfulness, hourly breakdown |
| 14.11 | View live scores: `GET /api/evaluations/live/scores` | Paginated live evaluation scores |
| 14.12 | Import RAGBench dataset: `POST /api/evaluations/datasets/import-ragbench` | Dataset imported from parquet files |
| 14.13 | Generate synthetic dataset from documents | Synthetic dataset created (async, HTTP 202) |
| 14.14 | Cancel a running evaluation | `is_cancel_requested: true`, run eventually stops |
| 14.15 | Delete a completed run | Run deleted (HTTP 204) |
| 14.16 | Delete a dataset (no active runs) | Dataset and samples deleted |
| 14.17 | **Negative:** Delete a dataset with active runs | Error: "Cannot delete dataset with existing evaluation runs" (HTTP 409) |
| 14.18 | **Negative:** Delete an active (pending/running) run | Error: "Cannot delete an active run. Cancel it first." (HTTP 409) |

---

## 15. RAG Quality Dashboard

**Prerequisites:** Logged in as admin, at least one evaluation run completed.

| # | Step | Expected Result |
|---|------|-----------------|
| 15.1 | Navigate to `/rag-quality` | RAG Quality dashboard renders |
| 15.2 | **[DEMO]** Verify summary section: latest run info, average scores | Displays faithfulness, answer relevancy averages |
| 15.3 | Verify score trend chart | Shows scores over time across evaluation runs |
| 15.4 | View dataset list within the dashboard | Datasets listed with sample counts |
| 15.5 | View run list with status indicators | Runs shown with status badges |

---

## 16. System Health Dashboard

**Prerequisites:** Logged in as admin.

| # | Step | Expected Result |
|---|------|-----------------|
| 16.1 | Navigate to `/health` (frontend view) | System health dashboard renders |
| 16.2 | **[DEMO]** Verify all components show green/healthy status | Ollama, Qdrant, SQLite, forms, auto-tagging all healthy |
| 16.3 | Verify Ollama models listed | Available models shown |
| 16.4 | Verify Qdrant collection info | Collection name, vector count |
| 16.5 | Verify forms status | Enabled, 2 approved templates |

---

## 17. Edge Cases & Negative Tests

### 17a. Access Control

| # | Step | Expected Result |
|---|------|-----------------|
| 17a.1 | Login as a non-admin user (Sarah from Section 4) | Login succeeds, limited UI (no admin features) |
| 17a.2 | **[DEMO]** As Sarah (with `client:acme-corp` tag), ask about ACME documents | Retrieves only ACME-tagged documents, answers based on accessible content |
| 17a.3 | **[DEMO]** As Sarah, ask about documents tagged with `client:beta-inc` (no access) | No relevant documents found, low confidence or "I don't have information" |
| 17a.4 | As Sarah, try to upload a document | Error: admin role required |
| 17a.5 | As Sarah, try to access `/api/users` | Error: admin required (HTTP 403) |
| 17a.6 | As Sarah, try to access `/api/admin/settings/retrieval` | Error: admin required (HTTP 403) |
| 17a.7 | As Sarah, list documents | Only documents matching her tags shown |
| 17a.8 | As Sarah, list form templates | Only approved templates shown, fields stripped |

### 17b. API Edge Cases

| # | Step | Expected Result |
|---|------|-----------------|
| 17b.1 | Send request without auth token | HTTP 401 Unauthorized |
| 17b.2 | Send request with expired token | HTTP 401 Unauthorized |
| 17b.3 | Send request with malformed JSON body | HTTP 422 Unprocessable Entity |
| 17b.4 | Upload zero-byte file | Handled gracefully (error or processed as empty) |
| 17b.5 | Send very long chat message (>10,000 chars) | Handled (processed or truncated, no server crash) |
| 17b.6 | Create session with very long title | Handled gracefully |
| 17b.7 | Use both `before` and `after` cursor in messages endpoint | Error: "Cannot use both 'before' and 'after' cursors simultaneously" (HTTP 400) |
| 17b.8 | Reference invalid message ID in cursor pagination | Error: "Invalid cursor: message not found" (HTTP 400) |

### 17c. Document Processing Edge Cases

| # | Step | Expected Result |
|---|------|-----------------|
| 17c.1 | Upload a password-protected PDF | Processing fails gracefully, document status = "failed" with error_message |
| 17c.2 | Upload a very large file (>50MB) | Handled (processed or rejected with size limit error) |
| 17c.3 | Upload an image file (.jpg) with OCR enabled | Image processed via OCR, text extracted |
| 17c.4 | Recover stuck documents: `POST /api/admin/documents/recover-stuck` | Stuck "processing" documents recovered |

### 17d. Concurrent Operations

| # | Step | Expected Result |
|---|------|-----------------|
| 17d.1 | Upload 10 documents simultaneously via batch | Semaphore limits concurrent processing; all eventually complete |
| 17d.2 | Send multiple chat messages rapidly | All messages processed in order, no data corruption |

---

## Demo Highlight Sequence

For a polished demo, execute these steps in order (all marked [DEMO] above):

1. **Health Check** (1.1) -- Show system overview
2. **Login** (3.2) -- Authenticate
3. **System Health Dashboard** (16.2) -- Show all green
4. **Create Tags** (5.2-5.5) -- Set up access control
5. **Create Users** (4.2-4.3) -- Create demo users
6. **Assign Tags to Users** (4.7) -- Wire up access control
7. **Upload Documents** (6.2-6.4) -- Single file with processing
8. **Batch Upload** (7.1) -- Multiple files at once
9. **Auto-Tagging** (8.1-8.4) -- Show intelligent tag suggestions
10. **Chat Q&A** (11.2-11.5) -- Ask questions, show citations and confidence
11. **Hybrid Search** (12.2) -- Show keyword + semantic search
12. **Access Control Demo** (17a.2-17a.3) -- Show tag-based document isolation
13. **Forms Preview** (10.2, 10.4) -- Extract data from ACORD forms
14. **Evaluation Run** (14.2, 14.5, 14.8) -- Show RAG quality measurement
15. **Change Model** (13a.3) -- Show model switching
16. **RAG Quality Dashboard** (15.2) -- Show quality metrics

---

**Total Steps:** 174
**Demo-Worthy Steps:** 32
**Feature Areas:** 17 sections
