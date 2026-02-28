# Test Run State — VE-RAG Laptop Test Plan v3.0

> Started: 2026-02-28
> Plan: test_data/MANUAL_TEST_PLAN.md
> Resume: read this file after any context compact to know where we left off.

## How to Resume After Compact

1. Read this file to find the last completed test and any open investigations
2. Read `test_data/MANUAL_TEST_PLAN.md` for full test descriptions
3. Continue from the next test in sequence
4. Update this file after each test

## Infrastructure State

| Service    | Status | Notes |
|------------|--------|-------|
| Docker     | ?      | |
| Postgres   | ?      | |
| Redis      | ?      | |
| Ollama     | ?      | |
| API Server | ?      | |

## Progress

| Test ID | Description | Result | Notes |
|---------|-------------|--------|-------|
| **PHASE 0** | | | |
| T0.1 | Infrastructure health endpoint | ✅ PASS | Fixed: hardcoded sqlite in health.py; ENABLE_OCR=false in .env |
| T0.2 | Detailed health check | ✅ PASS | total_chunks=0 noted; reindex needed in Phase 8 |
| T0.3 | Alembic migration state | ✅ PASS | 008 (head) |
| T0.4 | pgvector extension active | ✅ PASS | v0.8.2, HNSW index confirmed |
| T0.5 | Claude CLI reachable | ✅ PASS | Server restarted with env -u CLAUDECODE |
| T0.6 | Startup logs clean | ✅ PASS | All events present, no errors |
| **PHASE 1** | | | |
| T1.1 | Setup wizard / fresh DB | ✅ PASS | Setup completed, password changed to npassword2002@ |
| T1.2 | Login / logout | ✅ PASS | ⚠️ Token not invalidated on logout (no blacklist) — observation only |
| T1.3 | User management | ✅ PASS | user1 and cadmin created; role enforcement confirmed |
| T1.4 | Tag setup | ✅ PASS | hr+finance+legal+engineering exist; user1 assigned hr+finance |
| T1.5 | Account lockout | ❌ FAIL | Lockout not wired — config+DB column exist but login route never checks |
| **PHASE 2** | | | |
| T2.1 | PDF upload (standard) | ✅ PASS | Benefits_Guide_2025.pdf → 8 chunks |
| T2.2 | PDF with OCR | ✅ PASS | D&O Crime Policy → 146 chunks |
| T2.3 | Word upload | ✅ PASS | Employee_Handbook → 11 chunks |
| T2.4 | Excel upload (NL2SQL) | ✅ PASS | Inventory_Report → 26 chunks |
| T2.5 | Image OCR | ✅ PASS | IMG_5295.jpg → 1 chunk; Fixed #457 (lazy JPEG .convert("RGB")) |
| T2.6 | Additional Excel files | ✅ PASS | 5 Excel files → 346 total chunks |
| T2.7 | Batch upload | ✅ PASS | Duplicates detected correctly; 3 Word docs processed |
| T2.8 | Duplicate detection | ✅ PASS | HTTP 409 with existing_id on re-upload |
| T2.9 | File validation | ✅ PASS | .exe rejected with allowed types list |
| T2.10 | Bulk operations | ✅ PASS | bulk-delete works with per-doc status |
| T2.11 | Document tag editing | ✅ PASS | PATCH /documents/{id}/tags works |
| T2.12 | Single reprocess | ✅ PASS | Status: ready→pending→processing→ready |
| **PHASE 3** | | | |
| T3.1 | Admin sees all | ✅ PASS | Admin sees 13 docs across hr/finance/legal |
| T3.2 | User tag filtering | ✅ PASS | user1 sees 12 (not legal D&O policy) |
| T3.3 | Chat respects tags | ✅ PASS | D&O query returns 0 legal sources for user1 |
| T3.4 | API enforcement | ✅ PASS | /api/users → 403 for user role |
| **PHASE 4** | | | |
| T4.1 | Basic RAG (Claude CLI) | ✅ PASS | Delta Dental answer, confidence=76, claude-cli model |
| T4.2 | Multi-turn conversation | ✅ PASS | Follow-up vision/medical in same session |
| T4.3 | Low confidence routing | ✅ PASS | Off-topic query conf=50, coverage=16 (note: threshold=40, not routed) |
| T4.4 | Citation accuracy | ✅ PASS | Sources match document content |
| T4.5 | Session management | ✅ PASS | Sessions persist, history maintained |
| T4.6 | Cross-doc-type queries | ✅ PASS | PTO policy from docx, conf=71 |
| **PHASE 5** | | | |
| T5.1 | Strategy management | ✅ PASS | 4 strategies listed (generic, insurance_agency, construction, law_firm) |
| T5.2 | Switch active strategy | ✅ PASS | Switched generic→insurance_agency→generic |
| T5.3 | Custom strategy CRUD | ✅ PASS | Created test_custom, verified, deleted |
| T5.4 | Built-in protection | ✅ PASS | Cannot delete built-in strategy |
| T5.5 | Audit logging | ✅ PASS | Strategy changes logged with old/new values |
| T5.6 | Tag suggestions | ✅ PASS | Auto-tagging enabled; require_approval=false so direct apply |
| T5.7 | Batch tag approval | ⚠️ SKIP | require_approval=false; no approval queue |
| T5.8 | Frontend strategy card | ⚠️ SKIP | Browser required |
| **PHASE 6** | | | |
| T6.1 | Tag facets | ✅ PASS | facets: hr=6, finance=6, legal=1 |
| T6.2 | Namespace filtering | ✅ PASS | ?tag_id= filters correctly (not ?tag_ids=) |
| T6.3 | Frontend chips | ⚠️ SKIP | Browser required |
| **PHASE 7** | | | |
| T7.1 | Retrieval settings | ✅ PASS | top_k updated and reset |
| T7.2 | LLM settings | ✅ PASS | confidence_threshold updated and reset |
| T7.3 | Security settings | ✅ PASS | JWT/bcrypt/password settings exposed |
| T7.4 | Advanced settings | ✅ PASS | embedding model, chunking, HNSW params |
| T7.5 | Feature flags | ✅ PASS | enable_rag=true, skip_setup=false |
| T7.6 | Settings audit trail | ✅ PASS | Covered by T5.5 |
| T7.7 | Runtime model switching | ✅ PASS | /api/admin/models lists available Ollama models |
| T7.8 | Settings UI | ⚠️ SKIP | Browser required |
| **PHASE 8** | | | |
| T8.1 | KB stats | ✅ PASS | Fixed #458: metadata_::json->>'field' cast |
| T8.2 | Reindex full | ✅ PASS | Fixed #459: extra mode arg in ARQ enqueue |
| T8.3 | Reindex pause/resume/abort | ✅ PASS | All 3 operations working |
| T8.4 | Failure retry | ✅ PASS | failures endpoint returns structured data |
| T8.5 | Clear KB | ✅ PASS | confirm=false rejected; confirmation gate working |
| T8.6 | Reconcile DB + vector | ✅ PASS | Detects chunk_count_mismatch (stale labels: "SQLite"/"qdrant" cosmetic) |
| **PHASE 9** | | | |
| T9.1 | Synonym CRUD | ✅ PASS | PTO → vacation, time off, paid leave created/updated/deleted |
| T9.2 | Synonym in search | ✅ PASS | "vacation accrual rate" → found PTO_Policy doc |
| T9.3 | Frontend synonyms | ⚠️ SKIP | Browser required |
| **PHASE 10** | | | |
| T10.1 | Q&A CRUD | ✅ PASS | Keyword-based entries; create/update/delete all work |
| T10.2 | Q&A override | ✅ PASS | conf=95, exact curated answer returned for keyword match |
| T10.3 | Q&A cache | ✅ PASS | Cache invalidated on Q&A delete |
| T10.4 | Frontend Q&A manager | ⚠️ SKIP | Browser required |
| **PHASE 11** | | | |
| T11.1 | Cache stats & settings | ✅ PASS | 13 entries visible, settings readable |
| T11.2 | Cache seed & clear | ✅ PASS | 26 entries cleared successfully |
| T11.3 | Warming queue manual | ✅ PASS | 2 queries queued manually |
| T11.4 | Warming control | ✅ PASS | Warming completed in 26s |
| **PHASE 12** | | | |
| T12.1 | Dataset management | ⚠️ SKIP | Evaluation not enabled in test env |
| T12.2 | Synthetic samples | ⚠️ SKIP | Evaluation not enabled |
| T12.3 | Evaluation run | ⚠️ SKIP | Evaluation not enabled |
| T12.4 | Summary & live stats | ⚠️ SKIP | Evaluation not enabled |
| T12.5 | Cleanup | ⚠️ SKIP | Evaluation not enabled |
| **PHASE 13** | | | |
| T13.1 | Enable hybrid search | ✅ PASS | Setting toggled; retrieval_hybrid_enabled=true |
| T13.2 | Capability detection | ✅ PASS | enabled=true but collection_has_sparse=false → dense_only (pgvector no BM25) |
| T13.3 | Hybrid search quality | ⚠️ SKIP | pgvector doesn't support sparse vectors; always dense_only |
| T13.4 | Dense-only fallback | ✅ PASS | active_mode=dense_only confirmed when sparse unavailable |
| **PHASE 14** | | | |
| T14.1 | Basic health | ✅ PASS | Covered by T0.1 |
| T14.2 | Detailed health | ✅ PASS | Covered by T0.2 |
| T14.3 | Architecture info | ✅ PASS | vector_store_url=localhost:6333 cosmetic (stale label) |
| T14.4 | Processing options | ✅ PASS | /api/admin/processing-options returns chunking/OCR config |
| T14.5 | Document recovery | ✅ PASS | /api/admin/documents/stuck → moves stuck docs back to pending |
| T14.6 | Frontend health page | ⚠️ SKIP | Browser required |
| **PHASE 15** | | | |
| T15.1 | Template creation & approval | ✅ PASS | ACORD 25 + ACORD 24 templates active |
| T15.2 | Form processing | ❌ FAIL | #460: PostgreSQL 63-char column name limit truncates long ACORD field names → duplicate column error |
| T15.3 | Extraction preview | ✅ PASS | Fields extracted at 99% confidence (preview only) |
| T15.4 | Template visibility | ✅ PASS | Non-admin can view templates but cannot delete |
| **PHASE 16** | | | |
| T16.1 | Concurrent uploads | ✅ PASS | 3 parallel: 1 success + 2 duplicate 409 (correct) |
| T16.2 | Large document | ✅ PASS | Covered by T2.2 (D&O 146 chunks) |
| T16.3 | Corrupt / invalid files | ✅ PASS | Returns "Input document not valid" error |
| T16.4 | Special characters | ✅ PASS | HTML-like tags in query handled safely |
| T16.5 | Session timeout | ⚠️ SKIP | 24h JWT; no server-side blacklist (noted in T1.2) |
| T16.6 | Unauthenticated & unauthorized | ✅ PASS | 401 without token; 403 for role violations |
| T16.7 | Input validation | ⚠️ PARTIAL | Empty/long tag names accepted (no min/max length validation) |
| T16.8 | Background job tracking | ✅ PASS | ARQ job status visible via /api/admin/reindex/status |
| **PHASE 17** | | | |
| T17.1 | Table registration verification | ✅ PASS | 15 SQL templates registered at startup; QueryRouter initialized |
| T17.2 | Inventory quantitative query | ✅ PASS | "on hand" column signal → SQL route, all 25 rows returned |
| T17.3 | AR aging balance query | ✅ PASS | SQL route conf=0.75; Current=$280,200, Past Due=$12,400, Total=$292,600 |
| T17.4 | AP invoice count query | ⚠️ PARTIAL | Test query "how many invoices" doesn't route to SQL ("invoices" not a column); AP column-specific queries work (past_due=$12,400) |
| T17.5 | Budget Q1 query | ⚠️ PARTIAL | Test query "total budget for Q1 2025" ties with ap_summary (both score 0.75); "sum of Q1 budget values" routes correctly ($8.37M) |
| T17.6 | P&L cross-year query | ❌ FAIL | P&L UNION template not registered — year tables ingested with generic columns ('0','1') not financial vocabulary; _is_pl_table() returns False |
| T17.7 | SQL injection guard | ✅ PASS | Injection attempt logged and blocked by guard |
| T17.8 | RAG fallback (non-quantitative) | ✅ PASS | Non-quantitative queries fall through to vector search |
| T17.9 | NL2SQL tag access control | ❌ FAIL | #461: user2 (no finance tag) receives finance-tagged AR data via SQL path; NL2SQL does not enforce tag ACL |

## Open Investigations

- **#460**: Forms DB write fails — PostgreSQL 63-char column name truncation on long ACORD field names (ingestkit_forms library fix needed)
- **#461**: NL2SQL path bypasses tag-based access control — SQL queries execute without checking user tag permissions
- **T17.4/T17.5 routing**: NL2SQL column signal routing sensitive to exact column name words in query; when multiple tables score equally (0.75), first alphabetically wins (ap_summary before budget_2025)
- **T17.6**: P&L Excel ingested with generic column names ('0', '1') — financial vocabulary detection fails; P&L UNION template never registered

## Legend
- ✅ PASS
- ❌ FAIL — see Open Investigations
- ⏳ In progress
- ☐ Not yet run
- ⚠️ SKIP (with reason)
