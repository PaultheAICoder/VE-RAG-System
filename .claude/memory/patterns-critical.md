<!-- Critical Patterns for AI Ready RAG -->
<!-- ~600 tokens budget - keep concise -->

### Top Failure Patterns

| Pattern | Trigger | Prevention |
|---------|---------|------------|
| **Commit to main** | Any code change | ALWAYS use feature branch, NEVER commit directly to main |
| **Commit draft specs** | Creating specs | Do NOT commit until finalized after review |
| **Changelog skip** | Editing DEVELOPMENT_PLANS.md | ALWAYS update Change Log section, increment version |
| **Access control bypass** | Adding vector search | ALWAYS filter by user tags BEFORE search |
| **Hardcoded secrets** | Adding config values | Use environment variables, never commit secrets |
| **Wrong requirements** | Adding dependencies | Use `requirements-wsl.txt` for dev, NOT `requirements.txt` |

### Quick Reference

**File Edit Checklist:**
- DEVELOPMENT_PLANS.md → Update changelog + version
- docs/ARCHITECTURE.md → Add ADR if architectural decision
- requirements-wsl.txt → Add new deps here (NOT requirements.txt)

**Architecture Invariants:**
- Pre-retrieval access control (LLM never sees unauthorized docs)
- SQLite for app data (zero infrastructure)
- Qdrant for vectors (tag filtering)
- All services local (air-gap compatible)

### Project-Specific Gotchas

1. **Requirements files**: Use `requirements-wsl.txt` for dev (has qdrant-client, no chromadb)
2. **Gradio + FastAPI**: Mount Gradio at `/app`, not root
3. **Windows paths**: Use forward slashes or raw strings in Python
4. **Ollama models**: Must be pulled before use (`ollama pull model_name`)

### Current Sprint Focus
- FastAPI backend architecture
- JWT authentication
- SQLite schema setup
- Target: Feb 13, 2026
