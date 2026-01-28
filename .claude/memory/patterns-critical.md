<!-- Critical Patterns for AI Ready RAG -->
<!-- ~600 tokens budget - keep concise -->

### Top Failure Patterns

| Pattern | Trigger | Prevention |
|---------|---------|------------|
| **Changelog skip** | Editing DEVELOPMENT_PLANS.md | ALWAYS update Change Log section, increment version |
| **Access control bypass** | Adding vector search | ALWAYS filter by user tags BEFORE search |
| **Hardcoded secrets** | Adding config values | Use environment variables, never commit secrets |

### Quick Reference

**File Edit Checklist:**
- DEVELOPMENT_PLANS.md → Update changelog + version
- docs/ARCHITECTURE.md → Add ADR if architectural decision
- requirements.txt → Run `pip freeze` to verify versions

**Architecture Invariants:**
- Pre-retrieval access control (LLM never sees unauthorized docs)
- SQLite for app data (zero infrastructure)
- Qdrant for vectors (tag filtering)
- All services local (air-gap compatible)

### Project-Specific Gotchas

1. **Gradio + FastAPI**: Mount Gradio at `/app`, not root
2. **ChromaDB → Qdrant**: Migration in progress, check which is active
3. **Windows paths**: Use forward slashes or raw strings in Python
4. **Ollama models**: Must be pulled before use (`ollama pull model_name`)

### Current Sprint Focus
- FastAPI backend architecture
- JWT authentication
- SQLite schema setup
- Target: Feb 13, 2026
