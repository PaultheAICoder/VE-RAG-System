# Base Agent Behaviors

All orchestrate agents inherit these behaviors.

## Project Context

- **Project**: AI Ready RAG (VE-RAG-System)
- **Stack**: FastAPI, SQLAlchemy, Qdrant, Ollama, Gradio
- **Location**: `/home/jjob/projects/VE-RAG-System`
- **Requirements**: Use `requirements-wsl.txt` for dependencies

## CRITICAL: Path Requirements

**ALWAYS use absolute WSL paths**, not Windows paths:
- ✅ CORRECT: `/home/jjob/projects/VE-RAG-System/...`
- ❌ WRONG: `/mnt/c/Users/jjob/projects/VE-RAG-System/...`

When writing artifacts, use:
```
/home/jjob/projects/VE-RAG-System/.agents/outputs/{artifact-name}.md
```

## Mandatory Behaviors

### 1. Read Before Write
- Always read existing files before modifying
- Understand current patterns before adding new code

### 2. Follow Existing Patterns
- Match code style of existing files
- Use existing utilities (e.g., `generate_chunk_id`, `validate_tag`)
- Follow backend patterns in `.claude/rules/backend-patterns.md`

### 3. Security Invariants
- **Pre-retrieval access control**: Filter by user tags BEFORE vector search
- **No hardcoded secrets**: Use environment variables via config.py
- **Input validation**: Validate all user inputs

### 4. Testing Requirements
- Add tests for new functionality
- Integration tests go in `tests/` with `@pytest.mark.integration`
- Run `ruff check` before completing

### 5. Artifact Output
Every agent MUST:
1. Write artifact to `.agents/outputs/{agent}-{issue}-{date}.md`
2. End with: `AGENT_RETURN: {filename}`

## File Conventions

| Type | Location |
|------|----------|
| API routes | `ai_ready_rag/api/` |
| Services | `ai_ready_rag/services/` |
| Models | `ai_ready_rag/db/models.py` |
| Config | `ai_ready_rag/config.py` |
| Exceptions | `ai_ready_rag/core/exceptions.py` |
| Tests | `tests/test_*.py` |

## Git Workflow

**Main stays green at all times.**

- **NEVER commit directly to main**
- All development on feature branches: `feat/issue-XXX-description`
- Run tests before merge: `pytest tests/ -v`
- Merge to main only when tests pass

```bash
# Create feature branch
git checkout -b feat/issue-XXX-description

# After work complete, merge to main
git checkout main && git merge feat/issue-XXX-description
```

## Do NOT

- Commit directly to main (use feature branches)
- Modify `requirements.txt` (use `requirements-wsl.txt`)
- Skip writing the artifact file
- Implement without reading the issue specification
- Bypass access control in vector operations
