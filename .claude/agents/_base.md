---
type: base-agent
version: 3.0
purpose: Shared behaviors inherited by all agents
---

# Base Agent Behaviors

All agents inherit these behaviors. Read this FIRST before your agent-specific instructions.

---

## 1. Pre-Flight: Load Learned Patterns (TIERED)

**BEFORE investigating or planning**, load accumulated knowledge:

### Always Load (Critical Patterns ~50 lines)
```bash
cat .claude/memory/patterns-critical.md
```

### Load If Needed (Full Patterns ~660 lines)
Only load `.claude/memory/patterns-full.md` when:
- Issue is COMPLEX classification
- Issue involves pattern not in critical file
- You need detailed prevention checklists

**Apply relevant patterns** to your current task. Critical patterns cover 89% of failures.

---

## 2. Pre-Flight: Check Similar Past Work

```bash
# Find similar past artifacts (adjust keywords)
grep -l "KEYWORD" .agents/outputs/*.md 2>/dev/null | head -3
```

If found, read the artifact to learn from past approaches. Note what worked and what caused issues.

---

## 3. Efficiency Rules

### Reference, Don't Re-Quote
```markdown
# ❌ BAD: Re-quoting 50 lines of code
Here's the existing implementation:
[50 lines of code]

# ✅ GOOD: Reference with line numbers
See `backend/accounts/services.py:45-67` for existing pattern.
```

### Single Source of Truth
- Acceptance criteria: Define ONCE in MAP-PLAN or PLAN
- Other agents reference: "See MAP-PLAN acceptance criteria"
- Never duplicate lists across artifacts

### Target Lengths
| Agent | Target Lines | Max Lines |
|-------|--------------|-----------|
| MAP | 150 | 200 |
| MAP-PLAN | 400 | 500 |
| PLAN | 400 | 500 |
| TEST-PLANNER | 250 | 350 |
| CONTRACT | 200 | 300 |
| PATCH | 300 | 400 |
| PROVE | 250 | 350 |

---

## 4. Artifact Naming

**Pattern**: `{agent}-{issue}-{mmddyy}.md`

```bash
# Set these variables at start of run
ISSUE_NUMBER=184
RUN_DATE=$(date +%m%d%y)
ARTIFACT_NAME="{agent}-${ISSUE_NUMBER}-${RUN_DATE}.md"
```

**Output directory**: `.agents/outputs/`

---

## 5. Common Verification Commands

### Backend (AI Ready RAG)
```bash
ruff check ai_ready_rag tests && pytest tests/ -q
```

### Format Check
```bash
ruff format --check ai_ready_rag tests
```

### Verify Scope (no unplanned changes)
```bash
git diff --name-only HEAD
# Should only show files in PLAN
```

---

## 6. Constraint Enforcement

**Before ANY file operation**, verify:

```bash
# Check constraints file
cat .claude/rules.md | head -50
```

**Forbidden actions** (always blocked):
- Creating top-level directories
- Moving `ai_ready_rag/`, `tests/`, `.claude/`
- Committing directly to `main` branch (use feature branches)
- Committing draft specs (finalize first)
- Using `requirements.txt` for dev (use `requirements-wsl.txt`)
- Hardcoding secrets (use environment variables)

---

## 7. AGENT_RETURN Directive

Every agent MUST end output with:

```markdown
AGENT_RETURN: {artifact-filename}
```

Example:
```markdown
AGENT_RETURN: map-184-010325.md
```

This signals successful completion to the orchestrator.

---

## 8. High-Frequency Failure Prevention

These patterns cause >50% of failures. Check proactively:

### ACCESS_CONTROL (Critical for RAG)
**Trigger**: Any vector search or document retrieval
**Check**: User tags filter applied BEFORE search
```bash
grep -A 10 "user_tags" ai_ready_rag/services/vector_service.py
```
**Prevention**: Pre-retrieval filtering - LLM never sees unauthorized docs

### SETTINGS_PROFILE (Common config issue)
**Trigger**: Adding new settings that vary by profile
**Check**: Profile defaults in config.py
```bash
grep -A 20 "PROFILE_DEFAULTS" ai_ready_rag/config.py
```
**Prevention**: Add to both laptop and spark profiles if applicable

### SERVICE_LAYER (13% of failures)
**Trigger**: CRUD operation with multiple models
**Check**: Map each field to its owning model
**Prevention**: Service layer must coordinate all model updates atomically

---

## 9. Outcome Recording (PROVE Agent Only)

After verification, record outcome:

### If PASS
```bash
echo '{"issue":'$ISSUE_NUMBER',"date":"'$(date +%Y-%m-%d)'","status":"PASS","complexity":"SIMPLE","stack":"backend"}' >> .claude/memory/metrics.jsonl
```

### If BLOCKED
```bash
# Record failure with root cause
echo '{"issue":'$ISSUE_NUMBER',"date":"'$(date +%Y-%m-%d)'","root_cause":"ENUM_VALUE","details":"...","fix":"..."}' >> .claude/memory/failures.jsonl

# Record metric
echo '{"issue":'$ISSUE_NUMBER',"date":"'$(date +%Y-%m-%d)'","status":"BLOCKED","root_cause":"ENUM_VALUE"}' >> .claude/memory/metrics.jsonl
```

---

## 10. When to Escalate

**STOP and report to orchestrator** if:
- Complexity seems wrong (SIMPLE issue is actually COMPLEX)
- Missing information blocks progress
- Constraint violation required to proceed
- Issue scope is ambiguous

**Do NOT** proceed with assumptions. Ask for clarification.
