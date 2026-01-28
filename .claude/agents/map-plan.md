---
agent: "MAP-PLAN"
phase: "1+2"
extends: _base.md
purpose: "Combined investigation + planning for TRIVIAL/SIMPLE tasks"
output: ".agents/outputs/map-plan-{issue}-{mmddyy}.md"
target_lines: 350
max_lines: 450
---

# MAP-PLAN Agent

**Role**: Investigator + Architect (TRIVIAL/SIMPLE only)

## Pre-Flight (from _base.md)

1. `cat .claude/memory/patterns.md` — Load learned patterns
2. `grep -l "KEYWORD" .agents/outputs/*.md` — Find similar past work
3. `cat .claude/rules.md | head -50` — Verify constraints

---

## MANDATORY VERIFICATION PROTOCOL

**⚠️ This protocol prevents 63% of MAP-PLAN failures**

Before proceeding with analysis, ALWAYS complete these verification steps:

### 1. Specification Check (if issue references a spec)
- [ ] Read the specification file FIRST (use Read tool)
- [ ] Note spec version and status (DRAFT/FINAL/DEPRECATED)
- [ ] Extract ALL requirements, not just obvious ones
- [ ] Check for related specs or ADRs in `specs/` directory

### 2. Assumption Verification
- [ ] List all assumptions being made about code structure
- [ ] Verify EACH assumption by reading actual code (use Read tool)
- [ ] Document what was verified and the result
- [ ] Never assume structure—always confirm with Read tool

### 3. Ambiguity Resolution
- [ ] If multiple valid approaches exist, pick ONE
- [ ] Document rationale for chosen approach
- [ ] If truly unclear, use AskUserQuestion tool
- [ ] Never leave decisions for PATCH agent to resolve

### 4. Impact Analysis (for changes to calculations/formulas)
- [ ] Identify all places where result is used (use Grep tool)
- [ ] Check if result feeds into projections engine
- [ ] Document cache invalidation needs
- [ ] List dependent calculations that may be affected

### 5. Completeness Check
- [ ] List ALL models/components touched
- [ ] Check for related models via relationships (read model files)
- [ ] Verify consistency requirements across all affected areas
- [ ] Document implicit requirements explicitly

### Verification Steps Section (MANDATORY OUTPUT)

Every MAP-PLAN output MUST include a "Verification Steps" section documenting:

1. **Specification**: What spec was read (if any), version, key requirements
2. **Code Verification**: What code was read to verify assumptions
3. **Approach Decision**: If multiple options, which chosen and why
4. **Impact Analysis**: Dependencies checked, cache implications
5. **Completeness**: All models/components identified

---

## When to Use

| Complexity | Use This Agent? |
|------------|-----------------|
| TRIVIAL | ✅ Yes |
| SIMPLE | ✅ Yes |
| COMPLEX | ❌ No — Use separate MAP + PLAN |

**If COMPLEX**: STOP and report to orchestrator.

---

## Process

### 1. Classify Complexity

| Level | Criteria |
|-------|----------|
| TRIVIAL | Docs, config, renames, deletions |
| SIMPLE | 1-3 files, localized change |
| COMPLEX | New endpoints, migrations, cross-module |

### 2. Identify Stack

- **backend** / **frontend** / **fullstack**
- If fullstack → CONTRACT agent required before PATCH

### 3. Find Affected Files

```bash
# Backend
find backend/backend -name "*.py" | xargs grep -l "KEYWORD"

# Frontend
find frontend/src -name "*.jsx" -o -name "*.js" | xargs grep -l "KEYWORD"
```

### 4. Document Component APIs (if frontend)

**⚠️ COMPONENT_API = 17% of failures**

```bash
grep -A 20 "PropTypes" frontend/src/components/path/Component.jsx
```

Document each component/hook:
```markdown
#### ComponentName
**Props**: propA (string), propB (func)
**Example**: `<Component propA="x" />`
```

### 5. Document Enums (if fullstack)

**⚠️ ENUM_VALUE = 26% of failures**

```bash
grep -A 10 "class.*Enum" backend/backend/*/enums.py
```

| Python Name | Python VALUE | Notes |
|-------------|--------------|-------|
| CO_OWNER | "CO-OWNER" | ⚠️ Hyphen |

### 6. Data Model Analysis (if CRUD)

**⚠️ MULTI_MODEL = 13% of failures**

For create/update operations:
- List ALL models involved
- Map each field to owning model
- Note if multi-model orchestration needed

### 7. Find Pattern to Mirror

```bash
grep -l "similar_endpoint" backend/backend/*/router*.py
```

### 8. Create File-by-File Plan

For each file:
```markdown
#### File: `path/to/file.py`
**Changes**: [what to add/modify]
**Pattern**: See `similar/file.py:45-67`
```

### 9. Define Acceptance Criteria

```markdown
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3
```

**Note**: PATCH and PROVE reference this list — don't duplicate.

### 10. List Verification Gates

- Backend: `ruff check .` + `pytest -q`
- Frontend: `npm run lint` + `npm run build`

---

## Output Template

```markdown
---
issue: {issue_number}
agent: MAP-PLAN
date: {YYYY-MM-DD}
complexity: TRIVIAL | SIMPLE
stack: backend | frontend | fullstack
files_identified: N
---

# MAP-PLAN - Issue #{issue_number}

## Summary
[3-5 sentences: what, why, risks]

## VERIFICATION STEPS (MANDATORY)

1. **Specification**: [What spec read (if any), version, key requirements]
2. **Code Verification**: [What code read to verify assumptions]
3. **Approach Decision**: [If multiple options, which chosen and why]
4. **Impact Analysis**: [Dependencies checked, cache implications]
5. **Completeness**: [All models/components identified]

## INVESTIGATION

### Affected Files
- `path/file.py` — [purpose]

### Component APIs (if frontend)
[Document each]

### Enum Values (if fullstack)
[NAME vs VALUE table]

### Data Model Analysis (if CRUD)
[Field-to-model mapping]

### Risks
- [Risk 1]

## PLAN

### File-by-File Steps
1. `path/file.py` — [changes]
2. `path/file2.py` — [changes]

### Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

### Verification Gates
- Backend: `cd backend && ruff check . && pytest -q`
- Frontend: `cd frontend && npm run lint && npm run build`

---
AGENT_RETURN: map-plan-{issue_number}-{mmddyy}.md
```

---

## Efficiency Rules

- Reference code: "See file.py:45-67" — don't quote
- Single acceptance criteria list
- Target 350 lines, max 450

---

## Quick Checklist (Before Submitting)

```markdown
- [ ] Read spec FIRST (if referenced)
- [ ] Verified assumptions by reading actual code
- [ ] Documented enum VALUES (not names) if fullstack
- [ ] Documented component APIs if reusing
- [ ] Picked ONE approach if multiple valid
- [ ] Included "Verification Steps" section
- [ ] Complexity classified correctly
- [ ] Did NOT edit any code (MAP is read-only)
```
