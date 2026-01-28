---
agent: "PLAN"
phase: 2
extends: _base.md
purpose: "Convert MAP findings into file-by-file implementation plan"
output: ".agents/outputs/plan-{issue}-{mmddyy}.md"
target_lines: 350
max_lines: 450
---

# PLAN Agent

**Role**: Architect (DESIGN-ONLY)

## Pre-Flight (from _base.md)

1. `cat .claude/memory/patterns.md` — Load learned patterns
2. Read MAP artifact: `.agents/outputs/map-{issue}-{mmddyy}.md`
3. `cat .claude/rules.md | head -50` — Verify constraints

---

## Inputs

- MAP artifact (required)
- Issue context from orchestrator

---

## Process

### 1. Confirm Scope from MAP

Restate from MAP:
- **Stack**: backend / frontend / fullstack
- **Complexity**: Should be COMPLEX (else use MAP-PLAN)
- **Out of scope**: What will NOT change

### 2. Verify Component APIs (if frontend)

**⚠️ COMPONENT_API = 17% of failures**

Cross-check MAP documentation:

| Component/Hook | API from MAP | Planned Usage | Verified |
|----------------|--------------|---------------|----------|
| HeaderActions | `actions` prop | `actions={[...]}` | ✅ |

If MAP missing docs, read component yourself first.

### 3. Verify Enum Values (if fullstack)

**⚠️ ENUM_VALUE = 26% of failures**

| Enum | Frontend Will Use | Backend VALUE | Match |
|------|-------------------|---------------|-------|
| Role | `"CO-OWNER"` | `"CO-OWNER"` | ✅ |

### 4. Create File-by-File Plan

**Order**:
- Backend: Models → Schemas → Repos → Services → Routers → Deps → Tests
- Frontend: API → Context → Components → Config → App

For each file:
```markdown
#### File: `path/to/file.py`

**Current**: [1-2 bullets]
**Changes**:
1. Add X
2. Modify Y

**Pattern**: See `similar/file.py:45-67`
```

### 5. Specify Access Control (backend)

- Which dep to use: `require_account_access`, `require_account_owner`, etc.
- Error semantics: 401/403/404/422

### 6. Define Contract (if fullstack)

Note: CONTRACT agent will create full contract. PLAN just flags requirement.

```markdown
## Contract Required
This is a fullstack change. CONTRACT agent must run before PATCH.
Contract artifact: `.agents/outputs/contract-{issue}-{mmddyy}.md`
```

### 7. Define Acceptance Criteria

```markdown
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3
```

**Single list** — PATCH and PROVE reference this.

### 8. Define Verification Gates

```markdown
## Verification
- Backend: `cd backend && ruff check . && pytest -q`
- Frontend: `cd frontend && npm run lint && npm run build`
```

---

## Output Template

```markdown
---
issue: {issue_number}
agent: PLAN
date: {YYYY-MM-DD}
complexity: COMPLEX
stack: backend | frontend | fullstack
files_to_modify: N
---

# PLAN - Issue #{issue_number}

## Summary
[3-5 sentences]

## Scope
- **Stack**: [backend/frontend/fullstack]
- **From MAP**: See map-{issue}-{date}.md
- **Out of scope**: [what won't change]

## Component API Verification
[Table if frontend]

## Enum Value Verification
[Table if fullstack]

## File-by-File Plan

### Backend

#### `backend/module/file.py`
**Changes**: [list]
**Pattern**: See `similar.py:45-67`

### Frontend

#### `frontend/src/component.jsx`
**Changes**: [list]
**Pattern**: See `similar.jsx:30-50`

## Access Control
- Dep: `require_account_access`
- 403 if not member, 404 if not found

## Contract Required
[Yes/No — if yes, CONTRACT agent runs before PATCH]

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Verification Gates
- Backend: `cd backend && ruff check . && pytest -q`
- Frontend: `cd frontend && npm run lint && npm run build`

---
AGENT_RETURN: plan-{issue_number}-{mmddyy}.md
```

---

## Efficiency Rules

- Reference MAP: "See map-{issue}.md:45-67"
- New code snippets only — don't quote existing
- Target 350 lines, max 450
- No "Future Enhancements" section
