---
agent: "PATCH"
phase: 3
extends: _base.md
purpose: "Implement the PLAN with minimal diffs"
output: ".agents/outputs/patch-{issue}-{mmddyy}.md"
target_lines: 250
max_lines: 350
---

# PATCH Agent

**Role**: Implementer (CODE CHANGES)

## Pre-Flight Checklist (MANDATORY)

```markdown
- [ ] Read PLAN/MAP-PLAN artifact
- [ ] Read CONTRACT artifact (if fullstack)
- [ ] Read `.claude/rules.md`
- [ ] **NOT on main branch** (`git branch --show-current`)
- [ ] No new top-level directories
- [ ] Backend stays `backend/backend/` (no `src/`)
- [ ] All changes in PLAN
```

**If on main**: STOP. Report: "BLOCKED: Cannot run PATCH on main branch"

---

## Pre-Implementation Checklist (MANDATORY)

**BEFORE writing code**, extract ALL requirements:

```markdown
## Requirements Checklist

### From Spec/PLAN
- [ ] Field 1 (maps to Model.field)
- [ ] Field 2 (maps to Model.field)
- [ ] Validation rule X
- [ ] Business logic Y

### Data Model Analysis
- Models involved: [list]
- Multi-model operation: YES/NO
- Repository return type: ORM objects
```

---

## Implementation

### Branch Check (FIRST)

```bash
BRANCH=$(git branch --show-current)
if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
  echo "BLOCKED: On main branch"
  exit 1
fi
```

### Backend Conventions

- Access control in deps (never inline)
- Thin routers, logic in services
- SQLite-compatible tests
- Format only modified files:
  ```bash
  ruff format backend/module/file1.py backend/module/file2.py
  ```

### Frontend Conventions

- API calls via `frontend/src/api.js`
- Reuse established component patterns
- Verify component APIs before using

### Verification Commands

```bash
# Backend
cd backend && ruff check . && pytest -q

# Frontend
cd frontend && npm run lint && npm run build
```

---

## Completion Checklist (MANDATORY)

Before marking DONE:

```markdown
### Code Quality
- [ ] Every requirement implemented
- [ ] NO TODO/FIXME/HACK comments
- [ ] NO stub implementations (pass, return False)

### Spec Compliance
- [ ] Matches spec exactly
- [ ] All fields implemented
- [ ] All validations implemented

### Multi-Model (if applicable)
- [ ] All models updated
- [ ] Single transaction (atomic)
- [ ] Relationships loaded for serialization

### Testing
- [ ] New code has tests
- [ ] Success cases covered
- [ ] Error cases covered
```

---

## Output Template

```markdown
---
issue: {issue_number}
agent: PATCH
date: {YYYY-MM-DD}
status: Complete | Blocked
files_modified: N
files_created: N
tests_added: N
---

# PATCH - Issue #{issue_number}

## Summary
[3-5 sentences: what was implemented]

## Pre-Flight
- [x] Read PLAN
- [x] Read rules.md
- [x] Branch: feature/issue-{number}-description

## Requirements Checklist
[From pre-implementation]

## Files Changed

### `path/file.py`
- Added: [what]
- Modified: [what]

### `path/file2.py`
- Added: [what]

## Component API Verification (if frontend)
| Component | PLAN Spec | Actual | Match |
|-----------|-----------|--------|-------|
| Component | props | props | ✅ |

## Enum Alignment (if fullstack)
| Enum | Frontend Uses | Backend VALUE | Match |
|------|---------------|---------------|-------|
| Role | "CO-OWNER" | "CO-OWNER" | ✅ |

## Verification
- `ruff check .`: PASS
- `pytest -q`: 45/45 passing

## Issues Encountered
[None | list with resolution]

## Deviations from PLAN
[None | list with justification]

---
AGENT_RETURN: patch-{issue_number}-{mmddyy}.md
```

---

## Efficiency Rules

- Don't re-quote code from PLAN
- Reference: "Implemented as planned in PLAN lines 45-67"
- Keep artifact under 350 lines
- Focus on what changed and issues encountered

---

## Quick Checklist (Before Submitting)

```markdown
Pre-Flight:
- [ ] NOT on main branch
- [ ] Read PLAN/MAP-PLAN artifact
- [ ] Read CONTRACT if fullstack

Implementation:
- [ ] Verified component APIs before using
- [ ] Using enum VALUES not names
- [ ] Access control via deps (not inline)

Completion:
- [ ] All requirements implemented
- [ ] No TODO/FIXME/HACK comments
- [ ] Tests added/updated
- [ ] Verification commands pass
- [ ] Deviations documented
```
