---
description: Create a new bug report issue in GitHub
argument-hint: [bug title]
---

## Reported Issue
**What's broken**:
-

**Expected behavior**:
-

**Scope / Stack**:
- [ ] Backend
- [ ] Frontend
- [ ] Fullstack

**Severity**: Critical / High / Medium / Low

## Error Details (if applicable)
**Error type**:
**Error message**:
**Location (file:line)**:
**Endpoint/route (method path)**:

## Reproduction Steps
1.
2.
3.

## Notes / Logs
```text

```

## Technical Constraints
- [ ] No changes to monorepo structure (`backend/`, `frontend/` stay in place)
- [ ] No backend `src/` layout
- [ ] React Router changes follow migration guidelines (see `.claude/rules.md`)
- [ ] See `.claude/rules.md` for full constraints

## Acceptance Criteria
- [ ] Repro confirmed + root cause identified
- [ ] Fix is minimal and scoped
- [ ] No project structure violations
- [ ] If backend touched: tests updated + `cd backend && pytest -q` passes and `cd backend && ruff check .` passes
- [ ] If frontend touched: `cd frontend && npm run lint` passes and `cd frontend && npm run build` passes; no console errors
