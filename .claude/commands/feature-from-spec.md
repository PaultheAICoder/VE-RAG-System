---
description: Create a feature request issue from specification analysis (used by /spec-review)
argument-hint: [feature title]
---

## Specification Reference

**Source**: `docs/features/[spec-file].md`
**Section**: [Specific section in spec]
**Requirement**:
> [Exact requirement from spec - quote directly]

## Implementation Gap

**Current State**: [What exists now based on codebase analysis]
**Required State**: [What spec requires]
**Gap**: [What's missing / needs to change]

## Feature Description
[Clear 1-2 sentence description of what this issue will implement]

## User Story
**As a** <persona>
**I want** <capability>
**So that** <value>

## Scope / Stack
- [ ] Backend
- [ ] Frontend
- [ ] Fullstack

## Requirements

### Functional
- [ ] [Specific requirement from spec]
- [ ] [Another requirement]
- [ ] [Another requirement]

### Non-Functional
- [ ] Minimal diffs / no drive-by refactors
- [ ] Security / tenancy enforced via deps (backend)
- [ ] SQLite in-memory tests supported (backend)
- [ ] React Router migration-friendly changes (frontend)
- [ ] See `.claude/rules.md` for structural constraints

## Technical Context

### Affected Areas

**Backend (if applicable):**
- Models: [file path or "None"]
- Schemas: [file path or "Create new"]
- Repositories: [file path or "Create new"]
- Services: [file path or "Create new"]
- Routers: [file path or "Create new"]
- Deps / access control: [dependency to use, e.g., require_account_access]
- Tests: [file path or "Create new"]

**Frontend (if applicable):**
- Components/pages: [file path or "Create new"]
- Context/state: [what state management needed]
- API layer: [file path or "Create new"]
- Configs/constants: [if needed]
- Navigation (App.jsx): [route to add]

### Dependencies
[List any dependencies on other features/issues]

### Pattern to Follow
[Which existing component/module to mirror - be specific with file path]

Example:
- Mirror `backend/account_members/` pattern for CRUD operations
- Follow `frontend/src/components/accounts/AccountsList.jsx` for list component

## Acceptance Criteria
- [ ] [Specific testable criteria from spec]
- [ ] [Another criteria]
- [ ] No project structure violations
- [ ] If backend touched: `cd backend && pytest -q` passes and `cd backend && ruff check .` passes
- [ ] If frontend touched: `cd frontend && npm run lint` passes and `cd frontend && npm run build` passes
- [ ] Loading / empty / error states handled in UI (if UI change)

## Complexity: [TRIVIAL/SIMPLE/COMPLEX]

**Justification**:
[Why this complexity level - reference file count, pattern availability, etc.]

## Related Issues
[List related issues if part of larger spec, e.g., "Part of Invitation System - see #184, #185"]
