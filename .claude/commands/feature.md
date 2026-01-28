---
description: Create a new feature request issue in GitHub
argument-hint: [feature title]
---

## Specification Reference (if applicable)
**Source**: `docs/features/[spec-file].md` or "N/A (ad-hoc feature)"
**Section**: [Specific section] or "N/A"

## Feature Description
[Clear 1-2 sentence description]

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
- [ ]
- [ ]

### Non-Functional
- [ ] Minimal diffs / no drive-by refactors
- [ ] Security / tenancy enforced via deps (backend)
- [ ] SQLite in-memory tests supported (backend)
- [ ] React Router migration-friendly changes (frontend)
- [ ] See `.claude/rules.md` for structural constraints

## Technical Context
### Affected Areas
**Backend (if applicable):**
- Models:
- Schemas:
- Repositories:
- Services:
- Routers:
- Deps / access control:
- Tests:

**Frontend (if applicable):**
- Components/pages:
- Context/state:
- API layer:
- Configs/constants:
- Navigation (App.jsx):

## Acceptance Criteria
- [ ] Behavior works end-to-end
- [ ] No project structure violations
- [ ] If backend touched: `cd backend && pytest -q` passes and `cd backend && ruff check .` passes
- [ ] If frontend touched: `cd frontend && npm run lint` passes and `cd frontend && npm run build` passes
- [ ] Loading / empty / error states handled in UI (if UI change)

## Complexity: [TRIVIAL/SIMPLE/COMPLEX]

**Justification**:
[Why this complexity level - reference file count, pattern availability, new vs existing patterns]

- TRIVIAL: Config changes, docs, single-file edits
- SIMPLE: 1-3 files, follows existing pattern exactly
- COMPLEX: 4+ files, new patterns, database migrations, fullstack
