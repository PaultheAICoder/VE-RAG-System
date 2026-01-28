# Artifact Templates

Slim templates for agent outputs. Copy structure, fill in content.

---

## MAP Artifact Template

```markdown
---
issue: {N}
agent: MAP
date: {YYYY-MM-DD}
complexity: TRIVIAL | SIMPLE | COMPLEX
stack: backend | frontend | fullstack
---

# MAP - Issue #{N}

## Summary
[3-5 sentences]

## Affected Files
- `path/file` — [purpose]

## Pattern to Mirror
See `path/similar.py:lines`

## Component APIs (if frontend)
#### ComponentName
**Props**: [list]
**Example**: `<Component prop={} />`

## Enum Values (if fullstack)
| Name | VALUE |
|------|-------|

## Risks
- [list]

---
AGENT_RETURN: map-{N}-{mmddyy}.md
```

---

## MAP-PLAN Artifact Template

```markdown
---
issue: {N}
agent: MAP-PLAN
date: {YYYY-MM-DD}
complexity: TRIVIAL | SIMPLE
stack: backend | frontend | fullstack
---

# MAP-PLAN - Issue #{N}

## Summary
[3-5 sentences]

## INVESTIGATION
### Affected Files
- `path/file` — [purpose]

### Component APIs
[if frontend]

### Enum Values
[if fullstack]

## PLAN
### File-by-File
1. `path/file` — [changes]

### Acceptance Criteria
- [ ] Criterion 1

### Verification
- Backend: `ruff check . && pytest -q`
- Frontend: `npm run lint && npm run build`

---
AGENT_RETURN: map-plan-{N}-{mmddyy}.md
```

---

## PLAN Artifact Template

```markdown
---
issue: {N}
agent: PLAN
date: {YYYY-MM-DD}
complexity: COMPLEX
stack: backend | frontend | fullstack
---

# PLAN - Issue #{N}

## Summary
[3-5 sentences]

## Scope
- Stack: [backend/frontend/fullstack]
- Out of scope: [what won't change]

## File-by-File
### `path/file.py`
**Changes**: [list]
**Pattern**: See `similar.py:lines`

## Access Control
- Dep: `require_account_access`

## Acceptance Criteria
- [ ] Criterion 1

## Verification
[commands]

---
AGENT_RETURN: plan-{N}-{mmddyy}.md
```

---

## CONTRACT Artifact Template

```markdown
---
issue: {N}
agent: CONTRACT
date: {YYYY-MM-DD}
scope: fullstack
breaking_changes: NO
---

# API Contract - Issue #{N}

## Summary
[3-5 sentences]

## Endpoints

### METHOD /path
**Auth**: Required
**Request**: `{fields}`
**Response**: `{fields}`
**Errors**: 401, 403, 404, 422

## Enums
| Name | VALUE |
|------|-------|
| CO_OWNER | "CO-OWNER" |

## Frontend Usage
```javascript
const { data } = useQuery(...)
```

---
AGENT_RETURN: contract-{N}-{mmddyy}.md
```

---

## PATCH Artifact Template

```markdown
---
issue: {N}
agent: PATCH
date: {YYYY-MM-DD}
status: Complete | Blocked
files_modified: N
---

# PATCH - Issue #{N}

## Summary
[what was implemented]

## Pre-Flight
- [x] Read PLAN
- [x] Not on main branch

## Files Changed
### `path/file.py`
- Added: [what]

## Verification
- `ruff check .`: PASS
- `pytest -q`: N/N passing

## Issues Encountered
[None | list]

---
AGENT_RETURN: patch-{N}-{mmddyy}.md
```

---

## PROVE Artifact Template

```markdown
---
issue: {N}
agent: PROVE
date: {YYYY-MM-DD}
status: PASS | BLOCKED
---

# PROVE - Issue #{N}

## Status: PASS ✅ | BLOCKED ❌

## Verification
### Commands
- `ruff check .`: [output]
- `pytest -q`: [output]

### Pattern Checks
- ENUM_VALUE: ✅ N/A | PASS | FAIL
- COMPONENT_API: ✅ N/A | PASS | FAIL

### Acceptance Criteria
| Criterion | Status |
|-----------|--------|
| ... | ✅ |

## Outcome Recorded
- metrics.jsonl: ✅

---
AGENT_RETURN: prove-{N}-{mmddyy}.md
```
