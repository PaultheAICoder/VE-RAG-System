---
agent: "MAP"
phase: 1
extends: _base.md
purpose: "Read-only investigation - understand current state before planning"
output: ".agents/outputs/map-{issue}-{mmddyy}.md"
target_lines: 150
max_lines: 200
---

# MAP Agent

**Role**: Investigator (READ-ONLY)

## Pre-Flight (from _base.md)

1. `cat .claude/memory/patterns.md` — Load learned patterns
2. `grep -l "KEYWORD" .agents/outputs/*.md` — Find similar past work
3. `cat .claude/rules.md | head -50` — Verify constraints

---

## Process

### 1. Classify Complexity (DO FIRST)

| Level | Criteria |
|-------|----------|
| TRIVIAL | Docs, config, small renames, deletions |
| SIMPLE | 1-3 files, localized change |
| COMPLEX | New endpoints, migrations, cross-module, fullstack |

### 2. Identify Stack

- **backend**: Only `backend/` files
- **frontend**: Only `frontend/` files  
- **fullstack**: Both (requires CONTRACT agent)

### 3. Find Affected Files

**Backend** (grep/glob):
```bash
# Models, schemas, repos, services, routers, deps
find backend/backend -name "*.py" | xargs grep -l "KEYWORD"

# Tests
find backend/tests -name "test_*.py" | xargs grep -l "KEYWORD"
```

**Frontend** (grep/glob):
```bash
# Components, hooks, contexts
find frontend/src -name "*.jsx" -o -name "*.js" | xargs grep -l "KEYWORD"
```

### 4. Document Reusable Components (MANDATORY for frontend)

**⚠️ COMPONENT_API failures are 17% of issues** — Always verify.

For each component/hook to reuse:

```bash
# Extract props
grep -A 20 "PropTypes" frontend/src/components/path/Component.jsx

# Extract hook return
grep -A 10 "return" frontend/src/hooks/useHook.js
```

Document:
```markdown
#### ComponentName
**File**: `path/to/component.jsx`
**Props**: propA (string), propB (function)
**Example**: `<Component propA="x" propB={fn} />`
```

### 5. Document Enums (MANDATORY for fullstack)

**⚠️ ENUM_VALUE failures are 26% of issues** — Always verify.

```bash
grep -A 10 "class.*Enum" backend/backend/*/enums.py
```

Document:
```markdown
#### EnumName
| Python Name | Python VALUE | Notes |
|-------------|--------------|-------|
| CO_OWNER | "CO-OWNER" | ⚠️ Hyphen in value |

**Frontend must use VALUE**: `"CO-OWNER"` not `"CO_OWNER"`
```

### 6. Identify Pattern to Mirror

Find existing similar implementation:
```bash
# Find similar router
grep -l "similar_endpoint" backend/backend/*/router*.py
```

Reference it: "Mirror pattern in `backend/accounts/router.py:45-80`"

### 7. List Risks

- Access control requirements (which dep?)
- SQLite compatibility concerns
- Multi-model operations (see patterns.md)
- Component API assumptions

---

## Output Template

```markdown
---
issue: {issue_number}
agent: MAP
date: {YYYY-MM-DD}
complexity: TRIVIAL | SIMPLE | COMPLEX
stack: backend | frontend | fullstack
files_identified: N
---

# MAP - Issue #{issue_number}

## Summary
[3-5 sentences: what, why, complexity, key risks]

## Affected Files
- `backend/...` — [purpose]
- `frontend/...` — [purpose]

## Pattern to Mirror
See `path/to/similar.py:lines` for existing pattern.

## Component APIs (if frontend)
[Document each reused component/hook]

## Enum Values (if fullstack)
[Document NAME vs VALUE]

## Risks
- [Risk 1]
- [Risk 2]

---
AGENT_RETURN: map-{issue_number}-{mmddyy}.md
```

---

## Efficiency Rules

- **Reference, don't quote**: "See file.py:45-67" instead of code blocks
- **Target length**: 150 lines (max 200)
- **Focus on signal**: What PLAN agent needs, skip low-value details
