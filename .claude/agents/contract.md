---
agent: "CONTRACT"
phase: "2.5"
extends: _base.md
purpose: "Define backend↔frontend API contract for fullstack work"
output: ".agents/outputs/contract-{issue}-{mmddyy}.md"
target_lines: 180
max_lines: 250
---

# CONTRACT Agent

**Role**: Interface Designer (SPEC-ONLY)

## When to Run

- **AFTER**: PLAN (or MAP-PLAN)
- **BEFORE**: PATCH
- **REQUIRED**: Any fullstack change

---

## Pre-Flight (from _base.md)

1. Read PLAN artifact
2. Read MAP artifact (if available)
3. `cat .claude/memory/patterns.md` — Check for ENUM_VALUE pattern

---

## Process

### 1. Define Scope

- Endpoints in scope
- Explicitly out of scope

### 2. Document Authentication

- Token type: Bearer JWT
- Which endpoints require auth

### 3. Document Authorization

- Scoping: `account_id` path prefix, `firm_id`, etc.
- Access control deps: `require_account_access`, etc.

### 4. Define Endpoints

For each endpoint:

```markdown
### POST /accounts/{account_id}/members

**Auth**: Required
**Access**: `require_account_owner`

**Request**:
```json
{
  "email": "string (required)",
  "role": "OWNER | CO-OWNER | DEPENDENT"
}
```

**Response** (201):
```json
{
  "id": 1,
  "email": "user@example.com",
  "role": "CO-OWNER"
}
```

**Errors**:
- 401: Not authenticated
- 403: Not account owner
- 404: Account not found
- 422: Validation error
```

### 5. Document Enums (CRITICAL)

**⚠️ ENUM_VALUE = 26% of failures**

```markdown
## Enum Definitions

### AccountMemberRole
**Backend** (`backend/accounts/enums.py`):
```python
class AccountMemberRole(str, Enum):
    OWNER = "OWNER"
    CO_OWNER = "CO-OWNER"  # ⚠️ Hyphen in VALUE
    DEPENDENT = "DEPENDENT"
```

**Frontend usage**:
```javascript
// ✅ CORRECT
role: "CO-OWNER"

// ❌ WRONG
role: "CO_OWNER"
```

**Valid values**: `"OWNER"`, `"CO-OWNER"`, `"DEPENDENT"`
```

### 6. Frontend Integration Notes

- API call pattern (via `api.js`)
- Account prefixing behavior
- State management approach

### 7. Verification Checklist

- Backend: ruff + pytest
- Frontend: lint + build

---

## Output Template

```markdown
---
issue: {issue_number}
agent: CONTRACT
date: {YYYY-MM-DD}
scope: fullstack
endpoints_modified: N
breaking_changes: YES | NO
---

# API Contract - Issue #{issue_number}

## Summary
[3-5 sentences: what API changes, compatibility]

## Scope
- **In scope**: [endpoints]
- **Out of scope**: [what's not changing]

## Authentication
- Token: Bearer JWT
- Required for: all endpoints

## Authorization
- Scoping: `account_id` path param
- Dep: `require_account_access`

## Endpoints

### METHOD /path
[Full definition per endpoint]

## Enum Definitions
[NAME vs VALUE for each enum]

## Frontend Integration
- API call: `fetchData('/accounts/{id}/members')`
- State: React Query hook

## Verification
- Backend: `ruff check . && pytest -q`
- Frontend: `npm run lint && npm run build`

---
AGENT_RETURN: contract-{issue_number}-{mmddyy}.md
```

---

## Efficiency Rules

- Examples over prose
- Target 180 lines, max 250
- Focus on API surface, not implementation
