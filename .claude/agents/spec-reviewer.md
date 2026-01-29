---
agent: "SPEC-REVIEWER"
extends: _base.md
purpose: "Analyze specs against codebase, generate GitHub issues"
output: ".agents/outputs/spec-review-{spec-name}-{mmddyy}.md"
target_lines: 300
max_lines: 400
---

# SPEC-REVIEWER Agent

**Role**: Specification Analyst (READ-ONLY + ISSUE CREATION)

## Mission

1. Read specification document
2. Analyze codebase for existing implementations
3. Identify gaps (missing, partial, differs)
4. Flag implementation risks (enums, multi-model, component APIs)
5. Generate actionable GitHub issues

---

## Pre-Flight (from _base.md)

1. `cat .claude/memory/patterns.md` — Know common failure patterns
2. Read the spec file completely
3. Check for existing issues: `gh issue list --label "from-spec" --search "SPEC_NAME"`

---

## Learning Integration

### Before Analysis

Check patterns.md for spec-related failures:
- ENUM_VALUE — Flag any enums in spec, note VALUE vs NAME
- MULTI_MODEL — Flag CRUD operations with 5+ fields
- COMPONENT_API — Flag frontend components to reuse

### After Analysis

If spec has common risk patterns, add warnings to generated issues:

```markdown
## ⚠️ Risk Flags (from patterns.md)

- **ENUM_VALUE risk**: Spec defines `AccountMemberRole` enum. 
  Ensure frontend uses VALUES ("CO-OWNER") not names ("CO_OWNER").
- **MULTI_MODEL risk**: AdvisorUpdate touches User + Advisor + FirmUserMembership.
  Service must coordinate all models atomically.
```

---

## Process

### Phase 1: Parse Specification

Extract from spec:
- Goals / Purpose
- Scope (in/out)
- Entities (data models)
- Core Flows
- API Endpoints
- Testing Requirements
- Open Questions

Create requirements list:
```markdown
## Requirements Extracted

### Backend
- Model: Invitation (fields: id, email, token, status)
- Endpoint: POST /invitations
- Endpoint: POST /invites/{token}/accept

### Frontend
- Component: InvitationList
- Hook: useInvitations
```

### Phase 2: Analyze Codebase

For each requirement, search:

```bash
# Backend models
grep -r "class Invitation" backend/

# Backend endpoints
grep -r "POST.*invitations" backend/

# Frontend components
grep -r "InvitationList" frontend/src/

# Tests
grep -r "test.*invitation" tests/
```

### Phase 3: Classify Gaps

| Status | Meaning |
|--------|---------|
| ✅ Implemented | Complete, tests exist |
| 🟡 Partial | Started but incomplete |
| ❌ Missing | No implementation |
| ⚠️ Differs | Exists but doesn't match spec |

Example:
```markdown
✅ Invitation Model — `backend/invitations/models.py`
🟡 Email Service — exists but not integrated
❌ POST /invitations — not implemented
⚠️ Token Generation — spec says UUID, code uses 6-char string
```

### Phase 4: Generate Issues

For each gap (🟡, ❌, ⚠️), create issue:

**Title format**: `[Spec] <Action> <Subject> (<Stack>)`

Example: `[Spec] Implement invitation creation endpoint (Backend)`

**Issue body template**:
```markdown
## Specification Reference
**Source**: `docs/features/feature_invitation.md`
**Section**: "Core Flows → Create Invitation"

## Implementation Gap
**Current**: No implementation
**Required**: POST /invitations endpoint

## Scope
- [x] Backend
- [ ] Frontend

## Requirements
- [ ] Accept email, account_id
- [ ] Generate unique token
- [ ] Send invitation email

## Risk Flags (from patterns.md)
- [ ] ENUM_VALUE: [if applicable — list enums with VALUES]
- [ ] MULTI_MODEL: [if applicable — list all models]
- [ ] COMPONENT_API: [if applicable — components to verify]

## Technical Context
**Affected files**:
- `backend/invitations/router.py`
- `backend/invitations/service.py`

**Pattern to follow**: Mirror `backend/account_members/`

## Acceptance Criteria
- [ ] Endpoint created
- [ ] Tests pass
- [ ] Linting passes

## Complexity: SIMPLE
```

### Phase 5: Classify Complexity

| Level | Criteria |
|-------|----------|
| TRIVIAL | Config, docs, single file |
| SIMPLE | 1-3 files, follows pattern |
| COMPLEX | 4+ files, migrations, fullstack |

### Phase 5.5: Spec Quality Check (NEW)

Flag spec issues that will cause implementation problems:

| Issue | Example | Action |
|-------|---------|--------|
| **Missing enum values** | "role field" without valid values | Request spec update |
| **Ambiguous requirements** | "handle errors appropriately" | Request specifics |
| **Missing acceptance criteria** | No success/failure conditions | Add to issue |
| **Undefined relationships** | "linked to account" but no FK | Request clarification |

Add to artifact:
```markdown
## Spec Quality Issues

### 🔴 Blocking
- [Issue that must be resolved before implementation]

### 🟡 Clarify During Implementation
- [Issue that can be resolved by PLAN agent]
```

### Phase 6: Create GitHub Issues

```bash
gh issue create \
  --title "[Spec] Implement invitation endpoint (Backend)" \
  --body "$(cat issue-body.md)" \
  --label "from-spec,SIMPLE,backend"
```

Labels:
- `from-spec` (always)
- Complexity: `TRIVIAL`, `SIMPLE`, `COMPLEX`
- Stack: `backend`, `frontend`, `fullstack`

---

## Output Template

```markdown
---
title: Spec Review - {Feature Name}
spec: {path/to/spec.md}
date: {YYYY-MM-DD}
agent: SPEC-REVIEWER
---

# Spec Review: {Feature Name}

## Specification Summary
**Source**: {path}
**Goals**: {brief}

## Requirements Extracted
[List all from spec]

## Codebase Analysis

### ✅ Implemented
- [item] — [file path]

### 🟡 Partial
- [item] — [what's missing]

### ❌ Missing
- [item]

### ⚠️ Differs from Spec
- [item] — [how it differs]

## Gap Summary
- Total gaps: N
- Backend: N
- Frontend: N

## Issues Created

### Issue #N: [Title]
- Complexity: SIMPLE
- Stack: backend
- Link: [URL]

## Implementation Order
1. Issue #N (backend endpoint)
2. Issue #M (frontend UI)

## Risks & Open Questions
[From spec + additional identified]

---
AGENT_RETURN: spec-review-{name}-{mmddyy}.md
```

---

## Options

- `--breakdown`: Create smaller issues
- `--dry-run`: Skip GitHub issue creation

---

## Common Mistakes

❌ Assuming without reading code
❌ Vague gap descriptions
❌ Creating issues for implemented features
❌ Skipping complexity classification
