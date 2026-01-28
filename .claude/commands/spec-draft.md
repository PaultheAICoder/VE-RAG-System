---
description: Draft a feature specification with guided questions and codebase discovery
argument-hint: "<feature title>"
---

# Spec Draft Command

Guides you through creating a complete specification by asking structured questions and discovering relevant patterns in the codebase.

## Usage

```bash
/spec-draft "Add advisor co-ownership feature"
/spec-draft "Expense category filtering"
```

---

## Process

### Step 1: Classify Feature Type

Ask user:
```
What type of feature is this?

1. **CRUD** — Create/Read/Update/Delete for a resource
2. **Integration** — Connect to external service or existing module
3. **UI Component** — Frontend-only addition
4. **Enhancement** — Modify existing behavior
5. **Fullstack** — New end-to-end feature

Type (1-5):
```

Based on answer, adjust required sections.

---

### Step 2: Discover Related Patterns

```bash
# Find similar features in codebase
grep -rl "KEYWORD" backend/backend/*/router*.py | head -5
grep -rl "KEYWORD" frontend/src/components/ | head -5

# Find existing models that might be affected
grep -l "account_id" backend/backend/*/models.py

# Find existing enums
grep -r "class.*Enum" backend/backend/*/enums.py
```

Report findings:
```
📁 Related Code Found:

Backend:
- backend/backend/advisors/router.py — Similar CRUD pattern
- backend/backend/advisors/models.py — Advisor model

Frontend:
- frontend/src/components/advisors/AdvisorList.jsx
- frontend/src/hooks/useAdvisors.js

Enums:
- FirmRole (backend/firms/enums.py): ADVISOR, SUPER_USER, ADMIN
```

---

### Step 3: Guided Questions (by feature type)

#### For CRUD Features:

```markdown
## Resource Definition

**Resource name** (singular): 
**Resource name** (plural): 
**Parent resource** (if nested): Account / Firm / None

## Fields

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| | | | | |

## Enums (if any)

| Enum Name | Values (comma-separated) | Notes |
|-----------|--------------------------|-------|
| | | |

⚠️ IMPORTANT: List the actual VALUES that will be sent in JSON, not Python names.
Example: "CO-OWNER" not "CO_OWNER"

## Relationships

| Related Model | Relationship | FK Location |
|---------------|--------------|-------------|
| | | |

## Access Control

Who can perform each action?

| Action | Allowed Roles | Dep to Use |
|--------|---------------|------------|
| Create | | |
| Read | | |
| Update | | |
| Delete | | |
```

#### For Fullstack Features:

```markdown
## Backend

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| | | |

### Models Affected

| Model | Changes |
|-------|---------|
| | |

⚠️ If >1 model affected by single operation, flag as MULTI_MODEL risk.

## Frontend

### Components to Create

| Component | Purpose | Reuses |
|-----------|---------|--------|
| | | |

### Components to Reuse

| Component | Current Location | Props/API |
|-----------|------------------|-----------|
| | | |

⚠️ VERIFY component APIs before specifying. Don't assume.

### State Management

| Data | Hook/Query | Cache Key |
|------|------------|-----------|
| | | |
```

#### For UI Components:

```markdown
## Component Specification

**Component name**: 
**Location**: frontend/src/components/...

### Props

| Prop | Type | Required | Default |
|------|------|----------|---------|
| | | | |

### State

| State | Type | Initial |
|-------|------|---------|
| | | |

### Events

| Event | Handler | Notes |
|-------|---------|-------|
| | | |

### Components to Reuse

| Component | Import From |
|-----------|-------------|
| | |

⚠️ Read PropTypes of reused components before assuming their API.
```

---

### Step 4: Auto-Fill from Codebase

Based on discovery, pre-fill what we can:

```markdown
## Auto-Discovered (verify these)

### Existing Patterns to Mirror
- CRUD pattern: See `backend/account_members/` (5 files)
- Hook pattern: See `frontend/src/hooks/useAccountMembers.js`

### Existing Enums (use these VALUES)
- AccountMemberRole: "OWNER", "CO-OWNER", "DEPENDENT", "EXECUTOR"
- FirmRole: "ADVISOR", "SUPER_USER", "ADMIN"

### Existing Components (verify APIs)
- HeaderActions: `frontend/src/components/common/HeaderActions.jsx`
- DataTable: `frontend/src/components/common/DataTable.jsx`

### Access Control Deps Available
- require_account_access — Any account member
- require_account_owner — Owner only
- require_firm_member — Firm membership required
```

---

### Step 5: Risk Flags (Auto-Generated)

Based on spec content, flag risks:

```markdown
## ⚠️ Risk Flags

### ENUM_VALUE Risk
Spec defines enums. Ensure:
- [ ] VALUES listed (not Python names)
- [ ] Frontend will use exact VALUES

### MULTI_MODEL Risk
Operations affect multiple models:
- [ ] Service coordinates all updates
- [ ] Single transaction (atomic)
- [ ] All relationships loaded for response

### COMPONENT_API Risk
Reusing existing components:
- [ ] PropTypes verified (not assumed)
- [ ] Hook return structures verified
```

---

### Step 6: Completeness Check

Before outputting, verify all required sections:

```markdown
## Completeness Checklist

### Required for All
- [ ] Feature title and summary
- [ ] Scope (in/out)
- [ ] Acceptance criteria

### Required for Backend
- [ ] Endpoints defined
- [ ] Models/fields specified
- [ ] Enum VALUES listed (not names)
- [ ] Access control specified

### Required for Frontend
- [ ] Components listed
- [ ] Reused component APIs verified
- [ ] State management approach

### Required for Fullstack
- [ ] All backend sections
- [ ] All frontend sections
- [ ] API contract (request/response shapes)
```

If incomplete:
```
❌ Spec incomplete. Missing sections:
- [ ] Enum VALUES not specified
- [ ] Access control deps not chosen

Please provide these before proceeding.
```

---

### Step 7: Output Spec File

**Location**: `docs/features/feature_{name}.md`

```markdown
---
title: {Feature Title}
status: draft
created: {YYYY-MM-DD}
author: {user}
type: {CRUD|Integration|UI|Enhancement|Fullstack}
complexity: {TRIVIAL|SIMPLE|COMPLEX}
---

# {Feature Title}

## Summary
{2-3 sentences describing the feature}

## Goals
- Goal 1
- Goal 2

## Scope

### In Scope
- Item 1

### Out of Scope
- Item 1

## Technical Specification

{Filled sections based on feature type}

## Risk Flags

{Auto-generated risk flags}

## Acceptance Criteria

- [ ] Criterion 1
- [ ] Criterion 2

## Open Questions

- Question 1?

---

**Next Steps**:
1. Review this spec
2. Fill any [TODO] sections
3. Run `/spec-review docs/features/feature_{name}.md`
```

---

## Output

```
✓ Spec drafted: docs/features/feature_advisor_co_ownership.md

Completeness: 85%
Missing:
- [ ] Acceptance criteria (3+ required)

Risk flags detected:
- ⚠️ ENUM_VALUE: FirmRole enum used
- ⚠️ MULTI_MODEL: Touches User, Advisor, FirmUserMembership

Next steps:
1. Review and complete spec
2. Run: /spec-review docs/features/feature_advisor_co_ownership.md
```

---

## Integration with Workflow

```
/spec-draft "feature name"
       ↓
Human reviews, completes [TODO]s
       ↓
/spec-review docs/features/feature_X.md
       ↓
GitHub issues created
       ↓
/orchestrate <issue>
```

---

## Tips

- **Be specific early**: Vague specs create vague issues
- **List enum VALUES**: This prevents 26% of failures
- **Verify component APIs**: This prevents 17% of failures
- **Map fields to models**: This prevents 13% of failures
- **When unsure**: Mark as [TODO] and flag as Open Question
