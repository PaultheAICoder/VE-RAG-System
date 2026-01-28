---
title: Feature Specification Template
purpose: Complete template for new features
version: 1.0
---

# Feature Specification Templates

Use these templates based on feature type. Copy the appropriate template and fill in all sections.

---

## Template: CRUD Feature

```markdown
---
title: {Resource} Management
status: draft
created: {YYYY-MM-DD}
type: CRUD
complexity: SIMPLE
---

# {Resource} Management

## Summary
Add ability to create, read, update, and delete {resources} for {parent resource}.

## Goals
- Users can create new {resources}
- Users can view list of {resources}
- Users can edit existing {resources}
- Users can delete {resources}

## Scope

### In Scope
- CRUD operations for {resource}
- Validation of required fields
- Access control

### Out of Scope
- Bulk operations
- Import/export
- {Other limitations}

---

## Data Model

### {Resource} Entity

| Field | Type | Required | Default | Validation |
|-------|------|----------|---------|------------|
| id | UUID | Yes | auto | |
| {field} | {type} | {Yes/No} | {value} | {rules} |
| account_id | UUID | Yes | - | FK to Account |
| created_at | datetime | Yes | now() | |
| updated_at | datetime | Yes | now() | |

### Enums

#### {EnumName}

| Python Name | **VALUE** (use this in frontend) | Description |
|-------------|----------------------------------|-------------|
| {NAME} | "{VALUE}" | {description} |

⚠️ **CRITICAL**: Frontend must use VALUES in the right column, not Python names.

### Relationships

| Related Model | Type | FK Location | Cascade |
|---------------|------|-------------|---------|
| Account | belongs_to | {resource}.account_id | - |
| {Model} | {type} | {location} | {behavior} |

---

## API Specification

### Endpoints

| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| POST | /accounts/{account_id}/{resources} | Create | require_account_access |
| GET | /accounts/{account_id}/{resources} | List | require_account_access |
| GET | /accounts/{account_id}/{resources}/{id} | Get one | require_account_access |
| PATCH | /accounts/{account_id}/{resources}/{id} | Update | require_account_access |
| DELETE | /accounts/{account_id}/{resources}/{id} | Delete | require_account_owner |

### Request/Response Schemas

#### Create Request
```json
{
  "field1": "string (required)",
  "field2": "number (optional)",
  "role": "ENUM_VALUE"
}
```

#### Response
```json
{
  "id": "uuid",
  "field1": "string",
  "field2": "number",
  "role": "ENUM_VALUE",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

#### Error Responses
- 401: Not authenticated
- 403: Not authorized (wrong role)
- 404: Resource not found
- 422: Validation error

---

## Access Control

| Action | Allowed Roles | Dependency |
|--------|---------------|------------|
| Create | Account member | require_account_access |
| Read | Account member | require_account_access |
| Update | Account member | require_account_access |
| Delete | Account owner only | require_account_owner |

---

## Frontend Specification

### Components

| Component | Location | Purpose |
|-----------|----------|---------|
| {Resource}List | components/{resources}/ | Display list |
| {Resource}Form | components/{resources}/ | Create/edit form |

### Hooks

| Hook | Purpose | Returns |
|------|---------|---------|
| use{Resources} | Fetch list | { data, isLoading, error } |
| use{Resource} | Fetch one | { data, isLoading, error } |
| useCreate{Resource} | Create mutation | { mutate, isLoading } |
| useUpdate{Resource} | Update mutation | { mutate, isLoading } |
| useDelete{Resource} | Delete mutation | { mutate, isLoading } |

### Components to Reuse

| Component | Import From | Props Used |
|-----------|-------------|------------|
| DataTable | components/common/ | columns, data, onRowClick |
| HeaderActions | components/common/ | title, actions |

⚠️ Verify PropTypes before using.

---

## Acceptance Criteria

- [ ] Can create {resource} with required fields
- [ ] Can view list of {resources}
- [ ] Can edit existing {resource}
- [ ] Can delete {resource} (owner only)
- [ ] Validation errors shown for invalid input
- [ ] Loading states displayed
- [ ] Error states handled gracefully
- [ ] Backend tests pass
- [ ] Frontend builds without errors

---

## Risk Flags

- [ ] **ENUM_VALUE**: Verify frontend uses VALUES from table above
- [ ] **ACCESS_CONTROL**: Verify deps match table above

---

## Open Questions

- {Question 1}?
```

---

## Template: Fullstack Feature

```markdown
---
title: {Feature Name}
status: draft
created: {YYYY-MM-DD}
type: Fullstack
complexity: COMPLEX
---

# {Feature Name}

## Summary
{2-3 sentences}

## Goals
- Goal 1
- Goal 2

## Scope

### In Scope
- Item 1

### Out of Scope
- Item 1

---

## Backend Specification

### Models Affected

| Model | Changes | Risk |
|-------|---------|------|
| {Model1} | {Add field X} | |
| {Model2} | {Update relationship} | MULTI_MODEL |

⚠️ **MULTI_MODEL Risk**: If >1 model updated in single operation, service must coordinate atomically.

### New/Modified Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| | | |

### Enums

| Enum | VALUES (use in frontend) |
|------|--------------------------|
| {Name} | "VALUE1", "VALUE2" |

### Service Logic

```
1. Validate input
2. Check permissions
3. Begin transaction
4. Update Model1
5. Update Model2
6. Commit transaction
7. Return response with all relationships loaded
```

---

## Frontend Specification

### Components to Create

| Component | Purpose |
|-----------|---------|
| | |

### Components to Reuse

| Component | Location | API (verify!) |
|-----------|----------|---------------|
| | | |

### State Management

| Data | Source | Cache Key |
|------|--------|-----------|
| | | |

---

## API Contract

### Request
```json
{
}
```

### Response
```json
{
}
```

---

## Acceptance Criteria

- [ ] {Criterion 1}
- [ ] {Criterion 2}
- [ ] Backend: `ruff check . && pytest -q` passes
- [ ] Frontend: `npm run lint && npm run build` passes

---

## Risk Flags

- [ ] **ENUM_VALUE**: {List enums if any}
- [ ] **MULTI_MODEL**: {List models if >1}
- [ ] **COMPONENT_API**: {List components to verify}

---

## Open Questions

- {Question}?
```

---

## Template: UI Component

```markdown
---
title: {Component Name} Component
status: draft
created: {YYYY-MM-DD}
type: UI Component
complexity: SIMPLE
---

# {Component Name} Component

## Summary
{1-2 sentences}

## Location
`frontend/src/components/{path}/{ComponentName}.jsx`

---

## Props

| Prop | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| | | | | |

## State

| State | Type | Initial | Description |
|-------|------|---------|-------------|
| | | | |

## Events/Callbacks

| Event | Signature | Description |
|-------|-----------|-------------|
| | | |

---

## Components to Reuse

| Component | Import | Props Used |
|-----------|--------|------------|
| | | |

⚠️ Verify PropTypes before assuming API.

---

## Behavior

### User Interactions
1. User does X → Component does Y
2. ...

### Loading State
- Show {loading indicator}

### Error State
- Show {error message}

### Empty State
- Show {empty message}

---

## Acceptance Criteria

- [ ] Renders correctly with all prop combinations
- [ ] Handles loading state
- [ ] Handles error state
- [ ] Handles empty state
- [ ] Accessible (keyboard nav, aria labels)
- [ ] `npm run lint && npm run build` passes

---

## Open Questions

- {Question}?
```

---

## Quick Reference: Required Sections by Type

| Section | CRUD | Fullstack | UI | Enhancement |
|---------|------|-----------|----|----|
| Data Model | ✅ | ✅ | ❌ | Maybe |
| Enums (with VALUES) | ✅ | ✅ | ❌ | Maybe |
| API Endpoints | ✅ | ✅ | ❌ | Maybe |
| Access Control | ✅ | ✅ | ❌ | Maybe |
| Components | ✅ | ✅ | ✅ | Maybe |
| Props/State | ❌ | Maybe | ✅ | Maybe |
| Risk Flags | ✅ | ✅ | ✅ | ✅ |
| Acceptance Criteria | ✅ | ✅ | ✅ | ✅ |
