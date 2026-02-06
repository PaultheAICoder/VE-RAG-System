---
agent: "SESSION-MIGRATION"
extends: _base.md
purpose: "Migrate frontend to new backend identity/roles + React Router incrementally"
output: ".agents/outputs/session-migration-{mmddyy}.md"
target_lines: 100
max_lines: 150
---

# SESSION-MIGRATION Agent

**Role**: Migration Coordinator (INCREMENTAL CHANGES)

## Core Principles

- **Small, green steps**: Every change leaves builds/tests passing
- **Centralize session**: Single `SessionContext` owns identity state
- **One API source**: Use `frontend/src/api.js` only
- **Contract-first**: Fullstack changes need CONTRACT artifact

---

## Backend Mental Model

```
User (JWT subject)
  └── FirmUserMembership (FirmRole)
        └── Firm
              └── Accounts[]
                    └── AccountMembers (AccountMemberRole)
                          └── ClientProfiles
```

---

## Migration Phases

### Phase 1: Session Bootstrap

Implement `SessionContext` providing:
- `user` — from `GET /users/me`
- `firmMembership` — firm context
- `accounts[]` — accessible accounts
- `activeAccountId` — selected account
- `setActiveAccountId()` — persists to localStorage
- `isBootstrapped`, `bootstrapError`

**Bootstrap flow**:
1. `GET /users/me`
2. `GET /firms/{firm_id}/me` (if firm-scoped)
3. List accessible accounts
4. Set active account (from localStorage or default)

### Phase 2: React Router

- Add Router at top level
- Mount existing screens without refactors
- Add protected routes using SessionContext

### Phase 3: Role-Aware Navigation

Show/hide nav based on:
- `SystemRole` (admin)
- `FirmRole` (advisor)
- `AccountMemberRole` (client)

Ensure API calls use `activeAccountId`.

### Phase 4: Module Adoption

For each module (assets, incomes, debts, expenses):
- Use contexts + `api.js`
- Ensure account-scoped
- Keep config-driven rendering

---

## Guardrails

- ❌ Never break account-scoped prefixing
- ❌ No big-bang rewrites
- ✅ Keep 401 handling consistent
- ✅ Feature flags for parallel components if needed

---

## PR Deliverables

- Migration note in PR description
- CONTRACT artifact (if fullstack)
- Updated/new tests
