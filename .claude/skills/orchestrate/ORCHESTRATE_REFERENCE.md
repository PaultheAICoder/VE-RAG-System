# Orchestrate Command — Complete Reference

This is the detailed reference for the orchestrate workflow. The main skill definition is in [SKILL.md](./SKILL.md).

## Workflow Overview

**ORCHESTRATION ONLY**: You are a **conductor**, not a performer.

### Conditional Workflow Paths
- **TRIVIAL/SIMPLE tasks**: **MAP-PLAN → PATCH → PROVE** (3-agent)
- **COMPLEX tasks**: **MAP → PLAN → PATCH → PROVE** (4-agent)

## Repo Structure
- **Monorepo root**: `mymoney-dev/`
- **Backend repo root**: `backend/` (FastAPI + SQLAlchemy + pytest)
- **Frontend repo root**: `frontend/` (Vite + React)

## Tech Stack Snapshot
- **Backend**: FastAPI, SQLAlchemy, Alembic, pytest (SQLite in-memory), ruff
- **Frontend**: Vite, React, Tailwind, `react-router-dom` (migration in progress), axios wrapper in `frontend/src/api.js`, ESLint/Vitest

---

## Critical Rules

**YOU MUST:**
- Require a **GitHub Issue** as source of truth (issue number = build number)
- Extract/verify the issue via GitHub CLI
- Coordinate agents sequentially
- Require each agent to write a `.agents/outputs/...` artifact
- Summarize between phases and carry forward filenames
- Enforce `.claude/rules.md`

**YOU MUST NEVER:**
- Implement the feature/bug yourself
- Edit code directly
- Bypass the MAP/PLAN/PATCH/PROVE sequence

---

## Output Files (Mandatory)

All outputs are written to:

`mymoney-dev/.agents/outputs/`

Naming convention:

`{skill}-{issue_number}-{mmddyy}.md`

**TRIVIAL/SIMPLE (MAP-PLAN workflow):**
- `map-plan-{issue}-{date}.md`
- `patch-{issue}-{date}.md`
- `prove-{issue}-{date}.md`

**COMPLEX (Full workflow):**
- `map-{issue}-{date}.md`
- `plan-{issue}-{date}.md`
- `patch-{issue}-{date}.md`
- `prove-{issue}-{date}.md`

Each agent MUST end with:

`AGENT_RETURN: <filename>`

---

## Usage

```
/orchestrate gh issue #184
/orchestrate #184
/orchestrate https://github.com/<org>/<repo>/issues/184
```

If the user does not provide an issue, instruct them to create one with:
- `bug "..."` or
- `feature "..."`

---

## Step 0 — Verify Issue Exists (authoritative)

Extract `ISSUE_NUMBER`, then:

```bash
gh issue view $ISSUE_NUMBER --json number,title,body -q '.number'
```

Capture context to pass into MAP/MAP-PLAN:

```bash
gh issue view $ISSUE_NUMBER --json title,body -q '.title + "\n\n" + .body'
```

---

## Step 1 — Initialize Run Date + Filenames

Set:
- `RUN_DATE = $(date +%m%d%y)`
- Ensure: `mkdir -p .agents/outputs`

Filenames:
- `MAP_FILE        = .agents/outputs/map-${ISSUE_NUMBER}-${RUN_DATE}.md`
- `PLAN_FILE       = .agents/outputs/plan-${ISSUE_NUMBER}-${RUN_DATE}.md`
- `MAP_PLAN_FILE   = .agents/outputs/map-plan-${ISSUE_NUMBER}-${RUN_DATE}.md`
- `CONTRACT_FILE   = .agents/outputs/contract-${ISSUE_NUMBER}-${RUN_DATE}.md`
- `PATCH_FILE      = .agents/outputs/patch-${ISSUE_NUMBER}-${RUN_DATE}.md`
- `PROVE_FILE      = .agents/outputs/prove-${ISSUE_NUMBER}-${RUN_DATE}.md`

---

## Step 1.5 — Classify Task Complexity (MANDATORY)

Classification criteria:

**TRIVIAL:** docs, config tweaks, small renames, deleting unused code

**SIMPLE:** single-area changes, 1–3 files, straightforward bug fix or UI tweak

**COMPLEX:** new endpoints, DB migrations, cross-module refactors, multi-step UI features, changes spanning backend + frontend

Routing:
- TRIVIAL/SIMPLE → MAP-PLAN
- COMPLEX → MAP then PLAN

---

## Step 2 — Run Agents

### Path A: TRIVIAL/SIMPLE
1) Run **MAP-PLAN agent** (`.claude/agents/map-plan.md`) → `MAP_PLAN_FILE`
2) If the task is **fullstack** (backend + frontend), run **CONTRACT agent** (`.claude/agents/contract.md`) → `CONTRACT_FILE`
3) Run **PATCH agent** (`.claude/agents/patch.md`) → `PATCH_FILE`
4) Run **PROVE agent** (`.claude/agents/prove.md`) → `PROVE_FILE`

### Path B: COMPLEX
1) Run **MAP agent** (`.claude/agents/map.md`) → `MAP_FILE`
2) Run **PLAN agent** (`.claude/agents/plan.md`) → `PLAN_FILE`
3) If the task is **fullstack** (backend + frontend), run **CONTRACT agent** (`.claude/agents/contract.md`) → `CONTRACT_FILE`
4) Run **PATCH agent** (`.claude/agents/patch.md`) → `PATCH_FILE`
5) Run **PROVE agent** (`.claude/agents/prove.md`) → `PROVE_FILE`

---

## Step 3 — Fullstack Coordination Rule (MANDATORY CONTRACT)

If the change impacts **both** backend and frontend:
- Backend changes define the contract (routes, request/response, error semantics)
- Frontend must follow the contract

**MANDATORY contract artifact for any fullstack change:**
- `.agents/outputs/contract-{issue}-{date}.md`

PATCH must treat the Contract Artifact as **authoritative** for:
- Route paths, request/response schemas
- Error semantics (401/403/404/422) and payload shapes
- Account scoping (path prefix vs query param)
- Frontend integration notes (how to call via `fetchData`)

---

## Step 4 — Verification Gates

PROVE must run only the relevant commands, based on what changed:

**Backend touched:**
```bash
cd backend && ruff check .
cd backend && pytest -q
```

**Frontend touched:**
```bash
cd frontend && npm run lint
cd frontend && npm run build
```

If a command fails, PROVE status is **BLOCKED** and must include error output + unblock steps.

---

## PR Workflow
- Keep `main` green. If this change is non-trivial, use `/pr <number>` to produce a PR-ready plan, contract artifact (if fullstack), and verification checklist.

---

## Agent Communication

Each agent receives:
1. **Issue context** (title, body, number)
2. **Previous artifacts** (MAP/MAP-PLAN, PLAN if applicable)
3. **Project constraints** (`.claude/rules.md`)
4. **Tech stack context** (`.claude/context/project_stack.md`)

Each agent produces:
1. **Markdown artifact** in `.agents/outputs/`
2. **AGENT_RETURN: <filename>** at the end
3. **Summary** of work completed
4. **Handoff notes** for next agent (if applicable)

---

## Troubleshooting

**Issue not found:**
```bash
gh issue view <number>
# Error: Could not resolve to an issue
```
→ Verify issue exists, check repo access

**Agent fails:**
- Review agent artifact for error details
- Check if constraints were violated
- Verify all prerequisites met

**PROVE blocked:**
- Review linting/test output in PROVE artifact
- Fix issues before proceeding
- May need to run PATCH again with fixes

**Fullstack confusion:**
- Always generate CONTRACT artifact for backend + frontend changes
- Backend defines API contract
- Frontend implements to contract
