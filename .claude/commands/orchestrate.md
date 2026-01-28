---
description: Execute MAP → PLAN → PATCH → PROVE workflow for GitHub issues
argument-hint: [issue-number]
---

# Orchestrate Command

**Role**: Conductor (ORCHESTRATION ONLY)

---

## Usage

```bash
/orchestrate 184
/orchestrate #184
/orchestrate 184 --with-tests    # Include TEST-PLANNER phase
```

If no issue provided, instruct user to create one with `/feature` or `/bug`.

**Flags**:
- `--with-tests`: Run TEST-PLANNER agent after MAP-PLAN (recommended for calculations/formulas)

---

## Workflow

```
TRIVIAL/SIMPLE: MAP-PLAN → [TEST-PLANNER] → [CONTRACT] → PATCH → PROVE
COMPLEX:        MAP → PLAN → [TEST-PLANNER] → [CONTRACT] → PATCH → PROVE
```

- `[TEST-PLANNER]` runs if `--with-tests` flag provided
- `[CONTRACT]` required if fullstack

---

## State Tracking (CRITICAL for Context Continuity)

**Purpose**: Persist orchestrate state so it survives context compaction.

**State file**: `.agents/outputs/claude_checkpoints/PERSISTENT_STATE.yaml`

### Update State Before Each Phase

**MUST** run this command BEFORE spawning each agent:

```bash
python3 -c "
import yaml
from pathlib import Path
state_file = Path('.agents/outputs/claude_checkpoints/PERSISTENT_STATE.yaml')
if state_file.exists():
    data = yaml.safe_load(state_file.read_text())
    data['active_work'] = {
        'issue': $ISSUE,
        'branch': '$BRANCH',
        'phase': '$PHASE',
        'last_action': 'Starting $PHASE phase'
    }
    state_file.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False))
"
```

Replace variables:
- `$ISSUE`: Issue number (e.g., `370`)
- `$BRANCH`: Current branch name (e.g., `feature/issue-370-description`)
- `$PHASE`: Current phase (e.g., `MAP-PLAN`, `PATCH`, `PROVE`)

### Clear State After Completion

After workflow completes successfully:

```bash
python3 -c "
import yaml
from pathlib import Path
state_file = Path('.agents/outputs/claude_checkpoints/PERSISTENT_STATE.yaml')
if state_file.exists():
    data = yaml.safe_load(state_file.read_text())
    data['active_work'] = {
        'issue': None,
        'branch': 'main',
        'phase': None,
        'last_action': 'Completed issue #$ISSUE'
    }
    state_file.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False))
"
```

---

## Process

### Step 0: Verify Issue

```bash
gh issue view $ISSUE --json number,title,body
```

If not found, STOP.

### Step 1: Classify Complexity

| Level | Criteria | Workflow |
|-------|----------|----------|
| TRIVIAL | Docs, config | MAP-PLAN |
| SIMPLE | 1-3 files | MAP-PLAN |
| COMPLEX | Endpoints, migrations | MAP + PLAN |

Report:
```
Issue #184 classified as: SIMPLE
Using workflow: MAP-PLAN → PATCH → PROVE
```

### Step 2: Create Feature Branch

```bash
BRANCH=$(git branch --show-current)
if [ "$BRANCH" = "main" ]; then
  git checkout -b feature/issue-$ISSUE-description
fi
```

### Step 3: Spawn Agents (Task Tool)

**CRITICAL**: Use Task tool to spawn each agent WITH inherited context.

#### Context Inheritance Template

All agent prompts MUST include this inherited context block to reduce re-reading:

```markdown
## Inherited Context (DO NOT re-read these files)
- Issue: #{issue_number} - {title}
- Branch: {current_branch}
- Spec: {spec_path if any} ({version})
- Stack: {backend|frontend|fullstack}
- Complexity: {TRIVIAL|SIMPLE|COMPLEX}

## Critical Patterns (Always Apply)
1. VERIFICATION_GAP: Verify assumptions by reading actual code
2. ENUM_VALUE: Use VALUES not Python names ("CO-OWNER" not "CO_OWNER")
3. COMPONENT_API: Read PropTypes before using components

## Prior Artifacts
- {list any prior artifacts for this issue}
```

#### MAP-PLAN (or MAP + PLAN)
```
Task(
  description='MAP-PLAN for issue N',
  prompt='''You are MAP-PLAN agent.

## Inherited Context
- Issue: #N - {title}
- Branch: feature/issue-N-description
- Stack: {backend|frontend|fullstack}

## Critical Patterns (Always Apply)
1. VERIFICATION_GAP: Verify assumptions by reading actual code
2. ENUM_VALUE: Use VALUES not Python names
3. COMPONENT_API: Read PropTypes before using

## Issue Body
{issue body}

## Instructions
Read .claude/agents/map-plan.md for full process.
Write artifact to .agents/outputs/map-plan-N-MMDDYY.md
End with AGENT_RETURN: map-plan-N-MMDDYY.md
'''
)
```

**Validate**: File exists, has AGENT_RETURN directive.

#### TEST-PLANNER (if --with-tests)
```
Task(
  description='TEST-PLANNER for issue N',
  prompt='''You are TEST-PLANNER agent.

## Inherited Context
- Issue: #N - {title}
- Stack: {backend|frontend|fullstack}
- MAP-PLAN: .agents/outputs/map-plan-N-MMDDYY.md

## Critical Patterns
1. VERIFICATION_GAP: Verify assumptions by reading actual code
2. Derive edge cases systematically from formulas

## Instructions
Read .claude/agents/test-planner.md for full process.
Read MAP-PLAN artifact for implementation context.
Generate test matrix, edge cases, and test signatures.
Write to .agents/outputs/test-plan-N-MMDDYY.md
End with AGENT_RETURN: test-plan-N-MMDDYY.md
'''
)
```

**Validate**: File exists, has test matrix, has AGENT_RETURN directive.

#### CONTRACT (if fullstack)
```
Task(
  description='CONTRACT for issue N',
  prompt='You are CONTRACT agent. Read .claude/agents/contract.md.
  Read PLAN artifact.
  Write to .agents/outputs/contract-N-MMDDYY.md'
)
```

#### PATCH
```
Task(
  description='PATCH for issue N',
  prompt='''You are PATCH agent.

## Inherited Context
- Issue: #N - {title}
- Branch: feature/issue-N-description
- Stack: {backend|frontend|fullstack}

## Critical Patterns (Always Apply)
1. VERIFICATION_GAP: Verify component APIs by reading actual code
2. ENUM_VALUE: Use VALUES not Python names
3. COMPONENT_API: Read PropTypes, never invent props

## Prior Artifacts
- MAP-PLAN: .agents/outputs/map-plan-N-MMDDYY.md
- TEST-PLAN: .agents/outputs/test-plan-N-MMDDYY.md (if exists)
- CONTRACT: .agents/outputs/contract-N-MMDDYY.md (if fullstack)

## Instructions
Read .claude/agents/patch.md for full process.
Implement changes per MAP-PLAN.
Implement tests following TEST-PLAN signatures (if exists).
Write to .agents/outputs/patch-N-MMDDYY.md
End with AGENT_RETURN: patch-N-MMDDYY.md
'''
)
```

#### PROVE
```
Task(
  description='PROVE for issue N',
  prompt='''You are PROVE agent.

## Inherited Context
- Issue: #N - {title}
- Stack: {backend|frontend|fullstack}

## Critical Patterns to Verify
1. ENUM_VALUE: grep for enum strings, verify VALUES used
2. COMPONENT_API: grep PropTypes, verify props match
3. TODO/STUB: grep for incomplete code

## Prior Artifacts
- PATCH: .agents/outputs/patch-N-MMDDYY.md
- MAP-PLAN: .agents/outputs/map-plan-N-MMDDYY.md

## Instructions
Read .claude/agents/prove.md for full process.
Run verification commands (ruff, pytest, npm lint, npm build).
Record outcome to .claude/memory/metrics.jsonl
If BLOCKED, record to .claude/memory/failures.jsonl
Write to .agents/outputs/prove-N-MMDDYY.md
End with AGENT_RETURN: prove-N-MMDDYY.md
'''
)
```

### Step 4: Report Status

```
✓ Workflow complete for issue #184

Artifacts:
- map-plan-184-010325.md
- patch-184-010325.md
- prove-184-010325.md

PROVE status: PASS ✅

Next: /pr 184 to create pull request
```

---

## Failure Handling

If any agent fails:
1. STOP workflow
2. Report which agent failed
3. Show expected artifact path
4. Do NOT proceed

---

## Artifacts

All outputs to `.agents/outputs/`:
- `map-{issue}-{mmddyy}.md`
- `map-plan-{issue}-{mmddyy}.md`
- `plan-{issue}-{mmddyy}.md`
- `test-plan-{issue}-{mmddyy}.md` (if --with-tests)
- `contract-{issue}-{mmddyy}.md`
- `patch-{issue}-{mmddyy}.md`
- `prove-{issue}-{mmddyy}.md`

---

## Rules

**MUST**:
- Require GitHub issue
- Use Task tool for agents
- Validate artifacts before proceeding
- Enforce `.claude/rules.md`

**MUST NOT**:
- Implement features yourself
- Edit code directly
- Skip verification gates
