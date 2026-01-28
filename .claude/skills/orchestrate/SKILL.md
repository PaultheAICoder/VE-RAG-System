---
name: orchestrate
version: 3.0
description: Multi-agent workflow with self-learning capabilities
---

# Orchestrate Skill (v3.0)

Issue-driven workflow with pattern learning and outcome tracking.

## What's New in v3.0

- **Self-learning**: Agents read patterns.md before each run
- **Outcome tracking**: PROVE records success/failure to memory/
- **Slimmed agents**: 50% smaller, faster context loading
- **Shared behaviors**: All agents inherit from _base.md

## Workflow

```
Issue → Classify → Branch → Agents → Verify → Record → PR
```

## Agent Sequence

**TRIVIAL/SIMPLE**:
1. MAP-PLAN → `.agents/outputs/map-plan-{issue}-{date}.md`
2. TEST-PLANNER (if `--with-tests`) → `.agents/outputs/test-plan-{issue}-{date}.md`
3. CONTRACT (if fullstack) → `.agents/outputs/contract-{issue}-{date}.md`
4. PATCH → `.agents/outputs/patch-{issue}-{date}.md`
5. PROVE → `.agents/outputs/prove-{issue}-{date}.md`

**COMPLEX**:
1. MAP → `.agents/outputs/map-{issue}-{date}.md`
2. PLAN → `.agents/outputs/plan-{issue}-{date}.md`
3. TEST-PLANNER (if `--with-tests`) → `.agents/outputs/test-plan-{issue}-{date}.md`
4. CONTRACT (if fullstack)
5. PATCH
6. PROVE

**Recommended**: Use `--with-tests` for issues involving calculations, formulas, or complex business rules.

## Learning Loop

After each issue:
1. PROVE records outcome to `.claude/memory/metrics.jsonl`
2. If BLOCKED, records failure to `.claude/memory/failures.jsonl`
3. Weekly `/learn` extracts patterns
4. Agents read `patterns.md` on next run

## Key Files

- `.claude/agents/_base.md` — Shared agent behaviors
- `.claude/memory/patterns.md` — Learned patterns
- `.claude/memory/failures.jsonl` — Failure log
- `.claude/memory/metrics.jsonl` — Success metrics

## Usage

```bash
/orchestrate 184
/orchestrate 184 --with-tests    # Include TEST-PLANNER phase
```

See `.claude/commands/orchestrate.md` for full details.
