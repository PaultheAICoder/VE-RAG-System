---
description: Generate pre-implementation test plan with edge cases
argument-hint: [issue-number or spec-path]
---

# Test Plan Command

**Role**: Test Architect (TDD approach)

---

## Usage

```bash
/test-plan 239
/test-plan #239
/test-plan specs/engine-v2-proration-addendum.md
```

---

## Process

### Step 1: Identify Source

**If issue number provided:**
```bash
gh issue view $ISSUE --json number,title,body
```

**If spec path provided:**
```bash
cat $SPEC_PATH
```

### Step 2: Find Related MAP-PLAN (if exists)

```bash
ls -la .agents/outputs/map-plan-$ISSUE-*.md 2>/dev/null | tail -1
```

If found, read it for implementation context.

### Step 3: Spawn TEST-PLANNER Agent

**CRITICAL**: Use Task tool to spawn the agent.

```
Task(
  subagent_type='general-purpose',
  description='TEST-PLANNER for issue N',
  prompt='''You are the TEST-PLANNER agent.

Read your agent definition: .claude/agents/test-planner.md

Issue/Spec Context:
[Insert issue body or spec content]

MAP-PLAN Reference (if exists):
[Insert MAP-PLAN artifact path]

Generate a comprehensive test plan with:
1. Testable requirements extracted from spec/issue
2. Existing test coverage analysis
3. Test matrix (happy path, boundary, error cases)
4. Edge cases derived from formulas
5. Test function signatures

Write artifact to: .agents/outputs/test-plan-{issue}-{mmddyy}.md

End with: AGENT_RETURN: test-plan-{issue}-{mmddyy}.md
'''
)
```

### Step 4: Validate Artifact

```bash
# Check file exists
ls -la .agents/outputs/test-plan-$ISSUE-*.md

# Check for AGENT_RETURN
grep "AGENT_RETURN" .agents/outputs/test-plan-$ISSUE-*.md
```

### Step 5: Report

```
Test plan generated for issue #239

Artifact: .agents/outputs/test-plan-239-011226.md

Test cases identified: 12
- P0 (critical): 4
- P1 (important): 6
- P2 (edge): 2

Next steps:
- Review test plan
- Run /orchestrate 239 to implement (will use test plan)
- Or run /patch 239 to implement manually
```

---

## Output

Artifact: `.agents/outputs/test-plan-{issue}-{mmddyy}.md`

Contains:
- Requirements analysis
- Existing coverage gaps
- Test matrix with priorities
- Formula-derived edge cases
- Test function signatures
- Regression risks

---

## Integration

### With Orchestrate

Test plan is automatically used by PATCH agent when it exists:

```bash
# PATCH agent checks for test plan
ls .agents/outputs/test-plan-$ISSUE-*.md
```

### Standalone

Can be run independently before manual implementation:

```bash
/test-plan 239
# Review output
# Implement tests manually
```
