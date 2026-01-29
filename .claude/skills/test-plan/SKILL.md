---
name: test-plan
version: 1.0
description: Generate test plan with edge cases before implementation
argument-hint: [issue-number]
---

# Test Plan Skill (v1.0)

Pre-implementation test planning with systematic edge case generation.

## When to Use

- Before implementing features with calculations/formulas
- When spec defines business rules that need test coverage
- For complex bug fixes with multiple edge cases
- When you want TDD approach

## Workflow Position

```
MAP-PLAN → TEST-PLANNER → PATCH → PROVE
```

TEST-PLANNER runs AFTER planning, BEFORE implementation.

## Usage

```bash
# Generate test plan for an issue
/test-plan 239

# Generate test plan from a spec (without issue)
/test-plan specs/engine-v2-proration-addendum.md
```

## Process

1. **Extract requirements** — Parse spec/issue for testable behavior
2. **Analyze coverage** — Find existing tests, identify gaps
3. **Generate matrix** — Derive cases (happy, boundary, error)
4. **Formula edge cases** — Systematically derive from calculations
5. **Test signatures** — Provide function stubs for PATCH to implement

## Output

- Artifact: `.agents/outputs/test-plan-{issue}-{date}.md`
- Test matrix with priorities (P0, P1, P2)
- Test function signatures (not full implementations)
- Regression risk list

## Example Output

```markdown
## Test Matrix - Income Proration

| Test Case | Input | Expected | Priority |
|-----------|-------|----------|----------|
| July start | start_month=7 | 0.5 | P0 |
| Null month | start_month=None | 1.0 | P0 |
| Invalid range | start=9, end=3 | ValidationError | P1 |

## Test Signatures

def test_income_july_start_prorates_half():
    """Income starting July = 6/12 of annual"""
    # Arrange: income with start_month=7
    # Act: project_values()
    # Assert: result == annual * 0.5
```

## Integration with Orchestrate

When `/orchestrate` is run with `--with-tests` flag:
```
/orchestrate 239 --with-tests
```

The TEST-PLANNER agent runs automatically after MAP-PLAN.

## See Also

- `.claude/agents/test-planner.md` — Agent definition
- `.claude/rules/testing.md` — Testing patterns
