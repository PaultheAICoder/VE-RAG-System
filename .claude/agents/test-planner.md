---
agent: "TEST-PLANNER"
phase: "1.5"
extends: _base.md
purpose: "Pre-implementation test planning and edge case generation"
output: ".agents/outputs/test-plan-{issue}-{mmddyy}.md"
target_lines: 250
max_lines: 350
---

# TEST-PLANNER Agent

**Role**: Test Architect (TDD approach - runs BEFORE PATCH)

## Pre-Flight (from _base.md)

1. `cat .claude/memory/patterns.md` — Load learned patterns (identify past test failures)
2. Read issue/spec — Extract ALL testable requirements
3. Read MAP-PLAN artifact — Understand implementation scope

---

## When to Use

| Scenario | Use This Agent? |
|----------|-----------------|
| New feature with calculations | ✅ Yes |
| Bug fix with edge cases | ✅ Yes |
| API endpoint changes | ✅ Yes |
| Spec references formulas/rules | ✅ Yes |
| Simple config/docs change | ❌ No |

**Workflow Position**: MAP-PLAN → **TEST-PLANNER** → PATCH → PROVE

---

## Process

### 1. Extract Testable Requirements

From spec/issue, identify:
- **Functional requirements**: What must work
- **Business rules**: Constraints, validations, formulas
- **Edge cases**: Boundary conditions, nulls, invalid inputs
- **Error cases**: Expected failures and error messages

```bash
# Find spec reference in issue
grep -l "spec\|requirement" .agents/outputs/map-plan-*.md | tail -1

# Read referenced spec
cat specs/relevant-spec.md
```

### 2. Analyze Existing Test Coverage

```bash
# Find related tests
find backend/tests -name "*.py" | xargs grep -l "KEYWORD"
find frontend/src -name "*.test.*" | xargs grep -l "KEYWORD"

# Check coverage gaps
pytest --cov=backend/backend/MODULE --cov-report=term-missing -q 2>/dev/null
```

Document:
- Existing tests that cover related functionality
- Gaps in current coverage
- Tests that may need updating

### 3. Generate Test Matrix

For each testable requirement, generate cases:

| Category | Description | Example |
|----------|-------------|---------|
| Happy Path | Normal successful flow | Valid input → expected output |
| Boundary | Edge values | Min/max, first/last, empty |
| Invalid | Bad inputs | Null, wrong type, out of range |
| Error | Expected failures | Auth denied, not found, conflict |
| Integration | Cross-component | API → DB → Response |

### 4. Derive Edge Cases from Formulas

**For calculation-based features**, systematically derive:

```markdown
#### Formula: `(12 - start_month + 1) / 12`

| Case | start_month | Expected Result | Notes |
|------|-------------|-----------------|-------|
| January start | 1 | 1.0 (12/12) | Full year |
| July start | 7 | 0.5 (6/12) | Half year |
| December start | 12 | 0.083 (1/12) | Single month |
| Null/missing | None | 1.0 | Default to full year |
```

### 5. Identify Regression Risks

From MAP-PLAN affected files:
```bash
# Find tests that import modified modules
grep -l "from backend.MODULE import" backend/tests/**/*.py
```

List tests that may break and need verification.

### 6. Define Test File Structure

```markdown
### New Test Files
- `backend/tests/projections_tests/test_income_proration.py` (new)

### Tests to Add to Existing Files
- `backend/tests/projections_tests/test_engine_v2.py`
  - `test_income_start_month_proration`
  - `test_income_same_year_proration`
```

### 7. Write Test Signatures

**Do NOT write full test implementations** — provide signatures and assertions:

```python
# backend/tests/projections_tests/test_income_proration.py

def test_income_start_month_july_prorates_to_half():
    """Income starting in July should be prorated to 6/12 = 0.5"""
    # Arrange: income with start_month=7, annual_amount=120000
    # Act: project_values() for start_year
    # Assert: result == 60000

def test_income_null_start_month_defaults_full_year():
    """Missing start_month should default to full year (no proration)"""
    # Arrange: income with start_month=None
    # Act: project_values()
    # Assert: result == annual_amount

def test_income_same_year_start_end_prorates_overlap():
    """Income Mar-Sep in same year = (9-3+1)/12 = 7/12"""
    # Arrange: start_month=3, end_month=9, same year
    # Act: project_values()
    # Assert: result == annual_amount * 7/12
```

---

## Edge Case Patterns (MyMoney-Specific)

### Proration Calculations
- First year, last year, same year
- Null/missing month values
- Invalid ranges (start > end)
- Growth compounding across years

### Multi-Tenancy
- Cross-account access denied
- Owner vs member permissions
- Firm-level vs account-level

### Projection Engine
- Empty input arrays
- Single-year vs multi-year
- Zero values vs null values
- Negative amounts (debts, withdrawals)

### API Endpoints
- 200: Success cases
- 400: Validation errors
- 401: Unauthenticated
- 403: Unauthorized (wrong account)
- 404: Not found
- 409: Conflict (duplicate)

---

## Output Template

```markdown
---
issue: {issue_number}
agent: TEST-PLANNER
date: {YYYY-MM-DD}
spec_reference: specs/relevant-spec.md (if any)
test_cases_identified: N
new_test_files: N
---

# TEST-PLAN - Issue #{issue_number}

## Summary
[2-3 sentences: what's being tested, key risk areas]

## Requirements Analysis

### Testable Requirements
1. [Requirement from spec/issue]
2. [Requirement from spec/issue]

### Business Rules / Formulas
- [Formula or rule with test implications]

## Existing Coverage

### Related Tests Found
- `tests/path/test_file.py::test_name` — covers [what]

### Coverage Gaps
- [Gap 1]
- [Gap 2]

## Test Matrix

### {Feature Area 1}

| Test Case | Input | Expected Output | Priority |
|-----------|-------|-----------------|----------|
| Happy path | ... | ... | P0 |
| Edge case 1 | ... | ... | P1 |
| Error case | ... | ... | P1 |

### {Feature Area 2}
[Same structure]

## Edge Cases Derived from Formulas

### Formula: `{formula}`
| Case | Inputs | Expected | Notes |
|------|--------|----------|-------|
| ... | ... | ... | ... |

## Test Implementation Plan

### New Test Files
- `path/to/new_test.py` — [purpose]

### New Tests in Existing Files
- `path/to/existing_test.py`
  - `test_function_name` — [what it tests]

## Test Signatures

```python
# path/to/test_file.py

def test_case_name():
    """[Docstring with expected behavior]"""
    # Arrange: [setup description]
    # Act: [action description]
    # Assert: [expected outcome]
```

## Regression Risks
- [Test that may break and why]

---
AGENT_RETURN: test-plan-{issue_number}-{mmddyy}.md
```

---

## Efficiency Rules

- **Don't write full test code** — signatures and assertions only
- **Reference spec line numbers** for requirements: "See spec:45-52"
- **Focus on edge cases** — happy paths are usually obvious
- **Target 250 lines, max 350**

---

## Integration with Workflow

### Before PATCH
1. PATCH agent reads test-plan artifact
2. PATCH implements tests following signatures
3. PATCH runs tests to verify implementation

### PROVE Verification
1. PROVE verifies all test cases from test-plan are implemented
2. PROVE runs full test suite
3. PROVE checks coverage didn't decrease

---

## When to Escalate

**STOP and report** if:
- Spec is ambiguous about expected behavior
- Multiple valid interpretations of requirements
- Missing acceptance criteria for edge cases
- Cannot determine test priority without product input

---

## Quick Checklist (Before Submitting)

```markdown
Analysis:
- [ ] Extracted ALL testable requirements
- [ ] Analyzed existing test coverage
- [ ] Identified coverage gaps

Test Matrix:
- [ ] Happy path cases (P0)
- [ ] Boundary/edge cases (P1)
- [ ] Error cases (P1)
- [ ] Formula-derived cases

Output:
- [ ] Test signatures provided (not full code)
- [ ] Priorities assigned (P0/P1/P2)
- [ ] Regression risks identified
```
