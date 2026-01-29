---
paths: ".agents/**/*.md"
---

# Orchestrate Workflow & Agent Efficiency Guidelines

This rule applies to all agent outputs in .agents/outputs/.

## Workflow Overview

The orchestrate workflow implements a **MAP → PLAN → PATCH → PROVE** pattern for issue-driven development.

```
GitHub Issue → MAP-PLAN → PATCH → PROVE → PR → Merge
```

### When to Use

Use the orchestrate workflow when:
- ✅ You have a GitHub issue (bug or feature)
- ✅ Change spans backend, frontend, or both
- ✅ You need automated verification and artifact tracking
- ✅ Issue requires multi-step implementation

**Primary command**: `/orchestrate <issue_number>`

---

## Workflow Phases

### Phase 1: MAP-PLAN (Investigation + Planning)

**Agent**: `.claude/agents/map-plan.md`
**Output**: `.agents/outputs/map-plan-{issue}-{mmddyy}.md`

**Purpose**:
- Investigate codebase and understand current state
- Design implementation approach
- Create file-by-file plan
- List acceptance criteria

**For SIMPLE/TRIVIAL issues**: Use combined MAP-PLAN agent
**For COMPLEX issues**: Use separate MAP + PLAN agents

**Key Sections**:
- Executive Summary (3-5 sentences)
- Investigation findings
- File-by-file implementation steps
- Acceptance criteria (checklist format)
- Verification gates for PROVE

**Target Length**: 500-600 lines

---

### Phase 1.5 (Optional): TEST-PLANNER (Pre-Implementation Test Planning)

**Agent**: `.claude/agents/test-planner.md`
**Output**: `.agents/outputs/test-plan-{issue}-{mmddyy}.md`

**When to use**: When `--with-tests` flag is provided, recommended for:
- Issues involving calculations or formulas
- Complex business rules with edge cases
- Bug fixes requiring regression tests
- TDD approach

**Purpose**:
- Extract testable requirements from spec/issue
- Analyze existing test coverage gaps
- Generate systematic test matrix (happy, boundary, error)
- Derive edge cases from formulas
- Provide test function signatures for PATCH

**Key Sections**:
- Requirements analysis
- Existing coverage gaps
- Test matrix with priorities (P0, P1, P2)
- Formula-derived edge cases
- Test signatures (stubs, not full code)

**Target Length**: 250-350 lines

**Timing**: Run AFTER MAP-PLAN, BEFORE CONTRACT/PATCH

---

### Phase 2 (Optional): CONTRACT (API Contract Definition)

**Agent**: `.claude/agents/contract.md`
**Output**: `.agents/outputs/contract-{issue}-{mmddyy}.md`

**When to use**: **MANDATORY for fullstack changes** (backend + frontend)

**Purpose**:
- Define backend ↔ frontend API contract
- Specify request/response schemas
- Document validation rules and error codes
- Clarify enum values (backend VALUE vs NAME)
- Examples for integration

**Key Sections**:
- Executive Summary
- Endpoint definitions (request/response)
- Enum definitions (backend VALUE must match frontend usage)
- Frontend integration notes
- Backward compatibility analysis

**Target Length**: 200-300 lines

**Timing**: Run AFTER PLAN, BEFORE PATCH

---

### Phase 3: PATCH (Implementation)

**Agent**: `.claude/agents/patch.md`
**Output**: `.agents/outputs/patch-{issue}-{mmddyy}.md`

**Purpose**:
- Implement changes exactly as planned
- Update tests
- Document issues encountered
- Verify implementation

**Key Sections**:
- Executive Summary
- Metrics table (files changed, tests added, lines modified)
- Files changed (with references to MAP-PLAN, NOT full code re-quotes)
- Issues encountered (CRITICAL - keep detailed)
- Verification performed (linting, tests)
- Deviations from PLAN
- Acceptance criteria status (reference MAP-PLAN, don't repeat)

---

### Phase 4: PROVE (Verification)

**Agent**: `.claude/agents/prove.md`
**Output**: `.agents/outputs/prove-{issue}-{mmddyy}.md`

**Purpose**:
- Run verification commands (linting, tests)
- Validate acceptance criteria
- Check for regressions
- Verify compliance with project rules
- Final sign-off for PR

**Key Sections**:
- Executive Summary with PASS/BLOCKED status
- Verification results (tests, linting)
- Acceptance criteria (simple pass/fail table)
- Comparison with PATCH (exceptions only)
- Compliance checklist
- Issues found (if any)
- Recommendation (APPROVED or BLOCKED)

---

## Agent Efficiency Guidelines (Phase 1 - Implemented Dec 2025)

### Critical Rules for ALL Agents

**DO**:
- ✅ Use YAML frontmatter for metadata
- ✅ Keep reports concise - focus on signal, not noise
- ✅ Include Executive Summary (3-5 sentences) in all reports
- ✅ Document issues encountered (high-value learning)
- ✅ Use consistent status indicators (✅/❌)

**DON'T**:
- ❌ Be verbose - avoid redundancy
- ❌ Include low-value appendices
- ❌ Create excessive subsections (max 3 heading levels)
- ❌ Repeat information across phases

---

### MAP-PLAN Agent Efficiency Rules

**Output Target**: 500-600 lines (currently averaging 698)

**DO**:
- ✅ Use YAML frontmatter:
  ```yaml
  ---
  issue: 26
  agent: MAP-PLAN
  date: 2025-12-23
  complexity: SIMPLE
  stack: backend
  ---
  ```
- ✅ Keep Executive Summary to 3-5 sentences
- ✅ List acceptance criteria as simple checklist (don't repeat in PATCH)
- ✅ Reference file line numbers instead of quoting existing code

**DON'T**:
- ❌ Quote full existing code (use: "See models.py:45-67")
- ❌ Create "Future Enhancements" sections (out of scope)
- ❌ Duplicate risk sections (consolidate to one)
- ❌ Include API documentation examples (reference Swagger `/docs`)

**Target Reduction**: 17% (698 → 580 lines)

---

### PATCH Agent Efficiency Rules

**Output Target**: 280-300 lines (currently averaging 475)

**DO**:
- ✅ Use YAML frontmatter with metrics:
  ```yaml
  ---
  issue: 26
  agent: PATCH
  date: 2025-12-23
  status: Complete
  files_modified: 2
  files_created: 1
  tests_added: 6
  lines_added: 241
  lines_removed: 4
  ---
  ```
- ✅ Include Metrics table after Executive Summary
- ✅ Reference MAP-PLAN for code details:
  ```markdown
  ### `accounts/schemas.py`
  - Added: AccountMemberRead schema (lines 18-39)
  - See map-plan-26-122225.md lines 102-127 for details
  ```
- ✅ Summarize test implementations:
  ```markdown
  Created 3 router tests: test_success, test_not_found, test_unauthorized (117 lines)
  ```
- ✅ **Document issues encountered in detail** (HIGH VALUE!)

**DON'T**:
- ❌ Re-quote code already in MAP-PLAN (biggest waste - 15% of content)
- ❌ Repeat acceptance criteria (reference MAP-PLAN instead)
- ❌ Include full test code listings (summarize)
- ❌ Enumerate every single change (git diff provides this)

**Acceptance Criteria Section**:
```markdown
## Acceptance Criteria Status
All criteria met. See MAP-PLAN for full list.

(If any failed, list specific failures here)
```

**Target Reduction**: 41% (475 → 280 lines)

---

### PROVE Agent Efficiency Rules

**Output Target**: 370-400 lines MAX (currently averaging 642)

**DO**:
- ✅ Use YAML frontmatter with test metrics:
  ```yaml
  ---
  issue: 26
  agent: PROVE
  date: 2025-12-23
  status: PASS
  tests_passed: 6
  tests_failed: 0
  regressions: 0
  ---
  ```
- ✅ Use exceptions-only reporting:
  ```markdown
  ## Comparison with PATCH
  All PATCH claims verified. No discrepancies found.
  ```
- ✅ Simple acceptance criteria table (10 lines max):
  ```markdown
  | Criterion | Status |
  |-----------|--------|
  | Schema created | ✅ |
  | Tests passing | ✅ |
  ```
- ✅ Focus on PASS/FAIL and exceptions

**DON'T**:
- ❌ Create detailed verification for each criterion (120 lines → 15 lines)
- ❌ Document PATCH comparison when all confirmed (use one-liner)
- ❌ Create appendices ("Appendix A: Test Output", etc.)
- ❌ Include redundant test output (reference PATCH if already documented)
- ❌ Exceed 400 lines total

**Comparison Section**:
```markdown
## Comparison with PATCH
All PATCH claims verified. No discrepancies found.

**OR if issues found:**
- Discrepancy 1: [detailed description]
- Discrepancy 2: [detailed description]
```

**Target Reduction**: 42% (642 → 370 lines)

---

## Output File Naming

**Pattern**: `{phase}-{issue_number}-{mmddyy}.md`

**Examples**:
- `map-plan-26-122225.md`
- `test-plan-26-122225.md` (if --with-tests)
- `patch-26-122225.md`
- `prove-26-122225.md`
- `contract-26-122225.md` (for fullstack coordination)

**Location**: `.agents/outputs/`

---

## Fullstack Coordination

If change touches both backend and frontend:
1. PLAN defines implementation approach
2. **CONTRACT agent (Phase 2)** defines API surface (MANDATORY)
3. PATCH implements both sides using CONTRACT as authoritative spec

**Workflow for fullstack**:
```
MAP-PLAN → CONTRACT → PATCH → PROVE
```

**Contract Agent**: `.claude/agents/contract.md`
**Output**: `.agents/outputs/contract-{issue}-{mmddyy}.md`
**Timing**: After PLAN, before PATCH

---

## Verification Gates

### Backend Verification (PROVE)
```bash
cd backend && ruff check .
cd backend && pytest -q
```

### Frontend Verification (PROVE)
```bash
cd frontend && npm run lint
cd frontend && npm run build
```

**Status**: If any command fails, set overall status to **BLOCKED**

---

## Complexity Classification

**TRIVIAL**: docs, config tweaks, small renames, deleting unused code
→ Use MAP-PLAN (single phase)

**SIMPLE**: 1-3 files, straightforward bug fix or UI tweak
→ Use MAP-PLAN (single phase)

**COMPLEX**: new endpoints, DB migrations, cross-module refactors, fullstack changes
→ Use MAP + PLAN (two phases)

---

## Success Metrics (Phase 1 Targets)

### Document Length Targets

| Agent | Current Avg | Phase 1 Target | Reduction |
|-------|-------------|----------------|-----------|
| MAP-PLAN | 698 lines | 580 lines | -17% |
| PATCH | 475 lines | 280 lines | -41% |
| PROVE | 642 lines | 370 lines | -42% |
| **Total** | **1,815 lines** | **1,230 lines** | **-32%** |

### Quality Metrics

- ✅ Zero code re-quotes in PATCH
- ✅ Acceptance criteria NOT repeated across phases
- ✅ PROVE reports under 400 lines
- ✅ No appendices in PROVE reports
- ✅ All high-value content preserved (Issues Encountered, Test Results, etc.)

### Token Savings

- Current: ~8,408 tokens per issue
- Target: ~5,525 tokens per issue
- **Savings**: 2,883 tokens/issue (34% reduction)

---

## Example Sessions

### Standard Workflow
```
User: /orchestrate 26
Skill: Verifying issue #26...
Skill: Issue verified: "Add account_members to ClientProfileRead schema"
Skill: Classifying as SIMPLE
Skill: Running MAP-PLAN agent...
Skill: MAP-PLAN artifact: .agents/outputs/map-plan-26-122225.md
Skill: Running PATCH agent...
Skill: PATCH artifact: .agents/outputs/patch-26-122225.md
Skill: Running PROVE agent...
Skill: PROVE artifact: .agents/outputs/prove-26-122225.md
Skill: ✓ Workflow complete - Ready for PR
```

### With Test Planning (Recommended for Calculations)
```
User: /orchestrate 239 --with-tests
Skill: Verifying issue #239...
Skill: Issue verified: "Income start-year proration + bug fix"
Skill: Classifying as SIMPLE (backend)
Skill: Running MAP-PLAN agent...
Skill: MAP-PLAN artifact: .agents/outputs/map-plan-239-011226.md
Skill: Running TEST-PLANNER agent...
Skill: TEST-PLAN artifact: .agents/outputs/test-plan-239-011226.md
Skill: Test cases identified: 12 (P0: 4, P1: 6, P2: 2)
Skill: Running PATCH agent...
Skill: PATCH artifact: .agents/outputs/patch-239-011226.md
Skill: Running PROVE agent...
Skill: PROVE artifact: .agents/outputs/prove-239-011226.md
Skill: ✓ Workflow complete - Ready for PR
```

---

## Post-Workflow Actions

Once PROVE passes:
1. Review artifacts in `.agents/outputs/`
2. Use `/pr` skill to create pull request
3. Keep `main` branch green
4. Merge after review

---

## Monitoring & Improvement

Track these metrics for each issue:

```yaml
metrics:
  issue: 29
  map_plan_lines: 580
  patch_lines: 280
  patch_code_requotes: 0  # Should be zero
  prove_lines: 370
  prove_appendices: 0     # Should be zero
  total_lines: 1230
  reduction_pct: 32
```

**Performance Reviews**:
- Quarterly review: `.agents/outputs/agent-performance-review-{date}.md`
- Implementation summaries: `.agents/outputs/phase{N}-implementation-summary-{date}.md`

---

## References

- **Detailed orchestrate guide**: `.claude/commands/orchestrate.md`
- **Agent definitions**: `.claude/agents/`
- **Performance review**: `.agents/outputs/agent-performance-review-122325.md`
- **Phase 1 implementation**: `.agents/outputs/phase1-implementation-summary-122325.md`
- **Project rules**: `.claude/rules.md`
- **Project stack**: `.claude/context/project_stack.md`

---

## Phase 1 Efficiency Improvements Summary

**Date**: December 2025
**Status**: Implemented

**Changes**:
1. ✅ YAML frontmatter for all agents
2. ✅ Eliminated code re-quoting in PATCH (41% reduction)
3. ✅ Consolidated acceptance criteria (10% reduction across all phases)
4. ✅ Removed PROVE appendices (5% reduction)
5. ✅ Implemented exceptions-only PROVE reporting (42% reduction in PROVE)

**Expected Results**:
- 32% reduction in total documentation (1,815 → 1,230 lines)
- 34% token savings (8,408 → 5,525 tokens/issue)
- Zero quality loss - all high-value content preserved

**Next Phases**:
- Phase 2: Cross-reference system, standardized status indicators
- Phase 3: Structured JSON output, smart content reuse library

---

## Critical Fix: Task Tool Invocation (December 31, 2025)

**Issue**: Orchestrate workflow failed to produce artifacts (issue #116, 40+ minutes, zero output)

**Root Cause**: Orchestrate SKILL.md described WHAT to do ("Run MAP-PLAN agent") but not HOW to do it (use Task tool)

**Impact**: Complete workflow failure - agents never spawned, no artifacts created

**Fix Applied**: Updated `.claude/skills/orchestrate/SKILL.md` with:
1. ✅ Explicit Task tool invocation instructions for each agent
2. ✅ Artifact validation after each phase
3. ✅ Error handling and failure reporting
4. ✅ Clear "CRITICAL" markers for required steps

**Key Changes**:
```markdown
### Step 2: Spawn Agents Using Task Tool

**CRITICAL**: You MUST use the Task tool to spawn each agent.

Use the Task tool to spawn the MAP-PLAN agent:
Task(
  subagent_type='general-purpose',
  description='MAP-PLAN analysis for issue <number>',
  prompt='''You are acting as the MAP-PLAN agent.
  Read your agent definition at: `.claude/agents/map-plan.md`
  ...'''
)

**Validate artifact**:
- Check file exists
- Check file size > 0 bytes
- Check file contains "AGENT_RETURN:" directive
```

**Documentation**: See `.agents/outputs/orchestrate-rca-123125.md` for full root cause analysis

**Status**: Fixed and ready for testing with issue #116
