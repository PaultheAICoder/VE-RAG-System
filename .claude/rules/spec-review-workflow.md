# Spec Review Workflow & Best Practices

This document defines the **specification review and issue creation workflow** based on lessons learned from the flow-of-funds specification process (December 2025).

## Overview

The spec review workflow ensures specifications are **finalized and validated** before creating GitHub issues, preventing wasted effort and maintaining consistency between specs and implementation work.

## Workflow Pattern

### ✅ CORRECT: Spec Finalization Gate

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Initial Spec Review                                      │
│    - Analyze spec against codebase                          │
│    - Identify gaps, inconsistencies, missing prerequisites  │
│    - Document findings in review artifact                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Senior Engineer / Stakeholder Feedback                   │
│    - Code review findings                                   │
│    - Identify architectural gaps                            │
│    - Clarify business requirements                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Make Decisions & Update Spec                             │
│    - Document all stakeholder decisions                     │
│    - Fix inconsistencies                                    │
│    - Add prerequisites and dependencies                     │
│    - Update effort estimates                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────────────┐
                    │ Spec Final?   │
                    └───────┬───────┘
                            │
                ┌───────────┴────────────┐
                │                        │
              NO                        YES
                │                        │
                └──► Repeat 1-3          │
                                         ↓
                        ┌────────────────────────────────┐
                        │ 4. Commit Final Spec to Git    │
                        │    - Single version (vX)       │
                        │    - Mark as FINAL             │
                        │    - Tag commit                │
                        └────────────────────────────────┘
                                         ↓
                        ┌────────────────────────────────┐
                        │ 5. Create GitHub Issues        │
                        │    - Reference spec + commit   │
                        │    - All issues consistent     │
                        └────────────────────────────────┘
```

### ❌ INCORRECT: Create Issues Before Finalizing Spec

```
Spec Review → Decisions → Create Issues → Fix Spec
                              ↓
                    ⚠️ Issues now inconsistent with spec
                    ⚠️ Must update all issues manually
                    ⚠️ Wasted effort
```

---

## Single Source of Truth Pattern

### Specification Versioning

**DO:**
- ✅ Use **one spec file** with clear version in filename (e.g., `feature-name-v3.md`)
- ✅ Use **git commits** to track version history
- ✅ Use **git tags** for major versions (e.g., `spec-flow-of-funds-v3.0`)
- ✅ Mark final version as **FINAL** in title and status

**DON'T:**
- ❌ Keep multiple version files (v1.md, v2.md, v3.md)
- ❌ Use timestamps in filenames (spec-123125.md)
- ❌ Maintain outdated versions in main branch

### Review Document Management

**DO:**
- ✅ Keep **only the final review document** in `.agents/outputs/`
- ✅ Name clearly: `spec-review-FINAL-{feature-name}-{mmddyy}.md`
- ✅ Archive intermediate reviews if needed

**DON'T:**
- ❌ Keep multiple review iterations (creates confusion)
- ❌ Reference outdated reviews in issues

---

## Spec Finalization Checklist

Before creating GitHub issues, ensure:

### Content Completeness
- [ ] All prerequisites documented with evidence
- [ ] Current behavior accurately reflects codebase state
- [ ] All stakeholder decisions documented
- [ ] Implementation phases defined with dependencies
- [ ] Effort estimates realistic and approved
- [ ] Risk assessment complete
- [ ] Open questions resolved

### Internal Consistency
- [ ] No contradictions between sections
- [ ] Terminology used consistently
- [ ] Dependencies correctly sequenced
- [ ] Acceptance criteria align with goals

### Stakeholder Approval
- [ ] Engineering lead approval
- [ ] Product team approval (timeline, scope)
- [ ] Architecture team approval (if applicable)
- [ ] Security team approval (if applicable)

### Documentation Quality
- [ ] Status marked as "FINAL - Ready for Implementation"
- [ ] Version number in frontmatter and title
- [ ] Document history table updated
- [ ] References to other docs accurate

---

## Issue Creation Best Practices

### Issue Template Requirements

Every issue created from a spec MUST include:

```markdown
## Reference
Spec: `specs/{spec-name}-v{X}.md` (lines Y-Z)
**Spec Version**: v{X}.0 (commit {hash})

## Overview
[Clear description of what this issue implements]

## Problem
**Current State**: [What exists now]
**Required**: [What spec requires]

## Implementation
[Detailed implementation steps from spec]

## Acceptance Criteria
[Checklist from spec - do not duplicate, reference spec lines]

## Dependencies
**Depends on**: Issue #X, Issue #Y
**Blocks**: Issue #Z

## Effort Estimate
**X-Y days** (COMPLEXITY_LEVEL)
```

### Spec Version Reference Format

**Required fields:**
1. **Spec file path** with version: `specs/flow-of-funds-v3.md`
2. **Line numbers** for relevant sections: `(lines 76-88, 243-258)`
3. **Spec version**: `v3.0`
4. **Git commit hash**: `(commit abc1234)`

**Example:**
```markdown
## Reference
Spec: `specs/flow-of-funds-v3.md` (lines 434-448)
**Spec Version**: v3.0 (commit 97bc25e)
```

**Why this matters:**
- Ensures implementers reference correct version
- Git commit provides immutable reference
- Line numbers help locate exact requirements
- Version number prevents confusion with outdated specs

---

## Git Workflow for Specs

### 1. Create Initial Spec

```bash
# Create spec in specs/ directory
# Naming: feature-name-v1.md (start with v1)

# Commit initial version
git add specs/feature-name-v1.md
git commit -m "feat: Add feature-name specification v1"
git push
```

### 2. Iterate Based on Feedback

```bash
# Update spec file IN PLACE
# DO NOT create v2, v3 files yet

# Commit each iteration
git commit -am "docs: Update feature-name spec with senior feedback"
git push
```

### 3. Finalize Spec

```bash
# Update status to FINAL
# Update version number in frontmatter
# Rename file if moving to final version number

git mv specs/feature-name-v1.md specs/feature-name-v3.md  # If jumped versions
# OR just update content if keeping v1

# Commit final version
git add specs/feature-name-v3.md
git commit -m "docs: Finalize feature-name specification v3 (FINAL)"

# Tag the commit
git tag -a spec-feature-name-v3.0 -m "Feature Name Specification v3.0 FINAL"
git push && git push --tags
```

### 4. Create Issues

```bash
# Now that spec is finalized, create issues
# Each issue references: specs/feature-name-v3.md (commit abc1234)

# If spec needs updates after issues created
# Update the SAME file (v3.md)
# Increment version in frontmatter (v3.1, v3.2)
# Update affected issues
```

---

## Version Control Strategy

### Semantic Versioning for Specs

- **v1.0**: Initial draft
- **v2.0**: Major revision (after senior review)
- **v3.0**: Final approved version (ready for implementation)
- **v3.1**: Minor updates after finalization (bug fixes, clarifications)
- **v3.2**: Additional minor updates

### Git Tags

**Format**: `spec-{feature-name}-v{X}.{Y}`

**Examples**:
- `spec-flow-of-funds-v3.0` - Final approved version
- `spec-rbac-v2.0` - Second major revision
- `spec-invitation-v1.0` - Initial version

**When to tag**:
- ✅ When spec status changes to FINAL
- ✅ Before creating GitHub issues
- ✅ Before starting implementation
- ❌ Not for every minor update

---

## Lessons Learned: Flow-of-Funds Case Study

### What Went Wrong

**Problem**: Created GitHub issues from v2 spec, then found Phase 3 inconsistencies, created v3 spec.

**Impact**:
- Had to manually update issues #125 and #126
- Had 3 spec files (v1, v2, v3) causing confusion
- Had 3 review documents with overlapping content
- Wasted time cleaning up afterward

### What Would Have Been Better

**Improved Flow**:
1. ✅ Spec review → senior feedback → UPDATE spec v1 → iterate
2. ✅ Make all decisions → UPDATE spec v1 → mark as FINAL → rename to v3
3. ✅ Commit final spec
4. ✅ Create all issues referencing v3 + commit hash
5. ✅ No cleanup needed

**Time Saved**: 2-3 hours of manual issue updates and cleanup

---

## Quick Reference

### Before Creating Issues

1. Is spec marked FINAL? → If NO, keep iterating
2. Are all decisions documented? → If NO, get decisions
3. Is spec committed to git? → If NO, commit it
4. Is commit tagged? → If NO, tag it
5. Ready to create issues? → YES

### Issue Creation Checklist

- [ ] Spec is FINAL and committed
- [ ] Have git commit hash
- [ ] Issue template includes spec reference with commit
- [ ] Issue references specific line numbers
- [ ] Issue dependencies align with spec phases
- [ ] Issue effort estimates match spec

### Cleanup After Finalization

1. Delete old spec versions (if any)
2. Delete intermediate review documents
3. Keep only: final spec + final review
4. Commit cleanup
5. Push to remote

---

## Related Documentation

- **Orchestrate Workflow**: `.claude/rules/orchestrate-workflow.md`
- **Spec-Reviewer Agent**: `.claude/agents/spec-reviewer.md`
- **Feature Command**: `.claude/commands/feature-from-spec.md`
- **Backend Patterns**: `.claude/rules/backend-patterns.md`

---

## Summary

**Golden Rules**:

1. **Finalize spec BEFORE creating issues** (not after)
2. **One spec file** (use git for history, not filenames)
3. **Tag the commit** when finalizing spec
4. **Reference commit hash** in every issue
5. **Delete old versions** after finalization

Following this workflow prevents:
- ❌ Inconsistent issues
- ❌ Manual issue updates
- ❌ Confusion about canonical version
- ❌ Wasted cleanup effort

And ensures:
- ✅ Single source of truth
- ✅ Immutable references (git commits)
- ✅ Clear version history
- ✅ Efficient workflow
