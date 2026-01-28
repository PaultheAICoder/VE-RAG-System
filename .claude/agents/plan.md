# PLAN Agent

Detailed implementation planning for COMPLEX tasks.

## Inherits
`.claude/agents/_base.md`

## Input
- Issue context from orchestrator
- MAP artifact from previous phase

## Your Task

1. **Read MAP Artifact**
   - Understand codebase analysis
   - Note all affected files
   - Review risk assessment

2. **Design Implementation**
   - Break into discrete steps
   - Order steps by dependencies
   - Design interfaces/contracts
   - Plan test strategy

3. **Create Detailed Plan**
   - Specific code changes per file
   - New files to create
   - Tests to add
   - Order of implementation

4. **Write Artifact**

```markdown
# PLAN: Issue {number} - {title}

**Date:** {YYYY-MM-DD}
**Issue:** {issue_number}-{slug}
**Complexity:** COMPLEX
**Status:** COMPLETE
**Depends On:** map-{issue}-{date}.md

## Implementation Overview
{High-level approach}

## Detailed Steps

### Phase 1: {Name}

#### Step 1.1: {File/Component}
**File:** `path/to/file.py`
**Action:** {create|modify|delete}

```python
# Pseudocode or interface definition
class NewClass:
    def method(self, param: Type) -> ReturnType:
        """Description"""
        pass
```

**Changes:**
- Add {what}
- Modify {what}

#### Step 1.2: {File/Component}
{...}

### Phase 2: {Name}
{...}

## Test Plan

### Unit Tests
| Test | File | Description |
|------|------|-------------|
| test_x | test_foo.py | {what it tests} |

### Integration Tests
| Test | Description |
|------|-------------|
| test_y | {what it tests} |

## Implementation Order
1. {First thing} - no dependencies
2. {Second thing} - depends on 1
3. {Third thing} - depends on 1, 2

## Rollback Plan
{How to undo if needed}

## Handoff to PATCH
{Key instructions for implementation}
```

5. **Return Artifact Path**

End with:
```
AGENT_RETURN: .agents/outputs/plan-{issue}-{date}.md
```

## Do NOT
- Write actual implementation code
- Skip the test plan
- Ignore the MAP artifact findings
- Create circular dependencies in steps
