---
description: Review a specification against the codebase and optionally create GitHub issues
argument-hint: <spec-file> [--dry-run] [--create-issues]
---

# Spec Review Command

**Role**: Invoke SPEC-REVIEWER agent to analyze a specification

---

## Usage

```bash
/spec-review specs/FEATURE_NAME.md
/spec-review specs/FEATURE_NAME.md --dry-run       # Don't create issues
/spec-review specs/FEATURE_NAME.md --create-issues # Create GitHub issues
```

If no spec file provided, list available specs:
```bash
ls specs/*.md
```

---

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run` | ✓ | Analyze only, don't create GitHub issues |
| `--create-issues` | | Create GitHub issues for each gap found |

---

## Process

### Step 1: Validate Spec File

```bash
# Check spec exists
if [ ! -f "$SPEC_FILE" ]; then
  echo "Spec file not found: $SPEC_FILE"
  exit 1
fi
```

### Step 2: Spawn SPEC-REVIEWER Agent

**CRITICAL**: Use Task tool to spawn the agent.

```
Task(
  subagent_type='general-purpose',
  description='Spec review for <spec-name>',
  prompt='''You are acting as the SPEC-REVIEWER agent.

## Instructions
Read your agent definition at: `.claude/agents/spec-reviewer.md`

## Specification to Review
File: {spec_file_path}

## Options
- dry_run: {true|false}
- create_issues: {true|false}

## Process
1. Read the spec file completely
2. Extract all requirements (backend, frontend, API endpoints)
3. Search codebase for existing implementations
4. Classify gaps (✅ Implemented, 🟡 Partial, ❌ Missing, ⚠️ Differs)
5. Generate spec review artifact
6. If --create-issues: Create GitHub issues for gaps

## Output
Write artifact to: .agents/outputs/spec-review-{name}-{mmddyy}.md
End with: AGENT_RETURN: spec-review-{name}-{mmddyy}.md
'''
)
```

### Step 3: Validate Output

Check artifact exists and contains required sections:
- Requirements Extracted
- Codebase Analysis
- Gap Summary

### Step 4: Report Results

```
✓ Spec review complete for {spec-name}

Artifact: .agents/outputs/spec-review-{name}-{mmddyy}.md

Gap Summary:
- ✅ Implemented: N
- 🟡 Partial: N
- ❌ Missing: N
- ⚠️ Differs: N

{If --create-issues}
Issues Created:
- #123: [Spec] Implement X (Backend)
- #124: [Spec] Add Y component (Frontend)

Next Steps:
- Review artifact for accuracy
- Use /orchestrate <issue> to implement gaps
```

---

## Output

Artifact written to: `.agents/outputs/spec-review-{spec-name}-{mmddyy}.md`

Contains:
- Specification summary
- Requirements extracted from spec
- Codebase analysis with gap classification
- Implementation order recommendation
- Risk flags from patterns.md
- GitHub issues (if --create-issues)

---

## Examples

### Analyze Admin Dashboard Spec
```
/spec-review specs/ADMIN_DASHBOARD_SPEC.md
```

### Review and Create Issues
```
/spec-review specs/ADMIN_DASHBOARD_SPEC.md --create-issues
```

---

## Integration with Orchestrate

After spec review creates issues:
```
/orchestrate <issue-number>
```

This implements the MAP → PLAN → PATCH → PROVE workflow for each issue.

---

## Rules

**MUST**:
- Read spec file completely before analysis
- Search codebase for each requirement
- Classify all gaps accurately
- Generate actionable output

**MUST NOT**:
- Create issues without --create-issues flag
- Skip codebase analysis
- Make assumptions about implementation status
