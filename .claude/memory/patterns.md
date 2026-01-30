# Learned Patterns

**Last updated**: 2026-01-30
**Total issues analyzed**: 3 (#23, #24, #25)
**Success rate**: 100% (features shipped, workflow disruption)

---

## Critical Patterns (Always Apply)

See `.claude/memory/patterns-critical.md` for the essential patterns:

1. **Commit to main** → Use feature branches
2. **Access control bypass** → Pre-retrieval filtering
3. **Wrong requirements** → Use `requirements-wsl.txt`
4. **Parallel branch collision** → Check file overlap before parallel PATCH

---

## High-Frequency Failure Patterns

### 1. PARALLEL_BRANCH_COLLISION — Git branches share working directory

**Frequency**: 1 occurrence (2026-01-30)
**Trigger**: Running parallel PATCH agents that modify the same file
**Prevention**: Check file overlap in MAP-PLAN; run sequentially if overlap; use git worktrees

**Root Cause**: Git branches share the same working directory. When multiple agents checkout different branches and modify the same file, all changes exist in the working directory simultaneously. The first commit captures ALL changes.

**Checklist**:
- [ ] [Orchestrator]: Before parallel PATCH, compare `files_identified` across MAP-PLANs
- [ ] [Orchestrator]: If ANY file appears in multiple issues → run PATCH sequentially
- [ ] [PATCH Agent]: MUST commit changes before returning
- [ ] [Orchestrator]: Verify `git log main..{branch}` shows commits after each PATCH

**Postmortem**: `.claude/memory/postmortems/parallel-patch-branch-collision-013026.md`

### Recording Outcomes

After each issue, PROVE agent records:
- **Success**: `metrics.jsonl` with PASS status
- **Failure**: `failures.jsonl` with root cause + details

### Extracting Patterns

Run `/learn` weekly to:
1. Cluster failures by root cause
2. Calculate success rate trends
3. Extract new patterns
4. Update this file

---

## Pattern Template

When patterns are extracted, they follow this format:

```markdown
### N. PATTERN_NAME — Description

**Frequency**: X% of failures (N occurrences)
**Trigger**: [What triggers this failure]
**Prevention**: [How to prevent it]

**Checklist**:
- [ ] [Agent]: [Verification step]
```

---

## Next Steps

1. Complete issues using `/orchestrate <issue>`
2. PROVE agent records outcomes automatically
3. Run `/learn` after 5-10 issues
4. Patterns appear here automatically
