# Learned Patterns

**Last updated**: 2026-01-30
**Total issues analyzed**: 9 (#2-7, #23-25)
**Success rate**: 100% (9/9 issues shipped)
**Workflow incidents**: 1 (branch collision, no feature impact)

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

**Frequency**: 1 occurrence (11% of orchestrated batches)
**Severity**: WORKFLOW_DISRUPTION (features still shipped)
**Phase**: PATCH

**Pattern**: When multiple PATCH agents run in parallel and modify the same file, Git's shared working directory causes all changes to accumulate. The first branch to commit captures ALL changes, leaving other branches empty.

**Trigger conditions**:
- Parallel PATCH execution (`--parallel` flag or multiple Task calls)
- Multiple issues modifying same file (e.g., `gradio_app.py`)
- Agents don't commit before orchestrator continues

**Common files affected**:
- `ai_ready_rag/ui/gradio_app.py` (UI changes often overlap)
- Any shared service file

**Prevention checklist**:
- [ ] [Orchestrator]: Before parallel PATCH, extract `files_identified` from each MAP-PLAN
- [ ] [Orchestrator]: If ANY file appears in multiple issues → run PATCH sequentially
- [ ] [PATCH Agent]: MUST `git add && git commit` before returning
- [ ] [Orchestrator]: After each PATCH, verify: `git log main..{branch} --oneline` shows commits
- [ ] [Alternative]: Use `git worktree` for true parallel isolation

**Responsible agents**: Orchestrator (file overlap check), PATCH (must commit)

**Postmortem**: `.claude/memory/postmortems/parallel-patch-branch-collision-013026.md`

---

## Success Patterns

### Backend-Only Issues
- **Success rate**: 100% (6/6)
- **Complexity**: SIMPLE to COMPLEX
- **Observation**: Single-stack issues have cleaner execution

### Fullstack Issues
- **Success rate**: 100% (3/3) - but with workflow incident
- **Complexity**: SIMPLE
- **Observation**: Same-file modifications need sequential execution

---

## Metrics Summary

| Dimension | Count | Pass | Rate |
|-----------|-------|------|------|
| **Total** | 9 | 9 | 100% |
| Backend | 6 | 6 | 100% |
| Frontend/Fullstack | 3 | 3 | 100% |
| SIMPLE | 8 | 8 | 100% |
| COMPLEX | 1 | 1 | 100% |

| Week | Issues | Pass | Rate | Trend |
|------|--------|------|------|-------|
| 2026-01-29 | 6 | 6 | 100% | — |
| 2026-01-30 | 3 | 3 | 100% | → Stable |

---

## Suggested Agent Updates

### 1. Orchestrate Skill — Add File Overlap Check

**Reason**: PARALLEL_BRANCH_COLLISION occurred when parallel PATCH modified same file

**Suggested addition** to `.claude/skills/orchestrate/SKILL.md`:

```markdown
## Parallel Safety Check (Before PATCH Phase)

If running multiple issues in parallel:

1. Extract files from each MAP-PLAN:
   ```bash
   grep -A20 "### Affected Files" .agents/outputs/map-plan-{issue}-*.md
   ```

2. Check for overlap:
   - If ANY file appears in multiple MAP-PLANs → Run PATCH sequentially
   - If no overlap → Can run PATCH in parallel

3. After each PATCH returns, verify commit exists:
   ```bash
   git log main..feat/issue-{N}-* --oneline | head -1
   ```
   If empty → PATCH failed to commit, investigate.
```

**Impact**: Would have prevented 1 workflow disruption

### 2. PATCH Agent — Mandatory Commit Before Return

**Reason**: Agents made changes but didn't commit, causing cross-contamination

**Suggested addition** to `.claude/agents/patch.md`:

```markdown
## Commit Requirement (MANDATORY)

Before returning, PATCH agent MUST:

1. Stage changes: `git add <specific files>`
2. Commit with message: `git commit -m "feat: <description> (#<issue>)"`
3. Verify commit: `git log -1 --oneline`

If commit fails (e.g., pre-commit hook), fix and retry.
NEVER return without a commit on the feature branch.
```

**Impact**: Would ensure branch isolation for parallel work

---

## Next Steps

1. Review suggested agent updates above
2. Run `/agent-update` to apply changes
3. Continue using `/orchestrate` to gather more data
4. Re-run `/learn` after 10 more issues

---

## Data Sources

- **Metrics**: `.claude/memory/metrics.jsonl` (6 records)
- **Failures**: `.claude/memory/failures.jsonl` (1 record)
- **Postmortems**: `.claude/memory/postmortems/` (1 file)
