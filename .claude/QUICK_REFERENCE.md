# Claude Code Quick Reference

## Essential Commands

| Command | Purpose |
|---------|---------|
| `/orchestrate <issue>` | Full workflow: MAP → PLAN → PATCH → PROVE |
| `/orchestrate <issue> --with-tests` | Include TEST-PLANNER phase |
| `/spec-review <spec-file>` | Analyze spec, identify gaps |
| `/spec-review <spec-file> --create-issues` | Create GitHub issues from gaps |
| `/feature` | Create feature issue |
| `/bug` | Create bug issue |
| `/pr` | Create pull request |
| `/learn` | Extract patterns from outcomes |
| `/metrics` | View success rates |

## Verification Commands

```bash
# Quick check
ruff check ai_ready_rag tests && pytest tests/ -q

# Full verification
ruff check ai_ready_rag tests && ruff format --check ai_ready_rag tests && pytest tests/ -v
```

## Git Workflow

```bash
# Feature branch
git checkout -b feat/issue-N-description

# After implementation
git add <files>
git commit -m "feat(#N): Description"
git push origin feat/issue-N-description

# Create PR
gh pr create --title "..." --body "..."
```

## Critical Patterns (Never Forget)

| Pattern | Rule |
|---------|------|
| **Branching** | NEVER commit to main directly |
| **Requirements** | Use `requirements-wsl.txt`, NOT `requirements.txt` |
| **Access Control** | Pre-retrieval filtering (user tags → vector search) |
| **Specs** | Don't commit drafts, finalize first |

## File Locations

| What | Where |
|------|-------|
| Agents | `.claude/agents/` |
| Commands | `.claude/commands/` |
| Patterns | `.claude/memory/patterns-critical.md` |
| Saved State | `.agents/outputs/claude_checkpoints/PERSISTENT_STATE.yaml` |
| Artifacts | `.agents/outputs/` |
| Metrics | `.claude/memory/metrics.jsonl` |
| Failures | `.claude/memory/failures.jsonl` |

## Complexity Classification

| Level | Criteria | Workflow |
|-------|----------|----------|
| TRIVIAL | Docs, config, renames | MAP-PLAN only |
| SIMPLE | 1-3 files, follows pattern | MAP-PLAN → PATCH → PROVE |
| COMPLEX | 4+ files, migrations, fullstack | MAP → PLAN → PATCH → PROVE |

## Artifact Naming

```
{agent}-{issue}-{mmddyy}.md

Examples:
- map-plan-42-012926.md
- patch-42-012926.md
- prove-42-012926.md
```

## Self-Learning Loop

```
Issue → /orchestrate → PROVE records outcome
                              ↓
                        metrics.jsonl (PASS)
                        failures.jsonl (BLOCKED)
                              ↓
                        /learn (weekly)
                              ↓
                        patterns.md updated
                              ↓
                        Agents read patterns
```

## Emergency Recovery

If context is lost:
1. Check `.agents/outputs/claude_checkpoints/PERSISTENT_STATE.yaml`
2. Read recent artifacts in `.agents/outputs/`
3. Resume with `/orchestrate <issue>` or continue manually
