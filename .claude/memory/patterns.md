# Learned Patterns

**Last updated**: 2026-01-29
**Total issues analyzed**: 0
**Success rate**: N/A (no data yet)

---

## Critical Patterns (Always Apply)

See `.claude/memory/patterns-critical.md` for the essential patterns:

1. **Commit to main** → Use feature branches
2. **Access control bypass** → Pre-retrieval filtering
3. **Wrong requirements** → Use `requirements-wsl.txt`

---

## High-Frequency Failure Patterns

*Patterns will be populated as issues are completed via `/orchestrate`.*

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
