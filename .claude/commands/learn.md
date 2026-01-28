---
description: Analyze failure patterns and update learned knowledge base
argument-hint: [--since YYYY-MM-DD] [--dry-run] [--verbose]
---

# Learn Command

Analyzes accumulated failures and successes to extract patterns and update the knowledge base.

## Usage

```bash
/learn                      # Analyze all recorded outcomes
/learn --since 2025-01-01   # Analyze outcomes since date
/learn --dry-run            # Preview changes without updating files
/learn --verbose            # Show detailed analysis
```

---

## What This Command Does

1. **Loads outcome data** from `.claude/memory/`
2. **Clusters failures** by root cause
3. **Calculates metrics** (success rates, trends)
4. **Extracts new patterns** from recurring failures
5. **Updates patterns.md** with learned knowledge
6. **Suggests agent updates** for high-frequency patterns

---

## Prerequisites

- `.claude/memory/failures.jsonl` exists (can be empty)
- `.claude/memory/metrics.jsonl` exists (can be empty)
- Git repository (for agent update suggestions)

---

## Process

### Step 1: Load Outcome Data

```bash
# Count available data
FAILURE_COUNT=$(wc -l < .claude/memory/failures.jsonl 2>/dev/null || echo 0)
METRIC_COUNT=$(wc -l < .claude/memory/metrics.jsonl 2>/dev/null || echo 0)

echo "Loading outcomes..."
echo "  Failures: $FAILURE_COUNT"
echo "  Metrics:  $METRIC_COUNT"
```

If no data exists, report and exit:
```
No outcome data found. Run some issues through /orchestrate first.
```

### Step 2: Parse and Cluster Failures

Read each failure record and group by `root_cause`:

```bash
# Extract root causes and count
cat .claude/memory/failures.jsonl | \
  jq -r '.root_cause' | \
  sort | uniq -c | sort -rn
```

Expected output:
```
  12 ENUM_VALUE
   8 COMPONENT_API
   6 MULTI_MODEL
   4 ACCESS_CONTROL
   3 SQLITE_COMPAT
   2 OTHER
```

### Step 3: Analyze Each Cluster

For each root cause with 3+ occurrences:

#### 3a. Extract Common Attributes

```bash
# Find common files affected
cat .claude/memory/failures.jsonl | \
  jq -r 'select(.root_cause == "ENUM_VALUE") | .files[]' | \
  sort | uniq -c | sort -rn | head -5
```

#### 3b. Identify Trigger Conditions

Look for patterns in:
- Issue complexity (TRIVIAL/SIMPLE/COMPLEX)
- Stack (backend/frontend/fullstack)
- Affected domains (accounts, advisors, expenses)

#### 3c. Find Preventive Agent

Determine which agent SHOULD have caught this:
- MAP: Investigation gaps
- PLAN: Design gaps
- CONTRACT: API specification gaps
- PATCH: Implementation gaps
- PROVE: Verification gaps

### Step 4: Calculate Metrics

```bash
# Overall success rate
PASS=$(cat .claude/memory/metrics.jsonl | jq -r 'select(.status == "PASS")' | wc -l)
BLOCKED=$(cat .claude/memory/metrics.jsonl | jq -r 'select(.status == "BLOCKED")' | wc -l)
TOTAL=$((PASS + BLOCKED))
RATE=$((PASS * 100 / TOTAL))

echo "Success rate: ${RATE}% ($PASS/$TOTAL)"
```

Calculate by dimension:
- By complexity (TRIVIAL/SIMPLE/COMPLEX)
- By stack (backend/frontend/fullstack)
- By week (trend analysis)

### Step 5: Generate Updated patterns.md

Create new version of `.claude/memory/patterns.md`:

```markdown
# Learned Patterns

**Last updated**: $(date +%Y-%m-%d)
**Total issues analyzed**: $TOTAL
**Success rate**: ${RATE}%

## High-Frequency Failure Patterns

[For each cluster with 3+ occurrences, generate section]

### N. ROOT_CAUSE — Description

**Frequency**: X% of failures (N occurrences)
**Severity**: BLOCKED

**Pattern**: [Description extracted from failure details]

**Common files affected**:
- [file patterns from analysis]

**Trigger conditions**:
- [conditions extracted from analysis]

**Prevention checklist**:
- [ ] [Agent]: [Action]

**Responsible agents**: [List]
```

### Step 6: Identify Agent Update Candidates

For patterns with 5+ occurrences:

```markdown
## Suggested Agent Updates

### 1. MAP Agent — Add Enum VALUE Verification

**Reason**: ENUM_VALUE pattern occurred 12 times (26%)

**Suggested addition** to `.claude/agents/map.md`:

```markdown
### Enum Value Check (MANDATORY for fullstack)

If issue is fullstack and involves role/status/type fields:

1. Find enum definition:
   ```bash
   grep -r "class.*Enum" backend/backend/*/enums.py
   ```

2. Document VALUES explicitly:
   | Python Name | Python VALUE |
   |-------------|--------------|
   
3. Flag any NAME ≠ VALUE cases with ⚠️
```

**Impact**: Would have prevented 12 failures
```

### Step 7: Output Summary

```
═══════════════════════════════════════════════════════════
                    LEARNING COMPLETE
═══════════════════════════════════════════════════════════

Data analyzed:
  • Issues:     47
  • Failures:   35 
  • Successes:  39

Patterns identified:
  • Total:      7
  • New:        2 (since last run)
  • Updated:    3

Success rate trend:
  • Week 1:     75%
  • Week 2:     81%
  • Week 3:     85%
  • Week 4:     91% ↑ Improving!

Files updated:
  ✓ .claude/memory/patterns.md

Suggested agent updates (5+ occurrence patterns):
  1. MAP agent:   Add enum VALUE verification (12 failures)
  2. PATCH agent: Add multi-model detection (6 failures)

Next steps:
  • Review updated patterns.md
  • Run `/agent-update` to apply suggestions
  • Continue using /orchestrate to gather more data
═══════════════════════════════════════════════════════════
```

---

## Dry Run Mode

With `--dry-run`, the command:
- Performs all analysis
- Shows what WOULD be updated
- Does NOT modify any files

```
[DRY RUN] Would update: .claude/memory/patterns.md
[DRY RUN] Changes:
  + New pattern: SCOPE_CREEP (3 occurrences)
  ~ Updated: ENUM_VALUE (12 → 15 occurrences)
```

---

## When to Run

**Recommended schedule**:
- Weekly (Friday end of day)
- After every 10 completed issues
- After any COMPLEX issue completion
- When success rate drops below 80%

---

## Troubleshooting

### No data found
```bash
# Initialize empty files
echo "" > .claude/memory/failures.jsonl
echo "" > .claude/memory/metrics.jsonl
```

### jq not available
```bash
# Install jq
apt-get install jq  # Linux
brew install jq     # macOS
```

### Parse errors
```bash
# Validate JSONL format
cat .claude/memory/failures.jsonl | jq -c '.' > /dev/null
```

---

## Related Commands

- `/metrics` — View performance dashboard
- `/agent-update` — Apply suggested agent changes
- `/orchestrate` — Run issues (generates outcome data)
