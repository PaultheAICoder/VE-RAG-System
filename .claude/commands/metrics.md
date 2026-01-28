---
description: Display agent performance metrics and trends
argument-hint: [--week] [--month] [--agent NAME] [--json]
---

# Metrics Command

Displays performance metrics from accumulated outcome data.

## Usage

```bash
/metrics              # Show all metrics (last 30 days)
/metrics --week       # Last 7 days only
/metrics --month      # Last 30 days (default)
/metrics --all        # All time
/metrics --agent MAP  # Filter by specific agent
/metrics --json       # Output as JSON (for automation)
```

---

## Dashboard Output

```
╔═══════════════════════════════════════════════════════════════╗
║                 AGENT PERFORMANCE DASHBOARD                    ║
║                   Period: Last 30 Days                         ║
╚═══════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────┐
│ OVERALL METRICS                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Issues completed:        47                                     │
│ First-attempt success:   83% (39/47)                           │
│ Avg recovery attempts:   1.2                                    │
│ Avg time to completion:  24 min                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ BY COMPLEXITY                                                   │
├─────────────────────────────────────────────────────────────────┤
│ TRIVIAL:   95% success   ████████████████████░   (19/20)       │
│ SIMPLE:    85% success   █████████████████░░░░   (17/20)       │
│ COMPLEX:   43% success   ████████░░░░░░░░░░░░░   (3/7)         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ BY STACK                                                        │
├─────────────────────────────────────────────────────────────────┤
│ Backend:    91% success  ██████████████████░░░   (21/23)       │
│ Frontend:   78% success  ███████████████░░░░░░   (14/18)       │
│ Fullstack:  67% success  █████████████░░░░░░░░   (4/6)         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ TOP FAILURE CAUSES                                              │
├─────────────────────────────────────────────────────────────────┤
│ 1. ENUM_VALUE      26%  ██████████░░░░░░░░░░░   (12)           │
│ 2. COMPONENT_API   17%  ██████░░░░░░░░░░░░░░░   (8)            │
│ 3. MULTI_MODEL     13%  █████░░░░░░░░░░░░░░░░   (6)            │
│ 4. ACCESS_CONTROL   9%  ███░░░░░░░░░░░░░░░░░░   (4)            │
│ 5. SQLITE_COMPAT    7%  ██░░░░░░░░░░░░░░░░░░░   (3)            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ AGENT BLOCKING RATE                                             │
├─────────────────────────────────────────────────────────────────┤
│ MAP:        2%  █░░░░░░░░░░░░░░░░░░░░   (1 block)              │
│ PLAN:       4%  █░░░░░░░░░░░░░░░░░░░░   (2 blocks)             │
│ CONTRACT:   0%  ░░░░░░░░░░░░░░░░░░░░░   (0 blocks)             │
│ PATCH:     11%  ██░░░░░░░░░░░░░░░░░░░   (5 blocks)             │
│ PROVE:     83%  ████████████████░░░░░   (39 blocks) [expected] │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ WEEKLY TREND                                                    │
├─────────────────────────────────────────────────────────────────┤
│ Week 1:  75%  ███████████████░░░░░░░░░░                        │
│ Week 2:  81%  ████████████████░░░░░░░░░                        │
│ Week 3:  85%  █████████████████░░░░░░░░                        │
│ Week 4:  91%  ██████████████████░░░░░░░  ↑ Improving!          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ RECENT FAILURES                                                 │
├─────────────────────────────────────────────────────────────────┤
│ #201  ENUM_VALUE     fullstack  2025-01-02  (recovered in 1)   │
│ #198  COMPONENT_API  frontend   2025-01-01  (recovered in 2)   │
│ #195  MULTI_MODEL    backend    2024-12-30  (recovered in 1)   │
└─────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════╗
║ RECOMMENDATIONS                                                ║
╠═══════════════════════════════════════════════════════════════╣
║ • ENUM_VALUE at 26%: Consider adding enum check to MAP agent  ║
║ • Fullstack at 67%: Always use CONTRACT agent for fullstack   ║
║ • COMPLEX at 43%: Break into smaller issues when possible     ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## Implementation

### Step 1: Load Data

```bash
# Check for data files
if [ ! -f ".claude/memory/metrics.jsonl" ]; then
  echo "No metrics data found. Run /orchestrate to generate data."
  exit 1
fi

# Count records
TOTAL=$(wc -l < .claude/memory/metrics.jsonl)
echo "Analyzing $TOTAL outcome records..."
```

### Step 2: Calculate Overall Metrics

```bash
# Success rate
PASS=$(cat .claude/memory/metrics.jsonl | jq -r 'select(.status == "PASS")' | wc -l)
BLOCKED=$(cat .claude/memory/metrics.jsonl | jq -r 'select(.status == "BLOCKED")' | wc -l)

# Recovery attempts (for blocked issues that eventually passed)
AVG_RECOVERY=$(cat .claude/memory/metrics.jsonl | \
  jq -r 'select(.recovery_attempts != null) | .recovery_attempts' | \
  awk '{sum+=$1; count++} END {print sum/count}')
```

### Step 3: Calculate by Dimension

```bash
# By complexity
for COMPLEXITY in TRIVIAL SIMPLE COMPLEX; do
  PASS=$(cat .claude/memory/metrics.jsonl | \
    jq -r "select(.complexity == \"$COMPLEXITY\" and .status == \"PASS\")" | wc -l)
  TOTAL=$(cat .claude/memory/metrics.jsonl | \
    jq -r "select(.complexity == \"$COMPLEXITY\")" | wc -l)
  echo "$COMPLEXITY: $PASS / $TOTAL"
done

# By stack
for STACK in backend frontend fullstack; do
  PASS=$(cat .claude/memory/metrics.jsonl | \
    jq -r "select(.stack == \"$STACK\" and .status == \"PASS\")" | wc -l)
  TOTAL=$(cat .claude/memory/metrics.jsonl | \
    jq -r "select(.stack == \"$STACK\")" | wc -l)
  echo "$STACK: $PASS / $TOTAL"
done
```

### Step 4: Calculate Failure Causes

```bash
# Top failure causes
cat .claude/memory/failures.jsonl | \
  jq -r '.root_cause' | \
  sort | uniq -c | sort -rn | head -5
```

### Step 5: Calculate Trends

```bash
# Weekly trend (last 4 weeks)
for WEEK in 1 2 3 4; do
  START_DATE=$(date -d "$WEEK weeks ago" +%Y-%m-%d)
  END_DATE=$(date -d "$((WEEK-1)) weeks ago" +%Y-%m-%d)
  
  PASS=$(cat .claude/memory/metrics.jsonl | \
    jq -r "select(.date >= \"$START_DATE\" and .date < \"$END_DATE\" and .status == \"PASS\")" | wc -l)
  TOTAL=$(cat .claude/memory/metrics.jsonl | \
    jq -r "select(.date >= \"$START_DATE\" and .date < \"$END_DATE\")" | wc -l)
  
  echo "Week $WEEK: $PASS / $TOTAL"
done
```

---

## JSON Output Mode

With `--json` flag, output structured data for automation:

```json
{
  "period": "30d",
  "generated": "2025-01-03T10:00:00Z",
  "overall": {
    "total": 47,
    "passed": 39,
    "blocked": 8,
    "success_rate": 0.83,
    "avg_recovery": 1.2
  },
  "by_complexity": {
    "TRIVIAL": {"passed": 19, "total": 20, "rate": 0.95},
    "SIMPLE": {"passed": 17, "total": 20, "rate": 0.85},
    "COMPLEX": {"passed": 3, "total": 7, "rate": 0.43}
  },
  "by_stack": {
    "backend": {"passed": 21, "total": 23, "rate": 0.91},
    "frontend": {"passed": 14, "total": 18, "rate": 0.78},
    "fullstack": {"passed": 4, "total": 6, "rate": 0.67}
  },
  "top_failures": [
    {"cause": "ENUM_VALUE", "count": 12, "percentage": 0.26},
    {"cause": "COMPONENT_API", "count": 8, "percentage": 0.17}
  ],
  "trend": {
    "week_1": 0.75,
    "week_2": 0.81,
    "week_3": 0.85,
    "week_4": 0.91,
    "direction": "improving"
  }
}
```

---

## Agent Filter Mode

With `--agent MAP`, show metrics for specific agent:

```
╔═══════════════════════════════════════════════════════════════╗
║                    MAP AGENT METRICS                           ║
╚═══════════════════════════════════════════════════════════════╝

Invocations:     47
Blocks caused:   1 (2%)
Avg duration:    3 min

Issues identified by MAP:
• COMPONENT_API gaps:   8 (caught 6, missed 2)
• ENUM_VALUE gaps:     12 (caught 9, missed 3)

Common investigation patterns:
• grep for models:      47 uses
• grep for enums:       12 uses
• PropTypes extraction: 18 uses

Recommendations:
• Add automatic enum VALUE check for fullstack issues
• Increase PropTypes extraction for component reuse
```

---

## Recommendations Engine

Based on metrics, generate actionable recommendations:

| Condition | Recommendation |
|-----------|----------------|
| ENUM_VALUE > 20% | "Add enum VALUE verification to MAP agent" |
| Fullstack < 70% | "Always use CONTRACT agent for fullstack issues" |
| COMPLEX < 50% | "Break COMPLEX issues into SIMPLE sub-issues" |
| Trend declining | "Run /learn to update patterns" |
| Agent blocking > 10% | "Review agent definition for gaps" |

---

## Related Commands

- `/learn` — Update patterns from failures
- `/orchestrate` — Run issues (generates metrics data)
- `/agent-update` — Apply suggested improvements
