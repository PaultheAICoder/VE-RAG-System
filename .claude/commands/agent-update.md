---
description: Apply suggested agent improvements from learning analysis
argument-hint: [--dry-run] [--agent NAME] [--pattern PATTERN_NAME]
---

# Agent Update Command

Applies suggested improvements to agent definitions based on learned patterns.

## Usage

```bash
/agent-update              # Apply all suggestions interactively
/agent-update --dry-run    # Preview changes without applying
/agent-update --agent MAP  # Update specific agent only
/agent-update --pattern ENUM_VALUE  # Apply fix for specific pattern
```

---

## How It Works

1. Reads `.claude/memory/patterns.md` for suggested updates
2. Identifies patterns with 5+ occurrences (high impact)
3. Generates specific text additions for agent definitions
4. Applies changes with user confirmation

---

## Suggested Update Format

From `/learn` output, updates are structured as:

```markdown
### Suggested Update: MAP Agent — Enum VALUE Check

**Pattern**: ENUM_VALUE (12 occurrences, 26%)
**Target file**: `.claude/agents/map.md`
**Location**: After "Document Enums" section

**Addition**:
```markdown
### Enum VALUE Verification (MANDATORY for fullstack)

**⚠️ This check prevents 26% of failures**

1. Find enum definition:
   ```bash
   grep -A 10 "class.*Enum" backend/backend/*/enums.py
   ```

2. For each enum, document:
   | Python Name | Python VALUE |
   |-------------|--------------|
   
3. Flag mismatches: NAME ≠ VALUE (e.g., `CO_OWNER = "CO-OWNER"`)

4. Add to risks if any mismatch found
```

**Impact**: Would have prevented 12 failures
```

---

## Interactive Mode

```
╔═══════════════════════════════════════════════════════════════╗
║                    AGENT UPDATE                                ║
╚═══════════════════════════════════════════════════════════════╝

Found 3 suggested updates from patterns.md:

1. MAP Agent — Enum VALUE Check
   Pattern: ENUM_VALUE (12 occurrences)
   Impact: High (26% of failures)

2. PATCH Agent — Multi-Model Detection
   Pattern: MULTI_MODEL (6 occurrences)
   Impact: Medium (13% of failures)

3. PROVE Agent — Component API Verification
   Pattern: COMPONENT_API (8 occurrences)
   Impact: High (17% of failures)

Apply which updates? [all/1,2,3/none]: 
```

---

## Process

### Step 1: Load Suggestions

```bash
# Extract suggestions from patterns.md
grep -A 20 "Suggested Update" .claude/memory/patterns.md
```

### Step 2: Validate Target Files

```bash
# Check agent files exist
for AGENT in map plan patch prove; do
  if [ ! -f ".claude/agents/${AGENT}.md" ]; then
    echo "WARNING: Agent file not found: ${AGENT}.md"
  fi
done
```

### Step 3: Preview Changes

For each suggestion:
1. Show current section (if exists)
2. Show proposed addition
3. Show expected location

```
Preview: MAP Agent — Enum VALUE Check
─────────────────────────────────────

File: .claude/agents/map.md
Location: Line 85 (after "Document Enums" section)

Current content at location:
│ ### 5. Document Enums (MANDATORY for fullstack)
│ ...
│ ### 6. Identify Pattern to Mirror

Proposed addition:
│ ### 5.5 Enum VALUE Verification (AUTO-ADDED)
│ 
│ **⚠️ This check prevents 26% of failures**
│ ...

Apply this change? [y/n]: 
```

### Step 4: Apply Changes

Using `sed` or similar to insert at correct location:

```bash
# Insert after line N
sed -i "${LINE}r addition.md" .claude/agents/map.md
```

### Step 5: Verify Changes

```bash
# Check file still valid markdown
head -20 .claude/agents/map.md
tail -20 .claude/agents/map.md
```

### Step 6: Commit Changes

```bash
# Create commit with clear message
git add .claude/agents/
git commit -m "feat(agents): apply learned patterns

Applied automatic improvements from /learn analysis:
- MAP: Added enum VALUE verification (ENUM_VALUE pattern)
- PATCH: Added multi-model detection (MULTI_MODEL pattern)
- PROVE: Added component API check (COMPONENT_API pattern)

Based on analysis of 47 issues with 83% success rate.
Patterns addressed account for 56% of failures."
```

---

## Dry Run Mode

With `--dry-run`:

```
╔═══════════════════════════════════════════════════════════════╗
║                    AGENT UPDATE (DRY RUN)                      ║
╚═══════════════════════════════════════════════════════════════╝

Would apply 3 updates:

1. MAP Agent — Enum VALUE Check
   File: .claude/agents/map.md
   Lines added: 15
   Location: After line 85

2. PATCH Agent — Multi-Model Detection
   File: .claude/agents/patch.md
   Lines added: 22
   Location: After line 120

3. PROVE Agent — Component API Verification
   File: .claude/agents/prove.md
   Lines added: 18
   Location: After line 95

Total lines added: 55
Files modified: 3

Run without --dry-run to apply changes.
```

---

## Rollback

If updates cause issues:

```bash
# Revert last agent update commit
git revert HEAD

# Or restore specific agent
git checkout HEAD~1 .claude/agents/map.md
```

---

## Update History

Track applied updates in `.claude/memory/updates.jsonl`:

```json
{
  "date": "2025-01-03",
  "agent": "map",
  "pattern": "ENUM_VALUE",
  "lines_added": 15,
  "commit": "abc123"
}
```

---

## Related Commands

- `/learn` — Generate update suggestions
- `/metrics` — View pattern impact
- `/orchestrate` — Test updated agents
