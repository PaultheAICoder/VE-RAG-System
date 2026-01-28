---
description: Extract learning data from postmortem files into memory system
argument-hint: [--new-only] [--dry-run]
---

# Postmortem Extract Command

Converts human-readable postmortem analyses into machine-readable failure records for the learning system.

## Purpose

Postmortems contain valuable failure analysis that should feed into:
1. `memory/failures.jsonl` — Structured failure records
2. `memory/patterns.md` — Aggregated prevention patterns

This command bridges the gap between human documentation and automated learning.

## Usage

```bash
/postmortem-extract              # Process all postmortems
/postmortem-extract --new-only   # Only unprocessed files
/postmortem-extract --dry-run    # Preview without writing
```

---

## Process

### Step 1: Find Postmortem Files

```bash
find ai_docs/postmortems -name "*.md" -type f | grep -v "TEMPLATE\|SUMMARY\|USER_TEMPLATES"
```

### Step 2: Parse Each Postmortem

For each file, extract:

| Field | Source | Example |
|-------|--------|---------|
| `date` | Frontmatter `date:` | "2025-12-24" |
| `issue` | Frontmatter `issue:` or filename | "Enum Mismatch" |
| `root_cause` | Analysis section | "ENUM_VALUE" |
| `severity` | Frontmatter `severity:` | "HIGH" |
| `details` | Executive Summary | "Frontend sent CO_OWNER..." |
| `fix` | What Should Have Happened | "Use enum VALUES not names" |
| `prevention` | Agent Improvement section | "Add enum check to PATCH" |
| `files` | Affected files list | ["frontend/src/..."] |
| `agents_involved` | Timeline section | ["CONTRACT", "PATCH", "PROVE"] |

### Step 3: Classify Root Cause

Map postmortem findings to standard root cause codes:

| Pattern in Postmortem | Root Cause Code |
|----------------------|-----------------|
| "enum", "VALUE", "NAME" | `ENUM_VALUE` |
| "component", "props", "PropTypes" | `COMPONENT_API` |
| "hook", "return", "destructur" | `COMPONENT_API` |
| "multi-model", "relationship", "update" | `MULTI_MODEL` |
| "permission", "access", "403" | `ACCESS_CONTROL` |
| "SQLite", "PostgreSQL" | `SQLITE_COMPAT` |
| "structure", "directory", "src/" | `STRUCTURE_VIOLATION` |
| "spec", "requirement", "missing" | `SPEC_DEVIATION` |

### Step 4: Generate Failure Record

```json
{
  "source": "postmortem",
  "source_file": "ai_docs/postmortems/2025-12-enum-mismatch-errors.md",
  "date": "2025-12-24",
  "issue": "Enum Mismatch Analysis",
  "root_cause": "ENUM_VALUE",
  "severity": "HIGH",
  "details": "Frontend sent CO_OWNER (underscore) but backend expected CO-OWNER (hyphen)",
  "fix": "Changed role string to match backend enum VALUE",
  "prevention": "PATCH agent must verify enum VALUES against CONTRACT",
  "files": ["frontend/src/components/owners/OwnersList2.jsx"],
  "agents_involved": ["CONTRACT", "PATCH", "PROVE"],
  "extracted_at": "2025-01-03T10:00:00Z"
}
```

### Step 5: Append to failures.jsonl

```bash
echo '$RECORD' >> .claude/memory/failures.jsonl
```

### Step 6: Mark as Processed

Track processed files to avoid duplicates:

```bash
echo "ai_docs/postmortems/2025-12-enum-mismatch-errors.md" >> .claude/memory/processed_postmortems.txt
```

---

## Output

```
╔═══════════════════════════════════════════════════════════════╗
║                POSTMORTEM EXTRACTION COMPLETE                  ║
╚═══════════════════════════════════════════════════════════════╝

Files processed: 12
Records extracted: 15

By root cause:
  ENUM_VALUE:     4 records
  COMPONENT_API:  3 records
  MULTI_MODEL:    2 records
  ACCESS_CONTROL: 2 records
  SPEC_DEVIATION: 2 records
  OTHER:          2 records

Output:
  ✓ Appended 15 records to .claude/memory/failures.jsonl

Next steps:
  • Run /learn to update patterns.md
  • Review patterns.md for new prevention rules
```

---

## Handling Edge Cases

### Multiple Issues per Postmortem

Some postmortems (like `2024-12-27-issue-103-SUMMARY.md`) document multiple failures.

**Approach**: Create one record per distinct root cause.

### Missing Fields

If postmortem lacks structured data:

```json
{
  "source": "postmortem",
  "source_file": "...",
  "root_cause": "OTHER",
  "details": "[First paragraph of file]",
  "extraction_quality": "partial"
}
```

### Already Processed

With `--new-only` flag:
```bash
# Skip files in processed list
grep -F "$FILE" .claude/memory/processed_postmortems.txt && continue
```

---

## Integration with /learn

After extraction:

```bash
/postmortem-extract
/learn
```

The `/learn` command will:
1. Read new records from failures.jsonl
2. Cluster by root cause
3. Update patterns.md with prevention rules

---

## Example Extraction

**Input**: `ai_docs/postmortems/2025-12-enum-mismatch-errors.md`

**Extracted**:
```json
{
  "source": "postmortem",
  "source_file": "ai_docs/postmortems/2025-12-enum-mismatch-errors.md",
  "date": "2025-12-24",
  "issue": "52",
  "root_cause": "ENUM_VALUE",
  "severity": "HIGH",
  "details": "Frontend sent role: CO_OWNER (underscore) but backend expected role: CO-OWNER (hyphen). Caused 422 validation errors.",
  "fix": "Changed frontend to use enum VALUE (CO-OWNER) not Python name (CO_OWNER)",
  "prevention": "1. PATCH must read CONTRACT for enum specs. 2. PATCH must verify against backend enum file. 3. PROVE must grep frontend for enum strings.",
  "files": [
    "frontend/src/components/owners/OwnersList2.jsx",
    "frontend/src/components/owners/AddEditMemberModal.jsx"
  ],
  "agents_involved": ["CONTRACT", "PATCH", "PROVE"],
  "blocking_agent": "PROVE",
  "extracted_at": "2025-01-03T10:00:00Z"
}
```

---

## Related Commands

- `/learn` — Process failures into patterns
- `/metrics` — View failure statistics
- `/agent-update` — Apply prevention rules to agents
