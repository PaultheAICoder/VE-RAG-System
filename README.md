# Claude Code Configuration Template

A reusable agent system for issue-driven development workflows with pattern learning.

## Quick Start

1. **Copy to your project root:**
   ```bash
   tar -xzvf claude-config-template.tar.gz -C /path/to/your/project/
   ```

2. **Make hooks executable:**
   ```bash
   chmod +x .claude/hooks/*.py
   ```

3. **Install PyYAML (recommended):**
   ```bash
   pip install pyyaml
   ```

4. **Customize for your project:**
   - Edit `.claude/context/project_stack.md` - Your tech stack
   - Edit `.claude/rules/backend-patterns.md` - Your backend patterns
   - Edit `.claude/rules/frontend-patterns.md` - Your frontend patterns
   - Edit `.claude/rules/testing.md` - Your test setup
   - Create `CLAUDE.md` in your project root with project-specific instructions

5. **Start using:**
   ```bash
   claude
   /orchestrate <issue_number>
   ```

## What's Included

### Agents (`.claude/agents/`)
| Agent | Purpose |
|-------|---------|
| `_base.md` | Shared behaviors inherited by all agents |
| `map.md` | Codebase exploration |
| `map-plan.md` | Combined mapping + planning for simple issues |
| `plan.md` | Implementation planning |
| `patch.md` | Code implementation |
| `prove.md` | Verification and testing |
| `test-planner.md` | Test generation |
| `spec-reviewer.md` | Specification analysis |
| `contract.md` | API contract definition |

### Commands (`.claude/commands/`)
| Command | Usage |
|---------|-------|
| `/orchestrate <issue>` | Main workflow - issue to PR |
| `/learn` | Extract patterns from postmortems |
| `/metrics` | View agent performance |
| `/pr` | Create pull request |
| `/bug` | Create bug issue |
| `/feature` | Create feature issue |
| `/spec-draft` | Draft a specification |
| `/test-plan` | Generate test plan |

### Hooks (`.claude/hooks/`)
- `precompact_checkpoint.py` - Saves state before context compaction
- `sessionstart_restore_state.py` - Restores state on session start

### Workflow

```
GitHub Issue → /orchestrate → MAP-PLAN → PATCH → PROVE → PR
```

## Configuration Files to Customize

### Required
- `CLAUDE.md` (project root) - Main project instructions
- `.claude/context/project_stack.md` - Your technology stack
- `.claude/rules/backend-patterns.md` - Backend coding patterns
- `.claude/rules/frontend-patterns.md` - Frontend coding patterns

### Optional
- `.claude/rules/testing.md` - Testing conventions
- `.claude/memory/patterns-critical.md` - Critical patterns (learned over time)

## Building Your Pattern Library

1. Use `/orchestrate` to complete issues
2. When issues fail, create postmortems:
   ```
   .claude/memory/postmortems/YYYY-MM-DD-description.md
   ```
3. Run `/learn` to extract patterns
4. Patterns auto-load on session start

## Directory Structure

```
your-project/
├── .claude/
│   ├── agents/           # Agent definitions
│   ├── commands/         # Slash commands
│   ├── context/          # Project context
│   ├── hooks/            # Context preservation
│   ├── memory/           # Learned patterns
│   │   └── postmortems/  # Failure analysis
│   ├── rules/            # Coding patterns
│   ├── skills/           # Skill definitions
│   ├── templates/        # Spec/artifact templates
│   └── settings.json     # Hook configuration
├── .agents/
│   └── outputs/          # Agent artifacts (not in git)
│       └── claude_checkpoints/
└── CLAUDE.md             # Project instructions
```

## Artifacts

Agent outputs go to `.agents/outputs/`:
- `map-{issue}-{date}.md`
- `plan-{issue}-{date}.md`
- `patch-{issue}-{date}.md`
- `prove-{issue}-{date}.md`

Add `.agents/` to `.gitignore`.

## Requirements

- Claude Code CLI (`npm install -g @anthropic-ai/claude-code`)
- Python 3.8+ (for hooks)
- PyYAML (optional but recommended): `pip install pyyaml`
