#!/usr/bin/env python3
"""
Claude Code SessionStart hook (Optimized v2):
- Loads compact YAML state (~300 tokens vs ~650)
- Loads critical patterns only (~200 tokens vs ~2600)
- Detects active orchestrate workflow and provides continue instructions
- Total: ~500 tokens vs ~3250 (85% reduction)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def get_active_work(yaml_content: str) -> dict:
    """Extract active_work from YAML content."""
    if not HAS_YAML:
        return {}
    try:
        data = yaml.safe_load(yaml_content)
        return data.get("active_work", {}) if data else {}
    except Exception:
        return {}


def main() -> int:
    _hook_in = json.load(sys.stdin)

    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd()))
    checkpoints_dir = project_dir / ".agents" / "outputs" / "claude_checkpoints"
    memory_dir = project_dir / ".claude" / "memory"

    print("## Restored Context\n")

    # 1. Load compact YAML state (preferred) or fallback to markdown
    yaml_state = checkpoints_dir / "PERSISTENT_STATE.yaml"
    md_state = checkpoints_dir / "PERSISTENT_STATE.md"

    active_work = {}
    if yaml_state.exists():
        yaml_content = yaml_state.read_text(encoding="utf-8")
        active_work = get_active_work(yaml_content)
        print("### Project State\n")
        print("```yaml")
        print(yaml_content)
        print("```\n")
    elif md_state.exists():
        print("### Project State\n")
        print(md_state.read_text(encoding="utf-8"))
        print()

    # 2. Load critical patterns only (top 3 patterns = 89% of failures)
    patterns_critical = memory_dir / "patterns-critical.md"
    if patterns_critical.exists():
        print("### Critical Patterns (Always Apply)\n")
        print(patterns_critical.read_text(encoding="utf-8"))
        print()
    else:
        # Fallback: print inline critical patterns
        print("### Critical Patterns\n")
        print("1. **VERIFICATION_GAP**: Read spec/code before assuming")
        print("2. **ENUM_VALUE**: Use VALUES not Python names (CO-OWNER not CO_OWNER)")
        print("3. **COMPONENT_API**: Read PropTypes before using components")
        print()
        print("Full patterns: `.claude/memory/patterns-full.md`\n")

    # 3. Check for active orchestrate workflow and provide continue instructions
    issue = active_work.get("issue")
    phase = active_work.get("phase")
    branch = active_work.get("branch")

    if issue and phase:
        print("### ACTIVE ORCHESTRATE WORKFLOW\n")
        print(f"**Issue**: #{issue}")
        print(f"**Phase**: {phase}")
        print(f"**Branch**: {branch}")
        print()
        print("**CRITICAL**: You were in the middle of an orchestrate workflow.")
        print("Continue with the current phase using the Task tool:")
        print()
        print("1. Read `.claude/commands/orchestrate.md` for phase instructions")
        print("2. Check for existing artifacts in `.agents/outputs/`")
        print(f"3. Continue the `{phase}` phase for issue #{issue}")
        print()
        print("If the phase was completed, proceed to the next phase in the workflow.")
        print()

    # 4. Hint about full patterns location
    print("---")
    print("*Full patterns available at `.claude/memory/patterns-full.md` if needed.*\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
