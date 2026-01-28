#!/usr/bin/env python3
"""
SessionStart Hook: Restores saved state on session start.
Injects PERSISTENT_STATE.yaml and critical patterns into context.
"""

import os
from pathlib import Path

# Try to import yaml, fall back to basic parsing if not available
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def get_project_dir():
    """Get the Claude project directory."""
    return os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())


def load_yaml_file(file_path):
    """Load a YAML file, with fallback for missing PyYAML."""
    if not file_path.exists():
        return None

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        if HAS_YAML:
            return yaml.safe_load(content)
        else:
            # Return raw content if no YAML parser
            return {"_raw": content}
    except Exception:
        return None


def load_markdown_file(file_path):
    """Load a markdown file."""
    if not file_path.exists():
        return None

    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def format_state_output(state):
    """Format state for minimal token output."""
    if not state:
        return None

    # Handle raw content fallback
    if "_raw" in state:
        return state["_raw"]

    lines = []

    active = state.get("active_work", {})
    if active.get("issue") or active.get("phase"):
        lines.append("active_work:")
        if active.get("issue"):
            lines.append(f"  issue: #{active['issue']}")
        if active.get("branch") and active["branch"] != "main":
            lines.append(f"  branch: {active['branch']}")
        if active.get("phase"):
            lines.append(f"  phase: {active['phase']}")
        if active.get("last_action"):
            lines.append(f"  last_action: {active['last_action']}")

    modified = state.get("modified_files", [])
    if modified:
        lines.append(f"modified_files: [{', '.join(modified[:5])}]")

    pending = state.get("pending_tasks", [])
    if pending:
        lines.append("pending_tasks:")
        for task in pending[:3]:
            lines.append(f"  - {task}")

    return "\n".join(lines) if lines else None


def has_active_workflow(state):
    """Check if there's an active workflow to continue."""
    if not state:
        return False
    active = state.get("active_work", {})
    return bool(active.get("issue") and active.get("phase"))


def main():
    """Main entry point for session start hook."""
    project_dir = Path(get_project_dir())
    output_parts = []

    # Load persistent state
    state_file = (
        project_dir / ".agents" / "outputs" / "claude_checkpoints" / "PERSISTENT_STATE.yaml"
    )
    state = load_yaml_file(state_file)

    if state:
        formatted_state = format_state_output(state)
        if formatted_state:
            output_parts.append("**Restored State:**")
            output_parts.append("```yaml")
            output_parts.append(formatted_state)
            output_parts.append("```")

    # Load critical patterns
    patterns_file = project_dir / ".claude" / "memory" / "patterns-critical.md"
    patterns = load_markdown_file(patterns_file)

    if patterns:
        # Truncate if too long (aim for ~600 tokens)
        max_chars = 2400  # ~600 tokens
        if len(patterns) > max_chars:
            patterns = patterns[:max_chars] + "\n...(truncated)"
        output_parts.append("\n**Critical Patterns:**")
        output_parts.append(patterns)

    # Add continue instructions if active workflow detected
    if has_active_workflow(state):
        active = state.get("active_work", {})
        output_parts.append("\n---")
        output_parts.append(
            f"**Continue:** Issue #{active.get('issue')} | Phase: {active.get('phase')}"
        )
        if active.get("last_action"):
            output_parts.append(f"Last action: {active.get('last_action')}")
        output_parts.append("Resume where you left off or ask for status update.")

    # Output everything
    if output_parts:
        print("\n".join(output_parts))
    else:
        # Minimal output if nothing to restore
        print("No saved state found. Starting fresh session.")


if __name__ == "__main__":
    main()
