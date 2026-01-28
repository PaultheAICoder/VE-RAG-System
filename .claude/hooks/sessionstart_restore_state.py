#!/usr/bin/env python3
"""
SessionStart Hook: Restores saved state when a new session begins.
Injects PERSISTENT_STATE and critical patterns into context.
"""

import os
import sys
from pathlib import Path

# Try to import yaml, fall back to basic file reading if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def get_project_dir():
    """Get the project directory from environment or current working directory."""
    return os.environ.get('CLAUDE_PROJECT_DIR', os.getcwd())


def load_persistent_state():
    """Load the persistent state file."""
    project_dir = get_project_dir()
    state_file = Path(project_dir) / '.agents' / 'outputs' / 'claude_checkpoints' / 'PERSISTENT_STATE.yaml'

    if not state_file.exists():
        return None

    try:
        with open(state_file, 'r') as f:
            if HAS_YAML:
                return yaml.safe_load(f)
            else:
                # Return raw content if no YAML
                return {'_raw': f.read()}
    except Exception:
        return None


def load_critical_patterns():
    """Load the critical patterns file."""
    project_dir = get_project_dir()
    patterns_file = Path(project_dir) / '.claude' / 'memory' / 'patterns-critical.md'

    if not patterns_file.exists():
        return None

    try:
        with open(patterns_file, 'r') as f:
            return f.read()
    except Exception:
        return None


def format_state_output(state):
    """Format state for injection into context."""
    if not state:
        return ""

    if '_raw' in state:
        return f"```yaml\n{state['_raw']}\n```"

    lines = ["```yaml", "# Restored Session State"]

    if 'active_work' in state:
        lines.append("active_work:")
        for k, v in state['active_work'].items():
            lines.append(f"  {k}: {v}")

    if state.get('files_modified'):
        lines.append(f"files_modified: {state['files_modified']}")

    if state.get('pending_tasks'):
        lines.append(f"pending_tasks: {state['pending_tasks']}")

    lines.append("```")
    return '\n'.join(lines)


def get_continue_instructions(state):
    """Generate continuation instructions if active work detected."""
    if not state or not state.get('active_work'):
        return ""

    active = state['active_work']
    issue = active.get('issue')
    phase = active.get('phase')

    if not issue and not phase:
        return ""

    instructions = ["\n**Active Workflow Detected:**"]

    if issue:
        instructions.append(f"- Working on: {issue}")
    if phase:
        instructions.append(f"- Phase: {phase}")
    if active.get('branch') and active['branch'] != 'main':
        instructions.append(f"- Branch: {active['branch']}")

    instructions.append("\nConsider continuing from where you left off.")

    return '\n'.join(instructions)


def main():
    try:
        output_parts = []

        # Load and format persistent state
        state = load_persistent_state()
        if state:
            state_output = format_state_output(state)
            if state_output:
                output_parts.append("## Restored Session State")
                output_parts.append(state_output)

        # Load critical patterns
        patterns = load_critical_patterns()
        if patterns:
            output_parts.append("\n## Critical Patterns")
            output_parts.append(patterns)

        # Add continuation instructions
        if state:
            continue_inst = get_continue_instructions(state)
            if continue_inst:
                output_parts.append(continue_inst)

        # Output everything
        if output_parts:
            print('\n'.join(output_parts))

    except Exception as e:
        # Never fail the hook - just log and continue
        print(f"<!-- Session restore warning: {e} -->", file=sys.stderr)
        sys.exit(0)


if __name__ == '__main__':
    main()
