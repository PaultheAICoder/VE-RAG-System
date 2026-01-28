#!/usr/bin/env python3
"""
PreCompact Hook: Saves conversation state before context compaction.
Extracts structured state from transcript and saves to PERSISTENT_STATE.yaml
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Try to import yaml, fall back to basic file writing if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def get_project_dir():
    """Get the project directory from environment or current working directory."""
    return os.environ.get('CLAUDE_PROJECT_DIR', os.getcwd())


def get_checkpoint_dir():
    """Get the checkpoint directory path."""
    project_dir = get_project_dir()
    checkpoint_dir = Path(project_dir) / '.agents' / 'outputs' / 'claude_checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def extract_state_from_transcript(transcript_lines):
    """Extract structured state from transcript content."""
    state = {
        'active_work': {
            'issue': None,
            'branch': 'main',
            'phase': None,
            'last_action': None,
        },
        'files_modified': [],
        'pending_tasks': [],
        'key_decisions': [],
        'extracted_at': datetime.now().isoformat(),
    }

    # Patterns to extract
    issue_pattern = re.compile(r'[Ii]ssue\s*#?(\d+)|#(\d+)')
    phase_pattern = re.compile(r'\b(MAP|PLAN|PATCH|PROVE|REVIEW|DEBUG|REFACTOR)\b', re.IGNORECASE)
    file_pattern = re.compile(r'(?:modified|created|edited|updated|changed)\s+[`"\']?([^\s`"\']+\.[a-z]+)[`"\']?', re.IGNORECASE)
    todo_pattern = re.compile(r'(?:TODO|FIXME|PENDING|NEXT):\s*(.+)', re.IGNORECASE)
    branch_pattern = re.compile(r'branch[:\s]+[`"\']?([a-zA-Z0-9_/-]+)[`"\']?', re.IGNORECASE)

    files_seen = set()

    for line in transcript_lines:
        # Extract issue numbers
        issue_match = issue_pattern.search(line)
        if issue_match:
            issue_num = issue_match.group(1) or issue_match.group(2)
            state['active_work']['issue'] = f"#{issue_num}"

        # Extract phase
        phase_match = phase_pattern.search(line)
        if phase_match:
            state['active_work']['phase'] = phase_match.group(1).upper()

        # Extract modified files
        file_match = file_pattern.search(line)
        if file_match:
            filename = file_match.group(1)
            if filename not in files_seen:
                files_seen.add(filename)
                state['files_modified'].append(filename)

        # Extract TODOs
        todo_match = todo_pattern.search(line)
        if todo_match:
            todo_text = todo_match.group(1).strip()
            if todo_text and len(todo_text) < 200:
                state['pending_tasks'].append(todo_text)

        # Extract branch
        branch_match = branch_pattern.search(line)
        if branch_match:
            state['active_work']['branch'] = branch_match.group(1)

    # Limit lists to avoid bloat
    state['files_modified'] = state['files_modified'][-10:]
    state['pending_tasks'] = state['pending_tasks'][-5:]

    return state


def save_state(state, checkpoint_dir):
    """Save state to YAML file."""
    state_file = checkpoint_dir / 'PERSISTENT_STATE.yaml'

    if HAS_YAML:
        with open(state_file, 'w') as f:
            yaml.dump(state, f, default_flow_style=False, sort_keys=False)
    else:
        # Fallback: write as simple text format
        with open(state_file, 'w') as f:
            f.write("# PERSISTENT_STATE (YAML library not available)\n")
            f.write(f"# extracted_at: {state['extracted_at']}\n")
            f.write(f"active_work:\n")
            for k, v in state['active_work'].items():
                f.write(f"  {k}: {v}\n")
            f.write(f"files_modified: {state['files_modified']}\n")
            f.write(f"pending_tasks: {state['pending_tasks']}\n")


def save_transcript_backup(transcript_path, checkpoint_dir):
    """Save a backup of the raw transcript."""
    if not transcript_path or not os.path.exists(transcript_path):
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = checkpoint_dir / f'{timestamp}.transcript.jsonl'

    try:
        with open(transcript_path, 'r') as src:
            # Only save last 500 lines to avoid huge files
            lines = src.readlines()[-500:]
        with open(backup_file, 'w') as dst:
            dst.writelines(lines)
    except Exception:
        pass  # Don't fail if backup fails


def create_markdown_summary(state, checkpoint_dir):
    """Create a human-readable markdown checkpoint."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = checkpoint_dir / f'{timestamp}_checkpoint.md'

    content = f"""# Checkpoint: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Active Work
- **Issue**: {state['active_work']['issue'] or 'None'}
- **Branch**: {state['active_work']['branch']}
- **Phase**: {state['active_work']['phase'] or 'None'}

## Files Modified
{chr(10).join(f'- {f}' for f in state['files_modified']) or '- None tracked'}

## Pending Tasks
{chr(10).join(f'- {t}' for t in state['pending_tasks']) or '- None tracked'}
"""

    with open(summary_file, 'w') as f:
        f.write(content)


def main():
    try:
        # Read input from stdin (Claude passes JSON with transcript_path)
        input_data = sys.stdin.read().strip()

        transcript_path = None
        transcript_lines = []

        if input_data:
            try:
                data = json.loads(input_data)
                transcript_path = data.get('transcript_path')

                if transcript_path and os.path.exists(transcript_path):
                    with open(transcript_path, 'r') as f:
                        # Read last 300 lines for efficiency
                        transcript_lines = f.readlines()[-300:]
            except json.JSONDecodeError:
                # Input might be raw transcript
                transcript_lines = input_data.split('\n')[-300:]

        checkpoint_dir = get_checkpoint_dir()

        # Extract state
        state = extract_state_from_transcript(transcript_lines)

        # Save everything
        save_state(state, checkpoint_dir)
        save_transcript_backup(transcript_path, checkpoint_dir)
        create_markdown_summary(state, checkpoint_dir)

        print(f"Checkpoint saved to {checkpoint_dir}")

    except Exception as e:
        # Never fail the hook - just log and continue
        print(f"Warning: Checkpoint hook encountered an error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == '__main__':
    main()
