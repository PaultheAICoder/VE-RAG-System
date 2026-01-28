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

# Try to import yaml, fall back to basic serialization if not available
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def get_project_dir():
    """Get the Claude project directory."""
    return os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())


def get_checkpoint_dir():
    """Get the checkpoint output directory."""
    project_dir = get_project_dir()
    checkpoint_dir = Path(project_dir) / ".agents" / "outputs" / "claude_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def parse_transcript(transcript_path, max_lines=300):
    """Parse transcript file looking for state patterns."""
    state = {
        "active_work": {
            "issue": None,
            "branch": "main",
            "phase": None,
            "last_action": None,
        },
        "modified_files": [],
        "pending_tasks": [],
        "key_decisions": [],
        "extracted_at": datetime.now().isoformat(),
    }

    if not transcript_path or not Path(transcript_path).exists():
        return state

    try:
        with open(transcript_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Focus on last N lines
        recent_lines = lines[-max_lines:] if len(lines) > max_lines else lines
        content = "".join(recent_lines)

        # Extract issue numbers (e.g., "Issue #123", "#456", "issue-789")
        issue_patterns = [
            r"[Ii]ssue\s*#?(\d+)",
            r"#(\d+)",
            r"issue-(\d+)",
        ]
        for pattern in issue_patterns:
            matches = re.findall(pattern, content)
            if matches:
                state["active_work"]["issue"] = matches[-1]  # Use most recent
                break

        # Extract phase (MAP-PLAN, PATCH, PROVE, etc.)
        phase_patterns = [
            r"\b(MAP-PLAN|PATCH|PROVE|PLANNING|IMPLEMENTING|TESTING|REVIEW)\b",
            r"[Pp]hase:\s*(\w+)",
        ]
        for pattern in phase_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                state["active_work"]["phase"] = matches[-1].upper()
                break

        # Extract branch name
        branch_patterns = [
            r"[Bb]ranch:\s*([^\s\n]+)",
            r"git checkout\s+([^\s\n]+)",
            r"on branch\s+([^\s\n]+)",
        ]
        for pattern in branch_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                state["active_work"]["branch"] = matches[-1]
                break

        # Extract modified files
        file_patterns = [
            r'(?:modified|edited|created|wrote)\s+[`"]?([^\s`"]+\.\w+)[`"]?',
            r'Edit.*?file_path["\']:\s*["\']([^"\']+)["\']',
        ]
        modified_files = set()
        for pattern in file_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            modified_files.update(matches)
        state["modified_files"] = list(modified_files)[:20]  # Limit to 20

        # Extract TODO items
        todo_patterns = [
            r"TODO:\s*(.+?)(?:\n|$)",
            r"- \[ \]\s*(.+?)(?:\n|$)",
            r"PENDING:\s*(.+?)(?:\n|$)",
        ]
        todos = []
        for pattern in todo_patterns:
            matches = re.findall(pattern, content)
            todos.extend(matches)
        state["pending_tasks"] = todos[:10]  # Limit to 10

        # Extract key decisions
        decision_patterns = [
            r"[Dd]ecision:\s*(.+?)(?:\n|$)",
            r"[Dd]ecided to\s+(.+?)(?:\.|$)",
            r"[Ww]ill use\s+(.+?)(?:\.|$)",
        ]
        decisions = []
        for pattern in decision_patterns:
            matches = re.findall(pattern, content)
            decisions.extend(matches)
        state["key_decisions"] = decisions[:5]  # Limit to 5

        # Determine last action from recent content
        action_patterns = [
            (r'commit.*?["\'](.+?)["\']', "committed"),
            (r"created?\s+(\S+)", "created file"),
            (r"test.*?(pass|fail)", "ran tests"),
        ]
        for pattern, action_type in action_patterns:
            matches = re.findall(pattern, content[-5000:], re.IGNORECASE)
            if matches:
                state["active_work"]["last_action"] = f"{action_type}: {matches[-1]}"
                break

    except Exception as e:
        # Fault-tolerant: don't fail if parsing fails
        state["parse_error"] = str(e)

    return state


def save_state(state, checkpoint_dir):
    """Save state to YAML file."""
    state_file = checkpoint_dir / "PERSISTENT_STATE.yaml"

    try:
        if HAS_YAML:
            with open(state_file, "w", encoding="utf-8") as f:
                yaml.dump(state, f, default_flow_style=False, allow_unicode=True)
        else:
            # Fallback: simple YAML-like format
            with open(state_file, "w", encoding="utf-8") as f:
                f.write("# Auto-generated state (install PyYAML for better formatting)\n")
                f.write("active_work:\n")
                for k, v in state.get("active_work", {}).items():
                    f.write(f"  {k}: {v}\n")
                f.write(f"modified_files: {state.get('modified_files', [])}\n")
                f.write(f"pending_tasks: {state.get('pending_tasks', [])}\n")
                f.write(f"extracted_at: {state.get('extracted_at', '')}\n")
    except Exception as e:
        print(f"Warning: Could not save state: {e}", file=sys.stderr)


def backup_transcript(transcript_path, checkpoint_dir):
    """Create a backup of the raw transcript."""
    if not transcript_path or not Path(transcript_path).exists():
        return

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = checkpoint_dir / f"{timestamp}.transcript.jsonl"

        with open(transcript_path, encoding="utf-8") as src:
            with open(backup_file, "w", encoding="utf-8") as dst:
                dst.write(src.read())
    except Exception as e:
        print(f"Warning: Could not backup transcript: {e}", file=sys.stderr)


def create_summary_checkpoint(state, checkpoint_dir):
    """Create a markdown summary checkpoint."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = checkpoint_dir / f"{timestamp}_checkpoint.md"

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"# Checkpoint: {datetime.now().isoformat()}\n\n")
            f.write("## Active Work\n")
            active = state.get("active_work", {})
            f.write(f"- Issue: {active.get('issue', 'None')}\n")
            f.write(f"- Branch: {active.get('branch', 'main')}\n")
            f.write(f"- Phase: {active.get('phase', 'None')}\n")
            f.write(f"- Last Action: {active.get('last_action', 'None')}\n\n")

            if state.get("modified_files"):
                f.write("## Modified Files\n")
                for file in state["modified_files"]:
                    f.write(f"- {file}\n")
                f.write("\n")

            if state.get("pending_tasks"):
                f.write("## Pending Tasks\n")
                for task in state["pending_tasks"]:
                    f.write(f"- [ ] {task}\n")
                f.write("\n")

            if state.get("key_decisions"):
                f.write("## Key Decisions\n")
                for decision in state["key_decisions"]:
                    f.write(f"- {decision}\n")
    except Exception as e:
        print(f"Warning: Could not create summary: {e}", file=sys.stderr)


def main():
    """Main entry point for the precompact hook."""
    # Read input from stdin (JSON with transcript_path)
    transcript_path = None

    try:
        input_data = sys.stdin.read()
        if input_data.strip():
            data = json.loads(input_data)
            transcript_path = data.get("transcript_path")
    except (json.JSONDecodeError, KeyError):
        # No input or invalid JSON - try to find transcript another way
        pass

    checkpoint_dir = get_checkpoint_dir()

    # Parse and save state
    state = parse_transcript(transcript_path)
    save_state(state, checkpoint_dir)

    # Create backups
    backup_transcript(transcript_path, checkpoint_dir)
    create_summary_checkpoint(state, checkpoint_dir)

    print(f"Checkpoint saved to {checkpoint_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
