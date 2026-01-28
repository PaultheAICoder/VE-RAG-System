#!/usr/bin/env python3
"""
Claude Code PreCompact hook:
- Extracts structured state from transcript (JSONL format)
- Uses git for reliable branch detection
- Updates PERSISTENT_STATE.yaml with extracted info
- Creates checkpoint files for recovery
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

MAX_TAIL_LINES = 300


def tail_lines(path: Path, max_lines: int) -> list[str]:
    """Get last N lines from file."""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        block = 8192
        data = b""
        while size > 0 and data.count(b"\n") <= max_lines:
            step = min(block, size)
            size -= step
            f.seek(size)
            data = f.read(step) + data
        lines = data.splitlines()[-max_lines:]
        return [ln.decode("utf-8", errors="replace") for ln in lines]


def get_git_branch(project_dir: Path) -> str:
    """Get current git branch directly from git."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "main"
    except Exception:
        pass
    return "main"


def extract_state_from_transcript(transcript_lines: list[str]) -> dict:
    """Extract structured state from conversation transcript (JSONL format)."""
    state = {
        "last_issue": None,
        "last_phase": None,
        "last_action": None,
        "pending_tasks": [],
        "key_decisions": [],
        "files_modified": [],
        "artifacts_created": [],
    }

    for line in transcript_lines:
        try:
            msg = json.loads(line)
            content = str(msg.get("content", ""))

            # Extract issue numbers from issue file references (most reliable)
            if (matches := re.findall(r"issues/(\d{3})-", content)) or (
                matches := re.findall(r"[Ii]ssue[#\s]*(\d{1,3})\b", content)
            ):
                state["last_issue"] = int(matches[-1])

            # Extract phase (most recent wins)
            for phase in ["MAP-PLAN", "MAP", "PLAN", "PATCH", "PROVE"]:
                if re.search(rf"\b{phase}\b", content):
                    state["last_phase"] = phase

            # Extract artifacts created
            if match := re.search(r"AGENT_RETURN:\s*(\S+\.md)", content):
                artifact = match.group(1)
                if artifact not in state["artifacts_created"]:
                    state["artifacts_created"].append(artifact)

            # Extract file modifications from tool calls
            if '"file_path"' in content:
                for match in re.finditer(r'"file_path":\s*"([^"]+)"', content):
                    filepath = match.group(1)
                    if filepath not in state["files_modified"] and not filepath.endswith(".log"):
                        state["files_modified"].append(filepath)

            # Extract last action from recent activity
            if "commit" in content.lower() and (
                "message" in content.lower() or "git commit" in content.lower()
            ):
                state["last_action"] = "Git commit"
            elif "tests" in content.lower() and "pass" in content.lower():
                state["last_action"] = "Tests passed"

        except (json.JSONDecodeError, TypeError):
            continue

    # Limit lists
    state["files_modified"] = state["files_modified"][-15:]
    state["pending_tasks"] = state["pending_tasks"][-5:]
    state["key_decisions"] = state["key_decisions"][-5:]

    return state


def update_persistent_state(state_file: Path, extracted: dict, branch: str) -> None:
    """Update PERSISTENT_STATE.yaml with extracted info."""
    try:
        import yaml
    except ImportError:
        return

    # Create default structure if file doesn't exist
    if state_file.exists():
        try:
            content = state_file.read_text(encoding="utf-8")
            data = yaml.safe_load(content) or {}
        except Exception:
            data = {}
    else:
        data = {}

    # Update active_work section
    if "active_work" not in data:
        data["active_work"] = {}

    if extracted["last_issue"]:
        data["active_work"]["issue"] = str(extracted["last_issue"])
    data["active_work"]["branch"] = branch
    if extracted["last_phase"]:
        data["active_work"]["phase"] = extracted["last_phase"]
    if extracted["last_action"]:
        data["active_work"]["last_action"] = extracted["last_action"]
    elif extracted["artifacts_created"]:
        data["active_work"]["last_action"] = f"Created {extracted['artifacts_created'][-1]}"

    # Update modified files
    if extracted["files_modified"]:
        data["modified_files"] = extracted["files_modified"]

    # Update timestamp
    data["extracted_at"] = datetime.now().isoformat()

    # Write back
    with state_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def main() -> int:
    # Read hook input from stdin
    try:
        hook_in = json.load(sys.stdin)
    except json.JSONDecodeError:
        print("[precompact] Error: Invalid JSON input", file=sys.stderr)
        return 1

    transcript_path = Path(hook_in.get("transcript_path", "")).expanduser()
    if not transcript_path.exists():
        print(f"[precompact] Error: Transcript not found: {transcript_path}", file=sys.stderr)
        return 1

    # Get project directory
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd()))
    # Handle Windows mount path
    if str(project_dir).startswith("/mnt/c/Users/jjob/projects/"):
        wsl_path = Path(
            str(project_dir).replace("/mnt/c/Users/jjob/projects/", "/home/jjob/projects/")
        )
        if wsl_path.exists():
            project_dir = wsl_path

    out_dir = project_dir / ".agents" / "outputs" / "claude_checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = hook_in.get("session_id", "unknown")
    trigger = hook_in.get("trigger", "compact")

    # 1) Copy the transcript as a raw checkpoint
    raw_dst = out_dir / f"{ts}__{session_id}__{trigger}.transcript.jsonl"
    shutil.copyfile(transcript_path, raw_dst)

    # 2) Get git branch (reliable source)
    branch = get_git_branch(project_dir)

    # 3) Extract structured state from transcript
    tail = tail_lines(transcript_path, MAX_TAIL_LINES)
    extracted = extract_state_from_transcript(tail)

    # 4) Update PERSISTENT_STATE.yaml
    state_yaml = out_dir / "PERSISTENT_STATE.yaml"
    update_persistent_state(state_yaml, extracted, branch)

    # 5) Create markdown summary
    md_dst = out_dir / f"{ts}__{session_id}__{trigger}.md"
    summary = f"""# Claude Code Checkpoint

- **Time**: {ts}
- **Session**: {session_id}
- **Trigger**: {trigger}
- **Branch**: {branch}

## Extracted State

- **Issue**: #{extracted["last_issue"] or "None"}
- **Phase**: {extracted["last_phase"] or "None"}
- **Last Action**: {extracted["last_action"] or "None"}

## Files Modified
{chr(10).join(f"- {f}" for f in extracted["files_modified"][-10:]) or "- None"}
"""
    md_dst.write_text(summary, encoding="utf-8")

    # Output for logs
    print(f"[precompact] Branch: {branch}")
    print(f"[precompact] Issue: #{extracted['last_issue']}, Phase: {extracted['last_phase']}")
    print(f"[precompact] Saved: {state_yaml.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
