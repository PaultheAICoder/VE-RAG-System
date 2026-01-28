#!/usr/bin/env python3
"""
Claude Code PreCompact hook (Optimized v2):
- Extracts structured state from transcript
- Updates PERSISTENT_STATE.yaml with extracted info
- Keeps transcript checkpoint for recovery
- Identifies issue numbers, phases, and key decisions
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

MAX_TAIL_LINES = 300  # Reduced from 600 - we extract structured state instead


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


def extract_state_from_transcript(transcript_lines: list[str]) -> dict:
    """Extract structured state from conversation transcript."""
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

            # Extract issue numbers (most recent wins)
            if matches := re.findall(r"[Ii]ssue[#\s]*(\d+)", content):
                state["last_issue"] = int(matches[-1])

            # Extract phase (most recent wins)
            for phase in ["MAP-PLAN", "TEST-PLAN", "CONTRACT", "PATCH", "PROVE"]:
                if phase in content:
                    state["last_phase"] = phase

            # Extract artifacts created
            if match := re.search(r"AGENT_RETURN:\s*(\S+\.md)", content):
                artifact = match.group(1)
                if artifact not in state["artifacts_created"]:
                    state["artifacts_created"].append(artifact)

            # Extract file modifications
            if match := re.search(
                r"(?:created|modified|updated).*?([a-zA-Z0-9_/]+\.(?:py|jsx?|tsx?|md))",
                content,
                re.I,
            ):
                filepath = match.group(1)
                if filepath not in state["files_modified"]:
                    state["files_modified"].append(filepath)

            # Extract TODO items (last 5)
            if "- [ ]" in content:
                todos = re.findall(r"- \[ \] (.+?)(?:\n|$)", content)
                for todo in todos[-5:]:
                    if todo not in state["pending_tasks"]:
                        state["pending_tasks"].append(todo)
                state["pending_tasks"] = state["pending_tasks"][-5:]

            # Extract key decisions (look for decision keywords)
            if any(kw in content.lower() for kw in ["decided", "decision:", "chose", "approach:"]):
                # Extract sentence containing decision
                sentences = re.split(r"[.!?]", content)
                for sent in sentences:
                    if any(
                        kw in sent.lower() for kw in ["decided", "decision", "chose", "approach"]
                    ):
                        clean = sent.strip()[:100]
                        if clean and clean not in state["key_decisions"]:
                            state["key_decisions"].append(clean)
                state["key_decisions"] = state["key_decisions"][-5:]

        except (json.JSONDecodeError, TypeError):
            continue

    return state


def update_persistent_state(state_file: Path, extracted: dict) -> None:
    """Update PERSISTENT_STATE.yaml with extracted info."""
    import yaml  # Only import if needed

    if not state_file.exists():
        return

    try:
        content = state_file.read_text(encoding="utf-8")
        data = yaml.safe_load(content) or {}

        # Update active_work section
        if "active_work" not in data:
            data["active_work"] = {}

        if extracted["last_issue"]:
            data["active_work"]["issue"] = extracted["last_issue"]
        if extracted["last_phase"]:
            data["active_work"]["phase"] = extracted["last_phase"]

        # Update with last action description
        if extracted["artifacts_created"]:
            data["active_work"]["last_action"] = f"Created {extracted['artifacts_created'][-1]}"

        # Update timestamp
        if "meta" not in data:
            data["meta"] = {}
        data["meta"]["updated"] = datetime.now().strftime("%Y-%m-%d")

        # Write back
        with state_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    except Exception as e:
        # Don't fail the hook if YAML update fails
        print(f"[precompact] Warning: Could not update PERSISTENT_STATE.yaml: {e}", file=sys.stderr)


def main() -> int:
    hook_in = json.load(sys.stdin)
    transcript_path = Path(hook_in["transcript_path"]).expanduser()

    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd()))
    out_dir = project_dir / ".agents" / "outputs" / "claude_checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = hook_in.get("session_id", "unknown-session")
    trigger = hook_in.get("trigger", "unknown")

    # 1) Copy the transcript as a raw checkpoint (for recovery)
    raw_dst = out_dir / f"{ts}__{session_id}__{trigger}.transcript.jsonl"
    shutil.copyfile(transcript_path, raw_dst)

    # 2) Extract structured state from transcript
    tail = tail_lines(transcript_path, MAX_TAIL_LINES)
    extracted = extract_state_from_transcript(tail)

    # 3) Create a compact summary (not full tail dump)
    md_dst = out_dir / f"{ts}__{session_id}__{trigger}.md"
    summary = f"""# Claude Code Checkpoint

- **Time**: {ts}
- **Session**: {session_id}
- **Trigger**: {trigger}
- **Transcript**: {raw_dst.name}

## Extracted State

- **Last Issue**: #{extracted["last_issue"] or "None"}
- **Last Phase**: {extracted["last_phase"] or "None"}
- **Artifacts Created**: {", ".join(extracted["artifacts_created"]) or "None"}

## Pending Tasks
{chr(10).join(f"- [ ] {t}" for t in extracted["pending_tasks"]) or "- None"}

## Files Modified
{chr(10).join(f"- {f}" for f in extracted["files_modified"][-10:]) or "- None"}

## Key Decisions
{chr(10).join(f"- {d}" for d in extracted["key_decisions"]) or "- None"}
"""
    md_dst.write_text(summary, encoding="utf-8")

    # 4) Update PERSISTENT_STATE.yaml with extracted info
    state_yaml = out_dir / "PERSISTENT_STATE.yaml"
    try:
        update_persistent_state(state_yaml, extracted)
        print("[precompact] Updated PERSISTENT_STATE.yaml")
    except ImportError:
        # PyYAML not available, skip update
        print("[precompact] Skipped PERSISTENT_STATE.yaml update (PyYAML not installed)")

    # PreCompact stdout is not injected into context; it's mostly for verbose logs.
    print(f"[precompact] Checkpointed transcript -> {raw_dst}")
    print(f"[precompact] Wrote summary -> {md_dst}")
    print(
        f"[precompact] Extracted: issue={extracted['last_issue']}, phase={extracted['last_phase']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
