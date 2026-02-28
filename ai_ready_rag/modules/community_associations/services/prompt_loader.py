"""Loader for CA extraction prompt files.

Prompts are plain text files in the prompts/ directory.
They are loaded at startup and cached in memory.
"""

from __future__ import annotations

import logging
import pathlib

logger = logging.getLogger(__name__)

PROMPTS_DIR = pathlib.Path(__file__).parent.parent / "prompts"

# Canonical mapping from document_type → prompt filename
DOCUMENT_TYPE_TO_PROMPT: dict[str, str] = {
    "ccr": "ccr_bylaws.txt",
    "bylaws": "ccr_bylaws.txt",
    "reserve_study": "reserve_study.txt",
    "board_minutes": "board_minutes.txt",
    "appraisal": "appraisal.txt",
    "unit_owner_letter": "unit_owner_letter.txt",
}


def load_prompt(document_type: str) -> str | None:
    """Load the extraction prompt for a given document type.

    Args:
        document_type: CA document type (e.g., 'ccr', 'reserve_study')

    Returns:
        Prompt text string, or None if no prompt exists for this type.
    """
    filename = DOCUMENT_TYPE_TO_PROMPT.get(document_type)
    if filename is None:
        return None

    prompt_path = PROMPTS_DIR / filename
    if not prompt_path.exists():
        logger.warning(
            "ca.prompt.not_found", extra={"document_type": document_type, "path": str(prompt_path)}
        )
        return None

    return prompt_path.read_text(encoding="utf-8")


def list_available_prompts() -> list[str]:
    """Return list of document types with available prompts."""
    return [dt for dt, fn in DOCUMENT_TYPE_TO_PROMPT.items() if (PROMPTS_DIR / fn).exists()]
