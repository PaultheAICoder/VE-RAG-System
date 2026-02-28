"""ClaudeCliEnrichmentBackend — call claude -p subprocess instead of Anthropic HTTP API.

Allows full enrichment pipeline testing using an existing Claude Code login session
with no API key required.

Usage: set CLAUDE_BACKEND=cli (and CLAUDE_ENRICHMENT_ENABLED=true)
"""

from __future__ import annotations

import json
import logging
import subprocess

from ai_ready_rag.services.enrichment_service import (
    ENTITY_EXTRACTION_PROMPT,
    SYNOPSIS_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT = 120  # seconds


def _strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from text.

    Handles:
    - ```json\\n...\\n```
    - ```\\n...\\n```
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        # Remove closing fence
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")].rstrip()
    return stripped


class ClaudeCliEnrichmentBackend:
    """Enrichment backend that shells out to the claude CLI.

    Both methods are synchronous (subprocess is blocking). Callers in
    ClaudeEnrichmentService wrap them with asyncio.to_thread.
    """

    def call_synopsis(self, document_text: str, tenant_id: str = "default") -> str:
        """Call claude -p with the synopsis prompt and return stdout.

        Args:
            document_text: Full document text (truncated to 100k chars internally).
            tenant_id: Tenant identifier (unused by subprocess backend, kept for interface parity).

        Returns:
            Synopsis text string (stdout from claude -p, stripped).

        Raises:
            RuntimeError: If subprocess exits with non-zero code or times out.
        """
        text = document_text[:100_000]
        prompt = f"{SYNOPSIS_SYSTEM_PROMPT}\n\nDocument text:\n\n{text}"

        logger.debug("enrichment_cli.synopsis.start", extra={"tenant_id": tenant_id})
        result = subprocess.run(  # noqa: S603
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"claude -p exited with code {result.returncode}: {result.stderr.strip()}"
            )

        synopsis = result.stdout.strip()
        logger.debug(
            "enrichment_cli.synopsis.done",
            extra={"tenant_id": tenant_id, "chars": len(synopsis)},
        )
        return synopsis

    def call_entity_extraction(self, synopsis: str, chunk_text: str) -> list[dict]:
        """Call claude -p for entity extraction and parse JSON from stdout.

        Args:
            synopsis: Document synopsis text (truncated to 1000 chars internally).
            chunk_text: Concatenated chunk text to extract entities from.

        Returns:
            List of entity dicts parsed from stdout. Returns [] on parse errors.

        Raises:
            RuntimeError: If subprocess exits with non-zero code or times out.
        """
        prompt = (
            f"{ENTITY_EXTRACTION_PROMPT}\n\nSynopsis:\n{synopsis[:1000]}\n\nChunks:\n{chunk_text}"
        )

        logger.debug("enrichment_cli.entity_extraction.start")
        result = subprocess.run(  # noqa: S603
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"claude -p exited with code {result.returncode}: {result.stderr.strip()}"
            )

        raw = _strip_markdown_fences(result.stdout)
        try:
            items = json.loads(raw)
            if not isinstance(items, list):
                logger.warning("enrichment_cli.entity_extraction.non_list_response")
                return []
            # Filter out non-dict items defensively
            return [item for item in items if isinstance(item, dict)]
        except json.JSONDecodeError:
            logger.warning(
                "enrichment_cli.entity_extraction.json_parse_error",
                extra={"raw_preview": raw[:200]},
            )
            return []
