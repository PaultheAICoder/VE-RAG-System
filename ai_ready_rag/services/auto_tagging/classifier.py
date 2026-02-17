"""LLM-based document classification using strategy prompts.

Orchestrates the LLM call, JSON parsing with repair logic, and the
confidence decision table to produce ClassificationResult objects.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

import httpx

from ai_ready_rag.config import Settings
from ai_ready_rag.services.auto_tagging.models import AutoTag
from ai_ready_rag.services.auto_tagging.strategy import AutoTagStrategy

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of LLM-based document classification."""

    tags: list[AutoTag] = field(default_factory=list)
    suggested: list[AutoTag] = field(default_factory=list)
    discarded: list[AutoTag] = field(default_factory=list)
    raw_response: str = ""
    parsed_response: dict | None = None
    status: str = "completed"
    error: str | None = None


class DocumentClassifier:
    """LLM-based document classification using strategy prompts."""

    def __init__(self, settings: Settings) -> None:
        self.ollama_url = settings.ollama_base_url
        self.model = settings.auto_tagging_llm_model
        self.timeout = settings.auto_tagging_llm_timeout_seconds
        self.max_retries = settings.auto_tagging_llm_max_retries
        self.confidence_threshold = settings.auto_tagging_confidence_threshold
        self.suggestion_threshold = settings.auto_tagging_suggestion_threshold
        self.require_approval = settings.auto_tagging_require_approval

    async def classify(
        self,
        strategy: AutoTagStrategy,
        filename: str,
        source_path: str,
        content_preview: str,
    ) -> ClassificationResult:
        """Classify a document using LLM and strategy prompts.

        Args:
            strategy: The loaded auto-tag strategy.
            filename: Original filename.
            source_path: Upload source path.
            content_preview: First 2000 chars of extracted text.

        Returns:
            ClassificationResult with applied, suggested, and discarded tags.
        """
        prompt = strategy.build_llm_prompt(filename, source_path, content_preview)

        raw_text = await self._call_llm(prompt)
        if raw_text is None:
            return ClassificationResult(
                status="partial",
                error="LLM call failed after retries",
            )

        parsed = self._parse_json_response(raw_text)
        if parsed is None:
            return ClassificationResult(
                raw_response=raw_text,
                status="partial",
                error="Failed to parse LLM response as JSON",
            )

        all_tags = strategy.parse_llm_response(parsed)
        applied, suggested, discarded = self._apply_decision_table(all_tags)

        logger.info(
            "Classification complete: %d applied, %d suggested, %d discarded",
            len(applied),
            len(suggested),
            len(discarded),
        )

        return ClassificationResult(
            tags=applied,
            suggested=suggested,
            discarded=discarded,
            raw_response=raw_text,
            parsed_response=parsed,
            status="completed",
            error=None,
        )

    async def _call_llm(self, prompt: str) -> str | None:
        """Call Ollama /api/generate with timeout and exponential backoff retry."""
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.ollama_url}/api/generate",
                        json={"model": self.model, "prompt": prompt, "stream": False},
                    )
                    response.raise_for_status()
                    return response.json().get("response", "").strip()
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt < self.max_retries:
                    backoff = 2**attempt
                    logger.warning(
                        "LLM call attempt %d failed: %s, retrying in %ds",
                        attempt + 1,
                        e,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        "LLM call failed after %d attempts: %s",
                        self.max_retries + 1,
                        e,
                    )
                    return None
            except httpx.HTTPStatusError as e:
                logger.error("LLM HTTP error (no retry): %s", e)
                return None
        return None

    def _parse_json_response(self, raw_text: str) -> dict | None:
        """Parse JSON from LLM response with repair attempts.

        Tries three strategies in order:
        1. Strict JSON parse
        2. Strip markdown code fences and parse
        3. Extract first {...} block and parse
        """
        # 1. Strict parse
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # 2. Strip markdown fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 3. Extract first {...} block
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.error("Failed to parse LLM response as JSON after all repair attempts")
        return None

    def _apply_decision_table(
        self, tags: list[AutoTag]
    ) -> tuple[list[AutoTag], list[AutoTag], list[AutoTag]]:
        """Apply the 6-cell confidence decision table.

        Returns:
            Tuple of (applied, suggested, discarded) tag lists.
        """
        applied: list[AutoTag] = []
        suggested: list[AutoTag] = []
        discarded: list[AutoTag] = []

        for tag in tags:
            if tag.confidence < self.suggestion_threshold:
                discarded.append(tag)
            elif self.require_approval:
                suggested.append(tag)
            else:
                applied.append(tag)

        return applied, suggested, discarded
