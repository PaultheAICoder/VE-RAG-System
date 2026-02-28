"""ClaudeEnrichmentService — two-call document enrichment pipeline.

Call 1 (Synopsis): claude-sonnet-4-6 with prompt caching
  - Input: full document text (up to 200k tokens)
  - Output: structured synopsis stored in enrichment_synopses

Call 2 (Entities): claude-haiku-4-5-20251001 per chunk batch
  - Input: synopsis + chunk batch
  - Output: typed entities stored in enrichment_entities

No-op on sqlite profile — Ollama/RAG path is unaffected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

SYNOPSIS_SYSTEM_PROMPT = """You are a document analyst specializing in insurance and property management documents.
Extract a structured synopsis from the document. Return valid JSON with:
{
  "document_type": "<string>",
  "key_entities": [{"type": "<string>", "value": "<string>", "confidence": <0.0-1.0>}],
  "summary": "<string, 2-3 sentences>",
  "coverage_lines": ["<string>"],
  "date_references": ["<string>"],
  "amounts": [{"label": "<string>", "amount": <number>, "currency": "USD"}]
}
Return ONLY valid JSON. No explanation."""

ENTITY_EXTRACTION_PROMPT = """Given this document synopsis and chunk batch, extract typed entities.
Return a JSON array:
[{"entity_type": "<string>", "value": "<string>", "canonical_value": "<string or null>", "confidence": <0.0-1.0>, "chunk_index": <int>}]
Entity types: insurance_carrier, coverage_line, coverage_limit, deductible, policy_number, effective_date, expiration_date, account_name, association_name
Return ONLY valid JSON array. No explanation."""


@dataclass
class SynopsisResult:
    synopsis_text: str
    model_id: str
    token_cost: int
    cost_usd: float
    raw_json: dict[str, Any]


@dataclass
class EntityResult:
    entity_type: str
    value: str
    canonical_value: str | None
    confidence: float
    source_chunk_index: int | None


class ClaudeEnrichmentService:
    """Two-call enrichment pipeline using Claude API.

    Gracefully degrades to no-op on SQLite / laptop dev profile.
    """

    def __init__(self, settings: Any, db_session: Any = None) -> None:
        self._settings = settings
        self._db = db_session
        self._client = None

    def _is_enabled(self) -> bool:
        """Return True only when Claude enrichment is explicitly enabled."""
        enabled = getattr(self._settings, "claude_enrichment_enabled", None)
        api_key = getattr(self._settings, "claude_api_key", None)
        database_backend = getattr(self._settings, "database_backend", "sqlite")
        if database_backend == "sqlite":
            return False
        return bool(enabled and api_key)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic

                api_key = getattr(self._settings, "claude_api_key", None)
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError as err:
                raise RuntimeError(
                    "anthropic package not installed. Add anthropic>=0.40.0 to requirements-wsl.txt"
                ) from err
        return self._client

    async def enrich_document(
        self,
        document_id: str,
        document_text: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the full two-call enrichment pipeline for a document.

        Returns enrichment result dict. No-op (returns empty dict) if not enabled.
        """
        if not self._is_enabled():
            logger.debug(
                "enrichment.skipped",
                extra={"document_id": document_id, "reason": "not_enabled_or_sqlite"},
            )
            return {}

        try:
            synopsis = await self._call_synopsis(document_id, document_text)
            entities = await self._call_entity_extraction(document_id, synopsis, chunks)

            result = {
                "document_id": document_id,
                "synopsis": synopsis,
                "entities": entities,
                "enrichment_model": synopsis.model_id,
                "token_cost": synopsis.token_cost,
                "cost_usd": synopsis.cost_usd,
            }

            if self._db is not None:
                await self._persist(document_id, synopsis, entities)

            logger.info(
                "enrichment.completed",
                extra={
                    "document_id": document_id,
                    "entities_count": len(entities),
                    "cost_usd": synopsis.cost_usd,
                },
            )
            return result
        except Exception as exc:
            logger.error(
                "enrichment.failed",
                extra={"document_id": document_id, "error": str(exc)},
            )
            raise

    async def _call_synopsis(self, document_id: str, document_text: str) -> SynopsisResult:
        """Call claude-sonnet-4-6 for document synopsis with prompt caching."""
        import json

        client = self._get_client()
        model = getattr(self._settings, "claude_enrichment_model", "claude-sonnet-4-6")
        timeout = getattr(self._settings, "claude_enrichment_timeout", 60)

        # Truncate to ~100k chars to stay within token limits
        text = document_text[:100_000]

        message = client.messages.create(
            model=model,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": SYNOPSIS_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},  # prompt caching
                }
            ],
            messages=[{"role": "user", "content": f"Document text:\n\n{text}"}],
            timeout=timeout,
        )

        content = message.content[0].text if message.content else "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"raw": content}

        input_tokens = getattr(message.usage, "input_tokens", 0)
        output_tokens = getattr(message.usage, "output_tokens", 0)
        token_cost = input_tokens + output_tokens
        # Approximate cost: sonnet-4-6 input ~$3/1M, output ~$15/1M
        cost_usd = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000

        return SynopsisResult(
            synopsis_text=content,
            model_id=model,
            token_cost=token_cost,
            cost_usd=cost_usd,
            raw_json=parsed,
        )

    async def _call_entity_extraction(
        self,
        document_id: str,
        synopsis: SynopsisResult,
        chunks: list[dict[str, Any]],
    ) -> list[EntityResult]:
        """Call claude-haiku for entity extraction on chunk batches."""
        import json

        client = self._get_client()
        simple_model = getattr(
            self._settings,
            "claude_query_model_simple",
            "claude-haiku-4-5-20251001",
        )
        batch_size = getattr(self._settings, "claude_enrichment_batch_size", 8)

        all_entities: list[EntityResult] = []

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            batch_text = "\n---\n".join(
                f"[Chunk {batch_start + i}] {c.get('text', c.get('content', ''))[:500]}"
                for i, c in enumerate(batch)
            )

            prompt = f"Synopsis:\n{synopsis.synopsis_text[:1000]}\n\nChunks:\n{batch_text}"

            message = client.messages.create(
                model=simple_model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": ENTITY_EXTRACTION_PROMPT + "\n\n" + prompt,
                    }
                ],
            )

            content = message.content[0].text if message.content else "[]"
            try:
                items = json.loads(content)
                if not isinstance(items, list):
                    items = []
            except json.JSONDecodeError:
                items = []

            for item in items:
                if not isinstance(item, dict):
                    continue
                all_entities.append(
                    EntityResult(
                        entity_type=str(item.get("entity_type", "unknown")),
                        value=str(item.get("value", "")),
                        canonical_value=item.get("canonical_value"),
                        confidence=float(item.get("confidence", 0.5)),
                        source_chunk_index=item.get("chunk_index"),
                    )
                )

        return all_entities

    async def _persist(
        self,
        document_id: str,
        synopsis: SynopsisResult,
        entities: list[EntityResult],
    ) -> None:
        """Persist synopsis and entities to the database."""
        try:
            from ai_ready_rag.db.models.base import generate_uuid
            from ai_ready_rag.db.models.enrichment import (
                EnrichmentEntity,
                EnrichmentSynopsis,
            )

            synopsis_id = generate_uuid()
            synopsis_obj = EnrichmentSynopsis(
                id=synopsis_id,
                document_id=document_id,
                synopsis_text=synopsis.synopsis_text,
                model_id=synopsis.model_id,
                token_cost=synopsis.token_cost,
                cost_usd=synopsis.cost_usd,
            )
            self._db.add(synopsis_obj)

            for entity in entities:
                self._db.add(
                    EnrichmentEntity(
                        id=generate_uuid(),
                        synopsis_id=synopsis_id,
                        entity_type=entity.entity_type,
                        value=entity.value,
                        canonical_value=entity.canonical_value,
                        confidence=entity.confidence,
                        source_chunk_index=entity.source_chunk_index,
                    )
                )

            self._db.commit()
        except Exception as exc:
            logger.error(
                "enrichment.persist.failed",
                extra={"document_id": document_id, "error": str(exc)},
            )
            if self._db:
                self._db.rollback()
            raise
