"""ClaudeQueryService — Standard tier query engine using Claude API.

Routes queries through ClaudeModelRouter:
- Simple/factual queries → claude-haiku-4-5-20251001 (fast, cheap)
- Complex/analytical queries → claude-sonnet-4-6 (accurate, expensive)

Only active when claude_query_enabled=True and claude_api_key is set.
SQLite / Ollama path is completely unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Query complexity signals — if ANY of these patterns are found, use complex model
COMPLEX_QUERY_SIGNALS = [
    "compare",
    "analyze",
    "explain why",
    "summarize",
    "what are the implications",
    "compliance",
    "gap",
    "recommend",
    "strategy",
    "year over year",
    "trend",
    "across all",
    "versus",
    "pros and cons",
    "risk assessment",
]


@dataclass
class QueryResponse:
    answer: str
    model_used: str
    is_complex: bool
    token_cost: int
    cost_usd: float
    route_type: str  # "claude_simple" | "claude_complex" | "ollama_fallback"


class ClaudeModelRouter:
    """Determines which Claude model to use based on query complexity.

    Configurable via TenantConfig.ai_models or Settings fallback.
    """

    def __init__(
        self,
        simple_model: str = "claude-haiku-4-5-20251001",
        complex_model: str = "claude-sonnet-4-6",
        complex_signals: list[str] | None = None,
    ) -> None:
        self._simple_model = simple_model
        self._complex_model = complex_model
        self._signals = [s.lower() for s in (complex_signals or COMPLEX_QUERY_SIGNALS)]

    def select_model(self, query: str) -> tuple[str, bool]:
        """Return (model_id, is_complex) for the given query."""
        query_lower = query.lower()
        is_complex = any(signal in query_lower for signal in self._signals)
        model = self._complex_model if is_complex else self._simple_model
        return model, is_complex

    @classmethod
    def from_settings(cls, settings: Any) -> ClaudeModelRouter:
        """Build router from global settings."""
        return cls(
            simple_model=getattr(
                settings, "claude_query_model_simple", "claude-haiku-4-5-20251001"
            ),
            complex_model=getattr(settings, "claude_query_model_complex", "claude-sonnet-4-6"),
        )

    @classmethod
    def from_tenant_config(cls, tenant_config: Any, settings: Any) -> ClaudeModelRouter:
        """Build router from tenant config, falling back to global settings."""
        try:
            ai_models = tenant_config.ai_models
            simple = getattr(ai_models, "query_model_simple", None)
            complex_ = getattr(ai_models, "query_model_complex", None)
            return cls(
                simple_model=simple
                or getattr(settings, "claude_query_model_simple", "claude-haiku-4-5-20251001"),
                complex_model=complex_
                or getattr(settings, "claude_query_model_complex", "claude-sonnet-4-6"),
            )
        except Exception:
            return cls.from_settings(settings)


class ClaudeQueryService:
    """Query answering service using Claude API (Standard tier).

    Gracefully degrades to no-op (returns None) when disabled,
    allowing the caller to fall back to Ollama/RAG.
    """

    def __init__(self, settings: Any, tenant_config: Any = None) -> None:
        self._settings = settings
        self._tenant_config = tenant_config
        self._client = None
        self._model_router = (
            ClaudeModelRouter.from_tenant_config(tenant_config, settings)
            if tenant_config
            else ClaudeModelRouter.from_settings(settings)
        )

    def _is_enabled(self) -> bool:
        """Return True only when all conditions for Claude query are satisfied."""
        enabled = getattr(self._settings, "claude_query_enabled", None)
        api_key = getattr(self._settings, "claude_api_key", None)
        db_backend = getattr(self._settings, "database_backend", "sqlite")
        if db_backend == "sqlite":
            return False
        return bool(enabled and api_key)

    def _get_client(self):
        """Lazily initialise the Anthropic client; raises RuntimeError if package missing."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=getattr(self._settings, "claude_api_key", None)
                )
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic package not installed. Add anthropic>=0.40.0 to requirements-wsl.txt"
                ) from exc
        return self._client

    async def answer(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        system_prompt_suffix: str = "",
    ) -> QueryResponse | None:
        """Answer a query using Claude with retrieved context chunks.

        Returns None if service is disabled — caller should use Ollama fallback.
        """
        if not self._is_enabled():
            logger.debug("claude_query.disabled", extra={"reason": "not_enabled_or_sqlite"})
            return None

        model_id, is_complex = self._model_router.select_model(query)

        # Build context string from chunks
        context_text = "\n\n".join(
            f"[Source {i + 1}]: {chunk.get('content', chunk.get('text', ''))}"
            for i, chunk in enumerate(context_chunks[:10])  # limit to 10 chunks
        )

        system_prompt = (
            "You are a knowledgeable insurance and property management assistant. "
            "Answer questions using ONLY the provided context. "
            "If the context doesn't contain the answer, say so clearly. "
            "Cite specific sources when possible."
            + (f"\n\n{system_prompt_suffix}" if system_prompt_suffix else "")
        )

        user_message = f"Context:\n{context_text}\n\nQuestion: {query}"

        try:
            client = self._get_client()
            message = client.messages.create(
                model=model_id,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                timeout=getattr(self._settings, "claude_enrichment_timeout", 60),
            )

            answer_text = message.content[0].text if message.content else ""
            input_tokens = getattr(message.usage, "input_tokens", 0)
            output_tokens = getattr(message.usage, "output_tokens", 0)
            token_cost = input_tokens + output_tokens

            # Cost calculation based on model
            if "haiku" in model_id:
                cost_usd = (input_tokens * 0.8 + output_tokens * 4.0) / 1_000_000
            else:
                cost_usd = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000

            route_type = "claude_complex" if is_complex else "claude_simple"

            logger.info(
                "claude_query.answered",
                extra={
                    "model": model_id,
                    "is_complex": is_complex,
                    "cost_usd": cost_usd,
                },
            )
            return QueryResponse(
                answer=answer_text,
                model_used=model_id,
                is_complex=is_complex,
                token_cost=token_cost,
                cost_usd=cost_usd,
                route_type=route_type,
            )

        except Exception as exc:
            logger.error("claude_query.failed", extra={"error": str(exc), "model": model_id})
            raise
