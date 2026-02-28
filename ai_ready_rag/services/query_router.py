"""Deterministic QueryRouter — rule-based SQL-first query routing.

No LLM call is made during routing. Routing decisions are:
1. SQL-first: if entity extraction + trigger phrases + confidence >= threshold -> structured query
2. RAG fallback: all other queries -> vector search

Modules contribute SQL templates and trigger phrases via ModuleRegistry.register_sql_templates().
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RouteType(str, Enum):
    SQL = "sql"
    RAG = "rag"
    REVIEW = "review"  # routed to human review (confidence below floor)


@dataclass
class RoutingDecision:
    route: RouteType
    template_name: str | None = None  # set when route == SQL
    confidence: float = 0.0
    matched_phrases: list[str] = field(default_factory=list)
    reason: str = ""


class QueryRouter:
    """Deterministic query router.

    Routing logic (in order):
    1. Extract trigger phrases from query text
    2. Find matching SQL templates in registry
    3. Score match confidence
    4. If confidence >= sql_threshold -> SQL route
    5. Otherwise -> RAG route
    6. If confidence below review_floor AND structured_query_enabled -> REVIEW route

    This is intentionally simple and auditable. No LLM, no ambiguity.
    """

    def __init__(
        self,
        sql_confidence_threshold: float = 0.6,
        review_floor: float = 0.0,  # disabled by default
    ) -> None:
        self._sql_threshold = sql_confidence_threshold
        self._review_floor = review_floor
        self._registry = None

    def _get_registry(self):
        if self._registry is None:
            from ai_ready_rag.modules.registry import ModuleRegistry

            # Use get_instance() so the router always shares the same singleton
            # as main.py and module startup code, even when called before
            # init_registry() (e.g. in tests — returns an empty registry).
            self._registry = ModuleRegistry.get_instance()
        return self._registry

    def route(self, query: str, structured_query_enabled: bool = False) -> RoutingDecision:
        """Route a query to SQL or RAG based on deterministic rules."""
        if not structured_query_enabled:
            return RoutingDecision(
                route=RouteType.RAG,
                confidence=0.0,
                reason="structured_query_disabled",
            )

        registry = self._get_registry()
        templates = registry.sql_templates

        if not templates:
            return RoutingDecision(
                route=RouteType.RAG,
                confidence=0.0,
                reason="no_sql_templates_registered",
            )

        best_template: str | None = None
        best_confidence: float = 0.0
        best_phrases: list[str] = []

        query_lower = query.lower()

        for name, template in templates.items():
            matched = []
            for phrase in template.trigger_phrases:
                # Word-boundary match (case-insensitive)
                if re.search(r"\b" + re.escape(phrase.lower()) + r"\b", query_lower):
                    matched.append(phrase)

            if not matched:
                continue

            # Presence-based confidence: trigger_phrases are domain vocabulary, not a
            # checklist.  Any matching phrase means the query is in-domain → base 0.7.
            # A density bonus (up to 0.3) rewards more matches and is used for template
            # disambiguation when multiple templates share trigger phrases.
            n_total = max(len(template.trigger_phrases), 1)
            density_bonus = min((len(matched) - 1) / max(n_total - 1, 1) * 0.3, 0.3)
            confidence = 0.7 + density_bonus

            if confidence > best_confidence:
                best_confidence = confidence
                best_template = name
                best_phrases = matched

        if best_template is None:
            return RoutingDecision(
                route=RouteType.RAG,
                confidence=0.0,
                reason="no_trigger_match",
            )

        if best_confidence >= self._sql_threshold:
            logger.info(
                "router.sql_route",
                extra={
                    "template": best_template,
                    "confidence": best_confidence,
                    "matched_phrases": best_phrases,
                },
            )
            return RoutingDecision(
                route=RouteType.SQL,
                template_name=best_template,
                confidence=best_confidence,
                matched_phrases=best_phrases,
                reason="trigger_match",
            )

        # Below SQL threshold but above review floor
        if self._review_floor > 0 and best_confidence >= self._review_floor:
            return RoutingDecision(
                route=RouteType.REVIEW,
                template_name=best_template,
                confidence=best_confidence,
                matched_phrases=best_phrases,
                reason="below_sql_threshold_above_review_floor",
            )

        return RoutingDecision(
            route=RouteType.RAG,
            confidence=best_confidence,
            matched_phrases=best_phrases,
            reason="below_sql_threshold",
        )
