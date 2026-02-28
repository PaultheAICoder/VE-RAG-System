"""Deterministic QueryRouter — rule-based SQL-first query routing.

No LLM call is made during routing. Routing decisions are:
1. SQL-first: if entity extraction + trigger phrases + confidence >= threshold -> structured query
2. SQL-first: if quantitative signals + column signals match and confidence >= threshold -> NL2SQL
3. RAG fallback: all other queries -> vector search

Modules contribute SQL templates and trigger phrases via ModuleRegistry.register_sql_templates().
Templates with column_signals set are eligible for quantitative signal scoring (NL2SQL path).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Quantitative signals that indicate a user wants a data-lookup/aggregation answer.
# When a template has column_signals set, both a quantitative signal AND a column signal
# must appear in the query to produce a confidence score >= sql_confidence_threshold.
QUANTITATIVE_SIGNALS = [
    "how many",
    "how much",
    "total",
    "sum",
    "count",
    "average",
    "on hand",
    "balance",
    "amount due",
    "list all",
    "show me all",
    "give me",
    "quantity",
    "in stock",
    "remaining",
    "what is the",
    "show me",
    "outstanding",
    "aged",
]


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

            if matched:
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

            elif template.column_signals is not None:
                # No phrase match — try quantitative signal scoring for NL2SQL templates
                signal_confidence = self._score_quantitative_signals(
                    query_lower, template.column_signals
                )
                if signal_confidence > best_confidence:
                    best_confidence = signal_confidence
                    best_template = name
                    best_phrases = ["<quantitative_signal>"]

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

    def _score_quantitative_signals(
        self, query_lower: str, column_signals: dict[str, list[str]]
    ) -> float:
        """Score a query against column_signals for NL2SQL routing.

        Algorithm:
          - base score 0.5 if any QUANTITATIVE_SIGNALS phrase appears in the query
          - column bonus 0.25 if any column synonym from column_signals appears
          - Returns 0.75 when both hit (above 0.6 threshold → SQL route)
          - Returns 0.5 when only quantitative hit (below 0.6 threshold → RAG)
          - Returns 0.0 when neither hit

        Args:
            query_lower: Lowercased query string.
            column_signals: Dict from SQLTemplate.column_signals.

        Returns:
            Confidence float in range [0.0, 0.75].
        """
        # Check quantitative signals
        quantitative_hit = False
        quant_signals = column_signals.get("__quantitative__", QUANTITATIVE_SIGNALS)
        for signal in quant_signals:
            if re.search(r"\b" + re.escape(signal.lower()) + r"\b", query_lower):
                quantitative_hit = True
                break

        if not quantitative_hit:
            return 0.0

        # Check column name signals (exclude __quantitative__ sentinel key)
        column_hit = False
        for col_key, synonyms in column_signals.items():
            if col_key == "__quantitative__":
                continue
            for syn in synonyms:
                if re.search(r"\b" + re.escape(syn.lower()) + r"\b", query_lower):
                    column_hit = True
                    break
            if column_hit:
                break

        # Score: 0.5 base + 0.25 column bonus = 0.75 total (above 0.6 threshold)
        base = 0.5
        column_bonus = 0.25 if column_hit else 0.0
        return base + column_bonus
