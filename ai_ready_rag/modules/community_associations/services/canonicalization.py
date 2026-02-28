"""Canonicalization engine for CA module entities.

Resolves raw extracted entity values to canonical forms:
- Carrier names: "State Farm Fire" → "State Farm Fire and Casualty Company"
- Account names: "Oak Hills HOA" → "Oak Hills Homeowners Association" (normalized)
- Coverage lines: "prop" → "property", "gl" → "general_liability"

Also provides the entity-to-SQL mapper that translates typed entities
into SQL template parameters for the QueryRouter.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Coverage line aliases — maps abbreviations and variants to canonical names
COVERAGE_LINE_ALIASES: dict[str, str] = {
    "prop": "property",
    "property damage": "property",
    "building": "property",
    "structure": "property",
    "gl": "general_liability",
    "general liability": "general_liability",
    "cgl": "general_liability",
    "commercial general liability": "general_liability",
    "d&o": "directors_and_officers",
    "directors and officers": "directors_and_officers",
    "d and o": "directors_and_officers",
    "wc": "workers_compensation",
    "workers comp": "workers_compensation",
    "workers compensation": "workers_compensation",
    "umbrella": "umbrella",
    "excess": "umbrella",
    "excess liability": "umbrella",
    "flood": "flood",
    "nfip": "flood",
    "earthquake": "earthquake",
    "eq": "earthquake",
    "fidelity": "fidelity",
    "crime": "fidelity",
    "employee dishonesty": "fidelity",
}

# Entity type → SQL template parameter mapping
ENTITY_TO_SQL_PARAM: dict[str, str] = {
    "insurance_carrier": "carrier_name",
    "coverage_line": "coverage_line",
    "coverage_limit": "limit_amount",
    "deductible": "deductible_amount",
    "policy_number": "policy_number",
    "effective_date": "effective_date",
    "expiration_date": "expiration_date",
    "account_name": "account_name",
    "association_name": "account_name",
}


@dataclass
class CanonicalEntity:
    entity_type: str
    raw_value: str
    canonical_value: str
    confidence: float
    sql_param: str | None = None  # SQL template parameter name if mappable


@dataclass
class EntityToSQLMapping:
    template_name: str
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    matched_entities: list[CanonicalEntity] = field(default_factory=list)


class CarrierAliasResolver:
    """Resolves carrier name aliases to canonical names.

    Uses in-memory aliases (seeded from ca_carrier_aliases table or direct registration).
    """

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        # alias (lower) → canonical
        self._aliases: dict[str, str] = {}
        if aliases:
            for alias, canonical in aliases.items():
                self._aliases[alias.lower().strip()] = canonical

    def register_aliases(self, aliases: dict[str, str]) -> None:
        for alias, canonical in aliases.items():
            self._aliases[alias.lower().strip()] = canonical

    def resolve(self, carrier_name: str) -> str:
        """Return canonical carrier name, or original if no alias found."""
        key = carrier_name.lower().strip()
        if key in self._aliases:
            return self._aliases[key]
        # Try prefix match (e.g. "State Farm Fire and Casualty" → alias "State Farm Fire")
        for alias, canonical in self._aliases.items():
            if key.startswith(alias) or alias.startswith(key):
                return canonical
        return carrier_name


class CoverageLineCanonicalizer:
    """Normalizes coverage line names to canonical identifiers."""

    def canonicalize(self, raw: str) -> str:
        key = raw.lower().strip()
        return COVERAGE_LINE_ALIASES.get(key, key.replace(" ", "_").replace("-", "_"))


class CanonicalizationEngine:
    """Full canonicalization pipeline for CA entities."""

    def __init__(
        self,
        carrier_resolver: CarrierAliasResolver | None = None,
        coverage_canonicalizer: CoverageLineCanonicalizer | None = None,
    ) -> None:
        self._carrier_resolver = carrier_resolver or CarrierAliasResolver()
        self._coverage_canonicalizer = coverage_canonicalizer or CoverageLineCanonicalizer()

    def canonicalize_entity(self, entity_type: str, raw_value: str) -> CanonicalEntity:
        """Canonicalize a single entity."""
        canonical = raw_value
        confidence = 0.9

        if entity_type == "insurance_carrier":
            canonical = self._carrier_resolver.resolve(raw_value)
            confidence = 1.0 if canonical != raw_value else 0.7

        elif entity_type == "coverage_line":
            canonical = self._coverage_canonicalizer.canonicalize(raw_value)
            confidence = 1.0 if canonical != raw_value.lower().replace(" ", "_") else 0.8

        sql_param = ENTITY_TO_SQL_PARAM.get(entity_type)

        return CanonicalEntity(
            entity_type=entity_type,
            raw_value=raw_value,
            canonical_value=canonical,
            confidence=confidence,
            sql_param=sql_param,
        )

    def canonicalize_all(self, entities: list[dict[str, Any]]) -> list[CanonicalEntity]:
        """Canonicalize a list of entity dicts (from enrichment output)."""
        result = []
        for entity in entities:
            entity_type = entity.get("entity_type", "unknown")
            raw_value = entity.get("value", "")
            if not raw_value:
                continue
            result.append(self.canonicalize_entity(entity_type, raw_value))
        return result


class EntityToSQLMapper:
    """Maps canonicalized entities to SQL template parameters."""

    def __init__(self, sql_templates: dict[str, Any]) -> None:
        self._templates = sql_templates

    def map(
        self,
        entities: list[CanonicalEntity],
        row_cap: int = 100,
    ) -> list[EntityToSQLMapping]:
        """Try to map entities to SQL templates. Returns ranked list of mappings."""
        # Build parameter dict from entities
        params: dict[str, Any] = {"row_cap": row_cap}
        for entity in entities:
            if entity.sql_param:
                params[entity.sql_param] = entity.canonical_value

        mappings = []
        for name, template in self._templates.items():
            # Check which template parameters we can satisfy
            # Simple heuristic: count :param_name patterns in SQL
            required_params = set(
                re.findall(r":(\w+)", template.sql if hasattr(template, "sql") else "")
            )
            required_params.discard("row_cap")  # always provided

            if not required_params:
                continue

            satisfied = required_params.intersection(params.keys())
            if not satisfied:
                continue

            confidence = len(satisfied) / len(required_params)
            mapping_params = {k: params[k] for k in satisfied}
            mapping_params["row_cap"] = row_cap

            mappings.append(
                EntityToSQLMapping(
                    template_name=name,
                    params=mapping_params,
                    confidence=confidence,
                    matched_entities=[e for e in entities if e.sql_param in satisfied],
                )
            )

        return sorted(mappings, key=lambda m: m.confidence, reverse=True)
