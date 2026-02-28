"""Tests for CA canonicalization engine."""

from ai_ready_rag.modules.community_associations.services.canonicalization import (
    CanonicalizationEngine,
    CarrierAliasResolver,
    CoverageLineCanonicalizer,
)


class TestCarrierAliasResolver:
    def test_exact_alias(self):
        resolver = CarrierAliasResolver({"state farm": "State Farm Fire and Casualty Company"})
        assert resolver.resolve("state farm") == "State Farm Fire and Casualty Company"

    def test_case_insensitive(self):
        resolver = CarrierAliasResolver({"State Farm": "State Farm Fire and Casualty Company"})
        assert resolver.resolve("STATE FARM") == "State Farm Fire and Casualty Company"

    def test_no_match_returns_original(self):
        resolver = CarrierAliasResolver({})
        assert resolver.resolve("Unknown Carrier") == "Unknown Carrier"

    def test_register_aliases(self):
        resolver = CarrierAliasResolver()
        resolver.register_aliases({"travelers": "The Travelers Indemnity Company"})
        assert resolver.resolve("travelers") == "The Travelers Indemnity Company"


class TestCoverageLineCanonicalizer:
    def test_gl_to_general_liability(self):
        c = CoverageLineCanonicalizer()
        assert c.canonicalize("gl") == "general_liability"

    def test_prop_to_property(self):
        c = CoverageLineCanonicalizer()
        assert c.canonicalize("prop") == "property"

    def test_unknown_normalizes(self):
        c = CoverageLineCanonicalizer()
        result = c.canonicalize("Unknown Coverage")
        assert " " not in result  # spaces converted to underscores


class TestCanonicalizationEngine:
    def test_carrier_entity(self):
        engine = CanonicalizationEngine(
            carrier_resolver=CarrierAliasResolver({"state farm": "State Farm Fire and Casualty"})
        )
        result = engine.canonicalize_entity("insurance_carrier", "state farm")
        assert result.canonical_value == "State Farm Fire and Casualty"
        assert result.sql_param == "carrier_name"

    def test_coverage_line_entity(self):
        engine = CanonicalizationEngine()
        result = engine.canonicalize_entity("coverage_line", "gl")
        assert result.canonical_value == "general_liability"

    def test_canonicalize_all(self):
        engine = CanonicalizationEngine()
        entities = [
            {"entity_type": "coverage_line", "value": "gl"},
            {"entity_type": "insurance_carrier", "value": "State Farm"},
        ]
        results = engine.canonicalize_all(entities)
        assert len(results) == 2

    def test_empty_value_skipped(self):
        engine = CanonicalizationEngine()
        results = engine.canonicalize_all([{"entity_type": "coverage_line", "value": ""}])
        assert len(results) == 0
