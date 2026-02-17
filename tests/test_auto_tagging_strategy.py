"""Comprehensive tests for the auto-tagging strategy engine."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ai_ready_rag.services.auto_tagging import (
    AutoTag,
    AutoTagStrategy,
    PathRule,
    get_transform,
    identity,
    lowercase,
    resolve_entity,
    slugify,
    year_range,
)

STRATEGIES_DIR = Path(__file__).resolve().parent.parent / "data" / "auto_tag_strategies"


@pytest.fixture
def generic_strategy() -> AutoTagStrategy:
    """Load the built-in generic strategy."""
    return AutoTagStrategy.load(str(STRATEGIES_DIR / "generic.yaml"))


@pytest.fixture
def insurance_strategy() -> AutoTagStrategy:
    """Load the built-in insurance agency strategy."""
    return AutoTagStrategy.load(str(STRATEGIES_DIR / "insurance_agency.yaml"))


@pytest.fixture
def law_firm_strategy() -> AutoTagStrategy:
    """Load the built-in law firm strategy."""
    return AutoTagStrategy.load(str(STRATEGIES_DIR / "law_firm.yaml"))


@pytest.fixture
def construction_strategy() -> AutoTagStrategy:
    """Load the built-in construction strategy."""
    return AutoTagStrategy.load(str(STRATEGIES_DIR / "construction.yaml"))


@pytest.fixture
def tmp_strategy_file(tmp_path):
    """Helper that writes a YAML dict to a temp file and returns the path."""

    def _write(data: dict, filename: str = "test_strategy.yaml") -> str:
        filepath = tmp_path / filename
        filepath.write_text(yaml.dump(data, default_flow_style=False))
        return str(filepath)

    return _write


# ============================================================
# Transform Tests
# ============================================================


class TestTransforms:
    def test_slugify_basic(self):
        assert slugify("Bethany Terrace") == "bethany-terrace"

    def test_slugify_special_chars(self):
        assert slugify("D&O Quote") == "do-quote"

    def test_slugify_underscores(self):
        assert slugify("some_name") == "some-name"

    def test_slugify_multiple_hyphens(self):
        assert slugify("a--b") == "a-b"

    def test_slugify_leading_trailing(self):
        assert slugify("-test-") == "test"

    def test_year_range_basic(self):
        assert year_range("24") == "2024-2025"

    def test_year_range_single_digit(self):
        assert year_range("5") == "2005-2006"

    def test_year_range_zero(self):
        assert year_range("0") == "2000-2001"

    def test_year_range_99(self):
        assert year_range("99") == "2099-2100"

    def test_year_range_invalid(self):
        with pytest.raises(ValueError, match="numeric"):
            year_range("abc")

    def test_lowercase(self):
        assert lowercase("CNA") == "cna"

    def test_identity(self):
        assert identity("Quote") == "Quote"

    def test_get_transform_valid(self):
        assert get_transform("slugify") is slugify
        assert get_transform("year_range") is year_range
        assert get_transform("lowercase") is lowercase
        assert get_transform("none") is identity

    def test_get_transform_none(self):
        assert get_transform(None) is identity

    def test_get_transform_unknown(self):
        with pytest.raises(ValueError, match="Unknown transform"):
            get_transform("bogus")


# ============================================================
# Entity Resolution Tests
# ============================================================


class TestResolveEntity:
    ALIASES = {
        "CNA": "cna",
        "Continental Casualty Company": "cna",
        "USLI": "usli",
    }

    def test_exact_match(self):
        assert resolve_entity("CNA", self.ALIASES) == "cna"

    def test_case_insensitive(self):
        assert resolve_entity("cna", self.ALIASES) == "cna"

    def test_full_name_match(self):
        assert resolve_entity("Continental Casualty Company", self.ALIASES) == "cna"

    def test_unknown_entity_slugified(self):
        assert resolve_entity("Some New Carrier", self.ALIASES) == "some-new-carrier"

    def test_empty_aliases(self):
        assert resolve_entity("Some Carrier", {}) == "some-carrier"


# ============================================================
# AutoTag Model Tests
# ============================================================


class TestAutoTag:
    def test_tag_name_property(self):
        tag = AutoTag(namespace="client", value="bethany-terrace", source="path")
        assert tag.tag_name == "client:bethany-terrace"

    def test_display_name_property(self):
        tag = AutoTag(namespace="client", value="bethany-terrace", source="path")
        assert tag.display_name == "Bethany Terrace"

    def test_default_confidence(self):
        tag = AutoTag(namespace="doctype", value="policy", source="llm")
        assert tag.confidence == 1.0

    def test_source_literal(self):
        for source in ("path", "llm", "email", "manual"):
            tag = AutoTag(namespace="test", value="v", source=source)
            assert tag.source == source

    def test_source_invalid(self):
        with pytest.raises(ValueError):
            AutoTag(namespace="test", value="v", source="invalid")


# ============================================================
# PathRule Tests
# ============================================================


class TestPathRule:
    def test_from_dict_minimal(self):
        rule = PathRule.from_dict({"namespace": "client", "level": 0})
        assert rule.namespace == "client"
        assert rule.level == 0
        assert rule.pattern is None
        assert rule.mapping is None

    def test_from_dict_full(self):
        rule = PathRule.from_dict(
            {
                "namespace": "year",
                "level": 1,
                "pattern": r"^(\d{2})\s+(?:NB|Renewal)$",
                "capture_group": 1,
                "transform": "year_range",
                "parent_match": "^Client$",
            }
        )
        assert rule.namespace == "year"
        assert rule.level == 1
        assert rule._compiled_pattern is not None
        assert rule._compiled_parent is not None

    def test_from_dict_invalid_level(self):
        with pytest.raises(ValueError, match="non-negative"):
            PathRule.from_dict({"namespace": "test", "level": -1})


# ============================================================
# Strategy Load Tests
# ============================================================


class TestStrategyLoad:
    def test_load_generic_yaml(self, generic_strategy):
        assert generic_strategy.id == "generic"
        assert generic_strategy.name == "Generic"
        assert "unknown" in generic_strategy.document_types
        assert len(generic_strategy.path_rules) >= 1

    def test_load_insurance_yaml(self, insurance_strategy):
        assert insurance_strategy.id == "insurance_agency"
        assert insurance_strategy.name == "Insurance Agency"
        assert insurance_strategy.entity_extraction is not None
        assert insurance_strategy.topic_extraction is not None
        assert len(insurance_strategy.path_rules) == 4

    def test_load_law_firm_yaml(self, law_firm_strategy):
        assert law_firm_strategy.id == "law_firm"
        assert law_firm_strategy.name == "Law Firm"
        assert law_firm_strategy.entity_extraction is not None
        assert law_firm_strategy.topic_extraction is not None
        assert len(law_firm_strategy.path_rules) == 2
        assert len(law_firm_strategy.document_types) == 10
        assert "unknown" in law_firm_strategy.document_types

    def test_load_construction_yaml(self, construction_strategy):
        assert construction_strategy.id == "construction"
        assert construction_strategy.name == "Construction"
        assert construction_strategy.entity_extraction is not None
        assert construction_strategy.topic_extraction is not None
        assert len(construction_strategy.path_rules) == 2
        assert len(construction_strategy.document_types) == 12
        assert "unknown" in construction_strategy.document_types

    def test_load_validates_id_matches_filename(self, tmp_strategy_file):
        data = _minimal_strategy("wrong_id")
        path = tmp_strategy_file(data, "test_strategy.yaml")
        with pytest.raises(ValueError, match="does not match"):
            AutoTagStrategy.load(path)

    def test_load_rejects_unknown_fields(self, tmp_strategy_file):
        data = _minimal_strategy("test_strategy")
        data["bogus_field"] = "should fail"
        path = tmp_strategy_file(data)
        with pytest.raises(ValueError):
            AutoTagStrategy.load(path)

    def test_load_rejects_missing_required(self, tmp_strategy_file):
        data = {"namespaces": {}, "document_types": {"unknown": {"display": "U"}}}
        path = tmp_strategy_file(data)
        with pytest.raises(ValueError):
            AutoTagStrategy.load(path)

    def test_load_rejects_missing_unknown_doctype(self, tmp_strategy_file):
        data = _minimal_strategy("test_strategy")
        data["document_types"] = {"policy": {"display": "Policy"}}
        path = tmp_strategy_file(data)
        with pytest.raises(Exception, match="unknown"):
            AutoTagStrategy.load(path)

    def test_load_rejects_invalid_namespace_id(self, tmp_strategy_file):
        data = _minimal_strategy("test_strategy")
        data["namespaces"]["Invalid"] = {"display": "Bad"}
        path = tmp_strategy_file(data)
        with pytest.raises(Exception, match="must match"):
            AutoTagStrategy.load(path)

    def test_load_rejects_missing_prompt_placeholder(self, tmp_strategy_file):
        data = _minimal_strategy("test_strategy")
        data["llm_prompt"] = "No placeholder here"
        path = tmp_strategy_file(data)
        with pytest.raises(Exception, match="document_type_ids"):
            AutoTagStrategy.load(path)

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            AutoTagStrategy.load("/nonexistent/path.yaml")


# ============================================================
# parse_path Tests
# ============================================================


class TestParsePath:
    def test_insurance_full_path(self, insurance_strategy):
        tags = insurance_strategy.parse_path("Bethany Terrace (12-13)/24 NB/Quote/CNA/DO_Quote.pdf")
        tag_map = {t.namespace: t.value for t in tags}
        assert tag_map["client"] == "bethany-terrace"
        assert tag_map["year"] == "2024-2025"
        assert tag_map["stage"] == "quote"
        assert tag_map["entity"] == "cna"

    def test_insurance_partial_path(self, insurance_strategy):
        tags = insurance_strategy.parse_path("Bethany Terrace/24 NB/file.pdf")
        namespaces = {t.namespace for t in tags}
        assert "client" in namespaces
        assert "year" in namespaces
        assert "stage" not in namespaces

    def test_mapping_rule(self, insurance_strategy):
        tags = insurance_strategy.parse_path("Client/24 NB/Quote/file.pdf")
        stage_tags = [t for t in tags if t.namespace == "stage"]
        assert len(stage_tags) == 1
        assert stage_tags[0].value == "quote"

    def test_parent_match_filters(self, insurance_strategy):
        tags = insurance_strategy.parse_path("Client/24 NB/Quote/CNA/file.pdf")
        entity_tags = [t for t in tags if t.namespace == "entity"]
        assert len(entity_tags) == 1
        assert entity_tags[0].value == "cna"

    def test_parent_match_skips(self, insurance_strategy):
        tags = insurance_strategy.parse_path("Client/24 NB/Bind/CNA/file.pdf")
        entity_tags = [t for t in tags if t.namespace == "entity"]
        assert len(entity_tags) == 0

    def test_empty_path(self, insurance_strategy):
        assert insurance_strategy.parse_path("") == []

    def test_short_path(self, insurance_strategy):
        tags = insurance_strategy.parse_path("OnlyOneFolder/file.pdf")
        assert len(tags) >= 1
        assert all(t.namespace == "client" for t in tags)

    def test_regex_error_skipped(self, tmp_strategy_file):
        data = _minimal_strategy("test_strategy")
        data["path_rules"] = [{"namespace": "test", "level": 0, "pattern": "[invalid(regex"}]
        path = tmp_strategy_file(data)
        with pytest.raises(ValueError):
            AutoTagStrategy.load(path)

    def test_all_tags_have_source_path(self, insurance_strategy):
        tags = insurance_strategy.parse_path("Bethany Terrace (12-13)/24 NB/Quote/CNA/file.pdf")
        for tag in tags:
            assert tag.source == "path"
            assert tag.confidence == 1.0
            assert tag.strategy_id == "insurance_agency"


# ============================================================
# parse_path Tests — Law Firm
# ============================================================


class TestParsePathLawFirm:
    def test_law_firm_client_extraction(self, law_firm_strategy):
        tags = law_firm_strategy.parse_path("Acme Corp/Discovery/file.pdf")
        tag_map = {t.namespace: t.value for t in tags}
        assert tag_map["client"] == "acme-corp"

    def test_law_firm_stage_mapping(self, law_firm_strategy):
        tags = law_firm_strategy.parse_path("Client/Pleadings/file.pdf")
        stage_tags = [t for t in tags if t.namespace == "stage"]
        assert len(stage_tags) == 1
        assert stage_tags[0].value == "pleadings"

    def test_law_firm_full_path(self, law_firm_strategy):
        tags = law_firm_strategy.parse_path("Acme Corp/Discovery/Depositions/file.pdf")
        tag_map = {t.namespace: t.value for t in tags}
        assert tag_map["client"] == "acme-corp"
        assert tag_map["stage"] == "discovery"

    def test_law_firm_unmapped_stage(self, law_firm_strategy):
        tags = law_firm_strategy.parse_path("Client/UnknownFolder/file.pdf")
        stage_tags = [t for t in tags if t.namespace == "stage"]
        assert len(stage_tags) == 0

    def test_law_firm_all_tags_source(self, law_firm_strategy):
        tags = law_firm_strategy.parse_path("Acme Corp/Discovery/file.pdf")
        for tag in tags:
            assert tag.source == "path"
            assert tag.confidence == 1.0
            assert tag.strategy_id == "law_firm"


# ============================================================
# parse_path Tests — Construction
# ============================================================


class TestParsePathConstruction:
    def test_construction_client_extraction(self, construction_strategy):
        tags = construction_strategy.parse_path("Riverdale Phase 2/Bids/file.pdf")
        tag_map = {t.namespace: t.value for t in tags}
        assert tag_map["client"] == "riverdale"

    def test_construction_client_no_phase(self, construction_strategy):
        tags = construction_strategy.parse_path("Downtown Tower/Contracts/file.pdf")
        tag_map = {t.namespace: t.value for t in tags}
        assert tag_map["client"] == "downtown-tower"

    def test_construction_stage_mapping(self, construction_strategy):
        tags = construction_strategy.parse_path("Project/Bids/file.pdf")
        stage_tags = [t for t in tags if t.namespace == "stage"]
        assert len(stage_tags) == 1
        assert stage_tags[0].value == "bidding"

    def test_construction_full_path(self, construction_strategy):
        tags = construction_strategy.parse_path("Project/RFIs/file.pdf")
        tag_map = {t.namespace: t.value for t in tags}
        assert "client" in tag_map
        assert tag_map["stage"] == "rfi"

    def test_construction_unmapped_stage(self, construction_strategy):
        tags = construction_strategy.parse_path("Project/Random/file.pdf")
        stage_tags = [t for t in tags if t.namespace == "stage"]
        assert len(stage_tags) == 0

    def test_construction_all_tags_source(self, construction_strategy):
        tags = construction_strategy.parse_path("Project/Bids/file.pdf")
        for tag in tags:
            assert tag.source == "path"
            assert tag.confidence == 1.0
            assert tag.strategy_id == "construction"


# ============================================================
# build_llm_prompt Tests
# ============================================================


class TestBuildLlmPrompt:
    def test_basic_placeholders(self, generic_strategy):
        prompt = generic_strategy.build_llm_prompt(
            filename="test.pdf",
            source_path="/docs/test.pdf",
            content_preview="Sample content",
        )
        assert "test.pdf" in prompt
        assert "Sample content" in prompt

    def test_document_type_ids_filled(self, generic_strategy):
        prompt = generic_strategy.build_llm_prompt("f.pdf", "", "c")
        for dt_id in generic_strategy.document_types:
            assert dt_id in prompt

    def test_document_type_rules_filled(self, insurance_strategy):
        prompt = insurance_strategy.build_llm_prompt("f.pdf", "", "c")
        assert "Active insurance policy" in prompt

    def test_entity_extraction_dot_notation(self, insurance_strategy):
        prompt = insurance_strategy.build_llm_prompt("f.pdf", "", "c")
        assert "carrier" in prompt
        assert "{entity_extraction.prompt_label}" not in prompt

    def test_topic_ids_filled(self, insurance_strategy):
        prompt = insurance_strategy.build_llm_prompt("f.pdf", "", "c")
        assert "gl" in prompt
        assert "property" in prompt

    def test_no_entity_extraction(self, generic_strategy):
        prompt = generic_strategy.build_llm_prompt("f.pdf", "", "c")
        assert "{entity_extraction.prompt_label}" not in prompt
        assert "{entity_extraction.prompt_instruction}" not in prompt

    def test_json_braces_preserved(self, generic_strategy):
        prompt = generic_strategy.build_llm_prompt("f.pdf", "", "c")
        assert '"document_type"' in prompt


# ============================================================
# parse_llm_response Tests
# ============================================================


class TestParseLlmResponse:
    def test_full_response(self, insurance_strategy):
        response = {
            "document_type": "policy",
            "entity": "CNA",
            "topics": ["gl", "property"],
            "year_start": "2024",
            "year_end": "2025",
            "confidence": 0.92,
        }
        tags = insurance_strategy.parse_llm_response(response)
        tag_map = {t.namespace: t.value for t in tags}
        assert tag_map["doctype"] == "policy"
        assert tag_map["entity"] == "cna"
        assert tag_map["year"] == "2024-2025"
        topic_tags = [t for t in tags if t.namespace == "topic"]
        assert len(topic_tags) == 2

    def test_unknown_doctype_maps_to_unknown(self, insurance_strategy):
        response = {"document_type": "martian_document", "confidence": 0.5}
        tags = insurance_strategy.parse_llm_response(response)
        doctype_tags = [t for t in tags if t.namespace == "doctype"]
        assert doctype_tags[0].value == "unknown"

    def test_entity_with_alias(self, insurance_strategy):
        response = {
            "document_type": "policy",
            "entity": "Continental Casualty Company",
            "confidence": 0.9,
        }
        tags = insurance_strategy.parse_llm_response(response)
        entity_tags = [t for t in tags if t.namespace == "entity"]
        assert entity_tags[0].value == "cna"

    def test_entity_without_alias(self, insurance_strategy):
        response = {
            "document_type": "policy",
            "entity": "Some New Carrier",
            "confidence": 0.8,
        }
        tags = insurance_strategy.parse_llm_response(response)
        entity_tags = [t for t in tags if t.namespace == "entity"]
        assert entity_tags[0].value == "some-new-carrier"

    def test_topics_filtered(self, insurance_strategy):
        response = {
            "document_type": "policy",
            "topics": ["gl", "unknown_topic", "cyber"],
            "confidence": 0.85,
        }
        tags = insurance_strategy.parse_llm_response(response)
        topic_tags = [t for t in tags if t.namespace == "topic"]
        topic_values = {t.value for t in topic_tags}
        assert "gl" in topic_values
        assert "cyber" in topic_values
        assert "unknown_topic" not in topic_values

    def test_missing_optional_fields(self, insurance_strategy):
        response = {"document_type": "quote", "confidence": 0.7}
        tags = insurance_strategy.parse_llm_response(response)
        assert len(tags) == 1
        assert tags[0].namespace == "doctype"

    def test_year_extraction(self, insurance_strategy):
        response = {
            "document_type": "policy",
            "year_start": "2024",
            "year_end": "2025",
            "confidence": 0.9,
        }
        tags = insurance_strategy.parse_llm_response(response)
        year_tags = [t for t in tags if t.namespace == "year"]
        assert len(year_tags) == 1
        assert year_tags[0].value == "2024-2025"

    def test_confidence_applied(self, insurance_strategy):
        response = {"document_type": "policy", "confidence": 0.75}
        tags = insurance_strategy.parse_llm_response(response)
        for tag in tags:
            assert tag.confidence == 0.75

    def test_default_confidence(self, insurance_strategy):
        response = {"document_type": "policy"}
        tags = insurance_strategy.parse_llm_response(response)
        for tag in tags:
            assert tag.confidence == 0.5


# ============================================================
# parse_email_subject Tests
# ============================================================


class TestParseEmailSubject:
    def test_bind_pattern(self, insurance_strategy):
        tags = insurance_strategy.parse_email_subject("RE: Bind confirmation")
        tag_map = {t.tag_name: t for t in tags}
        assert "stage:bind" in tag_map

    def test_quote_pattern(self, insurance_strategy):
        tags = insurance_strategy.parse_email_subject("Quote from CNA")
        tag_map = {t.tag_name: t for t in tags}
        assert "stage:quote" in tag_map

    def test_no_match(self, insurance_strategy):
        tags = insurance_strategy.parse_email_subject("General update")
        assert tags == []

    def test_multiple_matches(self, insurance_strategy):
        tags = insurance_strategy.parse_email_subject("RE: Loss run and endorsement changes")
        tag_names = {t.tag_name for t in tags}
        assert "doctype:loss-run" in tag_names
        assert "doctype:endorsement" in tag_names

    def test_deduplication(self, insurance_strategy):
        tags = insurance_strategy.parse_email_subject("bind bound confirmation")
        bind_tags = [t for t in tags if t.tag_name == "stage:bind"]
        assert len(bind_tags) == 1

    def test_email_tags_have_source(self, insurance_strategy):
        tags = insurance_strategy.parse_email_subject("RE: Bind confirmation")
        for tag in tags:
            assert tag.source == "email"
            assert tag.confidence == 1.0


# ============================================================
# Helpers
# ============================================================


def _minimal_strategy(strategy_id: str) -> dict:
    """Create a minimal valid strategy dict for testing."""
    return {
        "strategy": {
            "id": strategy_id,
            "name": "Test Strategy",
            "version": "1.0",
        },
        "namespaces": {
            "doctype": {"display": "Document Type"},
        },
        "document_types": {
            "unknown": {"display": "Unknown"},
        },
        "llm_prompt": "Classify: {document_type_ids}",
    }
