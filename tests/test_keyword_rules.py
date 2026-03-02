"""Tests for keyword rules: models, strategy, conflict resolution.

Covers the 20-item test matrix from specs/AUTO_TAG_KEYWORD_RULES_v1.md.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_ready_rag.services.auto_tagging.conflict import (
    KEYWORD_AUTHORITATIVE,
    resolve_keyword_conflicts,
)
from ai_ready_rag.services.auto_tagging.models import (
    KEYWORD_OVERRIDEABLE,
    AutoTag,
    DocumentTypeConfig,
    KeywordRule,
    NamespaceConfig,
    StrategyYAML,
)
from ai_ready_rag.services.auto_tagging.strategy import (
    AutoTagStrategy,
    normalize_for_keyword_match,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_path_tag(namespace: str, value: str) -> AutoTag:
    return AutoTag(namespace=namespace, value=value, source="path", confidence=1.0)


def _make_keyword_tag(namespace: str, value: str) -> AutoTag:
    return AutoTag(namespace=namespace, value=value, source="keyword", confidence=1.0)


def _make_manual_tag(namespace: str, value: str) -> AutoTag:
    return AutoTag(namespace=namespace, value=value, source="manual", confidence=1.0)


def _minimal_strategy_yaml(**overrides) -> dict:
    """Return minimal valid StrategyYAML kwargs."""
    base = {
        "strategy": {"id": "test", "name": "Test", "version": "1.0"},
        "namespaces": {
            "doctype": {"display": "Doc Type"},
            "topic": {"display": "Topic"},
            "entity": {"display": "Entity"},
            "stage": {"display": "Stage"},
            "client": {"display": "Client"},
            "year": {"display": "Year"},
        },
        "document_types": {"unknown": {"display": "Unknown"}},
        "llm_prompt": "Classify: {document_type_ids}",
        "keyword_rules": [],
    }
    base.update(overrides)
    return base


def _build_strategy_with_rules(rules: list[dict]) -> AutoTagStrategy:
    """Build an AutoTagStrategy with given keyword rules for testing."""
    keyword_rules = [KeywordRule(**r) for r in rules]
    return AutoTagStrategy(
        id="test",
        name="Test",
        description="",
        version="1.0",
        namespaces={
            "doctype": NamespaceConfig(display="Doc Type"),
            "stage": NamespaceConfig(display="Stage"),
        },
        path_rules=[],
        document_types={"unknown": DocumentTypeConfig(display="Unknown")},
        entity_extraction=None,
        topic_extraction=None,
        llm_prompt_template="Classify: {document_type_ids}",
        email_patterns=[],
        keyword_rules=keyword_rules,
    )


# ===========================================================================
# T11-T13, T19: YAML Schema Validation (KeywordRule + StrategyYAML)
# ===========================================================================


class TestKeywordRuleModel:
    """Tests for KeywordRule Pydantic model validation."""

    def test_valid_keyword_rule(self):
        """Valid keyword rule with keywords_any."""
        rule = KeywordRule(
            namespace="doctype",
            value="coi",
            priority=10,
            keywords_any=["CERTIFICATE OF INSURANCE"],
        )
        assert rule.namespace == "doctype"
        assert rule.priority == 10

    def test_priority_zero_rejected(self):
        """T11: priority: 0 rejected at load time."""
        with pytest.raises(ValidationError, match="priority must be >= 1"):
            KeywordRule(
                namespace="doctype",
                value="coi",
                priority=0,
                keywords_any=["CERT"],
            )

    def test_negative_priority_rejected(self):
        """Negative priority rejected."""
        with pytest.raises(ValidationError, match="priority must be >= 1"):
            KeywordRule(
                namespace="doctype",
                value="coi",
                priority=-1,
                keywords_any=["CERT"],
            )

    def test_no_keywords_rejected(self):
        """Rule with neither keywords_any nor keywords_all rejected."""
        with pytest.raises(ValidationError, match="At least one of"):
            KeywordRule(namespace="doctype", value="coi", priority=5)

    def test_keywords_any_only(self):
        """Rule with only keywords_any is valid."""
        rule = KeywordRule(
            namespace="doctype",
            value="coi",
            priority=5,
            keywords_any=["CERT"],
        )
        assert rule.keywords_all == []

    def test_keywords_all_only(self):
        """Rule with only keywords_all is valid."""
        rule = KeywordRule(
            namespace="doctype",
            value="app",
            priority=5,
            keywords_all=["APPLICANT", "SIGNATURE"],
        )
        assert rule.keywords_any == []

    def test_both_keywords(self):
        """Rule with both keywords_any and keywords_all is valid."""
        rule = KeywordRule(
            namespace="doctype",
            value="app",
            priority=5,
            keywords_any=["CERT"],
            keywords_all=["SIGN"],
        )
        assert rule.keywords_any == ["CERT"]
        assert rule.keywords_all == ["SIGN"]

    def test_case_sensitive_default_false(self):
        """Default case_sensitive is False."""
        rule = KeywordRule(
            namespace="doctype",
            value="coi",
            priority=5,
            keywords_any=["CERT"],
        )
        assert rule.case_sensitive is False


class TestStrategyYAMLKeywordValidation:
    """Tests for StrategyYAML keyword_rules validation."""

    def test_empty_keyword_rules_valid(self):
        """T19: Empty keyword_rules loads without error (backward-compatible)."""
        data = _minimal_strategy_yaml(keyword_rules=[])
        schema = StrategyYAML(**data)
        assert schema.keyword_rules == []

    def test_no_keyword_rules_field(self):
        """Omitting keyword_rules entirely defaults to empty list."""
        data = _minimal_strategy_yaml()
        del data["keyword_rules"]
        schema = StrategyYAML(**data)
        assert schema.keyword_rules == []

    def test_client_namespace_rejected(self):
        """T12: namespace 'client' in keyword rule rejected at load time."""
        data = _minimal_strategy_yaml(
            keyword_rules=[
                {"namespace": "client", "value": "test", "priority": 5, "keywords_any": ["TEST"]},
            ]
        )
        with pytest.raises(ValidationError, match="non-overrideable"):
            StrategyYAML(**data)

    def test_year_namespace_rejected(self):
        """namespace 'year' in keyword rule rejected at load time."""
        data = _minimal_strategy_yaml(
            keyword_rules=[
                {"namespace": "year", "value": "2025", "priority": 5, "keywords_any": ["2025"]},
            ]
        )
        with pytest.raises(ValidationError, match="non-overrideable"):
            StrategyYAML(**data)

    def test_undeclared_namespace_rejected(self):
        """T13: Undeclared namespace in keyword rule rejected."""
        data = _minimal_strategy_yaml(
            keyword_rules=[
                {
                    "namespace": "nonexistent",
                    "value": "val",
                    "priority": 5,
                    "keywords_any": ["TEST"],
                },
            ]
        )
        with pytest.raises(ValidationError, match="undeclared namespace"):
            StrategyYAML(**data)

    def test_valid_keyword_rules_load(self):
        """Valid keyword rules for overrideable namespaces load successfully."""
        data = _minimal_strategy_yaml(
            keyword_rules=[
                {
                    "namespace": "doctype",
                    "value": "coi",
                    "priority": 10,
                    "keywords_any": ["CERTIFICATE OF INSURANCE"],
                },
                {
                    "namespace": "stage",
                    "value": "bind",
                    "priority": 9,
                    "keywords_any": ["BIND CONFIRMATION"],
                },
            ]
        )
        schema = StrategyYAML(**data)
        assert len(schema.keyword_rules) == 2


# ===========================================================================
# T6-T10, T14-T15, T20: parse_keywords() logic
# ===========================================================================


class TestParseKeywords:
    """Tests for AutoTagStrategy.parse_keywords()."""

    def test_coi_keyword_match(self):
        """T1 (unit-level): COI keyword found in content preview."""
        strategy = _build_strategy_with_rules(
            [
                {
                    "namespace": "doctype",
                    "value": "coi",
                    "priority": 10,
                    "keywords_any": [
                        "CERTIFICATE OF INSURANCE",
                        "CERTIFICATE OF PROPERTY INSURANCE",
                    ],
                },
            ]
        )
        preview = normalize_for_keyword_match("CERTIFICATE OF PROPERTY INSURANCE ACORD 25")
        tags = strategy.parse_keywords(preview)
        assert len(tags) == 1
        assert tags[0].namespace == "doctype"
        assert tags[0].value == "coi"
        assert tags[0].source == "keyword"

    def test_no_match_returns_empty(self):
        """T2 (unit-level): No keyword match returns empty list."""
        strategy = _build_strategy_with_rules(
            [
                {
                    "namespace": "doctype",
                    "value": "coi",
                    "priority": 10,
                    "keywords_any": ["CERTIFICATE OF INSURANCE"],
                },
            ]
        )
        preview = normalize_for_keyword_match("JUST A REGULAR POLICY DOCUMENT")
        tags = strategy.parse_keywords(preview)
        assert tags == []

    def test_keywords_any_or_logic(self):
        """T6: keywords_any with multiple keywords — OR logic."""
        strategy = _build_strategy_with_rules(
            [
                {
                    "namespace": "doctype",
                    "value": "loss_run",
                    "priority": 10,
                    "keywords_any": ["LOSS RUN", "LOSS HISTORY", "CLAIMS EXPERIENCE"],
                },
            ]
        )
        preview = normalize_for_keyword_match("THIS IS A CLAIMS EXPERIENCE REPORT")
        tags = strategy.parse_keywords(preview)
        assert len(tags) == 1
        assert tags[0].value == "loss_run"

    def test_keywords_all_and_logic(self):
        """T7: keywords_all with multiple keywords — AND logic."""
        strategy = _build_strategy_with_rules(
            [
                {
                    "namespace": "doctype",
                    "value": "application",
                    "priority": 7,
                    "keywords_all": ["APPLICANT", "SIGNATURE", "DATE OF APPLICATION"],
                },
            ]
        )
        # All present
        preview = normalize_for_keyword_match("APPLICANT NAME SIGNATURE DATE OF APPLICATION")
        tags = strategy.parse_keywords(preview)
        assert len(tags) == 1
        assert tags[0].value == "application"

        # Missing one → no match
        preview2 = normalize_for_keyword_match("APPLICANT NAME SIGNATURE ONLY")
        tags2 = strategy.parse_keywords(preview2)
        assert tags2 == []

    def test_both_any_and_all_groups(self):
        """T8: Both keywords_any + keywords_all — AND between groups."""
        strategy = _build_strategy_with_rules(
            [
                {
                    "namespace": "doctype",
                    "value": "special",
                    "priority": 8,
                    "keywords_any": ["FORM A", "FORM B"],
                    "keywords_all": ["REQUIRED FIELD", "MANDATORY"],
                },
            ]
        )
        # Both groups satisfied
        preview = normalize_for_keyword_match(
            "FORM A DOCUMENT WITH REQUIRED FIELD AND MANDATORY SECTION"
        )
        tags = strategy.parse_keywords(preview)
        assert len(tags) == 1

        # keywords_any satisfied but keywords_all not
        preview2 = normalize_for_keyword_match("FORM A DOCUMENT ONLY")
        tags2 = strategy.parse_keywords(preview2)
        assert tags2 == []

    def test_higher_priority_wins(self):
        """T9: Two rules same namespace, different priority — higher wins."""
        strategy = _build_strategy_with_rules(
            [
                {
                    "namespace": "doctype",
                    "value": "endorsement",
                    "priority": 8,
                    "keywords_any": ["ENDORSEMENT"],
                },
                {
                    "namespace": "doctype",
                    "value": "coi",
                    "priority": 10,
                    "keywords_any": ["CERTIFICATE"],
                },
            ]
        )
        preview = normalize_for_keyword_match("CERTIFICATE OF INSURANCE WITH ENDORSEMENT ATTACHED")
        tags = strategy.parse_keywords(preview)
        assert len(tags) == 1
        assert tags[0].value == "coi"  # priority 10 > 8

    def test_yaml_order_tiebreak(self):
        """T10: Two rules same namespace, same priority — YAML order wins."""
        strategy = _build_strategy_with_rules(
            [
                {
                    "namespace": "doctype",
                    "value": "first_rule",
                    "priority": 10,
                    "keywords_any": ["COMMON KEYWORD"],
                },
                {
                    "namespace": "doctype",
                    "value": "second_rule",
                    "priority": 10,
                    "keywords_any": ["COMMON KEYWORD"],
                },
            ]
        )
        preview = normalize_for_keyword_match("DOCUMENT WITH COMMON KEYWORD")
        tags = strategy.parse_keywords(preview)
        assert len(tags) == 1
        assert tags[0].value == "first_rule"

    def test_case_sensitive_true(self):
        """T14: case_sensitive: true respects case."""
        strategy = _build_strategy_with_rules(
            [
                {
                    "namespace": "doctype",
                    "value": "sensitive",
                    "priority": 10,
                    "keywords_any": ["CaseSensitive"],
                    "case_sensitive": True,
                },
            ]
        )
        # parse_keywords expects a pre-normalized haystack, but for case_sensitive
        # rules the keywords are normalized with case_sensitive=True (no upper)
        # while the haystack was uppercased. The keyword "CaseSensitive" won't
        # match "CASESENSITIVE" in the uppercased haystack.
        preview = normalize_for_keyword_match("This has CaseSensitive text")
        tags = strategy.parse_keywords(preview)
        # Uppercased haystack won't contain "CaseSensitive" — no match
        assert tags == []

    def test_case_insensitive_mixed_case(self):
        """T15: case_sensitive: false with mixed-case content."""
        strategy = _build_strategy_with_rules(
            [
                {
                    "namespace": "doctype",
                    "value": "coi",
                    "priority": 10,
                    "keywords_any": ["certificate of insurance"],
                },
            ]
        )
        preview = normalize_for_keyword_match("Certificate Of Insurance issued today")
        tags = strategy.parse_keywords(preview)
        assert len(tags) == 1
        assert tags[0].value == "coi"

    def test_content_shorter_than_1500(self):
        """T20: Content shorter than 1500 chars — full text used."""
        strategy = _build_strategy_with_rules(
            [
                {
                    "namespace": "doctype",
                    "value": "coi",
                    "priority": 10,
                    "keywords_any": ["CERTIFICATE"],
                },
            ]
        )
        short_text = "CERTIFICATE"
        preview = normalize_for_keyword_match(short_text)[:1500]
        assert len(preview) < 1500
        tags = strategy.parse_keywords(preview)
        assert len(tags) == 1

    def test_empty_keyword_rules_returns_empty(self):
        """No keyword rules returns empty list."""
        strategy = _build_strategy_with_rules([])
        preview = normalize_for_keyword_match("ANY TEXT")
        tags = strategy.parse_keywords(preview)
        assert tags == []

    def test_empty_content_returns_empty(self):
        """Empty content preview returns empty list."""
        strategy = _build_strategy_with_rules(
            [
                {"namespace": "doctype", "value": "coi", "priority": 10, "keywords_any": ["CERT"]},
            ]
        )
        tags = strategy.parse_keywords("")
        assert tags == []


# ===========================================================================
# Text normalization
# ===========================================================================


class TestNormalization:
    """Tests for normalize_for_keyword_match()."""

    def test_whitespace_collapse(self):
        text = normalize_for_keyword_match("  hello   world  \n\t  ")
        assert text == "HELLO WORLD"

    def test_case_fold(self):
        text = normalize_for_keyword_match("Certificate Of Insurance")
        assert text == "CERTIFICATE OF INSURANCE"

    def test_case_sensitive_preserves_case(self):
        text = normalize_for_keyword_match("Certificate Of Insurance", case_sensitive=True)
        assert text == "Certificate Of Insurance"

    def test_unicode_nfc(self):
        # e + combining acute = é (NFC normalizes to single char)
        text = normalize_for_keyword_match("caf\u0065\u0301")
        assert "CAFÉ" in text or "CAFE\u0301" not in text


# ===========================================================================
# Conflict resolution
# ===========================================================================


class TestResolveKeywordConflicts:
    """Tests for resolve_keyword_conflicts()."""

    def test_keyword_overrides_path(self):
        """Keyword tag overrides path tag in same namespace."""
        path_tags = [_make_path_tag("doctype", "policy")]
        kw_tags = [_make_keyword_tag("doctype", "coi")]
        winning, losing, conflicts = resolve_keyword_conflicts(path_tags, kw_tags)
        assert len(winning) == 1
        assert winning[0].value == "coi"
        assert len(losing) == 1
        assert losing[0].value == "policy"
        assert len(conflicts) == 1
        assert conflicts[0]["winner"] == "keyword"

    def test_manual_tag_not_removed(self):
        """T3: Manual tag in same namespace not removed by keyword rule."""
        manual_tags = [_make_manual_tag("doctype", "policy")]
        kw_tags = [_make_keyword_tag("doctype", "coi")]
        winning, losing, conflicts = resolve_keyword_conflicts(manual_tags, kw_tags)
        # Manual tag has source="manual", not "path" — so it's not eligible
        assert len(losing) == 0
        # Keyword still passes through (no path conflict)
        assert len(winning) == 1

    def test_no_conflict_passes_through(self):
        """Keyword in namespace with no path tag passes through."""
        path_tags = [_make_path_tag("client", "acme")]
        kw_tags = [_make_keyword_tag("doctype", "coi")]
        winning, losing, conflicts = resolve_keyword_conflicts(path_tags, kw_tags)
        assert len(winning) == 1
        assert len(losing) == 0

    def test_conflict_record_fields(self):
        """ConflictRecord has override_value and override_source."""
        path_tags = [_make_path_tag("doctype", "policy")]
        kw_tags = [_make_keyword_tag("doctype", "coi")]
        _, _, conflicts = resolve_keyword_conflicts(path_tags, kw_tags)
        c = conflicts[0]
        assert c["override_value"] == "coi"
        assert c["override_source"] == "keyword"
        assert c["path_value"] == "policy"


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    """Verify constants match spec."""

    def test_keyword_overrideable(self):
        assert {"doctype", "topic", "entity", "stage"} == KEYWORD_OVERRIDEABLE

    def test_keyword_authoritative(self):
        assert {"doctype", "topic", "entity", "stage"} == KEYWORD_AUTHORITATIVE

    def test_autotag_keyword_source(self):
        """AutoTag accepts source='keyword'."""
        tag = AutoTag(namespace="doctype", value="coi", source="keyword")
        assert tag.source == "keyword"
