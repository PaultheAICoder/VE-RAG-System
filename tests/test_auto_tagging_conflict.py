"""Tests for auto-tagging conflict resolution, guardrail enforcement, and provenance."""

import pytest

from ai_ready_rag.services.auto_tagging.classifier import ClassificationResult
from ai_ready_rag.services.auto_tagging.conflict import (
    build_provenance,
    enforce_guardrail,
    resolve_conflicts,
)
from ai_ready_rag.services.auto_tagging.models import AutoTag


def _make_path_tag(namespace: str, value: str) -> AutoTag:
    """Create a path-derived AutoTag."""
    return AutoTag(namespace=namespace, value=value, source="path", confidence=1.0)


def _make_llm_tag(namespace: str, value: str, confidence: float = 0.9) -> AutoTag:
    """Create an LLM-derived AutoTag."""
    return AutoTag(
        namespace=namespace,
        value=value,
        source="llm",
        confidence=confidence,
        strategy_id="test",
        strategy_version="1.0",
    )


@pytest.fixture
def path_tags() -> list[AutoTag]:
    """Sample path-derived tags."""
    return [
        _make_path_tag("client", "bethany-terrace"),
        _make_path_tag("year", "2024-2025"),
        _make_path_tag("entity", "cna"),
    ]


@pytest.fixture
def llm_result_no_conflict() -> ClassificationResult:
    """LLM result with tags that don't conflict with path tags."""
    return ClassificationResult(
        tags=[
            _make_llm_tag("doctype", "policy", 0.92),
            _make_llm_tag("topic", "cyber-liability", 0.85),
        ],
        suggested=[],
        discarded=[],
        status="completed",
    )


@pytest.fixture
def llm_result_with_conflicts() -> ClassificationResult:
    """LLM result with tags that conflict in multiple namespaces."""
    return ClassificationResult(
        tags=[
            _make_llm_tag("client", "other-client", 0.8),
            _make_llm_tag("entity", "hartford", 0.9),
            _make_llm_tag("doctype", "quote", 0.75),
            _make_llm_tag("topic", "umbrella", 0.6),
        ],
        suggested=[_make_llm_tag("topic", "property", 0.5)],
        discarded=[_make_llm_tag("doctype", "invoice", 0.2)],
        status="completed",
    )


class TestResolveConflicts:
    """Tests for the resolve_conflicts function."""

    def test_no_conflicts_all_llm_tags_pass_through(self, path_tags, llm_result_no_conflict):
        """LLM tags in namespaces not covered by path tags are all added."""
        winning, losing, conflicts = resolve_conflicts(path_tags, llm_result_no_conflict, 0.7)
        assert len(winning) == 2
        assert len(losing) == 0
        assert len(conflicts) == 0
        assert winning[0].namespace == "doctype"
        assert winning[1].namespace == "topic"

    def test_path_wins_client_namespace(self, path_tags):
        """client: conflict: path tag kept, LLM tag discarded."""
        llm = ClassificationResult(tags=[_make_llm_tag("client", "other-client", 0.95)])
        winning, losing, conflicts = resolve_conflicts(path_tags, llm, 0.7)
        assert len(winning) == 0
        assert len(losing) == 0
        assert len(conflicts) == 1
        assert conflicts[0]["winner"] == "path"
        assert conflicts[0]["namespace"] == "client"

    def test_path_wins_year_namespace(self, path_tags):
        """year: conflict: path tag kept, LLM tag discarded."""
        llm = ClassificationResult(tags=[_make_llm_tag("year", "2023-2024", 0.99)])
        winning, losing, conflicts = resolve_conflicts(path_tags, llm, 0.7)
        assert len(winning) == 0
        assert len(conflicts) == 1
        assert conflicts[0]["winner"] == "path"
        assert conflicts[0]["namespace"] == "year"

    def test_path_wins_stage_namespace(self):
        """stage: conflict: path tag kept, LLM tag discarded."""
        ptags = [_make_path_tag("stage", "renewal")]
        llm = ClassificationResult(tags=[_make_llm_tag("stage", "new-business", 0.95)])
        winning, losing, conflicts = resolve_conflicts(ptags, llm, 0.7)
        assert len(winning) == 0
        assert len(conflicts) == 1
        assert conflicts[0]["winner"] == "path"

    def test_llm_wins_doctype_above_threshold(self):
        """doctype: conflict with LLM confidence >= 0.7: LLM wins."""
        ptags = [_make_path_tag("doctype", "unknown")]
        llm = ClassificationResult(tags=[_make_llm_tag("doctype", "policy", 0.85)])
        winning, losing, conflicts = resolve_conflicts(ptags, llm, 0.7)
        assert len(winning) == 1
        assert winning[0].value == "policy"
        assert len(losing) == 1
        assert losing[0].value == "unknown"
        assert conflicts[0]["winner"] == "llm"

    def test_path_wins_doctype_below_threshold(self):
        """doctype: conflict with LLM confidence < 0.7: path wins."""
        ptags = [_make_path_tag("doctype", "unknown")]
        llm = ClassificationResult(tags=[_make_llm_tag("doctype", "policy", 0.5)])
        winning, losing, conflicts = resolve_conflicts(ptags, llm, 0.7)
        assert len(winning) == 0
        assert len(losing) == 0
        assert conflicts[0]["winner"] == "path"

    def test_llm_wins_entity_above_threshold(self, path_tags):
        """entity: conflict with high confidence: LLM wins."""
        llm = ClassificationResult(tags=[_make_llm_tag("entity", "hartford", 0.9)])
        winning, losing, conflicts = resolve_conflicts(path_tags, llm, 0.7)
        assert len(winning) == 1
        assert winning[0].value == "hartford"
        assert len(losing) == 1
        assert losing[0].value == "cna"
        assert conflicts[0]["winner"] == "llm"

    def test_path_wins_entity_below_threshold(self, path_tags):
        """entity: conflict with low confidence: path wins."""
        llm = ClassificationResult(tags=[_make_llm_tag("entity", "hartford", 0.5)])
        winning, losing, conflicts = resolve_conflicts(path_tags, llm, 0.7)
        assert len(winning) == 0
        assert len(losing) == 0
        assert conflicts[0]["winner"] == "path"

    def test_llm_wins_topic_above_threshold(self):
        """topic: conflict: LLM wins when above threshold."""
        ptags = [_make_path_tag("topic", "general")]
        llm = ClassificationResult(tags=[_make_llm_tag("topic", "cyber-liability", 0.88)])
        winning, losing, conflicts = resolve_conflicts(ptags, llm, 0.7)
        assert len(winning) == 1
        assert winning[0].value == "cyber-liability"
        assert len(losing) == 1
        assert conflicts[0]["winner"] == "llm"

    def test_conflict_records_populated(self, path_tags, llm_result_with_conflicts):
        """Verify conflict records contain correct namespace, values, winner, reason."""
        _, _, conflicts = resolve_conflicts(path_tags, llm_result_with_conflicts, 0.7)
        assert len(conflicts) >= 2
        namespaces = {c["namespace"] for c in conflicts}
        assert "client" in namespaces
        assert "entity" in namespaces
        for c in conflicts:
            assert "namespace" in c
            assert "path_value" in c
            assert "llm_value" in c
            assert "winner" in c
            assert "reason" in c

    def test_unknown_namespace_defaults_to_path(self):
        """Namespace not in authority table defaults to path winning."""
        ptags = [_make_path_tag("custom_ns", "val1")]
        llm = ClassificationResult(tags=[_make_llm_tag("custom_ns", "val2", 0.99)])
        winning, losing, conflicts = resolve_conflicts(ptags, llm, 0.7)
        assert len(winning) == 0
        assert len(losing) == 0
        assert conflicts[0]["winner"] == "path"
        assert "unknown namespace" in conflicts[0]["reason"]

    def test_empty_llm_result(self, path_tags):
        """No LLM tags: returns empty winning list, no conflicts."""
        llm = ClassificationResult(tags=[])
        winning, losing, conflicts = resolve_conflicts(path_tags, llm, 0.7)
        assert len(winning) == 0
        assert len(losing) == 0
        assert len(conflicts) == 0

    def test_empty_path_tags(self, llm_result_no_conflict):
        """No path tags: all LLM tags pass through, no conflicts."""
        winning, losing, conflicts = resolve_conflicts([], llm_result_no_conflict, 0.7)
        assert len(winning) == 2
        assert len(losing) == 0
        assert len(conflicts) == 0


class TestEnforceGuardrail:
    """Tests for the enforce_guardrail function."""

    def test_under_limit_all_kept(self):
        """Total tags below max: all kept, no truncation."""
        path = [_make_path_tag("client", "acme")]
        llm = [_make_llm_tag("doctype", "policy", 0.9)]
        kept_p, kept_l, truncated = enforce_guardrail({"manual:tag"}, path, llm, 20)
        assert len(kept_p) == 1
        assert len(kept_l) == 1
        assert len(truncated) == 0

    def test_over_limit_llm_truncated_first(self):
        """LLM tags truncated before path tags."""
        manual = {"manual:one", "manual:two"}
        path = [_make_path_tag("client", "acme")]
        llm = [
            _make_llm_tag("doctype", "policy", 0.9),
            _make_llm_tag("topic", "cyber", 0.8),
        ]
        # max=4, manual=2, budget=2: path gets 1, llm gets 1 of 2
        kept_p, kept_l, truncated = enforce_guardrail(manual, path, llm, 4)
        assert len(kept_p) == 1
        assert len(kept_l) == 1
        assert len(truncated) == 1
        # The lower confidence LLM tag should be truncated
        assert kept_l[0].confidence == 0.9

    def test_over_limit_path_truncated_after_llm(self):
        """When budget exhausted by path tags, LLM tags are fully truncated."""
        manual = {"m1"}
        path = [
            _make_path_tag("client", "acme"),
            _make_path_tag("year", "2024"),
            _make_path_tag("stage", "renewal"),
        ]
        llm = [_make_llm_tag("doctype", "policy", 0.9)]
        # max=3, manual=1, budget=2: path gets 2, 1 path truncated, llm fully truncated
        kept_p, kept_l, truncated = enforce_guardrail(manual, path, llm, 3)
        assert len(kept_p) == 2
        assert len(kept_l) == 0
        assert len(truncated) == 2  # 1 path + 1 llm

    def test_manual_tags_never_truncated(self):
        """Manual tags always survive (counted but never removed)."""
        manual = {"m1", "m2", "m3"}
        path = [_make_path_tag("client", "acme")]
        llm = [_make_llm_tag("doctype", "policy", 0.9)]
        # max=3, manual=3, budget=0: all auto tags truncated
        kept_p, kept_l, truncated = enforce_guardrail(manual, path, llm, 3)
        assert len(kept_p) == 0
        assert len(kept_l) == 0
        assert len(truncated) == 2

    def test_llm_sorted_by_confidence_desc(self):
        """Higher confidence LLM tags kept over lower."""
        llm = [
            _make_llm_tag("topic", "low", 0.5),
            _make_llm_tag("doctype", "high", 0.95),
            _make_llm_tag("entity", "mid", 0.75),
        ]
        # max=2, no manual, no path: budget=2, keep top 2 LLM by confidence
        kept_p, kept_l, truncated = enforce_guardrail(set(), [], llm, 2)
        assert len(kept_l) == 2
        assert kept_l[0].confidence == 0.95
        assert kept_l[1].confidence == 0.75
        assert len(truncated) == 1
        assert "topic:low" in truncated

    def test_exact_limit(self):
        """Exactly at max: no truncation."""
        manual = {"m1"}
        path = [_make_path_tag("client", "acme")]
        llm = [_make_llm_tag("doctype", "policy", 0.9)]
        # max=3, manual=1, budget=2: fits exactly
        kept_p, kept_l, truncated = enforce_guardrail(manual, path, llm, 3)
        assert len(kept_p) == 1
        assert len(kept_l) == 1
        assert len(truncated) == 0

    def test_truncated_names_returned(self):
        """Truncated tag names appear in return value."""
        llm = [
            _make_llm_tag("doctype", "policy", 0.9),
            _make_llm_tag("topic", "cyber", 0.3),
        ]
        kept_p, kept_l, truncated = enforce_guardrail(set(), [], llm, 1)
        assert len(truncated) == 1
        assert "topic:cyber" in truncated


class TestBuildProvenance:
    """Tests for the build_provenance function."""

    def test_provenance_schema_matches_spec(self):
        """Output dict has all required keys from Section 8.5."""
        result = build_provenance(
            strategy_id="generic",
            strategy_version="1.0",
            path_tags=[_make_path_tag("client", "acme")],
            llm_result=ClassificationResult(tags=[_make_llm_tag("doctype", "policy", 0.9)]),
            conflicts=[],
            applied_tag_names=["client:acme", "doctype:policy"],
            discarded_tag_names=[],
            suggested_tag_names=[],
        )
        required_keys = {
            "strategy_id",
            "strategy_version",
            "path_candidates",
            "llm_candidates",
            "conflicts",
            "applied",
            "discarded",
            "suggested",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_path_candidates_populated(self):
        """Path tags appear in path_candidates."""
        ptags = [_make_path_tag("client", "acme"), _make_path_tag("year", "2024")]
        result = build_provenance(
            strategy_id="generic",
            strategy_version="1.0",
            path_tags=ptags,
            llm_result=None,
            conflicts=[],
            applied_tag_names=[],
            discarded_tag_names=[],
            suggested_tag_names=[],
        )
        assert len(result["path_candidates"]) == 2
        assert result["path_candidates"][0]["namespace"] == "client"
        assert result["path_candidates"][0]["confidence"] == 1.0

    def test_llm_candidates_include_all(self):
        """Applied + suggested + discarded LLM tags all appear in llm_candidates."""
        llm_result = ClassificationResult(
            tags=[_make_llm_tag("doctype", "policy", 0.9)],
            suggested=[_make_llm_tag("topic", "cyber", 0.55)],
            discarded=[_make_llm_tag("entity", "old", 0.2)],
        )
        result = build_provenance(
            strategy_id="generic",
            strategy_version="1.0",
            path_tags=[],
            llm_result=llm_result,
            conflicts=[],
            applied_tag_names=[],
            discarded_tag_names=[],
            suggested_tag_names=[],
        )
        assert len(result["llm_candidates"]) == 3
        namespaces = {c["namespace"] for c in result["llm_candidates"]}
        assert namespaces == {"doctype", "topic", "entity"}

    def test_conflicts_recorded(self):
        """Conflict records appear in conflicts list."""
        conflict = {
            "namespace": "entity",
            "path_value": "cna",
            "llm_value": "hartford",
            "winner": "llm",
            "reason": "entity: LLM confidence 0.9 >= threshold 0.7",
        }
        result = build_provenance(
            strategy_id="generic",
            strategy_version="1.0",
            path_tags=[],
            llm_result=None,
            conflicts=[conflict],
            applied_tag_names=[],
            discarded_tag_names=[],
            suggested_tag_names=[],
        )
        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0]["winner"] == "llm"

    def test_truncated_key_added_when_present(self):
        """truncated key present only when tags were truncated."""
        # Without truncated
        result_no = build_provenance(
            strategy_id="generic",
            strategy_version="1.0",
            path_tags=[],
            llm_result=None,
            conflicts=[],
            applied_tag_names=[],
            discarded_tag_names=[],
            suggested_tag_names=[],
        )
        assert "truncated" not in result_no

        # With truncated
        result_yes = build_provenance(
            strategy_id="generic",
            strategy_version="1.0",
            path_tags=[],
            llm_result=None,
            conflicts=[],
            applied_tag_names=[],
            discarded_tag_names=[],
            suggested_tag_names=[],
            truncated_tag_names=["topic:overflow"],
        )
        assert "truncated" in result_yes
        assert result_yes["truncated"] == ["topic:overflow"]

    def test_provenance_with_no_llm_result(self):
        """When llm_result=None, llm_candidates is empty."""
        result = build_provenance(
            strategy_id="generic",
            strategy_version="1.0",
            path_tags=[_make_path_tag("client", "acme")],
            llm_result=None,
            conflicts=[],
            applied_tag_names=["client:acme"],
            discarded_tag_names=[],
            suggested_tag_names=[],
        )
        assert result["llm_candidates"] == []
        assert len(result["path_candidates"]) == 1
