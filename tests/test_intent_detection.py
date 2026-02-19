"""Tests for query-aware intent detection and intent score boosting."""

import pytest

from ai_ready_rag.services.rag_service import QueryIntent, detect_query_intent
from ai_ready_rag.services.vector_service import SearchResult

# ---------------------------------------------------------------------------
# TestDetectQueryIntent
# ---------------------------------------------------------------------------


class TestDetectQueryIntent:
    """Unit tests for detect_query_intent()."""

    def test_detect_active_policy(self):
        intent = detect_query_intent("current policy limits for Bethany Terrace")
        assert "stage:policy" in intent.preferred_tags
        assert "doctype:policy" in intent.preferred_tags
        assert intent.intent_label == "active_policy"
        assert intent.confidence > 0

    def test_detect_policy_detail(self):
        intent = detect_query_intent("policy deductible for the property coverage")
        assert "stage:policy" in intent.preferred_tags
        assert "doctype:policy" in intent.preferred_tags

    def test_detect_quote(self):
        intent = detect_query_intent("compare the quoted premiums")
        assert "stage:quote" in intent.preferred_tags
        assert "doctype:quote" in intent.preferred_tags
        assert intent.intent_label == "quote"

    def test_detect_submission(self):
        intent = detect_query_intent("what was submitted in the application")
        assert "stage:submission" in intent.preferred_tags
        assert intent.intent_label == "submission"

    def test_detect_bind(self):
        intent = detect_query_intent("when was the binder issued")
        assert "stage:bind" in intent.preferred_tags
        assert intent.intent_label == "bind"

    def test_detect_topic_earthquake(self):
        intent = detect_query_intent("earthquake coverage details")
        assert "topic:earthquake" in intent.preferred_tags

    def test_detect_loss_run(self):
        intent = detect_query_intent("show the claims history for this account")
        assert "doctype:loss_run" in intent.preferred_tags
        assert intent.intent_label == "loss_run"

    def test_detect_combined(self):
        """Multiple patterns should accumulate tags."""
        intent = detect_query_intent("current earthquake policy limits")
        assert "stage:policy" in intent.preferred_tags
        assert "doctype:policy" in intent.preferred_tags
        assert "topic:earthquake" in intent.preferred_tags
        # Should have at least 3 tags
        assert len(intent.preferred_tags) >= 3

    def test_detect_no_intent(self):
        intent = detect_query_intent("hello how are you")
        assert intent.preferred_tags == []
        assert intent.intent_label is None
        assert intent.confidence == 0.0

    def test_detect_case_insensitive(self):
        intent = detect_query_intent("CURRENT POLICY LIMITS")
        assert "stage:policy" in intent.preferred_tags

    def test_detect_do(self):
        intent = detect_query_intent("D&O coverage for the board")
        assert "topic:do" in intent.preferred_tags
        assert intent.intent_label == "do"

    def test_detect_workers_comp(self):
        intent = detect_query_intent("workers comp claims")
        assert "topic:wc" in intent.preferred_tags

    def test_detect_certificate(self):
        intent = detect_query_intent("send the COI to the client")
        assert "doctype:certificate" in intent.preferred_tags

    def test_detect_endorsement(self):
        intent = detect_query_intent("is there an endorsement for flood")
        assert "doctype:endorsement" in intent.preferred_tags

    def test_detect_umbrella(self):
        intent = detect_query_intent("umbrella policy aggregate limit")
        assert "topic:umbrella" in intent.preferred_tags

    def test_detect_epli(self):
        intent = detect_query_intent("employment practices liability coverage")
        assert "topic:epli" in intent.preferred_tags

    def test_detect_crime(self):
        intent = detect_query_intent("employee dishonesty coverage limit")
        assert "topic:crime" in intent.preferred_tags

    def test_detect_renewal(self):
        intent = detect_query_intent("renewal options for next year")
        assert "stage:quote" in intent.preferred_tags
        assert intent.intent_label == "renewal"

    def test_detect_dec_page(self):
        intent = detect_query_intent("pull the dec page for this account")
        assert "doctype:coverage_summary" in intent.preferred_tags

    def test_no_duplicate_tags(self):
        """Tags should be deduplicated even if multiple patterns add the same tag."""
        intent = detect_query_intent("active bound policy limits")
        tag_counts = {}
        for tag in intent.preferred_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        for tag, count in tag_counts.items():
            assert count == 1, f"Tag '{tag}' appears {count} times"

    def test_returns_query_intent_dataclass(self):
        intent = detect_query_intent("anything")
        assert isinstance(intent, QueryIntent)
        assert isinstance(intent.preferred_tags, list)
        assert isinstance(intent.confidence, float)


# ---------------------------------------------------------------------------
# Helper: build SearchResult for testing
# ---------------------------------------------------------------------------


def _make_chunk(
    score: float,
    tags: list[str] | None = None,
    doc_id: str = "doc1",
    chunk_index: int = 0,
) -> SearchResult:
    return SearchResult(
        chunk_id=f"{doc_id}:{chunk_index}",
        document_id=doc_id,
        document_name="test.pdf",
        chunk_text="Sample chunk text for testing.",
        chunk_index=chunk_index,
        score=score,
        page_number=1,
        section=None,
        tags=tags,
    )


# ---------------------------------------------------------------------------
# TestApplyIntentBoost
# ---------------------------------------------------------------------------


class TestApplyIntentBoost:
    """Unit tests for RAGService._apply_intent_boost()."""

    def _make_rag_service(self, intent_weight: float = 0.20):
        """Create a minimal RAGService for testing."""
        from unittest.mock import MagicMock

        from ai_ready_rag.config import Settings

        settings = MagicMock(spec=Settings)
        settings.rag_intent_boost_weight = intent_weight
        settings.rag_recency_weight = 0.0
        settings.ollama_base_url = "http://localhost:11434"
        settings.chat_model = "test"
        settings.rag_max_chunks_per_doc = 5
        settings.rag_chunk_overlap_threshold = 0.9
        settings.rag_dedup_candidates_cap = 15
        settings.rag_enable_hallucination_check = False

        from ai_ready_rag.services.rag_service import RAGService

        # Patch get_rag_setting to return the weight directly
        service = RAGService(settings=settings)
        return service

    def test_boost_matching_tags(self):
        service = self._make_rag_service(intent_weight=0.20)
        intent = QueryIntent(
            preferred_tags=["stage:policy", "doctype:policy"],
            intent_label="active_policy",
            confidence=0.5,
        )

        chunks = [
            _make_chunk(0.8, tags=["stage:policy", "doctype:policy"], chunk_index=0),
            _make_chunk(0.85, tags=["stage:quote", "doctype:quote"], chunk_index=1),
            _make_chunk(0.7, tags=["stage:policy"], chunk_index=2),
        ]

        result = service._apply_intent_boost(chunks, intent)

        # Chunk 0 has full match (2/2 tags) -> 0.8*0.8 + 0.2*1.0 = 0.84
        # Chunk 1 has no match (0/2 tags) -> 0.85*0.8 + 0.2*0.0 = 0.68
        # Chunk 2 has partial match (1/2 tags) -> 0.7*0.8 + 0.2*0.5 = 0.66
        assert result[0].score == pytest.approx(0.84, abs=0.01)
        assert result[1].score == pytest.approx(0.68, abs=0.01)
        assert result[2].score == pytest.approx(0.66, abs=0.01)

        # First result should be the policy-tagged chunk
        assert result[0].tags == ["stage:policy", "doctype:policy"]

    def test_no_effect_when_disabled(self):
        service = self._make_rag_service(intent_weight=0.0)
        intent = QueryIntent(
            preferred_tags=["stage:policy"],
            intent_label="active_policy",
            confidence=0.5,
        )

        chunks = [
            _make_chunk(0.8, tags=["stage:policy"], chunk_index=0),
            _make_chunk(0.9, tags=["stage:quote"], chunk_index=1),
        ]

        # Weight=0 -> (1-0)*score + 0*overlap = score unchanged
        result = service._apply_intent_boost(chunks, intent)
        assert result[0].score == pytest.approx(0.9)
        assert result[1].score == pytest.approx(0.8)

    def test_no_effect_empty_intent(self):
        service = self._make_rag_service(intent_weight=0.20)
        intent = QueryIntent(
            preferred_tags=[],
            intent_label=None,
            confidence=0.0,
        )

        chunks = [
            _make_chunk(0.8, tags=["stage:policy"], chunk_index=0),
            _make_chunk(0.9, tags=["stage:quote"], chunk_index=1),
        ]

        result = service._apply_intent_boost(chunks, intent)
        # Empty preferred_tags -> early return, scores unchanged
        assert result[0].score == pytest.approx(0.8)
        assert result[1].score == pytest.approx(0.9)

    def test_partial_match(self):
        service = self._make_rag_service(intent_weight=0.20)
        intent = QueryIntent(
            preferred_tags=["stage:policy", "topic:earthquake"],
            intent_label="active_policy",
            confidence=0.5,
        )

        chunk = _make_chunk(0.8, tags=["stage:policy", "topic:property"], chunk_index=0)
        result = service._apply_intent_boost([chunk], intent)

        # 1 of 2 preferred tags match -> overlap = 0.5
        # score = 0.8*0.8 + 0.2*0.5 = 0.74
        assert result[0].score == pytest.approx(0.74, abs=0.01)

    def test_chunks_with_none_tags(self):
        service = self._make_rag_service(intent_weight=0.20)
        intent = QueryIntent(
            preferred_tags=["stage:policy"],
            intent_label="active_policy",
            confidence=0.5,
        )

        chunk = _make_chunk(0.8, tags=None, chunk_index=0)
        result = service._apply_intent_boost([chunk], intent)

        # None tags -> overlap = 0
        # score = 0.8*0.8 + 0.2*0 = 0.64
        assert result[0].score == pytest.approx(0.64, abs=0.01)

    def test_sort_order_after_boost(self):
        """After boosting, chunks should be sorted by score descending."""
        service = self._make_rag_service(intent_weight=0.40)
        intent = QueryIntent(
            preferred_tags=["stage:policy"],
            intent_label="active_policy",
            confidence=0.5,
        )

        chunks = [
            _make_chunk(0.6, tags=["stage:policy"], chunk_index=0),  # 0.6*0.6 + 0.4*1.0 = 0.76
            _make_chunk(0.9, tags=["stage:quote"], chunk_index=1),  # 0.9*0.6 + 0.4*0.0 = 0.54
        ]

        result = service._apply_intent_boost(chunks, intent)
        assert result[0].score > result[1].score
        assert result[0].tags == ["stage:policy"]
