"""Tests for the DocumentClassifier LLM-based classification service."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ai_ready_rag.services.auto_tagging.classifier import (
    ClassificationResult,
    DocumentClassifier,
)
from ai_ready_rag.services.auto_tagging.models import AutoTag


@pytest.fixture
def mock_settings():
    """Settings mock with auto-tagging defaults."""
    settings = MagicMock()
    settings.ollama_base_url = "http://localhost:11434"
    settings.auto_tagging_llm_model = None  # None = use chat_model (new default)
    settings.chat_model = "qwen3:8b"  # Fallback model
    settings.auto_tagging_llm_timeout_seconds = 30
    settings.auto_tagging_llm_max_retries = 1
    settings.auto_tagging_confidence_threshold = 0.7
    settings.auto_tagging_suggestion_threshold = 0.4
    settings.auto_tagging_require_approval = False
    return settings


@pytest.fixture
def classifier(mock_settings):
    """DocumentClassifier with mocked settings."""
    return DocumentClassifier(mock_settings)


def test_classifier_uses_chat_model_when_auto_tagging_model_is_none(mock_settings):
    """classifier.model falls back to chat_model when auto_tagging_llm_model is None."""
    mock_settings.auto_tagging_llm_model = None
    mock_settings.chat_model = "qwen3-rag"
    c = DocumentClassifier(mock_settings)
    assert c.model == "qwen3-rag"


def test_classifier_uses_override_when_auto_tagging_model_is_set(mock_settings):
    """classifier.model uses auto_tagging_llm_model when explicitly set."""
    mock_settings.auto_tagging_llm_model = "llama3.2"
    mock_settings.chat_model = "qwen3-rag"
    c = DocumentClassifier(mock_settings)
    assert c.model == "llama3.2"


@pytest.fixture
def mock_strategy():
    """Mocked AutoTagStrategy."""
    strategy = MagicMock()
    strategy.id = "test_strategy"
    strategy.version = "1.0"
    strategy.build_llm_prompt.return_value = "test prompt"
    strategy.parse_llm_response.return_value = []
    return strategy


def _make_tag(confidence: float = 0.9) -> AutoTag:
    """Helper to create an AutoTag with a given confidence."""
    return AutoTag(
        namespace="doctype",
        value="policy",
        source="llm",
        confidence=confidence,
        strategy_id="test",
        strategy_version="1.0",
    )


def _mock_httpx_response(body: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


class TestJsonParsing:
    """Test _parse_json_response with various LLM output formats."""

    def test_parse_valid_json(self, classifier):
        raw = '{"document_type": "policy", "confidence": 0.9}'
        result = classifier._parse_json_response(raw)
        assert result == {"document_type": "policy", "confidence": 0.9}

    def test_parse_markdown_fenced_json(self, classifier):
        raw = '```json\n{"document_type": "policy"}\n```'
        result = classifier._parse_json_response(raw)
        assert result == {"document_type": "policy"}

    def test_parse_embedded_json(self, classifier):
        raw = 'Here is my analysis:\n{"document_type": "policy"}\nThat is all.'
        result = classifier._parse_json_response(raw)
        assert result == {"document_type": "policy"}

    def test_parse_completely_invalid(self, classifier):
        raw = "This is not JSON at all"
        result = classifier._parse_json_response(raw)
        assert result is None


class TestConfidenceDecisionTable:
    """Test all 6 cells of the confidence decision table."""

    def test_high_confidence_no_approval(self, classifier):
        tag = _make_tag(0.9)
        applied, suggested, discarded = classifier._apply_decision_table([tag])
        assert len(applied) == 1
        assert len(suggested) == 0
        assert len(discarded) == 0

    def test_high_confidence_with_approval(self, mock_settings):
        mock_settings.auto_tagging_require_approval = True
        cls = DocumentClassifier(mock_settings)
        tag = _make_tag(0.9)
        applied, suggested, discarded = cls._apply_decision_table([tag])
        assert len(applied) == 0
        assert len(suggested) == 1
        assert len(discarded) == 0

    def test_mid_confidence_no_approval(self, classifier):
        tag = _make_tag(0.5)
        applied, suggested, discarded = classifier._apply_decision_table([tag])
        assert len(applied) == 1
        assert len(suggested) == 0
        assert len(discarded) == 0

    def test_mid_confidence_with_approval(self, mock_settings):
        mock_settings.auto_tagging_require_approval = True
        cls = DocumentClassifier(mock_settings)
        tag = _make_tag(0.5)
        applied, suggested, discarded = cls._apply_decision_table([tag])
        assert len(applied) == 0
        assert len(suggested) == 1
        assert len(discarded) == 0

    def test_low_confidence_no_approval(self, classifier):
        tag = _make_tag(0.2)
        applied, suggested, discarded = classifier._apply_decision_table([tag])
        assert len(applied) == 0
        assert len(suggested) == 0
        assert len(discarded) == 1

    def test_low_confidence_with_approval(self, mock_settings):
        mock_settings.auto_tagging_require_approval = True
        cls = DocumentClassifier(mock_settings)
        tag = _make_tag(0.2)
        applied, suggested, discarded = cls._apply_decision_table([tag])
        assert len(applied) == 0
        assert len(suggested) == 0
        assert len(discarded) == 1


class TestRetryBehavior:
    """Test LLM call retry logic with mocked httpx."""

    @pytest.mark.asyncio
    async def test_success_first_attempt(self, classifier):
        response = _mock_httpx_response({"response": "hello world"})
        mock_client = AsyncMock()
        mock_client.post.return_value = response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "ai_ready_rag.services.auto_tagging.classifier.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await classifier._call_llm("test prompt")

        assert result == "hello world"
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_success_on_retry(self, classifier):
        response = _mock_httpx_response({"response": "retry success"})
        mock_client = AsyncMock()
        mock_client.post.side_effect = [
            httpx.TimeoutException("timeout"),
            response,
        ]
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "ai_ready_rag.services.auto_tagging.classifier.httpx.AsyncClient",
                return_value=mock_client,
            ),
            patch(
                "ai_ready_rag.services.auto_tagging.classifier.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await classifier._call_llm("test prompt")

        assert result == "retry success"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_failure_after_max_retries(self, classifier):
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("timeout")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "ai_ready_rag.services.auto_tagging.classifier.httpx.AsyncClient",
                return_value=mock_client,
            ),
            patch(
                "ai_ready_rag.services.auto_tagging.classifier.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await classifier._call_llm("test prompt")

        assert result is None
        assert mock_client.post.call_count == 2  # initial + 1 retry


class TestClassifyEndToEnd:
    """Test the full classify() flow with mocked LLM."""

    @pytest.mark.asyncio
    async def test_classify_success(self, classifier, mock_strategy):
        llm_json = json.dumps({"document_type": "policy", "confidence": 0.9})
        response = _mock_httpx_response({"response": llm_json})
        mock_client = AsyncMock()
        mock_client.post.return_value = response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        high_tag = _make_tag(0.9)
        mock_strategy.parse_llm_response.return_value = [high_tag]

        with patch(
            "ai_ready_rag.services.auto_tagging.classifier.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await classifier.classify(
                mock_strategy, "test.pdf", "/uploads/test.pdf", "preview text"
            )

        assert result.status == "completed"
        assert result.error is None
        assert len(result.tags) == 1
        assert len(result.suggested) == 0
        assert len(result.discarded) == 0
        assert result.parsed_response is not None

    @pytest.mark.asyncio
    async def test_classify_llm_failure(self, classifier, mock_strategy):
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("timeout")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "ai_ready_rag.services.auto_tagging.classifier.httpx.AsyncClient",
                return_value=mock_client,
            ),
            patch(
                "ai_ready_rag.services.auto_tagging.classifier.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await classifier.classify(
                mock_strategy, "test.pdf", "/uploads/test.pdf", "preview text"
            )

        assert result.status == "partial"
        assert result.error == "LLM call failed after retries"
        assert len(result.tags) == 0
        assert len(result.suggested) == 0

    @pytest.mark.asyncio
    async def test_classify_invalid_json(self, classifier, mock_strategy):
        response = _mock_httpx_response({"response": "This is not JSON at all"})
        mock_client = AsyncMock()
        mock_client.post.return_value = response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "ai_ready_rag.services.auto_tagging.classifier.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await classifier.classify(
                mock_strategy, "test.pdf", "/uploads/test.pdf", "preview text"
            )

        assert result.status == "partial"
        assert "JSON" in result.error
        assert result.raw_response == "This is not JSON at all"
        assert result.parsed_response is None

    @pytest.mark.asyncio
    async def test_classify_with_suggestions(self, mock_settings, mock_strategy):
        mock_settings.auto_tagging_require_approval = True
        cls = DocumentClassifier(mock_settings)

        llm_json = json.dumps({"document_type": "policy", "confidence": 0.8})
        response = _mock_httpx_response({"response": llm_json})
        mock_client = AsyncMock()
        mock_client.post.return_value = response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        tag = _make_tag(0.8)
        mock_strategy.parse_llm_response.return_value = [tag]

        with patch(
            "ai_ready_rag.services.auto_tagging.classifier.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await cls.classify(
                mock_strategy, "test.pdf", "/uploads/test.pdf", "preview text"
            )

        assert result.status == "completed"
        assert len(result.tags) == 0
        assert len(result.suggested) == 1
        assert len(result.discarded) == 0


class TestClassificationResult:
    """Test ClassificationResult dataclass."""

    def test_result_dataclass_fields(self):
        result = ClassificationResult()
        assert result.tags == []
        assert result.suggested == []
        assert result.discarded == []
        assert result.raw_response == ""
        assert result.parsed_response is None
        assert result.status == "completed"
        assert result.error is None
