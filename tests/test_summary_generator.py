"""Tests for SummaryGenerator.

Unit tests mock Ollama for CI compatibility.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ai_ready_rag.services.processing_service import ChunkInfo
from ai_ready_rag.services.summary_generator import SummaryGenerator

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def generator():
    """SummaryGenerator instance with test defaults."""
    return SummaryGenerator(
        ollama_url="http://localhost:11434",
        model="test-model",
    )


@pytest.fixture
def sample_chunks():
    """Build a list of ChunkInfo objects for testing."""

    def _make(count: int) -> list[ChunkInfo]:
        return [
            ChunkInfo(
                text=f"Chunk {i} text content about insurance policy details.",
                chunk_index=i,
                page_number=i + 1,
                section=f"Section {i}",
                token_count=10,
            )
            for i in range(count)
        ]

    return _make


VALID_RESPONSE = """DOCUMENT_TYPE: policy
CARRIER: CNA
POLICY_PERIOD: 12/01/2024-12/01/2025
SUMMARY: This is a Directors & Officers liability policy issued by CNA for Cervantes Villas HOA. Coverage includes D&O liability with a $2M aggregate limit."""


# =============================================================================
# Chunk Sampling Tests
# =============================================================================


class TestSampleChunks:
    def test_sample_chunks_small_doc(self, generator, sample_chunks):
        """Documents with <= 9 chunks return all chunks."""
        chunks = sample_chunks(5)
        sampled = generator._sample_chunks(chunks)
        assert len(sampled) == 5
        assert sampled == chunks

    def test_sample_chunks_exact_nine(self, generator, sample_chunks):
        """Document with exactly 9 chunks returns all."""
        chunks = sample_chunks(9)
        sampled = generator._sample_chunks(chunks)
        assert len(sampled) == 9

    def test_sample_chunks_large_doc(self, generator, sample_chunks):
        """Documents with > 9 chunks return exactly 9 (first 3 + middle 3 + last 3)."""
        chunks = sample_chunks(30)
        sampled = generator._sample_chunks(chunks)
        assert len(sampled) == 9

        # First 3
        assert sampled[0].chunk_index == 0
        assert sampled[1].chunk_index == 1
        assert sampled[2].chunk_index == 2

        # Last 3
        assert sampled[6].chunk_index == 27
        assert sampled[7].chunk_index == 28
        assert sampled[8].chunk_index == 29


# =============================================================================
# Prompt Building Tests
# =============================================================================


class TestBuildPrompt:
    def test_build_prompt_contains_filename(self, generator, sample_chunks):
        """Prompt includes the filename."""
        chunks = sample_chunks(3)
        prompt = generator._build_prompt("test_policy.pdf", chunks, 10, 5000)
        assert "test_policy.pdf" in prompt

    def test_build_prompt_contains_chunk_text(self, generator, sample_chunks):
        """Prompt includes sampled chunk text."""
        chunks = sample_chunks(3)
        prompt = generator._build_prompt("test.pdf", chunks, 3, 1000)
        assert "[Chunk 0]:" in prompt
        assert "insurance policy details" in prompt

    def test_build_prompt_contains_stats(self, generator, sample_chunks):
        """Prompt includes chunk count and word count."""
        chunks = sample_chunks(3)
        prompt = generator._build_prompt("test.pdf", chunks, 50, 12345)
        assert "Total chunks: 50" in prompt
        assert "12,345" in prompt


# =============================================================================
# Response Parsing Tests
# =============================================================================


class TestParseResponse:
    def test_parse_response_valid(self, generator):
        """Full structured response parses all fields correctly."""
        parsed = generator._parse_response(VALID_RESPONSE)
        assert parsed["document_type"] == "policy"
        assert parsed["carrier"] == "CNA"
        assert parsed["policy_period"] == "12/01/2024-12/01/2025"
        assert "Directors & Officers" in parsed["text"]

    def test_parse_response_missing_carrier(self, generator):
        """Missing CARRIER falls back to 'unknown'."""
        response = """DOCUMENT_TYPE: quote
POLICY_PERIOD: 2024-2025
SUMMARY: A basic quote document."""
        parsed = generator._parse_response(response)
        assert parsed["document_type"] == "quote"
        assert parsed["carrier"] == "unknown"
        assert parsed["policy_period"] == "2024-2025"

    def test_parse_response_malformed(self, generator):
        """Completely malformed response uses full text as summary."""
        response = "This document is about insurance coverage and limits."
        parsed = generator._parse_response(response)
        assert parsed["text"] == response
        assert parsed["document_type"] == "other"

    def test_parse_response_invalid_document_type(self, generator):
        """Invalid document type falls back to 'other'."""
        response = """DOCUMENT_TYPE: banana
CARRIER: Test Corp
SUMMARY: A test document."""
        parsed = generator._parse_response(response)
        assert parsed["document_type"] == "other"

    def test_parse_response_case_insensitive(self, generator):
        """Field labels are matched case-insensitively."""
        response = """document_type: policy
carrier: Acme Insurance
summary: A policy document."""
        parsed = generator._parse_response(response)
        assert parsed["document_type"] == "policy"
        assert parsed["carrier"] == "Acme Insurance"


# =============================================================================
# Generate Tests (with mocked Ollama)
# =============================================================================


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_success(self, generator, sample_chunks):
        """Successful generation returns ChunkInfo + metadata tuple."""
        chunks = sample_chunks(5)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": VALID_RESPONSE}

        with patch("ai_ready_rag.services.summary_generator.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await generator.generate(chunks, "test.pdf", "doc-123")

        assert result is not None
        summary_chunk, metadata = result
        assert isinstance(summary_chunk, ChunkInfo)
        assert summary_chunk.text.startswith("DOCUMENT SUMMARY:")
        assert summary_chunk.chunk_index == -1
        assert summary_chunk.section == "Document Summary"
        assert summary_chunk.page_number is None
        assert metadata["is_summary"] is True
        assert metadata["document_type"] == "policy"
        assert metadata["carrier"] == "CNA"

    @pytest.mark.asyncio
    async def test_generate_ollama_failure(self, generator, sample_chunks):
        """Ollama failure returns None (graceful fallback)."""
        chunks = sample_chunks(5)

        with patch("ai_ready_rag.services.summary_generator.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await generator.generate(chunks, "test.pdf", "doc-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_empty_chunks(self, generator):
        """Empty chunk list returns None."""
        result = await generator.generate([], "test.pdf", "doc-123")
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_timeout(self, generator, sample_chunks):
        """Ollama timeout returns None (graceful fallback)."""
        chunks = sample_chunks(5)

        with patch("ai_ready_rag.services.summary_generator.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Request timed out")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await generator.generate(chunks, "test.pdf", "doc-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_empty_response(self, generator, sample_chunks):
        """Empty Ollama response returns None."""
        chunks = sample_chunks(5)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": ""}

        with patch("ai_ready_rag.services.summary_generator.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await generator.generate(chunks, "test.pdf", "doc-123")

        assert result is None
