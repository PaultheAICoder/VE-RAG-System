"""Tests for vector service utilities."""

import pytest

from ai_ready_rag.core.exceptions import (
    EmbeddingError,
    IndexingError,
    SearchError,
    VectorServiceError,
)
from ai_ready_rag.services.vector_utils import (
    RESERVED_TAGS,
    generate_chunk_id,
    validate_tag,
    validate_tags_for_ingestion,
)


class TestExceptions:
    """Test exception hierarchy."""

    def test_embedding_error_is_vector_service_error(self):
        assert issubclass(EmbeddingError, VectorServiceError)

    def test_indexing_error_is_vector_service_error(self):
        assert issubclass(IndexingError, VectorServiceError)

    def test_search_error_is_vector_service_error(self):
        assert issubclass(SearchError, VectorServiceError)

    def test_can_catch_all_with_base_class(self):
        """All exceptions can be caught with VectorServiceError."""
        for exc_class in [EmbeddingError, IndexingError, SearchError]:
            try:
                raise exc_class("test")
            except VectorServiceError:
                pass  # Expected

    def test_can_catch_individually(self):
        """Each exception can be caught individually."""
        try:
            raise EmbeddingError("test")
        except EmbeddingError:
            pass

        try:
            raise IndexingError("test")
        except IndexingError:
            pass

        try:
            raise SearchError("test")
        except SearchError:
            pass


class TestChunkIdGeneration:
    """Test chunk ID generation."""

    def test_deterministic_same_inputs(self):
        """Same inputs always produce same output."""
        id1 = generate_chunk_id("doc-123", 0)
        id2 = generate_chunk_id("doc-123", 0)
        assert id1 == id2

    def test_different_chunk_index_different_id(self):
        """Different chunk indices produce different IDs."""
        id1 = generate_chunk_id("doc-123", 0)
        id2 = generate_chunk_id("doc-123", 1)
        assert id1 != id2

    def test_different_document_different_id(self):
        """Different documents produce different IDs."""
        id1 = generate_chunk_id("doc-123", 0)
        id2 = generate_chunk_id("doc-456", 0)
        assert id1 != id2

    def test_returns_valid_uuid_string(self):
        """Returns a valid UUID string format."""
        chunk_id = generate_chunk_id("doc-123", 0)
        # UUID format: 8-4-4-4-12 hex chars
        assert len(chunk_id) == 36
        assert chunk_id.count("-") == 4


class TestValidateTag:
    """Test single tag validation."""

    def test_valid_simple_tag(self):
        assert validate_tag("hr") == "hr"

    def test_normalizes_to_lowercase(self):
        assert validate_tag("HR") == "hr"
        assert validate_tag("Human-Resources") == "human-resources"

    def test_strips_whitespace(self):
        assert validate_tag("  hr  ") == "hr"

    def test_valid_with_hyphen(self):
        assert validate_tag("human-resources") == "human-resources"

    def test_valid_single_char(self):
        assert validate_tag("a") == "a"
        assert validate_tag("1") == "1"

    def test_valid_with_numbers(self):
        assert validate_tag("team1") == "team1"
        assert validate_tag("2024-budget") == "2024-budget"

    def test_invalid_empty_string(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_tag("")

    def test_invalid_whitespace_only(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_tag("   ")

    def test_invalid_consecutive_hyphens(self):
        with pytest.raises(ValueError, match="consecutive hyphens"):
            validate_tag("hr--dept")

    def test_invalid_starts_with_hyphen(self):
        with pytest.raises(ValueError, match="starting and ending with alphanumeric"):
            validate_tag("-hr")

    def test_invalid_ends_with_hyphen(self):
        with pytest.raises(ValueError, match="starting and ending with alphanumeric"):
            validate_tag("hr-")

    def test_invalid_underscore(self):
        with pytest.raises(ValueError, match="alphanumeric with hyphens"):
            validate_tag("hr_dept")

    def test_invalid_special_chars(self):
        with pytest.raises(ValueError):
            validate_tag("hr@dept")

    def test_invalid_too_long(self):
        long_tag = "a" * 65
        with pytest.raises(ValueError, match="exceeds 64 characters"):
            validate_tag(long_tag)

    def test_valid_max_length(self):
        max_tag = "a" * 64
        assert validate_tag(max_tag) == max_tag


class TestValidateTagsForIngestion:
    """Test tag list validation for document ingestion."""

    def test_valid_single_tag(self):
        result = validate_tags_for_ingestion(["hr"])
        assert result == ["hr"]

    def test_valid_multiple_tags(self):
        result = validate_tags_for_ingestion(["hr", "finance"])
        assert result == ["hr", "finance"]

    def test_normalizes_all_tags(self):
        result = validate_tags_for_ingestion(["HR", "Finance"])
        assert result == ["hr", "finance"]

    def test_deduplicates_tags(self):
        result = validate_tags_for_ingestion(["hr", "HR", "hr"])
        assert result == ["hr"]

    def test_invalid_empty_list(self):
        with pytest.raises(ValueError, match="At least one tag is required"):
            validate_tags_for_ingestion([])

    def test_reserved_tag_public_as_non_admin(self):
        with pytest.raises(ValueError, match="reserved"):
            validate_tags_for_ingestion(["public"], is_admin=False)

    def test_reserved_tag_system_as_non_admin(self):
        with pytest.raises(ValueError, match="reserved"):
            validate_tags_for_ingestion(["system"], is_admin=False)

    def test_reserved_tag_public_as_admin(self):
        result = validate_tags_for_ingestion(["public"], is_admin=True)
        assert result == ["public"]

    def test_reserved_tag_system_as_admin(self):
        result = validate_tags_for_ingestion(["system"], is_admin=True)
        assert result == ["system"]

    def test_mixed_reserved_and_regular_as_admin(self):
        result = validate_tags_for_ingestion(["hr", "public", "finance"], is_admin=True)
        assert result == ["hr", "public", "finance"]

    def test_preserves_order(self):
        result = validate_tags_for_ingestion(["zebra", "alpha", "middle"])
        assert result == ["zebra", "alpha", "middle"]

    def test_reserved_tags_constant(self):
        """Verify reserved tags are as expected."""
        assert "public" in RESERVED_TAGS
        assert "system" in RESERVED_TAGS
        assert len(RESERVED_TAGS) == 2
