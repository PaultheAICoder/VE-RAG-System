"""Utility functions for the Vector Service.

Provides chunk ID generation and tag validation for document indexing.
"""

import re
import uuid

# Chunk ID namespace (UUID namespace for deterministic ID generation)
CHUNK_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# Tag validation constants
TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9\-]{0,62}[a-z0-9]$|^[a-z0-9]$")
RESERVED_TAGS = frozenset({"public", "system"})
MAX_TAG_LENGTH = 64


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate a deterministic chunk ID using UUIDv5.

    Same document_id + chunk_index always produces the same chunk_id.
    This enables idempotent upserts and reliable citations.

    Args:
        document_id: Parent document's unique identifier.
        chunk_index: Zero-based position of chunk within document.

    Returns:
        A deterministic UUID string for the chunk.

    Example:
        >>> generate_chunk_id("doc-123", 0)
        'a1b2c3d4-...'  # Always same output for same inputs
    """
    name = f"{document_id}:{chunk_index}"
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, name))


def validate_tag(tag: str) -> str:
    """Validate and normalize a single tag.

    Rules:
        - Lowercase only (normalized automatically)
        - Alphanumeric and hyphens only
        - Must start and end with alphanumeric
        - 1-64 characters
        - No consecutive hyphens

    Args:
        tag: The tag string to validate.

    Returns:
        Normalized tag (lowercase, stripped).

    Raises:
        ValueError: If tag is invalid with descriptive message.
    """
    normalized = tag.lower().strip()

    if not normalized:
        raise ValueError("Tag cannot be empty")

    if len(normalized) > MAX_TAG_LENGTH:
        raise ValueError(f"Tag exceeds {MAX_TAG_LENGTH} characters")

    if "--" in normalized:
        raise ValueError("Tag cannot contain consecutive hyphens")

    if not TAG_PATTERN.match(normalized):
        raise ValueError(
            "Tag must be alphanumeric with hyphens, starting and ending with alphanumeric"
        )

    return normalized


def validate_tags_for_ingestion(tags: list[str], is_admin: bool = False) -> list[str]:
    """Validate tags for document ingestion.

    Validates each tag and checks reserved tag permissions.

    Args:
        tags: List of tags to validate.
        is_admin: Whether the user has admin privileges.

    Returns:
        List of normalized, deduplicated, validated tags.

    Raises:
        ValueError: If any tag is invalid or reserved (for non-admins).
    """
    if not tags:
        raise ValueError("At least one tag is required")

    validated = []
    for tag in tags:
        normalized = validate_tag(tag)

        if normalized in RESERVED_TAGS and not is_admin:
            raise ValueError(f"Tag '{normalized}' is reserved and requires admin privileges")

        validated.append(normalized)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for tag in validated:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)

    return unique
