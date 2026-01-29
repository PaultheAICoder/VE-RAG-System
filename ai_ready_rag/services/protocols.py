"""Service protocols (interfaces) for swappable backends.

Uses typing.Protocol for structural subtyping (duck typing with type safety).
Implementations don't need to inherit - they just need to have matching methods.
"""

from typing import Any, Protocol


class VectorServiceProtocol(Protocol):
    """Interface for vector database backends (Qdrant, Chroma)."""

    async def initialize(self) -> None:
        """Initialize the vector service (create collection if needed)."""
        ...

    async def add_document(
        self,
        document_id: str,
        document_name: str,
        chunks: list[str],
        tags: list[str],
        uploaded_by: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Index document chunks to vector store.

        Args:
            document_id: Unique document identifier
            document_name: Original filename
            chunks: List of text chunks to embed and index
            tags: Access control tags
            uploaded_by: User ID who uploaded the document
            metadata: Optional document metadata

        Returns:
            Number of vectors indexed
        """
        ...

    async def search(
        self,
        query: str,
        user_tags: list[str],
        tenant_id: str = "default",
        limit: int = 5,
        score_threshold: float = 0.3,
    ) -> list[Any]:
        """Search vectors filtered by user's tags.

        Args:
            query: Search query text
            user_tags: Filter to documents with these tags
            tenant_id: Tenant isolation
            limit: Max results
            score_threshold: Minimum similarity score

        Returns:
            List of SearchResult objects
        """
        ...

    async def delete_document(self, document_id: str) -> int:
        """Delete all vectors for a document.

        Args:
            document_id: Document to delete

        Returns:
            Number of vectors deleted
        """
        ...

    async def update_document_tags(self, document_id: str, tags: list[str]) -> int:
        """Update tags in vector payloads without re-embedding.

        Args:
            document_id: Document to update
            tags: New tag list

        Returns:
            Number of vectors updated
        """
        ...

    async def health_check(self) -> dict[str, Any]:
        """Check backend health.

        Returns:
            Dict with 'healthy' (bool), 'backend' (str), 'details' (dict)
        """
        ...

    async def clear_collection(self) -> bool:
        """Clear all vectors from collection.

        Returns:
            True if successful
        """
        ...


class ChunkerProtocol(Protocol):
    """Interface for document chunking backends (Simple, Docling)."""

    def chunk_document(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Chunk a document file.

        Args:
            file_path: Path to document file
            metadata: Optional metadata to include in chunks

        Returns:
            List of chunk dicts with keys:
            - text: str - The chunk text
            - chunk_index: int - Index of chunk in document
            - page_number: int | None - Page number if available
            - section: str | None - Section heading if available
        """
        ...

    def chunk_text(
        self,
        text: str,
        source: str = "unknown",
    ) -> list[dict[str, Any]]:
        """Chunk plain text (fallback for simple files).

        Args:
            text: Text content to chunk
            source: Source identifier for metadata

        Returns:
            List of chunk dicts
        """
        ...
