"""Custom exceptions for the AI Ready RAG application."""


class VectorServiceError(Exception):
    """Base exception for vector service errors.

    All vector service exceptions inherit from this class,
    allowing callers to catch all vector errors with a single handler.
    """

    pass


class EmbeddingError(VectorServiceError):
    """Failed to generate embeddings.

    Causes:
        - Ollama service unavailable
        - Embedding model not found or not loaded
        - Request timeout (>10 seconds)
        - Batch retry exhausted after 3 attempts
    """

    pass


class IndexingError(VectorServiceError):
    """Failed to index documents to Qdrant.

    Causes:
        - Qdrant service unavailable
        - Collection doesn't exist
        - Batch retry exhausted after 3 attempts
        - Payload validation failed
    """

    pass


class SearchError(VectorServiceError):
    """Failed to execute search query.

    Causes:
        - Qdrant service unavailable
        - Invalid filter construction
        - Query embedding generation failed
    """

    pass
