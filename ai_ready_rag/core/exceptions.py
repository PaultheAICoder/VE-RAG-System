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


# -----------------------------------------------------------------------------
# RAG Service Exceptions
# -----------------------------------------------------------------------------


class RAGServiceError(Exception):
    """Base exception for RAG service errors.

    All RAG service exceptions inherit from this class,
    allowing callers to catch all RAG errors with a single handler.
    """

    pass


class LLMConnectionError(RAGServiceError):
    """Cannot connect to Ollama.

    Causes:
        - Ollama service unavailable
        - Network connectivity issues
    """

    pass


class LLMTimeoutError(RAGServiceError):
    """LLM response timed out.

    Causes:
        - Model inference taking too long
        - Server under heavy load
    """

    pass


class ModelNotAllowedError(RAGServiceError):
    """Requested model not in allowlist.

    Causes:
        - Model name not in MODEL_LIMITS configuration
    """

    pass


class ModelUnavailableError(RAGServiceError):
    """Model not available in Ollama.

    Causes:
        - Model not pulled (run: ollama pull <model>)
        - Model name misspelled
    """

    pass


class ContextError(RAGServiceError):
    """Failed to retrieve or format context.

    Causes:
        - VectorService search failed
        - Context formatting error
    """

    pass


class TokenBudgetExceededError(RAGServiceError):
    """System prompt + response reserve exceeds context window.

    Causes:
        - System prompt too long for model
        - Invalid token budget configuration
    """

    pass
