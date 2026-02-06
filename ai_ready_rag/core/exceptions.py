"""Custom exceptions for the AI Ready RAG application."""


# -----------------------------------------------------------------------------
# Application Base Error
# -----------------------------------------------------------------------------


class AppError(Exception):
    """Base application error with HTTP semantics.

    All domain exceptions that should map to HTTP responses inherit from this.
    The global error handler in error_handlers.py catches these and returns
    a consistent JSON response.
    """

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, detail: str = "An unexpected error occurred", context: dict | None = None):
        self.detail = detail
        self.context = context
        super().__init__(self.detail)


# -----------------------------------------------------------------------------
# Generic CRUD Exceptions
# -----------------------------------------------------------------------------


class EntityNotFound(AppError):
    """Entity not found by primary key (404)."""

    status_code = 404
    error_code = "NOT_FOUND"


# -----------------------------------------------------------------------------
# Document Service Exceptions
# -----------------------------------------------------------------------------


class ValidationError(AppError):
    """Generic validation error (400)."""

    status_code = 400
    error_code = "VALIDATION_ERROR"


class InvalidFileTypeError(AppError):
    """File type not in allowed extensions (400)."""

    status_code = 400
    error_code = "INVALID_FILE_TYPE"


class NoTagsError(AppError):
    """No tags provided for document upload (400)."""

    status_code = 400
    error_code = "NO_TAGS"


class InvalidTagsError(AppError):
    """One or more tag IDs not found (400)."""

    status_code = 400
    error_code = "INVALID_TAGS"


class FileTooLargeError(AppError):
    """File exceeds maximum upload size (400)."""

    status_code = 400
    error_code = "FILE_TOO_LARGE"


class StorageQuotaExceededError(AppError):
    """Storage quota would be exceeded (507)."""

    status_code = 507
    error_code = "STORAGE_QUOTA_EXCEEDED"


class FileStorageError(AppError):
    """Failed to write file to disk (500)."""

    status_code = 500
    error_code = "PROCESSING_FAILED"


class DuplicateFileError(AppError):
    """File with same content hash already exists (409)."""

    status_code = 409
    error_code = "DUPLICATE_FILE"


# -----------------------------------------------------------------------------
# Vector Service Exceptions
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Reindex Service Exceptions
# -----------------------------------------------------------------------------


class ReindexError(Exception):
    """Base exception for reindex service errors."""

    pass


class ReindexPausedError(ReindexError):
    """Raised when reindex pauses due to document failure.

    Allows callers to handle pause state and present options
    to the user (skip, retry, abort).
    """

    def __init__(self, job_id: str, document_id: str, error: str):
        self.job_id = job_id
        self.document_id = document_id
        self.error = error
        super().__init__(f"Reindex paused on document {document_id}: {error}")


# -----------------------------------------------------------------------------
# Cache Warming Exceptions
# -----------------------------------------------------------------------------


class WarmingError(Exception):
    """Base exception for cache warming errors."""

    pass


class ConnectionTimeoutError(WarmingError):
    """Connection to service timed out (retryable).

    Causes:
        - Ollama or Qdrant not responding
        - Network latency spike
    """

    pass


class ServiceUnavailableError(WarmingError):
    """Service temporarily unavailable (retryable).

    Causes:
        - Ollama restarting
        - Qdrant under load
    """

    pass


class RateLimitExceededError(WarmingError):
    """Rate limit exceeded (retryable with backoff).

    Causes:
        - Too many concurrent requests
    """

    pass


class WarmingCancelledException(WarmingError):
    """Job was cancelled by user (non-retryable).

    Raised to break out of processing loop cleanly.
    """

    pass
