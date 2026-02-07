"""Admin endpoint schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator

# =============================================================================
# Document Recovery
# =============================================================================


class RecoverResponse(BaseModel):
    recovered: int
    message: str


# =============================================================================
# Knowledge Base
# =============================================================================


class FileStats(BaseModel):
    """Per-file statistics."""

    document_id: str
    filename: str
    chunk_count: int
    status: str | None = None


class KnowledgeBaseStatsResponse(BaseModel):
    """Knowledge base statistics response."""

    total_chunks: int
    unique_files: int
    total_vectors: int
    collection_name: str
    files: list[FileStats]
    storage_size_bytes: int | None = None
    last_updated: datetime


class ClearKnowledgeBaseRequest(BaseModel):
    """Request to clear knowledge base."""

    confirm: bool
    delete_source_files: bool = False


class ClearKnowledgeBaseResponse(BaseModel):
    """Response after clearing knowledge base."""

    deleted_chunks: int
    deleted_files: int
    success: bool


# =============================================================================
# Architecture Info
# =============================================================================


class TesseractStatus(BaseModel):
    """Tesseract OCR availability status."""

    available: bool
    version: str | None = None
    languages: list[str] | None = None


class EasyOCRStatus(BaseModel):
    """EasyOCR availability status."""

    available: bool
    version: str | None = None


class OCRStatus(BaseModel):
    """Combined OCR status."""

    tesseract: TesseractStatus
    easyocr: EasyOCRStatus


class DocumentParsingInfo(BaseModel):
    """Document parsing engine information."""

    engine: str = "Docling"
    version: str
    type: str = "local ML"
    capabilities: list[str]


class EmbeddingsInfo(BaseModel):
    """Embeddings configuration."""

    model: str
    dimensions: int
    vector_store: str
    vector_store_url: str


class ChatModelInfo(BaseModel):
    """Chat model configuration."""

    name: str
    provider: str = "Ollama"
    capabilities: list[str]


class InfrastructureStatus(BaseModel):
    """Infrastructure health status."""

    ollama_url: str
    ollama_status: str  # "healthy" | "unhealthy"
    vector_db_status: str  # "healthy" | "unhealthy"


class ArchitectureInfoResponse(BaseModel):
    """Complete architecture information response."""

    document_parsing: DocumentParsingInfo
    embeddings: EmbeddingsInfo
    chat_model: ChatModelInfo
    infrastructure: InfrastructureStatus
    ocr_status: OCRStatus
    profile: str


# =============================================================================
# Processing Options
# =============================================================================


class ProcessingOptionsRequest(BaseModel):
    """Request model for processing options update."""

    enable_ocr: bool | None = None
    force_full_page_ocr: bool | None = None
    ocr_language: str | None = None
    table_extraction_mode: Literal["accurate", "fast"] | None = None
    include_image_descriptions: bool | None = None
    query_routing_mode: Literal["retrieve_only", "retrieve_and_direct"] | None = None


class ProcessingOptionsResponse(BaseModel):
    """Response model for processing options."""

    enable_ocr: bool
    force_full_page_ocr: bool
    ocr_language: str
    table_extraction_mode: str
    include_image_descriptions: bool
    query_routing_mode: str


# =============================================================================
# Model Configuration
# =============================================================================


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str
    display_name: str
    size_gb: float
    parameters: str | None = None
    quantization: str | None = None
    recommended: bool = False


class ModelsResponse(BaseModel):
    """Response containing available models and current selection."""

    available_models: list[ModelInfo]
    embedding_models: list[ModelInfo]
    chat_models: list[ModelInfo]
    current_chat_model: str
    current_embedding_model: str


class ChangeModelRequest(BaseModel):
    """Request to change the active chat model."""

    model_name: str


class ChangeModelResponse(BaseModel):
    """Response after changing the chat model."""

    previous_model: str
    current_model: str
    success: bool
    message: str


class ChangeEmbeddingRequest(BaseModel):
    """Request to change the embedding model."""

    model_name: str
    confirm_reindex: bool = False  # Must be True to proceed


class ChangeEmbeddingResponse(BaseModel):
    """Response after changing the embedding model."""

    previous_model: str
    current_model: str
    success: bool
    message: str
    reindex_required: bool = True
    documents_affected: int = 0


# =============================================================================
# Model Limits
# =============================================================================


class ModelLimits(BaseModel):
    """Limits for a specific model."""

    context_window: int
    max_response: int
    temperature_min: float = 0.0
    temperature_max: float = 1.0


class ModelLimitsResponse(BaseModel):
    """Response containing model limits for settings validation."""

    current_model: str
    limits: ModelLimits
    all_models: dict[str, ModelLimits]


# =============================================================================
# Detailed Health
# =============================================================================


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    name: str
    status: Literal["healthy", "unhealthy", "degraded"]
    version: str | None = None
    details: dict | None = None


class RAGPipelineStatus(BaseModel):
    """RAG pipeline status information."""

    embedding_model: str
    chat_model: str
    chunker: str
    stages: list[str]
    all_stages_healthy: bool


class KnowledgeBaseSummary(BaseModel):
    """Knowledge base summary statistics."""

    total_documents: int
    total_chunks: int
    storage_size_mb: float | None = None


class ProcessingQueueStatus(BaseModel):
    """Document processing queue status."""

    pending: int
    processing: int
    failed: int
    ready: int


class DetailedHealthResponse(BaseModel):
    """Comprehensive health response for admin dashboard."""

    # Overall status
    status: Literal["healthy", "unhealthy", "degraded"]
    version: str
    profile: str

    # Component health
    api_server: ComponentHealth
    ollama_llm: ComponentHealth
    vector_db: ComponentHealth

    # RAG Pipeline status
    rag_pipeline: RAGPipelineStatus

    # Knowledge base summary
    knowledge_base: KnowledgeBaseSummary

    # Processing queue
    processing_queue: ProcessingQueueStatus

    # System info
    uptime_seconds: int
    last_checked: datetime


# =============================================================================
# RAG Tuning Settings
# =============================================================================


class RetrievalSettingsRequest(BaseModel):
    """Request model for updating retrieval settings."""

    retrieval_top_k: int | None = None
    retrieval_min_score: float | None = None
    retrieval_enable_expansion: bool | None = None


class RetrievalSettingsResponse(BaseModel):
    """Response model for retrieval settings."""

    retrieval_top_k: int
    retrieval_min_score: float
    retrieval_enable_expansion: bool


class LLMSettingsRequest(BaseModel):
    """Request model for updating LLM settings."""

    llm_temperature: float | None = None
    llm_max_response_tokens: int | None = None
    llm_confidence_threshold: int | None = None


class LLMSettingsResponse(BaseModel):
    """Response model for LLM settings."""

    llm_temperature: float
    llm_max_response_tokens: int
    llm_confidence_threshold: int


class SettingsAuditEntry(BaseModel):
    """Response model for settings audit entry."""

    id: str
    setting_key: str
    old_value: str | None
    new_value: str
    changed_by: str | None
    changed_at: datetime
    change_reason: str | None


class SettingsAuditResponse(BaseModel):
    """Response model for settings audit list."""

    entries: list[SettingsAuditEntry]
    total: int
    limit: int
    offset: int


# =============================================================================
# Security Settings
# =============================================================================


class SecuritySettingsResponse(BaseModel):
    """Response containing security settings."""

    jwt_expiration_hours: int
    password_min_length: int
    bcrypt_rounds: int


class SecuritySettingsRequest(BaseModel):
    """Request to update security settings."""

    jwt_expiration_hours: int | None = None
    password_min_length: int | None = None
    bcrypt_rounds: int | None = None


# =============================================================================
# Feature Flags
# =============================================================================


class FeatureFlagsResponse(BaseModel):
    """Response containing feature flag settings."""

    enable_rag: bool
    skip_setup_wizard: bool


class FeatureFlagsRequest(BaseModel):
    """Request to update feature flags."""

    enable_rag: bool | None = None
    skip_setup_wizard: bool | None = None


# =============================================================================
# Advanced Settings & Reindex
# =============================================================================


class AdvancedSettingsResponse(BaseModel):
    """Response containing advanced (destructive) RAG settings."""

    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    hnsw_ef_construct: int
    hnsw_m: int
    vector_backend: str


class AdvancedSettingsRequest(BaseModel):
    """Request to update advanced settings (requires reindex)."""

    embedding_model: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    hnsw_ef_construct: int | None = None
    hnsw_m: int | None = None
    vector_backend: str | None = None
    confirm_reindex: bool = False


class ReindexJobResponse(BaseModel):
    """Reindex job information."""

    id: str
    status: str
    total_documents: int
    processed_documents: int
    failed_documents: int
    progress_percent: float
    current_document_id: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime
    settings_changed: dict | None = None
    # Phase 3: Failure handling fields
    last_error: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    paused_at: datetime | None = None
    paused_reason: str | None = None
    auto_skip_failures: bool = False


class ReindexEstimate(BaseModel):
    """Time estimate for reindex operation."""

    total_documents: int
    avg_processing_time_ms: int
    estimated_total_seconds: int
    estimated_time_str: str


class StartReindexRequest(BaseModel):
    """Request to start reindex operation."""

    confirm: bool = False


class ResumeReindexRequest(BaseModel):
    """Request to resume a paused reindex."""

    action: Literal["skip", "retry", "skip_all"]


class ReindexFailureInfo(BaseModel):
    """Information about a failed document during reindex."""

    document_id: str
    filename: str
    status: str
    error_message: str | None = None


class ReindexFailuresResponse(BaseModel):
    """Response containing reindex failure details."""

    job_id: str
    failures: list[ReindexFailureInfo]
    total_failures: int


# =============================================================================
# Cache Warming
# =============================================================================


class TopQueryItem(BaseModel):
    """Single query with access frequency data."""

    query_text: str
    access_count: int
    last_accessed: datetime | None = None


class TopQueryResponse(BaseModel):
    """Response containing top queries for warming."""

    queries: list[TopQueryItem]


class CacheWarmRequest(BaseModel):
    """Request to warm cache with specific queries."""

    queries: list[str]


class CacheWarmResponse(BaseModel):
    """Response after starting cache warming."""

    queued: int
    message: str


class WarmFileResponse(BaseModel):
    """Response after starting file-based cache warming."""

    job_id: str
    queued: int
    message: str
    sse_url: str


class WarmRetryRequest(BaseModel):
    """Request to retry failed warming queries."""

    queries: list[str]


class WarmingJobResponse(BaseModel):
    """Response model for a warming job."""

    id: str
    source_file: str | None = None  # Not tracked in current implementation
    status: str
    total: int
    processed: int
    success_count: int
    failed_count: int
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    triggered_by: str


class WarmingJobListResponse(BaseModel):
    """Response model for list of warming jobs."""

    jobs: list[WarmingJobResponse]
    total_count: int


# =============================================================================
# DB-based Warming Queue (Issue #121)
# =============================================================================


class WarmingQueueJobResponse(BaseModel):
    """Response model for a DB-based warming queue job."""

    id: str
    file_path: str
    source_type: str
    original_filename: str | None = None
    total_queries: int
    processed_queries: int
    failed_queries: int
    status: str
    is_paused: bool
    is_cancel_requested: bool
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_by: str | None = None
    error_message: str | None = None


class WarmingQueueListResponse(BaseModel):
    """Response model for list of warming queue jobs."""

    jobs: list[WarmingQueueJobResponse]
    total_count: int


class ManualWarmingRequest(BaseModel):
    """Request to add manual queries to warming queue."""

    queries: list[str]


class BulkDeleteRequest(BaseModel):
    """Request to bulk delete warming jobs."""

    job_ids: list[str]


# =============================================================================
# Cache Settings & Stats
# =============================================================================


class CacheSettingsResponse(BaseModel):
    """Response model for cache settings."""

    cache_enabled: bool
    cache_ttl_hours: int
    cache_max_entries: int
    cache_semantic_threshold: float
    cache_min_confidence: int
    cache_auto_warm_enabled: bool
    cache_auto_warm_count: int


class CacheSettingsRequest(BaseModel):
    """Request model for updating cache settings."""

    cache_enabled: bool | None = None
    cache_ttl_hours: int | None = None
    cache_max_entries: int | None = None
    cache_semantic_threshold: float | None = None
    cache_min_confidence: int | None = None
    cache_auto_warm_enabled: bool | None = None
    cache_auto_warm_count: int | None = None


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""

    enabled: bool
    total_entries: int
    memory_entries: int
    sqlite_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    avg_response_time_cached_ms: float | None = None
    avg_response_time_uncached_ms: float | None = None
    storage_size_bytes: int | None = None
    oldest_entry: datetime | None = None
    newest_entry: datetime | None = None


class CacheClearResponse(BaseModel):
    """Response after clearing cache."""

    cleared_entries: int
    message: str


class CacheSeedRequest(BaseModel):
    """Request to seed cache with a curated response."""

    query: str
    answer: str
    source_reference: str
    confidence: int = 85

    @field_validator("source_reference")
    @classmethod
    def source_required(cls, v: str) -> str:
        """Validate that source_reference is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_reference is required for compliance")
        return v.strip()

    @field_validator("query")
    @classmethod
    def query_required(cls, v: str) -> str:
        """Validate that query is non-empty."""
        if not v or not v.strip():
            raise ValueError("query is required")
        return v.strip()


class CacheSeedResponse(BaseModel):
    """Response after seeding cache."""

    query_hash: str
    message: str


# =============================================================================
# Synonyms
# =============================================================================


class SynonymCreate(BaseModel):
    """Request model for creating a synonym."""

    term: str
    synonyms: list[str]


class SynonymUpdate(BaseModel):
    """Request model for updating a synonym."""

    term: str | None = None
    synonyms: list[str] | None = None
    enabled: bool | None = None


class SynonymResponse(BaseModel):
    """Response model for a synonym."""

    id: str
    term: str
    synonyms: list[str]
    enabled: bool
    created_by: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SynonymListResponse(BaseModel):
    """Response model for paginated synonym list."""

    synonyms: list[SynonymResponse]
    total: int
    page: int
    page_size: int


# =============================================================================
# Curated Q&A
# =============================================================================


class CuratedQACreate(BaseModel):
    """Request model for creating a curated Q&A."""

    keywords: list[str]
    answer: str
    source_reference: str
    confidence: int = 85
    priority: int = 0

    @field_validator("source_reference")
    @classmethod
    def source_required(cls, v: str) -> str:
        """Validate that source_reference is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_reference is required for compliance")
        return v.strip()


class CuratedQAUpdate(BaseModel):
    """Request model for updating a curated Q&A."""

    keywords: list[str] | None = None
    answer: str | None = None
    source_reference: str | None = None
    confidence: int | None = None
    priority: int | None = None
    enabled: bool | None = None


class CuratedQAResponse(BaseModel):
    """Response model for a curated Q&A."""

    id: str
    keywords: list[str]
    answer: str
    source_reference: str
    confidence: int
    priority: int
    enabled: bool
    access_count: int
    last_accessed_at: datetime | None
    created_by: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CuratedQAListResponse(BaseModel):
    """Response model for paginated Q&A list."""

    qa_pairs: list[CuratedQAResponse]
    total: int
    page: int
    page_size: int
