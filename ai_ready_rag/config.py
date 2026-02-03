"""Configuration management using Pydantic settings."""

from functools import lru_cache
from typing import Any, Literal

from pydantic_settings import BaseSettings

# Profile defaults for laptop and spark deployments
PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "laptop": {
        "vector_backend": "chroma",
        "chunker_backend": "simple",
        "enable_ocr": False,
        "chat_model": "llama3.2:latest",
        "embedding_model": "nomic-embed-text",
        "rag_max_context_tokens": 2000,
        "rag_max_history_tokens": 600,
        "rag_max_response_tokens": 512,
        "rag_enable_hallucination_check": False,  # Faster dev
        # Database pool - modest for laptop hardware
        "db_pool_size": 5,
        "db_pool_max_overflow": 10,
        "db_pool_timeout": 30,
        # Concurrent processing - limited for laptop
        "max_concurrent_processing": 3,
    },
    "spark": {
        "vector_backend": "qdrant",
        "chunker_backend": "docling",
        "enable_ocr": True,
        "chat_model": "qwen3:8b",
        "embedding_model": "nomic-embed-text",
        "rag_max_context_tokens": 6000,
        "rag_max_history_tokens": 1500,
        "rag_max_response_tokens": 2048,
        "rag_enable_hallucination_check": True,  # Full quality
        # Database pool - larger for DGX Spark hardware
        "db_pool_size": 10,
        "db_pool_max_overflow": 20,
        "db_pool_timeout": 60,
        # Concurrent processing - more for Spark
        "max_concurrent_processing": 8,
    },
}


class Settings(BaseSettings):
    # Application
    app_name: str = "AI Ready RAG"
    app_version: str = "0.5.0"
    debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "sqlite:///./data/ai_ready_rag.db"
    db_pool_size: int | None = None  # None = use profile default
    db_pool_max_overflow: int | None = None  # None = use profile default
    db_pool_timeout: int | None = None  # None = use profile default

    # Processing concurrency
    max_concurrent_processing: int | None = None  # None = use profile default

    # JWT
    jwt_secret_key: str = "dev-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # Initial Admin (for seeding)
    admin_email: str = "admin@test.com"
    admin_password: str = "npassword"
    admin_display_name: str = "Administrator"

    # Security
    password_min_length: int = 12
    lockout_attempts: int = 5
    lockout_minutes: int = 15
    bcrypt_rounds: int = 12

    # Audit
    audit_level: Literal["essential", "comprehensive", "full_debug"] = "full_debug"

    # Feature Flags
    enable_rag: bool = True  # Enabled for RAG functionality
    enable_gradio: bool = True

    # Setup Wizard
    skip_setup_wizard: bool = False  # Set to True to bypass setup check for automated deployments

    # Profile Selection
    env_profile: Literal["laptop", "spark"] = "laptop"

    # Pipeline Backends (None = use profile default)
    vector_backend: Literal["chroma", "qdrant"] | None = None
    chunker_backend: Literal["simple", "docling"] | None = None

    # API
    api_base_url: str = "http://localhost:8000"

    # Vector Service
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    chroma_persist_dir: str = "./data/chroma_db"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str | None = None  # None = use profile default
    embedding_dimension: int = 768
    embedding_max_tokens: int = 8192
    default_tenant_id: str = "default"

    # RAG Service
    chat_model: str | None = None  # None = use profile default
    rag_temperature: float = 0.1
    rag_timeout_seconds: int = 30
    rag_confidence_threshold: int = 40
    rag_admin_email: str = "admin@company.com"

    # Token Budget (None = use profile default)
    rag_max_context_tokens: int | None = None
    rag_max_history_tokens: int | None = None
    rag_max_response_tokens: int | None = None
    rag_system_prompt_tokens: int = 500

    # Retrieval Quality
    rag_min_similarity_score: float = 0.3
    rag_max_chunks_per_doc: int = 3
    rag_total_context_chunks: int = 8  # Increased from 5 per Spark config
    rag_dedup_candidates_cap: int = 15
    rag_chunk_overlap_threshold: float = 0.9
    rag_enable_query_expansion: bool = True  # Expand queries for better recall
    rag_enable_hallucination_check: bool | None = None  # None = use profile default

    # Cache Warming - Core settings
    warming_delay_seconds: float = 2.0  # Delay between warming queries to reduce Ollama contention
    warming_queue_dir: str = "data/warming_queue"  # Directory for persistent job files
    warming_scan_interval_seconds: int = 60  # Folder watcher polling interval
    warming_lock_timeout_minutes: int = 30  # Reclaim stale locks after N minutes
    warming_max_file_size_mb: float = 10.0  # Max uploaded file size for CLI drops
    warming_allowed_extensions: list[str] = [".txt", ".csv"]  # Allowed CLI file types
    warming_archive_completed: bool = False  # Archive completed jobs for audit
    warming_checkpoint_interval: int = 10  # Save progress every N queries

    # Cache Warming - Worker settings
    warming_checkpoint_time_seconds: int = 5  # Max seconds between checkpoints
    warming_lease_duration_minutes: int = 10  # Job lease duration
    warming_lease_renewal_seconds: int = 60  # Lease renewal interval

    # Cache Warming - Retry settings
    warming_max_retries: int = 3
    warming_retry_delays: str = "5,30,120"  # Comma-separated seconds for exponential backoff

    # Cache Warming - SCTP settings (optional, disabled by default)
    sctp_enabled: bool = False
    sctp_host: str = "0.0.0.0"
    sctp_port: int = 9900
    sctp_max_file_size_mb: int = 10
    sctp_max_queries_per_file: int = 10000
    sctp_tls_cert: str | None = None
    sctp_tls_key: str | None = None
    sctp_tls_ca: str | None = None
    sctp_shared_secret: str | None = None
    sctp_allowed_ips: str | None = None

    # Cache Warming - Cleanup settings
    warming_completed_retention_days: int = 7
    warming_failed_retention_days: int = 30  # Note: replaces existing 7-day default
    warming_cleanup_interval_hours: int = 6

    # Cache Warming - SSE settings
    sse_event_buffer_size: int = 1000  # Ring buffer size for replay
    sse_heartbeat_seconds: int = 30

    # SSE settings
    sse_event_buffer_size: int = 1000  # Ring buffer size for replay
    sse_heartbeat_seconds: int = 30  # Heartbeat interval

    # Document Management
    upload_dir: str = "./data/uploads"
    max_upload_size_mb: int = 100
    max_storage_gb: int = 10
    allowed_extensions: list[str] = [
        "pdf",
        "docx",
        "xlsx",
        "pptx",
        "txt",
        "md",
        "html",
        "htm",
        "csv",
    ]

    # Document Processing
    enable_ocr: bool | None = None  # None = use profile default
    ocr_language: str = "eng"
    force_full_page_ocr: bool = False
    table_extraction_mode: Literal["accurate", "fast"] = "accurate"
    include_image_descriptions: bool = True
    chunk_size: int = 200  # Smaller chunks = more precise matching
    chunk_overlap: int = 40  # ~20% overlap for context continuity

    def model_post_init(self, __context: Any) -> None:
        """Apply profile defaults after Pydantic initialization."""
        profile = PROFILE_DEFAULTS.get(self.env_profile, PROFILE_DEFAULTS["laptop"])

        # Apply profile defaults for settings that are None
        for key, default_value in profile.items():
            current = getattr(self, key, None)
            if current is None:
                object.__setattr__(self, key, default_value)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
