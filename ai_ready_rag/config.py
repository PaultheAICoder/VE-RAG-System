"""Configuration management using Pydantic settings."""

import logging
import os
import warnings
from functools import lru_cache
from typing import Any, Literal

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Governance — Issue #379
# Pinned Claude model IDs — do not use aliases or short names
# ---------------------------------------------------------------------------

CLAUDE_ENRICHMENT_MODEL = "claude-sonnet-4-6"
CLAUDE_QUERY_MODEL_SIMPLE = "claude-haiku-4-5-20251001"
CLAUDE_QUERY_MODEL_COMPLEX = "claude-sonnet-4-6"

VALID_CLAUDE_MODELS: frozenset[str] = frozenset(
    {
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    }
)

REJECTED_CLAUDE_ALIASES: frozenset[str] = frozenset(
    {
        "claude-sonnet",
        "sonnet",
        "claude-haiku",
        "haiku",
        "claude-3",
        "claude-3-sonnet",
        "claude-3-haiku",
    }
)


def validate_claude_model_id(model_id: str, field: str) -> None:
    """Raise ValueError if model_id is an alias or unknown Claude model."""
    if model_id in REJECTED_CLAUDE_ALIASES:
        raise ValueError(
            f"Model alias '{model_id}' is not permitted in {field}. "
            f"Use a pinned ID from VALID_CLAUDE_MODELS."
        )
    if model_id.startswith("claude-") and model_id not in VALID_CLAUDE_MODELS:
        raise ValueError(
            f"Unknown Claude model ID '{model_id}' in {field}. "
            f"Valid IDs: {sorted(VALID_CLAUDE_MODELS)}"
        )


# Profile defaults for laptop, spark, and hosted deployments
PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "laptop": {
        # Portable — mirrors spark profile. Only hardware-bound limits differ.
        "vector_backend": "pgvector",
        "chunker_backend": "docling",
        "enable_ocr": True,
        "chat_model": "llama3.2:latest",  # Fallback LLM; Claude is primary
        "embedding_model": "nomic-embed-text",
        "rag_max_context_tokens": 2000,  # Hardware limit: smaller than Spark
        "rag_max_history_tokens": 600,
        "rag_max_response_tokens": 512,  # Hardware limit: smaller than Spark
        "rag_enable_hallucination_check": True,
        # Database pool - modest for laptop hardware
        "db_pool_size": 5,
        "db_pool_max_overflow": 10,
        "db_pool_timeout": 30,
        # Concurrent processing - 2 to avoid tesseract/OCR race conditions (same as Spark)
        "max_concurrent_processing": 2,
        # Summary generation
        "generate_summaries": True,
        # ingestkit - all enabled (mirrors Spark)
        "use_ingestkit_forms": True,
        "forms_ocr_engine": "tesseract",  # tesseract on laptop, paddleocr on Spark
        "forms_vlm_enabled": True,
        "use_ingestkit_image": True,
        "use_ingestkit_email": True,
        # Deployment tier and feature flags
        "deployment_tier": "enterprise",
        "claude_enrichment_enabled": True,
        "claude_enrichment_model": "claude-sonnet-4-6",
        "claude_query_enabled": True,
        "structured_query_enabled": True,
        "database_backend": "postgresql",
    },
    "spark": {
        "vector_backend": "pgvector",
        "chunker_backend": "docling",
        "enable_ocr": True,
        "chat_model": "qwen3-rag",
        "embedding_model": "nomic-embed-text",
        "rag_max_context_tokens": 6000,
        "rag_max_history_tokens": 1500,
        "rag_max_response_tokens": 2048,
        "rag_enable_hallucination_check": True,  # Full quality
        # Database pool - larger for DGX Spark hardware
        "db_pool_size": 10,
        "db_pool_max_overflow": 20,
        "db_pool_timeout": 60,
        # Concurrent processing - conservative to avoid tesseract/OCR race conditions
        "max_concurrent_processing": 2,
        # Summary generation
        "generate_summaries": True,
        # ingestkit-forms - enabled on Spark with PaddleOCR + VLM
        "use_ingestkit_forms": True,
        "forms_ocr_engine": "paddleocr",
        "forms_vlm_enabled": True,
        # ingestkit-image / ingestkit-email - enabled on Spark
        "use_ingestkit_image": True,
        "use_ingestkit_email": True,
        # Deployment tier and feature flags
        "deployment_tier": "enterprise",
        "claude_enrichment_enabled": True,
        "claude_query_enabled": True,
        "structured_query_enabled": True,
        "database_backend": "postgresql",
        "claude_enrichment_model": "claude-sonnet-4-6",
    },
    "hosted": {
        "deployment_tier": "standard",
        "claude_enrichment_enabled": True,
        "claude_query_enabled": True,
        "structured_query_enabled": True,
        "database_backend": "postgresql",
        "vector_backend": "pgvector",
        "claude_enrichment_model": "claude-sonnet-4-6",
        "claude_query_model_simple": "claude-haiku-4-5-20251001",
        "claude_query_model_complex": "claude-sonnet-4-6",
        "chat_model": "qwen3-rag",
    },
}


# Per-document-type chunk sizes based on tag name.
# Documents matching these tags use larger chunks for better context retention.
# Falls back to settings.chunk_size for unmatched documents.
CHUNK_SIZE_BY_TAG: dict[str, int] = {
    "insurance": 1024,
    "legal": 1024,
    "financial": 1024,
    "policy": 1024,
    "hr": 512,
    "handbook": 512,
}


class Settings(BaseSettings):
    # Application
    app_name: str = "AI Ready RAG"
    app_version: str = "0.5.0"
    debug: bool = True
    log_level: str = "INFO"
    log_format: str = "json"  # "json" for production, "console" for dev

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
    generate_summaries: bool | None = None  # None = use profile default
    # Setup Wizard
    skip_setup_wizard: bool = False  # Set to True to bypass setup check for automated deployments

    # Profile Selection
    env_profile: Literal["laptop", "spark", "hosted"] = "laptop"

    # Deployment tier — set by profile defaults
    deployment_tier: Literal["standard", "enterprise"] | None = None  # None = use profile default

    # Pipeline Backends (None = use profile default)
    vector_backend: Literal["pgvector"] | None = None
    chunker_backend: Literal["simple", "docling"] | None = None

    # API
    api_base_url: str = "http://localhost:8000"

    # Redis / ARQ Task Queue
    redis_url: str = "redis://localhost:6379"
    arq_job_timeout: int = 600  # 10 min max for document processing
    arq_max_jobs: int = 2  # Max concurrent ARQ jobs (low to avoid tesseract race conditions)
    arq_health_check_interval: int = 60  # Seconds between worker health checks
    use_arq_worker: bool = True  # Set False to bypass ARQ and use BackgroundTasks

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
    summary_model: str | None = None  # Override model for summaries (default: use chat_model)
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
    rag_max_chunks_per_doc: int = 5
    rag_total_context_chunks: int = 8  # Increased from 5 per Spark config
    rag_dedup_candidates_cap: int = 15
    rag_chunk_overlap_threshold: float = 0.9
    rag_enable_query_expansion: bool = True  # Expand queries for better recall
    rag_enable_hallucination_check: bool | None = None  # None = use profile default
    rag_recency_weight: float = 0.15  # 0=disabled, blend weight for recency boost
    coverage_rechunk_enabled: bool = True  # Post-process Coverage Summary xlsx with section chunker
    rag_intent_boost_weight: float = 0.35  # 0=disabled, blend weight for intent tag boost

    # Cache Warming - Core settings
    warming_delay_seconds: float = 2.0  # Delay between warming queries to reduce Ollama contention
    warming_scan_interval_seconds: int = 5  # DB polling interval for pending batches
    warming_max_upload_size_mb: float = 10.0  # Max uploaded file size for batch submissions
    warming_max_queries_per_batch: int = 10000  # Max queries per batch submission

    # Cache Warming - Worker settings
    warming_max_concurrent_queries: int = 2  # Max concurrent Ollama calls during warming
    warming_lease_duration_minutes: int = 10  # Batch lease duration
    warming_lease_renewal_seconds: int = 60  # Lease renewal interval

    # Cache Warming - Retry settings
    warming_max_retries: int = 3
    warming_retry_delays: str = "5,30,120"  # Comma-separated seconds for exponential backoff
    warming_cancel_timeout_seconds: int = 5  # Max wait after cancel before abandoning query

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
    warming_failed_retention_days: int = 30
    warming_cleanup_interval_hours: int = 6

    # SSE settings
    sse_event_buffer_size: int = 1000  # Ring buffer size for replay
    sse_heartbeat_seconds: int = 30  # Heartbeat interval

    # Evaluation Framework
    eval_enabled: bool = True
    eval_scan_interval_seconds: int = 30
    eval_timeout_seconds: int = 120
    eval_sample_deadline_seconds: int = 300
    eval_delay_between_samples_seconds: float = 1.0
    eval_lease_duration_minutes: int = 15
    eval_lease_renewal_seconds: int = 60
    eval_max_retries_per_sample: int = 1
    eval_retry_backoff_seconds: int = 10
    eval_max_samples_per_run: int = 500
    eval_max_run_duration_hours: float = 8.0
    eval_live_max_concurrent: int = 2
    eval_live_queue_size: int = 10
    eval_live_retention_days: int = 30

    # Evaluation - RAGBench
    ragbench_data_dir: str = "./data/ragbench"

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
        # Image formats (OCR extraction)
        "png",
        "jpg",
        "jpeg",
        "tiff",
        "tif",
        # Email formats
        "eml",
        "msg",
    ]

    # ingestkit-excel integration
    use_ingestkit_excel: bool = False  # Feature flag (off by default)
    excel_classification_model: str = "claude-sonnet-4-6"  # Claude is primary LLM
    excel_reasoning_model: str = "claude-sonnet-4-6"  # Claude is primary LLM
    excel_enable_tier3: bool = True
    excel_tier2_confidence_threshold: float = 0.6

    # ingestkit-forms integration
    use_ingestkit_forms: bool = False  # Feature flag (off by default)
    forms_match_confidence_threshold: float = 0.8
    forms_ocr_engine: str = "tesseract"
    forms_vlm_enabled: bool = False
    forms_vlm_model: str = "claude-sonnet-4-6"  # Claude is primary LLM
    forms_template_storage_path: str = "./data/form_templates"
    forms_redact_high_risk_fields: bool = True  # Redact SSN, tax ID, account numbers
    forms_template_require_approval: bool = True  # Templates must be approved before matching
    forms_rechunk_enabled: bool = True  # Split form mega-chunks into field groups

    # ingestkit-image integration
    use_ingestkit_image: bool | None = None  # None = use profile default
    image_ocr_language: str = "eng"
    image_ocr_config: str | None = None

    # ingestkit-email integration
    use_ingestkit_email: bool | None = None  # None = use profile default
    email_include_headers: bool = True

    # Auto-tagging
    auto_tagging_enabled: bool = False
    auto_tagging_strategy: str = "generic"
    auto_tagging_path_enabled: bool = True
    auto_tagging_llm_enabled: bool = True
    auto_tagging_llm_model: str | None = None  # None = use chat_model at runtime
    auto_tagging_require_approval: bool = False
    auto_tagging_create_missing_tags: bool = True
    auto_tagging_confidence_threshold: float = 0.7
    auto_tagging_suggestion_threshold: float = 0.4
    auto_tagging_strategies_dir: str = "./data/auto_tag_strategies"

    # Auto-tagging guardrails
    auto_tagging_max_tags_per_doc: int = 20
    auto_tagging_max_tag_name_length: int = 100
    auto_tagging_max_client_tags: int = 500
    auto_tagging_llm_timeout_seconds: int = 30
    auto_tagging_llm_max_retries: int = 1

    # Vertical modules to load at startup (comma-separated in env: ACTIVE_MODULES=ca,insurance)
    active_modules: list[str] = []

    # Document Processing
    enable_ocr: bool | None = None  # None = use profile default
    ocr_language: str = "eng"
    force_full_page_ocr: bool = False
    table_extraction_mode: Literal["accurate", "fast"] = "accurate"
    include_image_descriptions: bool = True
    chunk_size: int = 512  # Optimal for retrieval recall (400-512 tokens per research)
    chunk_overlap: int = 80  # ~15% overlap for boundary coverage

    # ---------------------------------------------------------------------------
    # Claude Enrichment (both tiers) — Issue #374
    # ---------------------------------------------------------------------------
    claude_enrichment_enabled: bool | None = None  # None = use profile default
    claude_api_key: str | None = None  # from ANTHROPIC_API_KEY env var
    claude_enrichment_model: str = CLAUDE_ENRICHMENT_MODEL
    claude_enrichment_batch_size: int = 8
    claude_enrichment_max_retries: int = 3
    claude_enrichment_timeout: int = 60
    claude_enrichment_cost_limit_usd: float = 10.0  # daily cap
    # Enrichment backend: "api" = Anthropic HTTP API, "cli" = claude -p subprocess (#435)
    claude_backend: Literal["api", "cli"] = "api"

    # ---------------------------------------------------------------------------
    # Claude Query (Standard tier primary) — Issue #374
    # ---------------------------------------------------------------------------
    claude_query_enabled: bool | None = None  # None = use profile default
    claude_query_model_simple: str = CLAUDE_QUERY_MODEL_SIMPLE
    claude_query_model_complex: str = CLAUDE_QUERY_MODEL_COMPLEX
    claude_query_cost_limit_usd: float = 50.0  # monthly cap

    # ---------------------------------------------------------------------------
    # Database / Vector backend — Issue #374
    # ---------------------------------------------------------------------------
    database_backend: Literal["sqlite", "postgresql"] | None = None  # None = use profile default
    pgvector_dimension: int = 768
    pgvector_index_type: str = "ivfflat"
    pgvector_lists: int = 100
    pgvector_probes: int = 10

    # ---------------------------------------------------------------------------
    # Query Router — Issue #374
    # ---------------------------------------------------------------------------
    structured_query_enabled: bool | None = None  # None = use profile default
    structured_query_row_cap: int = 1000
    structured_query_timeout_seconds: int = 5
    structured_query_confidence_threshold: float = 0.6

    # ---------------------------------------------------------------------------
    # Tenant / Module — Issue #374
    # ---------------------------------------------------------------------------
    active_modules: list[str] = ["core"]
    tenant_config_path: str = "tenant-instances/{tenant_id}/tenant.json"
    vaultiq_encryption_key: str | None = None  # from VAULTIQ_ENCRYPTION_KEY env var

    # Webhook / Integrations
    webhook_enabled: bool = False
    webhook_secret: str | None = None  # HMAC signing secret (sha256)
    webhook_timeout_seconds: int = 10
    webhook_max_retries: int = 3

    def model_post_init(self, __context: Any) -> None:
        """Apply profile defaults after Pydantic initialization."""
        profile = PROFILE_DEFAULTS.get(self.env_profile, PROFILE_DEFAULTS["laptop"])

        # Apply profile defaults for settings that are None
        for key, default_value in profile.items():
            current = getattr(self, key, None)
            if current is None:
                object.__setattr__(self, key, default_value)

        # Model governance validation — Issue #379
        # Log CRITICAL if a bad model ID is configured; don't raise so the app
        # can still start while Claude integration is not yet wired in.
        for field, value in [
            ("claude_enrichment_model", self.claude_enrichment_model),
            ("claude_query_model_simple", self.claude_query_model_simple),
            ("claude_query_model_complex", self.claude_query_model_complex),
        ]:
            try:
                validate_claude_model_id(value, field)
            except ValueError as exc:
                logger.critical("Model governance violation in settings.%s: %s", field, exc)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if os.environ.get("QDRANT_URL"):
        warnings.warn(
            "QDRANT_URL env var is set but Qdrant has been removed. "
            "The system now uses pgvector. This variable has no effect.",
            DeprecationWarning,
            stacklevel=2,
        )
    if os.environ.get("QDRANT_COLLECTION"):
        warnings.warn(
            "QDRANT_COLLECTION env var is set but Qdrant has been removed. "
            "The system now uses pgvector. This variable has no effect.",
            DeprecationWarning,
            stacklevel=2,
        )
    return settings
