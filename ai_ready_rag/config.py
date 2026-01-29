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

    # Profile Selection
    env_profile: Literal["laptop", "spark"] = "laptop"

    # Pipeline Backends (None = use profile default)
    vector_backend: Literal["chroma", "qdrant"] | None = None
    chunker_backend: Literal["simple", "docling"] | None = None

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
    rag_total_context_chunks: int = 5
    rag_dedup_candidates_cap: int = 15
    rag_chunk_overlap_threshold: float = 0.9

    # Document Management
    upload_dir: str = "./data/uploads"
    max_upload_size_mb: int = 100
    max_storage_gb: int = 10
    allowed_extensions: list[str] = ["pdf", "docx", "xlsx", "pptx", "txt", "md", "html", "csv"]

    # Document Processing
    enable_ocr: bool | None = None  # None = use profile default
    ocr_language: str = "eng"
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
