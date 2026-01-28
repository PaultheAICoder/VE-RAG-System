"""Configuration management using Pydantic settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


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

    # Security
    password_min_length: int = 12
    lockout_attempts: int = 5
    lockout_minutes: int = 15
    bcrypt_rounds: int = 12

    # Audit
    audit_level: Literal["essential", "comprehensive", "full_debug"] = "full_debug"

    # Feature Flags
    enable_rag: bool = False  # Disabled for auth testing
    enable_gradio: bool = False

    # Vector Service
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    embedding_max_tokens: int = 8192
    default_tenant_id: str = "default"

    # RAG Service
    chat_model: str = "llama3.2"
    rag_temperature: float = 0.1
    rag_timeout_seconds: int = 30
    rag_confidence_threshold: int = 60
    rag_admin_email: str = "admin@company.com"

    # Token Budget
    rag_max_context_tokens: int = 3000
    rag_max_history_tokens: int = 1000
    rag_max_response_tokens: int = 1024
    rag_system_prompt_tokens: int = 500

    # Retrieval Quality
    rag_min_similarity_score: float = 0.3
    rag_max_chunks_per_doc: int = 3
    rag_total_context_chunks: int = 5
    rag_dedup_candidates_cap: int = 15
    rag_chunk_overlap_threshold: float = 0.9

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
