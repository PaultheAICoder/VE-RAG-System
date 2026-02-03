"""SQLAlchemy ORM models."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import relationship

from ai_ready_rag.db.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


# Association tables (no extra columns to avoid FK ambiguity)
user_tags = Table(
    "user_tags",
    Base.metadata,
    Column("user_id", String, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", String, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)

document_tags = Table(
    "document_tags",
    Base.metadata,
    Column("document_id", String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", String, ForeignKey("tags.id"), primary_key=True),
)


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    display_name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default="user")
    is_active = Column(Boolean, default=True)
    must_reset_password = Column(Boolean, default=False)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, nullable=True)  # Removed FK to avoid circular ref
    last_login = Column(DateTime, nullable=True)
    login_count = Column(Integer, default=0)

    # Simple many-to-many with tags
    tags = relationship("Tag", secondary=user_tags, back_populates="users")


class Tag(Base):
    __tablename__ = "tags"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, unique=True, nullable=False)
    display_name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    color = Column(String, default="#6B7280")
    owner_id = Column(String, nullable=True)  # Removed FK to avoid circular ref
    is_system = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, nullable=True)  # Removed FK to avoid circular ref

    # Relationships
    users = relationship("User", secondary=user_tags, back_populates="tags")
    documents = relationship("Document", secondary=document_tags, back_populates="tags")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    status = Column(String, default="pending")
    error_message = Column(Text, nullable=True)
    chunk_count = Column(Integer, nullable=True)
    uploaded_by = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    # Extended fields (spec v1.2)
    title = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    content_hash = Column(String, nullable=True, index=True)

    tags = relationship("Tag", secondary=document_tags, back_populates="documents")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_archived = Column(Boolean, default=False)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    confidence_retrieval = Column(Float, nullable=True)
    confidence_coverage = Column(Float, nullable=True)
    confidence_llm = Column(Integer, nullable=True)
    generation_time_ms = Column(Float, nullable=True)
    was_routed = Column(Boolean, default=False)
    routed_to = Column(String, nullable=True)
    route_reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String, nullable=False)
    event_type = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=True)
    user_email = Column(String, nullable=True)
    action = Column(String, nullable=False)
    resource_type = Column(String, nullable=True)
    resource_id = Column(String, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    request_id = Column(String, nullable=True)


class AdminSetting(Base):
    __tablename__ = "admin_settings"

    id = Column(String, primary_key=True, default=generate_uuid)
    key = Column(String, unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)  # JSON encoded
    updated_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SystemSetup(Base):
    """Tracks system setup state for first-run wizard."""

    __tablename__ = "system_setup"

    id = Column(String, primary_key=True, default=generate_uuid)
    setup_complete = Column(Boolean, default=False)
    admin_password_changed = Column(Boolean, default=False)
    setup_completed_at = Column(DateTime, nullable=True)
    setup_completed_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SettingsAudit(Base):
    """Tracks all admin settings changes for audit trail."""

    __tablename__ = "settings_audit"

    id = Column(String, primary_key=True, default=generate_uuid)
    setting_key = Column(String, nullable=False, index=True)
    old_value = Column(Text, nullable=True)  # JSON encoded
    new_value = Column(Text, nullable=False)  # JSON encoded
    changed_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    changed_at = Column(DateTime, default=datetime.utcnow)
    change_reason = Column(String, nullable=True)


class ReindexJob(Base):
    """Tracks background knowledge base reindex operations."""

    __tablename__ = "reindex_jobs"

    id = Column(String, primary_key=True, default=generate_uuid)
    status = Column(
        String, default="pending"
    )  # pending, running, paused, completed, failed, aborted
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    total_documents = Column(Integer, default=0)
    processed_documents = Column(Integer, default=0)
    failed_documents = Column(Integer, default=0)
    current_document_id = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    triggered_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    settings_changed = Column(Text, nullable=True)  # JSON encoded dict of changed settings
    temp_collection_name = Column(String, nullable=True)  # Temp collection being built
    created_at = Column(DateTime, default=datetime.utcnow)

    # Phase 3: Failure handling fields
    failed_document_ids = Column(Text, nullable=True)  # JSON list of failed doc IDs
    last_error = Column(Text, nullable=True)  # Last error message
    retry_count = Column(Integer, default=0)  # Retries for current document
    max_retries = Column(Integer, default=3)  # Max retries before skip
    paused_at = Column(DateTime, nullable=True)  # When job was paused
    paused_reason = Column(String, nullable=True)  # 'failure' or 'user_request'
    auto_skip_failures = Column(Boolean, default=False)  # Skip-all mode


class ResponseCache(Base):
    """Cached RAG responses for fast retrieval."""

    __tablename__ = "response_cache"

    id = Column(String, primary_key=True, default=generate_uuid)
    query_hash = Column(String, unique=True, nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    query_embedding = Column(Text, nullable=False)  # JSON-encoded float array
    answer = Column(Text, nullable=False)
    sources = Column(Text, nullable=False)  # JSON-encoded citations
    confidence_overall = Column(Integer, nullable=False)
    confidence_retrieval = Column(Float, nullable=False)
    confidence_coverage = Column(Float, nullable=False)
    confidence_llm = Column(Integer, nullable=False)
    generation_time_ms = Column(Float, nullable=False)
    model_used = Column(String, nullable=False)
    document_ids = Column(Text, nullable=False)  # JSON-encoded list[str]
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_accessed_at = Column(DateTime, default=datetime.utcnow, index=True)
    access_count = Column(Integer, default=1)


class EmbeddingCache(Base):
    """Cached query embeddings to skip Ollama embed calls."""

    __tablename__ = "embedding_cache"

    query_hash = Column(String, primary_key=True)
    query_text = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)  # JSON-encoded float array
    created_at = Column(DateTime, default=datetime.utcnow)


class CacheAccessLog(Base):
    """Tracks cache hit/miss for query frequency analysis."""

    __tablename__ = "cache_access_log"

    id = Column(String, primary_key=True, default=generate_uuid)
    query_hash = Column(String, nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    was_hit = Column(Boolean, nullable=False)
    accessed_at = Column(DateTime, default=datetime.utcnow, index=True)


class WarmingQueue(Base):
    """Persistent cache warming job queue."""

    __tablename__ = "warming_queue"
    __table_args__ = (
        Index("idx_warming_queue_fifo", "status", "created_at"),
        Index("idx_warming_queue_status", "status"),
        Index("idx_warming_queue_completed", "completed_at"),
        Index("idx_warming_queue_lease", "worker_lease_expires_at"),
    )

    id = Column(String, primary_key=True, default=generate_uuid)
    file_path = Column(Text, nullable=False)  # Path to query file (immutable)
    file_checksum = Column(String, nullable=False)  # SHA256 for integrity verification
    source_type = Column(String, nullable=False)  # 'manual' | 'upload' | 'sctp'
    original_filename = Column(String, nullable=True)  # User-friendly name
    total_queries = Column(Integer, nullable=False)  # Total queries in file
    processed_queries = Column(Integer, default=0)
    failed_queries = Column(Integer, default=0)
    byte_offset = Column(Integer, default=0)  # File position for resume
    status = Column(String, default="pending")  # pending|running|paused|completed|failed|cancelled
    is_paused = Column(Boolean, default=False)  # Persisted pause flag
    is_cancel_requested = Column(Boolean, default=False)  # Graceful cancel flag
    worker_id = Column(String, nullable=True)  # Lease: which worker owns this job
    worker_lease_expires_at = Column(DateTime, nullable=True)  # Lease expiry
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)


class WarmingFailedQuery(Base):
    """Failed queries during cache warming jobs."""

    __tablename__ = "warming_failed_queries"
    __table_args__ = (Index("idx_warming_failed_job", "job_id"),)

    id = Column(String, primary_key=True, default=generate_uuid)
    job_id = Column(String, ForeignKey("warming_queue.id", ondelete="CASCADE"), nullable=False)
    query = Column(Text, nullable=False)
    line_number = Column(Integer, nullable=False)
    error_message = Column(Text, nullable=True)
    error_type = Column(String, nullable=True)  # Exception class name
    retry_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class WarmingSSEEvent(Base):
    """SSE events ring buffer for cache warming progress replay."""

    __tablename__ = "warming_sse_events"
    __table_args__ = (
        Index("idx_sse_events_job", "job_id"),
        Index("idx_sse_events_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, unique=True, nullable=False)  # UUID for client tracking
    event_type = Column(String, nullable=False)  # 'progress', 'job_started', etc.
    job_id = Column(String, nullable=True)  # Nullable for heartbeats
    payload = Column(Text, nullable=False)  # JSON event data
    created_at = Column(DateTime, default=datetime.utcnow)
