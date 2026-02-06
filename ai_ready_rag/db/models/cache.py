"""Cache and warming queue models."""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text

from ai_ready_rag.db.database import Base
from ai_ready_rag.db.models.base import TimestampMixin, generate_uuid


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


class EmbeddingCache(TimestampMixin, Base):
    """Cached query embeddings to skip Ollama embed calls."""

    __tablename__ = "embedding_cache"

    query_hash = Column(String, primary_key=True)
    query_text = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)  # JSON-encoded float array


class CacheAccessLog(Base):
    """Tracks cache hit/miss for query frequency analysis."""

    __tablename__ = "cache_access_log"

    id = Column(String, primary_key=True, default=generate_uuid)
    query_hash = Column(String, nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    was_hit = Column(Boolean, nullable=False)
    accessed_at = Column(DateTime, default=datetime.utcnow, index=True)


class WarmingQueue(TimestampMixin, Base):
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
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)


class WarmingFailedQuery(TimestampMixin, Base):
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


class WarmingSSEEvent(TimestampMixin, Base):
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
