"""Cache models."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

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


class WarmingSSEEvent(TimestampMixin, Base):
    """SSE events ring buffer for cache warming progress replay."""

    __tablename__ = "warming_sse_events"
    __table_args__ = (
        Index("idx_sse_events_job", "job_id"),
        Index("idx_sse_events_created", "created_at"),
        UniqueConstraint("job_id", "batch_seq", name="uq_sse_events_job_batch_seq"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, nullable=False)  # str(batch_seq) for job-scoped, UUID for global
    event_type = Column(String, nullable=False)  # 'progress', 'job_started', etc.
    job_id = Column(String, nullable=True)  # Nullable for heartbeats
    batch_seq = Column(Integer, nullable=True)  # Per-batch monotonic sequence for replay
    payload = Column(Text, nullable=False)  # JSON event data
