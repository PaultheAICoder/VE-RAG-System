"""Warming batch and query models for the redesigned warming queue."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)

from ai_ready_rag.db.database import Base
from ai_ready_rag.db.models.base import generate_uuid


class WarmingBatch(Base):
    """A batch of warming queries submitted together.

    Tracks overall batch progress, worker lease ownership,
    and pause/cancel state for graceful lifecycle management.
    """

    __tablename__ = "warming_batches"
    __table_args__ = (
        Index("idx_warming_batches_status", "status", "created_at"),
        Index("idx_warming_batches_lease", "worker_lease_expires_at"),
        Index("idx_warming_batches_cleanup", "status", "completed_at"),
    )

    id = Column(String, primary_key=True, default=generate_uuid)
    source_type = Column(String, nullable=False)  # "manual" | "upload"
    original_filename = Column(String, nullable=True)  # For file uploads
    total_queries = Column(Integer, nullable=False)
    status = Column(String, nullable=False, default="pending")
    is_paused = Column(Boolean, nullable=False, default=False)
    is_cancel_requested = Column(Boolean, nullable=False, default=False)
    worker_id = Column(String, nullable=True)
    worker_lease_expires_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    submitted_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class WarmingQuery(Base):
    """Individual query within a warming batch.

    Each query is independently claimable for idempotent processing.
    sort_order determines processing sequence within the batch.
    """

    __tablename__ = "warming_queries"
    __table_args__ = (
        UniqueConstraint("batch_id", "sort_order", name="uq_warming_queries_batch_sort"),
        Index("idx_warming_queries_batch", "batch_id"),
        Index("idx_warming_queries_status", "status", "created_at"),
        Index(
            "idx_warming_queries_pending",
            "batch_id",
            "status",
            sqlite_where=text("status = 'pending'"),
        ),
        Index("idx_warming_queries_cleanup", "status", "processed_at"),
    )

    id = Column(String, primary_key=True, default=generate_uuid)
    batch_id = Column(
        String,
        ForeignKey("warming_batches.id", ondelete="CASCADE"),
        nullable=False,
    )
    query_text = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="pending")
    error_message = Column(Text, nullable=True)
    error_type = Column(String, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    sort_order = Column(Integer, nullable=False, default=0)
    submitted_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    processed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
