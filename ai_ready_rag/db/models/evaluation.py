"""Evaluation framework models for RAG quality assessment."""

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
    Text,
    UniqueConstraint,
)

from ai_ready_rag.db.database import Base
from ai_ready_rag.db.models.base import generate_uuid


class EvaluationDataset(Base):
    """Named collection of Q&A pairs for repeatable evaluation."""

    __tablename__ = "evaluation_datasets"
    __table_args__ = (Index("idx_evaluation_datasets_source", "source_type"),)

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)
    source_type = Column(String, nullable=False)  # manual | ragbench | synthetic | live_sample
    source_config = Column(Text, nullable=True)  # JSON
    sample_count = Column(Integer, nullable=False, default=0)
    created_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatasetSample(Base):
    """Individual Q&A pair within a dataset."""

    __tablename__ = "dataset_samples"
    __table_args__ = (
        UniqueConstraint("dataset_id", "sort_order", name="uq_dataset_samples_dataset_sort"),
        Index("idx_dataset_samples_dataset", "dataset_id"),
    )

    id = Column(String, primary_key=True, default=generate_uuid)
    dataset_id = Column(
        String,
        ForeignKey("evaluation_datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    question = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=True)
    reference_contexts = Column(Text, nullable=True)  # JSON list
    metadata_ = Column("metadata", Text, nullable=True)  # JSON — "metadata" is reserved in SA
    sort_order = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class EvaluationRun(Base):
    """Tracks batch evaluation executions."""

    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index("idx_evaluation_runs_status", "status", "created_at"),
        Index("idx_evaluation_runs_dataset", "dataset_id"),
        Index("idx_evaluation_runs_lease", "worker_lease_expires_at"),
    )

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    dataset_id = Column(
        String,
        ForeignKey("evaluation_datasets.id", ondelete="RESTRICT"),
        nullable=False,
    )
    status = Column(String, nullable=False, default="pending")
    total_samples = Column(Integer, nullable=False, default=0)
    completed_samples = Column(Integer, nullable=False, default=0)
    failed_samples = Column(Integer, nullable=False, default=0)

    # Access control scope
    tag_scope = Column(Text, nullable=True)  # JSON list of tags
    admin_bypass_tags = Column(Boolean, nullable=False, default=False)

    # Aggregate scores
    avg_faithfulness = Column(Float, nullable=True)
    avg_answer_relevancy = Column(Float, nullable=True)
    avg_llm_context_precision = Column(Float, nullable=True)
    avg_llm_context_recall = Column(Float, nullable=True)
    invalid_score_count = Column(Integer, nullable=False, default=0)

    # Reproducibility snapshot
    model_used = Column(String, nullable=False)
    embedding_model_used = Column(String, nullable=False)
    config_snapshot = Column(Text, nullable=False)  # JSON

    # Worker lease fields
    worker_id = Column(String, nullable=True)
    worker_lease_expires_at = Column(DateTime, nullable=True)
    is_cancel_requested = Column(Boolean, nullable=False, default=False)

    # Capacity controls
    max_duration_hours = Column(Float, nullable=True)

    error_message = Column(Text, nullable=True)
    triggered_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class EvaluationSample(Base):
    """Per-question results within an evaluation run."""

    __tablename__ = "evaluation_samples"
    __table_args__ = (
        UniqueConstraint("run_id", "sort_order", name="uq_evaluation_samples_run_sort"),
        Index("idx_evaluation_samples_run", "run_id"),
        Index("idx_evaluation_samples_status", "run_id", "status"),
    )

    id = Column(String, primary_key=True, default=generate_uuid)
    run_id = Column(
        String,
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    sort_order = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="pending")

    # Input (from dataset)
    question = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=True)
    reference_contexts = Column(Text, nullable=True)  # JSON list

    # RAG output
    generated_answer = Column(Text, nullable=True)
    retrieved_contexts = Column(Text, nullable=True)  # JSON list

    # RAGAS metric scores
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    llm_context_precision = Column(Float, nullable=True)
    llm_context_recall = Column(Float, nullable=True)

    generation_time_ms = Column(Float, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    error_type = Column(String, nullable=True)
    processed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
