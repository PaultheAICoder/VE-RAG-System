"""Enrichment models — synopsis, entity, and review queue tables."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from ai_ready_rag.db.database import Base
from ai_ready_rag.db.models.base import generate_uuid


class EnrichmentSynopsis(Base):
    __tablename__ = "enrichment_synopses"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    synopsis_text = Column(Text, nullable=False)
    model_id = Column(String, nullable=False)  # e.g. "claude-sonnet-4-6"
    prompt_version = Column(String, nullable=True)  # semver of the prompt template
    token_cost = Column(Integer, nullable=True)  # input + output tokens
    cost_usd = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    entities = relationship(
        "EnrichmentEntity", back_populates="synopsis", cascade="all, delete-orphan"
    )
    document = relationship(
        "Document",
        foreign_keys=[document_id],
        primaryjoin="EnrichmentSynopsis.document_id == Document.id",
        viewonly=True,
    )


class EnrichmentEntity(Base):
    __tablename__ = "enrichment_entities"

    id = Column(String, primary_key=True, default=generate_uuid)
    synopsis_id = Column(
        String,
        ForeignKey("enrichment_synopses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entity_type = Column(String, nullable=False)  # e.g. "insurance_carrier", "coverage_line"
    value = Column(String, nullable=False)  # raw extracted value
    canonical_value = Column(String, nullable=True)  # normalized via alias resolution
    confidence = Column(Float, nullable=True)  # 0.0–1.0
    source_chunk_index = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    synopsis = relationship("EnrichmentSynopsis", back_populates="entities")


class ReviewItem(Base):
    __tablename__ = "review_items"

    id = Column(String, primary_key=True, default=generate_uuid)
    query_id = Column(String, nullable=True, index=True)  # chat message ID if applicable
    answer_text = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    reason = Column(String, nullable=True)  # why it was routed to review
    status = Column(String, default="pending")  # pending | approved | rejected
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
