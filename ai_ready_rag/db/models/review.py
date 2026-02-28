"""SQLAlchemy model for the review queue.

The review queue collects items requiring human resolution before the platform
can respond confidently. Items are created by:
  - ClaudeEnrichmentService (canonicalization failures, low confidence entities)
  - QueryRouter (low confidence answers)
  - CADocumentClassifier (ambiguous document type)
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, String, Text

from ai_ready_rag.db.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class ReviewItem(Base):
    """Item requiring human review before automated processing can continue.

    review_type values:
        low_confidence_answer      — RAG answer below confidence threshold
        account_match_pending      — Account fuzzy match needs human confirmation
        canonicalization_failure   — Entity value could not be canonicalized
        unknown_document_type      — Classifier returned 'unknown' type
        ambiguous_classification   — Top-two classifier scores within ambiguity threshold
    """

    __tablename__ = "review_items"

    id = Column(String, primary_key=True, default=generate_uuid)
    review_type = Column(String, nullable=False, index=True)
    # Context for the reviewer
    query = Column(Text, nullable=True)
    tentative_answer = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    # Classification ambiguity fields
    candidate_types = Column(Text, nullable=True)  # JSON array: ["board_minutes", "reserve_study"]
    candidate_scores = Column(Text, nullable=True)  # JSON array: [0.78, 0.73]
    # Review outcome
    review_status = Column(String, default="pending", index=True)
    # values: pending|accepted|corrected|dismissed
    corrected_answer = Column(Text, nullable=True)
    reviewer_id = Column(
        String,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Module / tenant context
    module_context = Column(String, nullable=True)  # e.g., "community_associations"
    tenant_id = Column(String, nullable=True, index=True)
    # Source document link (optional)
    document_id = Column(
        String,
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime, nullable=True)
    # Soft delete
    is_deleted = Column(Boolean, default=False)
