"""TagSuggestion model for auto-tagging approval workflow."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, String

from ai_ready_rag.db.database import Base
from ai_ready_rag.db.models.base import generate_uuid


class TagSuggestion(Base):
    __tablename__ = "tag_suggestions"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tag_name = Column(String, nullable=False)
    display_name = Column(String, nullable=False)
    namespace = Column(String, nullable=False)
    source = Column(String, nullable=False)
    confidence = Column(Float, default=1.0)
    strategy_id = Column(String, nullable=False)
    status = Column(String, default="pending")
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_tag_suggestions_status", "status"),)
