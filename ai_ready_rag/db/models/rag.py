"""RAG feature models - synonyms and curated Q&A."""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship

from ai_ready_rag.db.database import Base
from ai_ready_rag.db.models.base import TimestampMixin, generate_uuid


class QuerySynonym(TimestampMixin, Base):
    """Synonym mappings for query expansion."""

    __tablename__ = "query_synonyms"
    __table_args__ = (
        Index("idx_query_synonyms_term", "term"),
        Index("idx_query_synonyms_enabled", "enabled"),
    )

    id = Column(String, primary_key=True, default=generate_uuid)
    term = Column(String, nullable=False)  # Source term (single word/phrase)
    synonyms = Column(Text, nullable=False)  # JSON array of synonyms
    enabled = Column(Boolean, default=True)
    created_by = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CuratedQA(TimestampMixin, Base):
    """Admin-curated Q&A pairs for direct answers."""

    __tablename__ = "curated_qa"
    __table_args__ = (Index("idx_curated_qa_priority", "priority", "enabled"),)

    id = Column(String, primary_key=True, default=generate_uuid)
    keywords = Column(Text, nullable=False)  # JSON array of keywords
    answer = Column(Text, nullable=False)  # Sanitized HTML from WYSIWYG
    source_reference = Column(String, nullable=False)  # Required for compliance
    confidence = Column(Integer, default=85)  # Confidence to return (0-100)
    priority = Column(Integer, default=0)  # Higher = checked first
    enabled = Column(Boolean, default=True)
    access_count = Column(Integer, default=0)
    last_accessed_at = Column(DateTime, nullable=True)
    created_by = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to keyword lookup table
    keyword_entries = relationship(
        "CuratedQAKeyword", back_populates="qa", cascade="all, delete-orphan"
    )


class CuratedQAKeyword(Base):
    """Normalized keywords for efficient Q&A matching."""

    __tablename__ = "curated_qa_keywords"
    __table_args__ = (
        Index("idx_curated_qa_keywords_keyword", "keyword"),
        Index("idx_curated_qa_keywords_qa_id", "qa_id"),
    )

    id = Column(String, primary_key=True, default=generate_uuid)
    qa_id = Column(String, ForeignKey("curated_qa.id", ondelete="CASCADE"), nullable=False)
    keyword = Column(String, nullable=False)  # Normalized token (lowercase)
    original_keyword = Column(String, nullable=False)  # Original multi-word keyword

    qa = relationship("CuratedQA", back_populates="keyword_entries")
