"""Document model."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from ai_ready_rag.db.database import Base
from ai_ready_rag.db.models.base import document_tags, generate_uuid


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
    uploaded_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    # Extended fields (spec v1.2)
    title = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    content_hash = Column(String, nullable=True, index=True)

    # ingestkit-forms fields (all nullable — only populated on forms extraction)
    forms_template_id = Column(String, nullable=True)
    forms_template_name = Column(String, nullable=True)
    forms_template_version = Column(Integer, nullable=True)
    forms_overall_confidence = Column(Float, nullable=True)
    forms_extraction_method = Column(
        String, nullable=True
    )  # native_fields|ocr_overlay|cell_mapping
    forms_match_method = Column(String, nullable=True)  # auto_detect|manual_override
    forms_ingest_key = Column(String, nullable=True, index=True)
    forms_db_table_names = Column(Text, nullable=True)  # JSON array of table names

    # Auto-tagging fields (all nullable — only populated when auto-tagging is enabled)
    auto_tag_status = Column(String, nullable=True)  # null|pending|completed|partial|failed
    auto_tag_strategy = Column(String, nullable=True)  # Strategy ID used
    auto_tag_version = Column(String, nullable=True)  # Strategy version used
    auto_tag_source = Column(Text, nullable=True)  # JSON provenance
    source_path = Column(String, nullable=True)  # Original folder path from upload

    tags = relationship("Tag", secondary=document_tags, back_populates="documents")
