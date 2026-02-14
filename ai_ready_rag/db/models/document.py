"""Document model."""

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
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

    # ingestkit-excel metadata (nullable - only set for Excel files processed via ingestkit)
    excel_file_type = Column(String, nullable=True)  # tabular_data | formatted_document | hybrid
    excel_ingest_key = Column(String, nullable=True)  # SHA-256 idempotency key
    excel_tables_created = Column(Integer, nullable=True)  # DB tables created (Path A)
    excel_classification_tier = Column(
        String, nullable=True
    )  # rule_based | llm_basic | llm_reasoning
    excel_db_table_names = Column(Text, nullable=True)  # JSON list of table names for cleanup

    tags = relationship("Tag", secondary=document_tags, back_populates="documents")
