"""SQLAlchemy model for the document_table_registry table.

Stores schema metadata and row value samples for tables from all source formats
(excel, pdf, docx, csv, image). Supersedes excel_table_registry for new ingest.

Spec: UNIFIED_TABLE_INGEST_v1 §8.1
"""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Index, Integer, String, Text

from ai_ready_rag.db.database import Base
from ai_ready_rag.db.models.base import generate_uuid


class DocumentTableRegistry(Base):
    __tablename__ = "document_table_registry"

    id = Column(String, primary_key=True, default=generate_uuid)
    tenant_id = Column(String, nullable=False, default="default")
    table_name = Column(String, nullable=False)
    schema_name = Column(String, nullable=False, default="document_tables")
    source_format = Column(String, nullable=False)
    source_page = Column(Integer, nullable=True)
    table_index = Column(Integer, nullable=True, default=0)
    columns = Column(Text, nullable=False)
    column_types = Column(Text, nullable=True)
    row_value_samples = Column(Text, nullable=True)
    document_id = Column(String, nullable=False)
    document_name = Column(String, nullable=True)
    row_count = Column(Integer, nullable=True)
    table_metadata = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_dtr_tenant_id", "tenant_id"),
        Index("ix_dtr_document_id", "document_id"),
        Index("ix_dtr_source_format", "source_format"),
        Index(
            "ix_dtr_tenant_schema_table",
            "tenant_id",
            "schema_name",
            "table_name",
            unique=True,
        ),
    )
