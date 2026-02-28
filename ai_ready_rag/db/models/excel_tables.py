"""SQLAlchemy model for the excel_table_registry table.

Stores schema metadata for tables written to the excel_tables schema by
ingestkit-excel, enabling schema-aware NL2SQL routing.
"""

from sqlalchemy import Column, DateTime, Index, Integer, String, Text

from ai_ready_rag.db.database import Base
from ai_ready_rag.db.models.base import generate_uuid


class ExcelTableRegistry(Base):
    """Registry of tables written to the excel_tables schema.

    Each row represents one table written by ingestkit-excel. The columns/
    column_types fields carry the DataFrame schema as JSON, used by
    ExcelTablesService to build column_signals for schema-aware routing.
    """

    __tablename__ = "excel_table_registry"

    id = Column(String, primary_key=True, default=generate_uuid)
    table_name = Column(String, nullable=False)
    schema_name = Column(String, nullable=False, default="excel_tables")
    columns = Column(Text, nullable=False)  # JSON list of column names
    column_types = Column(Text, nullable=True)  # JSON dict of col -> dtype string
    document_name = Column(String, nullable=True)
    document_id = Column(String, nullable=True)
    tenant_id = Column(String, nullable=False, default="default")
    row_count = Column(Integer, nullable=True)
    table_metadata = Column(Text, nullable=True)  # JSON extras (sheet name, etc.)
    created_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_excel_table_registry_table_name", "table_name"),
        Index("ix_excel_table_registry_tenant_id", "tenant_id"),
        Index(
            "ix_excel_table_registry_schema_table",
            "schema_name",
            "table_name",
            unique=True,
        ),
    )
