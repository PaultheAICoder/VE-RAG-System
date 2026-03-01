"""TableExtractionAdapter — extracts tables from Docling output and persists them to
document_table_registry + document_tables PostgreSQL schema.

Called from ProcessingService after Docling chunking for PDF/image/DOCX documents.
Spec: UNIFIED_TABLE_INGEST_v1 §5.1, §5.2, §5.3
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_ready_rag.db.models.document import Document

logger = logging.getLogger(__name__)


class TableExtractionAdapter:
    """Extracts TableItem objects from a DoclingDocument and persists them.

    For each table in the Docling output:
    1. Export to DataFrame
    2. Filter: skip if < 2 rows or < 2 columns
    3. Apply header heuristic for PDF tables
    4. Generate stable table name via make_table_name()
    5. Write to document_tables PostgreSQL schema
    6. Sample row values and write to document_table_registry
    """

    def __init__(self, database_url: str, settings: Any) -> None:
        self._database_url = database_url
        self._settings = settings

    def extract_and_persist(
        self,
        docling_document: Any,
        document: Document,
        source_format: str,
        access_tags: list[str] | None = None,
        tenant_id: str = "default",
    ) -> list[str]:
        """Extract all tables from a DoclingDocument and persist to PostgreSQL.

        Args:
            docling_document: Docling ConversionResult or DoclingDocument with .tables attribute
            document: The Document ORM object (for doc_id, original_filename)
            source_format: 'pdf', 'image', 'docx', etc.
            access_tags: Tags required to access these tables (empty = public)
            tenant_id: Tenant scope

        Returns:
            List of registered table names (empty if no tables found or extraction fails).
        """
        from ai_ready_rag.utils.signal_canon import make_table_name, sample_row_values

        if docling_document is None:
            return []

        # Get tables list — Docling exposes tables via .document.tables or .tables
        tables = []
        if hasattr(docling_document, "document") and hasattr(docling_document.document, "tables"):
            tables = docling_document.document.tables
        elif hasattr(docling_document, "tables"):
            tables = docling_document.tables

        if not tables:
            logger.debug("table_extraction: no tables found in document=%s", document.id)
            return []

        registered: list[str] = []
        doc_stem = document.original_filename or document.id
        # Strip extension from stem
        if "." in doc_stem:
            doc_stem = doc_stem.rsplit(".", 1)[0]

        for idx, table_item in enumerate(tables):
            try:
                df = self._export_table(table_item)
                if df is None:
                    continue

                # Minimum size gate
                if len(df) < 2 or len(df.columns) < 2:
                    logger.debug(
                        "table_extraction.skip_min_size: doc=%s idx=%d rows=%d cols=%d",
                        document.id,
                        idx,
                        len(df),
                        len(df.columns),
                    )
                    continue

                # Apply header heuristic for non-Excel formats
                if source_format in ("pdf", "image", "docx"):
                    df = self._apply_header_heuristic(df)

                # Re-check size after header promotion (may reduce row count)
                if len(df) < 2 or len(df.columns) < 2:
                    logger.debug(
                        "table_extraction.skip_after_header: doc=%s idx=%d rows=%d cols=%d",
                        document.id,
                        idx,
                        len(df),
                        len(df.columns),
                    )
                    continue

                # Generate stable table name
                table_name = make_table_name(doc_stem, document.id, idx)

                # Write table to document_tables schema
                self._write_table(df, table_name)

                # Sample row values and register
                samples = sample_row_values(df)
                self._write_registry(
                    table_name=table_name,
                    document=document,
                    df=df,
                    source_format=source_format,
                    table_idx=idx,
                    row_value_samples=samples,
                    access_tags=access_tags or [],
                    tenant_id=tenant_id,
                )

                registered.append(table_name)
                logger.info(
                    "table_extraction.registered: doc=%s table=%s idx=%d rows=%d cols=%d",
                    document.id,
                    table_name,
                    idx,
                    len(df),
                    len(df.columns),
                )

            except Exception as exc:
                logger.warning(
                    "table_extraction.failed: doc=%s idx=%d error=%s",
                    document.id,
                    idx,
                    exc,
                )
                continue

        return registered

    def _export_table(self, table_item: Any):
        """Export a Docling TableItem to a pandas DataFrame. Returns None on failure."""
        try:
            import pandas as pd

            if hasattr(table_item, "export_to_dataframe"):
                return table_item.export_to_dataframe()
            elif hasattr(table_item, "to_dataframe"):
                return table_item.to_dataframe()
            # Fallback: try to get data attribute
            elif hasattr(table_item, "data"):
                return pd.DataFrame(table_item.data)
        except Exception as exc:
            logger.debug("table_extraction.export_failed: %s", exc)
        return None

    def _apply_header_heuristic(self, df):
        """If first row appears to be a header (all-string, rest has numeric), promote it."""
        import pandas as pd

        if len(df) < 2:
            return df
        first_row = df.iloc[0]
        # Check if first row is all strings and second row has at least one numeric value
        first_all_str = all(
            isinstance(v, str) or (not pd.isna(v) and str(v).strip()) for v in first_row
        )
        second_has_num = any(
            str(v).replace(".", "").replace("-", "").replace(",", "").isdigit()
            for v in df.iloc[1]
            if v is not None and not (isinstance(v, float) and pd.isna(v))
        )
        if first_all_str and second_has_num:
            new_cols = [str(v).strip() for v in first_row]
            df = df.iloc[1:].copy()
            df.columns = new_cols
            df = df.reset_index(drop=True)
        return df

    def _write_table(self, df, table_name: str) -> None:
        """Write DataFrame to document_tables PostgreSQL schema."""
        try:
            import psycopg2

            conn = psycopg2.connect(self._database_url)
            with conn.cursor() as cur:
                # Ensure schema exists
                cur.execute("CREATE SCHEMA IF NOT EXISTS document_tables")
                # Drop and recreate table
                cur.execute(f'DROP TABLE IF EXISTS document_tables."{table_name}"')
                cols_ddl = ", ".join(f'"{col}" TEXT' for col in df.columns)
                cur.execute(f'CREATE TABLE document_tables."{table_name}" ({cols_ddl})')
                # Insert rows
                for _, row in df.iterrows():
                    placeholders = ", ".join(["%s"] * len(df.columns))
                    quoted_cols = ", ".join(f'"{c}"' for c in df.columns)
                    cur.execute(
                        f'INSERT INTO document_tables."{table_name}"'
                        f" ({quoted_cols}) VALUES ({placeholders})",
                        [str(v) if v is not None else None for v in row],
                    )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning(
                "table_extraction.write_table_failed: table=%s error=%s", table_name, exc
            )
            raise

    def _write_registry(
        self,
        table_name: str,
        document: Document,
        df,
        source_format: str,
        table_idx: int,
        row_value_samples: dict,
        access_tags: list[str],
        tenant_id: str,
    ) -> None:
        """Write metadata to document_table_registry (upsert)."""
        try:
            import psycopg2

            from ai_ready_rag.db.models.base import generate_uuid

            conn = psycopg2.connect(self._database_url)
            with conn.cursor() as cur:
                meta = {"access_tags": access_tags}
                cur.execute(
                    """
                    INSERT INTO document_table_registry
                        (id, tenant_id, table_name, schema_name, source_format,
                         table_index, columns, column_types, row_value_samples,
                         document_id, document_name, row_count, table_metadata,
                         created_at, updated_at)
                    VALUES (%s,%s,%s,'document_tables',%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),NOW())
                    ON CONFLICT (tenant_id, schema_name, table_name)
                    DO UPDATE SET
                        columns = EXCLUDED.columns,
                        row_value_samples = EXCLUDED.row_value_samples,
                        document_id = EXCLUDED.document_id,
                        document_name = EXCLUDED.document_name,
                        row_count = EXCLUDED.row_count,
                        table_metadata = EXCLUDED.table_metadata,
                        updated_at = NOW()
                    """,
                    (
                        generate_uuid(),
                        tenant_id,
                        table_name,
                        source_format,
                        table_idx,
                        json.dumps(list(df.columns)),
                        json.dumps({}),
                        json.dumps(row_value_samples),
                        document.id,
                        document.original_filename,
                        len(df),
                        json.dumps(meta),
                    ),
                )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning(
                "table_extraction.write_registry_failed: table=%s error=%s", table_name, exc
            )
            raise
