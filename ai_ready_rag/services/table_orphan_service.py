"""TableOrphanService — detects and drops orphaned SQL tables.

Orphan = table in document_tables or excel_tables schema with no matching registry row.
Spec: UNIFIED_TABLE_INGEST_v1 §12

Note: Named TableOrphanService (not ReconciliationService) to avoid collision with
the existing reconciliation_service.py (PostgreSQL ↔ pgvector reconciliation).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TableOrphanService:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def find_orphans(self) -> list[dict]:
        """Find tables with no registry row in either document_tables or excel_tables schema."""
        try:
            import psycopg2

            conn = psycopg2.connect(self._database_url)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT t.table_name, 'document_tables' AS schema_name, 'no_registry_row' AS reason
                    FROM information_schema.tables t
                    LEFT JOIN document_table_registry r
                        ON r.table_name = t.table_name AND r.schema_name = 'document_tables'
                    WHERE t.table_schema = 'document_tables' AND r.id IS NULL
                    UNION ALL
                    SELECT t.table_name, 'excel_tables' AS schema_name, 'no_registry_row' AS reason
                    FROM information_schema.tables t
                    LEFT JOIN excel_table_registry r
                        ON r.table_name = t.table_name AND r.schema_name = 'excel_tables'
                    WHERE t.table_schema = 'excel_tables' AND r.id IS NULL
                """)
                rows = cur.fetchall()
            conn.close()
            return [{"table_name": r[0], "schema_name": r[1], "reason": r[2]} for r in rows]
        except Exception as exc:
            logger.warning("table_orphan.find_orphans_failed: %s", exc)
            return []

    def total_table_count(self) -> int:
        try:
            import psycopg2

            conn = psycopg2.connect(self._database_url)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema IN ('document_tables', 'excel_tables')
                """)
                count = cur.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0

    def drop_table(self, schema_name: str, table_name: str) -> None:
        """Drop a table. Uses psycopg2.sql.Identifier to prevent SQL injection."""
        import psycopg2
        from psycopg2 import sql

        conn = psycopg2.connect(self._database_url)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name),
                )
            )
        conn.commit()
        conn.close()
        logger.info("table_orphan.dropped: schema=%s table=%s", schema_name, table_name)
