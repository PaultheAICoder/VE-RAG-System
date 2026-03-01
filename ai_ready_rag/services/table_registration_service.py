"""TableRegistrationService — discovers and registers SQL templates from both registries.

Reads document_table_registry (new) and excel_table_registry (legacy) at startup.
document_table_registry takes precedence on conflict (spec §7.4).
All tables get NL2SQL routing via column_signals (no P&L special-casing).

Spec: UNIFIED_TABLE_INGEST_v1 §7.1, §7.4
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class TableRegistrationService:
    """Discovers tables from both registries and registers SQL templates.

    Called once at startup by main.py (lifespan). Safe to call again (idempotent).
    """

    def __init__(self, database_url: str, tenant_id: str = "default") -> None:
        self._database_url = database_url
        self._tenant_id = tenant_id

    def discover_and_register_all(self) -> int:
        """Query both registries and register all tables as NL2SQL SQL templates.

        document_table_registry rows win over excel_table_registry on conflict.
        All registered tables use column_signals path (no trigger phrases).

        Returns:
            Total number of SQL templates registered.
        """
        rows = self._fetch_all_rows()
        if not rows:
            logger.info("table_registration_service: no tables found in any registry")
            return 0

        registered = 0
        for row in rows:
            try:
                self._register_template(row)
                registered += 1
            except Exception as exc:
                logger.warning(
                    "table_registration_service.register_failed: table=%s error=%s",
                    row.get("table_name"),
                    exc,
                )

        logger.info(
            "table_registration_service: registered %d templates (tenant=%s)",
            registered,
            self._tenant_id,
        )
        return registered

    def _fetch_all_rows(self) -> list[dict]:
        """Fetch rows from both registries, with document_table_registry winning on conflict."""
        try:
            import psycopg2

            conn = psycopg2.connect(self._database_url)
            rows_by_key: dict[tuple, dict] = {}

            with conn.cursor() as cur:
                # 1. Load excel_table_registry first (lower priority)
                try:
                    cur.execute(
                        "SELECT table_name, schema_name, columns, column_types, table_metadata "
                        "FROM excel_table_registry ORDER BY table_name"
                    )
                    for row in cur.fetchall():
                        table_name, schema_name, columns, column_types, table_metadata_raw = row
                        try:
                            meta = json.loads(table_metadata_raw) if table_metadata_raw else {}
                        except (json.JSONDecodeError, TypeError):
                            meta = {}
                        key = (self._tenant_id, schema_name or "excel_tables", table_name)
                        rows_by_key[key] = {
                            "table_name": table_name,
                            "schema_name": schema_name or "excel_tables",
                            "columns": columns,
                            "column_types": column_types,
                            "row_value_samples": None,
                            "access_tags": meta.get("access_tags", []),
                            "source": "excel_table_registry",
                        }
                except Exception as exc:
                    logger.debug(
                        "table_registration_service: excel_table_registry unavailable: %s", exc
                    )

                # 2. Load document_table_registry (higher priority — overwrites on conflict)
                try:
                    cur.execute(
                        "SELECT table_name, schema_name, columns, column_types, "
                        "row_value_samples, table_metadata "
                        "FROM document_table_registry WHERE tenant_id = %s ORDER BY table_name",
                        (self._tenant_id,),
                    )
                    for row in cur.fetchall():
                        table_name, schema_name, columns, column_types, row_samples, meta_raw = row
                        try:
                            meta = json.loads(meta_raw) if meta_raw else {}
                        except (json.JSONDecodeError, TypeError):
                            meta = {}
                        key = (self._tenant_id, schema_name, table_name)
                        if key in rows_by_key:
                            logger.warning(
                                "table_registration_service.dual_registry_conflict: "
                                "table=%s schema=%s — document_table_registry wins",
                                table_name,
                                schema_name,
                            )
                        rows_by_key[key] = {
                            "table_name": table_name,
                            "schema_name": schema_name,
                            "columns": columns,
                            "column_types": column_types,
                            "row_value_samples": row_samples,
                            "access_tags": meta.get("access_tags", []),
                            "source": "document_table_registry",
                        }
                except Exception as exc:
                    logger.debug(
                        "table_registration_service: document_table_registry unavailable: %s", exc
                    )

            conn.close()
            return list(rows_by_key.values())

        except Exception as exc:
            logger.warning("table_registration_service.fetch_failed: %s", exc)
            return []

    def _register_template(self, row: dict) -> None:
        """Register a single NL2SQL template from a registry row."""
        from ai_ready_rag.modules.registry import SQLTemplate, get_registry
        from ai_ready_rag.services.excel_tables_service import ExcelTablesService

        table_name = row["table_name"]
        schema_name = row["schema_name"]
        access_tags = row.get("access_tags") or []

        try:
            col_list = json.loads(row["columns"]) if row["columns"] else []
        except (json.JSONDecodeError, TypeError):
            col_list = []

        try:
            col_types = json.loads(row["column_types"]) if row["column_types"] else {}
        except (json.JSONDecodeError, TypeError):
            col_types = {}

        # Build column_signals using existing ExcelTablesService._compute_column_signals
        # (reuses the existing __table__ sentinel and __quantitative__ signals logic)
        ets = ExcelTablesService.__new__(ExcelTablesService)
        column_signals = ets._compute_column_signals(table_name, col_list, col_types)

        # Build fallback SQL
        quoted_cols = ", ".join(f'"{c}"' for c in col_list[:10]) if col_list else "*"
        sql = f'SELECT {quoted_cols} FROM "{schema_name}"."{table_name}" LIMIT :row_cap'

        template_name = f"excel_{table_name}"
        template = SQLTemplate(
            name=template_name,
            sql=sql,
            trigger_phrases=[],
            description=f"Table {schema_name}.{table_name} — NL2SQL via column signals",
            column_signals=column_signals,
            access_tags=access_tags,
        )

        registry = get_registry()
        registry.register_sql_templates("core.unified_tables", {template_name: template})
        logger.debug(
            "table_registration_service.registered: name=%s schema=%s columns=%d tags=%s",
            template_name,
            schema_name,
            len(col_list),
            access_tags,
        )
