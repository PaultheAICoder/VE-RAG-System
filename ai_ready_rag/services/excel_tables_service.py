"""ExcelTablesService — registers SQL templates for excel_tables schema at startup.

Queries the excel_table_registry table (written by ingestkit-excel at ingest time)
and registers:
  - One UNION ALL SQLTemplate for P&L tables (backward-compatible, trigger-phrase based)
  - One per-table SQLTemplate with column_signals for all other tables (NL2SQL path)

The QueryRouter scores column_signals for non-P&L queries; _execute_nl2sql_route in
RAGService then generates and executes the SQL via Claude.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# P&L trigger phrases (kept for backward compatibility with financial tables)
# ---------------------------------------------------------------------------
_PL_TRIGGER_PHRASES = [
    "cogs",
    "cost of goods",
    "cost of goods sold",
    "revenue",
    "total revenue",
    "gross profit",
    "operating income",
    "operating expenses",
    "net income",
    "profit and loss",
    "p&l",
    "income statement",
    "profit & loss",
    "financials",
    "financial statement",
    "ebitda",
    "materials",
    "direct labor",
    "manufacturing overhead",
    "income before taxes",
    "interest expense",
]

# Column names that strongly indicate a P&L financial table
_PL_COLUMN_INDICATORS = {
    "revenue",
    "cogs",
    "net income",
    "gross profit",
    "operating income",
    "operating expenses",
    "ebitda",
    "item",
    "value",
    "year",
    "income before taxes",
    "interest expense",
    "materials",
    "direct labor",
    "manufacturing overhead",
}

_PL_TEMPLATE_NAME = "excel_pl_financials"

# Quantitative signals that indicate a user wants a SQL-answerable query
QUANTITATIVE_SIGNALS = [
    "how many",
    "how much",
    "total",
    "sum",
    "count",
    "average",
    "on hand",
    "balance",
    "amount due",
    "list all",
    "show me all",
    "give me",
    "quantity",
    "in stock",
    "remaining",
    "what is the",
    "show me",
    "outstanding",
    "aged",
]


# ---------------------------------------------------------------------------
# SQL injection guard
# ---------------------------------------------------------------------------


class SqlInjectionGuard:
    """Validates that Claude-generated SQL is a plain SELECT with no side effects."""

    _FORBIDDEN = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|MERGE|EXEC|EXECUTE|GRANT|REVOKE)\b",
        re.IGNORECASE,
    )

    def validate(self, sql: str) -> None:
        """Raise ValueError if sql is not a safe SELECT statement."""
        stripped = sql.strip()
        if not re.search(r"^\s*SELECT\b", stripped, re.IGNORECASE):
            raise ValueError(
                f"SQL injection guard: only SELECT statements are allowed. Got: {stripped[:60]}"
            )
        if self._FORBIDDEN.search(stripped):
            raise ValueError(
                f"SQL injection guard: forbidden statement in generated SQL: {stripped[:80]}"
            )


# ---------------------------------------------------------------------------
# ExcelTablesService
# ---------------------------------------------------------------------------


class ExcelTablesService:
    """Discovers excel_tables schema tables and registers SQL templates.

    Called once at startup. Safe to call again (idempotent — re-registers
    with current table list, replacing any prior registration).
    """

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def discover_and_register(self) -> int:
        """Discover tables from excel_table_registry and register SQL templates.

        Queries the excel_table_registry table (populated by ingestkit-excel at
        ingest time) to get the current set of tables and their schemas.

        Returns:
            Total number of SQL templates registered (P&L union + per-table).
        """
        return self.discover_and_register_all()

    def discover_and_register_all(self) -> int:
        """Query excel_table_registry and register all tables as SQL templates.

        - P&L tables (detected by column overlap with financial vocabulary) are
          registered as a single UNION ALL template with trigger phrases.
        - Non-P&L tables are registered one-per-table with column_signals for
          the NL2SQL path.

        Returns:
            Total number of SQL templates registered.
        """
        registry_rows = self._discover_registry_tables()

        if not registry_rows:
            # Fall back to the old information_schema discovery for P&L only
            logger.info("excel_tables.discover_all: registry empty — trying legacy discovery")
            pl_tables = self._discover_year_tables_legacy()
            if not pl_tables:
                logger.info("excel_tables.discover_all: no tables found — skipping registration")
                return 0
            sql = self._build_union_sql(pl_tables)
            self._register_pl_template(sql, pl_tables)
            return 1

        pl_tables: list[str] = []
        non_pl_rows: list[dict] = []

        for row in registry_rows:
            table_name = row["table_name"]
            schema_name = row.get("schema_name", "excel_tables")
            try:
                col_list = json.loads(row["columns"])
            except (json.JSONDecodeError, KeyError):
                col_list = []

            try:
                col_types = json.loads(row.get("column_types") or "{}")
            except json.JSONDecodeError:
                col_types = {}

            access_tags = row.get("access_tags", [])
            # When column names are generic numeric indices ("0", "1"), sample
            # actual row values to detect P&L tables from header-less workbooks.
            sample_values: list[str] | None = None
            if col_list and all(c.isdigit() for c in col_list):
                sample_values = self._sample_first_column(table_name, schema_name, col_list[0])
            if self._is_pl_table(col_list, sample_values=sample_values):
                pl_tables.append(table_name)
            else:
                non_pl_rows.append(
                    {
                        "table_name": table_name,
                        "schema_name": schema_name,
                        "columns": col_list,
                        "column_types": col_types,
                        "access_tags": access_tags,
                    }
                )

        registered = 0

        # Register P&L union template (backward compat)
        if pl_tables:
            sql = self._build_union_sql(pl_tables)
            self._register_pl_template(sql, pl_tables)
            registered += 1

        # Register one template per non-P&L table with column_signals
        for row in non_pl_rows:
            self._register_table_template(
                table_name=row["table_name"],
                schema_name=row["schema_name"],
                columns=row["columns"],
                column_types=row["column_types"],
                access_tags=row.get("access_tags", []),
            )
            registered += 1

        logger.info(
            "excel_tables.discover_all: registered %d templates (%d P&L, %d per-table)",
            registered,
            1 if pl_tables else 0,
            len(non_pl_rows),
        )
        return registered

    # -----------------------------------------------------------------------
    # Discovery helpers
    # -----------------------------------------------------------------------

    def _discover_registry_tables(self) -> list[dict]:
        """Query excel_table_registry for all registered tables.

        Returns:
            List of dicts with table_name, schema_name, columns, column_types keys.
            Returns [] on exception (table may not exist yet).
        """
        try:
            import psycopg2

            conn = psycopg2.connect(self._database_url)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT table_name, schema_name, columns, column_types, table_metadata "
                    "FROM excel_table_registry "
                    "ORDER BY table_name"
                )
                rows = cur.fetchall()
            conn.close()
            result = []
            for row in rows:
                table_metadata_raw = row[4]
                try:
                    table_metadata = json.loads(table_metadata_raw) if table_metadata_raw else {}
                except (json.JSONDecodeError, TypeError):
                    table_metadata = {}
                result.append(
                    {
                        "table_name": row[0],
                        "schema_name": row[1],
                        "columns": row[2],
                        "column_types": row[3],
                        "access_tags": table_metadata.get("access_tags", []),
                    }
                )
            return result
        except Exception as exc:
            logger.warning("excel_tables.registry_query.failed: %s", exc)
            return []

    def _discover_year_tables_legacy(self) -> list[str]:
        """Legacy: query information_schema for tables in excel_tables schema.

        Used as fallback when excel_table_registry does not exist yet.
        """
        try:
            import psycopg2

            conn = psycopg2.connect(self._database_url)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'excel_tables' "
                    "ORDER BY table_name"
                )
                tables = [row[0] for row in cur.fetchall()]
            conn.close()
            return tables
        except Exception as exc:
            logger.warning("excel_tables.legacy_discovery.failed: %s", exc)
            return []

    # -----------------------------------------------------------------------
    # Classification helpers
    # -----------------------------------------------------------------------

    def _is_pl_table(self, col_list: list[str], sample_values: list[str] | None = None) -> bool:
        """Return True if the column list looks like a P&L financial table.

        Falls back to checking sampled row values when column names are generic
        (e.g. "0", "1" from header-less workbooks like PL_Statements).
        """
        col_lower = {c.lower() for c in col_list}
        overlap = col_lower & _PL_COLUMN_INDICATORS
        if len(overlap) >= 2:
            return True
        if sample_values:
            val_lower = {v.lower() for v in sample_values}
            return len(val_lower & _PL_COLUMN_INDICATORS) >= 2
        return False

    def _sample_first_column(self, table_name: str, schema_name: str, first_col: str) -> list[str]:
        """Return up to 30 non-null string values from the first column of a table.

        Used as a fallback for P&L detection when column names are generic.
        Returns [] on any error.
        """
        try:
            import psycopg2

            conn = psycopg2.connect(self._database_url)
            with conn.cursor() as cur:
                cur.execute(f'SELECT "{first_col}" FROM {schema_name}."{table_name}" LIMIT 30')
                rows = cur.fetchall()
            conn.close()
            return [str(r[0]) for r in rows if r[0] is not None]
        except Exception:
            return []

    def _compute_column_signals(
        self, table_name: str, columns: list[str], column_types: dict
    ) -> dict[str, list[str]]:
        """Build a column_signals dict for the given table.

        The signals dict has:
        - One key per column name with synonyms derived from the column name
        - A "__quantitative__" key with the global quantitative signal list

        Args:
            table_name: Name of the table (for generating table-level synonyms).
            columns: List of column names from the DataFrame.
            column_types: Dict mapping column name -> dtype string (may be empty).

        Returns:
            Dict suitable for SQLTemplate.column_signals.
        """
        signals: dict[str, list[str]] = {}

        for col in columns:
            col_lower = col.lower().replace("_", " ").strip()
            synonyms = [col_lower]
            # Add the original column name (with underscores) as well
            if "_" in col:
                synonyms.append(col.lower())
            signals[col] = synonyms

        # Table name tokens — helps QueryRouter disambiguate between tables
        # with overlapping column names (e.g. Budget_2025 vs AR_Aging_Report).
        # Stored under __table__ sentinel so _score_quantitative_signals can
        # give a tiebreaker bonus when the table name appears in the query.
        table_lower = table_name.lower()
        table_display = table_lower.replace("_", " ").strip()
        table_tokens = [t for t in table_lower.split("_") if len(t) > 1]
        signals["__table__"] = list(dict.fromkeys([table_display] + table_tokens))

        # Always add quantitative signals under sentinel key
        signals["__quantitative__"] = list(QUANTITATIVE_SIGNALS)

        return signals

    # -----------------------------------------------------------------------
    # SQL builders
    # -----------------------------------------------------------------------

    def _build_union_sql(self, tables: list[str]) -> str:
        """Build a UNION ALL SELECT across all year tables (P&L path)."""
        branches = [
            f'SELECT \'{t}\' AS year, "0" AS item, "1" AS value FROM excel_tables."{t}"'
            for t in tables
        ]
        union_body = "\n  UNION ALL\n  ".join(branches)
        return (
            f"SELECT year, item, value\n"
            f"FROM (\n  {union_body}\n) AS pl_data\n"
            f"WHERE item IS NOT NULL\n"
            f"LIMIT :row_cap"
        )

    # -----------------------------------------------------------------------
    # Registration helpers
    # -----------------------------------------------------------------------

    def _register_pl_template(self, sql: str, tables: list[str]) -> None:
        """Register the UNION ALL P&L template with trigger phrases."""
        from ai_ready_rag.modules.registry import SQLTemplate, get_registry

        template = SQLTemplate(
            name=_PL_TEMPLATE_NAME,
            sql=sql,
            trigger_phrases=_PL_TRIGGER_PHRASES,
            description=f"P&L data from excel_tables schema ({', '.join(tables)})",
            column_signals=None,
        )
        registry = get_registry()
        registry.register_sql_templates("core.excel_tables", {_PL_TEMPLATE_NAME: template})
        logger.info(
            "excel_tables.registered: template=%s tables=%s",
            _PL_TEMPLATE_NAME,
            tables,
        )

    def _register_table_template(
        self,
        table_name: str,
        schema_name: str,
        columns: list[str],
        column_types: dict,
        access_tags: list[str] | None = None,
    ) -> None:
        """Register a per-table NL2SQL template with column_signals.

        The SQL is a simple placeholder SELECT; actual SQL is generated by Claude
        at query time via _execute_nl2sql_route in RAGService.

        Args:
            access_tags: Tags required to query this table. Empty list means
                the table is public (all users may access it).
        """
        from ai_ready_rag.modules.registry import SQLTemplate, get_registry

        # Build a simple fallback SQL (used when column_signals is set, the
        # actual NL2SQL generation happens at query time — but we still need
        # a valid SQL to pass _validate_sql_template which requires LIMIT).
        quoted_cols = ", ".join(f'"{c}"' for c in columns[:10]) if columns else "*"
        sql = f'SELECT {quoted_cols} FROM "{schema_name}"."{table_name}" LIMIT :row_cap'

        column_signals = self._compute_column_signals(table_name, columns, column_types)

        template_name = f"excel_{table_name}"
        template = SQLTemplate(
            name=template_name,
            sql=sql,
            trigger_phrases=[],
            description=f"Table {schema_name}.{table_name} — NL2SQL via column signals",
            column_signals=column_signals,
            access_tags=access_tags or [],
        )

        registry = get_registry()
        registry.register_sql_templates("core.excel_tables", {template_name: template})
        logger.info(
            "excel_tables.registered_table_template: name=%s columns=%d access_tags=%s",
            template_name,
            len(columns),
            access_tags or [],
        )

    # -----------------------------------------------------------------------
    # Legacy method kept for callers that call _register_template directly
    # -----------------------------------------------------------------------

    def _register_template(self, sql: str, tables: list[str]) -> None:
        """Legacy alias for _register_pl_template (backward compat)."""
        self._register_pl_template(sql, tables)
