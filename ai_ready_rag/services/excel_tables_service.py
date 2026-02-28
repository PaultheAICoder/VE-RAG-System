"""ExcelTablesService — registers SQL templates for excel_tables schema at startup.

Discovers year-based P&L tables written by ingestkit-excel and registers a single
'excel_pl_financials' SQLTemplate with financial trigger phrases. The QueryRouter
will route financial queries to this template; _execute_sql_route then runs the
UNION ALL query across all years and passes the results to Claude for synthesis.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Trigger phrases that indicate the user is asking about financial statement data.
# Any query containing one of these phrases will be routed to the SQL template.
_FINANCIAL_TRIGGER_PHRASES = [
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

_TEMPLATE_NAME = "excel_pl_financials"


class ExcelTablesService:
    """Discovers excel_tables schema tables and registers SQL templates.

    Called once at startup. Safe to call again (idempotent — re-registers
    with current table list, replacing any prior registration).
    """

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def discover_and_register(self) -> int:
        """Discover all year tables and register a UNION ALL SQL template.

        Returns:
            Number of year tables found (0 if schema doesn't exist yet).
        """
        tables = self._discover_year_tables()
        if not tables:
            logger.info("excel_tables.discovery: no tables found — skipping registration")
            return 0

        sql = self._build_union_sql(tables)
        self._register_template(sql, tables)
        return len(tables)

    def _discover_year_tables(self) -> list[str]:
        """Return sorted list of table names in the excel_tables schema."""
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
            logger.warning("excel_tables.discovery.failed: %s", exc)
            return []

    def _build_union_sql(self, tables: list[str]) -> str:
        """Build a UNION ALL SELECT across all year tables."""
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

    def _register_template(self, sql: str, tables: list[str]) -> None:
        """Register the UNION ALL template with the ModuleRegistry."""
        from ai_ready_rag.modules.registry import SQLTemplate, get_registry

        template = SQLTemplate(
            name=_TEMPLATE_NAME,
            sql=sql,
            trigger_phrases=_FINANCIAL_TRIGGER_PHRASES,
            description=f"P&L data from excel_tables schema ({', '.join(tables)})",
        )
        registry = get_registry()
        registry.register_sql_templates("core.excel_tables", {_TEMPLATE_NAME: template})
        logger.info(
            "excel_tables.registered: template=%s tables=%s",
            _TEMPLATE_NAME,
            tables,
        )
