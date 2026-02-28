"""Tests for ExcelTablesService, SqlInjectionGuard, and NL2SQL routing.

Tests from issue #453: Text-to-SQL with schema-aware routing.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from ai_ready_rag.modules.registry import ModuleRegistry, SQLTemplate
from ai_ready_rag.services.excel_tables_service import ExcelTablesService, SqlInjectionGuard
from ai_ready_rag.services.query_router import QueryRouter

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry singleton between tests."""
    ModuleRegistry.reset()
    yield
    ModuleRegistry.reset()


@pytest.fixture
def sql_guard():
    """SqlInjectionGuard instance."""
    return SqlInjectionGuard()


@pytest.fixture
def router_with_inventory_template():
    """QueryRouter with an inventory SQLTemplate with column_signals registered."""
    registry = ModuleRegistry.get_instance()
    registry.register_sql_templates(
        "test_inventory",
        {
            "excel_inventory_report": SQLTemplate(
                name="excel_inventory_report",
                sql='SELECT "SKU", "Item", "On Hand" FROM "excel_tables"."inventory_report" LIMIT :row_cap',
                trigger_phrases=[],
                description="Inventory report with stock levels",
                column_signals={
                    "SKU": ["sku", "part number", "part"],
                    "Item": ["item", "product", "description"],
                    "On Hand": ["on hand", "quantity", "stock", "in stock", "inventory"],
                    "__quantitative__": [
                        "how many",
                        "how much",
                        "total",
                        "count",
                        "on hand",
                        "quantity",
                        "in stock",
                    ],
                },
            )
        },
    )
    return QueryRouter(sql_confidence_threshold=0.6)


# =============================================================================
# TestSqlInjectionGuard
# =============================================================================


class TestSqlInjectionGuard:
    """Unit tests for SqlInjectionGuard.validate()."""

    def test_valid_select_passes(self, sql_guard):
        """A plain SELECT statement passes without raising."""
        sql = "SELECT * FROM excel_tables.inventory LIMIT 100"
        sql_guard.validate(sql)  # Should not raise

    def test_valid_select_with_where_passes(self, sql_guard):
        """A SELECT with WHERE clause passes."""
        sql = 'SELECT "Item", "On Hand" FROM excel_tables.inventory WHERE "SKU" ILIKE \'%plug%\' LIMIT 100'
        sql_guard.validate(sql)  # Should not raise

    def test_rejects_delete(self, sql_guard):
        """DELETE statement raises ValueError."""
        with pytest.raises(ValueError, match="forbidden|only SELECT"):
            sql_guard.validate("DELETE FROM excel_tables.inventory WHERE id = 1")

    def test_rejects_drop(self, sql_guard):
        """DROP statement raises ValueError."""
        with pytest.raises(ValueError, match="forbidden|only SELECT"):
            sql_guard.validate("DROP TABLE excel_tables.inventory")

    def test_rejects_insert(self, sql_guard):
        """INSERT statement raises ValueError."""
        with pytest.raises(ValueError, match="forbidden|only SELECT"):
            sql_guard.validate("INSERT INTO t VALUES (1, 2, 3)")

    def test_rejects_update(self, sql_guard):
        """UPDATE statement raises ValueError."""
        with pytest.raises(ValueError, match="forbidden|only SELECT"):
            sql_guard.validate("UPDATE excel_tables.inventory SET qty = 0")

    def test_rejects_non_select_start(self, sql_guard):
        """SQL that starts with non-SELECT raises ValueError."""
        with pytest.raises(ValueError, match="only SELECT"):
            sql_guard.validate("TRUNCATE TABLE excel_tables.inventory")

    def test_rejects_select_with_embedded_drop(self, sql_guard):
        """SQL containing DROP keyword anywhere raises ValueError."""
        with pytest.raises(ValueError, match="forbidden"):
            sql_guard.validate("SELECT * FROM t; DROP TABLE t")


# =============================================================================
# TestQueryRouterQuantitativeSignals
# =============================================================================


class TestQueryRouterQuantitativeSignals:
    """Tests for QueryRouter with quantitative signal scoring."""

    def test_how_many_routes_to_sql(self, router_with_inventory_template):
        """'how many X on hand' routes to SQL via quantitative signals."""
        from ai_ready_rag.services.query_router import RouteType

        decision = router_with_inventory_template.route(
            "how many EndCore Plugger sets on hand?",
            structured_query_enabled=True,
        )
        assert decision.route == RouteType.SQL
        assert decision.confidence >= 0.6

    def test_total_routes_to_sql(self, router_with_inventory_template):
        """'total on hand for item' routes to SQL via quantitative signals."""
        from ai_ready_rag.services.query_router import RouteType

        decision = router_with_inventory_template.route(
            "what is the total on hand quantity for item X?",
            structured_query_enabled=True,
        )
        assert decision.route == RouteType.SQL

    def test_on_hand_routes_to_sql(self, router_with_inventory_template):
        """'items on hand' routes to SQL via column signal match."""
        from ai_ready_rag.services.query_router import RouteType

        decision = router_with_inventory_template.route(
            "how many items are currently in stock?",
            structured_query_enabled=True,
        )
        assert decision.route == RouteType.SQL

    def test_vacation_policy_stays_rag(self, router_with_inventory_template):
        """Unrelated query does not match any signal and routes to RAG."""
        from ai_ready_rag.services.query_router import RouteType

        decision = router_with_inventory_template.route(
            "what is the vacation policy?",
            structured_query_enabled=True,
        )
        assert decision.route == RouteType.RAG

    def test_structured_disabled_stays_rag(self, router_with_inventory_template):
        """When structured_query_enabled=False, all queries route to RAG."""
        from ai_ready_rag.services.query_router import RouteType

        decision = router_with_inventory_template.route(
            "how many items on hand?",
            structured_query_enabled=False,
        )
        assert decision.route == RouteType.RAG

    def test_count_routes_to_sql(self, router_with_inventory_template):
        """'count of items in inventory' routes to SQL."""
        from ai_ready_rag.services.query_router import RouteType

        decision = router_with_inventory_template.route(
            "count of all items in stock",
            structured_query_enabled=True,
        )
        assert decision.route == RouteType.SQL

    def test_quantitative_only_stays_rag(self):
        """Quantitative signal alone (no column signal) does NOT route to SQL."""
        from ai_ready_rag.services.query_router import RouteType

        registry = ModuleRegistry.get_instance()
        registry.register_sql_templates(
            "test_table",
            {
                "excel_my_table": SQLTemplate(
                    name="excel_my_table",
                    sql='SELECT "col" FROM "excel_tables"."my_table" LIMIT :row_cap',
                    trigger_phrases=[],
                    description="test",
                    column_signals={
                        "SomeObscureColumn": ["xyzzy123"],
                        "__quantitative__": ["how many", "total"],
                    },
                )
            },
        )
        router = QueryRouter(sql_confidence_threshold=0.6)

        # Query has quantitative signal ("how many") but NOT the column synonym ("xyzzy123")
        decision = router.route(
            "how many things are there?",
            structured_query_enabled=True,
        )
        # Score should be 0.5 (quantitative hit only) which is below threshold 0.6
        assert decision.route == RouteType.RAG


# =============================================================================
# TestExcelTableRegistryDiscover
# =============================================================================


class TestExcelTableRegistryDiscover:
    """Tests for ExcelTablesService registry-based discovery."""

    def test_discover_queries_registry_not_information_schema(self):
        """discover_and_register_all() queries excel_table_registry, not information_schema."""
        service = ExcelTablesService("postgresql://fake/db")

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with (
            patch("psycopg2.connect", return_value=mock_conn),
            # Prevent get_registry() from raising RuntimeError
            patch(
                "ai_ready_rag.services.excel_tables_service.ExcelTablesService"
                "._discover_year_tables_legacy",
                return_value=[],
            ),
        ):
            service.discover_and_register_all()

        # Check that the cursor.execute was called with a query containing
        # "excel_table_registry" and NOT "information_schema"
        execute_calls = mock_cursor.execute.call_args_list
        assert len(execute_calls) > 0, "Expected at least one cursor.execute call"
        first_call_sql = execute_calls[0][0][0]
        assert "excel_table_registry" in first_call_sql
        assert "information_schema" not in first_call_sql

    def test_discover_skips_info_schema_when_registry_has_rows(self):
        """When registry has rows, information_schema is not queried."""
        service = ExcelTablesService("postgresql://fake/db")

        inventory_columns = json.dumps(["SKU", "Item", "On Hand"])
        inventory_types = json.dumps({"SKU": "object", "Item": "object", "On Hand": "int64"})

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            ("inventory_report", "excel_tables", inventory_columns, inventory_types)
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.close = MagicMock()

        with patch("psycopg2.connect", return_value=mock_conn):
            # Call _discover_registry_tables directly — no get_registry involved
            service._discover_registry_tables()

        execute_calls = mock_cursor.execute.call_args_list
        for c in execute_calls:
            sql = c[0][0] if c[0] else ""
            assert "information_schema" not in sql


# =============================================================================
# TestBackwardCompatibility
# =============================================================================


class TestBackwardCompatibility:
    """Verify P&L trigger phrases still work after the refactor."""

    def test_pl_revenue_still_routes_to_sql(self):
        """Financial trigger phrase 'revenue' still routes to SQL."""
        from ai_ready_rag.services.query_router import RouteType

        registry = ModuleRegistry.get_instance()
        registry.register_sql_templates(
            "core.excel_tables",
            {
                "excel_pl_financials": SQLTemplate(
                    name="excel_pl_financials",
                    sql=(
                        "SELECT year, item, value FROM ("
                        'SELECT \'2024\' AS year, "0" AS item, "1" AS value '
                        'FROM excel_tables."pl_2024"'
                        ") AS pl_data WHERE item IS NOT NULL LIMIT :row_cap"
                    ),
                    trigger_phrases=[
                        "revenue",
                        "cogs",
                        "gross profit",
                        "net income",
                        "profit and loss",
                    ],
                    description="P&L data",
                    column_signals=None,
                )
            },
        )
        router = QueryRouter(sql_confidence_threshold=0.6)

        decision = router.route(
            "what was total revenue in 2024?",
            structured_query_enabled=True,
        )
        assert decision.route == RouteType.SQL

    def test_pl_cogs_still_routes_to_sql(self):
        """Financial trigger phrase 'cogs' still routes to SQL."""
        from ai_ready_rag.services.query_router import RouteType

        registry = ModuleRegistry.get_instance()
        registry.register_sql_templates(
            "core.excel_tables",
            {
                "excel_pl_financials": SQLTemplate(
                    name="excel_pl_financials",
                    sql=(
                        "SELECT year, item, value FROM ("
                        'SELECT \'2024\' AS year, "0" AS item, "1" AS value '
                        'FROM excel_tables."pl_2024"'
                        ") AS pl_data WHERE item IS NOT NULL LIMIT :row_cap"
                    ),
                    trigger_phrases=["cogs", "cost of goods", "revenue", "net income"],
                    description="P&L data",
                    column_signals=None,
                )
            },
        )
        router = QueryRouter(sql_confidence_threshold=0.6)

        decision = router.route(
            "show me COGS for last year",
            structured_query_enabled=True,
        )
        assert decision.route == RouteType.SQL


# =============================================================================
# TestIngestKitRegistryWrite
# =============================================================================


class TestIngestKitRegistryWrite:
    """Tests for VERagPostgresStructuredDB._register_table_in_registry."""

    def _make_df(self):
        """Return a minimal pandas DataFrame for testing."""
        try:
            import pandas as pd

            return pd.DataFrame(
                {
                    "SKU": ["A1", "A2"],
                    "Item": ["Widget", "Gadget"],
                    "On Hand": [100, 50],
                }
            )
        except ImportError:
            pytest.skip("pandas not available")

    def test_registry_write_on_ingest(self):
        """_register_table_in_registry calls INSERT into excel_table_registry."""
        from ai_ready_rag.services.ingestkit_adapters import VERagPostgresStructuredDB

        df = self._make_df()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg2.connect", return_value=mock_conn):
            adapter = VERagPostgresStructuredDB.__new__(VERagPostgresStructuredDB)
            adapter._database_url = "postgresql://fake/db"
            adapter._SCHEMA = "excel_tables"
            adapter._register_table_in_registry("inventory_report", df)

        # Verify cursor.execute was called with INSERT SQL
        assert mock_cursor.execute.called
        call_sql = mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO excel_table_registry" in call_sql
        assert "ON CONFLICT" in call_sql

    def test_registry_write_contains_correct_table_name(self):
        """_register_table_in_registry includes correct table_name in params."""
        from ai_ready_rag.services.ingestkit_adapters import VERagPostgresStructuredDB

        df = self._make_df()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg2.connect", return_value=mock_conn):
            adapter = VERagPostgresStructuredDB.__new__(VERagPostgresStructuredDB)
            adapter._database_url = "postgresql://fake/db"
            adapter._SCHEMA = "excel_tables"
            adapter._register_table_in_registry("my_test_table", df)

        call_params = mock_cursor.execute.call_args[0][1]
        assert "my_test_table" in call_params

    def test_registry_write_failure_does_not_raise(self):
        """Registry write failure is silently logged, never raised."""
        from ai_ready_rag.services.ingestkit_adapters import VERagPostgresStructuredDB

        df = self._make_df()

        with patch("psycopg2.connect", side_effect=Exception("db unreachable")):
            adapter = VERagPostgresStructuredDB.__new__(VERagPostgresStructuredDB)
            adapter._database_url = "postgresql://fake/db"
            adapter._SCHEMA = "excel_tables"
            # Should not raise
            adapter._register_table_in_registry("inventory_report", df)
