"""Tests for TableExtractionAdapter."""

from unittest.mock import MagicMock, patch

import pandas as pd

from ai_ready_rag.services.table_extraction_adapter import TableExtractionAdapter


def make_adapter():
    settings = MagicMock()
    settings.database_url = "postgresql://test"
    return TableExtractionAdapter("postgresql://test", settings)


def make_document(doc_id="doc-123", filename="report.pdf"):
    doc = MagicMock()
    doc.id = doc_id
    doc.original_filename = filename
    return doc


class TestHeaderHeuristic:
    def test_promotes_first_row_when_header_pattern(self):
        adapter = make_adapter()
        df = pd.DataFrame(
            [
                ["Item", "Value", "Year"],
                ["Revenue", "1000000", "2024"],
                ["COGS", "500000", "2024"],
            ]
        )
        result = adapter._apply_header_heuristic(df)
        assert list(result.columns) == ["Item", "Value", "Year"]
        assert len(result) == 2

    def test_leaves_df_unchanged_when_not_header_pattern(self):
        """When both rows are all-string (no numeric second row), no promotion happens."""
        adapter = make_adapter()
        # Second row has no numeric values → heuristic does NOT promote
        df = pd.DataFrame({"A": ["Label", "Gamma"], "B": ["Header", "Delta"]})
        original_len = len(df)
        result = adapter._apply_header_heuristic(df)
        # No rows should be dropped — length stays the same
        assert len(result) == original_len

    def test_single_row_not_modified(self):
        adapter = make_adapter()
        df = pd.DataFrame([["Header", "Col"]])
        result = adapter._apply_header_heuristic(df)
        assert len(result) == 1

    def test_preserves_dataframe_when_second_row_not_numeric(self):
        """When second row has no numeric values, header heuristic does not promote."""
        adapter = make_adapter()
        df = pd.DataFrame(
            [
                ["Alpha", "Beta"],
                ["Gamma", "Delta"],
                ["Epsilon", "Zeta"],
            ]
        )
        original_len = len(df)
        result = adapter._apply_header_heuristic(df)
        assert len(result) == original_len


class TestExtractAndPersist:
    def test_returns_empty_for_none_document(self):
        adapter = make_adapter()
        doc = make_document()
        result = adapter.extract_and_persist(None, doc, "pdf")
        assert result == []

    def test_returns_empty_when_no_tables(self):
        adapter = make_adapter()
        doc = make_document()
        # Use spec=["tables"] so the code takes the .tables branch (not .document.tables)
        docling_doc = MagicMock(spec=["tables"])
        docling_doc.tables = []
        result = adapter.extract_and_persist(docling_doc, doc, "pdf")
        assert result == []

    def test_skips_table_below_min_size(self):
        adapter = make_adapter()
        doc = make_document()
        table_item = MagicMock()
        # 1 row = too small
        table_item.export_to_dataframe.return_value = pd.DataFrame({"A": ["x"]})
        # Use spec=["tables"] so the code takes the .tables branch
        docling_doc = MagicMock(spec=["tables"])
        docling_doc.tables = [table_item]
        with patch.object(adapter, "_write_table"), patch.object(adapter, "_write_registry"):
            result = adapter.extract_and_persist(docling_doc, doc, "pdf")
        assert result == []

    def test_reads_tables_from_docling_document_attribute(self):
        """When docling_doc has .document.tables, uses that."""
        adapter = make_adapter()
        doc = make_document()
        inner_doc = MagicMock()
        inner_doc.tables = []
        docling_result = MagicMock()
        docling_result.document = inner_doc
        # Ensure .tables not directly on result (to test .document.tables path)
        del docling_result.tables
        result = adapter.extract_and_persist(docling_result, doc, "pdf")
        assert result == []

    def test_registers_valid_table(self):
        """A 3-row × 3-column table with header pattern gets registered."""
        adapter = make_adapter()
        doc = make_document(filename="report.pdf")
        table_item = MagicMock()
        table_item.export_to_dataframe.return_value = pd.DataFrame(
            [
                ["Category", "Amount", "Year"],
                ["Revenue", "5000000", "2024"],
                ["Expense", "3000000", "2024"],
            ]
        )
        # Use spec=[] to prevent MagicMock from auto-creating a .document attribute,
        # so the code falls through to the .tables branch.
        docling_doc = MagicMock(spec=["tables"])
        docling_doc.tables = [table_item]

        with (
            patch.object(adapter, "_write_table") as mock_write,
            patch.object(adapter, "_write_registry") as mock_reg,
        ):
            result = adapter.extract_and_persist(docling_doc, doc, "pdf")

        # Should register one table
        assert len(result) == 1
        mock_write.assert_called_once()
        mock_reg.assert_called_once()

    def test_continues_after_single_table_failure(self):
        """If one table fails extraction, remaining tables are still processed."""
        adapter = make_adapter()
        doc = make_document(filename="multi.pdf")

        # First table will fail export, second is valid
        bad_table = MagicMock()
        bad_table.export_to_dataframe.side_effect = RuntimeError("export failed")

        good_table = MagicMock()
        # Use all-string data (no numeric second row) to avoid header promotion
        # which would reduce row count below min threshold.
        # 4 rows × 3 columns — after no header promotion: still 4 rows, 3 cols
        good_table.export_to_dataframe.return_value = pd.DataFrame(
            {
                "Col A": ["Alpha", "Gamma", "Epsilon", "Zeta"],
                "Col B": ["Beta", "Delta", "Eta", "Theta"],
                "Col C": ["One", "Two", "Three", "Four"],
            }
        )

        # Use spec to prevent auto-creation of .document attribute
        docling_doc = MagicMock(spec=["tables"])
        docling_doc.tables = [bad_table, good_table]

        with patch.object(adapter, "_write_table"), patch.object(adapter, "_write_registry"):
            result = adapter.extract_and_persist(docling_doc, doc, "pdf")

        # Only the good table gets registered
        assert len(result) == 1

    def test_skips_single_column_table(self):
        """Tables with < 2 columns are skipped."""
        adapter = make_adapter()
        doc = make_document()
        table_item = MagicMock()
        # 3 rows, 1 column — below min column threshold
        table_item.export_to_dataframe.return_value = pd.DataFrame({"A": ["x", "y", "z"]})
        # Use spec=["tables"] so the code takes the .tables branch
        docling_doc = MagicMock(spec=["tables"])
        docling_doc.tables = [table_item]

        with patch.object(adapter, "_write_table"), patch.object(adapter, "_write_registry"):
            result = adapter.extract_and_persist(docling_doc, doc, "pdf")

        assert result == []
