"""Tests for CsvProcessingService."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def tmp_csv(tmp_path):
    """Helper to create temp CSV files."""

    def _make(content: str, filename: str = "test.csv") -> Path:
        p = tmp_path / filename
        p.write_text(content, encoding="utf-8")
        return p

    return _make


@pytest.fixture
def bom_csv(tmp_path):
    p = tmp_path / "bom.csv"
    p.write_bytes(b"\xef\xbb\xbfName,Value\nRevenue,1000\nCOGS,500\n")
    return p


def make_service():
    from ai_ready_rag.services.csv_processing_service import CsvProcessingService

    settings = MagicMock()
    settings.csv_max_file_size_mb = 100
    settings.database_url = "sqlite:///test.db"
    return CsvProcessingService(settings)


class TestDelimiterDetection:
    def test_comma_delimited(self, tmp_csv):
        p = tmp_csv("Name,Value,Year\nRevenue,1000,2024\nCOGS,500,2024\n")
        svc = make_service()
        assert svc._detect_delimiter(p) == ","

    def test_semicolon_delimited(self, tmp_csv):
        p = tmp_csv("Name;Value;Year\nRevenue;1000;2024\nCOGS;500;2024\n")
        svc = make_service()
        assert svc._detect_delimiter(p) == ";"

    def test_tab_delimited(self, tmp_csv):
        content = "Name\tValue\tYear\nRevenue\t1000\t2024\nCOGS\t500\t2024\n"
        p = tmp_csv(content)
        svc = make_service()
        assert svc._detect_delimiter(p) == "\t"

    def test_single_column_returns_none_or_invalid(self, tmp_csv):
        """Single-column CSV: detect_delimiter may return a delimiter but validate catches it."""
        p = tmp_csv("SingleColumn\nvalue1\nvalue2\n")
        svc = make_service()
        # Either None or a delimiter — if delimiter returned, validate will flag it
        result = svc._detect_delimiter(p)
        if result is not None:
            df, stats = svc._read_csv(p, result)
            # Should be caught by validation (single column)
            assert svc._validate(df, stats) == "csv_delimiter_undetected"


class TestBomHandling:
    def test_bom_stripped_from_column_names(self, bom_csv):
        svc = make_service()
        df, stats = svc._read_csv(bom_csv, ",")
        assert not any(col.startswith("\ufeff") for col in df.columns)
        assert "Name" in df.columns
        assert "Value" in df.columns


class TestValidation:
    def test_empty_df_returns_csv_empty(self):
        svc = make_service()
        df = pd.DataFrame()
        stats = {"bad_row_count": 0, "total_rows": 0}
        assert svc._validate(df, stats) == "csv_empty"

    def test_bad_row_rate_exceeds_threshold(self):
        svc = make_service()
        df = pd.DataFrame({"A": range(90), "B": range(90)})
        stats = {"bad_row_count": 10, "total_rows": 100}
        assert svc._validate(df, stats) == "csv_bad_row_rate"

    def test_valid_csv_returns_none(self):
        svc = make_service()
        df = pd.DataFrame({"A": range(10), "B": range(10)})
        stats = {"bad_row_count": 0, "total_rows": 10}
        assert svc._validate(df, stats) is None

    def test_single_column_returns_delimiter_undetected(self):
        svc = make_service()
        df = pd.DataFrame({"SingleColumn": range(5)})
        stats = {"bad_row_count": 0, "total_rows": 5}
        assert svc._validate(df, stats) == "csv_delimiter_undetected"

    def test_bad_row_rate_under_threshold_is_valid(self):
        svc = make_service()
        df = pd.DataFrame({"A": range(98), "B": range(98)})
        stats = {"bad_row_count": 2, "total_rows": 100}
        assert svc._validate(df, stats) is None


class TestVectorPathChunkCount:
    def test_large_df_produces_multiple_chunks(self):
        svc = make_service()
        df = pd.DataFrame({"A": range(100), "B": range(100)})
        assert svc._vector_path_chunk_count(df) == 5

    def test_small_df_returns_at_least_one(self):
        svc = make_service()
        df = pd.DataFrame({"A": range(3), "B": range(3)})
        assert svc._vector_path_chunk_count(df) == 1

    def test_empty_df_returns_one(self):
        svc = make_service()
        df = pd.DataFrame()
        assert svc._vector_path_chunk_count(df) == 1


class TestStructuredPathSkipsNonPostgres:
    def test_sqlite_url_returns_empty_list(self, tmp_csv):
        """Structured path is skipped for non-postgresql databases."""
        from ai_ready_rag.services.csv_processing_service import CsvProcessingService

        settings = MagicMock()
        settings.csv_max_file_size_mb = 100
        settings.database_url = "sqlite:///test.db"
        svc = CsvProcessingService(settings)

        p = tmp_csv("Name,Value\nRevenue,1000\nCOGS,500\n")
        df, _ = svc._read_csv(p, ",")
        document = MagicMock()
        document.original_filename = "test.csv"
        document.id = "doc-123"
        result = svc._structured_path(df, document, ["hr"], "default", MagicMock())
        assert result == []
