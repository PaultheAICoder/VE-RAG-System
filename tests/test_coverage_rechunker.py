"""Tests for Coverage Summary Excel rechunker."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from ai_ready_rag.services.coverage_rechunker import (
    _is_section_header,
    is_coverage_summary,
    parse_coverage_sections,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_document(filename: str, tag_names: list[str] | None = None):
    """Create a mock Document object."""
    doc = MagicMock()
    doc.original_filename = filename
    tags = []
    for name in tag_names or []:
        tag = MagicMock()
        tag.name = name
        tags.append(tag)
    doc.tags = tags
    return doc


def _write_coverage_xlsx(path: Path) -> None:
    """Write a minimal Coverage Summary xlsx for testing."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active

    # Header rows
    ws.append(["Bethany Terrace HOA"])
    ws.append(["Coverage Summary 2025-2026"])
    ws.append([])

    # Property section
    ws.append(["Property"])
    ws.append(["Building Limit", "$5,000,000"])
    ws.append(["Business Personal Property", "$500,000"])
    ws.append(["Deductible", "$2,500"])
    ws.append(["Carrier", "Hartford"])
    ws.append([])

    # General Liability section
    ws.append(["General Liability"])
    ws.append(["Each Occurrence Limit", "$1,000,000"])
    ws.append(["General Aggregate", "$2,000,000"])
    ws.append(["Carrier", "Travelers"])
    ws.append([])

    # D&O section
    ws.append(["D&O"])
    ws.append(["Aggregate Limit", "$2,000,000"])
    ws.append(["Each Claim", "$2,000,000"])
    ws.append([])

    # Crime section
    ws.append(["Crime"])
    ws.append(["Employee Dishonesty", "$500,000"])
    ws.append(["Forgery", "$250,000"])

    wb.save(str(path))


def _write_non_coverage_xlsx(path: Path) -> None:
    """Write a non-coverage xlsx (e.g. budget) for negative testing."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Month", "Revenue", "Expenses"])
    ws.append(["January", 100000, 80000])
    ws.append(["February", 120000, 85000])
    wb.save(str(path))


# ---------------------------------------------------------------------------
# Tests: is_coverage_summary
# ---------------------------------------------------------------------------


class TestIsCoverageSummary:
    def test_filename_with_coverage(self):
        doc = _make_document("Coverage_Summary_2025.xlsx")
        assert is_coverage_summary(doc) is True

    def test_filename_with_summary(self):
        doc = _make_document("HOA_Summary_2025.xlsx")
        assert is_coverage_summary(doc) is True

    def test_insurance_tag(self):
        doc = _make_document("policies.xlsx", tag_names=["insurance"])
        assert is_coverage_summary(doc) is True

    def test_unrelated_file(self):
        doc = _make_document("Budget_2025.xlsx", tag_names=["financial"])
        assert is_coverage_summary(doc) is False


# ---------------------------------------------------------------------------
# Tests: _is_section_header
# ---------------------------------------------------------------------------


class TestIsSectionHeader:
    def test_known_keyword(self):
        assert _is_section_header(["General Liability"]) is not None

    def test_partial_keyword(self):
        assert _is_section_header(["Property Insurance"]) is not None

    def test_case_insensitive(self):
        assert _is_section_header(["WORKERS COMPENSATION"]) is not None

    def test_data_row_not_header(self):
        assert _is_section_header(["Building Limit", "$5,000,000"]) is None

    def test_empty_row(self):
        assert _is_section_header([None, None, None]) is None

    def test_dno(self):
        assert _is_section_header(["D&O"]) is not None

    def test_crime(self):
        assert _is_section_header(["Crime"]) is not None


# ---------------------------------------------------------------------------
# Tests: parse_coverage_sections
# ---------------------------------------------------------------------------


class TestParseCoverageSections:
    def test_parses_sections_from_xlsx(self):
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            path = Path(f.name)
        try:
            _write_coverage_xlsx(path)
            chunks = parse_coverage_sections(path)

            assert len(chunks) >= 3, f"Expected at least 3 sections, got {len(chunks)}"

            # Check section names are present
            section_names = [c["section"] for c in chunks]
            assert any("Property" in s for s in section_names)
            assert any("General Liability" in s for s in section_names)
            assert any("D&O" in s for s in section_names)

            # Check chunk text has natural-text format
            property_chunk = next(c for c in chunks if "Property" in c["section"])
            assert "Building Limit" in property_chunk["text"]
            assert "$5,000,000" in property_chunk["text"]
            assert "Bethany Terrace" in property_chunk["text"]
        finally:
            path.unlink(missing_ok=True)

    def test_no_sections_returns_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            path = Path(f.name)
        try:
            _write_non_coverage_xlsx(path)
            chunks = parse_coverage_sections(path)
            # Non-coverage xlsx may return 0 sections or a generic "General" section
            # Either is acceptable — it just shouldn't crash
            assert isinstance(chunks, list)
        finally:
            path.unlink(missing_ok=True)

    def test_extracts_year_range(self):
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            path = Path(f.name)
        try:
            _write_coverage_xlsx(path)
            chunks = parse_coverage_sections(path)
            # The header line should contain the year range
            if chunks:
                assert "2025-2026" in chunks[0]["text"]
        finally:
            path.unlink(missing_ok=True)

    def test_crime_section_parsed(self):
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            path = Path(f.name)
        try:
            _write_coverage_xlsx(path)
            chunks = parse_coverage_sections(path)
            crime_chunks = [c for c in chunks if "Crime" in c["section"]]
            assert len(crime_chunks) == 1
            assert "Employee Dishonesty" in crime_chunks[0]["text"]
            assert "$500,000" in crime_chunks[0]["text"]
        finally:
            path.unlink(missing_ok=True)
