"""Tests for signal canonicalization utilities."""

from ai_ready_rag.utils.signal_canon import canonicalize, make_table_name


class TestCanonicalize:
    def test_basic_lowercase(self):
        assert canonicalize("Revenue") == "revenue"

    def test_unicode_nfkc(self):
        # Full-width chars normalize to ASCII
        result = canonicalize("Ｒｅｖｅｎｕｅ")
        assert result == "revenue"

    def test_underscore_to_space(self):
        assert canonicalize("net_income") == "net income"

    def test_hyphen_to_space(self):
        assert canonicalize("cost-of-goods") == "cost of goods"

    def test_collapse_whitespace(self):
        assert canonicalize("total   revenue") == "total revenue"

    def test_stopword_filtered(self):
        assert canonicalize("the") == ""
        assert canonicalize("of") == ""

    def test_too_short_filtered(self):
        assert canonicalize("ab") == ""

    def test_too_long_filtered(self):
        assert canonicalize("x" * 81) == ""

    def test_exactly_3_chars_passes(self):
        result = canonicalize("abc")
        assert result == "abc"


class TestMakeTableName:
    def test_length_always_lte_49(self):
        name = make_table_name("some_document_with_a_very_long_name", "doc-123", 0)
        assert len(name) <= 49

    def test_same_input_stable(self):
        n1 = make_table_name("budget_2025", "doc-abc", 0)
        n2 = make_table_name("budget_2025", "doc-abc", 0)
        assert n1 == n2

    def test_different_idx_different_name(self):
        n1 = make_table_name("report", "doc-abc", 0)
        n2 = make_table_name("report", "doc-abc", 1)
        assert n1 != n2

    def test_special_chars_slugified(self):
        name = make_table_name("P&L Statements (2025)", "doc-1", 0)
        assert all(c.isalnum() or c == "_" for c in name)
