"""Tests for Phase 3 NL2SQL prompt enrichment helpers."""

import json
from unittest.mock import MagicMock


def make_settings(**overrides):
    s = MagicMock()
    s.nl2sql_max_prompt_columns = overrides.get("nl2sql_max_prompt_columns", 20)
    s.nl2sql_max_samples_per_column = overrides.get("nl2sql_max_samples_per_column", 20)
    s.nl2sql_sample_token_budget = overrides.get("nl2sql_sample_token_budget", 2000)
    s.nl2sql_max_prompt_tokens = overrides.get("nl2sql_max_prompt_tokens", 8000)
    return s


def make_rag_service(settings=None, **setting_overrides):
    from ai_ready_rag.services.rag_service import RAGService

    svc = RAGService.__new__(RAGService)
    svc.settings = settings or make_settings(**setting_overrides)
    svc.default_model = "mock-model"
    return svc


class TestTruncateSamplesToBudget:
    def test_empty_returns_empty(self):
        svc = make_rag_service()
        assert svc._truncate_samples_to_budget({}, ["col_a"], "query") == {}

    def test_within_budget_unchanged(self):
        svc = make_rag_service()
        samples = {"item": ["Revenue", "COGS"], "year": ["2019", "2020"]}
        result = svc._truncate_samples_to_budget(samples, ["item", "year"], "show revenue 2020")
        assert result["item"] == ["Revenue", "COGS"]
        assert result["year"] == ["2019", "2020"]

    def test_absent_columns_excluded(self):
        svc = make_rag_service()
        result = svc._truncate_samples_to_budget({"item": ["Revenue"]}, ["item", "value"], "q")
        assert "value" not in result

    def test_zero_string_samples_dropped(self):
        svc = make_rag_service()
        samples = {"item": [None, "", "   "], "year": ["2019"]}
        result = svc._truncate_samples_to_budget(samples, ["item", "year"], "q")
        assert "item" not in result
        assert "year" in result

    def test_max_samples_cap(self):
        svc = make_rag_service(nl2sql_max_samples_per_column=3)
        samples = {"item": ["A", "B", "C", "D", "E"]}
        result = svc._truncate_samples_to_budget(samples, ["item"], "q")
        assert result["item"] == ["A", "B", "C"]

    def test_never_raises_on_tight_budget(self):
        svc = make_rag_service(nl2sql_sample_token_budget=1)
        samples = {"item": ["Revenue"] * 100}
        result = svc._truncate_samples_to_budget(samples, ["item"], "q")
        assert isinstance(result, dict)


class TestLoadRowValueSamples:
    def test_returns_empty_when_row_not_found(self):
        svc = make_rag_service()
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        result = svc._load_row_value_samples("t", "s", db)
        assert result == {}

    def test_returns_empty_when_samples_null(self):
        svc = make_rag_service()
        row = MagicMock()
        row.row_value_samples = None
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = row
        result = svc._load_row_value_samples("t", "s", db)
        assert result == {}

    def test_returns_parsed_dict(self):
        svc = make_rag_service()
        row = MagicMock()
        row.row_value_samples = json.dumps({"item": ["Revenue"]})
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = row
        result = svc._load_row_value_samples("t", "s", db)
        assert result == {"item": ["Revenue"]}

    def test_graceful_on_db_error(self):
        svc = make_rag_service()
        db = MagicMock()
        db.query.side_effect = Exception("DB down")
        result = svc._load_row_value_samples("t", "s", db)
        assert result == {}
