"""Tests for TableExtractionMetrics."""

import threading

from ai_ready_rag.services.table_extraction_metrics import TableExtractionMetrics


class TestTableExtractionMetrics:
    def setup_method(self):
        self.m = TableExtractionMetrics()

    def test_inc_extraction_counter(self):
        self.m.inc_extraction("pdf", "success")
        snap = self.m.snapshot()
        assert snap["extraction_total"]["success:pdf"] == 1

    def test_inc_extraction_with_reason(self):
        self.m.inc_extraction("pdf", "skip", "min_rows")
        snap = self.m.snapshot()
        assert snap["extraction_failure_reasons"]["min_rows"] == 1

    def test_inc_registration(self):
        self.m.inc_registration("excel")
        assert self.m.snapshot()["registration_total"]["excel"] == 1

    def test_inc_rag_fallback(self):
        self.m.inc_rag_fallback()
        self.m.inc_rag_fallback()
        assert self.m.snapshot()["rag_fallback_total"] == 2

    def test_set_orphaned_count(self):
        self.m.set_orphaned_count(7)
        assert self.m.snapshot()["orphaned_tables_count"] == 7

    def test_observe_registration_duration(self):
        self.m.observe_registration_duration(1.5)
        self.m.observe_registration_duration(2.5)
        snap = self.m.snapshot()["registration_duration_seconds"]
        assert snap["count"] == 2
        assert snap["min"] == 1.5
        assert snap["max"] == 2.5

    def test_prometheus_text_contains_all_metric_names(self):
        text = self.m.prometheus_text()
        for name in [
            "table_extraction_total",
            "table_extraction_failure_reasons",
            "table_registration_total",
            "sql_route_hit_total",
            "rag_fallback_total",
            "dual_registry_conflicts_total",
            "tag_predicate_denied_total",
            "orphaned_tables_count",
            "table_registration_duration_seconds",
            "nl2sql_generation_duration_seconds",
            "csv_bad_row_rate",
        ]:
            assert name in text, f"Missing metric: {name}"

    def test_thread_safety(self):
        errors = []

        def worker():
            try:
                for _ in range(100):
                    self.m.inc_extraction("pdf", "success")
                    self.m.inc_registration("excel")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        snap = self.m.snapshot()
        assert snap["extraction_total"]["success:pdf"] == 1000

    def test_reset_clears_all(self):
        self.m.inc_extraction("pdf", "success")
        self.m.reset()
        assert self.m.snapshot()["extraction_total"] == {}
