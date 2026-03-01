"""Thread-safe in-memory metrics for the unified table extraction pipeline.

11 metrics per UNIFIED_TABLE_INGEST_v1 §12. No external dependencies.
Module-level singleton accessed via get_table_metrics().
"""

from __future__ import annotations

import threading
from collections import defaultdict


class TableExtractionMetrics:
    """Thread-safe metrics for table extraction pipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Counters
        self._extraction_total: dict[str, int] = defaultdict(int)  # key: "status:format"
        self._extraction_failure_reasons: dict[str, int] = defaultdict(int)
        self._registration_total: dict[str, int] = defaultdict(int)  # key: source_format
        self._sql_route_hit_total: dict[str, int] = defaultdict(int)  # key: "template:format"
        self._rag_fallback_total: int = 0
        self._dual_registry_conflicts_total: int = 0
        self._tag_predicate_denied_total: dict[str, int] = defaultdict(int)
        # Gauge
        self._orphaned_tables_count: int = 0
        # Histograms
        self._registration_duration_seconds: list[float] = []
        self._nl2sql_generation_duration_seconds: list[float] = []
        self._csv_bad_row_rate: list[float] = []

    def reset(self) -> None:
        with self._lock:
            self.__init__()

    def inc_extraction(self, source_format: str, status: str, reason: str | None = None) -> None:
        with self._lock:
            self._extraction_total[f"{status}:{source_format}"] += 1
            if reason:
                self._extraction_failure_reasons[reason] += 1

    def inc_registration(self, source_format: str) -> None:
        with self._lock:
            self._registration_total[source_format] += 1

    def inc_sql_route_hit(self, template_name: str, source_format: str = "unknown") -> None:
        with self._lock:
            self._sql_route_hit_total[f"{template_name}:{source_format}"] += 1

    def inc_rag_fallback(self) -> None:
        with self._lock:
            self._rag_fallback_total += 1

    def inc_dual_registry_conflict(self) -> None:
        with self._lock:
            self._dual_registry_conflicts_total += 1

    def inc_tag_denied(self, template_name: str) -> None:
        with self._lock:
            self._tag_predicate_denied_total[template_name] += 1

    def set_orphaned_count(self, count: int) -> None:
        with self._lock:
            self._orphaned_tables_count = count

    def observe_registration_duration(self, seconds: float) -> None:
        with self._lock:
            self._registration_duration_seconds.append(seconds)

    def observe_nl2sql_duration(self, seconds: float) -> None:
        with self._lock:
            self._nl2sql_generation_duration_seconds.append(seconds)

    def observe_csv_bad_row_rate(self, rate: float) -> None:
        with self._lock:
            self._csv_bad_row_rate.append(rate)

    def snapshot(self) -> dict:
        """Return all metrics as a JSON-serializable dict."""
        with self._lock:
            return {
                "extraction_total": dict(self._extraction_total),
                "extraction_failure_reasons": dict(self._extraction_failure_reasons),
                "registration_total": dict(self._registration_total),
                "sql_route_hit_total": dict(self._sql_route_hit_total),
                "rag_fallback_total": self._rag_fallback_total,
                "dual_registry_conflicts_total": self._dual_registry_conflicts_total,
                "tag_predicate_denied_total": dict(self._tag_predicate_denied_total),
                "orphaned_tables_count": self._orphaned_tables_count,
                "registration_duration_seconds": self._percentiles(
                    self._registration_duration_seconds
                ),
                "nl2sql_generation_duration_seconds": self._percentiles(
                    self._nl2sql_generation_duration_seconds
                ),
                "csv_bad_row_rate": self._percentiles(self._csv_bad_row_rate),
            }

    def prometheus_text(self) -> str:
        """Prometheus text exposition format."""
        with self._lock:
            lines = []

            def counter(name, help_text, data: dict):
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} counter")
                for k, v in data.items():
                    lines.append(f'{name}{{key="{k}"}} {v}')

            def simple_counter(name, help_text, value: int):
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")

            def gauge(name, help_text, value):
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")

            def histogram(name, help_text, values: list):
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} summary")
                if values:
                    s = sorted(values)
                    n = len(s)
                    lines.append(f'{name}{{quantile="0.5"}} {s[n // 2]}')
                    lines.append(f'{name}{{quantile="0.95"}} {s[min(int(n * 0.95), n - 1)]}')
                    lines.append(f'{name}{{quantile="0.99"}} {s[min(int(n * 0.99), n - 1)]}')
                    lines.append(f"{name}_count {n}")

            counter(
                "table_extraction_total",
                "Tables extracted by format and status",
                self._extraction_total,
            )
            counter(
                "table_extraction_failure_reasons",
                "Extraction failures by reason",
                self._extraction_failure_reasons,
            )
            counter(
                "table_registration_total",
                "SQL templates registered at startup",
                self._registration_total,
            )
            counter(
                "sql_route_hit_total",
                "SQL route hits by template and format",
                self._sql_route_hit_total,
            )
            counter(
                "tag_predicate_denied_total",
                "Access denied at SQL execution time",
                self._tag_predicate_denied_total,
            )
            simple_counter(
                "rag_fallback_total",
                "Queries that fell back from SQL to RAG",
                self._rag_fallback_total,
            )
            simple_counter(
                "dual_registry_conflicts_total",
                "Dual-registry conflicts at startup",
                self._dual_registry_conflicts_total,
            )
            gauge(
                "orphaned_tables_count",
                "Tables in schema with no registry row",
                self._orphaned_tables_count,
            )
            histogram(
                "table_registration_duration_seconds",
                "Startup registration time",
                self._registration_duration_seconds,
            )
            histogram(
                "nl2sql_generation_duration_seconds",
                "Claude NL2SQL generation latency",
                self._nl2sql_generation_duration_seconds,
            )
            histogram("csv_bad_row_rate", "CSV parse bad-row fraction", self._csv_bad_row_rate)

            return "\n".join(lines)

    @staticmethod
    def _percentiles(values: list) -> dict:
        if not values:
            return {"count": 0}
        s = sorted(values)
        n = len(s)
        return {
            "count": n,
            "min": s[0],
            "max": s[-1],
            "avg": sum(s) / n,
            "p50": s[n // 2],
            "p95": s[min(int(n * 0.95), n - 1)],
            "p99": s[min(int(n * 0.99), n - 1)],
        }


_table_metrics = TableExtractionMetrics()


def get_table_metrics() -> TableExtractionMetrics:
    return _table_metrics
