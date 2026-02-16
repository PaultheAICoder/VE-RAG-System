"""Lightweight in-memory metrics for the forms processing pipeline.

Provides thread-safe counters and histograms that track forms processing
activity. No external dependencies (no prometheus_client). Designed to be
importable even when ingestkit_forms is not installed.

A future issue can wire these to a /metrics Prometheus-export endpoint.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _Histogram:
    """Simple histogram that stores observations and computes summary stats."""

    _values: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float) -> None:
        with self._lock:
            self._values.append(value)

    def snapshot(self) -> dict:
        with self._lock:
            if not self._values:
                return {"count": 0, "sum": 0.0, "min": 0.0, "max": 0.0, "avg": 0.0}
            total = sum(self._values)
            return {
                "count": len(self._values),
                "sum": round(total, 6),
                "min": round(min(self._values), 6),
                "max": round(max(self._values), 6),
                "avg": round(total / len(self._values), 6),
            }

    def reset(self) -> None:
        with self._lock:
            self._values.clear()


class FormsMetrics:
    """Thread-safe in-memory metrics registry for forms processing.

    Metrics (per spec Section 15.2):
        forms_documents_processed_total  -- Counter(status)
        forms_match_confidence           -- Histogram(template_id)
        forms_extraction_duration_seconds -- Histogram(extraction_method)
        forms_fallback_total             -- Counter(reason)
        forms_compensation_total         -- Counter(target, status)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.monotonic()

        # Counters: label-key -> int
        self._documents_processed: dict[str, int] = defaultdict(int)
        self._fallback: dict[str, int] = defaultdict(int)
        self._compensation: dict[str, int] = defaultdict(int)

        # Histograms: label-key -> _Histogram
        self._match_confidence: dict[str, _Histogram] = defaultdict(_Histogram)
        self._extraction_duration: dict[str, _Histogram] = defaultdict(_Histogram)

    # -- Counter increments ---------------------------------------------------

    def inc_documents_processed(self, status: str) -> None:
        """Increment forms_documents_processed_total{status=...}."""
        with self._lock:
            self._documents_processed[status] += 1

    def inc_fallback(self, reason: str) -> None:
        """Increment forms_fallback_total{reason=...}."""
        with self._lock:
            self._fallback[reason] += 1

    def inc_compensation(self, target: str, status: str) -> None:
        """Increment forms_compensation_total{target=..., status=...}."""
        key = f"{target}:{status}"
        with self._lock:
            self._compensation[key] += 1

    # -- Histogram observations -----------------------------------------------

    def inc_match_confidence(self, template_id: str, confidence: float) -> None:
        """Observe forms_match_confidence{template_id=...}."""
        self._match_confidence[template_id].observe(confidence)

    def observe_extraction_duration(self, extraction_method: str, seconds: float) -> None:
        """Observe forms_extraction_duration_seconds{extraction_method=...}."""
        self._extraction_duration[extraction_method].observe(seconds)

    # -- Export ---------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot of all metrics."""
        with self._lock:
            docs = dict(self._documents_processed)
            fallback = dict(self._fallback)
            compensation = dict(self._compensation)

        match_conf = {k: v.snapshot() for k, v in self._match_confidence.items()}
        extract_dur = {k: v.snapshot() for k, v in self._extraction_duration.items()}

        return {
            "forms_documents_processed_total": docs,
            "forms_match_confidence": match_conf,
            "forms_extraction_duration_seconds": extract_dur,
            "forms_fallback_total": fallback,
            "forms_compensation_total": compensation,
            "uptime_seconds": round(time.monotonic() - self._started_at, 1),
        }

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._documents_processed.clear()
            self._fallback.clear()
            self._compensation.clear()
        for h in self._match_confidence.values():
            h.reset()
        self._match_confidence.clear()
        for h in self._extraction_duration.values():
            h.reset()
        self._extraction_duration.clear()


# Module-level singleton
metrics = FormsMetrics()


def get_forms_metrics() -> dict:
    """Return a snapshot of all forms metrics (for health/debug endpoints)."""
    return metrics.snapshot()
