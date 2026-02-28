"""Claude API cost tracking with daily and monthly cap enforcement.

Tracks cumulative Claude API spend via the claude_usage_log table.
Enforces configurable daily and monthly caps before allowing API calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    operation: str  # "enrichment" | "query" | "extraction"
    document_id: str | None = None
    session_id: str | None = None
    tenant_id: str = "default"
    recorded_at: datetime | None = None


@dataclass
class SpendSummary:
    period: str  # "daily" | "monthly"
    period_label: str  # "2026-02-27" | "2026-02"
    total_cost_usd: float
    call_count: int
    cap_usd: float
    is_over_cap: bool

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.total_cost_usd)

    @property
    def utilization_pct(self) -> float:
        if self.cap_usd <= 0:
            return 0.0
        return min(100.0, (self.total_cost_usd / self.cap_usd) * 100)


class CostTracker:
    """Tracks Claude API usage and enforces spending caps.

    In-memory mode (db=None): tracks per-session only, no persistence.
    Database mode: persists to claude_usage_log table.
    """

    def __init__(
        self,
        db: Any = None,
        daily_cap_usd: float = 10.0,
        monthly_cap_usd: float = 50.0,
    ) -> None:
        self._db = db
        self._daily_cap = daily_cap_usd
        self._monthly_cap = monthly_cap_usd
        self._session_total: float = 0.0
        self._session_count: int = 0

    @classmethod
    def from_settings(cls, settings: Any, db: Any = None) -> CostTracker:
        return cls(
            db=db,
            daily_cap_usd=getattr(settings, "claude_enrichment_cost_limit_usd", 10.0),
            monthly_cap_usd=getattr(settings, "claude_query_cost_limit_usd", 50.0),
        )

    def record(self, record: UsageRecord) -> None:
        """Record a usage event. Persists if DB available."""
        self._session_total += record.cost_usd
        self._session_count += 1
        logger.debug(
            "cost_tracker.recorded",
            extra={
                "model": record.model_id,
                "cost_usd": record.cost_usd,
                "operation": record.operation,
                "session_total": self._session_total,
            },
        )
        if self._db is not None:
            self._persist(record)

    def _persist(self, record: UsageRecord) -> None:
        """Persist to claude_usage_log table."""
        try:
            from sqlalchemy import text

            self._db.execute(
                text("""
                    INSERT INTO claude_usage_log
                        (model_id, input_tokens, output_tokens, cost_usd,
                         operation, document_id, session_id, tenant_id, recorded_at)
                    VALUES
                        (:model_id, :input_tokens, :output_tokens, :cost_usd,
                         :operation, :document_id, :session_id, :tenant_id, :recorded_at)
                """),
                {
                    "model_id": record.model_id,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "cost_usd": record.cost_usd,
                    "operation": record.operation,
                    "document_id": record.document_id,
                    "session_id": record.session_id,
                    "tenant_id": record.tenant_id,
                    "recorded_at": record.recorded_at or datetime.utcnow(),
                },
            )
            self._db.commit()
        except Exception as exc:
            logger.warning("cost_tracker.persist_failed", extra={"error": str(exc)})
            if self._db:
                try:
                    self._db.rollback()
                except Exception:
                    pass

    def check_daily_cap(self, tenant_id: str = "default") -> SpendSummary:
        """Return daily spend summary. Uses in-memory if no DB."""
        today = date.today().isoformat()
        if self._db is None:
            return SpendSummary(
                period="daily",
                period_label=today,
                total_cost_usd=self._session_total,
                call_count=self._session_count,
                cap_usd=self._daily_cap,
                is_over_cap=self._session_total >= self._daily_cap,
            )
        total, count = self._query_period(tenant_id, f"{today}%")
        return SpendSummary(
            period="daily",
            period_label=today,
            total_cost_usd=total,
            call_count=count,
            cap_usd=self._daily_cap,
            is_over_cap=total >= self._daily_cap,
        )

    def check_monthly_cap(self, tenant_id: str = "default") -> SpendSummary:
        """Return monthly spend summary."""
        month = date.today().strftime("%Y-%m")
        if self._db is None:
            return SpendSummary(
                period="monthly",
                period_label=month,
                total_cost_usd=self._session_total,
                call_count=self._session_count,
                cap_usd=self._monthly_cap,
                is_over_cap=self._session_total >= self._monthly_cap,
            )
        total, count = self._query_period(tenant_id, f"{month}%")
        return SpendSummary(
            period="monthly",
            period_label=month,
            total_cost_usd=total,
            call_count=count,
            cap_usd=self._monthly_cap,
            is_over_cap=total >= self._monthly_cap,
        )

    def _query_period(self, tenant_id: str, period_prefix: str) -> tuple[float, int]:
        try:
            from sqlalchemy import text

            result = self._db.execute(
                text("""
                    SELECT COALESCE(SUM(cost_usd), 0.0) as total,
                           COUNT(*) as call_count
                    FROM claude_usage_log
                    WHERE tenant_id = :tenant_id
                      AND CAST(recorded_at AS TEXT) LIKE :period_prefix
                """),
                {"tenant_id": tenant_id, "period_prefix": period_prefix},
            )
            row = result.fetchone()
            return float(row.total or 0), int(row.call_count or 0)
        except Exception as exc:
            logger.warning("cost_tracker.query_failed", extra={"error": str(exc)})
            return 0.0, 0

    def is_allowed(self, estimated_cost: float = 0.0, tenant_id: str = "default") -> bool:
        """Return True if a call with estimated_cost is within caps."""
        daily = self.check_daily_cap(tenant_id)
        if daily.total_cost_usd + estimated_cost > daily.cap_usd:
            logger.warning(
                "cost_tracker.daily_cap_exceeded",
                extra={"current": daily.total_cost_usd, "cap": daily.cap_usd},
            )
            return False
        monthly = self.check_monthly_cap(tenant_id)
        if monthly.total_cost_usd + estimated_cost > monthly.cap_usd:
            logger.warning(
                "cost_tracker.monthly_cap_exceeded",
                extra={"current": monthly.total_cost_usd, "cap": monthly.cap_usd},
            )
            return False
        return True

    @property
    def session_total_usd(self) -> float:
        return self._session_total
