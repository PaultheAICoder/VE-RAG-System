"""Tests for CostTracker."""

import pytest

from ai_ready_rag.services.cost_tracker import CostTracker, SpendSummary, UsageRecord


class TestUsageRecord:
    def test_dataclass(self):
        r = UsageRecord(
            model_id="claude-haiku-4-5-20251001",
            input_tokens=1000,
            output_tokens=200,
            cost_usd=0.0008,
            operation="query",
        )
        assert r.cost_usd == 0.0008
        assert r.tenant_id == "default"


class TestSpendSummary:
    def test_remaining_usd(self):
        s = SpendSummary(
            period="daily",
            period_label="2026-02-27",
            total_cost_usd=3.0,
            call_count=10,
            cap_usd=10.0,
            is_over_cap=False,
        )
        assert s.remaining_usd == 7.0

    def test_utilization_pct(self):
        s = SpendSummary(
            period="daily",
            period_label="2026-02-27",
            total_cost_usd=5.0,
            call_count=5,
            cap_usd=10.0,
            is_over_cap=False,
        )
        assert s.utilization_pct == 50.0

    def test_over_cap(self):
        s = SpendSummary(
            period="daily",
            period_label="2026-02-27",
            total_cost_usd=12.0,
            call_count=50,
            cap_usd=10.0,
            is_over_cap=True,
        )
        assert s.remaining_usd == 0.0


class TestCostTrackerInMemory:
    def test_record_increments_session_total(self):
        tracker = CostTracker(db=None, daily_cap_usd=10.0)
        tracker.record(
            UsageRecord(
                model_id="claude-haiku-4-5-20251001",
                input_tokens=1000,
                output_tokens=100,
                cost_usd=0.0008,
                operation="query",
            )
        )
        assert tracker.session_total_usd == pytest.approx(0.0008)

    def test_is_allowed_within_cap(self):
        tracker = CostTracker(db=None, daily_cap_usd=10.0, monthly_cap_usd=50.0)
        assert tracker.is_allowed(0.5) is True

    def test_is_allowed_over_daily_cap(self):
        tracker = CostTracker(db=None, daily_cap_usd=1.0, monthly_cap_usd=50.0)
        tracker.record(
            UsageRecord(
                model_id="test",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.9,
                operation="query",
            )
        )
        assert tracker.is_allowed(0.2) is False

    def test_from_settings(self):
        from unittest.mock import MagicMock

        settings = MagicMock()
        settings.claude_enrichment_cost_limit_usd = 5.0
        settings.claude_query_cost_limit_usd = 25.0
        tracker = CostTracker.from_settings(settings)
        assert tracker._daily_cap == 5.0
        assert tracker._monthly_cap == 25.0

    def test_check_daily_cap_returns_summary(self):
        tracker = CostTracker(db=None, daily_cap_usd=10.0)
        summary = tracker.check_daily_cap()
        assert isinstance(summary, SpendSummary)
        assert summary.period == "daily"
        assert summary.cap_usd == 10.0
