"""Tests for CA compliance engine."""

from ai_ready_rag.modules.community_associations.services.compliance import (
    ComplianceEngine,
    ComplianceReport,
    ComplianceStandard,
)


class TestPromptInjection:
    def test_fannie_mae_injection_contains_requirements(self):
        engine = ComplianceEngine()
        text = engine.inject_prompt(ComplianceStandard.FANNIE_MAE)
        assert "Fannie Mae" in text
        assert "$1,000,000" in text

    def test_fha_injection_contains_requirements(self):
        engine = ComplianceEngine()
        text = engine.inject_prompt(ComplianceStandard.FHA)
        assert "FHA" in text

    def test_both_injection_contains_both(self):
        engine = ComplianceEngine()
        text = engine.inject_prompt(ComplianceStandard.BOTH)
        assert "Fannie Mae" in text
        assert "FHA" in text

    def test_injection_is_nonempty_string(self):
        engine = ComplianceEngine()
        text = engine.inject_prompt()
        assert isinstance(text, str) and len(text) > 50


class TestComplianceCheck:
    def setup_method(self):
        self.engine = ComplianceEngine()
        self.full_coverage = [
            {"coverage_line": "property", "limit_amount": 5_000_000, "deductible_amount": 25_000},
            {"coverage_line": "general_liability", "limit_amount": 1_000_000},
            {"coverage_line": "fidelity", "limit_amount": 50_000},
        ]

    def test_compliant_with_full_coverage(self):
        report = self.engine.check(self.full_coverage, account_name="Test HOA")
        assert isinstance(report, ComplianceReport)
        assert report.is_compliant is True
        assert report.gap_count == 0

    def test_missing_gl_creates_gap(self):
        coverage_without_gl = [
            c for c in self.full_coverage if c["coverage_line"] != "general_liability"
        ]
        report = self.engine.check(coverage_without_gl)
        assert report.gap_count > 0
        gap_lines = [g.coverage_line for g in report.gaps]
        assert "general_liability" in gap_lines

    def test_gl_below_minimum_creates_gap(self):
        coverage = [
            {"coverage_line": "property", "limit_amount": 5_000_000},
            {"coverage_line": "general_liability", "limit_amount": 500_000},  # below $1M
            {"coverage_line": "fidelity", "limit_amount": 50_000},
        ]
        report = self.engine.check(coverage, standard=ComplianceStandard.FANNIE_MAE)
        below_min_gaps = [g for g in report.gaps if g.gap_type == "below_minimum"]
        assert len(below_min_gaps) > 0

    def test_flood_not_required_outside_sfha(self):
        report = self.engine.check(self.full_coverage, in_sfha=False)
        flood_gaps = [g for g in report.gaps if g.coverage_line == "flood"]
        assert len(flood_gaps) == 0

    def test_flood_required_inside_sfha(self):
        report = self.engine.check(self.full_coverage, in_sfha=True)
        flood_gaps = [g for g in report.gaps if g.coverage_line == "flood"]
        assert len(flood_gaps) > 0

    def test_report_has_account_name(self):
        report = self.engine.check(self.full_coverage, account_name="Oak Hills HOA")
        assert report.account_name == "Oak Hills HOA"

    def test_high_severity_count(self):
        report = self.engine.check([], account_name="Empty HOA")
        assert report.high_severity_count >= 2  # missing property + GL at minimum
