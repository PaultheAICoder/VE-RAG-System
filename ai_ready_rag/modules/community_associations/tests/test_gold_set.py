"""Tests for CA gold set fixtures and runner."""

import json
from pathlib import Path

from ai_ready_rag.modules.community_associations.tests.gold_set_runner import (
    GoldSetResult,
    GoldSetRunner,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestGoldSetFixtures:
    def test_gold_set_json_exists(self):
        assert (FIXTURES_DIR / "gold_set.json").exists()

    def test_sample_policy_json_exists(self):
        assert (FIXTURES_DIR / "sample_insurance_policy.json").exists()

    def test_gold_set_has_16_questions(self):
        with open(FIXTURES_DIR / "gold_set.json") as f:
            data = json.load(f)
        assert len(data["questions"]) == 16

    def test_gold_set_passing_threshold(self):
        with open(FIXTURES_DIR / "gold_set.json") as f:
            data = json.load(f)
        assert data["passing_threshold"] == 0.70

    def test_sample_policy_has_required_fields(self):
        with open(FIXTURES_DIR / "sample_insurance_policy.json") as f:
            data = json.load(f)
        assert "account_name" in data
        assert "policies" in data
        assert len(data["policies"]) > 0


class TestGoldSetRunner:
    def setup_method(self):
        self.runner = GoldSetRunner()

    def test_evaluate_perfect_score(self):
        """Mock that always returns the exact expected answer."""
        gold_questions = self.runner._gold_set["questions"]
        idx = [0]

        def perfect_fn(question):
            q = gold_questions[idx[0]]
            idx[0] += 1
            return q["expected_answer"]

        result = self.runner.evaluate(perfect_fn)
        assert result.accuracy == 1.0
        assert result.passed_threshold is True

    def test_evaluate_zero_score(self):
        def bad_fn(question):
            return "WRONG ANSWER THAT MATCHES NOTHING XYZ123"

        result = self.runner.evaluate(bad_fn)
        assert result.accuracy == 0.0
        assert result.passed_threshold is False

    def test_evaluate_returns_gold_set_result(self):
        result = self.runner.evaluate(lambda q: "unknown")
        assert isinstance(result, GoldSetResult)
        assert result.total_questions == 16

    def test_summary_contains_score(self):
        result = self.runner.evaluate(lambda q: "unknown")
        summary = result.summary()
        assert "0/16" in summary or "FAIL" in summary

    def test_check_answer_exact(self):
        assert self.runner._check_answer("State Farm", "State Farm", []) is True

    def test_check_answer_case_insensitive(self):
        assert self.runner._check_answer("state farm", "State Farm", []) is True

    def test_check_answer_variant(self):
        assert (
            self.runner._check_answer(
                "State Farm Fire",
                "State Farm Fire and Casualty Company",
                ["State Farm", "State Farm Fire"],
            )
            is True
        )

    def test_check_answer_none(self):
        assert self.runner._check_answer(None, "expected", []) is False
