"""Tests for GoldSetRunner.

Covers:
- Happy path: all questions answered correctly
- Partial pass: some questions answered incorrectly
- Below threshold detection (standard and enterprise tiers)
- Answer matching (case-insensitive, partial match, empty answers)
- Gold-set file loading (missing file, missing 'questions' key)
- Weighted scoring
- Question result fields populated correctly
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from ai_ready_rag.services.evaluation.gold_set_runner import (
    ENTERPRISE_THRESHOLD,
    STANDARD_THRESHOLD,
    GoldSetReport,
    GoldSetRunner,
    QuestionResult,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

SAMPLE_GOLD_SET = {
    "name": "Test Gold Set",
    "version": "1.0",
    "questions": [
        {
            "id": "q001",
            "question": "What is the GL limit?",
            "expected_answer": "$1,000,000",
            "category": "coverage",
            "weight": 1.0,
        },
        {
            "id": "q002",
            "question": "What is the D&O limit?",
            "expected_answer": "$500,000",
            "category": "coverage",
            "weight": 1.0,
        },
        {
            "id": "q003",
            "question": "When does the policy expire?",
            "expected_answer": "June 1, 2026",
            "category": "expiration",
            "weight": 1.0,
        },
        {
            "id": "q004",
            "question": "Is the policy Fannie Mae compliant?",
            "expected_answer": "Yes",
            "category": "compliance",
            "weight": 2.0,
        },
    ],
}


def _write_gold_set(data: dict, directory: str) -> str:
    """Write a gold-set dict to a temp JSON file and return the path."""
    path = os.path.join(directory, "gold_set.json")
    Path(path).write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def tmp_dir():
    """Temporary directory cleaned up after each test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def gold_set_path(tmp_dir):
    """Write SAMPLE_GOLD_SET to a temp file and return its path."""
    return _write_gold_set(SAMPLE_GOLD_SET, tmp_dir)


# ---------------------------------------------------------------------------
# Answer functions for testing
# ---------------------------------------------------------------------------

CORRECT_ANSWERS = {
    "What is the GL limit?": "The GL limit is $1,000,000 per occurrence.",
    "What is the D&O limit?": "D&O coverage is $500,000.",
    "When does the policy expire?": "The policy expires on June 1, 2026.",
    "Is the policy Fannie Mae compliant?": "Yes, it meets all Fannie Mae requirements.",
}


def all_correct_fn(question: str) -> str:
    return CORRECT_ANSWERS.get(question, "I don't know.")


def all_wrong_fn(question: str) -> str:
    return "I have no information about that."


def partial_correct_fn(question: str) -> str:
    """Returns correct answers for first two questions only."""
    partial = {
        "What is the GL limit?": "The GL limit is $1,000,000.",
        "What is the D&O limit?": "D&O coverage is $500,000.",
    }
    return partial.get(question, "I don't know.")


def raises_fn(question: str) -> str:
    raise RuntimeError("RAG service is unavailable")


# ---------------------------------------------------------------------------
# Happy path: all questions answered correctly
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_all_correct_returns_report(self, gold_set_path):
        runner = GoldSetRunner(gold_set_path, all_correct_fn)
        report = runner.run()
        assert isinstance(report, GoldSetReport)

    def test_all_correct_score_is_one(self, gold_set_path):
        runner = GoldSetRunner(gold_set_path, all_correct_fn)
        report = runner.run()
        assert report.score == pytest.approx(1.0)

    def test_all_correct_passed_count(self, gold_set_path):
        runner = GoldSetRunner(gold_set_path, all_correct_fn)
        report = runner.run()
        assert report.passed == 4
        assert report.failed == 0

    def test_all_correct_meets_standard_threshold(self, gold_set_path):
        runner = GoldSetRunner(gold_set_path, all_correct_fn)
        report = runner.run()
        assert report.meets_standard_threshold is True

    def test_all_correct_meets_enterprise_threshold(self, gold_set_path):
        runner = GoldSetRunner(gold_set_path, all_correct_fn)
        report = runner.run()
        assert report.meets_enterprise_threshold is True

    def test_report_name_and_version(self, gold_set_path):
        runner = GoldSetRunner(gold_set_path, all_correct_fn)
        report = runner.run()
        assert report.name == "Test Gold Set"
        assert report.version == "1.0"

    def test_question_results_populated(self, gold_set_path):
        runner = GoldSetRunner(gold_set_path, all_correct_fn)
        report = runner.run()
        assert len(report.question_results) == 4
        for qr in report.question_results:
            assert isinstance(qr, QuestionResult)
            assert qr.passed is True
            assert qr.score == pytest.approx(1.0)

    def test_question_result_fields(self, gold_set_path):
        runner = GoldSetRunner(gold_set_path, all_correct_fn)
        report = runner.run()
        first = report.question_results[0]
        assert first.id == "q001"
        assert first.question == "What is the GL limit?"
        assert first.expected_answer == "$1,000,000"
        assert first.category == "coverage"
        assert first.weight == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# All wrong: score 0
# ---------------------------------------------------------------------------


class TestAllWrong:
    def test_all_wrong_score_is_zero(self, gold_set_path):
        runner = GoldSetRunner(gold_set_path, all_wrong_fn)
        report = runner.run()
        assert report.score == pytest.approx(0.0)

    def test_all_wrong_failed_count(self, gold_set_path):
        runner = GoldSetRunner(gold_set_path, all_wrong_fn)
        report = runner.run()
        assert report.failed == 4
        assert report.passed == 0

    def test_all_wrong_does_not_meet_thresholds(self, gold_set_path):
        runner = GoldSetRunner(gold_set_path, all_wrong_fn)
        report = runner.run()
        assert report.meets_standard_threshold is False
        assert report.meets_enterprise_threshold is False


# ---------------------------------------------------------------------------
# Partial pass
# ---------------------------------------------------------------------------


class TestPartialPass:
    def test_partial_score(self, gold_set_path):
        """2 of 4 questions correct; q004 has weight=2.0 so total_weight=5.0."""
        runner = GoldSetRunner(gold_set_path, partial_correct_fn)
        report = runner.run()
        # q001 weight=1, q002 weight=1 pass → weighted_score=2.0
        # total_weight = 1+1+1+2 = 5.0
        # score = 2.0/5.0 = 0.4
        assert report.score == pytest.approx(0.4)

    def test_partial_passed_count(self, gold_set_path):
        runner = GoldSetRunner(gold_set_path, partial_correct_fn)
        report = runner.run()
        assert report.passed == 2
        assert report.failed == 2

    def test_partial_below_standard_threshold(self, gold_set_path):
        runner = GoldSetRunner(gold_set_path, partial_correct_fn)
        report = runner.run()
        assert report.meets_standard_threshold is False

    def test_partial_below_enterprise_threshold(self, gold_set_path):
        runner = GoldSetRunner(gold_set_path, partial_correct_fn)
        report = runner.run()
        assert report.meets_enterprise_threshold is False


# ---------------------------------------------------------------------------
# Threshold boundary conditions
# ---------------------------------------------------------------------------


class TestThresholdDetection:
    def _make_n_of_10_correct(self, tmp_dir: str, n_correct: int) -> str:
        """Gold set with 10 equal-weight questions; first n_correct have a known answer."""
        questions = []
        for i in range(10):
            questions.append(
                {
                    "id": f"q{i + 1:03d}",
                    "question": f"question {i + 1}",
                    "expected_answer": "answer" if i < n_correct else "unique_xyz",
                    "category": "test",
                    "weight": 1.0,
                }
            )
        data = {"name": "Threshold Test", "version": "1.0", "questions": questions}
        return _write_gold_set(data, tmp_dir)

    def _answer_fn_for_first_n(self, n: int) -> callable:
        def fn(question: str) -> str:
            # Questions 1..n have expected_answer="answer"
            idx = int(question.split()[-1])
            if idx <= n:
                return "The answer is: answer"
            return "I don't know."

        return fn

    def test_exactly_90_percent_meets_standard(self, tmp_dir):
        path = self._make_n_of_10_correct(tmp_dir, 9)
        runner = GoldSetRunner(path, self._answer_fn_for_first_n(9))
        report = runner.run()
        assert report.score == pytest.approx(0.9)
        assert report.meets_standard_threshold is True

    def test_below_90_fails_standard(self, tmp_dir):
        path = self._make_n_of_10_correct(tmp_dir, 8)
        runner = GoldSetRunner(path, self._answer_fn_for_first_n(8))
        report = runner.run()
        assert report.score == pytest.approx(0.8)
        assert report.meets_standard_threshold is False

    def test_exactly_70_percent_meets_enterprise(self, tmp_dir):
        path = self._make_n_of_10_correct(tmp_dir, 7)
        runner = GoldSetRunner(path, self._answer_fn_for_first_n(7))
        report = runner.run()
        assert report.score == pytest.approx(0.7)
        assert report.meets_enterprise_threshold is True

    def test_below_70_fails_enterprise(self, tmp_dir):
        path = self._make_n_of_10_correct(tmp_dir, 6)
        runner = GoldSetRunner(path, self._answer_fn_for_first_n(6))
        report = runner.run()
        assert report.score == pytest.approx(0.6)
        assert report.meets_enterprise_threshold is False

    def test_threshold_constants(self):
        assert pytest.approx(0.90) == STANDARD_THRESHOLD
        assert pytest.approx(0.70) == ENTERPRISE_THRESHOLD


# ---------------------------------------------------------------------------
# Answer matching
# ---------------------------------------------------------------------------


class TestAnswerMatching:
    def _check(self, expected: str, actual: str) -> tuple[bool, float]:
        return GoldSetRunner._check_answer(expected, actual)

    def test_exact_match(self):
        passed, score = self._check("$1,000,000", "$1,000,000")
        assert passed is True
        assert score == pytest.approx(1.0)

    def test_case_insensitive_match(self):
        passed, score = self._check("yes", "YES, it complies.")
        assert passed is True

    def test_partial_match_expected_in_actual(self):
        passed, score = self._check("$1,000,000", "The GL limit is $1,000,000 per occurrence.")
        assert passed is True

    def test_partial_match_actual_in_expected(self):
        """Short actual contained within longer expected."""
        passed, score = self._check("June 1, 2026", "June 1")
        assert passed is True

    def test_no_match_returns_false(self):
        passed, score = self._check("$1,000,000", "I have no information.")
        assert passed is False
        assert score == pytest.approx(0.0)

    def test_empty_actual_returns_false(self):
        passed, score = self._check("$1,000,000", "")
        assert passed is False

    def test_empty_expected_returns_pass(self):
        """No expected answer defined — should be treated as inconclusive pass."""
        passed, score = self._check("", "anything")
        assert passed is True
        assert score == pytest.approx(1.0)

    def test_whitespace_stripped(self):
        passed, score = self._check("  $1,000,000  ", "  The limit is $1,000,000.  ")
        assert passed is True


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_missing_file_raises_file_not_found(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            GoldSetRunner(os.path.join(tmp_dir, "nonexistent.json"), all_correct_fn)

    def test_missing_questions_key_raises_value_error(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.json")
        Path(path).write_text(json.dumps({"name": "bad", "version": "1.0"}), encoding="utf-8")
        with pytest.raises(ValueError, match="'questions'"):
            GoldSetRunner(path, all_correct_fn)

    def test_answer_fn_exception_treated_as_empty(self, gold_set_path):
        """If answer_fn raises, the question should be treated as unanswered (fail)."""
        runner = GoldSetRunner(gold_set_path, raises_fn)
        report = runner.run()
        # All questions should fail because answer_fn raises
        assert report.passed == 0
        assert report.score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Weighted scoring
# ---------------------------------------------------------------------------


class TestWeightedScoring:
    def test_heavier_question_affects_score_more(self, tmp_dir):
        """Question with weight=3.0 that passes should raise the overall score."""
        data = {
            "name": "Weight Test",
            "version": "1.0",
            "questions": [
                {
                    "id": "q001",
                    "question": "easy question",
                    "expected_answer": "correct",
                    "category": "test",
                    "weight": 3.0,
                },
                {
                    "id": "q002",
                    "question": "hard question",
                    "expected_answer": "exact_phrase_xyz",
                    "category": "test",
                    "weight": 1.0,
                },
            ],
        }
        path = _write_gold_set(data, tmp_dir)

        def answer_fn(question: str) -> str:
            if question == "easy question":
                return "correct answer"
            return "wrong"

        runner = GoldSetRunner(path, answer_fn)
        report = runner.run()
        # weight=3 passes, weight=1 fails → score = 3/(3+1) = 0.75
        assert report.score == pytest.approx(0.75)
        assert report.meets_enterprise_threshold is True
        assert report.meets_standard_threshold is False


# ---------------------------------------------------------------------------
# Marshall Wells gold set (integration-style, no live backend)
# ---------------------------------------------------------------------------


class TestMarshallWellsGoldSet:
    """Smoke-test the committed marshall_wells.json gold set."""

    GOLD_SET_PATH = Path(__file__).parent / "gold_sets" / "marshall_wells.json"

    def test_file_exists(self):
        assert self.GOLD_SET_PATH.exists(), f"Gold set not found: {self.GOLD_SET_PATH}"

    def test_file_is_valid_json(self):
        data = json.loads(self.GOLD_SET_PATH.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_has_required_fields(self):
        data = json.loads(self.GOLD_SET_PATH.read_text(encoding="utf-8"))
        assert "name" in data
        assert "version" in data
        assert "questions" in data

    def test_has_ten_questions(self):
        data = json.loads(self.GOLD_SET_PATH.read_text(encoding="utf-8"))
        assert len(data["questions"]) == 10

    def test_each_question_has_required_fields(self):
        data = json.loads(self.GOLD_SET_PATH.read_text(encoding="utf-8"))
        for q in data["questions"]:
            assert "id" in q, f"Missing 'id' in {q}"
            assert "question" in q, f"Missing 'question' in {q}"
            assert "expected_answer" in q, f"Missing 'expected_answer' in {q}"
            assert "category" in q, f"Missing 'category' in {q}"
            assert "weight" in q, f"Missing 'weight' in {q}"

    def test_runner_loads_without_error(self):
        runner = GoldSetRunner(str(self.GOLD_SET_PATH), all_wrong_fn)
        assert runner is not None

    def test_runner_produces_report_with_stub_answers(self):
        runner = GoldSetRunner(str(self.GOLD_SET_PATH), all_wrong_fn)
        report = runner.run()
        assert report.total_questions == 10
        assert report.passed == 0
        assert report.score == pytest.approx(0.0)

    def test_summary_string_contains_name(self):
        runner = GoldSetRunner(str(self.GOLD_SET_PATH), all_wrong_fn)
        report = runner.run()
        assert "Marshall Wells" in report.summary()
