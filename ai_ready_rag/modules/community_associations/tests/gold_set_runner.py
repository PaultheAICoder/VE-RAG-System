"""CA gold set evaluation harness.

Evaluates CA module extraction accuracy against the Marshall Wells
16-question gold set. Ship gate: >=70% accuracy required before merge.

Usage:
    python -m ai_ready_rag.modules.community_associations.tests.gold_set_runner
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)

GOLD_SET_PATH = Path(__file__).parent / "fixtures" / "gold_set.json"
SAMPLE_POLICY_PATH = Path(__file__).parent / "fixtures" / "sample_insurance_policy.json"


@dataclass
class QuestionResult:
    question_id: str
    question: str
    expected: str
    actual: str | None
    passed: bool
    confidence: float = 0.0
    method: str = "exact"


@dataclass
class GoldSetResult:
    total_questions: int
    passed_questions: int
    accuracy: float
    passing_threshold: float
    passed_threshold: bool
    results: list[QuestionResult] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed_threshold else "FAIL"
        return (
            f"Gold Set Evaluation: {status}\n"
            f"Score: {self.passed_questions}/{self.total_questions} "
            f"({self.accuracy:.1%}) — threshold: {self.passing_threshold:.0%}"
        )


class GoldSetRunner:
    """Runs the CA gold set evaluation harness.

    Evaluates against structured data (JSON) to avoid needing
    a live DB or Claude API. Tests extraction/canonicalization logic.
    """

    def __init__(self, gold_set_path: Path = GOLD_SET_PATH) -> None:
        with open(gold_set_path) as f:
            self._gold_set = json.load(f)

    def evaluate(self, answer_fn: Any) -> GoldSetResult:
        """Evaluate all questions using answer_fn(question, context) -> str."""
        questions = self._gold_set["questions"]
        threshold = self._gold_set.get("passing_threshold", 0.70)

        results = []
        for q in questions:
            try:
                actual = answer_fn(q["question"])
                passed = self._check_answer(
                    actual, q["expected_answer"], q.get("acceptable_variants", [])
                )
                results.append(
                    QuestionResult(
                        question_id=q["id"],
                        question=q["question"],
                        expected=q["expected_answer"],
                        actual=actual,
                        passed=passed,
                    )
                )
            except Exception as exc:
                results.append(
                    QuestionResult(
                        question_id=q["id"],
                        question=q["question"],
                        expected=q["expected_answer"],
                        actual=None,
                        passed=False,
                        method=f"error: {exc}",
                    )
                )

        passed_count = sum(1 for r in results if r.passed)
        accuracy = passed_count / len(questions) if questions else 0.0

        return GoldSetResult(
            total_questions=len(questions),
            passed_questions=passed_count,
            accuracy=accuracy,
            passing_threshold=threshold,
            passed_threshold=accuracy >= threshold,
            results=results,
        )

    def _check_answer(self, actual: str | None, expected: str, variants: list[str]) -> bool:
        """Check if actual answer matches expected or any variant."""
        if actual is None:
            return False
        actual_lower = str(actual).lower().strip()
        if actual_lower == expected.lower():
            return True
        for variant in variants:
            if variant.lower() in actual_lower or actual_lower in variant.lower():
                return True
        return False


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    runner = GoldSetRunner()

    # Load sample policy for a simple demo answer function
    with open(SAMPLE_POLICY_PATH) as f:
        policy = json.load(f)

    def _demo_answer_fn(question: str) -> str:
        """Minimal demo: returns placeholder. Replace with real extractor."""
        return "unknown"

    result = runner.evaluate(_demo_answer_fn)
    click.echo(result.summary())
    click.echo("")
    for r in result.results:
        status = "PASS" if r.passed else "FAIL"
        click.echo(f"  [{status}] {r.question_id}: expected={r.expected!r}  actual={r.actual!r}")

    sys.exit(0 if result.passed_threshold else 1)
