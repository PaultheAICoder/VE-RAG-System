"""Gold set runner: run the system against known-answer test sets.

Gates deployment by comparing actual answers against expected answers and
checking whether the overall score meets configured accuracy thresholds.

Usage (programmatic)::

    from ai_ready_rag.services.evaluation.gold_set_runner import GoldSetRunner

    runner = GoldSetRunner(
        gold_set_path="tests/gold_sets/marshall_wells.json",
        answer_fn=my_rag_answer_fn,
    )
    report = runner.run()
    print(report.score, report.meets_standard_threshold)

Usage (CLI)::

    python -m ai_ready_rag.cli.evaluate --gold-set tests/gold_sets/marshall_wells.json
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

STANDARD_THRESHOLD = 0.90  # ≥90% accuracy required for standard tier
ENTERPRISE_THRESHOLD = 0.70  # ≥70% accuracy required for enterprise tier


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    """Result for a single gold-set question."""

    id: str
    question: str
    expected_answer: str
    actual_answer: str
    category: str
    weight: float
    passed: bool
    score: float  # 0.0 or 1.0 (partial match → 1.0 / no match → 0.0)


@dataclass
class GoldSetReport:
    """Aggregated report for an entire gold-set run."""

    name: str
    version: str
    total_questions: int
    passed: int
    failed: int
    score: float  # weighted accuracy: 0.0 – 1.0
    question_results: list[QuestionResult] = field(default_factory=list)
    meets_standard_threshold: bool = False  # score >= STANDARD_THRESHOLD
    meets_enterprise_threshold: bool = False  # score >= ENTERPRISE_THRESHOLD

    def summary(self) -> str:
        """Human-readable one-liner summary."""
        standard = "PASS" if self.meets_standard_threshold else "FAIL"
        enterprise = "PASS" if self.meets_enterprise_threshold else "FAIL"
        return (
            f"Gold set '{self.name}' v{self.version}: "
            f"{self.passed}/{self.total_questions} passed "
            f"(score={self.score:.1%}) | "
            f"standard={standard} enterprise={enterprise}"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class GoldSetRunner:
    """Run a gold-set JSON against an answer function and produce a report.

    Parameters
    ----------
    gold_set_path:
        Path to the gold-set JSON file (absolute or relative to CWD).
    answer_fn:
        Callable that accepts a question ``str`` and returns an answer ``str``.
        Typically wraps ``RAGService`` or a simpler stub during testing.
    """

    def __init__(
        self,
        gold_set_path: str,
        answer_fn: Callable[[str], str],
    ) -> None:
        self.answer_fn = answer_fn
        self._gold_set = self._load(gold_set_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> GoldSetReport:
        """Run all questions in the gold set and return a ``GoldSetReport``."""
        questions = self._gold_set.get("questions", [])
        name = self._gold_set.get("name", "unknown")
        version = self._gold_set.get("version", "0.0")

        results: list[QuestionResult] = []
        total_weight = 0.0
        weighted_score = 0.0

        for item in questions:
            qid = item.get("id", "")
            question = item.get("question", "")
            expected = item.get("expected_answer", "")
            category = item.get("category", "")
            weight = float(item.get("weight", 1.0))

            logger.debug("Running question %s: %s", qid, question)

            try:
                actual = self.answer_fn(question)
            except Exception as exc:  # noqa: BLE001
                logger.warning("answer_fn raised for question %s: %s", qid, exc)
                actual = ""

            passed, score = self._check_answer(expected, actual)

            results.append(
                QuestionResult(
                    id=qid,
                    question=question,
                    expected_answer=expected,
                    actual_answer=actual,
                    category=category,
                    weight=weight,
                    passed=passed,
                    score=score,
                )
            )

            total_weight += weight
            weighted_score += score * weight

            logger.debug(
                "Question %s %s (score=%.2f)",
                qid,
                "PASSED" if passed else "FAILED",
                score,
            )

        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        final_score = (weighted_score / total_weight) if total_weight > 0 else 0.0

        report = GoldSetReport(
            name=name,
            version=version,
            total_questions=len(results),
            passed=passed_count,
            failed=failed_count,
            score=final_score,
            question_results=results,
            meets_standard_threshold=final_score >= STANDARD_THRESHOLD,
            meets_enterprise_threshold=final_score >= ENTERPRISE_THRESHOLD,
        )

        logger.info(report.summary())
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: str) -> dict:
        """Load and parse a gold-set JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Gold set file not found: {path}")
        with p.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if "questions" not in data:
            raise ValueError(f"Gold set JSON missing 'questions' key: {path}")
        return data

    @staticmethod
    def _check_answer(expected: str, actual: str) -> tuple[bool, float]:
        """Check whether *actual* satisfies *expected*.

        Matching rules (applied in order):
        1. Case-insensitive substring match: expected contained in actual.
        2. Case-insensitive substring match: actual contained in expected
           (handles abbreviated answers like "$1M" vs "$1,000,000").

        Returns
        -------
        (passed, score)
            ``passed`` is True when any matching rule succeeds.
            ``score`` is 1.0 on pass, 0.0 on fail.
        """
        if not expected:
            # No expected answer defined — treat as inconclusive pass.
            return True, 1.0

        expected_lower = expected.strip().lower()
        actual_lower = actual.strip().lower()

        if not actual_lower:
            return False, 0.0

        passed = expected_lower in actual_lower or actual_lower in expected_lower
        score = 1.0 if passed else 0.0
        return passed, score
