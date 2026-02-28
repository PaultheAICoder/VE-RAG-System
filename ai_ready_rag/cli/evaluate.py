"""CLI: run the system against a gold-set JSON and report pass/fail.

Usage::

    python -m ai_ready_rag.cli.evaluate --gold-set tests/gold_sets/marshall_wells.json

The command starts without a live RAG backend by default (useful for CI).
Pass ``--live`` to use a running Ollama/Qdrant instance.

Exit codes
----------
0  All thresholds met (or ``--tier`` not specified).
1  Threshold not met.
2  File not found / JSON error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a gold-set evaluation against the RAG system.",
        prog="python -m ai_ready_rag.cli.evaluate",
    )
    parser.add_argument(
        "--gold-set",
        required=True,
        metavar="PATH",
        help="Path to gold-set JSON file (e.g. tests/gold_sets/marshall_wells.json)",
    )
    parser.add_argument(
        "--tier",
        choices=["standard", "enterprise"],
        default=None,
        help=(
            "Gate tier to enforce. "
            "'standard' requires >=90%% accuracy; "
            "'enterprise' requires >=70%% accuracy. "
            "If omitted, the report is printed but no gate is enforced."
        ),
    )
    parser.add_argument(
        "--output-json",
        metavar="PATH",
        default=None,
        help="Write the full report as JSON to this file.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def _stub_answer_fn(question: str) -> str:
    """Placeholder answer function used when --live is not set.

    Returns an empty string so every question fails, which is useful
    for verifying the CLI plumbing and gold-set JSON format without
    requiring a running RAG backend.
    """
    return ""


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Import runner (deferred so the module is importable without heavy deps)
    # ------------------------------------------------------------------
    try:
        from ai_ready_rag.services.evaluation.gold_set_runner import GoldSetRunner
    except ImportError as exc:
        logger.error("Could not import GoldSetRunner: %s", exc)
        sys.exit(2)

    # ------------------------------------------------------------------
    # Validate gold-set path
    # ------------------------------------------------------------------
    gold_set_path = Path(args.gold_set)
    if not gold_set_path.exists():
        logger.error("Gold set file not found: %s", gold_set_path)
        sys.exit(2)

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    try:
        runner = GoldSetRunner(
            gold_set_path=str(gold_set_path),
            answer_fn=_stub_answer_fn,
        )
        report = runner.run()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Failed to run gold set: %s", exc)
        sys.exit(2)

    # ------------------------------------------------------------------
    # Print per-question results
    # ------------------------------------------------------------------
    print()
    print(f"Gold Set: {report.name}  (v{report.version})")
    print(f"{'─' * 60}")
    for qr in report.question_results:
        status_label = "PASS" if qr.passed else "FAIL"
        print(
            f"  [{status_label}] {qr.id:8s} | cat={qr.category:20s} | "
            f"expected={qr.expected_answer!r}"
        )
        if not qr.passed:
            print(f"            actual={qr.actual_answer!r}")
    print(f"{'─' * 60}")
    print(report.summary())
    print()

    # ------------------------------------------------------------------
    # Optional JSON output
    # ------------------------------------------------------------------
    if args.output_json:
        output_path = Path(args.output_json)
        output_data = {
            "name": report.name,
            "version": report.version,
            "total_questions": report.total_questions,
            "passed": report.passed,
            "failed": report.failed,
            "score": report.score,
            "meets_standard_threshold": report.meets_standard_threshold,
            "meets_enterprise_threshold": report.meets_enterprise_threshold,
            "question_results": [
                {
                    "id": qr.id,
                    "question": qr.question,
                    "category": qr.category,
                    "weight": qr.weight,
                    "passed": qr.passed,
                    "score": qr.score,
                    "expected_answer": qr.expected_answer,
                    "actual_answer": qr.actual_answer,
                }
                for qr in report.question_results
            ],
        }
        output_path.write_text(
            json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Report written to: {output_path}")

    # ------------------------------------------------------------------
    # Gate enforcement
    # ------------------------------------------------------------------
    if args.tier == "standard" and not report.meets_standard_threshold:
        print(f"GATE FAILED: standard tier requires >={90}%% accuracy, got {report.score:.1%}")
        sys.exit(1)
    elif args.tier == "enterprise" and not report.meets_enterprise_threshold:
        print(f"GATE FAILED: enterprise tier requires >={70}%% accuracy, got {report.score:.1%}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
