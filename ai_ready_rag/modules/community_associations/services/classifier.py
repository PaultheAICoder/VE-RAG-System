"""Community Associations document type classifier.

Implements keyword-based document classification with ambiguity gate.
No database dependency — pure Python classification logic.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result from document type classification."""

    document_type: str
    confidence: float
    display_name: str
    extraction_prompt: str | None = None
    is_ambiguous: bool = False
    ambiguous_candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DocumentTypeConfig:
    """Configuration for a single document type classifier."""

    document_type: str
    display_name: str
    description: str
    confidence_threshold: float
    extraction_prompt: str | None
    filename_patterns: list[str]
    content_keywords: list[str]
    negative_keywords: list[str]


def _score_document(
    cfg: DocumentTypeConfig,
    filename: str,
    content_sample: str,
) -> float:
    """Score a document against one classifier. Returns confidence 0.0–1.0."""
    if cfg.document_type == "unknown":
        return 0.0

    filename_lower = filename.lower()
    content_lower = content_sample.lower()
    score = 0.0

    # Filename match — high signal
    for pattern in cfg.filename_patterns:
        if pattern in filename_lower:
            score += 0.4
            break  # Only count once

    # Content keyword matches — additive but capped
    keyword_hits = sum(1 for kw in cfg.content_keywords if kw in content_lower)
    if cfg.content_keywords:
        keyword_ratio = keyword_hits / len(cfg.content_keywords)
        score += keyword_ratio * 0.7

    # Negative keyword penalty
    neg_hits = sum(1 for kw in cfg.negative_keywords if kw in content_lower)
    score -= neg_hits * 0.2

    return max(0.0, min(1.0, score))


class CADocumentClassifier:
    """Classifies documents using keyword matching against classifiers.yaml.

    Usage:
        classifier = CADocumentClassifier.from_yaml()
        result = classifier.classify("Reserve_Study_2024.pdf", content_text)
    """

    def __init__(self, configs: list[DocumentTypeConfig], ambiguity_threshold: float = 0.10):
        self._configs = configs
        self._ambiguity_threshold = ambiguity_threshold

    @classmethod
    def from_yaml(cls, yaml_path: str | pathlib.Path | None = None) -> CADocumentClassifier:
        """Load classifier from classifiers.yaml (defaults to module's classifiers.yaml)."""
        if yaml_path is None:
            yaml_path = pathlib.Path(__file__).parent.parent / "classifiers.yaml"

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        ambiguity_threshold = data.get("ambiguity_threshold", 0.10)
        configs = []
        for dt in data.get("document_types", []):
            patterns = dt.get("patterns", {})
            configs.append(
                DocumentTypeConfig(
                    document_type=dt["document_type"],
                    display_name=dt["display_name"],
                    description=dt.get("description", ""),
                    confidence_threshold=dt.get("confidence_threshold", 0.75),
                    extraction_prompt=dt.get("extraction_prompt"),
                    filename_patterns=patterns.get("filename", []),
                    content_keywords=patterns.get("content_keywords", []),
                    negative_keywords=patterns.get("negative_keywords", []),
                )
            )
        return cls(configs, ambiguity_threshold)

    def classify(
        self,
        filename: str,
        content_sample: str,
    ) -> ClassificationResult:
        """Classify a document. Returns top type or unknown with ambiguity flag.

        Args:
            filename: Original filename (used for pattern matching)
            content_sample: First N characters of document text (used for keyword matching)

        Returns:
            ClassificationResult with document_type, confidence, and ambiguity details.
        """
        scores: list[tuple[DocumentTypeConfig, float]] = []
        for cfg in self._configs:
            if cfg.document_type == "unknown":
                continue
            score = _score_document(cfg, filename, content_sample)
            if score >= cfg.confidence_threshold:
                scores.append((cfg, score))

        if not scores:
            # Nothing met threshold — return unknown
            unknown = next(c for c in self._configs if c.document_type == "unknown")
            return ClassificationResult(
                document_type="unknown",
                confidence=0.0,
                display_name=unknown.display_name,
            )

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        top_cfg, top_score = scores[0]

        # Ambiguity gate: if top two are within threshold, flag for review
        if len(scores) >= 2:
            second_cfg, second_score = scores[1]
            gap = top_score - second_score
            if gap < self._ambiguity_threshold:
                return ClassificationResult(
                    document_type=top_cfg.document_type,
                    confidence=top_score,
                    display_name=top_cfg.display_name,
                    extraction_prompt=top_cfg.extraction_prompt,
                    is_ambiguous=True,
                    ambiguous_candidates=[
                        {"document_type": top_cfg.document_type, "confidence": top_score},
                        {"document_type": second_cfg.document_type, "confidence": second_score},
                    ],
                )

        return ClassificationResult(
            document_type=top_cfg.document_type,
            confidence=top_score,
            display_name=top_cfg.display_name,
            extraction_prompt=top_cfg.extraction_prompt,
        )
