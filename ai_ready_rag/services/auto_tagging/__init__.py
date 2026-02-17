"""Auto-tagging strategy engine for hybrid path-based and LLM-based tagging."""

from ai_ready_rag.services.auto_tagging.classifier import ClassificationResult, DocumentClassifier
from ai_ready_rag.services.auto_tagging.conflict import (
    build_provenance,
    enforce_guardrail,
    resolve_conflicts,
)
from ai_ready_rag.services.auto_tagging.email_processor import EmailProcessingResult, EmailProcessor
from ai_ready_rag.services.auto_tagging.models import (
    AutoTag,
    DocumentTypeConfig,
    EmailPattern,
    EmailPatternTag,
    EntityExtractionConfig,
    NamespaceConfig,
    StrategyMetadata,
    StrategyYAML,
    TopicExtractionConfig,
    TopicValue,
)
from ai_ready_rag.services.auto_tagging.strategy import AutoTagStrategy, PathRule
from ai_ready_rag.services.auto_tagging.transforms import (
    TRANSFORM_REGISTRY,
    get_transform,
    identity,
    lowercase,
    resolve_entity,
    slugify,
    year_range,
)

__all__ = [
    # Strategy
    "AutoTagStrategy",
    "ClassificationResult",
    "DocumentClassifier",
    "PathRule",
    # Email processor
    "EmailProcessingResult",
    "EmailProcessor",
    # Conflict resolution
    "build_provenance",
    "enforce_guardrail",
    "resolve_conflicts",
    # Models
    "AutoTag",
    "DocumentTypeConfig",
    "EmailPattern",
    "EmailPatternTag",
    "EntityExtractionConfig",
    "NamespaceConfig",
    "StrategyMetadata",
    "StrategyYAML",
    "TopicExtractionConfig",
    "TopicValue",
    # Transforms
    "TRANSFORM_REGISTRY",
    "get_transform",
    "identity",
    "lowercase",
    "resolve_entity",
    "slugify",
    "year_range",
]
