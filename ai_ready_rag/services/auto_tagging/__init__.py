"""Auto-tagging strategy engine for hybrid path-based and LLM-based tagging."""

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
    "PathRule",
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
