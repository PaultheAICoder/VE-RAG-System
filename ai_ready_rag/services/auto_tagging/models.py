"""Pydantic models for auto-tagging strategy configuration and runtime tags."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class NamespaceConfig(BaseModel):
    """Configuration for a tag namespace."""

    model_config = ConfigDict(extra="forbid")

    display: str
    color: str = "#6B7280"
    description: str = ""


class DocumentTypeConfig(BaseModel):
    """Configuration for a document type in the taxonomy."""

    model_config = ConfigDict(extra="forbid")

    display: str
    description: str = ""
    keywords: list[str] = []


class EntityExtractionConfig(BaseModel):
    """Configuration for domain-specific entity extraction."""

    model_config = ConfigDict(extra="forbid")

    namespace: str
    prompt_label: str
    prompt_instruction: str
    aliases: dict[str, str] = {}


class TopicValue(BaseModel):
    """A single topic value with keywords for matching."""

    model_config = ConfigDict(extra="forbid")

    id: str
    display: str
    keywords: list[str] = []


class TopicExtractionConfig(BaseModel):
    """Configuration for topic extraction from document content."""

    model_config = ConfigDict(extra="forbid")

    namespace: str
    prompt_label: str
    prompt_instruction: str
    values: list[TopicValue] = []


class EmailPatternTag(BaseModel):
    """A tag to apply when an email pattern matches."""

    model_config = ConfigDict(extra="forbid")

    namespace: str
    value: str


class EmailPattern(BaseModel):
    """A regex pattern for matching email subjects to tags."""

    model_config = ConfigDict(extra="forbid")

    pattern: str
    tags: list[EmailPatternTag]


class StrategyMetadata(BaseModel):
    """Metadata block for a strategy YAML file."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    version: str
    description: str = ""


_NAMESPACE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_DOCTYPE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class StrategyYAML(BaseModel):
    """Top-level validation model for strategy YAML files."""

    model_config = ConfigDict(extra="forbid")

    strategy: StrategyMetadata
    namespaces: dict[str, NamespaceConfig]
    document_types: dict[str, DocumentTypeConfig]
    llm_prompt: str
    path_rules: list[dict] = []
    entity_extraction: EntityExtractionConfig | None = None
    topic_extraction: TopicExtractionConfig | None = None
    email_patterns: list[EmailPattern] = []

    @field_validator("namespaces")
    @classmethod
    def validate_namespaces(cls, v: dict[str, NamespaceConfig]) -> dict[str, NamespaceConfig]:
        """Validate namespace IDs are lowercase alphanumeric, max 20 chars."""
        for ns_id in v:
            if len(ns_id) > 20:
                msg = f"Namespace ID '{ns_id}' exceeds 20 characters"
                raise ValueError(msg)
            if not _NAMESPACE_ID_RE.match(ns_id):
                msg = f"Namespace ID '{ns_id}' must match [a-z][a-z0-9_]*"
                raise ValueError(msg)
        return v

    @field_validator("document_types")
    @classmethod
    def validate_document_types(
        cls, v: dict[str, DocumentTypeConfig]
    ) -> dict[str, DocumentTypeConfig]:
        """Validate document type IDs and ensure 'unknown' exists."""
        if "unknown" not in v:
            msg = "document_types must contain an 'unknown' entry"
            raise ValueError(msg)
        for dt_id in v:
            if len(dt_id) > 30:
                msg = f"Document type ID '{dt_id}' exceeds 30 characters"
                raise ValueError(msg)
            if not _DOCTYPE_ID_RE.match(dt_id):
                msg = f"Document type ID '{dt_id}' must match [a-z][a-z0-9_]*"
                raise ValueError(msg)
        return v

    @field_validator("llm_prompt")
    @classmethod
    def validate_llm_prompt(cls, v: str) -> str:
        """Ensure LLM prompt contains required placeholder."""
        if "{document_type_ids}" not in v:
            msg = "llm_prompt must contain {document_type_ids} placeholder"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_path_rules(self) -> StrategyYAML:
        """Validate path rule structure."""
        for i, rule in enumerate(self.path_rules):
            if "namespace" not in rule:
                msg = f"Path rule {i} missing required 'namespace' field"
                raise ValueError(msg)
            if not isinstance(rule.get("namespace"), str):
                msg = f"Path rule {i} 'namespace' must be a string"
                raise ValueError(msg)
            if "level" not in rule:
                msg = f"Path rule {i} missing required 'level' field"
                raise ValueError(msg)
            level = rule.get("level")
            if not isinstance(level, int) or level < 0:
                msg = f"Path rule {i} 'level' must be a non-negative integer"
                raise ValueError(msg)
        return self


class AutoTag(BaseModel):
    """A tag produced by the auto-tagging pipeline."""

    namespace: str
    value: str
    source: Literal["path", "llm", "email", "manual"]
    confidence: float = 1.0
    strategy_id: str = ""
    strategy_version: str = ""

    @property
    def tag_name(self) -> str:
        """Full namespaced tag name for DB storage."""
        return f"{self.namespace}:{self.value}"

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        return self.value.replace("-", " ").title()
