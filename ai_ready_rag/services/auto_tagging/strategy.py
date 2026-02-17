"""AutoTagStrategy class and PathRule dataclass for the auto-tagging engine."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from ai_ready_rag.services.auto_tagging.models import (
    AutoTag,
    DocumentTypeConfig,
    EmailPattern,
    EntityExtractionConfig,
    NamespaceConfig,
    StrategyYAML,
    TopicExtractionConfig,
)
from ai_ready_rag.services.auto_tagging.transforms import get_transform, resolve_entity

logger = logging.getLogger(__name__)


@dataclass
class PathRule:
    """A rule for extracting a tag from a specific folder level in a path."""

    namespace: str
    level: int
    pattern: str | None = None
    capture_group: int = 1
    transform: str | None = None
    mapping: dict[str, str] | None = None
    parent_match: str | None = None
    _compiled_pattern: re.Pattern | None = field(default=None, repr=False, compare=False)
    _compiled_parent: re.Pattern | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Compile regex patterns after initialization."""
        if self.pattern is not None:
            self._compiled_pattern = re.compile(self.pattern)
        if self.parent_match is not None:
            self._compiled_parent = re.compile(self.parent_match)

    @classmethod
    def from_dict(cls, data: dict) -> PathRule:
        """Create a PathRule from a dictionary, validating required fields.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        namespace = data.get("namespace")
        if not isinstance(namespace, str):
            msg = "PathRule requires a string 'namespace'"
            raise ValueError(msg)
        level = data.get("level")
        if not isinstance(level, int) or level < 0:
            msg = "PathRule requires a non-negative integer 'level'"
            raise ValueError(msg)
        return cls(
            namespace=namespace,
            level=level,
            pattern=data.get("pattern"),
            capture_group=data.get("capture_group", 1),
            transform=data.get("transform"),
            mapping=data.get("mapping"),
            parent_match=data.get("parent_match"),
        )


class AutoTagStrategy:
    """Loaded from YAML, drives both path-based and LLM-based tagging."""

    def __init__(
        self,
        *,
        id: str,
        name: str,
        description: str,
        version: str,
        namespaces: dict[str, NamespaceConfig],
        path_rules: list[PathRule],
        document_types: dict[str, DocumentTypeConfig],
        entity_extraction: EntityExtractionConfig | None,
        topic_extraction: TopicExtractionConfig | None,
        llm_prompt_template: str,
        email_patterns: list[EmailPattern],
    ) -> None:
        self.id = id
        self.name = name
        self.description = description
        self.version = version
        self.namespaces = namespaces
        self.path_rules = path_rules
        self.document_types = document_types
        self.entity_extraction = entity_extraction
        self.topic_extraction = topic_extraction
        self.llm_prompt_template = llm_prompt_template
        self.email_patterns = email_patterns

    @classmethod
    def load(cls, path: str) -> AutoTagStrategy:
        """Load a strategy from a YAML file.

        Validates the YAML against the StrategyYAML schema, checks that
        strategy.id matches the filename, and compiles PathRule regex patterns.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If validation fails (schema, id mismatch, etc.).
        """
        filepath = Path(path)
        if not filepath.exists():
            msg = f"Strategy file not found: {path}"
            raise FileNotFoundError(msg)

        with open(filepath) as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            msg = f"Strategy file must contain a YAML mapping, got {type(raw).__name__}"
            raise ValueError(msg)

        try:
            validated = StrategyYAML(**raw)
        except ValidationError as e:
            msg = f"Strategy validation failed for {filepath.name}: {e}"
            raise ValueError(msg) from e

        expected_id = filepath.stem
        if validated.strategy.id != expected_id:
            msg = f"Strategy ID '{validated.strategy.id}' does not match filename '{expected_id}'"
            raise ValueError(msg)

        compiled_rules: list[PathRule] = []
        for rule_dict in validated.path_rules:
            try:
                compiled_rules.append(PathRule.from_dict(rule_dict))
            except (ValueError, re.error) as e:
                msg = f"Invalid path rule: {e}"
                raise ValueError(msg) from e

        return cls(
            id=validated.strategy.id,
            name=validated.strategy.name,
            description=validated.strategy.description,
            version=validated.strategy.version,
            namespaces=validated.namespaces,
            path_rules=compiled_rules,
            document_types=validated.document_types,
            entity_extraction=validated.entity_extraction,
            topic_extraction=validated.topic_extraction,
            llm_prompt_template=validated.llm_prompt,
            email_patterns=validated.email_patterns,
        )

    def parse_path(self, source_path: str) -> list[AutoTag]:
        """Apply path rules to extract tags from folder structure.

        Strips the filename from the path (last component if it contains a dot),
        then splits by '/' into folder components. Each PathRule is applied by
        its level index.
        """
        if not source_path or not source_path.strip():
            return []

        normalized = source_path.replace("\\", "/").strip("/")
        parts = normalized.split("/")

        if not parts:
            return []

        if "." in parts[-1]:
            parts = parts[:-1]

        if not parts:
            return []

        tags: list[AutoTag] = []

        for rule in self.path_rules:
            try:
                if rule.level >= len(parts):
                    continue

                folder = parts[rule.level]

                if rule.parent_match is not None:
                    if rule.level == 0:
                        continue
                    parent = parts[rule.level - 1]
                    if rule._compiled_parent and not rule._compiled_parent.match(parent):
                        continue

                if rule.mapping is not None:
                    mapped_value = rule.mapping.get(folder)
                    if mapped_value is not None:
                        tags.append(
                            AutoTag(
                                namespace=rule.namespace,
                                value=mapped_value,
                                source="path",
                                confidence=1.0,
                                strategy_id=self.id,
                                strategy_version=self.version,
                            )
                        )
                    continue

                transform_fn = get_transform(rule.transform)

                if rule.pattern is not None and rule._compiled_pattern is not None:
                    m = rule._compiled_pattern.match(folder)
                    if m:
                        try:
                            captured = m.group(rule.capture_group)
                        except IndexError:
                            captured = m.group(0)
                        value = transform_fn(captured)
                        tags.append(
                            AutoTag(
                                namespace=rule.namespace,
                                value=value,
                                source="path",
                                confidence=1.0,
                                strategy_id=self.id,
                                strategy_version=self.version,
                            )
                        )
                else:
                    value = transform_fn(folder)
                    tags.append(
                        AutoTag(
                            namespace=rule.namespace,
                            value=value,
                            source="path",
                            confidence=1.0,
                            strategy_id=self.id,
                            strategy_version=self.version,
                        )
                    )

            except Exception:
                logger.warning(
                    "Skipping path rule (namespace=%s, level=%d) due to error",
                    rule.namespace,
                    rule.level,
                    exc_info=True,
                )

        return tags

    def build_llm_prompt(self, filename: str, source_path: str, content_preview: str) -> str:
        """Render the LLM prompt template with strategy-specific values.

        Uses a two-pass approach:
        1. Replace dot-notation placeholders manually
        2. Use str.format_map with a defaultdict for simple placeholders
        """
        template = self.llm_prompt_template

        entity_label = self.entity_extraction.prompt_label if self.entity_extraction else "entity"
        entity_instruction = (
            self.entity_extraction.prompt_instruction if self.entity_extraction else ""
        )
        topic_instruction = (
            self.topic_extraction.prompt_instruction if self.topic_extraction else ""
        )

        template = template.replace("{entity_extraction.prompt_label}", entity_label)
        template = template.replace("{entity_extraction.prompt_instruction}", entity_instruction)
        template = template.replace("{topic_extraction.prompt_instruction}", topic_instruction)

        document_type_ids = ", ".join(self.document_types.keys())
        document_type_rules = "\n".join(
            f"- {dt_id}: {dt_config.description}"
            for dt_id, dt_config in self.document_types.items()
        )
        topic_ids = ""
        if self.topic_extraction:
            topic_ids = ", ".join(v.id for v in self.topic_extraction.values)

        values = defaultdict(
            str,
            {
                "filename": filename,
                "source_path": source_path,
                "content_preview": content_preview,
                "document_type_ids": document_type_ids,
                "document_type_rules": document_type_rules,
                "topic_ids": topic_ids,
            },
        )

        return template.format_map(values)

    def parse_llm_response(self, response: dict) -> list[AutoTag]:
        """Convert an LLM JSON response into a list of AutoTag objects.

        Handles document_type, entity, topics, confidence, and year extraction.
        """
        tags: list[AutoTag] = []
        confidence = float(response.get("confidence", 0.5))

        doc_type = response.get("document_type", "unknown")
        if doc_type not in self.document_types:
            logger.warning("Unknown document type '%s', mapping to 'unknown'", doc_type)
            doc_type = "unknown"
        tags.append(
            AutoTag(
                namespace="doctype",
                value=doc_type,
                source="llm",
                confidence=confidence,
                strategy_id=self.id,
                strategy_version=self.version,
            )
        )

        entity_raw = response.get("entity")
        if entity_raw and self.entity_extraction:
            entity_value = resolve_entity(entity_raw, self.entity_extraction.aliases)
            tags.append(
                AutoTag(
                    namespace=self.entity_extraction.namespace,
                    value=entity_value,
                    source="llm",
                    confidence=confidence,
                    strategy_id=self.id,
                    strategy_version=self.version,
                )
            )

        topics_raw = response.get("topics")
        if topics_raw and self.topic_extraction:
            known_ids = {v.id for v in self.topic_extraction.values}
            for topic_id in topics_raw:
                if topic_id in known_ids:
                    tags.append(
                        AutoTag(
                            namespace=self.topic_extraction.namespace,
                            value=topic_id,
                            source="llm",
                            confidence=confidence,
                            strategy_id=self.id,
                            strategy_version=self.version,
                        )
                    )

        year_start = response.get("year_start")
        year_end = response.get("year_end")
        if year_start and year_end:
            tags.append(
                AutoTag(
                    namespace="year",
                    value=f"{year_start}-{year_end}",
                    source="llm",
                    confidence=confidence,
                    strategy_id=self.id,
                    strategy_version=self.version,
                )
            )

        return tags

    def parse_email_subject(self, subject: str) -> list[AutoTag]:
        """Apply email subject patterns to extract tags.

        Returns deduplicated tags (by tag_name).
        """
        tags: list[AutoTag] = []
        seen: set[str] = set()

        for email_pattern in self.email_patterns:
            try:
                if re.search(email_pattern.pattern, subject):
                    for tag_def in email_pattern.tags:
                        tag = AutoTag(
                            namespace=tag_def.namespace,
                            value=tag_def.value,
                            source="email",
                            confidence=1.0,
                            strategy_id=self.id,
                            strategy_version=self.version,
                        )
                        if tag.tag_name not in seen:
                            seen.add(tag.tag_name)
                            tags.append(tag)
            except re.error:
                logger.warning(
                    "Invalid email pattern '%s', skipping",
                    email_pattern.pattern,
                    exc_info=True,
                )

        return tags
