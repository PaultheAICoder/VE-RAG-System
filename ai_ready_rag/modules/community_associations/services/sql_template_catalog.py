"""SQL template catalog loader and validator.

Loads sql_templates.yaml and validates all templates at startup.
Templates are registered via ModuleRegistry.register_sql_templates().
"""

from __future__ import annotations

import logging
import pathlib
import re
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)

# SQL statements that are forbidden in templates (read-only enforcement)
_DML_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

# Required safety clause in all templates
_LIMIT_PATTERN = re.compile(r"\bLIMIT\b", re.IGNORECASE)

# Forbidden interpolation patterns (f-string or % formatting)
_INTERPOLATION_PATTERN = re.compile(r"\{[^}]+\}|%s|%\(")


@dataclass
class SQLTemplateParam:
    """Parameter definition for a SQL template."""

    name: str
    type: str
    required: bool = True
    default: object = None


@dataclass
class SQLTemplate:
    """A validated SQL template with metadata."""

    name: str
    display_name: str
    trigger_phrases: list[str]
    params: list[SQLTemplateParam]
    sql: str

    def validate(self) -> list[str]:
        """Validate template safety. Returns list of violations (empty = valid)."""
        violations = []

        if _DML_PATTERN.search(self.sql):
            violations.append(f"Template '{self.name}' contains forbidden DML statement")

        if not _LIMIT_PATTERN.search(self.sql):
            violations.append(f"Template '{self.name}' missing required LIMIT clause")

        if _INTERPOLATION_PATTERN.search(self.sql):
            violations.append(
                f"Template '{self.name}' contains forbidden string interpolation (use :param_name)"
            )

        return violations


@dataclass
class SQLTemplateCatalog:
    """Collection of validated SQL templates for a module."""

    module_id: str
    default_row_cap: int
    default_timeout_seconds: int
    templates: list[SQLTemplate] = field(default_factory=list)

    def get(self, name: str) -> SQLTemplate | None:
        """Return template by name, or None if not found."""
        return next((t for t in self.templates if t.name == name), None)

    def as_dict(self) -> dict[str, str]:
        """Return {template_name: sql_string} dict for ModuleRegistry registration."""
        return {t.name: t.sql for t in self.templates}

    def trigger_map(self) -> dict[str, str]:
        """Return {trigger_phrase: template_name} for QueryRouter routing."""
        result = {}
        for template in self.templates:
            for phrase in template.trigger_phrases:
                result[phrase.lower()] = template.name
        return result


def load_sql_template_catalog(
    yaml_path: str | pathlib.Path | None = None,
) -> SQLTemplateCatalog:
    """Load and validate the CA SQL template catalog from YAML.

    Args:
        yaml_path: Path to sql_templates.yaml. Defaults to module's sql_templates.yaml.

    Returns:
        Validated SQLTemplateCatalog.

    Raises:
        ValueError: If any template fails validation.
    """
    if yaml_path is None:
        yaml_path = pathlib.Path(__file__).parent.parent / "sql_templates.yaml"

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    catalog = SQLTemplateCatalog(
        module_id=data["module_id"],
        default_row_cap=data.get("default_row_cap", 1000),
        default_timeout_seconds=data.get("default_timeout_seconds", 5),
    )

    all_violations = []
    for tpl_data in data.get("templates", []):
        params = [
            SQLTemplateParam(
                name=p["name"],
                type=p["type"],
                required=p.get("required", True),
                default=p.get("default"),
            )
            for p in tpl_data.get("params", [])
        ]
        template = SQLTemplate(
            name=tpl_data["name"],
            display_name=tpl_data["display_name"],
            trigger_phrases=tpl_data.get("trigger_phrases", []),
            params=params,
            sql=tpl_data["sql"].strip(),
        )
        violations = template.validate()
        all_violations.extend(violations)
        catalog.templates.append(template)

    if all_violations:
        msg = "SQL template validation failures:\n" + "\n".join(f"  - {v}" for v in all_violations)
        raise ValueError(msg)

    logger.info(
        "ca.sql_templates.loaded",
        extra={"count": len(catalog.templates), "module": data["module_id"]},
    )
    return catalog
