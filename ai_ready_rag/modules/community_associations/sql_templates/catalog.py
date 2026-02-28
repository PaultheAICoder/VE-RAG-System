"""Community Associations SQL template catalog.

Builds TEMPLATE_CATALOG from the sql_templates.yaml file so the CA module
register() function can load all templates into the ModuleRegistry at startup.

Usage (in module.py register()):
    from ai_ready_rag.modules.community_associations.sql_templates.catalog import TEMPLATE_CATALOG
    registry.register_sql_templates("community_associations", TEMPLATE_CATALOG)
"""

from __future__ import annotations

import logging
import pathlib

import yaml

from ai_ready_rag.modules.registry import SQLTemplate

logger = logging.getLogger(__name__)

_YAML_PATH = pathlib.Path(__file__).parent.parent / "sql_templates.yaml"


def _load_catalog() -> dict[str, SQLTemplate]:
    """Parse sql_templates.yaml and return a dict[name, SQLTemplate]."""
    if not _YAML_PATH.exists():
        logger.warning("CA sql_templates.yaml not found at %s — catalog will be empty", _YAML_PATH)
        return {}

    with open(_YAML_PATH) as f:
        data = yaml.safe_load(f)

    catalog: dict[str, SQLTemplate] = {}
    for entry in data.get("templates", []):
        name = entry["name"]
        sql = entry.get("sql", "").strip()
        trigger_phrases = entry.get("trigger_phrases", [])
        description = entry.get("display_name", "")
        catalog[name] = SQLTemplate(
            name=name,
            sql=sql,
            trigger_phrases=trigger_phrases,
            description=description,
        )

    logger.debug("CA sql_templates catalog loaded: %d templates", len(catalog))
    return catalog


# Module-level singleton — loaded once at import time.
TEMPLATE_CATALOG: dict[str, SQLTemplate] = _load_catalog()
