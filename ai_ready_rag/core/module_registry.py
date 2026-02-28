"""ModuleRegistry — sole extension point between core platform and vertical modules.

Architecture rule: Core never imports from modules/. Modules import from core only
through the registry API. This file defines the 5 extension points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from fastapi import APIRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols (interfaces modules must implement)
# ---------------------------------------------------------------------------


@runtime_checkable
class DocumentClassifier(Protocol):
    """A classifier that can assign a document type to a file."""

    def classify(self, filename: str, content_sample: str) -> Any:
        """Return a ClassificationResult-like object with .doc_type and .confidence."""
        ...


@runtime_checkable
class ComplianceChecker(Protocol):
    """A checker that injects compliance constraints into prompts or validates results."""

    def check(self, context: dict[str, Any]) -> Any:
        """Return a ComplianceReport-like object."""
        ...


@dataclass
class SQLTemplate:
    """A registered SQL template with its parameter contract."""

    name: str
    sql: str
    trigger_phrases: list[str] = field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ModuleRegistry:
    """Singleton registry for vertical module extension points.

    Modules call register() at startup to contribute:
    - Document classifiers
    - Entity alias maps
    - SQL query templates
    - Compliance checkers
    - FastAPI sub-routers
    """

    _instance: ModuleRegistry | None = None

    def __init__(self) -> None:
        self._classifiers: list[DocumentClassifier] = []
        self._entity_maps: dict[str, str] = {}  # canonical_name → table/column hint
        self._sql_templates: dict[str, SQLTemplate] = {}  # template_name → SQLTemplate
        self._compliance_checkers: list[ComplianceChecker] = []
        self._routers: list[tuple[APIRouter, str]] = []  # (router, prefix)
        self._registered_modules: list[str] = []

    @classmethod
    def get_instance(cls) -> ModuleRegistry:
        """Return the process-wide singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton — for testing only."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Extension point 1: Document classifiers
    # ------------------------------------------------------------------

    def register_document_classifiers(self, classifiers: list[DocumentClassifier]) -> None:
        """Register one or more document classifiers contributed by a module."""
        self._classifiers.extend(classifiers)
        logger.debug("registry.classifiers.added", extra={"count": len(classifiers)})

    @property
    def document_classifiers(self) -> list[DocumentClassifier]:
        return list(self._classifiers)

    # ------------------------------------------------------------------
    # Extension point 2: Entity alias map
    # ------------------------------------------------------------------

    def register_entity_map(self, entity_map: dict[str, str]) -> None:
        """Merge entity_map into the registry. Later registrations win on key collision."""
        self._entity_maps.update(entity_map)
        logger.debug("registry.entity_map.merged", extra={"keys": len(entity_map)})

    @property
    def entity_map(self) -> dict[str, str]:
        return dict(self._entity_maps)

    # ------------------------------------------------------------------
    # Extension point 3: SQL templates
    # ------------------------------------------------------------------

    def register_sql_templates(self, templates: dict[str, SQLTemplate]) -> None:
        """Register SQL templates. Later registrations win on name collision."""
        self._sql_templates.update(templates)
        logger.debug("registry.sql_templates.merged", extra={"keys": len(templates)})

    @property
    def sql_templates(self) -> dict[str, SQLTemplate]:
        return dict(self._sql_templates)

    # ------------------------------------------------------------------
    # Extension point 4: Compliance checkers
    # ------------------------------------------------------------------

    def register_compliance_checker(self, checker: ComplianceChecker) -> None:
        """Register a compliance checker."""
        self._compliance_checkers.append(checker)
        logger.debug("registry.compliance_checker.added")

    @property
    def compliance_checkers(self) -> list[ComplianceChecker]:
        return list(self._compliance_checkers)

    # ------------------------------------------------------------------
    # Extension point 5: API routers
    # ------------------------------------------------------------------

    def register_api_router(self, router: APIRouter, prefix: str) -> None:
        """Register a FastAPI router with its URL prefix."""
        self._routers.append((router, prefix))
        logger.debug("registry.router.added", extra={"prefix": prefix})

    @property
    def api_routers(self) -> list[tuple[APIRouter, str]]:
        return list(self._routers)

    # ------------------------------------------------------------------
    # Module lifecycle
    # ------------------------------------------------------------------

    def register_module(self, module_id: str) -> None:
        """Record that a module has been registered (for health/status endpoints)."""
        if module_id not in self._registered_modules:
            self._registered_modules.append(module_id)
            logger.info("registry.module.registered", extra={"module_id": module_id})

    @property
    def registered_modules(self) -> list[str]:
        return list(self._registered_modules)


def get_registry() -> ModuleRegistry:
    """FastAPI dependency: returns the process-wide ModuleRegistry."""
    return ModuleRegistry.get_instance()
