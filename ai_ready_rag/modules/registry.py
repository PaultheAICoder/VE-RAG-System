"""ModuleRegistry — vertical module extension point system.

The sole coupling point between the core platform and vertical modules.
Core services call the getters; modules call the registration methods.

Architecture rule (enforced here):
    Core NEVER imports from modules/<vertical>/ directly.
    Modules call registry.register_*() methods at startup.
    Core calls registry.get_*() at runtime.
"""

from __future__ import annotations

import importlib
import json
import logging
import pathlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ─── SQL template safety guards ─────────────────────────────────────────────

_DML_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)
_LIMIT_PATTERN = re.compile(r"\bLIMIT\b", re.IGNORECASE)
_INTERPOLATION_PATTERN = re.compile(r"\{[^}]+\}|%s|%\(")


def _validate_sql_template(name: str, sql: str) -> None:
    """Raise ValueError if SQL template is unsafe."""
    if _DML_PATTERN.search(sql):
        raise ValueError(f"SQL template '{name}' contains forbidden DML statement")
    if not _LIMIT_PATTERN.search(sql):
        raise ValueError(f"SQL template '{name}' missing required LIMIT clause")
    if _INTERPOLATION_PATTERN.search(sql):
        raise ValueError(
            f"SQL template '{name}' uses string interpolation — use :param_name bindings"
        )


# ─── Protocol stubs ─────────────────────────────────────────────────────────


class ComplianceChecker:
    """Base class for module compliance checkers. Override check() in module implementation."""

    def check(self, account_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Run compliance check. Returns ComplianceReport dict."""
        raise NotImplementedError


# ─── Module manifest ─────────────────────────────────────────────────────────


@dataclass
class ModuleManifest:
    """Parsed manifest.json for a vertical module."""

    module_id: str
    version: str
    display_name: str
    requires_core_version: str = ">=0.5.0"
    feature_flags: dict[str, bool] = field(default_factory=dict)
    extension_points: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


# ─── ModuleRegistry ──────────────────────────────────────────────────────────


class ModuleRegistry:
    """Singleton registry for vertical module extension points.

    Extension points (5 total):
        1. document_classifiers — per-type classification rules
        2. entity_map          — entity name → (table, column) for SQL dispatch
        3. sql_templates        — parameterized read-only SQL templates
        4. compliance_checker   — per-module Fannie Mae / FHA / state-law checks
        5. api_router           — FastAPI router mounted at startup
    """

    def __init__(self) -> None:
        self._classifiers: dict[str, Any] = {}  # module_name → classifier config/path
        self._entity_maps: dict[str, dict[str, str]] = {}  # module_name → {entity: table.col}
        self._sql_templates: dict[str, str] = {}  # template_name → sql
        self._compliance_checkers: dict[str, ComplianceChecker] = {}
        self._api_routers: list[tuple[str, APIRouter, str]] = []  # (module_name, router, prefix)
        self._active_modules: list[str] = ["core"]
        self._manifests: dict[str, ModuleManifest] = {}

    # ── Registration methods (called by modules at startup) ────────────────

    def register_document_classifiers(self, module_name: str, classifiers: Any) -> None:
        """Register document type classifiers for a module.

        Args:
            module_name: Unique module identifier.
            classifiers: Path to classifiers.yaml OR list of classifier dicts.
        """
        self._classifiers[module_name] = classifiers
        logger.info("registry.classifiers.registered", extra={"module_name": module_name})

    def register_entity_map(self, module_name: str, entity_map: dict[str, str]) -> None:
        """Register entity name → SQL table.column mapping.

        Args:
            module_name: Unique module identifier.
            entity_map: Dict mapping extracted entity names to fully-qualified table.column strings.
                Example: {"unit_count": "insurance_accounts.units_residential"}
        """
        self._entity_maps[module_name] = entity_map
        logger.info(
            "registry.entity_map.registered",
            extra={"module_name": module_name, "count": len(entity_map)},
        )

    def register_sql_templates(self, module_name: str, templates: dict[str, str] | str) -> None:
        """Register parameterized SQL templates.

        Args:
            module_name: Unique module identifier.
            templates: Dict of {template_name: sql_string} OR path to sql_templates.yaml.
                       All templates are validated for safety (no DML, must have LIMIT).

        Raises:
            ValueError: If any template fails safety validation.
        """
        if isinstance(templates, str):
            # Load from yaml path
            import yaml  # noqa: PLC0415

            with open(templates) as f:
                data = yaml.safe_load(f)
            templates = {t["name"]: t["sql"].strip() for t in data.get("templates", [])}

        for name, sql in templates.items():
            _validate_sql_template(name, sql)
            self._sql_templates[name] = sql

        logger.info(
            "registry.sql_templates.registered",
            extra={"module_name": module_name, "count": len(templates)},
        )

    def register_compliance_checker(
        self, module_name: str, checker: type[ComplianceChecker] | ComplianceChecker
    ) -> None:
        """Register a compliance checker instance or class.

        Args:
            module_name: Unique module identifier.
            checker: ComplianceChecker instance or class (will be instantiated).
        """
        instance = checker() if isinstance(checker, type) else checker
        self._compliance_checkers[module_name] = instance
        logger.info("registry.compliance_checker.registered", extra={"module_name": module_name})

    def register_api_router(self, module_name: str, router: APIRouter, prefix: str = "") -> None:
        """Register a FastAPI API router for a module.

        Args:
            module_name: Unique module identifier.
            router: FastAPI APIRouter with module-specific endpoints.
            prefix: URL prefix to mount the router at (e.g., "/api/ca").
        """
        self._api_routers.append((module_name, router, prefix))
        logger.info(
            "registry.api_router.registered",
            extra={"module_name": module_name, "prefix": prefix},
        )

    # ── Getter methods (called by core services at runtime) ─────────────────

    def get_classifiers(self, module_name: str | None = None) -> dict[str, Any]:
        """Return all registered classifiers, optionally filtered by module."""
        if module_name:
            return {module_name: self._classifiers.get(module_name)}
        return dict(self._classifiers)

    def get_entity_map(self) -> dict[str, str]:
        """Return merged entity map across all registered modules."""
        merged: dict[str, str] = {}
        for entity_map in self._entity_maps.values():
            merged.update(entity_map)
        return merged

    def get_sql_templates(self) -> dict[str, str]:
        """Return all registered SQL templates across all modules."""
        return dict(self._sql_templates)

    def get_sql_template(self, name: str) -> str | None:
        """Return a single SQL template by name."""
        return self._sql_templates.get(name)

    def get_compliance_checker(self, module_name: str) -> ComplianceChecker | None:
        """Return compliance checker for a module, or None if not registered."""
        return self._compliance_checkers.get(module_name)

    def get_api_routers(self) -> list[tuple[str, APIRouter, str]]:
        """Return list of (module_name, router, prefix) tuples for mounting."""
        return list(self._api_routers)

    @property
    def active_modules(self) -> list[str]:
        """Return list of currently active module names."""
        return list(self._active_modules)

    # ── Module loading ───────────────────────────────────────────────────────

    def load_module(self, module_name: str) -> None:
        """Discover, validate, and register a vertical module.

        Process:
            1. Load manifest.json from modules/<module_name>/manifest.json
            2. Validate manifest (required fields, version compatibility)
            3. Call module.register(registry=self)
            4. Add module_name to active_modules

        Module loading failures are caught and logged; they do NOT block
        core startup or other module loading.

        Args:
            module_name: Name matching the module directory under ai_ready_rag/modules/
        """
        try:
            self._load_module_unsafe(module_name)
        except Exception as exc:
            logger.error(
                "registry.module.load_failed",
                extra={"module_name": module_name, "error": str(exc)},
                exc_info=True,
            )

    def _load_module_unsafe(self, module_name: str) -> None:
        """Internal: load module without error suppression."""
        # 1. Find module directory
        modules_dir = pathlib.Path(__file__).parent
        module_dir = modules_dir / module_name
        if not module_dir.is_dir():
            raise FileNotFoundError(f"Module directory not found: {module_dir}")

        # 2. Load and validate manifest
        manifest_path = module_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                raw = json.load(f)
            manifest = ModuleManifest(
                module_id=raw.get("module_id", module_name),
                version=raw.get("version", "0.0"),
                display_name=raw.get("display_name", module_name),
                requires_core_version=raw.get("dependencies", {}).get(
                    "core_min_version", ">=0.5.0"
                ),
                feature_flags=raw.get("feature_flags", {}),
                extension_points=raw.get("extension_points", []),
                raw=raw,
            )
            self._manifests[module_name] = manifest
            logger.info(
                "registry.module.manifest_loaded",
                extra={"module_name": module_name, "version": manifest.version},
            )

        # 3. Import and call register()
        module_import_path = f"ai_ready_rag.modules.{module_name}.module"
        mod = importlib.import_module(module_import_path)
        if hasattr(mod, "register"):
            mod.register(registry=self)
        else:
            logger.warning(
                "registry.module.no_register",
                extra={"module_name": module_name, "import": module_import_path},
            )

        # 4. Mark as active
        if module_name not in self._active_modules:
            self._active_modules.append(module_name)
        logger.info("registry.module.loaded", extra={"module_name": module_name})

    def load_all_modules(self, module_names: list[str]) -> None:
        """Load all modules in order. Each failure is isolated.

        Args:
            module_names: Ordered list of module names to load.
        """
        for name in module_names:
            if name == "core":
                continue  # Core is always active, not a loadable module
            self.load_module(name)


# ─── Singleton accessor ──────────────────────────────────────────────────────

_registry: ModuleRegistry | None = None


def get_registry() -> ModuleRegistry:
    """Return the global ModuleRegistry singleton.

    Initialized by main.py lifespan. Raises RuntimeError if called before init.
    """
    if _registry is None:
        raise RuntimeError(
            "ModuleRegistry not initialized. Call init_registry() during application startup."
        )
    return _registry


def init_registry(module_names: list[str] | None = None) -> ModuleRegistry:
    """Initialize and return the global ModuleRegistry.

    Called once during FastAPI lifespan startup.

    Args:
        module_names: List of module names to load. Defaults to ["core"].
    """
    global _registry
    _registry = ModuleRegistry()
    if module_names:
        _registry.load_all_modules(module_names)
    return _registry
