"""ModuleRegistry — re-export shim (DEPRECATED).

Previously this module held a separate ModuleRegistry class with a class-level
singleton (_instance), while ai_ready_rag.modules.registry held a different
singleton (_registry, module-level global). The two never shared state, so
QueryRouter (which imported from here) always saw an empty template set even
after vertical modules registered templates via modules.registry at startup.

All implementation is now consolidated in ai_ready_rag.modules.registry.
This file is kept as a thin re-export shim to avoid breaking any existing
imports. New code should import directly from ai_ready_rag.modules.registry.

Fixed by: issue #422
"""

# ---------------------------------------------------------------------------
# Re-export everything from the canonical registry module
# ---------------------------------------------------------------------------

from ai_ready_rag.modules.registry import (  # noqa: F401
    ComplianceChecker,
    ModuleManifest,
    ModuleRegistry,
    SQLTemplate,
    _validate_sql_template,
    get_registry,
    init_registry,
)

__all__ = [
    "ComplianceChecker",
    "ModuleManifest",
    "ModuleRegistry",
    "SQLTemplate",
    "_validate_sql_template",
    "get_registry",
    "init_registry",
]
