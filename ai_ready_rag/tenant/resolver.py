"""TenantConfigResolver and PromptResolver — 3-tier configuration resolution.

Resolution order (lowest → highest priority):
  1. Core defaults  — built-in Pydantic defaults in TenantConfig
  2. Module defaults — from modules/{id}/manifest.json (future use)
  3. Tenant overrides — from tenant-instances/{tenant_id}/tenant.json
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from ai_ready_rag.tenant.config import TenantConfig

logger = logging.getLogger(__name__)

# Prompt search order: tenant → module → core
_PROMPT_SEARCH_DIRS = [
    "tenant-instances/{tenant_id}/prompts",
    "ai_ready_rag/modules/{module_id}/prompts",
    "ai_ready_rag/prompts",
]


class TenantConfigResolver:
    """Resolves 3-tier configuration: core defaults → module → tenant.json overrides.

    Usage:
        resolver = TenantConfigResolver()
        config = resolver.resolve("acme-corp")
        resolver.invalidate("acme-corp")  # force re-read on next call
    """

    def __init__(
        self,
        tenant_config_path_template: str = "tenant-instances/{tenant_id}/tenant.json",
    ) -> None:
        self._path_template = tenant_config_path_template
        self._cache: dict[str, TenantConfig] = {}

    def resolve(self, tenant_id: str) -> TenantConfig:
        """Return the merged TenantConfig for the given tenant_id.

        Result is cached per tenant_id; call invalidate() to force a re-read.
        """
        if tenant_id in self._cache:
            return self._cache[tenant_id]

        config = self._load(tenant_id)
        self._cache[tenant_id] = config
        return config

    def invalidate(self, tenant_id: str) -> None:
        """Remove cached config — forces re-read on next resolve()."""
        self._cache.pop(tenant_id, None)

    def _load(self, tenant_id: str) -> TenantConfig:
        """Load and parse tenant.json, falling back to Pydantic defaults on any error."""
        path = Path(self._path_template.format(tenant_id=tenant_id))
        if not path.exists():
            logger.warning(
                "tenant.config.not_found",
                extra={"path": str(path), "tenant_id": tenant_id},
            )
            return TenantConfig(tenant_id=tenant_id)

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["tenant_id"] = tenant_id
            config = TenantConfig.model_validate(raw)
            logger.info("tenant.config.loaded", extra={"tenant_id": tenant_id})
            return config
        except Exception as exc:
            logger.error(
                "tenant.config.parse_error",
                extra={"tenant_id": tenant_id, "error": str(exc)},
            )
            return TenantConfig(tenant_id=tenant_id)


class PromptResolver:
    """Resolves prompt templates from 3-tier search path: tenant → module → core.

    Search order (first found wins):
      1. tenant-instances/{tenant_id}/prompts/{prompt_name}.txt
      2. ai_ready_rag/modules/{module_id}/prompts/{prompt_name}.txt
      3. ai_ready_rag/prompts/{prompt_name}.txt
    """

    def __init__(self, tenant_id: str = "default", module_id: str = "core") -> None:
        self._tenant_id = tenant_id
        self._module_id = module_id

    def resolve(self, prompt_name: str) -> str | None:
        """Return the first matching prompt file content or None.

        Returns None (does not raise) when the prompt is not found in any tier.
        """
        for dir_template in _PROMPT_SEARCH_DIRS:
            dir_path = Path(
                dir_template.format(
                    tenant_id=self._tenant_id,
                    module_id=self._module_id,
                )
            )
            candidate = dir_path / f"{prompt_name}.txt"
            if candidate.exists():
                logger.debug(
                    "prompt.resolved",
                    extra={"name": prompt_name, "path": str(candidate)},
                )
                return candidate.read_text(encoding="utf-8")

        logger.debug("prompt.not_found", extra={"name": prompt_name})
        return None

    def list_available(self) -> list[str]:
        """Return all prompt names available across all tiers (deduplicated, sorted)."""
        names: set[str] = set()
        for dir_template in _PROMPT_SEARCH_DIRS:
            dir_path = Path(
                dir_template.format(
                    tenant_id=self._tenant_id,
                    module_id=self._module_id,
                )
            )
            if dir_path.exists():
                names.update(p.stem for p in dir_path.glob("*.txt"))
        return sorted(names)


@lru_cache(maxsize=32)
def get_tenant_config_resolver(
    tenant_config_path_template: str = "tenant-instances/{tenant_id}/tenant.json",
) -> TenantConfigResolver:
    """Cached TenantConfigResolver factory (singleton per path template)."""
    return TenantConfigResolver(tenant_config_path_template)
