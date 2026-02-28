"""TenantConfig — Pydantic models for per-tenant configuration.

Resolution order (lowest → highest priority):
  1. Core defaults  — built-in Pydantic defaults
  2. Module defaults — from modules/{id}/manifest.json (future use)
  3. Tenant overrides — from tenant-instances/{tenant_id}/tenant.json
"""

from __future__ import annotations

from pydantic import BaseModel

# ─── Feature flags ────────────────────────────────────────────────────────────


class FeatureFlags(BaseModel):
    """Feature flags controlling per-tenant module enablement."""

    ca_enabled: bool = True
    ca_auto_renewal: bool = True
    ca_compliance_check: bool = True
    ca_unit_owner_letters: bool = True
    ca_board_presentations: bool = True
    structured_query_enabled: bool = True
    claude_enrichment_enabled: bool = False


# ─── Supporting config models ─────────────────────────────────────────────────


class BrandingConfig(BaseModel):
    """Tenant branding configuration."""

    product_name: str = "AI Ready RAG"
    logo_url: str | None = None
    primary_color: str = "#6366F1"


# ─── Root tenant config ───────────────────────────────────────────────────────


class TenantConfig(BaseModel):
    """Root tenant configuration resolved from 3-tier lookup."""

    feature_flags: FeatureFlags = FeatureFlags()
    branding: BrandingConfig = BrandingConfig()
    active_modules: frozenset[str] = frozenset(["core", "community_associations"])
    monthly_query_cap_usd: float | None = None
    daily_enrichment_cap_usd: float | None = None

    class Config:
        extra = "ignore"


# Default config used when tenant.json is absent or unreadable
DEFAULT_TENANT_CONFIG = TenantConfig()
