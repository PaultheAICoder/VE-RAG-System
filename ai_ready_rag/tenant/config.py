"""Pydantic models for tenant.json configuration schema.

Tenant config is stored in tenant-instances/{tenant_id}/tenant.json
and loaded at startup by TenantConfigResolver.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

# Claude model governance constants.
# Pinned model IDs — aliases like "claude-sonnet" are rejected to prevent silent version drift.
VALID_CLAUDE_MODELS: frozenset[str] = frozenset(
    {
        # Claude 4 series
        "claude-opus-4-6",
        "claude-opus-4-5",
        # Claude Sonnet 4.x
        "claude-sonnet-4-6",
        "claude-sonnet-4-5-20251001",
        # Claude Haiku 4.x
        "claude-haiku-4-5-20251001",
        # Claude 3.7 series
        "claude-3-7-sonnet-20250219",
        # Claude 3.5 series
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        # Claude 3 series
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    }
)

# Default model constants (mirrors what will be in ai_ready_rag.config after PR #390)
CLAUDE_ENRICHMENT_MODEL = "claude-sonnet-4-6"
CLAUDE_QUERY_MODEL_SIMPLE = "claude-haiku-4-5-20251001"
CLAUDE_QUERY_MODEL_COMPLEX = "claude-sonnet-4-6"

# Alias IDs explicitly rejected (version-unspecified, may silently upgrade)
REJECTED_CLAUDE_ALIASES: frozenset[str] = frozenset(
    {
        "claude-opus",
        "claude-sonnet",
        "claude-haiku",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "claude-3-5-sonnet",
        "claude-3-5-haiku",
        "claude-3-7-sonnet",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    }
)


class AIModelConfig(BaseModel):
    """AI model preferences for this tenant."""

    enrichment_model: str = CLAUDE_ENRICHMENT_MODEL
    query_model_simple: str = CLAUDE_QUERY_MODEL_SIMPLE
    query_model_complex: str = CLAUDE_QUERY_MODEL_COMPLEX
    primary: str = "claude"
    fallback: str = "ollama"
    monthly_query_cap_usd: float = 150.0
    daily_enrichment_cap_usd: float = 10.0

    @field_validator("enrichment_model", "query_model_simple", "query_model_complex", mode="before")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        """Reject alias model IDs — only pinned IDs allowed."""
        if v in REJECTED_CLAUDE_ALIASES:
            raise ValueError(
                f"Model alias '{v}' is not permitted. Use a pinned ID from VALID_CLAUDE_MODELS."
            )
        if VALID_CLAUDE_MODELS and v.startswith("claude-") and v not in VALID_CLAUDE_MODELS:
            raise ValueError(f"Unknown Claude model ID '{v}'. Valid: {sorted(VALID_CLAUDE_MODELS)}")
        return v


class FeatureFlags(BaseModel):
    """Feature flag configuration for this tenant."""

    ca_enabled: bool = False
    ca_auto_renewal: bool = False
    ca_unit_owner_letters: bool = False
    ca_board_presentations: bool = False
    structured_query_enabled: bool = False
    claude_enrichment_enabled: bool = False


class BrandingConfig(BaseModel):
    """UI branding configuration."""

    primary_color: str = "#1a73e8"
    logo_url: str | None = None
    product_name: str = "VaultIQ"


class TenantConfig(BaseModel):
    """Complete tenant configuration schema (tenant.json)."""

    tenant_id: str = "default"
    display_name: str = ""
    active_modules: list[str] = ["core"]
    feature_flags: FeatureFlags = FeatureFlags()
    ai_models: AIModelConfig = AIModelConfig()
    branding: BrandingConfig = BrandingConfig()

    class Config:
        extra = "ignore"


# Default tenant config used when no tenant.json is found
DEFAULT_TENANT_CONFIG = TenantConfig(
    tenant_id="default",
    display_name="VaultIQ",
    active_modules=["core"],
    feature_flags=FeatureFlags(
        ca_enabled=False,
        claude_enrichment_enabled=True,
        structured_query_enabled=False,
    ),
)
