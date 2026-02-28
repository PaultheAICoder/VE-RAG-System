"""Tenant configuration API endpoint.

GET /api/tenant/config — returns resolved tenant config for the React frontend.
Used at app startup to load branding and feature flags.
"""

from fastapi import APIRouter, Depends

from ai_ready_rag.core.dependencies import get_current_user
from ai_ready_rag.db.models import User
from ai_ready_rag.tenant.config import TenantConfig
from ai_ready_rag.tenant.resolver import TenantConfigResolver

router = APIRouter(prefix="/api/tenant", tags=["tenant"])

_resolver = TenantConfigResolver()


@router.get("/config", response_model=TenantConfig)
async def get_tenant_config(
    current_user: User = Depends(get_current_user),
) -> TenantConfig:
    """Return the resolved tenant configuration for the current user's tenant.

    Loads core defaults → module config → tenant.json overrides.
    Used by React frontend at startup for branding and feature flags.

    Auth: requires valid JWT (any role).
    """
    tenant_id = getattr(current_user, "tenant_id", "default")
    return _resolver.resolve(tenant_id)
