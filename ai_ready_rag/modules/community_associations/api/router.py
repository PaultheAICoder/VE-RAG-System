"""Community Associations REST API router.

Full implementation in issue #373. This stub allows the module to load
without errors before the router is implemented.
"""

from fastapi import APIRouter

ca_router = APIRouter(prefix="/ca", tags=["community-associations"])


@ca_router.get("/health")
async def ca_health() -> dict:
    """CA module health check."""
    return {"module": "community_associations", "status": "stub"}
