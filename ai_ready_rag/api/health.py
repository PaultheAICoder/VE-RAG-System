"""Health check endpoints."""

from fastapi import APIRouter

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.redis import is_redis_available

router = APIRouter()
settings = get_settings()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    redis_ok = await is_redis_available()
    return {
        "status": "healthy",
        "version": settings.app_version,
        "database": "sqlite",
        "redis": "connected" if redis_ok else "unavailable",
        "rag_enabled": settings.enable_rag,
        "profile": settings.env_profile,
        "backends": {
            "vector": settings.vector_backend,
            "chunker": settings.chunker_backend,
            "ocr_enabled": settings.enable_ocr,
        },
    }


@router.get("/version")
async def version():
    """Get version info."""
    return {"name": settings.app_name, "version": settings.app_version}
