"""Health check endpoints."""
from fastapi import APIRouter
from ai_ready_rag.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "database": "sqlite",
        "rag_enabled": settings.enable_rag,
        "gradio_enabled": settings.enable_gradio
    }


@router.get("/version")
async def version():
    """Get version info."""
    return {
        "name": settings.app_name,
        "version": settings.app_version
    }
