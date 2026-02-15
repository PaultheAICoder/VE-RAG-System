"""Health check endpoints."""

from fastapi import APIRouter, Request

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.redis import is_redis_available

router = APIRouter()
settings = get_settings()


@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    redis_ok = await is_redis_available()

    eval_status = {"enabled": settings.eval_enabled}
    eval_worker = getattr(request.app.state, "eval_worker", None)
    if eval_worker:
        eval_status["worker_running"] = eval_worker._task is not None
        eval_status["current_run_id"] = eval_worker._current_run_id
    else:
        eval_status["worker_running"] = False
        eval_status["current_run_id"] = None

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
        "evaluation": eval_status,
    }


@router.get("/version")
async def version():
    """Get version info."""
    return {"name": settings.app_name, "version": settings.app_version}
