"""ARQ worker settings.

Start the worker with:
    arq ai_ready_rag.workers.settings.WorkerSettings
"""

import logging

from arq.connections import RedisSettings

from ai_ready_rag.config import get_settings

logger = logging.getLogger(__name__)


async def on_startup(ctx: dict) -> None:
    """ARQ worker startup: initialize shared resources."""
    from ai_ready_rag.db.database import init_db
    from ai_ready_rag.services.factory import get_vector_service

    settings = get_settings()
    init_db()

    vector_service = get_vector_service(settings)
    await vector_service.initialize()

    ctx["settings"] = settings
    ctx["vector_service"] = vector_service
    logger.info("ARQ worker started — vector service initialized")


async def on_shutdown(ctx: dict) -> None:
    """ARQ worker shutdown: clean up resources."""
    logger.info("ARQ worker shutting down")


def get_worker_settings() -> dict:
    """Build WorkerSettings configuration dict."""
    from ai_ready_rag.workers.tasks import process_document, reindex_knowledge_base, warm_cache

    settings = get_settings()

    return {
        "functions": [process_document, reindex_knowledge_base, warm_cache],
        "redis_settings": RedisSettings.from_dsn(settings.redis_url),
        "max_jobs": settings.arq_max_jobs,
        "job_timeout": settings.arq_job_timeout,
        "health_check_interval": settings.arq_health_check_interval,
        "on_startup": on_startup,
        "on_shutdown": on_shutdown,
    }


class WorkerSettings:
    """ARQ WorkerSettings for `arq ai_ready_rag.workers.settings.WorkerSettings`."""

    from ai_ready_rag.workers.tasks import process_document, reindex_knowledge_base, warm_cache

    settings = get_settings()

    functions = [process_document, reindex_knowledge_base, warm_cache]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = settings.arq_max_jobs
    job_timeout = settings.arq_job_timeout
    health_check_interval = settings.arq_health_check_interval
    on_startup = on_startup
    on_shutdown = on_shutdown
