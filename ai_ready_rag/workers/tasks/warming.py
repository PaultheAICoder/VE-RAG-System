"""Cache warming ARQ task.

Runs specified queries through the RAG pipeline to pre-populate cache.
Falls back to BackgroundTasks if Redis is unavailable.
"""

import asyncio
import logging

from ai_ready_rag.config import get_settings

logger = logging.getLogger(__name__)


async def warm_cache(
    ctx: dict,
    queries: list[str],
    triggered_by: str,
) -> dict:
    """ARQ task: warm cache by running queries through RAG pipeline.

    Args:
        ctx: ARQ context dict (contains settings, vector_service from on_startup)
        queries: List of query strings to warm
        triggered_by: User ID who triggered the warming

    Returns:
        Dict with warming result (success, warmed, total)
    """
    from ai_ready_rag.db.database import SessionLocal
    from ai_ready_rag.services.factory import get_vector_service
    from ai_ready_rag.services.rag_service import RAGRequest, RAGService

    logger.info(
        f"[ARQ] Starting cache warming: {len(queries)} queries (triggered by {triggered_by})"
    )

    settings = ctx.get("settings") or get_settings()
    db = SessionLocal()

    try:
        vector_service = ctx.get("vector_service") or get_vector_service(settings)
        if not ctx.get("vector_service"):
            await vector_service.initialize()

        rag_service = RAGService(settings, vector_service=vector_service)
        warmed = 0

        for i, query in enumerate(queries):
            try:
                request = RAGRequest(
                    query=query,
                    user_tags=[],  # Admin context - responses cached without tag restriction
                    tenant_id="default",
                )
                await rag_service.generate(request, db)
                warmed += 1
                logger.debug(f"Warmed cache for query: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to warm cache for query '{query[:50]}...': {e}")

            # Throttle to reduce Ollama contention with live user requests
            if i < len(queries) - 1 and settings.warming_delay_seconds > 0:
                await asyncio.sleep(settings.warming_delay_seconds)

        logger.info(
            f"[ARQ] Cache warming complete: {warmed}/{len(queries)} queries processed "
            f"(triggered by: {triggered_by})"
        )
        return {"success": True, "warmed": warmed, "total": len(queries)}

    except Exception as e:
        logger.error(f"[ARQ] Cache warming task failed: {e}")
        return {"success": False, "error": str(e), "warmed": 0, "total": len(queries)}
    finally:
        db.close()
