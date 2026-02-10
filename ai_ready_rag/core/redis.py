"""Redis connection pool with degraded mode fallback.

The application works without Redis (degraded mode):
- Background tasks fall back to FastAPI BackgroundTasks (in-process)
- Chat and all other functionality continues normally
- Health endpoint reports Redis as unavailable
"""

import logging

from arq.connections import ArqRedis, RedisSettings, create_pool

from ai_ready_rag.config import get_settings

logger = logging.getLogger(__name__)

_redis_pool: ArqRedis | None = None
_redis_checked: bool = False  # True after first connection attempt


def _parse_redis_settings() -> RedisSettings:
    """Parse redis_url into ARQ RedisSettings with fast failure."""
    settings = get_settings()
    base = RedisSettings.from_dsn(settings.redis_url)
    # Override retry settings for fast degraded-mode detection
    return RedisSettings(
        host=base.host,
        port=base.port,
        unix_socket_path=base.unix_socket_path,
        database=base.database,
        password=base.password,
        ssl=base.ssl,
        conn_timeout=2,
        conn_retries=0,
        conn_retry_delay=0,
    )


async def get_redis_pool() -> ArqRedis | None:
    """Get or create the Redis connection pool.

    Returns None if Redis is unavailable (degraded mode).
    Safe to call repeatedly — creates pool once and caches it.
    Only attempts connection once; after failure, returns None without retrying.
    """
    global _redis_pool, _redis_checked
    if _redis_pool is not None:
        return _redis_pool
    if _redis_checked:
        return None  # Already tried and failed

    _redis_checked = True
    try:
        _redis_pool = await create_pool(_parse_redis_settings())
        logger.info("Redis connection pool created")
        return _redis_pool
    except (ConnectionError, OSError, Exception) as e:
        logger.warning(f"Redis unavailable — running in degraded mode: {e}")
        return None


async def close_redis_pool() -> None:
    """Close the Redis connection pool on shutdown."""
    global _redis_pool, _redis_checked
    if _redis_pool is not None:
        await _redis_pool.aclose()
        _redis_pool = None
        logger.info("Redis connection pool closed")
    _redis_checked = False


async def is_redis_available() -> bool:
    """Check if Redis is connected and responding.

    Unlike get_redis_pool(), this retries when the cached pool is None.
    This allows recovery after Redis comes back online.
    """
    global _redis_pool, _redis_checked

    pool = await get_redis_pool()

    # If pool is None, reset the check flag and try once more.
    # This allows recovery when Redis was down at startup but is now back.
    if pool is None:
        _redis_checked = False
        pool = await get_redis_pool()
        if pool is None:
            return False

    try:
        await pool.ping()
        return True
    except Exception:
        # Pool exists but ping failed — Redis went down after connect.
        # Invalidate the cached pool so next call retries.
        _redis_pool = None
        _redis_checked = False
        return False
