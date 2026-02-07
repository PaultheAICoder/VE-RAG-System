"""Request logging middleware with request_id propagation.

Logs every HTTP request with method, path, status code, latency, and user_id.
Adds X-Request-ID header to responses for tracing.
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ai_ready_rag.core.logging import request_id_var

logger = logging.getLogger(__name__)

# Paths to skip logging (health checks, static files)
SKIP_PATHS = frozenset({"/api/health", "/api/version", "/favicon.ico"})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log HTTP requests with structured metadata."""

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = str(uuid.uuid4())
        request_id_var.set(rid)

        if request.url.path in SKIP_PATHS or request.url.path.startswith("/static"):
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response

        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        # Extract user_id from request state if auth middleware set it
        user_id = getattr(request.state, "user_id", None) if hasattr(request, "state") else None

        logger.info(
            "http_request",
            extra={
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "user_id": user_id,
            },
        )

        response.headers["X-Request-ID"] = rid
        return response
