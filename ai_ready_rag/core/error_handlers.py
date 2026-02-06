"""Global error handlers for FastAPI."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai_ready_rag.core.exceptions import AppError


def register_error_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app.

    Catches all AppError subclasses and returns a consistent JSON response
    with `detail`, `error_code`, and any extra context fields.
    """

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        content: dict = {
            "detail": exc.detail,
            "error_code": exc.error_code,
        }
        if exc.context:
            content.update(exc.context)
        return JSONResponse(
            status_code=exc.status_code,
            content=content,
        )
