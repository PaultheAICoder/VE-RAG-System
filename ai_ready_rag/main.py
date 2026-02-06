"""FastAPI application entry point."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai_ready_rag.api import admin, auth, chat, documents, experimental, health, setup, tags, users
from ai_ready_rag.config import get_settings
from ai_ready_rag.core.error_handlers import register_error_handlers
from ai_ready_rag.db.database import SessionLocal, init_db
from ai_ready_rag.db.models import Document
from ai_ready_rag.services.factory import get_vector_service
from ai_ready_rag.workers.warming_cleanup import WarmingCleanupService
from ai_ready_rag.workers.warming_worker import WarmingWorker, recover_stale_jobs

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize and cleanup services."""
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"  ENV_PROFILE: {settings.env_profile}")
    logger.info(f"  Vector backend: {settings.vector_backend}")
    logger.info(f"  Chunker backend: {settings.chunker_backend}")
    logger.info(f"  OCR enabled: {settings.enable_ocr}")
    logger.info(f"  Chat model: {settings.chat_model}")
    logger.info(f"  Embedding model: {settings.embedding_model}")
    logger.info("=" * 60)
    print(f"Starting {settings.app_name} v{settings.app_version}", flush=True)
    print(f"Debug mode: {settings.debug}", flush=True)
    print(f"RAG enabled: {settings.enable_rag}", flush=True)

    # Track server start time for uptime calculation
    app.state.start_time = time.time()

    # Store settings in app.state for Depends() access
    app.state.settings = settings

    init_db()

    # Initialize VectorService once (expensive — singleton for app lifetime)
    vector_service = get_vector_service(settings)
    await vector_service.initialize()
    app.state.vector_service = vector_service
    logger.info("VectorService initialized (singleton)")

    # Recover stuck documents from previous crashes
    db = SessionLocal()
    try:
        stuck_count = (
            db.query(Document)
            .filter(Document.status == "processing")
            .update({"status": "pending", "error_message": None})
        )
        db.commit()
        if stuck_count:
            logger.warning(f"Reset {stuck_count} stuck documents to pending status")
            print(f"Recovered {stuck_count} stuck documents", flush=True)
    finally:
        db.close()

    # Initialize and start DB-based WarmingWorker
    from ai_ready_rag.services.rag_service import RAGService

    rag_service = RAGService(settings, vector_service=vector_service)
    warming_worker = WarmingWorker(rag_service, settings)
    await warming_worker.start()

    # Recover jobs with expired leases (from server crash)
    recovered_count = await recover_stale_jobs()
    if recovered_count:
        logger.info(f"Recovered {recovered_count} warming jobs with expired leases")

    print("WarmingWorker started", flush=True)
    logger.info("WarmingWorker started")

    # Initialize and start WarmingCleanupService
    warming_cleanup = WarmingCleanupService(settings)
    await warming_cleanup.start()
    print("WarmingCleanupService started", flush=True)
    logger.info("WarmingCleanupService started")

    yield

    # Shutdown
    await warming_cleanup.stop()
    logger.info("WarmingCleanupService stopped")

    await warming_worker.stop()
    logger.info("WarmingWorker stopped")

    print("Shutting down...", flush=True)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise RAG system with authentication and access control",
    docs_url="/api/docs" if settings.debug else None,
    redoc_url="/api/redoc" if settings.debug else None,
    lifespan=lifespan,
)

# Register global error handlers (AppError → JSON responses)
register_error_handlers(app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(setup.router, prefix="/api/setup", tags=["Setup"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(tags.router, prefix="/api/tags", tags=["Tags"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(experimental.router, prefix="/api", tags=["Experimental"])

# Serve React frontend static files if dist exists
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIR.exists():
    # Mount static assets (js, css, etc.)
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="static")
    print(f"React frontend mounted from {FRONTEND_DIR}")

    @app.get("/")
    async def serve_frontend():
        """Serve React frontend."""
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        """Catch-all for SPA client-side routing."""
        # Don't intercept API routes - return proper 404
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")

        # Serve static files if they exist
        file_path = FRONTEND_DIR / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # Otherwise serve index.html for SPA routing
        return FileResponse(FRONTEND_DIR / "index.html")
else:

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.app_version,
            "docs": "/api/docs" if settings.debug else "disabled",
            "health": "/api/health",
            "frontend": "React (build frontend/dist for production)",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ai_ready_rag.main:app", host=settings.host, port=settings.port, reload=settings.debug
    )
