"""FastAPI application entry point."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai_ready_rag.api import admin, auth, chat, documents, health, setup, tags, users
from ai_ready_rag.config import get_settings
from ai_ready_rag.db.database import SessionLocal, init_db
from ai_ready_rag.db.models import Document
from ai_ready_rag.services.warming_cleanup import WarmingCleanupService
from ai_ready_rag.services.warming_worker import WarmingWorker, recover_stale_jobs

logger = logging.getLogger(__name__)
settings = get_settings()

# Global warming worker instance (managed by lifespan)
warming_worker: WarmingWorker | None = None
# Global warming cleanup service instance (managed by lifespan)
warming_cleanup: WarmingCleanupService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    # Startup
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"  ENV_PROFILE: {settings.env_profile}")
    logger.info(f"  Vector backend: {settings.vector_backend}")
    logger.info(f"  Chunker backend: {settings.chunker_backend}")
    logger.info(f"  OCR enabled: {settings.enable_ocr}")
    logger.info(f"  Chat model: {settings.chat_model}")
    logger.info(f"  Embedding model: {settings.embedding_model}")
    logger.info("=" * 60)
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Debug mode: {settings.debug}")
    print(f"RAG enabled: {settings.enable_rag}")

    # Track server start time for uptime calculation
    app.state.start_time = time.time()

    init_db()

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
            print(f"Recovered {stuck_count} stuck documents")
    finally:
        db.close()

    # Initialize and start DB-based WarmingWorker
    global warming_worker
    from ai_ready_rag.services.rag_service import RAGService

    rag_service = RAGService(settings)
    warming_worker = WarmingWorker(rag_service, settings)
    await warming_worker.start()

    # Recover jobs with expired leases (from server crash)
    recovered_count = await recover_stale_jobs()
    if recovered_count:
        logger.info(f"Recovered {recovered_count} warming jobs with expired leases")

    logger.info("WarmingWorker started")

    # Initialize and start WarmingCleanupService
    global warming_cleanup
    warming_cleanup = WarmingCleanupService(settings)
    await warming_cleanup.start()
    logger.info("WarmingCleanupService started")

    yield

    # Shutdown
    # Stop WarmingCleanupService
    if warming_cleanup:
        await warming_cleanup.stop()
        warming_cleanup = None
        logger.info("WarmingCleanupService stopped")

    # Stop WarmingWorker
    if warming_worker:
        await warming_worker.stop()
        warming_worker = None
        logger.info("WarmingWorker stopped")

    print("Shutting down...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise RAG system with authentication and access control",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.debug else None,
    redoc_url="/api/redoc" if settings.debug else None,
)

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

# Mount Gradio UI at /app if enabled
print(f"Gradio enabled: {settings.enable_gradio}")
if settings.enable_gradio:
    try:
        import gradio as gr

        from ai_ready_rag.ui import create_app

        gradio_app = create_app()
        app = gr.mount_gradio_app(app, gradio_app, path="/app")
        print("Gradio UI mounted at /app")
    except Exception as e:
        print(f"Failed to mount Gradio: {e}")
        import traceback

        traceback.print_exc()


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
        # Don't intercept API or Gradio routes - return proper 404
        if path.startswith("api/") or path.startswith("app/"):
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
            "ui": "/app" if settings.enable_gradio else "disabled",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ai_ready_rag.main:app", host=settings.host, port=settings.port, reload=settings.debug
    )
