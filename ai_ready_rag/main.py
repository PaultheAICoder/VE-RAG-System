"""FastAPI application entry point."""

import asyncio
import logging
import re
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
from ai_ready_rag.services.warming_queue import WarmingQueueService

logger = logging.getLogger(__name__)
settings = get_settings()


def _strip_numbering(text: str) -> str:
    """Strip leading numbering from a question (e.g., '1. Question' -> 'Question')."""
    return re.sub(r"^\d+[\.\)\-\s]+", "", text.strip())


async def folder_watcher(queue_service: WarmingQueueService) -> None:
    """Watch for new files in queue directory (CLI-based warming).

    Scans the queue directory for .txt/.csv files, validates them,
    parses queries, creates jobs, and starts processing.
    Invalid files are quarantined with reason.
    """
    from ai_ready_rag.api.admin import _warm_file_task

    while True:
        try:
            # Scan for non-JSON files in queue directory (not jobs subdirectory)
            for file_path in queue_service.queue_dir.iterdir():
                # Skip directories, hidden files, and JSON files
                if file_path.is_dir() or file_path.name.startswith("."):
                    continue
                if file_path.suffix.lower() == ".json":
                    continue

                # Validate extension
                if file_path.suffix.lower() not in settings.warming_allowed_extensions:
                    logger.warning(f"Invalid extension: {file_path.name}")
                    queue_service._quarantine_file(file_path, "invalid_extension")
                    continue

                # Validate size
                try:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                except OSError as e:
                    logger.warning(f"Cannot read file: {file_path.name}: {e}")
                    continue

                if size_mb > settings.warming_max_file_size_mb:
                    logger.warning(f"File too large: {file_path.name} ({size_mb:.1f}MB)")
                    queue_service._quarantine_file(
                        file_path,
                        f"file_too_large: {size_mb:.1f}MB > {settings.warming_max_file_size_mb}MB",
                    )
                    continue

                # Try to read and parse
                try:
                    content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError as e:
                    logger.warning(f"Encoding error: {file_path.name}: {e}")
                    queue_service._quarantine_file(file_path, f"encoding_error: {e}")
                    continue
                except OSError as e:
                    logger.warning(f"Cannot read file: {file_path.name}: {e}")
                    continue

                # Parse queries (one per line, strip numbering)
                queries = [
                    _strip_numbering(line)
                    for line in content.strip().split("\n")
                    if line.strip() and not line.strip().startswith("#")
                ]

                if not queries:
                    logger.warning(f"Empty file (no valid queries): {file_path.name}")
                    file_path.unlink()
                    continue

                # Create job and remove source file
                job = queue_service.create_job(queries, triggered_by="cli")
                file_path.unlink()
                logger.info(
                    f"Created warming job {job.id} from {file_path.name} ({len(queries)} queries)"
                )

                # Start processing in background
                asyncio.create_task(_warm_file_task(job.id, "cli"))

        except Exception as e:
            logger.error(f"Folder watcher error: {e}")

        # Sleep until next scan
        await asyncio.sleep(settings.warming_scan_interval_seconds)


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

    # Initialize warming queue service
    warming_service = WarmingQueueService(
        queue_dir=settings.warming_queue_dir,
        lock_timeout_minutes=settings.warming_lock_timeout_minutes,
        checkpoint_interval=settings.warming_checkpoint_interval,
        archive_completed=settings.warming_archive_completed,
    )

    # Cleanup old failed jobs (>7 days)
    deleted_count = warming_service.cleanup_old_failed_jobs(
        retention_days=settings.warming_failed_job_retention_days
    )
    if deleted_count:
        logger.info(f"Cleaned up {deleted_count} old failed warming jobs")

    # Auto-resume pending/running warming jobs
    from ai_ready_rag.api.admin import _warm_file_task

    pending_jobs = warming_service.list_pending_jobs()
    if pending_jobs:
        for job in pending_jobs:
            # Reset running jobs to pending (they crashed mid-processing)
            if job.status == "running":
                job.status = "pending"
                job.locked_by = None
                job.locked_at = None
                warming_service.update_job(job)

            # Start processing task
            asyncio.create_task(_warm_file_task(job.id, job.triggered_by))

        logger.info(f"Auto-resumed {len(pending_jobs)} warming jobs")
        print(f"Auto-resumed {len(pending_jobs)} warming jobs")

    # Start folder watcher as background task
    watcher_task = asyncio.create_task(folder_watcher(warming_service))
    app.state.watcher_task = watcher_task
    app.state.warming_service = warming_service
    logger.info(
        f"Folder watcher started (interval: {settings.warming_scan_interval_seconds}s, "
        f"dir: {settings.warming_queue_dir})"
    )

    yield

    # Shutdown
    # Cancel folder watcher
    if hasattr(app.state, "watcher_task") and app.state.watcher_task:
        app.state.watcher_task.cancel()
        try:
            await app.state.watcher_task
        except asyncio.CancelledError:
            pass
        logger.info("Folder watcher stopped")

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
