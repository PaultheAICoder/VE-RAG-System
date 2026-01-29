"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_ready_rag.api import admin, auth, chat, documents, health, tags, users
from ai_ready_rag.config import get_settings
from ai_ready_rag.db.database import SessionLocal, init_db
from ai_ready_rag.db.models import Document

logger = logging.getLogger(__name__)
settings = get_settings()


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

    yield
    # Shutdown
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
