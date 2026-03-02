"""FastAPI application entry point."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai_ready_rag.api import (
    admin,
    auth,
    chat,
    documents,
    evaluations,
    experimental,
    health,
    jobs,
    setup,
    suggestions,
    tags,
    users,
)
from ai_ready_rag.api.tenant import router as tenant_router
from ai_ready_rag.config import get_settings
from ai_ready_rag.core.error_handlers import register_error_handlers
from ai_ready_rag.core.logging import configure_logging
from ai_ready_rag.core.redis import close_redis_pool, get_redis_pool
from ai_ready_rag.core.security import hash_password
from ai_ready_rag.db.database import SessionLocal, init_db
from ai_ready_rag.db.models import Document, SystemSetup, User
from ai_ready_rag.middleware.request_logging import RequestLoggingMiddleware
from ai_ready_rag.modules.registry import init_registry
from ai_ready_rag.services.factory import get_vector_service
from ai_ready_rag.workers.arq_worker import EmbeddedArqWorker
from ai_ready_rag.workers.warming_cleanup import WarmingCleanupService
from ai_ready_rag.workers.warming_worker import WarmingWorker, recover_stale_batches

settings = get_settings()
configure_logging(settings.log_level, settings.log_format)
logger = logging.getLogger(__name__)


def seed_admin_user() -> None:
    """Seed admin user from config if no admin exists. Idempotent."""
    db = SessionLocal()
    try:
        # Check if any admin user exists
        existing_admin = db.query(User).filter(User.role == "admin").first()
        if existing_admin:
            logger.debug("Admin user already exists: %s", existing_admin.email)
            return

        # Create admin from config/env vars
        admin_user = User(
            email=settings.admin_email,
            display_name=settings.admin_display_name,
            password_hash=hash_password(settings.admin_password),
            role="admin",
            is_active=True,
            must_reset_password=True,
        )
        db.add(admin_user)

        # Ensure SystemSetup record exists with password_changed=False
        existing_setup = db.query(SystemSetup).first()
        if not existing_setup:
            setup_record = SystemSetup(
                setup_complete=False,
                admin_password_changed=False,
            )
            db.add(setup_record)

        db.commit()
        logger.info("Auto-created admin user: %s", admin_user.email)
    except Exception:
        db.rollback()
        logger.exception("Failed to seed admin user")
        raise
    finally:
        db.close()


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
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"RAG enabled: {settings.enable_rag}")

    # Track server start time for uptime calculation
    app.state.start_time = time.time()

    # Store settings in app.state for Depends() access
    app.state.settings = settings

    init_db()
    seed_admin_user()

    # Initialize ModuleRegistry and load active modules
    _active_modules = getattr(settings, "active_modules", ["core"])
    registry = init_registry(_active_modules)
    logger.info("module_registry_initialized", extra={"modules": registry.active_modules})

    # Discover excel_tables schema and register SQL templates (no-op if schema is empty)
    if settings.use_ingestkit_excel and "postgresql" in settings.database_url:
        from ai_ready_rag.services.excel_tables_service import ExcelTablesService

        n = ExcelTablesService(settings.database_url).discover_and_register_all()
        logger.info("excel_tables_service: registered %d sql templates", n)

    # Unified table registration — reads both registries, no P&L special-casing
    if "postgresql" in str(settings.database_url):
        from ai_ready_rag.services.table_registration_service import TableRegistrationService

        _trs = TableRegistrationService(
            database_url=settings.database_url,
            tenant_id="default",
            settings=settings,
        )
        _n_unified = _trs.discover_and_register_all()
        logger.info("table_registration_service: registered %d templates", _n_unified)

    # Verify evaluation tables exist (fail-fast if eval_enabled and tables missing)
    if settings.eval_enabled:
        from sqlalchemy import inspect as sa_inspect

        from ai_ready_rag.db.database import engine
        from ai_ready_rag.db.models.evaluation import (
            DatasetSample,
            EvaluationDataset,
            EvaluationRun,
            EvaluationSample,
            LiveEvaluationScore,
        )

        inspector = sa_inspect(engine)
        existing_tables = set(inspector.get_table_names())
        required_tables = {
            EvaluationDataset.__tablename__,
            DatasetSample.__tablename__,
            EvaluationRun.__tablename__,
            EvaluationSample.__tablename__,
            LiveEvaluationScore.__tablename__,
        }
        missing = required_tables - existing_tables
        if missing:
            msg = (
                f"Evaluation tables not found: {', '.join(sorted(missing))}. "
                "Run database migration or restart with EVAL_ENABLED=false"
            )
            logger.error(msg)
            raise RuntimeError(msg)
        logger.info("Evaluation schema verified: all tables present")

    # Validate ingestkit-forms config (fail-fast if misconfigured)
    if settings.use_ingestkit_forms:
        import os

        try:
            import ingestkit_forms  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "use_ingestkit_forms=True but ingestkit-forms is not installed. "
                "Install with: pip install ingestkit-forms"
            ) from exc

        # Validate template storage path
        tmpl_path = Path(settings.forms_template_storage_path).resolve()
        if not tmpl_path.exists():
            tmpl_path.mkdir(parents=True, exist_ok=True)
        if not os.access(tmpl_path, os.W_OK):
            raise RuntimeError(f"forms_template_storage_path not writable: {tmpl_path}")

        # Ensure forms_data schema exists in postgres
        from ai_ready_rag.services.ingestkit_adapters import VERagFormDBAdapter

        VERagFormDBAdapter(database_url=settings.database_url)  # creates schema if needed

        logger.info(
            "ingestkit-forms validated: db=postgres(forms_data), templates=%s",
            settings.forms_template_storage_path,
        )

    # Initialize VectorService once (expensive — singleton for app lifetime)
    vector_service = get_vector_service(settings)
    await vector_service.initialize()
    app.state.vector_service = vector_service
    logger.info("VectorService initialized (singleton)")

    # Preload chat_model into Ollama VRAM at startup (prevents cold-start eviction)
    # Non-fatal: if Ollama is not yet ready, the first real query will load the model.
    try:
        async with httpx.AsyncClient(timeout=10.0) as _preload_client:
            await _preload_client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.chat_model,
                    "prompt": "",
                    "keep_alive": "24h",
                    "stream": False,
                },
            )
        logger.info(f"Preloaded chat_model into VRAM: {settings.chat_model}")
    except Exception as _preload_err:
        logger.warning(f"Chat model preload failed (non-fatal): {_preload_err}")

    # Initialize Redis connection pool (None if unavailable — degraded mode)
    redis_pool = await get_redis_pool()
    arq_worker: EmbeddedArqWorker | None = None
    if redis_pool:
        logger.info("Redis connected — ARQ task queue available")

        # Flush ALL stale ARQ state from previous server runs.
        # - job keys: prevent stale queued jobs from running with outdated code/state
        # - queue sorted set: the worker polls this; stale entries re-run old jobs
        # - result keys: prevent old cached failures from blocking re-execution
        # - in-progress keys: prevent orphaned job markers from filling job slots
        # - retry keys: prevent stale retry backoff from delaying new attempts
        try:
            stale_keys = []
            for pattern in (
                "arq:job:*",
                "arq:result:*",
                "arq:in-progress:*",
                "arq:retry:*",
            ):
                stale_keys.extend(await redis_pool.keys(pattern))
            # Delete both the default queue key and the tenant-specific queue key.
            # arq_queue_name returns "arq:queue:{tenant_id}" (the full Redis key),
            # so both must be cleared to prevent stale jobs from running after restart.
            stale_keys.append("arq:queue")
            stale_keys.append(settings.arq_queue_name)
            if stale_keys:
                await redis_pool.delete(*stale_keys)
                logger.info(f"Cleared {len(stale_keys)} stale ARQ keys")
        except Exception as e:
            logger.warning(f"Failed to clear stale ARQ keys: {e}")

        arq_worker = EmbeddedArqWorker(redis_pool, settings, vector_service)
        await arq_worker.start()
    else:
        logger.warning("Redis unavailable — running in degraded mode (BackgroundTasks fallback)")

    # Store in app.state so watchdog and admin routes can access/update it
    app.state.arq_worker = arq_worker
    app.state.redis_pool = redis_pool

    # Recover stuck documents from previous crashes and re-enqueue all pending docs.
    # The ARQ flush above wiped any queued jobs, so every document currently in
    # 'pending' or 'processing' status needs a fresh job enqueued.
    db = SessionLocal()
    try:
        # Reset processing → pending (worker was killed mid-job)
        stuck_count = (
            db.query(Document)
            .filter(Document.status == "processing")
            .update({"status": "pending", "error_message": None})
        )
        db.commit()
        if stuck_count:
            logger.warning(f"Reset {stuck_count} stuck documents to pending status")

        # Re-enqueue every pending document — their ARQ jobs were cleared by the flush
        if redis_pool:
            pending_docs = db.query(Document).filter(Document.status == "pending").all()
            if pending_docs:
                enqueued = 0
                for doc in pending_docs:
                    try:
                        await redis_pool.enqueue_job(
                            "process_document",
                            doc.id,
                            None,  # processing_options_dict
                            False,  # delete_existing
                            _queue_name=settings.arq_queue_name,
                        )
                        enqueued += 1
                    except Exception as _eq:
                        logger.warning(f"Failed to re-enqueue pending doc {doc.id}: {_eq}")
                logger.info(f"Re-enqueued {enqueued} pending documents after startup flush")
    finally:
        db.close()

    # Initialize QueryRouter singleton (SQL-first deterministic routing)
    from ai_ready_rag.services.factory import get_query_router

    query_router = get_query_router()
    app.state.query_router = query_router
    logger.info("QueryRouter initialized (SQL-first routing)")

    # Initialize and start DB-based WarmingWorker
    from ai_ready_rag.services.rag_service import RAGService

    rag_service = RAGService(settings, vector_service=vector_service, query_router=query_router)
    warming_worker = WarmingWorker(rag_service, settings)
    await warming_worker.start()

    # Recover batches with expired leases (from server crash)
    recovered_count = await recover_stale_batches()
    if recovered_count:
        logger.info(f"Recovered {recovered_count} warming batches with expired leases")

    logger.info("WarmingWorker started")

    # Initialize and start EvaluationWorker (if eval enabled)
    eval_worker = None
    eval_service = None
    if settings.eval_enabled:
        from ai_ready_rag.services.evaluation_service import EvaluationService
        from ai_ready_rag.workers.evaluation_worker import (
            EvaluationWorker,
            recover_stale_evaluation_runs,
        )

        eval_service = EvaluationService(settings, rag_service)
        eval_worker = EvaluationWorker(eval_service, settings)
        await eval_worker.start()

        eval_recovered = await recover_stale_evaluation_runs()
        if eval_recovered:
            logger.info(f"Recovered {eval_recovered} stale evaluation runs")

        logger.info("EvaluationWorker started")

    app.state.eval_worker = eval_worker

    # Initialize and start LiveEvaluationQueue (if eval enabled)
    live_eval_queue = None
    if settings.eval_enabled and eval_service is not None:
        from ai_ready_rag.workers.live_eval_queue import LiveEvaluationQueue

        live_eval_queue = LiveEvaluationQueue(eval_service, settings)
        await live_eval_queue.start()
        rag_service._live_eval_queue = live_eval_queue
        logger.info("LiveEvaluationQueue started")

    app.state.live_eval_queue = live_eval_queue

    # Initialize and start WarmingCleanupService
    warming_cleanup = WarmingCleanupService(settings)
    await warming_cleanup.start()
    logger.info("WarmingCleanupService started")

    # Periodic recovery for stuck processing documents and orphaned pending docs
    async def _stale_processing_recovery_loop():
        """Two-pass recovery every 5 minutes:

        Pass 1 — Reset documents stuck in 'processing' for >15 min back to
        'pending' (existing behaviour, guards against worker crash mid-job).

        Pass 2 — Re-enqueue documents stuck in 'pending' for >20 min that have
        no active ARQ job (orphaned when the worker died before consuming them).
        Safe to re-enqueue duplicates: process_document skips docs already 'ready'.
        """
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            try:
                from datetime import datetime, timedelta

                recovery_db = SessionLocal()
                try:
                    now = datetime.utcnow()

                    # Pass 1: processing → pending for docs stuck >15 min
                    processing_cutoff = now - timedelta(minutes=15)
                    reset_count = (
                        recovery_db.query(Document)
                        .filter(
                            Document.status == "processing",
                            Document.uploaded_at < processing_cutoff,
                        )
                        .update(
                            {"status": "pending", "error_message": None},
                            synchronize_session=False,
                        )
                    )
                    recovery_db.commit()
                    if reset_count:
                        logger.warning(
                            "[RECOVERY] Reset %d stale processing documents to pending",
                            reset_count,
                        )

                    # Pass 2: re-enqueue orphaned pending docs stuck >20 min
                    current_redis = app.state.redis_pool
                    if current_redis:
                        pending_cutoff = now - timedelta(minutes=20)
                        orphaned = (
                            recovery_db.query(Document)
                            .filter(
                                Document.status == "pending",
                                Document.uploaded_at < pending_cutoff,
                            )
                            .all()
                        )
                        if orphaned:
                            enqueued = 0
                            for doc in orphaned:
                                try:
                                    await current_redis.enqueue_job(
                                        "process_document",
                                        doc.id,
                                        None,  # processing_options_dict
                                        False,  # delete_existing
                                        _queue_name=settings.arq_queue_name,
                                    )
                                    enqueued += 1
                                except Exception as _eq:
                                    logger.warning(
                                        "[RECOVERY] Failed to re-enqueue document %s: %s",
                                        doc.id,
                                        _eq,
                                    )
                            if enqueued:
                                logger.warning(
                                    "[RECOVERY] Re-enqueued %d orphaned pending documents",
                                    enqueued,
                                )
                finally:
                    recovery_db.close()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[RECOVERY] Error in stale processing recovery loop")

    # Watchdog: checks every 60 s whether the ARQ worker task is still alive
    # and restarts it if it has crashed.  Pass-2 of the recovery loop will
    # re-enqueue any pending docs that lost their ARQ jobs when the worker died.
    async def _arq_worker_watchdog_loop():
        """Restart the embedded ARQ worker if its asyncio task dies unexpectedly."""
        while True:
            await asyncio.sleep(60)
            try:
                current_worker: EmbeddedArqWorker | None = getattr(app.state, "arq_worker", None)
                current_redis = app.state.redis_pool

                # Nothing to watch in degraded mode (no Redis)
                if current_worker is None or current_redis is None:
                    continue

                if current_worker._task is not None and current_worker._task.done():
                    # Capture the exception (if any) before discarding the task
                    exc = None
                    try:
                        exc = current_worker._task.exception()
                    except (asyncio.CancelledError, asyncio.InvalidStateError):
                        pass

                    logger.error(
                        "[WATCHDOG] ARQ worker task died unexpectedly — restarting",
                        exc_info=exc,
                    )

                    new_worker = EmbeddedArqWorker(current_redis, settings, vector_service)
                    await new_worker.start()
                    app.state.arq_worker = new_worker
                    logger.info("[WATCHDOG] ARQ worker restarted successfully")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[WATCHDOG] Error in ARQ worker watchdog loop")

    recovery_task = asyncio.create_task(_stale_processing_recovery_loop())
    logger.info("Stale processing recovery loop started (every 5 minutes)")

    watchdog_task = asyncio.create_task(_arq_worker_watchdog_loop())
    logger.info("ARQ worker watchdog started (every 60 seconds)")

    yield

    # Shutdown
    recovery_task.cancel()
    watchdog_task.cancel()
    try:
        await asyncio.gather(recovery_task, watchdog_task, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    logger.info("Stale processing recovery loop stopped")
    logger.info("ARQ worker watchdog stopped")

    await warming_cleanup.stop()
    logger.info("WarmingCleanupService stopped")

    if live_eval_queue:
        await live_eval_queue.stop()
        logger.info("LiveEvaluationQueue stopped")

    if eval_worker:
        await eval_worker.stop()
        logger.info("EvaluationWorker stopped")

    await warming_worker.stop()
    logger.info("WarmingWorker stopped")

    # Use app.state.arq_worker — the watchdog may have replaced the original instance
    final_worker: EmbeddedArqWorker | None = getattr(app.state, "arq_worker", None)
    if final_worker:
        await final_worker.stop()

    await close_redis_pool()

    logger.info("Shutting down...")


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

# Request logging middleware (adds X-Request-ID, logs method/path/latency)
app.add_middleware(RequestLoggingMiddleware)

# API Routes
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(setup.router, prefix="/api/setup", tags=["Setup"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(tags.router, prefix="/api/tags", tags=["Tags"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(suggestions.router, prefix="/api/documents", tags=["Tag Suggestions"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["Jobs"])
app.include_router(experimental.router, prefix="/api", tags=["Experimental"])
app.include_router(evaluations.router, prefix="/api/evaluations", tags=["Evaluations"])
app.include_router(tenant_router)

# Form template management (optional -- requires ingestkit-forms)
if settings.use_ingestkit_forms:
    try:
        from ai_ready_rag.api.forms_templates import router as forms_router

        app.include_router(forms_router, prefix="/api/forms", tags=["Form Templates"])
        logger.info("forms.router.mounted")
    except ImportError:
        logger.warning("forms.router.import_failed")

# Serve React frontend static files if dist exists
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIR.exists():
    # Mount static assets (js, css, etc.)
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="static")
    logger.info(f"React frontend mounted from {FRONTEND_DIR}")

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
