"""Health check endpoints."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.redis import is_redis_available
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import Tag

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


def _check_forms_installed() -> bool:
    """Return True if ingestkit_forms package is importable."""
    try:
        import ingestkit_forms  # noqa: F401

        return True
    except ImportError:
        return False


def _check_forms_db(app_settings) -> bool:
    """Return True if the forms SQLite DB path is writable."""
    try:
        db_path = Path(app_settings.forms_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Try opening the file in append mode to verify writability
        with open(db_path, "a"):
            pass
        return True
    except Exception:
        return False


def _check_template_store(app_settings) -> bool:
    """Return True if the template storage directory is readable."""
    try:
        store_path = Path(app_settings.forms_template_storage_path)
        return store_path.is_dir() and any(True for _ in store_path.iterdir())
    except Exception:
        return False


def _count_templates(app_settings) -> tuple[int, int]:
    """Count approved and draft templates via FileSystemTemplateStore.

    Returns (approved, draft). Returns (0, 0) if ingestkit_forms is not
    installed or the store is empty/unreadable.
    """
    try:
        from ingestkit_forms import FileSystemTemplateStore

        store = FileSystemTemplateStore(
            base_path=app_settings.forms_template_storage_path,
        )
        templates = store.list_templates()
        approved = sum(1 for t in templates if getattr(t, "status", None) == "approved")
        draft = sum(1 for t in templates if getattr(t, "status", None) == "draft")
        return approved, draft
    except Exception:
        return 0, 0


def _build_auto_tagging_status(app_settings, db: Session) -> dict:
    """Build the auto_tagging section of the health response.

    Loads the configured strategy YAML and reports its metadata.
    Falls back to generic strategy or error status on failure.
    """
    if not app_settings.auto_tagging_enabled:
        return {"enabled": False}

    from ai_ready_rag.services.auto_tagging.strategy import AutoTagStrategy

    result: dict = {
        "enabled": True,
        "strategy": app_settings.auto_tagging_strategy,
        "path_enabled": app_settings.auto_tagging_path_enabled,
        "llm_enabled": app_settings.auto_tagging_llm_enabled,
        "llm_model": app_settings.auto_tagging_llm_model,
        "require_approval": app_settings.auto_tagging_require_approval,
    }

    strategy_path = (
        f"{app_settings.auto_tagging_strategies_dir}/{app_settings.auto_tagging_strategy}.yaml"
    )

    strategy = None
    strategy_status = "error"

    try:
        strategy = AutoTagStrategy.load(strategy_path)
        strategy_status = "loaded"
    except (FileNotFoundError, ValueError):
        logger.warning("Failed to load strategy '%s', trying generic fallback", strategy_path)
        if app_settings.auto_tagging_strategy != "generic":
            generic_path = f"{app_settings.auto_tagging_strategies_dir}/generic.yaml"
            try:
                strategy = AutoTagStrategy.load(generic_path)
                strategy_status = "fallback_generic"
            except (FileNotFoundError, ValueError):
                logger.warning("Generic fallback strategy also failed to load")

    if strategy is not None:
        result["strategy_name"] = strategy.name
        result["strategy_version"] = strategy.version
        result["namespaces"] = list(strategy.namespaces.keys())
        result["document_types"] = len(strategy.document_types)
        result["path_rules"] = len(strategy.path_rules)
    else:
        result["strategy_name"] = None
        result["strategy_version"] = None
        result["namespaces"] = []
        result["document_types"] = 0
        result["path_rules"] = 0

    result["strategy_status"] = strategy_status

    # Live DB count of client namespace tags
    current_client_tags = db.query(Tag).filter(Tag.name.like("client:%")).count()
    result["guardrails"] = {
        "max_tags_per_doc": app_settings.auto_tagging_max_tags_per_doc,
        "max_client_tags": app_settings.auto_tagging_max_client_tags,
        "current_client_tags": current_client_tags,
    }

    return result


@router.get("/health")
async def health_check(request: Request, db: Session = Depends(get_db)):
    """Health check endpoint."""
    redis_ok = await is_redis_available()

    eval_status = {"enabled": settings.eval_enabled}
    eval_worker = getattr(request.app.state, "eval_worker", None)
    if eval_worker:
        eval_status["worker_running"] = eval_worker._task is not None
        eval_status["current_run_id"] = eval_worker._current_run_id
    else:
        eval_status["worker_running"] = False
        eval_status["current_run_id"] = None

    live_queue = getattr(request.app.state, "live_eval_queue", None)
    if live_queue:
        eval_status["live_queue_depth"] = live_queue.depth
        eval_status["live_queue_capacity"] = live_queue.capacity
        eval_status["live_queue_drops"] = live_queue.drops_since_startup
        eval_status["live_queue_processed"] = live_queue.processed_since_startup
    else:
        eval_status["live_queue_depth"] = 0
        eval_status["live_queue_capacity"] = 0
        eval_status["live_queue_drops"] = 0
        eval_status["live_queue_processed"] = 0

    # Forms health
    forms_status: dict = {"enabled": settings.use_ingestkit_forms}
    if settings.use_ingestkit_forms:
        forms_status["package_installed"] = _check_forms_installed()
        forms_status["forms_db_writable"] = _check_forms_db(settings)
        forms_status["template_store_readable"] = _check_template_store(settings)
        approved, draft = _count_templates(settings)
        forms_status["templates_approved"] = approved
        forms_status["templates_draft"] = draft

    return {
        "status": "healthy",
        "version": settings.app_version,
        "database": "sqlite",
        "redis": "connected" if redis_ok else "unavailable",
        "rag_enabled": settings.enable_rag,
        "profile": settings.env_profile,
        "backends": {
            "vector": settings.vector_backend,
            "chunker": settings.chunker_backend,
            "ocr_enabled": settings.enable_ocr,
        },
        "evaluation": eval_status,
        "forms": forms_status,
        "auto_tagging": _build_auto_tagging_status(settings, db),
    }


@router.get("/version")
async def version():
    """Get version info."""
    return {"name": settings.app_name, "version": settings.app_version}
