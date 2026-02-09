"""Admin endpoints for system management."""

import asyncio
import json
import logging
import re
import shutil
import subprocess
import time
import uuid
from datetime import UTC, datetime

import bleach
import httpx
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.dependencies import (
    get_optional_current_user,
    require_admin,
    require_system_admin,
)
from ai_ready_rag.core.redis import get_redis_pool
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import (
    CuratedQA,
    CuratedQAKeyword,
    Document,
    QuerySynonym,
    User,
    WarmingBatch,
    WarmingQuery,
)
from ai_ready_rag.schemas.admin import (
    AdvancedSettingsRequest,
    AdvancedSettingsResponse,
    ArchitectureInfoResponse,
    BatchQueriesResponse,
    BulkDeleteRequest,
    CacheClearResponse,
    CacheSeedRequest,
    CacheSeedResponse,
    CacheSettingsRequest,
    CacheSettingsResponse,
    CacheStatsResponse,
    ChangeEmbeddingRequest,
    ChangeEmbeddingResponse,
    ChangeModelRequest,
    ChangeModelResponse,
    ChatModelInfo,
    ClearKnowledgeBaseRequest,
    ClearKnowledgeBaseResponse,
    ComponentHealth,
    CuratedQACreate,
    CuratedQAListResponse,
    CuratedQAResponse,
    CuratedQAUpdate,
    DetailedHealthResponse,
    DocumentParsingInfo,
    EasyOCRStatus,
    EmbeddingsInfo,
    FeatureFlagsRequest,
    FeatureFlagsResponse,
    FileStats,
    InfrastructureStatus,
    KnowledgeBaseStatsResponse,
    KnowledgeBaseSummary,
    LLMSettingsRequest,
    LLMSettingsResponse,
    ManualWarmingRequest,
    ModelInfo,
    ModelLimits,
    ModelLimitsResponse,
    ModelsResponse,
    OCRStatus,
    ProcessingOptionsRequest,
    ProcessingOptionsResponse,
    ProcessingQueueStatus,
    QueryRetryResponse,
    RAGPipelineStatus,
    RecoverResponse,
    ReindexEstimate,
    ReindexFailureInfo,
    ReindexFailuresResponse,
    ReindexJobResponse,
    ResumeReindexRequest,
    RetrievalSettingsRequest,
    RetrievalSettingsResponse,
    SecuritySettingsRequest,
    SecuritySettingsResponse,
    SettingsAuditEntry,
    SettingsAuditResponse,
    StartReindexRequest,
    SynonymCreate,
    SynonymListResponse,
    SynonymResponse,
    SynonymUpdate,
    TesseractStatus,
    TopQueryItem,
    TopQueryResponse,
    WarmingQueryResponse,
    WarmingQueueJobResponse,
    WarmingQueueListResponse,
)
from ai_ready_rag.services.document_service import DocumentService
from ai_ready_rag.services.factory import get_vector_service
from ai_ready_rag.services.model_service import ModelService, OllamaUnavailableError
from ai_ready_rag.services.settings_service import SettingsService, get_model_setting

logger = logging.getLogger(__name__)

# HTML sanitization constants for curated Q&A
ALLOWED_HTML_TAGS = ["p", "br", "strong", "em", "ul", "ol", "li", "a"]
ALLOWED_HTML_ATTRS = {"a": ["href", "title"]}


def sanitize_html(html: str) -> str:
    """Sanitize HTML to prevent XSS."""
    return bleach.clean(
        html,
        tags=ALLOWED_HTML_TAGS,
        attributes=ALLOWED_HTML_ATTRS,
        strip=True,
    )


def sync_qa_keywords(db: Session, qa_id: str, keywords: list[str]) -> None:
    """Sync curated_qa_keywords table when Q&A is created/updated.

    Deletes existing keywords for the Q&A and inserts new tokenized entries.

    Args:
        db: Database session
        qa_id: ID of the CuratedQA record
        keywords: List of keyword phrases
    """
    from ai_ready_rag.services.rag_service import tokenize_query

    # Delete existing keywords
    db.query(CuratedQAKeyword).filter(CuratedQAKeyword.qa_id == qa_id).delete()

    # Insert tokenized keywords
    for original_keyword in keywords:
        tokens = tokenize_query(original_keyword)
        for token in tokens:
            db.add(
                CuratedQAKeyword(
                    qa_id=qa_id,
                    keyword=token,
                    original_keyword=original_keyword.lower().strip(),
                )
            )
    db.flush()


def delete_qa_keywords(db: Session, qa_id: str) -> None:
    """Delete all keywords for a curated Q&A.

    Note: This is also handled by CASCADE DELETE, but provided for explicit cleanup.

    Args:
        db: Database session
        qa_id: ID of the CuratedQA record
    """
    db.query(CuratedQAKeyword).filter(CuratedQAKeyword.qa_id == qa_id).delete()


router = APIRouter()


# Architecture info caching
_architecture_cache: dict = {}
_cache_ttl_seconds: int = 60


def _get_tesseract_status() -> TesseractStatus:
    """Detect Tesseract availability and get version/languages."""
    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        return TesseractStatus(available=False)

    version = None
    languages = None

    try:
        # Get version
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse first line: "tesseract X.X.X"
            first_line = result.stdout.split("\n")[0]
            if "tesseract" in first_line.lower():
                parts = first_line.split()
                if len(parts) >= 2:
                    version = parts[1]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    try:
        # Get languages
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Skip header line, get language codes
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                languages = [lang.strip() for lang in lines[1:] if lang.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return TesseractStatus(
        available=True,
        version=version,
        languages=languages,
    )


def _get_easyocr_status() -> EasyOCRStatus:
    """Detect EasyOCR availability and version."""
    try:
        import easyocr  # noqa: F401

        version = getattr(easyocr, "__version__", None)
        return EasyOCRStatus(available=True, version=version)
    except ImportError:
        return EasyOCRStatus(available=False)


def _get_docling_version() -> str:
    """Get Docling version if installed."""
    try:
        import docling

        return getattr(docling, "__version__", "unknown")
    except ImportError:
        return "not installed"


@router.post("/documents/recover-stuck", response_model=RecoverResponse)
async def recover_stuck_documents(
    max_age_hours: int = Query(2, ge=1, le=168, description="Maximum age in hours"),
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Recover documents stuck in processing state.

    Resets documents that have been in 'processing' status longer than
    max_age_hours back to 'pending' so they can be reprocessed.

    Admin only.
    """
    settings = get_settings()
    service = DocumentService(db, settings)

    recovered = service.recover_stuck_documents(max_age_hours=max_age_hours)

    if recovered > 0:
        logger.warning(
            f"Admin {current_user.email} recovered {recovered} stuck documents "
            f"(max_age_hours={max_age_hours})"
        )

    return RecoverResponse(
        recovered=recovered,
        message=f"Reset {recovered} stuck documents to pending status",
    )


@router.get("/knowledge-base/stats", response_model=KnowledgeBaseStatsResponse)
async def get_knowledge_base_stats(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get knowledge base statistics including per-file details.

    Returns total chunks, unique files, collection info, and
    per-file statistics with chunk counts.

    Admin only.
    """
    settings = get_settings()
    vector_service = get_vector_service(settings)
    await vector_service.initialize()

    try:
        stats = await vector_service.get_extended_stats()
    except Exception as e:
        logger.error(f"Failed to get knowledge base stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve knowledge base statistics: {e}",
        ) from e

    # Convert files to FileStats models
    files = [
        FileStats(
            document_id=f["document_id"],
            filename=f["filename"],
            chunk_count=f["chunk_count"],
            status=None,  # Could join with Document table for status if needed
        )
        for f in stats.get("files", [])
    ]

    return KnowledgeBaseStatsResponse(
        total_chunks=stats.get("total_chunks", 0),
        unique_files=stats.get("unique_files", 0),
        total_vectors=stats.get("total_chunks", 0),  # Same as total_chunks for now
        collection_name=stats.get("collection_name", ""),
        files=files,
        storage_size_bytes=stats.get("collection_size_bytes"),
        last_updated=datetime.now(UTC),
    )


@router.delete("/knowledge-base", response_model=ClearKnowledgeBaseResponse)
async def clear_knowledge_base(
    request: ClearKnowledgeBaseRequest,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Clear all vectors from the knowledge base.

    Requires `confirm: true` in request body to proceed.
    Optionally delete source files from SQLite as well.

    Admin only. This is a destructive operation.
    """
    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation required. Set 'confirm: true' to proceed.",
        )

    settings = get_settings()
    vector_service = get_vector_service(settings)
    await vector_service.initialize()

    # Get stats before clearing to report deleted counts
    try:
        stats_before = await vector_service.get_extended_stats()
        chunks_before = stats_before.get("total_chunks", 0)
        files_before = stats_before.get("unique_files", 0)
    except Exception:
        chunks_before = 0
        files_before = 0

    # Clear vectors
    logger.warning(
        f"DESTRUCTIVE OPERATION: Admin {current_user.email} clearing knowledge base "
        f"(delete_source_files={request.delete_source_files})"
    )

    try:
        success = await vector_service.clear_collection()
    except Exception as e:
        logger.error(f"Failed to clear knowledge base: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear knowledge base: {e}",
        ) from e

    # Handle document records
    deleted_files = 0
    if request.delete_source_files and success:
        # Delete documents from SQLite entirely
        document_service = DocumentService(db, settings)
        deleted_files = document_service.delete_all_documents()
        logger.warning(f"Deleted {deleted_files} documents from database")
    elif success:
        # Reset all documents to pending status (ready for reprocessing)
        # This ensures Processing Queue shows correct state after KB clear
        reset_count = (
            db.query(Document)
            .filter(Document.status.in_(["ready", "failed", "processing"]))
            .update(
                {
                    "status": "pending",
                    "chunk_count": None,
                    "processed_at": None,
                    "error_message": None,
                    "processing_time_ms": None,
                },
                synchronize_session=False,
            )
        )
        db.commit()
        logger.info(f"Reset {reset_count} documents to pending status")

    return ClearKnowledgeBaseResponse(
        deleted_chunks=chunks_before if success else 0,
        deleted_files=deleted_files if request.delete_source_files else files_before,
        success=success,
    )


# Default values for processing options
PROCESSING_DEFAULTS = {
    "enable_ocr": True,
    "force_full_page_ocr": False,
    "ocr_language": "eng",
    "table_extraction_mode": "accurate",
    "include_image_descriptions": True,
    "query_routing_mode": "retrieve_only",  # Default: always search documents
}


def _get_setting_value(service: SettingsService, key: str, default: any) -> any:
    """Get setting value with fallback to default if None."""
    value = service.get(key)
    return value if value is not None else default


@router.get("/processing-options", response_model=ProcessingOptionsResponse)
async def get_processing_options(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get current processing options.

    Returns stored settings merged with defaults for any missing values.
    Admin only.
    """
    service = SettingsService(db)

    return ProcessingOptionsResponse(
        enable_ocr=_get_setting_value(service, "enable_ocr", PROCESSING_DEFAULTS["enable_ocr"]),
        force_full_page_ocr=_get_setting_value(
            service, "force_full_page_ocr", PROCESSING_DEFAULTS["force_full_page_ocr"]
        ),
        ocr_language=_get_setting_value(
            service, "ocr_language", PROCESSING_DEFAULTS["ocr_language"]
        ),
        table_extraction_mode=_get_setting_value(
            service,
            "table_extraction_mode",
            PROCESSING_DEFAULTS["table_extraction_mode"],
        ),
        include_image_descriptions=_get_setting_value(
            service,
            "include_image_descriptions",
            PROCESSING_DEFAULTS["include_image_descriptions"],
        ),
        query_routing_mode=_get_setting_value(
            service,
            "query_routing_mode",
            PROCESSING_DEFAULTS["query_routing_mode"],
        ),
    )


@router.patch("/processing-options", response_model=ProcessingOptionsResponse)
async def update_processing_options(
    options: ProcessingOptionsRequest,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Update processing options.

    Only updates fields that are provided (not None).
    Settings persist to admin_settings table.
    Admin only.
    """
    service = SettingsService(db)

    # Update only provided fields
    if options.enable_ocr is not None:
        service.set("enable_ocr", options.enable_ocr, updated_by=current_user.id)
    if options.force_full_page_ocr is not None:
        service.set("force_full_page_ocr", options.force_full_page_ocr, updated_by=current_user.id)
    if options.ocr_language is not None:
        service.set("ocr_language", options.ocr_language, updated_by=current_user.id)
    if options.table_extraction_mode is not None:
        service.set(
            "table_extraction_mode",
            options.table_extraction_mode,
            updated_by=current_user.id,
        )
    if options.include_image_descriptions is not None:
        service.set(
            "include_image_descriptions",
            options.include_image_descriptions,
            updated_by=current_user.id,
        )
    if options.query_routing_mode is not None:
        service.set(
            "query_routing_mode",
            options.query_routing_mode,
            updated_by=current_user.id,
        )

    logger.info(
        f"Admin {current_user.email} updated processing options: "
        f"{options.model_dump(exclude_none=True)}"
    )

    # Return current state (same as GET)
    return ProcessingOptionsResponse(
        enable_ocr=_get_setting_value(service, "enable_ocr", PROCESSING_DEFAULTS["enable_ocr"]),
        force_full_page_ocr=_get_setting_value(
            service, "force_full_page_ocr", PROCESSING_DEFAULTS["force_full_page_ocr"]
        ),
        ocr_language=_get_setting_value(
            service, "ocr_language", PROCESSING_DEFAULTS["ocr_language"]
        ),
        table_extraction_mode=_get_setting_value(
            service,
            "table_extraction_mode",
            PROCESSING_DEFAULTS["table_extraction_mode"],
        ),
        include_image_descriptions=_get_setting_value(
            service,
            "include_image_descriptions",
            PROCESSING_DEFAULTS["include_image_descriptions"],
        ),
        query_routing_mode=_get_setting_value(
            service,
            "query_routing_mode",
            PROCESSING_DEFAULTS["query_routing_mode"],
        ),
    )


@router.get("/architecture", response_model=ArchitectureInfoResponse)
async def get_architecture_info(
    current_user: User = Depends(require_system_admin),
):
    """Get comprehensive system architecture information.

    Returns system configuration, component health status, and OCR availability.
    Results are cached for 60 seconds to avoid repeated health checks.
    Admin only.
    """
    global _architecture_cache

    # Check cache validity
    cache_timestamp = _architecture_cache.get("timestamp", 0)
    if time.time() - cache_timestamp < _cache_ttl_seconds:
        cached_response = _architecture_cache.get("response")
        if cached_response:
            return cached_response

    # Get settings
    settings = get_settings()

    # Check infrastructure health
    vector_service = get_vector_service(settings)
    await vector_service.initialize()

    try:
        health = await vector_service.health_check()
        # Handle both Qdrant (named tuple) and ChromaDB (dict) responses
        if isinstance(health, dict):
            vector_db_status = "healthy" if health.get("healthy", False) else "unhealthy"
            # Check Ollama separately for ChromaDB
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{settings.ollama_base_url}/api/tags")
                    ollama_status = "healthy" if resp.status_code == 200 else "unhealthy"
            except Exception:
                ollama_status = "unhealthy"
        else:
            # Qdrant returns named tuple
            ollama_status = "healthy" if health.ollama_healthy else "unhealthy"
            vector_db_status = "healthy" if health.qdrant_healthy else "unhealthy"
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        ollama_status = "unhealthy"
        vector_db_status = "unhealthy"

    # Get OCR status
    tesseract_status = _get_tesseract_status()
    easyocr_status = _get_easyocr_status()

    # Get document parsing info based on profile
    if settings.chunker_backend == "docling":
        docling_version = _get_docling_version()
        doc_parsing_info = DocumentParsingInfo(
            engine="Docling",
            version=docling_version,
            type="local ML",
            capabilities=[
                "PDF extraction",
                "Table detection",
                "OCR integration",
                "Semantic chunking",
            ],
        )
    else:
        # SimpleChunker for laptop profile
        doc_parsing_info = DocumentParsingInfo(
            engine="SimpleChunker",
            version="built-in",
            type="basic text splitting",
            capabilities=[
                "Plain text extraction",
                "Token-based chunking",
                "Lightweight processing",
            ],
        )

    # Build response
    response = ArchitectureInfoResponse(
        document_parsing=doc_parsing_info,
        embeddings=EmbeddingsInfo(
            model=get_model_setting("embedding_model", settings.embedding_model),
            dimensions=settings.embedding_dimension,
            vector_store=settings.vector_backend,
            vector_store_url=settings.qdrant_url,
        ),
        chat_model=ChatModelInfo(
            name=get_model_setting("chat_model", settings.chat_model),
            provider="Ollama",
            capabilities=[
                "Text generation",
                "RAG context processing",
                "Citation generation",
            ],
        ),
        infrastructure=InfrastructureStatus(
            ollama_url=settings.ollama_base_url,
            ollama_status=ollama_status,
            vector_db_status=vector_db_status,
        ),
        ocr_status=OCRStatus(
            tesseract=tesseract_status,
            easyocr=easyocr_status,
        ),
        profile=settings.env_profile,
    )

    # Cache the response
    _architecture_cache = {
        "timestamp": time.time(),
        "response": response,
    }

    return response


# Model Configuration Models and Endpoints
CHAT_MODEL_KEY = "chat_model"
EMBEDDING_MODEL_KEY = "embedding_model"


@router.get("/models", response_model=ModelsResponse)
async def get_models(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get available Ollama models and current selection.

    Returns list of available models from Ollama and the currently
    active chat and embedding models.

    Admin only.
    """
    settings = get_settings()
    model_service = ModelService(settings.ollama_base_url)
    settings_service = SettingsService(db)

    try:
        models = await model_service.list_models()
    except OllamaUnavailableError as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e

    # Get current chat model from settings, fallback to config
    current_chat = settings_service.get(CHAT_MODEL_KEY)
    if current_chat is None:
        current_chat = settings.chat_model

    # Get current embedding model from settings, fallback to config
    current_embedding = settings_service.get(EMBEDDING_MODEL_KEY)
    if current_embedding is None:
        current_embedding = settings.embedding_model

    # Convert to ModelInfo response models
    available_models = [
        ModelInfo(
            name=m["name"],
            display_name=m["display_name"],
            size_gb=m["size_gb"],
            parameters=m["parameters"],
            quantization=m["quantization"],
            recommended=m["recommended"],
        )
        for m in models
    ]

    # Filter into embedding and chat models
    embedding_models = [m for m in available_models if ModelService.is_embedding_model(m.name)]
    chat_models = [m for m in available_models if not ModelService.is_embedding_model(m.name)]

    return ModelsResponse(
        available_models=available_models,
        embedding_models=embedding_models,
        chat_models=chat_models,
        current_chat_model=current_chat,
        current_embedding_model=current_embedding,
    )


@router.patch("/models/chat", response_model=ChangeModelResponse)
async def change_chat_model(
    request: ChangeModelRequest,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Change the active chat model.

    Validates that the requested model exists in Ollama before
    accepting the change. Model selection persists to database.

    Note: First request after switch may be slow due to model loading.

    Admin only.
    """
    settings = get_settings()
    model_service = ModelService(settings.ollama_base_url)
    settings_service = SettingsService(db)

    # Validate model exists in Ollama
    try:
        model_exists = await model_service.validate_model(request.model_name)
    except OllamaUnavailableError as e:
        logger.error(f"Cannot validate model - Ollama unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e

    if not model_exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{request.model_name}' not found in Ollama. "
            "Please ensure the model is pulled before selecting it.",
        )

    # Get previous model
    previous_model = settings_service.get(CHAT_MODEL_KEY)
    if previous_model is None:
        previous_model = settings.chat_model

    # Save new model
    settings_service.set(CHAT_MODEL_KEY, request.model_name, updated_by=current_user.id)

    # Audit log
    logger.info(
        f"Admin {current_user.email} changed chat model: {previous_model} -> {request.model_name}"
    )

    return ChangeModelResponse(
        previous_model=previous_model,
        current_model=request.model_name,
        success=True,
        message=f"Chat model changed to {request.model_name}. "
        "First request may be slow while the model loads.",
    )


@router.patch("/models/embedding", response_model=ChangeEmbeddingResponse)
async def change_embedding_model(
    request: ChangeEmbeddingRequest,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Change the embedding model used for vectorization.

    WARNING: Changing the embedding model invalidates all existing vectors.
    Documents must be re-indexed for search to work correctly.

    Requires confirm_reindex=true to acknowledge the re-indexing impact.

    Admin only.
    """
    # Require explicit confirmation
    if not request.confirm_reindex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Changing the embedding model requires re-indexing all documents. "
            "Set confirm_reindex=true to acknowledge this.",
        )

    settings = get_settings()
    model_service = ModelService(settings.ollama_base_url)
    settings_service = SettingsService(db)

    # Validate model exists in Ollama
    try:
        model_exists = await model_service.validate_model(request.model_name)
    except OllamaUnavailableError as e:
        logger.error(f"Cannot validate model - Ollama unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e

    if not model_exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{request.model_name}' not found in Ollama. "
            "Please ensure the model is pulled before selecting it.",
        )

    # Validate it's an embedding model
    if not ModelService.is_embedding_model(request.model_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{request.model_name}' does not appear to be an embedding model. "
            "Embedding models typically have 'embed' in their name.",
        )

    # Get previous model
    previous_model = settings_service.get(EMBEDDING_MODEL_KEY)
    if previous_model is None:
        previous_model = settings.embedding_model

    # Get document count for warning
    from ai_ready_rag.db.models import Document

    documents_affected = db.query(Document).count()

    # Save new model
    settings_service.set(EMBEDDING_MODEL_KEY, request.model_name, updated_by=current_user.id)

    # Audit log
    logger.warning(
        f"Admin {current_user.email} changed embedding model: {previous_model} -> {request.model_name}. "
        f"{documents_affected} documents need re-indexing."
    )

    return ChangeEmbeddingResponse(
        previous_model=previous_model,
        current_model=request.model_name,
        success=True,
        message=f"Embedding model changed to {request.model_name}. "
        f"WARNING: {documents_affected} documents need to be re-indexed. "
        "Clear the knowledge base and re-upload documents, or reprocess all documents.",
        reindex_required=True,
        documents_affected=documents_affected,
    )


@router.get("/model-limits", response_model=ModelLimitsResponse)
async def get_model_limits(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get model-specific limits for settings validation.

    Returns the current chat model's limits and all known model limits.
    Used by frontend to display warnings when settings exceed model limits.

    Admin only.
    """
    from ai_ready_rag.services.rag_constants import MODEL_LIMITS

    settings = get_settings()
    settings_service = SettingsService(db)

    # Get current chat model from settings, fallback to config
    current_model = settings_service.get(CHAT_MODEL_KEY)
    if current_model is None:
        current_model = settings.chat_model

    # Get limits for current model, fallback to qwen3:8b defaults
    default_limits = {"context_window": 32768, "max_response": 2048}
    current_limits = MODEL_LIMITS.get(current_model, default_limits)

    return ModelLimitsResponse(
        current_model=current_model,
        limits=ModelLimits(
            context_window=current_limits["context_window"],
            max_response=current_limits["max_response"],
        ),
        all_models={
            name: ModelLimits(
                context_window=limits["context_window"],
                max_response=limits["max_response"],
            )
            for name, limits in MODEL_LIMITS.items()
        },
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def get_detailed_health(
    request: Request,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Get detailed system health for admin dashboard.

    Returns comprehensive health information including component status,
    RAG pipeline status, knowledge base statistics, and processing queue.

    Admin only (system_admin or customer_admin).
    """
    settings = get_settings()

    # Get uptime from app state
    start_time = getattr(request.app.state, "start_time", time.time())
    uptime_seconds = int(time.time() - start_time)

    # Check component health
    vector_service = get_vector_service(settings)
    await vector_service.initialize()

    ollama_healthy = False
    vector_healthy = False
    total_chunks = 0
    storage_size_mb = None
    vector_db_version = None
    ollama_version = None

    try:
        health = await vector_service.health_check()
        # Handle both Qdrant (named tuple) and ChromaDB (dict) responses
        if isinstance(health, dict):
            # ChromaDB returns {'healthy': bool, ...}
            vector_healthy = health.get("healthy", False)
            # Check Ollama separately for ChromaDB
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{settings.ollama_base_url}/api/tags")
                    ollama_healthy = resp.status_code == 200
            except Exception:
                ollama_healthy = False
        else:
            # Qdrant returns named tuple with ollama_healthy, qdrant_healthy
            ollama_healthy = health.ollama_healthy
            vector_healthy = health.qdrant_healthy
    except Exception as e:
        logger.warning(f"Health check failed: {e}")

    # Get Vector DB version
    try:
        if settings.vector_backend == "qdrant":
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(settings.qdrant_url)
                if resp.status_code == 200:
                    vector_db_version = resp.json().get("version")
        elif settings.vector_backend == "chroma":
            # ChromaDB doesn't have a remote API for version, get from package
            try:
                import chromadb

                vector_db_version = chromadb.__version__
            except ImportError:
                vector_db_version = "installed"
    except Exception as e:
        logger.debug(f"Failed to get vector DB version: {e}")

    # Get Ollama version
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/version")
            if resp.status_code == 200:
                ollama_version = resp.json().get("version")
    except Exception as e:
        logger.debug(f"Failed to get Ollama version: {e}")

    # Get knowledge base stats
    try:
        stats = await vector_service.get_extended_stats()
        total_chunks = stats.get("total_chunks", 0)
        storage_bytes = stats.get("collection_size_bytes")
        if storage_bytes:
            storage_size_mb = round(storage_bytes / (1024 * 1024), 2)
    except Exception as e:
        logger.warning(f"Failed to get KB stats: {e}")

    # Get document counts from database
    total_documents = db.query(Document).count()
    pending_count = db.query(Document).filter(Document.status == "pending").count()
    processing_count = db.query(Document).filter(Document.status == "processing").count()
    failed_count = db.query(Document).filter(Document.status == "failed").count()
    ready_count = db.query(Document).filter(Document.status == "ready").count()

    # Determine overall status
    if ollama_healthy and vector_healthy:
        overall_status = "healthy"
    elif ollama_healthy or vector_healthy:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    # Build response
    return DetailedHealthResponse(
        status=overall_status,
        version=settings.app_version,
        profile=settings.env_profile,
        api_server=ComponentHealth(
            name="FastAPI",
            status="healthy",
            version="0.115.x",
            details={"uptime_hours": round(uptime_seconds / 3600, 2)},
        ),
        ollama_llm=ComponentHealth(
            name="Ollama",
            status="healthy" if ollama_healthy else "unhealthy",
            version=ollama_version,
            details={
                "model": get_model_setting("chat_model", settings.chat_model),
                "url": settings.ollama_base_url,
            },
        ),
        vector_db=ComponentHealth(
            name=settings.vector_backend.capitalize(),
            status="healthy" if vector_healthy else "unhealthy",
            version=vector_db_version,
            details={
                "collection": settings.qdrant_collection,
                "chunks": total_chunks,
            },
        ),
        rag_pipeline=RAGPipelineStatus(
            embedding_model=get_model_setting("embedding_model", settings.embedding_model),
            chat_model=get_model_setting("chat_model", settings.chat_model),
            chunker=settings.chunker_backend,
            stages=["query", "embed", "search", "rerank", "context", "llm", "response"],
            all_stages_healthy=ollama_healthy and vector_healthy,
        ),
        knowledge_base=KnowledgeBaseSummary(
            total_documents=total_documents,
            total_chunks=total_chunks,
            storage_size_mb=storage_size_mb,
        ),
        processing_queue=ProcessingQueueStatus(
            pending=pending_count,
            processing=processing_count,
            failed=failed_count,
            ready=ready_count,
        ),
        uptime_seconds=uptime_seconds,
        last_checked=datetime.now(UTC),
    )


# =============================================================================
# RAG Tuning Settings - Retrieval and LLM Response Configuration
# =============================================================================

# Default values for RAG tuning settings
RETRIEVAL_DEFAULTS = {
    "retrieval_top_k": 5,  # Number of chunks to retrieve
    "retrieval_min_score": 0.3,  # Minimum similarity score (0.0-1.0)
    "retrieval_enable_expansion": True,  # Enable query expansion
}

LLM_DEFAULTS = {
    "llm_temperature": 0.1,  # LLM temperature (0.0-1.0)
    "llm_max_response_tokens": 2048,  # Max tokens in response
    "llm_confidence_threshold": 40,  # Confidence threshold (0-100)
}


@router.get("/settings/retrieval", response_model=RetrievalSettingsResponse)
async def get_retrieval_settings(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get current retrieval settings for RAG tuning.

    Returns settings with fallback to environment variables, then defaults.
    Admin only.
    """
    service = SettingsService(db)

    return RetrievalSettingsResponse(
        retrieval_top_k=service.get_with_env_fallback(
            "retrieval_top_k",
            "RAG_TOP_K",
            RETRIEVAL_DEFAULTS["retrieval_top_k"],
        ),
        retrieval_min_score=service.get_with_env_fallback(
            "retrieval_min_score",
            "RAG_MIN_SCORE",
            RETRIEVAL_DEFAULTS["retrieval_min_score"],
        ),
        retrieval_enable_expansion=service.get_with_env_fallback(
            "retrieval_enable_expansion",
            "RAG_ENABLE_EXPANSION",
            RETRIEVAL_DEFAULTS["retrieval_enable_expansion"],
        ),
    )


@router.put("/settings/retrieval", response_model=RetrievalSettingsResponse)
async def update_retrieval_settings(
    request: RetrievalSettingsRequest,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Update retrieval settings for RAG tuning.

    Only updates fields that are provided (not None).
    All changes are recorded in the audit log.
    Admin only.
    """
    service = SettingsService(db)

    # Validate and update only provided fields
    if request.retrieval_top_k is not None:
        if not (3 <= request.retrieval_top_k <= 20):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="retrieval_top_k must be between 3 and 20",
            )
        service.set_with_audit(
            "retrieval_top_k",
            request.retrieval_top_k,
            changed_by=current_user.id,
            reason="Updated via admin settings",
        )

    if request.retrieval_min_score is not None:
        if not (0.1 <= request.retrieval_min_score <= 0.9):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="retrieval_min_score must be between 0.1 and 0.9",
            )
        service.set_with_audit(
            "retrieval_min_score",
            request.retrieval_min_score,
            changed_by=current_user.id,
            reason="Updated via admin settings",
        )

    if request.retrieval_enable_expansion is not None:
        service.set_with_audit(
            "retrieval_enable_expansion",
            request.retrieval_enable_expansion,
            changed_by=current_user.id,
            reason="Updated via admin settings",
        )

    logger.info(
        f"Admin {current_user.email} updated retrieval settings: "
        f"{request.model_dump(exclude_none=True)}"
    )

    # Return current state
    return RetrievalSettingsResponse(
        retrieval_top_k=service.get_with_env_fallback(
            "retrieval_top_k",
            "RAG_TOP_K",
            RETRIEVAL_DEFAULTS["retrieval_top_k"],
        ),
        retrieval_min_score=service.get_with_env_fallback(
            "retrieval_min_score",
            "RAG_MIN_SCORE",
            RETRIEVAL_DEFAULTS["retrieval_min_score"],
        ),
        retrieval_enable_expansion=service.get_with_env_fallback(
            "retrieval_enable_expansion",
            "RAG_ENABLE_EXPANSION",
            RETRIEVAL_DEFAULTS["retrieval_enable_expansion"],
        ),
    )


@router.get("/settings/llm", response_model=LLMSettingsResponse)
async def get_llm_settings(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get current LLM response settings for RAG tuning.

    Returns settings with fallback to environment variables, then defaults.
    Admin only.
    """
    service = SettingsService(db)

    return LLMSettingsResponse(
        llm_temperature=service.get_with_env_fallback(
            "llm_temperature",
            "RAG_TEMPERATURE",
            LLM_DEFAULTS["llm_temperature"],
        ),
        llm_max_response_tokens=service.get_with_env_fallback(
            "llm_max_response_tokens",
            "RAG_MAX_RESPONSE_TOKENS",
            LLM_DEFAULTS["llm_max_response_tokens"],
        ),
        llm_confidence_threshold=service.get_with_env_fallback(
            "llm_confidence_threshold",
            "RAG_CONFIDENCE_THRESHOLD",
            LLM_DEFAULTS["llm_confidence_threshold"],
        ),
    )


@router.put("/settings/llm", response_model=LLMSettingsResponse)
async def update_llm_settings(
    request: LLMSettingsRequest,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Update LLM response settings for RAG tuning.

    Only updates fields that are provided (not None).
    All changes are recorded in the audit log.
    Admin only.
    """
    service = SettingsService(db)

    # Validate and update only provided fields
    if request.llm_temperature is not None:
        if not (0.0 <= request.llm_temperature <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="llm_temperature must be between 0.0 and 1.0",
            )
        service.set_with_audit(
            "llm_temperature",
            request.llm_temperature,
            changed_by=current_user.id,
            reason="Updated via admin settings",
        )

    if request.llm_max_response_tokens is not None:
        if not (256 <= request.llm_max_response_tokens <= 4096):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="llm_max_response_tokens must be between 256 and 4096",
            )
        service.set_with_audit(
            "llm_max_response_tokens",
            request.llm_max_response_tokens,
            changed_by=current_user.id,
            reason="Updated via admin settings",
        )

    if request.llm_confidence_threshold is not None:
        if not (0 <= request.llm_confidence_threshold <= 100):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="llm_confidence_threshold must be between 0 and 100",
            )
        service.set_with_audit(
            "llm_confidence_threshold",
            request.llm_confidence_threshold,
            changed_by=current_user.id,
            reason="Updated via admin settings",
        )

    logger.info(
        f"Admin {current_user.email} updated LLM settings: {request.model_dump(exclude_none=True)}"
    )

    # Return current state
    return LLMSettingsResponse(
        llm_temperature=service.get_with_env_fallback(
            "llm_temperature",
            "RAG_TEMPERATURE",
            LLM_DEFAULTS["llm_temperature"],
        ),
        llm_max_response_tokens=service.get_with_env_fallback(
            "llm_max_response_tokens",
            "RAG_MAX_RESPONSE_TOKENS",
            LLM_DEFAULTS["llm_max_response_tokens"],
        ),
        llm_confidence_threshold=service.get_with_env_fallback(
            "llm_confidence_threshold",
            "RAG_CONFIDENCE_THRESHOLD",
            LLM_DEFAULTS["llm_confidence_threshold"],
        ),
    )


@router.get("/settings/audit", response_model=SettingsAuditResponse)
async def get_settings_audit(
    key: str | None = Query(None, description="Filter by setting key"),
    limit: int = Query(50, ge=1, le=100, description="Maximum entries to return"),
    offset: int = Query(0, ge=0, description="Number of entries to skip"),
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get settings change audit history.

    Returns a paginated list of all settings changes, with optional filtering by key.
    Admin only.
    """
    service = SettingsService(db)

    # Get audit entries
    entries = service.get_audit_history(key=key, limit=limit, offset=offset)

    # Get total count for pagination
    from ai_ready_rag.db.models import SettingsAudit

    query = db.query(SettingsAudit)
    if key:
        query = query.filter(SettingsAudit.setting_key == key)
    total = query.count()

    return SettingsAuditResponse(
        entries=[
            SettingsAuditEntry(
                id=e.id,
                setting_key=e.setting_key,
                old_value=e.old_value,
                new_value=e.new_value,
                changed_by=e.changed_by,
                changed_at=e.changed_at,
                change_reason=e.change_reason,
            )
            for e in entries
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


# =============================================================================
# Advanced Settings & Reindex Endpoints
# =============================================================================

# Default values for advanced settings
ADVANCED_DEFAULTS = {
    "embedding_model": "nomic-embed-text",
    "chunk_size": 200,
    "chunk_overlap": 40,
    "hnsw_ef_construct": 100,
    "hnsw_m": 16,
    "vector_backend": "qdrant",
}


def _reindex_job_to_response(job) -> ReindexJobResponse:
    """Convert a ReindexJob to ReindexJobResponse."""
    import json

    # Calculate progress percentage
    if job.total_documents > 0:
        progress = (job.processed_documents / job.total_documents) * 100
    else:
        progress = 0.0

    # Parse settings_changed JSON if present
    settings_changed = None
    if job.settings_changed:
        try:
            settings_changed = json.loads(job.settings_changed)
        except (json.JSONDecodeError, TypeError):
            settings_changed = None

    return ReindexJobResponse(
        id=job.id,
        status=job.status,
        total_documents=job.total_documents,
        processed_documents=job.processed_documents,
        failed_documents=job.failed_documents,
        progress_percent=round(progress, 1),
        current_document_id=job.current_document_id,
        error_message=job.error_message,
        started_at=job.started_at,
        completed_at=job.completed_at,
        created_at=job.created_at,
        settings_changed=settings_changed,
        last_error=job.last_error,
        retry_count=job.retry_count or 0,
        max_retries=job.max_retries or 3,
        paused_at=job.paused_at,
        paused_reason=job.paused_reason,
        auto_skip_failures=job.auto_skip_failures or False,
    )


@router.get("/settings/advanced", response_model=AdvancedSettingsResponse)
async def get_advanced_settings(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get advanced RAG settings that require reindex when changed.

    These settings affect document processing and vector storage.
    Changing them requires a full knowledge base reindex.

    Admin only.
    """
    service = SettingsService(db)
    settings = get_settings()

    return AdvancedSettingsResponse(
        embedding_model=service.get("embedding_model") or settings.embedding_model,
        chunk_size=service.get("chunk_size") or ADVANCED_DEFAULTS["chunk_size"],
        chunk_overlap=service.get("chunk_overlap") or ADVANCED_DEFAULTS["chunk_overlap"],
        hnsw_ef_construct=service.get("hnsw_ef_construct")
        or ADVANCED_DEFAULTS["hnsw_ef_construct"],
        hnsw_m=service.get("hnsw_m") or ADVANCED_DEFAULTS["hnsw_m"],
        vector_backend=service.get("vector_backend") or ADVANCED_DEFAULTS["vector_backend"],
    )


@router.put("/settings/advanced", response_model=AdvancedSettingsResponse)
async def update_advanced_settings(
    request: AdvancedSettingsRequest,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Update advanced RAG settings.

    WARNING: Changing these settings invalidates all existing vectors.
    A full reindex is required after changing these settings.

    Set confirm_reindex=true to acknowledge this.

    Admin only.
    """
    if not request.confirm_reindex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Changing advanced settings requires a knowledge base reindex. "
            "Set confirm_reindex=true to acknowledge this.",
        )

    service = SettingsService(db)
    settings = get_settings()

    # Validate chunk_overlap < chunk_size
    # Get effective values (requested or current or default)
    effective_chunk_size = (
        request.chunk_size
        if request.chunk_size is not None
        else (service.get("chunk_size") or ADVANCED_DEFAULTS["chunk_size"])
    )
    effective_chunk_overlap = (
        request.chunk_overlap
        if request.chunk_overlap is not None
        else (service.get("chunk_overlap") or ADVANCED_DEFAULTS["chunk_overlap"])
    )
    if effective_chunk_overlap >= effective_chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"chunk_overlap ({effective_chunk_overlap}) must be less than "
            f"chunk_size ({effective_chunk_size})",
        )

    # Update only provided settings with audit trail
    if request.embedding_model is not None:
        service.set_with_audit(
            "embedding_model",
            request.embedding_model,
            changed_by=current_user.id,
            reason="Advanced settings update",
        )
    if request.chunk_size is not None:
        service.set_with_audit(
            "chunk_size",
            request.chunk_size,
            changed_by=current_user.id,
            reason="Advanced settings update",
        )
    if request.chunk_overlap is not None:
        service.set_with_audit(
            "chunk_overlap",
            request.chunk_overlap,
            changed_by=current_user.id,
            reason="Advanced settings update",
        )
    if request.hnsw_ef_construct is not None:
        service.set_with_audit(
            "hnsw_ef_construct",
            request.hnsw_ef_construct,
            changed_by=current_user.id,
            reason="Advanced settings update",
        )
    if request.hnsw_m is not None:
        service.set_with_audit(
            "hnsw_m",
            request.hnsw_m,
            changed_by=current_user.id,
            reason="Advanced settings update",
        )
    if request.vector_backend is not None:
        service.set_with_audit(
            "vector_backend",
            request.vector_backend,
            changed_by=current_user.id,
            reason="Advanced settings update",
        )

    logger.warning(f"Admin {current_user.email} updated advanced settings. Reindex required.")

    return AdvancedSettingsResponse(
        embedding_model=service.get("embedding_model") or settings.embedding_model,
        chunk_size=service.get("chunk_size") or ADVANCED_DEFAULTS["chunk_size"],
        chunk_overlap=service.get("chunk_overlap") or ADVANCED_DEFAULTS["chunk_overlap"],
        hnsw_ef_construct=service.get("hnsw_ef_construct")
        or ADVANCED_DEFAULTS["hnsw_ef_construct"],
        hnsw_m=service.get("hnsw_m") or ADVANCED_DEFAULTS["hnsw_m"],
        vector_backend=service.get("vector_backend") or ADVANCED_DEFAULTS["vector_backend"],
    )


# =============================================================================
# Security Settings
# =============================================================================

SECURITY_DEFAULTS = {
    "jwt_expiration_hours": 24,
    "password_min_length": 12,
    "bcrypt_rounds": 12,
}


@router.get("/settings/security", response_model=SecuritySettingsResponse)
async def get_security_settings(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get current security settings.

    Returns settings with fallback to config.py defaults.
    System admin only.
    """
    service = SettingsService(db)
    settings = get_settings()

    return SecuritySettingsResponse(
        jwt_expiration_hours=service.get_with_env_fallback(
            "jwt_expiration_hours",
            "JWT_EXPIRATION_HOURS",
            settings.jwt_expiration_hours,
        ),
        password_min_length=service.get_with_env_fallback(
            "password_min_length",
            "PASSWORD_MIN_LENGTH",
            settings.password_min_length,
        ),
        bcrypt_rounds=service.get_with_env_fallback(
            "bcrypt_rounds",
            "BCRYPT_ROUNDS",
            settings.bcrypt_rounds,
        ),
    )


@router.put("/settings/security", response_model=SecuritySettingsResponse)
async def update_security_settings(
    request: SecuritySettingsRequest,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Update security settings.

    Only updates fields that are provided (not None).
    All changes are recorded in the audit log.
    System admin only.
    """
    service = SettingsService(db)
    settings = get_settings()

    if request.jwt_expiration_hours is not None:
        if not (1 <= request.jwt_expiration_hours <= 720):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="jwt_expiration_hours must be between 1 and 720 (30 days)",
            )
        service.set_with_audit(
            "jwt_expiration_hours",
            request.jwt_expiration_hours,
            changed_by=current_user.id,
            reason="Updated via admin settings",
        )

    if request.password_min_length is not None:
        if not (8 <= request.password_min_length <= 128):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="password_min_length must be between 8 and 128",
            )
        service.set_with_audit(
            "password_min_length",
            request.password_min_length,
            changed_by=current_user.id,
            reason="Updated via admin settings",
        )

    if request.bcrypt_rounds is not None:
        if not (4 <= request.bcrypt_rounds <= 31):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="bcrypt_rounds must be between 4 and 31",
            )
        service.set_with_audit(
            "bcrypt_rounds",
            request.bcrypt_rounds,
            changed_by=current_user.id,
            reason="Updated via admin settings",
        )

    logger.info(
        f"Admin {current_user.email} updated security settings: "
        f"{request.model_dump(exclude_none=True)}"
    )

    return SecuritySettingsResponse(
        jwt_expiration_hours=service.get_with_env_fallback(
            "jwt_expiration_hours",
            "JWT_EXPIRATION_HOURS",
            settings.jwt_expiration_hours,
        ),
        password_min_length=service.get_with_env_fallback(
            "password_min_length",
            "PASSWORD_MIN_LENGTH",
            settings.password_min_length,
        ),
        bcrypt_rounds=service.get_with_env_fallback(
            "bcrypt_rounds",
            "BCRYPT_ROUNDS",
            settings.bcrypt_rounds,
        ),
    )


# =============================================================================
# Feature Flags
# =============================================================================

FEATURE_FLAG_DEFAULTS = {
    "enable_rag": True,
    "skip_setup_wizard": False,
}


@router.get("/settings/feature-flags", response_model=FeatureFlagsResponse)
async def get_feature_flags(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get current feature flag settings.

    Returns settings with fallback to config.py defaults.
    System admin only.
    """
    service = SettingsService(db)
    settings = get_settings()

    return FeatureFlagsResponse(
        enable_rag=service.get_with_env_fallback(
            "enable_rag",
            "ENABLE_RAG",
            settings.enable_rag,
        ),
        skip_setup_wizard=service.get_with_env_fallback(
            "skip_setup_wizard",
            "SKIP_SETUP_WIZARD",
            settings.skip_setup_wizard,
        ),
    )


@router.put("/settings/feature-flags", response_model=FeatureFlagsResponse)
async def update_feature_flags(
    request: FeatureFlagsRequest,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Update feature flag settings.

    Only updates fields that are provided (not None).
    All changes are recorded in the audit log.
    System admin only.
    """
    service = SettingsService(db)
    settings = get_settings()

    if request.enable_rag is not None:
        service.set_with_audit(
            "enable_rag",
            request.enable_rag,
            changed_by=current_user.id,
            reason="Updated via admin settings",
        )

    if request.skip_setup_wizard is not None:
        service.set_with_audit(
            "skip_setup_wizard",
            request.skip_setup_wizard,
            changed_by=current_user.id,
            reason="Updated via admin settings",
        )

    logger.info(
        f"Admin {current_user.email} updated feature flags: {request.model_dump(exclude_none=True)}"
    )

    return FeatureFlagsResponse(
        enable_rag=service.get_with_env_fallback(
            "enable_rag",
            "ENABLE_RAG",
            settings.enable_rag,
        ),
        skip_setup_wizard=service.get_with_env_fallback(
            "skip_setup_wizard",
            "SKIP_SETUP_WIZARD",
            settings.skip_setup_wizard,
        ),
    )


@router.get("/reindex/estimate", response_model=ReindexEstimate)
async def get_reindex_estimate(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get time estimate for reindex operation.

    Based on historical document processing times.

    Admin only.
    """
    from ai_ready_rag.services.reindex_service import ReindexService

    service = ReindexService(db)
    total_docs = db.query(Document).filter(Document.status == "ready").count()
    estimate = service.estimate_time(total_docs)

    return ReindexEstimate(
        total_documents=estimate["total_documents"],
        avg_processing_time_ms=estimate["avg_processing_time_ms"],
        estimated_total_seconds=estimate["estimated_total_seconds"],
        estimated_time_str=estimate["estimated_time_str"],
    )


@router.post(
    "/reindex/start", response_model=ReindexJobResponse, status_code=status.HTTP_202_ACCEPTED
)
async def start_reindex(
    request: StartReindexRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Start a background reindex operation.

    Creates a new reindex job that will rebuild the knowledge base
    with current settings. Returns 202 Accepted immediately.

    Poll GET /api/admin/reindex/status to monitor progress.

    Admin only.
    """
    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must confirm=true to start reindex operation.",
        )

    from ai_ready_rag.services.reindex_service import ReindexService
    from ai_ready_rag.services.reindex_worker import run_reindex_job

    service = ReindexService(db)

    # Check for existing active job
    active_job = service.get_active_job()
    if active_job:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Reindex job {active_job.id} already in progress. "
            "Abort it first or wait for completion.",
        )

    # Create new job
    try:
        job = service.create_job(triggered_by=current_user.id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e

    logger.info(f"Admin {current_user.email} started reindex job {job.id}")

    # Enqueue via ARQ if Redis available, otherwise fall back to BackgroundTasks
    redis = await get_redis_pool()
    if redis:
        try:
            await redis.enqueue_job("reindex_knowledge_base", job.id)
            logger.info(f"Reindex job {job.id} enqueued via ARQ")
        except Exception as e:
            logger.warning(f"ARQ enqueue failed for reindex, falling back: {e}")
            background_tasks.add_task(run_reindex_job, job.id)
    else:
        background_tasks.add_task(run_reindex_job, job.id)

    return _reindex_job_to_response(job)


@router.get("/reindex/status", response_model=ReindexJobResponse | None)
async def get_reindex_status(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get current reindex job status.

    Returns the active reindex job if one exists, otherwise None.

    Admin only.
    """
    from ai_ready_rag.services.reindex_service import ReindexService

    service = ReindexService(db)
    job = service.get_active_job()

    if not job:
        return None

    return _reindex_job_to_response(job)


@router.post("/reindex/abort", response_model=ReindexJobResponse)
async def abort_reindex(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Abort the current reindex operation.

    Stops the reindex and cleans up any temporary resources.
    The existing collection remains unchanged.

    Admin only.
    """
    from ai_ready_rag.services.reindex_service import ReindexService

    service = ReindexService(db)
    job = service.get_active_job()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active reindex job to abort.",
        )

    job = service.abort_job(job.id)
    logger.warning(f"Admin {current_user.email} aborted reindex job {job.id}")

    return _reindex_job_to_response(job)


@router.get("/reindex/history", response_model=list[ReindexJobResponse])
async def get_reindex_history(
    limit: int = Query(10, ge=1, le=50, description="Maximum jobs to return"),
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get reindex job history.

    Returns recent reindex jobs ordered by creation date descending.

    Admin only.
    """
    from ai_ready_rag.services.reindex_service import ReindexService

    service = ReindexService(db)
    jobs = service.get_job_history(limit=limit)

    return [_reindex_job_to_response(job) for job in jobs]


# =============================================================================
# Phase 3: Reindex Failure Handling
# =============================================================================


@router.post("/reindex/pause", response_model=ReindexJobResponse)
async def pause_reindex(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Pause the current reindex operation.

    Pauses a running reindex job. Use resume to continue.

    Admin only.
    """
    from ai_ready_rag.services.reindex_service import ReindexService

    service = ReindexService(db)
    job = service.get_active_job()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active reindex job to pause.",
        )

    if job.status != "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot pause job in status '{job.status}'. Only running jobs can be paused.",
        )

    job = service.pause_job(job.id, reason="user_request")
    logger.info(f"Admin {current_user.email} paused reindex job {job.id}")

    return _reindex_job_to_response(job)


@router.post("/reindex/resume", response_model=ReindexJobResponse)
async def resume_reindex(
    request: ResumeReindexRequest,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Resume a paused reindex operation.

    Actions:
    - skip: Skip the failed document and continue
    - retry: Retry the failed document (up to max_retries)
    - skip_all: Enable auto-skip mode for all future failures

    Admin only.
    """
    from ai_ready_rag.services.reindex_service import ReindexService, ResumeAction

    service = ReindexService(db)
    job = service.get_active_job()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active reindex job to resume.",
        )

    if job.status != "paused":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot resume job in status '{job.status}'. Only paused jobs can be resumed.",
        )

    # Convert string action to enum
    action = ResumeAction(request.action)

    # Check retry limit before allowing retry
    if action == ResumeAction.RETRY and job.retry_count >= job.max_retries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Max retries ({job.max_retries}) reached. Use 'skip' to continue.",
        )

    job = service.resume_job(job.id, action)
    logger.info(f"Admin {current_user.email} resumed reindex job {job.id} with action '{action}'")

    return _reindex_job_to_response(job)


@router.get("/reindex/failures", response_model=ReindexFailuresResponse)
async def get_reindex_failures(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get details about failed documents in the current reindex.

    Returns information about all documents that have failed during
    the current or most recent reindex job.

    Admin only.
    """
    from ai_ready_rag.services.reindex_service import ReindexService

    service = ReindexService(db)

    # Get active job or most recent completed job with failures
    job = service.get_active_job()
    if not job:
        # Get most recent job
        history = service.get_job_history(limit=1)
        job = history[0] if history else None

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No reindex job found.",
        )

    failures = service.get_failures(job.id)

    return ReindexFailuresResponse(
        job_id=job.id,
        failures=[
            ReindexFailureInfo(
                document_id=f["document_id"],
                filename=f["filename"],
                status=f["status"],
                error_message=f.get("error_message"),
            )
            for f in failures
        ],
        total_failures=len(failures),
    )


@router.post("/reindex/retry/{document_id}", response_model=ReindexJobResponse)
async def retry_reindex_document(
    document_id: str,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Retry a specific failed document.

    Removes the document from the failed list and marks it for reprocessing.

    Admin only.
    """
    import json

    from ai_ready_rag.services.reindex_service import ReindexService

    service = ReindexService(db)
    job = service.get_active_job()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active reindex job.",
        )

    # Check if document is in failed list before attempting retry
    failed_ids = []
    if job.failed_document_ids:
        try:
            failed_ids = json.loads(job.failed_document_ids)
        except json.JSONDecodeError:
            pass

    if document_id not in failed_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document {document_id} not in failed list for this job.",
        )

    job = service.retry_document(job.id, document_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )

    logger.info(
        f"Admin {current_user.email} marked document {document_id} for retry in job {job.id}"
    )

    return _reindex_job_to_response(job)


# =============================================================================
# Cache Warming Endpoints
# =============================================================================


def _batch_to_response(batch: WarmingBatch, db: Session) -> WarmingQueueJobResponse:
    """Convert WarmingBatch + aggregated query counts to response."""
    counts = (
        db.query(
            func.count(case((WarmingQuery.status == "completed", 1))).label("completed"),
            func.count(case((WarmingQuery.status == "failed", 1))).label("failed"),
            func.count(case((WarmingQuery.status == "pending", 1))).label("pending"),
        )
        .filter(WarmingQuery.batch_id == batch.id)
        .first()
    )
    return WarmingQueueJobResponse(
        id=batch.id,
        source_type=batch.source_type,
        original_filename=batch.original_filename,
        total_queries=batch.total_queries,
        completed_queries=counts.completed if counts else 0,
        failed_queries=counts.failed if counts else 0,
        pending_queries=counts.pending if counts else 0,
        status=batch.status,
        is_paused=batch.is_paused,
        is_cancel_requested=batch.is_cancel_requested,
        created_at=batch.created_at,
        started_at=batch.started_at,
        completed_at=batch.completed_at,
        submitted_by=batch.submitted_by,
        error_message=batch.error_message,
        worker_id=batch.worker_id,
    )


def _strip_numbering(text: str) -> str:
    """Strip leading numbering from a question (e.g., '1. Question' -> 'Question')."""
    return re.sub(r"^\d+[\.\)\-\s]+", "", text.strip())


@router.get("/cache/top-queries", response_model=TopQueryResponse)
async def get_top_queries(
    limit: int = Query(20, ge=1, le=100, description="Maximum queries to return"),
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get most frequently accessed queries for cache warming.

    Returns queries sorted by access count (descending), useful for
    manual cache warming decisions.

    Admin only.
    """
    from ai_ready_rag.services.cache_service import CacheService

    cache_service = CacheService(db)
    top_queries = cache_service.get_top_queries(limit)

    return TopQueryResponse(
        queries=[
            TopQueryItem(
                query_text=q["query_text"],
                access_count=q["access_count"],
                last_accessed=q["last_accessed"],
            )
            for q in top_queries
        ]
    )


# =============================================================================
# Cache Admin Endpoints (Phase 4)
# =============================================================================


@router.get("/cache/settings", response_model=CacheSettingsResponse)
async def get_cache_settings(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get current cache settings.

    Returns all cache configuration options with current values.
    Admin only.
    """
    from ai_ready_rag.services.settings_service import (
        CACHE_SETTINGS_DEFAULTS,
        SettingsService,
    )

    service = SettingsService(db)

    return CacheSettingsResponse(
        cache_enabled=_get_setting_value(
            service, "cache_enabled", CACHE_SETTINGS_DEFAULTS["cache_enabled"]
        ),
        cache_ttl_hours=_get_setting_value(
            service, "cache_ttl_hours", CACHE_SETTINGS_DEFAULTS["cache_ttl_hours"]
        ),
        cache_max_entries=_get_setting_value(
            service, "cache_max_entries", CACHE_SETTINGS_DEFAULTS["cache_max_entries"]
        ),
        cache_semantic_threshold=_get_setting_value(
            service, "cache_semantic_threshold", CACHE_SETTINGS_DEFAULTS["cache_semantic_threshold"]
        ),
        cache_min_confidence=_get_setting_value(
            service, "cache_min_confidence", CACHE_SETTINGS_DEFAULTS["cache_min_confidence"]
        ),
        cache_auto_warm_enabled=_get_setting_value(
            service, "cache_auto_warm_enabled", CACHE_SETTINGS_DEFAULTS["cache_auto_warm_enabled"]
        ),
        cache_auto_warm_count=_get_setting_value(
            service, "cache_auto_warm_count", CACHE_SETTINGS_DEFAULTS["cache_auto_warm_count"]
        ),
    )


@router.put("/cache/settings", response_model=CacheSettingsResponse)
async def update_cache_settings(
    request: CacheSettingsRequest,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Update cache settings.

    Only updates fields that are provided (not None).
    Settings persist to admin_settings table.
    Admin only.
    """
    from ai_ready_rag.services.settings_service import (
        CACHE_SETTINGS_DEFAULTS,
        SettingsService,
    )

    service = SettingsService(db)

    # Validate and update only provided fields
    if request.cache_enabled is not None:
        service.set("cache_enabled", request.cache_enabled, updated_by=current_user.id)

    if request.cache_ttl_hours is not None:
        if not (1 <= request.cache_ttl_hours <= 168):  # 1 hour to 7 days
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="cache_ttl_hours must be between 1 and 168",
            )
        service.set("cache_ttl_hours", request.cache_ttl_hours, updated_by=current_user.id)

    if request.cache_max_entries is not None:
        if not (100 <= request.cache_max_entries <= 10000):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="cache_max_entries must be between 100 and 10000",
            )
        service.set("cache_max_entries", request.cache_max_entries, updated_by=current_user.id)

    if request.cache_semantic_threshold is not None:
        if not (0.85 <= request.cache_semantic_threshold <= 0.99):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="cache_semantic_threshold must be between 0.85 and 0.99",
            )
        service.set(
            "cache_semantic_threshold", request.cache_semantic_threshold, updated_by=current_user.id
        )

    if request.cache_min_confidence is not None:
        if not (0 <= request.cache_min_confidence <= 100):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="cache_min_confidence must be between 0 and 100",
            )
        service.set(
            "cache_min_confidence", request.cache_min_confidence, updated_by=current_user.id
        )

    if request.cache_auto_warm_enabled is not None:
        service.set(
            "cache_auto_warm_enabled", request.cache_auto_warm_enabled, updated_by=current_user.id
        )

    if request.cache_auto_warm_count is not None:
        if not (5 <= request.cache_auto_warm_count <= 50):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="cache_auto_warm_count must be between 5 and 50",
            )
        service.set(
            "cache_auto_warm_count", request.cache_auto_warm_count, updated_by=current_user.id
        )

    logger.info(
        f"Admin {current_user.email} updated cache settings: {request.model_dump(exclude_none=True)}"
    )

    # Return current state (same as GET)
    return CacheSettingsResponse(
        cache_enabled=_get_setting_value(
            service, "cache_enabled", CACHE_SETTINGS_DEFAULTS["cache_enabled"]
        ),
        cache_ttl_hours=_get_setting_value(
            service, "cache_ttl_hours", CACHE_SETTINGS_DEFAULTS["cache_ttl_hours"]
        ),
        cache_max_entries=_get_setting_value(
            service, "cache_max_entries", CACHE_SETTINGS_DEFAULTS["cache_max_entries"]
        ),
        cache_semantic_threshold=_get_setting_value(
            service, "cache_semantic_threshold", CACHE_SETTINGS_DEFAULTS["cache_semantic_threshold"]
        ),
        cache_min_confidence=_get_setting_value(
            service, "cache_min_confidence", CACHE_SETTINGS_DEFAULTS["cache_min_confidence"]
        ),
        cache_auto_warm_enabled=_get_setting_value(
            service, "cache_auto_warm_enabled", CACHE_SETTINGS_DEFAULTS["cache_auto_warm_enabled"]
        ),
        cache_auto_warm_count=_get_setting_value(
            service, "cache_auto_warm_count", CACHE_SETTINGS_DEFAULTS["cache_auto_warm_count"]
        ),
    )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get cache statistics.

    Returns hit/miss counts, entry counts, and storage information.
    Admin only.
    """
    from sqlalchemy import func

    from ai_ready_rag.db.models import ResponseCache
    from ai_ready_rag.services.cache_service import CacheService

    cache_service = CacheService(db)
    stats = cache_service.get_stats()

    # Get oldest and newest entries
    oldest = db.query(func.min(ResponseCache.created_at)).scalar()
    newest = db.query(func.max(ResponseCache.created_at)).scalar()

    # Estimate storage size (rough calculation)
    # Each entry: ~10KB average (query_text + answer + embedding + sources)
    storage_bytes = stats.sqlite_entries * 10240 if stats.sqlite_entries else None

    return CacheStatsResponse(
        enabled=stats.enabled,
        total_entries=stats.total_entries,
        memory_entries=stats.memory_entries,
        sqlite_entries=stats.sqlite_entries,
        hit_count=stats.hit_count,
        miss_count=stats.miss_count,
        hit_rate=stats.hit_rate,
        storage_size_bytes=storage_bytes,
        oldest_entry=oldest,
        newest_entry=newest,
    )


@router.post("/cache/clear", response_model=CacheClearResponse)
async def clear_cache(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Clear all cache entries.

    Removes all entries from both memory and SQLite cache.
    Admin only.
    """
    from ai_ready_rag.services.cache_service import CacheService

    cache_service = CacheService(db)
    cleared = cache_service.invalidate_all(reason=f"admin_clear_by_{current_user.email}")

    logger.warning(f"Admin {current_user.email} cleared cache: {cleared} entries removed")

    return CacheClearResponse(
        cleared_entries=cleared,
        message="Cache cleared successfully",
    )


@router.post("/cache/seed", response_model=CacheSeedResponse, status_code=status.HTTP_201_CREATED)
async def seed_cache(
    request: CacheSeedRequest,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Seed cache with a curated question-answer pair.

    Allows system admins to populate the cache with compliance-approved
    responses. The source_reference is required for audit trail.

    The response is stored with model_used="admin_seeded" to distinguish
    from LLM-generated responses.

    System admin only.
    """
    from ai_ready_rag.services.cache_service import CacheService

    settings = get_settings()

    # Sanitize HTML in the answer
    sanitized_answer = sanitize_html(request.answer)

    # Generate query embedding
    try:
        vector_service = get_vector_service(settings)
        await vector_service.initialize()
        embedding = await vector_service.embed(request.query)
    except Exception as e:
        logger.error(f"Failed to generate embedding for cache seed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {e}",
        ) from e

    # Seed the cache
    cache_service = CacheService(db)
    query_hash = await cache_service.seed_entry(
        query=request.query,
        embedding=embedding,
        answer=sanitized_answer,
        source_reference=request.source_reference,
        confidence=request.confidence,
    )

    logger.info(f"Admin {current_user.email} seeded cache for query: {request.query[:50]}...")

    return CacheSeedResponse(
        query_hash=query_hash,
        message="Cache entry seeded successfully",
    )


# =============================================================================
# DB-based Warming Queue Endpoints (Issue #121 - New URL Pattern)
# =============================================================================


@router.get("/warming/queue", response_model=WarmingQueueListResponse)
async def list_warming_queue(
    status_filter: str | None = Query(None, alias="status", description="Filter by batch status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of batches to return"),
    offset: int = Query(0, ge=0, description="Number of batches to skip"),
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """List all warming batches from database.

    Supports filtering by status and pagination.

    Admin only.
    """
    query = db.query(WarmingBatch)

    if status_filter:
        query = query.filter(WarmingBatch.status == status_filter)

    query = query.order_by(WarmingBatch.created_at.desc())

    total_count = query.count()
    batches = query.offset(offset).limit(limit).all()

    return WarmingQueueListResponse(
        jobs=[_batch_to_response(b, db) for b in batches],
        total_count=total_count,
    )


@router.get("/warming/queue/completed", response_model=WarmingQueueListResponse)
async def list_completed_warming_jobs(
    date_filter: str | None = Query(
        None, description="Filter by date (YYYY-MM-DD), defaults to today"
    ),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of batches to return"),
    offset: int = Query(0, ge=0, description="Number of batches to skip"),
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """List completed warming batches from database.

    Includes both 'completed' and 'completed_with_errors' statuses.
    Defaults to showing batches completed today.

    Admin only.
    """
    from datetime import date

    query = db.query(WarmingBatch).filter(
        WarmingBatch.status.in_(["completed", "completed_with_errors"])
    )

    if date_filter:
        try:
            filter_date = datetime.strptime(date_filter, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD.",
            ) from None
    else:
        filter_date = date.today()

    query = query.filter(
        WarmingBatch.completed_at >= datetime.combine(filter_date, datetime.min.time()),
        WarmingBatch.completed_at < datetime.combine(filter_date, datetime.max.time()),
    )

    query = query.order_by(WarmingBatch.completed_at.desc())

    total_count = query.count()
    batches = query.offset(offset).limit(limit).all()

    return WarmingQueueListResponse(
        jobs=[_batch_to_response(b, db) for b in batches],
        total_count=total_count,
    )


@router.post(
    "/warming/queue/upload",
    response_model=WarmingQueueJobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_warming_file(
    file: UploadFile,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Upload a file containing queries to the warming queue.

    Parses queries in-memory and stores them as WarmingBatch + WarmingQuery rows.
    No file is saved to disk.

    Supports .txt and .csv files with one query per line.

    Admin only.
    """
    settings = get_settings()

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    ext = file.filename.lower().split(".")[-1]
    if ext not in ("txt", "csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: .{ext}. Only .txt and .csv files are supported.",
        )

    try:
        content = await file.read()
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be UTF-8 encoded text.",
        ) from None

    # Check file size
    if len(content) > settings.warming_max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum size of {settings.warming_max_file_size_mb} MB.",
        )

    # Parse queries (one per line, strip numbering, skip blanks/comments)
    queries = []
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("//"):
            cleaned = _strip_numbering(line)
            if cleaned:
                queries.append(cleaned)

    if not queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid questions found in file.",
        )

    if len(queries) > settings.warming_max_queries_per_batch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many queries ({len(queries)}). Maximum is {settings.warming_max_queries_per_batch}.",
        )

    # Create batch
    batch = WarmingBatch(
        source_type="upload",
        original_filename=file.filename,
        total_queries=len(queries),
        status="pending",
        submitted_by=current_user.id,
        created_at=datetime.now(UTC),
    )
    db.add(batch)
    db.flush()

    # Bulk create query rows
    query_rows = [
        WarmingQuery(
            batch_id=batch.id,
            query_text=q,
            status="pending",
            sort_order=i,
            submitted_by=current_user.id,
            created_at=datetime.now(UTC),
        )
        for i, q in enumerate(queries)
    ]
    db.add_all(query_rows)
    db.commit()
    db.refresh(batch)

    # Best-effort ARQ enqueue
    try:
        redis = await get_redis_pool()
        if redis:
            await redis.enqueue_job("process_warming_batch", batch.id)
        else:
            logger.warning(
                f"Redis unavailable, batch {batch.id} saved but no worker will auto-pick up."
            )
    except Exception as e:
        logger.warning(f"ARQ enqueue failed for batch {batch.id}: {e}")

    logger.info(
        f"Admin {current_user.email} uploaded warming file: {len(queries)} queries, batch_id={batch.id}"
    )

    return _batch_to_response(batch, db)


@router.post(
    "/warming/queue/manual",
    response_model=WarmingQueueJobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_manual_warming_queries(
    request: ManualWarmingRequest,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Add manual queries to the warming queue.

    Creates WarmingBatch + WarmingQuery rows in DB. No file is saved to disk.

    Admin only.
    """
    settings = get_settings()

    if not request.queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one query is required.",
        )

    # Clean and validate queries (skip blanks and comments)
    queries = []
    for q in request.queries:
        cleaned = q.strip()
        if cleaned and not cleaned.startswith("#") and not cleaned.startswith("//"):
            queries.append(cleaned)

    if not queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid queries provided.",
        )

    if len(queries) > settings.warming_max_queries_per_batch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many queries ({len(queries)}). Maximum is {settings.warming_max_queries_per_batch}.",
        )

    # Create batch
    batch = WarmingBatch(
        source_type="manual",
        total_queries=len(queries),
        status="pending",
        submitted_by=current_user.id,
        created_at=datetime.now(UTC),
    )
    db.add(batch)
    db.flush()

    # Bulk create query rows
    query_rows = [
        WarmingQuery(
            batch_id=batch.id,
            query_text=q,
            status="pending",
            sort_order=i,
            submitted_by=current_user.id,
            created_at=datetime.now(UTC),
        )
        for i, q in enumerate(queries)
    ]
    db.add_all(query_rows)
    db.commit()
    db.refresh(batch)

    # Best-effort ARQ enqueue
    try:
        redis = await get_redis_pool()
        if redis:
            await redis.enqueue_job("process_warming_batch", batch.id)
        else:
            logger.warning(
                f"Redis unavailable, batch {batch.id} saved but no worker will auto-pick up."
            )
    except Exception as e:
        logger.warning(f"ARQ enqueue failed for batch {batch.id}: {e}")

    logger.info(
        f"Admin {current_user.email} added manual warming queries: "
        f"{len(queries)} queries, batch_id={batch.id}"
    )

    return _batch_to_response(batch, db)


@router.delete("/warming/queue/bulk", status_code=status.HTTP_200_OK)
async def bulk_delete_warming_jobs(
    request: BulkDeleteRequest,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Bulk delete warming batches and their queries (CASCADE).

    Skips running batches - cancel them first.

    Admin only.
    """
    if not request.job_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one job_id is required.",
        )

    deleted_count = 0
    skipped_count = 0
    not_found_count = 0

    for job_id in request.job_ids:
        batch = db.query(WarmingBatch).filter(WarmingBatch.id == job_id).first()
        if not batch:
            not_found_count += 1
            continue

        if batch.status == "running":
            skipped_count += 1
            continue

        db.delete(batch)
        deleted_count += 1

    db.commit()

    logger.info(
        f"Admin {current_user.email} bulk deleted warming batches: "
        f"deleted={deleted_count}, skipped={skipped_count}, not_found={not_found_count}"
    )

    return {
        "deleted_count": deleted_count,
        "skipped_count": skipped_count,
        "not_found_count": not_found_count,
    }


@router.get("/warming/queue/{job_id}", response_model=WarmingQueueJobResponse)
async def get_warming_queue_job(
    job_id: str,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get details of a specific warming batch.

    Admin only.
    """
    batch = db.query(WarmingBatch).filter(WarmingBatch.id == job_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {job_id} not found.",
        )

    return _batch_to_response(batch, db)


@router.delete("/warming/queue/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_warming_queue_job(
    job_id: str,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Delete a warming batch and its queries (CASCADE).

    Cannot delete running batches - cancel them first.

    Admin only.
    """
    batch = db.query(WarmingBatch).filter(WarmingBatch.id == job_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {job_id} not found.",
        )

    if batch.status == "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete running batch. Cancel it first.",
        )

    db.delete(batch)
    db.commit()

    logger.info(f"Admin {current_user.email} deleted warming batch {job_id}")


@router.get("/warming/current", response_model=WarmingQueueJobResponse | None)
async def get_current_warming_job(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get the currently running or paused warming batch.

    Returns null if no batch is currently running or paused.

    Admin only.
    """
    batch = (
        db.query(WarmingBatch)
        .filter(WarmingBatch.status.in_(["running", "paused"]))
        .order_by(WarmingBatch.started_at.desc())
        .first()
    )
    if not batch:
        return None

    return _batch_to_response(batch, db)


@router.post("/warming/current/pause", response_model=WarmingQueueJobResponse)
async def pause_current_warming_job(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Pause the currently running warming batch.

    Sets is_paused=True. Worker will detect this flag and pause gracefully
    after the current query completes.

    Admin only.
    """
    from ai_ready_rag.services.sse_buffer_service import store_sse_event

    batch = db.query(WarmingBatch).filter(WarmingBatch.status == "running").first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No running batch to pause.",
        )

    batch.is_paused = True
    db.commit()
    db.refresh(batch)

    store_sse_event(
        db,
        "pause_requested",
        batch.id,
        {"batch_id": batch.id, "status": "pausing"},
    )

    logger.info(f"Admin {current_user.email} paused warming batch {batch.id}")
    return _batch_to_response(batch, db)


@router.post("/warming/current/resume", response_model=WarmingQueueJobResponse)
async def resume_current_warming_job(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Resume a paused warming batch.

    Sets is_paused=False. Worker will detect this and continue processing.

    Admin only.
    """
    batch = (
        db.query(WarmingBatch)
        .filter(WarmingBatch.is_paused == True)  # noqa: E712
        .filter(WarmingBatch.status.in_(["running", "paused"]))
        .first()
    )
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No paused batch to resume.",
        )

    batch.is_paused = False
    db.commit()
    db.refresh(batch)

    logger.info(f"Admin {current_user.email} resumed warming batch {batch.id}")
    return _batch_to_response(batch, db)


@router.post("/warming/current/cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_current_warming_job(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Request graceful cancellation of the currently running warming batch.

    Sets is_cancel_requested=True. Worker will detect this flag and
    update status to 'cancelled'.

    Admin only.
    """
    from ai_ready_rag.services.sse_buffer_service import store_sse_event

    batch = db.query(WarmingBatch).filter(WarmingBatch.status.in_(["running", "paused"])).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No running or paused batch to cancel.",
        )

    batch.is_cancel_requested = True
    db.commit()

    store_sse_event(
        db,
        "cancel_requested",
        batch.id,
        {"batch_id": batch.id, "status": "cancelling"},
    )

    logger.info(f"Admin {current_user.email} requested cancel for warming batch {batch.id}")
    return {"batch_id": batch.id, "is_cancel_requested": True, "status": "cancelling"}


@router.get("/warming/progress")
async def stream_warming_progress_db(
    job_id: str | None = Query(None, description="Batch ID (defaults to current running batch)"),
    token: str | None = None,
    last_event_id: str | None = Query(None, description="Resume from event ID for replay"),
    current_user: User | None = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """Stream warming progress via Server-Sent Events (SSE).

    If job_id is not provided, streams progress for the currently running batch.

    Supports reconnection and replay via last_event_id parameter.

    Admin only. Accepts token via query param for EventSource compatibility.
    """
    from ai_ready_rag.core.dependencies import ROLE_SYSTEM_ADMIN, normalize_role
    from ai_ready_rag.core.security import decode_token

    user = current_user
    if not user and token:
        try:
            payload = decode_token(token)
            user_id = payload.get("sub")
            if user_id:
                user = db.query(User).filter(User.id == user_id).first()
        except Exception:
            pass

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    if normalize_role(user.role) != ROLE_SYSTEM_ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="System admin required")

    target_job_id = job_id
    if not target_job_id:
        running_batch = db.query(WarmingBatch).filter(WarmingBatch.status == "running").first()
        if running_batch:
            target_job_id = running_batch.id
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No running batch found. Provide job_id parameter.",
            )

    batch = db.query(WarmingBatch).filter(WarmingBatch.id == target_job_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {target_job_id} not found.",
        )

    return StreamingResponse(
        _sse_event_generator(target_job_id, last_event_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---- New Batch Query Endpoints ----


@router.get("/warming/batch/{batch_id}/queries", response_model=BatchQueriesResponse)
async def list_batch_queries(
    batch_id: str,
    status_filter: str | None = Query(None, alias="status", description="Filter by query status"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of queries to return"),
    offset: int = Query(0, ge=0, description="Number of queries to skip"),
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """List queries within a warming batch.

    Supports filtering by query status and pagination.

    Admin only.
    """
    batch = db.query(WarmingBatch).filter(WarmingBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_id} not found.",
        )

    query = db.query(WarmingQuery).filter(WarmingQuery.batch_id == batch_id)

    if status_filter:
        query = query.filter(WarmingQuery.status == status_filter)

    total_count = query.count()
    queries = query.order_by(WarmingQuery.sort_order.asc()).offset(offset).limit(limit).all()

    return BatchQueriesResponse(
        queries=[
            WarmingQueryResponse(
                id=q.id,
                query_text=q.query_text,
                status=q.status,
                error_message=q.error_message,
                error_type=q.error_type,
                retry_count=q.retry_count,
                sort_order=q.sort_order,
                processed_at=q.processed_at,
                created_at=q.created_at,
            )
            for q in queries
        ],
        total_count=total_count,
        batch_id=batch_id,
    )


@router.delete(
    "/warming/batch/{batch_id}/queries/{query_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_batch_query(
    batch_id: str,
    query_id: str,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Delete a single pending query from a warming batch.

    Only pending queries can be deleted. Decrements batch total_queries.

    Admin only.
    """
    batch = db.query(WarmingBatch).filter(WarmingBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_id} not found.",
        )

    query_row = (
        db.query(WarmingQuery)
        .filter(WarmingQuery.id == query_id, WarmingQuery.batch_id == batch_id)
        .first()
    )
    if not query_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query {query_id} not found in batch {batch_id}.",
        )

    if query_row.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete query with status '{query_row.status}'. Only pending queries can be deleted.",
        )

    db.delete(query_row)
    batch.total_queries = max(0, batch.total_queries - 1)
    db.commit()

    logger.info(f"Admin {current_user.email} deleted query {query_id} from batch {batch_id}")


@router.post("/warming/batch/{batch_id}/retry", response_model=QueryRetryResponse)
async def retry_batch_failed_queries(
    batch_id: str,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Retry all failed/skipped queries in a batch.

    Resets failed and skipped queries to pending, clears error fields,
    resets batch to pending, and enqueues for processing.

    Admin only.
    """
    batch = db.query(WarmingBatch).filter(WarmingBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_id} not found.",
        )

    terminal_statuses = {"completed", "completed_with_errors", "cancelled"}
    if batch.status not in terminal_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot retry batch with status '{batch.status}'. Batch must be in a terminal state.",
        )

    retried_count = (
        db.query(WarmingQuery)
        .filter(
            WarmingQuery.batch_id == batch_id,
            WarmingQuery.status.in_(["failed", "skipped"]),
        )
        .update(
            {
                "status": "pending",
                "error_message": None,
                "error_type": None,
                "retry_count": 0,
                "processed_at": None,
            },
            synchronize_session="fetch",
        )
    )

    batch.status = "pending"
    batch.completed_at = None
    batch.is_cancel_requested = False
    db.commit()

    # Best-effort ARQ enqueue
    try:
        redis = await get_redis_pool()
        if redis:
            await redis.enqueue_job("process_warming_batch", batch.id)
    except Exception as e:
        logger.warning(f"ARQ enqueue failed for batch retry {batch.id}: {e}")

    logger.info(f"Admin {current_user.email} retried {retried_count} queries in batch {batch_id}")

    return QueryRetryResponse(
        batch_id=batch_id,
        retried_count=retried_count,
        message=f"Reset {retried_count} failed/skipped queries to pending.",
    )


@router.post(
    "/warming/batch/{batch_id}/queries/{query_id}/retry",
    response_model=QueryRetryResponse,
)
async def retry_single_query(
    batch_id: str,
    query_id: str,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Retry a single failed/skipped query.

    Resets the query to pending and re-enqueues the batch if it was terminal.

    Admin only.
    """
    batch = db.query(WarmingBatch).filter(WarmingBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_id} not found.",
        )

    query_row = (
        db.query(WarmingQuery)
        .filter(WarmingQuery.id == query_id, WarmingQuery.batch_id == batch_id)
        .first()
    )
    if not query_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query {query_id} not found in batch {batch_id}.",
        )

    if query_row.status not in ("failed", "skipped"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot retry query with status '{query_row.status}'. Only failed/skipped queries can be retried.",
        )

    query_row.status = "pending"
    query_row.error_message = None
    query_row.error_type = None
    query_row.retry_count = 0
    query_row.processed_at = None

    # If batch is terminal, reset to pending
    terminal_statuses = {"completed", "completed_with_errors", "cancelled"}
    if batch.status in terminal_statuses:
        batch.status = "pending"
        batch.completed_at = None
        batch.is_cancel_requested = False

    db.commit()

    # Best-effort ARQ enqueue
    try:
        redis = await get_redis_pool()
        if redis:
            await redis.enqueue_job("process_warming_batch", batch.id)
    except Exception as e:
        logger.warning(f"ARQ enqueue failed for single query retry {batch.id}: {e}")

    logger.info(f"Admin {current_user.email} retried query {query_id} in batch {batch_id}")

    return QueryRetryResponse(
        batch_id=batch_id,
        retried_count=1,
        message=f"Reset query {query_id} to pending.",
    )


def _format_sse_event(event_type: str, data: dict, event_id: str | None = None) -> str:
    """Format SSE event with event_id for client tracking.

    event_id is now str(batch_seq) for job-scoped events (monotonic integer).
    """
    if event_id is None:
        event_id = str(uuid.uuid4())
    data["event_id"] = event_id
    return f"id: {event_id}\nevent: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _sse_event_generator(job_id: str, last_event_id: str | None = None):
    """Generate SSE events for warming batch progress using DB polling.

    Uses batch_seq-based event IDs for replay. Emits spec-compliant event types:
    connected, progress, paused, complete, error, heartbeat.

    Pause does NOT break the SSE connection -- emits 'paused' once and continues polling.
    Poll interval: 1.0 seconds per spec.
    """
    from ai_ready_rag.db.database import SessionLocal
    from ai_ready_rag.services.sse_buffer_service import (
        get_events_for_job,
        prune_old_events,
        store_sse_event,
    )

    settings = get_settings()

    db = SessionLocal()
    try:
        batch = db.query(WarmingBatch).filter(WarmingBatch.id == job_id).first()
    finally:
        db.close()

    if not batch:
        yield _format_sse_event("error", {"error": "Batch not found"}, str(uuid.uuid4()))
        return

    logger.info(f"[SSE] Started streaming for batch {job_id}, status={batch.status}")

    # Emit connected event with batch_seq-based event_id
    connected_data = {
        "worker_id": f"sse-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(UTC).isoformat(),
    }
    db = SessionLocal()
    try:
        connected_event_id = store_sse_event(db, "connected", job_id, connected_data)
    finally:
        db.close()
    yield _format_sse_event("connected", connected_data, connected_event_id)

    # Replay missed events using batch_seq-based replay
    if last_event_id:
        db = SessionLocal()
        try:
            missed_events = get_events_for_job(db, job_id, since_event_id=last_event_id)
            for event in missed_events:
                yield _format_sse_event(
                    event["event_type"],
                    event["payload"],
                    event["event_id"],
                )
            logger.info(f"[SSE] Replayed {len(missed_events)} events for batch {job_id}")
        finally:
            db.close()

    last_completed = -1
    last_heartbeat_time = asyncio.get_event_loop().time()
    heartbeat_interval = settings.sse_heartbeat_seconds
    paused_emitted = False  # Track pause transition to emit only once

    while True:
        db = SessionLocal()
        try:
            batch = db.query(WarmingBatch).filter(WarmingBatch.id == job_id).first()
            if not batch:
                logger.warning(f"[SSE] Batch {job_id} disappeared unexpectedly")
                yield _format_sse_event("error", {"error": "Batch disappeared"}, str(uuid.uuid4()))
                break

            # Aggregate counts from warming_queries including skipped
            counts = (
                db.query(
                    func.count(case((WarmingQuery.status == "completed", 1))).label("completed"),
                    func.count(case((WarmingQuery.status == "failed", 1))).label("failed"),
                    func.count(case((WarmingQuery.status == "skipped", 1))).label("skipped"),
                )
                .filter(WarmingQuery.batch_id == job_id)
                .first()
            )
            completed_count = counts.completed if counts else 0
            failed_count = counts.failed if counts else 0
            skipped_count = counts.skipped if counts else 0
            processed = completed_count + failed_count + skipped_count
            batch_status = batch.status
            total_queries = batch.total_queries
        finally:
            db.close()

        # Heartbeat check
        current_time = asyncio.get_event_loop().time()
        if current_time - last_heartbeat_time >= heartbeat_interval:
            heartbeat_data = {"timestamp": datetime.now(UTC).isoformat()}
            db = SessionLocal()
            try:
                hb_event_id = store_sse_event(db, "heartbeat", job_id, heartbeat_data)
            finally:
                db.close()
            yield _format_sse_event("heartbeat", heartbeat_data, hb_event_id)
            last_heartbeat_time = current_time

        # Progress event when counts change
        if processed != last_completed:
            last_completed = processed
            percent = int(processed / total_queries * 100) if total_queries > 0 else 0
            progress_data = {
                "batch_id": job_id,
                "processed": processed,
                "failed": failed_count,
                "skipped": skipped_count,
                "total": total_queries,
                "percent": percent,
                "batch_status": batch_status,
            }
            db = SessionLocal()
            try:
                progress_event_id = store_sse_event(db, "progress", job_id, progress_data)
            finally:
                db.close()
            yield _format_sse_event("progress", progress_data, progress_event_id)

        # Terminal states: emit "complete" (not "job_completed"/"job_cancelled")
        terminal = {"completed", "completed_with_errors", "cancelled"}
        if batch_status in terminal:
            final_data = {
                "batch_id": job_id,
                "processed": processed,
                "failed": failed_count,
                "skipped": skipped_count,
                "total": total_queries,
                "status": batch_status,
            }
            db = SessionLocal()
            try:
                complete_event_id = store_sse_event(db, "complete", job_id, final_data)
                prune_old_events(db)
            finally:
                db.close()
            yield _format_sse_event("complete", final_data, complete_event_id)
            break

        # Paused state: emit "paused" ONCE per pause transition, continue polling
        if batch_status == "paused" and not paused_emitted:
            paused_emitted = True
            paused_data = {
                "batch_id": job_id,
                "processed": processed,
                "failed": failed_count,
                "skipped": skipped_count,
                "total": total_queries,
            }
            db = SessionLocal()
            try:
                paused_event_id = store_sse_event(db, "paused", job_id, paused_data)
            finally:
                db.close()
            yield _format_sse_event("paused", paused_data, paused_event_id)
        elif batch_status != "paused":
            paused_emitted = False  # Reset when unpaused

        await asyncio.sleep(1.0)


# =============================================================================
# Synonym CRUD Endpoints - Query Synonym Management
# =============================================================================


@router.get("/synonyms", response_model=SynonymListResponse)
async def list_synonyms(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=50, ge=1, le=100, description="Items per page"),
    enabled: bool | None = Query(default=None, description="Filter by enabled status"),
    search: str | None = Query(default=None, description="Search term (case-insensitive)"),
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """List synonyms with pagination and filtering.

    System admin only.

    Args:
        page: Page number (1-indexed)
        page_size: Items per page (max 100)
        enabled: Filter by enabled status
        search: Case-insensitive search on term field
    """
    query = db.query(QuerySynonym)

    # Apply filters
    if enabled is not None:
        query = query.filter(QuerySynonym.enabled == enabled)

    if search:
        query = query.filter(QuerySynonym.term.ilike(f"%{search}%"))

    # Get total count
    total = query.count()

    # Apply pagination
    offset = (page - 1) * page_size
    synonyms = query.order_by(QuerySynonym.term).offset(offset).limit(page_size).all()

    # Parse JSON synonyms for each record
    result = []
    for syn in synonyms:
        result.append(
            SynonymResponse(
                id=syn.id,
                term=syn.term,
                synonyms=json.loads(syn.synonyms),
                enabled=syn.enabled,
                created_by=syn.created_by,
                created_at=syn.created_at,
                updated_at=syn.updated_at,
            )
        )

    return SynonymListResponse(
        synonyms=result,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/synonyms", response_model=SynonymResponse, status_code=status.HTTP_201_CREATED)
async def create_synonym(
    synonym_data: SynonymCreate,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Create a new synonym mapping.

    System admin only.

    Args:
        synonym_data: Synonym term and list of synonyms
    """
    # Check for duplicate term
    existing = db.query(QuerySynonym).filter(QuerySynonym.term == synonym_data.term).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Synonym for term '{synonym_data.term}' already exists",
        )

    from ai_ready_rag.services.rag_service import invalidate_synonym_cache

    synonym = QuerySynonym(
        term=synonym_data.term,
        synonyms=json.dumps(synonym_data.synonyms),
        enabled=True,
        created_by=current_user.id,
    )
    db.add(synonym)
    db.commit()
    db.refresh(synonym)

    # Invalidate cache after successful create
    invalidate_synonym_cache()

    return SynonymResponse(
        id=synonym.id,
        term=synonym.term,
        synonyms=json.loads(synonym.synonyms),
        enabled=synonym.enabled,
        created_by=synonym.created_by,
        created_at=synonym.created_at,
        updated_at=synonym.updated_at,
    )


@router.put("/synonyms/{synonym_id}", response_model=SynonymResponse)
async def update_synonym(
    synonym_id: str,
    synonym_data: SynonymUpdate,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Update an existing synonym mapping.

    System admin only.

    Args:
        synonym_id: ID of the synonym to update
        synonym_data: Fields to update (partial update supported)
    """
    synonym = db.query(QuerySynonym).filter(QuerySynonym.id == synonym_id).first()
    if not synonym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Synonym not found",
        )

    # Update only provided fields
    if synonym_data.term is not None:
        # Check for duplicate term if changing
        if synonym_data.term != synonym.term:
            existing = db.query(QuerySynonym).filter(QuerySynonym.term == synonym_data.term).first()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Synonym for term '{synonym_data.term}' already exists",
                )
        synonym.term = synonym_data.term

    if synonym_data.synonyms is not None:
        synonym.synonyms = json.dumps(synonym_data.synonyms)

    if synonym_data.enabled is not None:
        synonym.enabled = synonym_data.enabled

    db.commit()
    db.refresh(synonym)

    # Invalidate cache after successful update
    from ai_ready_rag.services.rag_service import invalidate_synonym_cache

    invalidate_synonym_cache()

    return SynonymResponse(
        id=synonym.id,
        term=synonym.term,
        synonyms=json.loads(synonym.synonyms),
        enabled=synonym.enabled,
        created_by=synonym.created_by,
        created_at=synonym.created_at,
        updated_at=synonym.updated_at,
    )


@router.delete("/synonyms/{synonym_id}")
async def delete_synonym(
    synonym_id: str,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Delete a synonym mapping.

    System admin only.

    Args:
        synonym_id: ID of the synonym to delete
    """
    synonym = db.query(QuerySynonym).filter(QuerySynonym.id == synonym_id).first()
    if not synonym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Synonym not found",
        )

    db.delete(synonym)
    db.commit()

    # Invalidate cache after successful delete
    from ai_ready_rag.services.rag_service import invalidate_synonym_cache

    invalidate_synonym_cache()

    return {"success": True, "message": f"Synonym '{synonym.term}' deleted"}


@router.post("/synonyms/invalidate-cache")
async def invalidate_synonym_cache(
    current_user: User = Depends(require_system_admin),
):
    """Invalidate the synonym cache.

    System admin only. Call this after CRUD operations to ensure
    RAG queries use the latest synonym mappings.
    """
    from ai_ready_rag.services.rag_service import invalidate_synonym_cache as do_invalidate

    do_invalidate()
    return {"success": True, "message": "Synonym cache invalidated"}


# =============================================================================
# Curated Q&A CRUD Endpoints - Admin-curated Q&A Management
# =============================================================================


@router.get("/qa", response_model=CuratedQAListResponse)
async def list_curated_qa(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=50, ge=1, le=100, description="Items per page"),
    enabled: bool | None = Query(default=None, description="Filter by enabled status"),
    search: str | None = Query(default=None, description="Search keywords (case-insensitive)"),
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """List curated Q&A pairs with pagination and filtering.

    System admin only.

    Args:
        page: Page number (1-indexed)
        page_size: Items per page (max 100)
        enabled: Filter by enabled status
        search: Case-insensitive search on keywords field
    """
    query = db.query(CuratedQA)

    # Apply filters
    if enabled is not None:
        query = query.filter(CuratedQA.enabled == enabled)

    if search:
        # Search in keywords JSON (simple LIKE search)
        query = query.filter(CuratedQA.keywords.ilike(f"%{search}%"))

    # Get total count
    total = query.count()

    # Apply pagination with ordering by priority DESC, created_at DESC
    offset = (page - 1) * page_size
    qa_pairs = (
        query.order_by(CuratedQA.priority.desc(), CuratedQA.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    # Parse JSON keywords for each record
    result = []
    for qa in qa_pairs:
        result.append(
            CuratedQAResponse(
                id=qa.id,
                keywords=json.loads(qa.keywords),
                answer=qa.answer,
                source_reference=qa.source_reference,
                confidence=qa.confidence,
                priority=qa.priority,
                enabled=qa.enabled,
                access_count=qa.access_count,
                last_accessed_at=qa.last_accessed_at,
                created_by=qa.created_by,
                created_at=qa.created_at,
                updated_at=qa.updated_at,
            )
        )

    return CuratedQAListResponse(
        qa_pairs=result,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/qa", response_model=CuratedQAResponse, status_code=status.HTTP_201_CREATED)
async def create_curated_qa(
    qa_data: CuratedQACreate,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Create a new curated Q&A pair.

    System admin only. HTML in answer field is sanitized.
    Keywords are synced to the curated_qa_keywords table for efficient matching.

    Args:
        qa_data: Q&A keywords, answer, source reference, and optional metadata
    """
    # Sanitize HTML in the answer
    sanitized_answer = sanitize_html(qa_data.answer)

    # Create the Q&A record
    qa = CuratedQA(
        keywords=json.dumps(qa_data.keywords),
        answer=sanitized_answer,
        source_reference=qa_data.source_reference,
        confidence=qa_data.confidence,
        priority=qa_data.priority,
        enabled=True,
        created_by=current_user.id,
    )
    db.add(qa)
    db.flush()  # Get the ID before syncing keywords

    # Sync keywords to the lookup table
    sync_qa_keywords(db, qa.id, qa_data.keywords)

    db.commit()
    db.refresh(qa)

    # Invalidate cache after successful create
    from ai_ready_rag.services.rag_service import invalidate_qa_cache

    invalidate_qa_cache()

    return CuratedQAResponse(
        id=qa.id,
        keywords=json.loads(qa.keywords),
        answer=qa.answer,
        source_reference=qa.source_reference,
        confidence=qa.confidence,
        priority=qa.priority,
        enabled=qa.enabled,
        access_count=qa.access_count,
        last_accessed_at=qa.last_accessed_at,
        created_by=qa.created_by,
        created_at=qa.created_at,
        updated_at=qa.updated_at,
    )


@router.put("/qa/{qa_id}", response_model=CuratedQAResponse)
async def update_curated_qa(
    qa_id: str,
    qa_data: CuratedQAUpdate,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Update an existing curated Q&A pair.

    System admin only. Partial updates supported.
    If keywords are changed, they are re-synced to the lookup table.

    Args:
        qa_id: ID of the Q&A to update
        qa_data: Fields to update (partial update supported)
    """
    qa = db.query(CuratedQA).filter(CuratedQA.id == qa_id).first()
    if not qa:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Curated Q&A not found",
        )

    # Track if keywords changed for re-sync
    keywords_changed = False

    # Update only provided fields
    if qa_data.keywords is not None:
        qa.keywords = json.dumps(qa_data.keywords)
        keywords_changed = True

    if qa_data.answer is not None:
        qa.answer = sanitize_html(qa_data.answer)

    if qa_data.source_reference is not None:
        if not qa_data.source_reference.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="source_reference cannot be empty",
            )
        qa.source_reference = qa_data.source_reference.strip()

    if qa_data.confidence is not None:
        qa.confidence = qa_data.confidence

    if qa_data.priority is not None:
        qa.priority = qa_data.priority

    if qa_data.enabled is not None:
        qa.enabled = qa_data.enabled

    # Re-sync keywords if they changed
    if keywords_changed:
        sync_qa_keywords(db, qa.id, qa_data.keywords)

    db.commit()
    db.refresh(qa)

    # Invalidate cache after successful update
    from ai_ready_rag.services.rag_service import invalidate_qa_cache

    invalidate_qa_cache()

    return CuratedQAResponse(
        id=qa.id,
        keywords=json.loads(qa.keywords),
        answer=qa.answer,
        source_reference=qa.source_reference,
        confidence=qa.confidence,
        priority=qa.priority,
        enabled=qa.enabled,
        access_count=qa.access_count,
        last_accessed_at=qa.last_accessed_at,
        created_by=qa.created_by,
        created_at=qa.created_at,
        updated_at=qa.updated_at,
    )


@router.delete("/qa/{qa_id}")
async def delete_curated_qa(
    qa_id: str,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Delete a curated Q&A pair.

    System admin only. Keywords are automatically deleted via CASCADE.

    Args:
        qa_id: ID of the Q&A to delete
    """
    from ai_ready_rag.services.rag_service import invalidate_qa_cache

    qa = db.query(CuratedQA).filter(CuratedQA.id == qa_id).first()
    if not qa:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Curated Q&A not found",
        )

    # Get keywords for the response message
    keywords = json.loads(qa.keywords)
    first_keyword = keywords[0] if keywords else "unknown"

    db.delete(qa)
    db.commit()

    # Invalidate cache after successful delete
    invalidate_qa_cache()

    return {"success": True, "message": f"Curated Q&A '{first_keyword}...' deleted"}


@router.post("/qa/invalidate-cache")
async def invalidate_curated_qa_cache(
    current_user: User = Depends(require_system_admin),
):
    """Invalidate the curated Q&A cache.

    System admin only. Call this after CRUD operations to ensure
    RAG queries use the latest curated Q&A mappings.
    """
    from ai_ready_rag.services.rag_service import invalidate_qa_cache

    invalidate_qa_cache()
    return {"success": True, "message": "Curated Q&A cache invalidated"}
