"""Admin endpoints for system management."""

import asyncio
import hashlib
import json
import logging
import re
import shutil
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

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
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.dependencies import (
    ROLE_SYSTEM_ADMIN,
    get_optional_current_user,
    normalize_role,
    require_admin,
    require_system_admin,
)
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import Document, QuerySynonym, User, WarmingQueue
from ai_ready_rag.services.document_service import DocumentService
from ai_ready_rag.services.factory import get_vector_service
from ai_ready_rag.services.model_service import ModelService, OllamaUnavailableError
from ai_ready_rag.services.settings_service import SettingsService, get_model_setting
from ai_ready_rag.services.warming_queue import WarmingQueueService

logger = logging.getLogger(__name__)

# Module-level warming queue service (initialized lazily)
_warming_queue: WarmingQueueService | None = None


def get_warming_queue() -> WarmingQueueService:
    """Get or create the warming queue service."""
    global _warming_queue
    if _warming_queue is None:
        settings = get_settings()
        queue_dir = settings.warming_queue_dir
        _warming_queue = WarmingQueueService(
            queue_dir=queue_dir,
            lock_timeout_minutes=settings.warming_lock_timeout_minutes,
            checkpoint_interval=settings.warming_checkpoint_interval,
            archive_completed=settings.warming_archive_completed,
        )
    return _warming_queue


router = APIRouter()


class RecoverResponse(BaseModel):
    recovered: int
    message: str


class FileStats(BaseModel):
    """Per-file statistics."""

    document_id: str
    filename: str
    chunk_count: int
    status: str | None = None


class KnowledgeBaseStatsResponse(BaseModel):
    """Knowledge base statistics response."""

    total_chunks: int
    unique_files: int
    total_vectors: int
    collection_name: str
    files: list[FileStats]
    storage_size_bytes: int | None = None
    last_updated: datetime


class ClearKnowledgeBaseRequest(BaseModel):
    """Request to clear knowledge base."""

    confirm: bool
    delete_source_files: bool = False


class ClearKnowledgeBaseResponse(BaseModel):
    """Response after clearing knowledge base."""

    deleted_chunks: int
    deleted_files: int
    success: bool


# Architecture Info Response Models
class TesseractStatus(BaseModel):
    """Tesseract OCR availability status."""

    available: bool
    version: str | None = None
    languages: list[str] | None = None


class EasyOCRStatus(BaseModel):
    """EasyOCR availability status."""

    available: bool
    version: str | None = None


class OCRStatus(BaseModel):
    """Combined OCR status."""

    tesseract: TesseractStatus
    easyocr: EasyOCRStatus


class DocumentParsingInfo(BaseModel):
    """Document parsing engine information."""

    engine: str = "Docling"
    version: str
    type: str = "local ML"
    capabilities: list[str]


class EmbeddingsInfo(BaseModel):
    """Embeddings configuration."""

    model: str
    dimensions: int
    vector_store: str
    vector_store_url: str


class ChatModelInfo(BaseModel):
    """Chat model configuration."""

    name: str
    provider: str = "Ollama"
    capabilities: list[str]


class InfrastructureStatus(BaseModel):
    """Infrastructure health status."""

    ollama_url: str
    ollama_status: str  # "healthy" | "unhealthy"
    vector_db_status: str  # "healthy" | "unhealthy"


class ArchitectureInfoResponse(BaseModel):
    """Complete architecture information response."""

    document_parsing: DocumentParsingInfo
    embeddings: EmbeddingsInfo
    chat_model: ChatModelInfo
    infrastructure: InfrastructureStatus
    ocr_status: OCRStatus
    profile: str


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


# Processing Options Models and Endpoints
class ProcessingOptionsRequest(BaseModel):
    """Request model for processing options update."""

    enable_ocr: bool | None = None
    force_full_page_ocr: bool | None = None
    ocr_language: str | None = None
    table_extraction_mode: Literal["accurate", "fast"] | None = None
    include_image_descriptions: bool | None = None
    query_routing_mode: Literal["retrieve_only", "retrieve_and_direct"] | None = None


class ProcessingOptionsResponse(BaseModel):
    """Response model for processing options."""

    enable_ocr: bool
    force_full_page_ocr: bool
    ocr_language: str
    table_extraction_mode: str
    include_image_descriptions: bool
    query_routing_mode: str


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


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str
    display_name: str
    size_gb: float
    parameters: str | None = None
    quantization: str | None = None
    recommended: bool = False


class ModelsResponse(BaseModel):
    """Response containing available models and current selection."""

    available_models: list[ModelInfo]
    embedding_models: list[ModelInfo]
    chat_models: list[ModelInfo]
    current_chat_model: str
    current_embedding_model: str


class ChangeModelRequest(BaseModel):
    """Request to change the active chat model."""

    model_name: str


class ChangeModelResponse(BaseModel):
    """Response after changing the chat model."""

    previous_model: str
    current_model: str
    success: bool
    message: str


class ChangeEmbeddingRequest(BaseModel):
    """Request to change the embedding model."""

    model_name: str
    confirm_reindex: bool = False  # Must be True to proceed


class ChangeEmbeddingResponse(BaseModel):
    """Response after changing the embedding model."""

    previous_model: str
    current_model: str
    success: bool
    message: str
    reindex_required: bool = True
    documents_affected: int = 0


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


# Model Limits Endpoint Models
class ModelLimits(BaseModel):
    """Limits for a specific model."""

    context_window: int
    max_response: int
    temperature_min: float = 0.0
    temperature_max: float = 1.0


class ModelLimitsResponse(BaseModel):
    """Response containing model limits for settings validation."""

    current_model: str
    limits: ModelLimits
    all_models: dict[str, ModelLimits]


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


# Detailed Health Endpoint Models
class ComponentHealth(BaseModel):
    """Health status of a single component."""

    name: str
    status: Literal["healthy", "unhealthy", "degraded"]
    version: str | None = None
    details: dict | None = None


class RAGPipelineStatus(BaseModel):
    """RAG pipeline status information."""

    embedding_model: str
    chat_model: str
    chunker: str
    stages: list[str]
    all_stages_healthy: bool


class KnowledgeBaseSummary(BaseModel):
    """Knowledge base summary statistics."""

    total_documents: int
    total_chunks: int
    storage_size_mb: float | None = None


class ProcessingQueueStatus(BaseModel):
    """Document processing queue status."""

    pending: int
    processing: int
    failed: int
    ready: int


class DetailedHealthResponse(BaseModel):
    """Comprehensive health response for admin dashboard."""

    # Overall status
    status: Literal["healthy", "unhealthy", "degraded"]
    version: str
    profile: str

    # Component health
    api_server: ComponentHealth
    ollama_llm: ComponentHealth
    vector_db: ComponentHealth

    # RAG Pipeline status
    rag_pipeline: RAGPipelineStatus

    # Knowledge base summary
    knowledge_base: KnowledgeBaseSummary

    # Processing queue
    processing_queue: ProcessingQueueStatus

    # System info
    uptime_seconds: int
    last_checked: datetime


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


class RetrievalSettingsRequest(BaseModel):
    """Request model for updating retrieval settings."""

    retrieval_top_k: int | None = None
    retrieval_min_score: float | None = None
    retrieval_enable_expansion: bool | None = None


class RetrievalSettingsResponse(BaseModel):
    """Response model for retrieval settings."""

    retrieval_top_k: int
    retrieval_min_score: float
    retrieval_enable_expansion: bool


class LLMSettingsRequest(BaseModel):
    """Request model for updating LLM settings."""

    llm_temperature: float | None = None
    llm_max_response_tokens: int | None = None
    llm_confidence_threshold: int | None = None


class LLMSettingsResponse(BaseModel):
    """Response model for LLM settings."""

    llm_temperature: float
    llm_max_response_tokens: int
    llm_confidence_threshold: int


class SettingsAuditEntry(BaseModel):
    """Response model for settings audit entry."""

    id: str
    setting_key: str
    old_value: str | None
    new_value: str
    changed_by: str | None
    changed_at: datetime
    change_reason: str | None


class SettingsAuditResponse(BaseModel):
    """Response model for settings audit list."""

    entries: list[SettingsAuditEntry]
    total: int
    limit: int
    offset: int


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


class AdvancedSettingsResponse(BaseModel):
    """Response containing advanced (destructive) RAG settings."""

    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    hnsw_ef_construct: int
    hnsw_m: int
    vector_backend: str


class AdvancedSettingsRequest(BaseModel):
    """Request to update advanced settings (requires reindex)."""

    embedding_model: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    hnsw_ef_construct: int | None = None
    hnsw_m: int | None = None
    vector_backend: str | None = None
    confirm_reindex: bool = False


class ReindexJobResponse(BaseModel):
    """Reindex job information."""

    id: str
    status: str
    total_documents: int
    processed_documents: int
    failed_documents: int
    progress_percent: float
    current_document_id: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime
    settings_changed: dict | None = None
    # Phase 3: Failure handling fields
    last_error: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    paused_at: datetime | None = None
    paused_reason: str | None = None
    auto_skip_failures: bool = False


class ReindexEstimate(BaseModel):
    """Time estimate for reindex operation."""

    total_documents: int
    avg_processing_time_ms: int
    estimated_total_seconds: int
    estimated_time_str: str


class StartReindexRequest(BaseModel):
    """Request to start reindex operation."""

    confirm: bool = False


class ResumeReindexRequest(BaseModel):
    """Request to resume a paused reindex."""

    action: Literal["skip", "retry", "skip_all"]


class ReindexFailureInfo(BaseModel):
    """Information about a failed document during reindex."""

    document_id: str
    filename: str
    status: str
    error_message: str | None = None


class ReindexFailuresResponse(BaseModel):
    """Response containing reindex failure details."""

    job_id: str
    failures: list[ReindexFailureInfo]
    total_failures: int


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

    # Start background worker
    background_tasks.add_task(run_reindex_job, job.id)

    return _job_to_response(job)


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

    return _job_to_response(job)


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

    return _job_to_response(job)


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

    return [_job_to_response(job) for job in jobs]


def _job_to_response(job) -> ReindexJobResponse:
    """Convert ReindexJob model to response."""
    import json

    progress = 0.0
    if job.total_documents > 0:
        progress = (job.processed_documents / job.total_documents) * 100

    settings_changed = None
    if job.settings_changed:
        try:
            settings_changed = json.loads(job.settings_changed)
        except json.JSONDecodeError:
            pass

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
        # Phase 3 fields
        last_error=job.last_error,
        retry_count=job.retry_count or 0,
        max_retries=job.max_retries or 3,
        paused_at=job.paused_at,
        paused_reason=job.paused_reason,
        auto_skip_failures=job.auto_skip_failures or False,
    )


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

    return _job_to_response(job)


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

    return _job_to_response(job)


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

    return _job_to_response(job)


# =============================================================================
# Cache Warming Endpoints
# =============================================================================


class TopQueryItem(BaseModel):
    """Single query with access frequency data."""

    query_text: str
    access_count: int
    last_accessed: datetime | None = None


class TopQueryResponse(BaseModel):
    """Response containing top queries for warming."""

    queries: list[TopQueryItem]


class CacheWarmRequest(BaseModel):
    """Request to warm cache with specific queries."""

    queries: list[str]


class CacheWarmResponse(BaseModel):
    """Response after starting cache warming."""

    queued: int
    message: str


class WarmFileResponse(BaseModel):
    """Response after starting file-based cache warming."""

    job_id: str
    queued: int
    message: str
    sse_url: str


class WarmRetryRequest(BaseModel):
    """Request to retry failed warming queries."""

    queries: list[str]


class WarmingJobResponse(BaseModel):
    """Response model for a warming job."""

    id: str
    source_file: str | None = None  # Not tracked in current implementation
    status: str
    total: int
    processed: int
    success_count: int
    failed_count: int
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    triggered_by: str


class WarmingJobListResponse(BaseModel):
    """Response model for list of warming jobs."""

    jobs: list[WarmingJobResponse]
    total_count: int


# =============================================================================
# DB-based Warming Queue Response Models (Issue #121)
# =============================================================================


class WarmingQueueJobResponse(BaseModel):
    """Response model for a DB-based warming queue job."""

    id: str
    file_path: str
    source_type: str
    original_filename: str | None = None
    total_queries: int
    processed_queries: int
    failed_queries: int
    status: str
    is_paused: bool
    is_cancel_requested: bool
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_by: str | None = None
    error_message: str | None = None


class WarmingQueueListResponse(BaseModel):
    """Response model for list of warming queue jobs."""

    jobs: list[WarmingQueueJobResponse]
    total_count: int


class ManualWarmingRequest(BaseModel):
    """Request to add manual queries to warming queue."""

    queries: list[str]


class BulkDeleteRequest(BaseModel):
    """Request to bulk delete warming jobs."""

    job_ids: list[str]


def _db_job_to_response(job: WarmingQueue) -> WarmingQueueJobResponse:
    """Convert WarmingQueue DB model to response."""
    return WarmingQueueJobResponse(
        id=job.id,
        file_path=job.file_path,
        source_type=job.source_type,
        original_filename=job.original_filename,
        total_queries=job.total_queries,
        processed_queries=job.processed_queries,
        failed_queries=job.failed_queries,
        status=job.status,
        is_paused=job.is_paused,
        is_cancel_requested=job.is_cancel_requested,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        created_by=job.created_by,
        error_message=job.error_message,
    )


def _job_to_response(job) -> WarmingJobResponse:
    """Convert a WarmingJob to a WarmingJobResponse."""
    return WarmingJobResponse(
        id=job.id,
        source_file=None,  # Not tracked in current implementation
        status=job.status,
        total=job.total,
        processed=job.processed,
        success_count=job.success_count,
        failed_count=len(job.failed_indices),
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        triggered_by=job.triggered_by,
    )


def _strip_numbering(text: str) -> str:
    """Strip leading numbering from a question (e.g., '1. Question' -> 'Question')."""
    return re.sub(r"^\d+[\.\)\-\s]+", "", text.strip())


# Cache Settings Models
class CacheSettingsResponse(BaseModel):
    """Response model for cache settings."""

    cache_enabled: bool
    cache_ttl_hours: int
    cache_max_entries: int
    cache_semantic_threshold: float
    cache_min_confidence: int
    cache_auto_warm_enabled: bool
    cache_auto_warm_count: int


class CacheSettingsRequest(BaseModel):
    """Request model for updating cache settings."""

    cache_enabled: bool | None = None
    cache_ttl_hours: int | None = None
    cache_max_entries: int | None = None
    cache_semantic_threshold: float | None = None
    cache_min_confidence: int | None = None
    cache_auto_warm_enabled: bool | None = None
    cache_auto_warm_count: int | None = None


# Cache Statistics Models
class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""

    enabled: bool
    total_entries: int
    memory_entries: int
    sqlite_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    avg_response_time_cached_ms: float | None = None
    avg_response_time_uncached_ms: float | None = None
    storage_size_bytes: int | None = None
    oldest_entry: datetime | None = None
    newest_entry: datetime | None = None


# Cache Clear Models
class CacheClearResponse(BaseModel):
    """Response after clearing cache."""

    cleared_entries: int
    message: str


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


@router.post("/cache/warm", response_model=CacheWarmResponse, status_code=status.HTTP_202_ACCEPTED)
async def warm_cache(
    request: CacheWarmRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Warm cache with specified queries.

    Runs each query through the RAG pipeline and caches the response.
    Executes in background, returns immediately with 202 Accepted.

    Admin only.
    """
    if not request.queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one query is required.",
        )

    # Start background warming
    background_tasks.add_task(
        warm_cache_task,
        queries=request.queries,
        triggered_by=current_user.id,
    )

    logger.info(
        f"Admin {current_user.email} started cache warming with {len(request.queries)} queries"
    )

    return CacheWarmResponse(
        queued=len(request.queries),
        message="Cache warming started in background",
    )


async def warm_cache_task(queries: list[str], triggered_by: str) -> None:
    """Background task to warm cache with queries.

    Runs each query through RAG pipeline and caches response.
    Uses empty user_tags (admin context) for warming - this allows
    the cache to be populated with responses that can then be filtered
    by access control on retrieval.

    Args:
        queries: List of query strings to warm
        triggered_by: User ID who triggered the warming
    """
    from ai_ready_rag.config import get_settings
    from ai_ready_rag.db.database import SessionLocal
    from ai_ready_rag.services.rag_service import RAGRequest, RAGService

    settings = get_settings()
    db = SessionLocal()

    try:
        rag_service = RAGService(settings)
        warmed = 0

        for i, query in enumerate(queries):
            try:
                request = RAGRequest(
                    query=query,
                    user_tags=[],  # Admin context - responses cached without tag restriction
                    tenant_id="default",
                )
                # Run query through RAG pipeline (will cache result)
                await rag_service.generate(request, db)
                warmed += 1
                logger.debug(f"Warmed cache for query: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to warm cache for query '{query[:50]}...': {e}")

            # Throttle to reduce Ollama contention with live user requests
            if i < len(queries) - 1 and settings.warming_delay_seconds > 0:
                await asyncio.sleep(settings.warming_delay_seconds)

        logger.info(
            f"Cache warming complete: {warmed}/{len(queries)} queries processed "
            f"(triggered by: {triggered_by})"
        )

    except Exception as e:
        logger.error(f"Cache warming task failed: {e}")
    finally:
        db.close()


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


# =============================================================================
# DB-based Warming Queue Endpoints (Issue #121 - New URL Pattern)
# =============================================================================


def _ensure_warming_dir() -> Path:
    """Ensure warming directory exists and return path."""
    settings = get_settings()
    warming_dir = Path(settings.warming_queue_dir)
    warming_dir.mkdir(parents=True, exist_ok=True)
    return warming_dir


@router.get("/warming/queue", response_model=WarmingQueueListResponse)
async def list_warming_queue(
    status_filter: str | None = Query(None, alias="status", description="Filter by job status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """List all warming queue jobs from database.

    Supports filtering by status and pagination.

    Admin only.
    """
    query = db.query(WarmingQueue)

    if status_filter:
        query = query.filter(WarmingQueue.status == status_filter)

    # Order by created_at descending (newest first)
    query = query.order_by(WarmingQueue.created_at.desc())

    total_count = query.count()
    jobs = query.offset(offset).limit(limit).all()

    return WarmingQueueListResponse(
        jobs=[_db_job_to_response(j) for j in jobs],
        total_count=total_count,
    )


@router.get("/warming/queue/completed", response_model=WarmingQueueListResponse)
async def list_completed_warming_jobs(
    date_filter: str | None = Query(
        None, description="Filter by date (YYYY-MM-DD), defaults to today"
    ),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """List completed warming jobs from database.

    Defaults to showing jobs completed today.

    Admin only.
    """
    from datetime import date

    query = db.query(WarmingQueue).filter(WarmingQueue.status == "completed")

    # Apply date filter
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

    # Filter by completed_at date
    query = query.filter(
        WarmingQueue.completed_at >= datetime.combine(filter_date, datetime.min.time()),
        WarmingQueue.completed_at < datetime.combine(filter_date, datetime.max.time()),
    )

    # Order by completed_at descending (newest first)
    query = query.order_by(WarmingQueue.completed_at.desc())

    total_count = query.count()
    jobs = query.offset(offset).limit(limit).all()

    return WarmingQueueListResponse(
        jobs=[_db_job_to_response(j) for j in jobs],
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

    Saves the file to disk and creates a DB record. WarmingWorker will
    automatically pick up the job.

    Supports .txt and .csv files with one query per line.

    Admin only.
    """
    # Validate file type
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

    # Read and parse file
    try:
        content = await file.read()
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be UTF-8 encoded text.",
        ) from None

    # Parse questions (one per line, strip numbering)
    queries = []
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):  # Skip empty lines and comments
            cleaned = _strip_numbering(line)
            if cleaned:
                queries.append(cleaned)

    if not queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid questions found in file.",
        )

    # Calculate checksum
    file_content = "\n".join(queries)
    checksum = hashlib.sha256(file_content.encode()).hexdigest()

    # Save to disk
    warming_dir = _ensure_warming_dir()
    job_id = str(uuid.uuid4())
    file_path = warming_dir / f"upload_{job_id}.txt"
    file_path.write_text(file_content, encoding="utf-8")

    # Create DB record
    job = WarmingQueue(
        id=job_id,
        file_path=str(file_path),
        file_checksum=checksum,
        source_type="upload",
        original_filename=file.filename,
        total_queries=len(queries),
        processed_queries=0,
        failed_queries=0,
        byte_offset=0,
        status="pending",
        is_paused=False,
        is_cancel_requested=False,
        created_at=datetime.now(UTC),
        created_by=current_user.id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    logger.info(
        f"Admin {current_user.email} uploaded warming file: {len(queries)} queries, job_id={job_id}"
    )

    return _db_job_to_response(job)


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

    Saves queries to a file and creates a DB record. WarmingWorker will
    automatically pick up the job.

    Admin only.
    """
    if not request.queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one query is required.",
        )

    # Clean and validate queries
    queries = []
    for q in request.queries:
        cleaned = q.strip()
        if cleaned and not cleaned.startswith("#"):
            queries.append(cleaned)

    if not queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid queries provided.",
        )

    # Calculate checksum
    file_content = "\n".join(queries)
    checksum = hashlib.sha256(file_content.encode()).hexdigest()

    # Save to disk
    warming_dir = _ensure_warming_dir()
    job_id = str(uuid.uuid4())
    file_path = warming_dir / f"manual_{job_id}.txt"
    file_path.write_text(file_content, encoding="utf-8")

    # Create DB record
    job = WarmingQueue(
        id=job_id,
        file_path=str(file_path),
        file_checksum=checksum,
        source_type="manual",
        original_filename=None,
        total_queries=len(queries),
        processed_queries=0,
        failed_queries=0,
        byte_offset=0,
        status="pending",
        is_paused=False,
        is_cancel_requested=False,
        created_at=datetime.now(UTC),
        created_by=current_user.id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    logger.info(
        f"Admin {current_user.email} added manual warming queries: "
        f"{len(queries)} queries, job_id={job_id}"
    )

    return _db_job_to_response(job)


@router.get("/warming/queue/{job_id}", response_model=WarmingQueueJobResponse)
async def get_warming_queue_job(
    job_id: str,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get details of a specific warming queue job from database.

    Admin only.
    """
    job = db.query(WarmingQueue).filter(WarmingQueue.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    return _db_job_to_response(job)


@router.delete("/warming/queue/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_warming_queue_job(
    job_id: str,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Delete a warming queue job from database.

    Also deletes the associated query file if it exists.
    Cannot delete running jobs - cancel them first.

    Admin only.
    """
    job = db.query(WarmingQueue).filter(WarmingQueue.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    if job.status == "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete running job. Cancel it first.",
        )

    # Delete associated file if exists
    if job.file_path:
        file_path = Path(job.file_path)
        if file_path.exists():
            try:
                file_path.unlink()
            except OSError as e:
                logger.warning(f"Could not delete warming file {file_path}: {e}")

    db.delete(job)
    db.commit()

    logger.info(f"Admin {current_user.email} deleted warming job {job_id}")
    return {"success": True, "job_id": job_id}


@router.delete("/warming/queue/bulk", status_code=status.HTTP_200_OK)
async def bulk_delete_warming_jobs(
    request: BulkDeleteRequest,
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Bulk delete warming queue jobs from database.

    Also deletes the associated query files if they exist.
    Skips running jobs - cancel them first.

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
        job = db.query(WarmingQueue).filter(WarmingQueue.id == job_id).first()
        if not job:
            not_found_count += 1
            continue

        if job.status == "running":
            skipped_count += 1
            continue

        # Delete associated file if exists
        if job.file_path:
            file_path = Path(job.file_path)
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError as e:
                    logger.warning(f"Could not delete warming file {file_path}: {e}")

        db.delete(job)
        deleted_count += 1

    db.commit()

    logger.info(
        f"Admin {current_user.email} bulk deleted warming jobs: "
        f"deleted={deleted_count}, skipped={skipped_count}, not_found={not_found_count}"
    )

    return {
        "deleted_count": deleted_count,
        "skipped_count": skipped_count,
        "not_found_count": not_found_count,
    }


@router.get("/warming/current", response_model=WarmingQueueJobResponse | None)
async def get_current_warming_job(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Get the currently running warming job from database.

    Returns null if no job is currently running.

    Admin only.
    """
    job = db.query(WarmingQueue).filter(WarmingQueue.status == "running").first()
    if not job:
        return None

    return _db_job_to_response(job)


@router.post("/warming/current/pause", response_model=WarmingQueueJobResponse)
async def pause_current_warming_job(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Pause the currently running warming job.

    Sets is_paused=True. Worker will detect this flag and pause gracefully
    after the current query completes.

    Admin only.
    """
    from ai_ready_rag.services.sse_buffer_service import store_sse_event

    job = db.query(WarmingQueue).filter(WarmingQueue.status == "running").first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No running job to pause.",
        )

    job.is_paused = True
    db.commit()
    db.refresh(job)

    # Emit SSE event for immediate UI update
    store_sse_event(
        db,
        "pause_requested",
        job.id,
        {"job_id": job.id, "status": "pausing"},
    )

    logger.info(f"Admin {current_user.email} paused warming job {job.id}")
    return _db_job_to_response(job)


@router.post("/warming/current/resume", response_model=WarmingQueueJobResponse)
async def resume_current_warming_job(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Resume a paused warming job.

    Sets is_paused=False. Worker will detect this and continue processing.

    Admin only.
    """
    # Find paused job (prioritize running+paused, then paused status)
    job = (
        db.query(WarmingQueue)
        .filter(WarmingQueue.is_paused == True)  # noqa: E712
        .filter(WarmingQueue.status.in_(["running", "paused"]))
        .first()
    )
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No paused job to resume.",
        )

    job.is_paused = False
    # If status was explicitly paused, reset to pending for worker pickup
    if job.status == "paused":
        job.status = "pending"
    db.commit()
    db.refresh(job)

    logger.info(f"Admin {current_user.email} resumed warming job {job.id}")
    return _db_job_to_response(job)


@router.post("/warming/current/cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_current_warming_job(
    current_user: User = Depends(require_system_admin),
    db: Session = Depends(get_db),
):
    """Request graceful cancellation of the currently running warming job.

    Sets is_cancel_requested=True. Worker will detect this flag,
    close the file handle gracefully, and update status to 'cancelled'.

    Returns status "cancelling" to indicate transitional state.

    Admin only.
    """
    from ai_ready_rag.services.sse_buffer_service import store_sse_event

    job = db.query(WarmingQueue).filter(WarmingQueue.status.in_(["running", "paused"])).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No running or paused job to cancel.",
        )

    job.is_cancel_requested = True
    db.commit()

    # Emit SSE event for immediate UI update
    store_sse_event(
        db,
        "cancel_requested",
        job.id,
        {"job_id": job.id, "status": "cancelling"},
    )

    logger.info(f"Admin {current_user.email} requested cancel for warming job {job.id}")
    return {"job_id": job.id, "is_cancel_requested": True, "status": "cancelling"}


@router.get("/warming/progress")
async def stream_warming_progress_db(
    job_id: str | None = Query(None, description="Job ID (defaults to current running job)"),
    token: str | None = None,
    last_event_id: str | None = Query(None, description="Resume from event ID for replay"),
    current_user: User | None = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """Stream warming progress via Server-Sent Events (SSE).

    If job_id is not provided, streams progress for the currently running job.

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

    # Get job_id from query or find current running job
    target_job_id = job_id
    if not target_job_id:
        running_job = db.query(WarmingQueue).filter(WarmingQueue.status == "running").first()
        if running_job:
            target_job_id = running_job.id
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No running job found. Provide job_id parameter.",
            )

    # Verify job exists
    job = db.query(WarmingQueue).filter(WarmingQueue.id == target_job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {target_job_id} not found.",
        )

    # Reuse existing SSE generator (it reads from file-based queue service
    # but we can still use it for progress - alternatively, we'd create
    # a DB-based generator)
    return StreamingResponse(
        _sse_event_generator(target_job_id, last_event_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# File-based Cache Warming Endpoints (DEPRECATED - use /warming/* endpoints)
# =============================================================================


@router.post(
    "/cache/warm-file",
    response_model=WarmFileResponse,
    status_code=status.HTTP_202_ACCEPTED,
    deprecated=True,
)
async def warm_cache_from_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_system_admin),
):
    """Upload a file containing queries to warm the cache.

    DEPRECATED: Use POST /api/admin/warming/queue/upload instead.

    Supports .txt and .csv files with one query per line.
    Numbered queries (e.g., "1. Question") are automatically cleaned.

    Returns a job_id and SSE URL for progress monitoring.

    Admin only.
    """
    logger.warning(
        f"DEPRECATED: /cache/warm-file called by {current_user.email}. "
        "Use /warming/queue/upload instead."
    )
    # Validate file type
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

    # Read and parse file
    try:
        content = await file.read()
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be UTF-8 encoded text.",
        ) from None

    # Parse questions (one per line, strip numbering)
    questions = []
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):  # Skip empty lines and comments
            cleaned = _strip_numbering(line)
            if cleaned:
                questions.append(cleaned)

    if not questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid questions found in file.",
        )

    # Create job via queue service (persists to disk)
    queue_service = get_warming_queue()
    job = queue_service.create_job(questions, triggered_by=current_user.id)

    # Start background warming
    def run_warming():
        import asyncio

        try:
            asyncio.run(_warm_file_task(job.id, current_user.id))
        except Exception as e:
            logger.error(f"Cache warming job {job.id} failed: {e}")

    background_tasks.add_task(run_warming)

    logger.info(
        f"Admin {current_user.email} started file cache warming: "
        f"{len(questions)} questions, job_id={job.id}"
    )

    return WarmFileResponse(
        job_id=job.id,
        queued=len(questions),
        message="Cache warming started. Connect to SSE for progress.",
        sse_url=f"/api/admin/cache/warm-progress/{job.id}",
    )


@router.get("/cache/warm-progress/{job_id}")
async def stream_warming_progress(
    job_id: str,
    token: str | None = None,
    last_event_id: str | None = Query(None, description="Resume from event ID for replay"),
    resume_job: str | None = Query(None, description="Get status of specific job"),
    current_user: User | None = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """Stream cache warming progress via Server-Sent Events (SSE).

    Supports reconnection and replay via last_event_id parameter.

    Event types (all include event_id):
    - connected: {"worker_id": "...", "timestamp": "..."}
    - job_started: {"job_id": "...", "file_path": "...", "total_queries": N}
    - progress: {"processed": N, "failed": F, "total": M, "percent": P, "estimated_remaining_seconds": S, "queries_per_second": Q}
    - query_failed: {"query": "...", "line_number": N, "error": "...", "error_type": "..."}
    - job_completed: {"processed": N, "failed": F, "total": M, "duration_seconds": D, "queries_per_second": Q}
    - job_paused: {"processed": N, "total": M}
    - job_cancelled: {"processed": N, "total": M, "file_deleted": bool}
    - job_failed: {"error": "..."}
    - heartbeat: {"timestamp": "..."}

    Admin only. Accepts token via query param for EventSource compatibility.
    """
    # SSE/EventSource can't send headers, so accept token from query param
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

    # Use resume_job if provided, otherwise use job_id from path
    target_job_id = resume_job or job_id

    queue_service = get_warming_queue()
    job = queue_service.get_job(target_job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {target_job_id} not found.",
        )

    return StreamingResponse(
        _sse_event_generator(target_job_id, last_event_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.post(
    "/cache/warm-retry", response_model=WarmFileResponse, status_code=status.HTTP_202_ACCEPTED
)
async def retry_warming_queries(
    request: WarmRetryRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_system_admin),
):
    """Retry specific failed queries from a previous warming job.

    Accepts a list of queries to retry and starts a new warming job.

    Admin only.
    """
    if not request.queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one query is required.",
        )

    # Create new job for retry via queue service
    queue_service = get_warming_queue()
    job = queue_service.create_job(request.queries, triggered_by=current_user.id)

    # Start background warming
    def run_warming():
        import asyncio

        try:
            asyncio.run(_warm_file_task(job.id, current_user.id))
        except Exception as e:
            logger.error(f"Cache warming retry job {job.id} failed: {e}")

    background_tasks.add_task(run_warming)

    logger.info(
        f"Admin {current_user.email} started retry warming: "
        f"{len(request.queries)} queries, job_id={job.id}"
    )

    return WarmFileResponse(
        job_id=job.id,
        queued=len(request.queries),
        message="Retry warming started. Connect to SSE for progress.",
        sse_url=f"/api/admin/cache/warm-progress/{job.id}",
    )


@router.get("/cache/warm-status/{job_id}")
async def get_warming_status(
    job_id: str,
    current_user: User = Depends(require_system_admin),
):
    """Get current status of a warming job.

    Used to recover state after navigation or page refresh.

    Admin only.
    """
    queue_service = get_warming_queue()
    job = queue_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    return {
        "job_id": job.id,
        "status": job.status,
        "total": job.total,
        "processed": job.processed,
        "success_count": job.success_count,
        "failed_queries": job.failed_queries,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@router.get("/cache/warm-jobs", response_model=WarmingJobListResponse, deprecated=True)
async def list_warming_jobs(
    status_filter: str | None = Query(None, alias="status", description="Filter by job status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    current_user: User = Depends(require_system_admin),
):
    """List all warming jobs with optional filtering.

    DEPRECATED: Use GET /api/admin/warming/queue instead.

    Supports filtering by status and pagination.

    Admin only.
    """
    logger.warning(
        f"DEPRECATED: /cache/warm-jobs called by {current_user.email}. Use /warming/queue instead."
    )
    queue_service = get_warming_queue()
    all_jobs = queue_service.list_all_jobs()

    # Filter by status if provided
    if status_filter:
        all_jobs = [j for j in all_jobs if j.status == status_filter]

    total_count = len(all_jobs)

    # Apply pagination
    paginated_jobs = all_jobs[offset : offset + limit]

    return WarmingJobListResponse(
        jobs=[_job_to_response(j) for j in paginated_jobs],
        total_count=total_count,
    )


@router.get("/cache/warm-jobs/active", response_model=WarmingJobResponse | None, deprecated=True)
async def get_active_warming_job(
    current_user: User = Depends(require_system_admin),
):
    """Get the currently running warming job, if any.

    DEPRECATED: Use GET /api/admin/warming/current instead.

    Returns null if no job is currently running.

    Admin only.
    """
    logger.warning(
        f"DEPRECATED: /cache/warm-jobs/active called by {current_user.email}. "
        "Use /warming/current instead."
    )
    queue_service = get_warming_queue()
    all_jobs = queue_service.list_all_jobs()

    # Find running job
    for job in all_jobs:
        if job.status == "running":
            return _job_to_response(job)

    return None


@router.get("/cache/warm-jobs/{job_id}", response_model=WarmingJobResponse, deprecated=True)
async def get_warming_job(
    job_id: str,
    current_user: User = Depends(require_system_admin),
):
    """Get details of a specific warming job.

    DEPRECATED: Use GET /api/admin/warming/queue/{id} instead.

    Admin only.
    """
    logger.warning(
        f"DEPRECATED: /cache/warm-jobs/{job_id} called by {current_user.email}. "
        "Use /warming/queue/{id} instead."
    )
    queue_service = get_warming_queue()
    job = queue_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    return _job_to_response(job)


@router.post("/cache/warm-jobs/{job_id}/pause", response_model=WarmingJobResponse, deprecated=True)
async def pause_warming_job(
    job_id: str,
    current_user: User = Depends(require_system_admin),
):
    """Pause a running warming job.

    DEPRECATED: Use POST /api/admin/warming/current/pause instead.

    Only running jobs can be paused.

    Admin only.
    """
    logger.warning(
        f"DEPRECATED: /cache/warm-jobs/{job_id}/pause called by {current_user.email}. "
        "Use /warming/current/pause instead."
    )
    queue_service = get_warming_queue()

    # Check if job exists
    job = queue_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    # Check if job is running
    if job.status != "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot pause job with status '{job.status}'. Only running jobs can be paused.",
        )

    paused_job = queue_service.pause_job(job_id)
    if not paused_job:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Could not pause job. It may have been modified by another process.",
        )

    logger.info(f"Admin {current_user.email} paused warming job {job_id}")
    return _job_to_response(paused_job)


@router.post("/cache/warm-jobs/{job_id}/resume", response_model=WarmingJobResponse, deprecated=True)
async def resume_warming_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_system_admin),
):
    """Resume a paused warming job.

    DEPRECATED: Use POST /api/admin/warming/current/resume instead.

    Only paused jobs can be resumed. This will restart the background processing.

    Admin only.
    """
    logger.warning(
        f"DEPRECATED: /cache/warm-jobs/{job_id}/resume called by {current_user.email}. "
        "Use /warming/current/resume instead."
    )
    queue_service = get_warming_queue()

    # Check if job exists
    job = queue_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    # Check if job is paused
    if job.status != "paused":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot resume job with status '{job.status}'. Only paused jobs can be resumed.",
        )

    worker_id = f"worker-{uuid.uuid4().hex[:8]}"
    resumed_job = queue_service.resume_job(job_id, worker_id)
    if not resumed_job:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Could not resume job. It may have been modified by another process.",
        )

    # Restart background warming task
    def run_warming():
        import asyncio

        try:
            asyncio.run(_warm_file_task(job_id, current_user.id))
        except Exception as e:
            logger.error(f"Cache warming job {job_id} failed after resume: {e}")

    background_tasks.add_task(run_warming)

    logger.info(f"Admin {current_user.email} resumed warming job {job_id}")
    return _job_to_response(resumed_job)


@router.post(
    "/cache/warm-jobs/{job_id}/cancel", status_code=status.HTTP_202_ACCEPTED, deprecated=True
)
async def cancel_warming_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_system_admin),
):
    """Request graceful cancellation of a warming job.

    DEPRECATED: Use POST /api/admin/warming/current/cancel instead.

    Sets is_cancel_requested=TRUE. The worker will detect this flag,
    close the file handle gracefully, update status to 'cancelled',
    and delete the query file.

    Only running or paused jobs can be cancelled.

    Admin only.
    """
    logger.warning(
        f"DEPRECATED: /cache/warm-jobs/{job_id}/cancel called by {current_user.email}. "
        "Use /warming/current/cancel instead."
    )
    # Query job from database
    job = db.query(WarmingQueue).filter(WarmingQueue.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    # Only running or paused jobs can be cancelled gracefully
    if job.status not in ("running", "paused"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status '{job.status}'. Only running or paused jobs can be cancelled.",
        )

    # Set the cancel flag for graceful shutdown
    job.is_cancel_requested = True
    db.commit()

    logger.info(f"Admin {current_user.email} requested cancel for warming job {job_id}")
    return {"job_id": job_id, "is_cancel_requested": True}


@router.delete("/cache/warm-jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT, deprecated=True)
async def delete_warming_job(
    job_id: str,
    current_user: User = Depends(require_system_admin),
):
    """Delete/cancel a warming job.

    DEPRECATED: Use DELETE /api/admin/warming/queue/{id} instead.

    Can delete jobs in any status.

    Admin only.
    """
    logger.warning(
        f"DEPRECATED: /cache/warm-jobs/{job_id} DELETE called by {current_user.email}. "
        "Use DELETE /warming/queue/{id} instead."
    )
    queue_service = get_warming_queue()

    # Check if job exists
    job = queue_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    queue_service.delete_job(job_id)
    logger.info(f"Admin {current_user.email} deleted warming job {job_id}")
    return None


def _format_sse_event(event_type: str, data: dict, event_id: str | None = None) -> str:
    """Format SSE event with event_id for client tracking.

    Args:
        event_type: Event type name
        data: Event payload data
        event_id: Optional event ID (generates UUID if None)

    Returns:
        Formatted SSE event string
    """
    if event_id is None:
        event_id = str(uuid.uuid4())
    data["event_id"] = event_id
    return f"id: {event_id}\nevent: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _sse_event_generator(job_id: str, last_event_id: str | None = None):
    """Generate SSE events for warming job progress.

    Supports replay via last_event_id and emits heartbeats every 30 seconds.

    Args:
        job_id: Job ID to stream events for
        last_event_id: Optional event ID to resume from (for replay)
    """
    from ai_ready_rag.db.database import SessionLocal
    from ai_ready_rag.services.sse_buffer_service import (
        get_events_for_job,
        prune_old_events,
        store_sse_event,
    )

    settings = get_settings()
    queue_service = get_warming_queue()
    job = queue_service.get_job(job_id)
    if not job:
        event_id = str(uuid.uuid4())
        yield _format_sse_event("error", {"error": "Job not found"}, event_id)
        return

    logger.info(f"[SSE] Started streaming for job {job_id}, status={job.status}")

    # Send connected event
    connected_event_id = str(uuid.uuid4())
    connected_data = {
        "worker_id": f"sse-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(UTC).isoformat(),
    }
    db = SessionLocal()
    try:
        store_sse_event(db, "connected", job_id, connected_data)
    finally:
        db.close()
    yield _format_sse_event("connected", connected_data, connected_event_id)

    # Replay missed events if last_event_id provided
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
            logger.info(f"[SSE] Replayed {len(missed_events)} events for job {job_id}")
        finally:
            db.close()

    last_processed = -1
    last_results_count = 0
    last_heartbeat_time = asyncio.get_event_loop().time()
    heartbeat_interval = settings.sse_heartbeat_seconds

    while True:
        job = queue_service.get_job(job_id)
        if not job:
            logger.warning(f"[SSE] Job {job_id} disappeared unexpectedly")
            error_event_id = str(uuid.uuid4())
            yield _format_sse_event("error", {"error": "Job disappeared"}, error_event_id)
            break

        # Send heartbeat if interval elapsed
        current_time = asyncio.get_event_loop().time()
        if current_time - last_heartbeat_time >= heartbeat_interval:
            heartbeat_event_id = str(uuid.uuid4())
            heartbeat_data = {"timestamp": datetime.now(UTC).isoformat()}
            db = SessionLocal()
            try:
                store_sse_event(db, "heartbeat", job_id, heartbeat_data)
            finally:
                db.close()
            yield _format_sse_event("heartbeat", heartbeat_data, heartbeat_event_id)
            last_heartbeat_time = current_time

        # Send progress update if changed
        if job.processed != last_processed:
            last_processed = job.processed
            elapsed = 0
            remaining = None
            qps = 0.0

            if job.started_at and job.processed > 0:
                elapsed = (datetime.now(UTC) - job.started_at).total_seconds()
                avg_per_query = elapsed / job.processed
                remaining_queries = job.total - job.processed
                remaining = int(avg_per_query * remaining_queries)
                qps = job.processed / elapsed if elapsed > 0 else 0.0

            percent = int(job.processed / job.total * 100) if job.total > 0 else 0
            progress_event_id = str(uuid.uuid4())
            progress_data = {
                "job_id": job_id,
                "processed": job.processed,
                "failed": len(job.failed_queries) if hasattr(job, "failed_queries") else 0,
                "total": job.total,
                "percent": percent,
                "estimated_remaining_seconds": remaining,
                "queries_per_second": round(qps, 2),
            }
            db = SessionLocal()
            try:
                store_sse_event(db, "progress", job_id, progress_data)
            finally:
                db.close()
            yield _format_sse_event("progress", progress_data, progress_event_id)

        # Send new results (query_failed events)
        if len(job.results) > last_results_count:
            for result in job.results[last_results_count:]:
                result_event_id = str(uuid.uuid4())
                db = SessionLocal()
                try:
                    # Store as query_failed if it's a failed result
                    if result.get("status") == "failed":
                        store_sse_event(db, "query_failed", job_id, result)
                        yield _format_sse_event("query_failed", result, result_event_id)
                    else:
                        store_sse_event(db, "result", job_id, result)
                        yield _format_sse_event("result", result, result_event_id)
                finally:
                    db.close()
            last_results_count = len(job.results)

        # Check for completion, pause, or cancellation
        if job.status in ("completed", "failed", "paused", "cancelled"):
            duration = 0
            qps = 0.0
            if job.started_at and job.completed_at:
                duration = (job.completed_at - job.started_at).total_seconds()
                qps = job.processed / duration if duration > 0 else 0.0

            final_event_id = str(uuid.uuid4())
            db = SessionLocal()
            try:
                if job.status == "completed":
                    complete_data = {
                        "job_id": job_id,
                        "processed": job.processed,
                        "failed": len(job.failed_queries) if hasattr(job, "failed_queries") else 0,
                        "total": job.total,
                        "duration_seconds": int(duration),
                        "queries_per_second": round(qps, 2),
                    }
                    store_sse_event(db, "job_completed", job_id, complete_data)
                    yield _format_sse_event("job_completed", complete_data, final_event_id)
                elif job.status == "failed":
                    failed_data = {
                        "job_id": job_id,
                        "error": getattr(job, "error_message", "Unknown error"),
                    }
                    store_sse_event(db, "job_failed", job_id, failed_data)
                    yield _format_sse_event("job_failed", failed_data, final_event_id)
                elif job.status == "paused":
                    paused_data = {
                        "job_id": job_id,
                        "processed": job.processed,
                        "total": job.total,
                    }
                    store_sse_event(db, "job_paused", job_id, paused_data)
                    yield _format_sse_event("job_paused", paused_data, final_event_id)
                elif job.status == "cancelled":
                    cancelled_data = {
                        "job_id": job_id,
                        "processed": job.processed,
                        "total": job.total,
                        "file_deleted": False,  # File cleanup handled separately
                    }
                    store_sse_event(db, "job_cancelled", job_id, cancelled_data)
                    yield _format_sse_event("job_cancelled", cancelled_data, final_event_id)

                # Periodic pruning of old events
                prune_old_events(db)
            finally:
                db.close()
            break

        await asyncio.sleep(0.5)


async def _warm_file_task(job_id: str, triggered_by: str) -> None:
    """Background task to warm cache with queries from file.

    Updates job state as queries are processed for SSE streaming.
    Persists progress to disk for crash recovery.
    """
    from ai_ready_rag.config import get_settings
    from ai_ready_rag.db.database import SessionLocal
    from ai_ready_rag.services.rag_service import RAGRequest, RAGService

    logger.info(f"[WARM] Starting warming task for job {job_id}")
    queue_service = get_warming_queue()
    worker_id = f"worker-{uuid.uuid4().hex[:8]}"

    # Acquire job with lock
    job = queue_service.acquire_job(job_id, worker_id)
    if not job:
        logger.warning(f"[WARM] Could not acquire warming job {job_id}")
        return

    logger.info(f"[WARM] Acquired job {job_id} with {job.total} queries")
    settings = get_settings()
    db = SessionLocal()

    try:
        # Initialize vector service properly
        from ai_ready_rag.services.vector_service import VectorService

        vector_service = VectorService(
            qdrant_url=settings.qdrant_url,
            ollama_url=settings.ollama_base_url,
            collection_name=settings.qdrant_collection,
            embedding_model=get_model_setting("embedding_model", settings.embedding_model),
            embedding_dimension=settings.embedding_dimension,
            max_tokens=settings.embedding_max_tokens,
            tenant_id=settings.default_tenant_id,
        )
        await vector_service.initialize()

        rag_service = RAGService(settings, vector_service)

        # Get admin user's tags for proper access control
        # System admins bypass tag filtering during cache warming
        admin_user = db.query(User).filter(User.id == triggered_by).first()
        admin_role = normalize_role(admin_user.role) if admin_user else None
        if admin_role == ROLE_SYSTEM_ADMIN:
            # System admin bypasses tag filtering (None = no filtering)
            admin_tags = None
        else:
            admin_tags = [t.name for t in admin_user.tags] if admin_user and admin_user.tags else []

        # Resume from processed_index (supports crash recovery)
        for i in range(job.processed_index, job.total):
            # Check for paused status before processing each query
            current_job = queue_service.get_job(job_id)
            if current_job and current_job.status == "paused":
                logger.info(f"[WARM] Job {job_id} was paused, stopping processing")
                return

            query = job.queries[i]
            try:
                rag_request = RAGRequest(
                    query=query,
                    user_tags=admin_tags,
                    tenant_id="default",
                )
                await rag_service.generate(rag_request, db)  # Triggers caching

                # Record success
                job.results.append(
                    {
                        "query": query,
                        "status": "success",
                        "cached": True,
                        "error": None,
                    }
                )
                job.success_count += 1
                logger.info(f"[WARM] Completed query {i + 1}: {query[:50]}...")

            except Exception as e:
                # Record failure by index
                error_msg = str(e)[:200]
                job.results.append(
                    {
                        "query": query,
                        "status": "failed",
                        "cached": False,
                        "error": error_msg,
                    }
                )
                job.failed_indices.append(i)
                logger.warning(f"Failed to warm cache for query '{query[:50]}...': {e}")

            # Update processed index and persist to disk
            job.processed_index = i + 1

            # Checkpoint progress to disk (crash recovery)
            if (i + 1) % queue_service.checkpoint_interval == 0 or i == job.total - 1:
                queue_service.update_job(job)

            # Throttle to reduce Ollama contention
            if i < job.total - 1 and settings.warming_delay_seconds > 0:
                await asyncio.sleep(settings.warming_delay_seconds)

        # Complete the job
        queue_service.complete_job(job)

        logger.info(
            f"File cache warming complete: {job.success_count}/{job.total} queries "
            f"(job_id={job_id}, triggered_by={triggered_by})"
        )

        # Wait for SSE to pick up completion before deleting
        # SSE polls every 0.5s, so 2s should be plenty
        await asyncio.sleep(2.0)

        # Delete job file on success (or archive if configured)
        queue_service.delete_job(job_id)

    except Exception as e:
        import traceback

        traceback.print_exc()
        queue_service.fail_job(job, str(e))
        logger.error(f"Cache warming job {job_id} failed: {e}")
    finally:
        db.close()


# =============================================================================
# Synonym CRUD Endpoints - Query Synonym Management
# =============================================================================


class SynonymCreate(BaseModel):
    """Request model for creating a synonym."""

    term: str
    synonyms: list[str]


class SynonymUpdate(BaseModel):
    """Request model for updating a synonym."""

    term: str | None = None
    synonyms: list[str] | None = None
    enabled: bool | None = None


class SynonymResponse(BaseModel):
    """Response model for a synonym."""

    id: str
    term: str
    synonyms: list[str]
    enabled: bool
    created_by: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SynonymListResponse(BaseModel):
    """Response model for paginated synonym list."""

    synonyms: list[SynonymResponse]
    total: int
    page: int
    page_size: int


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

    synonym = QuerySynonym(
        term=synonym_data.term,
        synonyms=json.dumps(synonym_data.synonyms),
        enabled=True,
        created_by=current_user.id,
    )
    db.add(synonym)
    db.commit()
    db.refresh(synonym)

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
