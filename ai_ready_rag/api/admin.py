"""Admin endpoints for system management."""

import logging
import shutil
import subprocess
import time
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.dependencies import require_admin, require_system_admin
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import Document, User
from ai_ready_rag.services.document_service import DocumentService
from ai_ready_rag.services.model_service import ModelService, OllamaUnavailableError
from ai_ready_rag.services.settings_service import SettingsService
from ai_ready_rag.services.vector_service import VectorService

logger = logging.getLogger(__name__)
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
    vector_service = VectorService(
        qdrant_url=settings.qdrant_url,
        ollama_url=settings.ollama_base_url,
        collection_name=settings.qdrant_collection,
        embedding_model=settings.embedding_model,
    )

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
    vector_service = VectorService(
        qdrant_url=settings.qdrant_url,
        ollama_url=settings.ollama_base_url,
        collection_name=settings.qdrant_collection,
        embedding_model=settings.embedding_model,
    )

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

    # Optionally delete source files from SQLite
    deleted_files = 0
    if request.delete_source_files and success:
        document_service = DocumentService(db, settings)
        deleted_files = document_service.delete_all_documents()
        logger.warning(f"Deleted {deleted_files} documents from database")

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
    vector_service = VectorService(
        qdrant_url=settings.qdrant_url,
        ollama_url=settings.ollama_base_url,
        collection_name=settings.qdrant_collection,
        embedding_model=settings.embedding_model,
    )

    try:
        health = await vector_service.health_check()
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
            model=settings.embedding_model,
            dimensions=settings.embedding_dimension,
            vector_store=settings.vector_backend,
            vector_store_url=settings.qdrant_url,
        ),
        chat_model=ChatModelInfo(
            name=settings.chat_model,
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
    vector_service = VectorService(
        qdrant_url=settings.qdrant_url,
        ollama_url=settings.ollama_base_url,
        collection_name=settings.qdrant_collection,
        embedding_model=settings.embedding_model,
    )

    ollama_healthy = False
    vector_healthy = False
    total_chunks = 0
    storage_size_mb = None

    try:
        health = await vector_service.health_check()
        ollama_healthy = health.ollama_healthy
        vector_healthy = health.qdrant_healthy
    except Exception as e:
        logger.warning(f"Health check failed: {e}")

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
            details={
                "model": settings.chat_model,
                "url": settings.ollama_base_url,
            },
        ),
        vector_db=ComponentHealth(
            name=settings.vector_backend.capitalize(),
            status="healthy" if vector_healthy else "unhealthy",
            details={
                "collection": settings.qdrant_collection,
                "chunks": total_chunks,
            },
        ),
        rag_pipeline=RAGPipelineStatus(
            embedding_model=settings.embedding_model,
            chat_model=settings.chat_model,
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
