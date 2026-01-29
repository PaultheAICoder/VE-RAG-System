"""Admin endpoints for system management."""

import logging
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.dependencies import require_admin
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import User
from ai_ready_rag.services.document_service import DocumentService
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


@router.post("/documents/recover-stuck", response_model=RecoverResponse)
async def recover_stuck_documents(
    max_age_hours: int = Query(2, ge=1, le=168, description="Maximum age in hours"),
    current_user: User = Depends(require_admin),
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
    current_user: User = Depends(require_admin),
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
    current_user: User = Depends(require_admin),
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


class ProcessingOptionsResponse(BaseModel):
    """Response model for processing options."""

    enable_ocr: bool
    force_full_page_ocr: bool
    ocr_language: str
    table_extraction_mode: str
    include_image_descriptions: bool


# Default values for processing options
PROCESSING_DEFAULTS = {
    "enable_ocr": True,
    "force_full_page_ocr": False,
    "ocr_language": "eng",
    "table_extraction_mode": "accurate",
    "include_image_descriptions": True,
}


def _get_setting_value(service: SettingsService, key: str, default: any) -> any:
    """Get setting value with fallback to default if None."""
    value = service.get(key)
    return value if value is not None else default


@router.get("/processing-options", response_model=ProcessingOptionsResponse)
async def get_processing_options(
    current_user: User = Depends(require_admin),
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
    )


@router.patch("/processing-options", response_model=ProcessingOptionsResponse)
async def update_processing_options(
    options: ProcessingOptionsRequest,
    current_user: User = Depends(require_admin),
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
    )
