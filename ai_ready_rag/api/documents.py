"""Document management endpoints."""

import asyncio
import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.dependencies import get_current_user, require_admin
from ai_ready_rag.core.redis import get_redis_pool
from ai_ready_rag.db.database import SessionLocal, get_db
from ai_ready_rag.db.models import Document, User
from ai_ready_rag.schemas.document import (
    BatchFileResult,
    BatchUploadResponse,
    BulkDeleteRequest,
    BulkDeleteResponse,
    BulkDeleteResult,
    BulkReprocessRequest,
    BulkReprocessResponse,
    CheckDuplicatesRequest,
    CheckDuplicatesResponse,
    DocumentListResponse,
    DocumentResponse,
    DuplicateInfo,
    ReprocessRequest,
    TagUpdateRequest,
)
from ai_ready_rag.services.document_service import DocumentService

logger = logging.getLogger(__name__)
router = APIRouter()

# Semaphore to limit concurrent document processing
# Initialized lazily to use profile-specific settings
_processing_semaphore: asyncio.Semaphore | None = None


def get_processing_semaphore() -> asyncio.Semaphore:
    """Get or create the processing semaphore with profile-specific limit."""
    global _processing_semaphore
    if _processing_semaphore is None:
        settings = get_settings()
        _processing_semaphore = asyncio.Semaphore(settings.max_concurrent_processing)
        logger.info(
            f"Processing semaphore initialized with limit: {settings.max_concurrent_processing}"
        )
    return _processing_semaphore


async def enqueue_document_processing(
    document_id: str,
    background_tasks: BackgroundTasks,
    processing_options_dict: dict | None = None,
    delete_existing: bool = False,
) -> str | None:
    """Enqueue document processing via ARQ if available, else BackgroundTasks.

    Returns the ARQ job_id if enqueued via ARQ, None if using BackgroundTasks fallback.
    """
    settings = get_settings()
    redis = await get_redis_pool() if settings.use_arq_worker else None
    if redis:
        try:
            job = await redis.enqueue_job(
                "process_document",
                document_id,
                processing_options_dict,
                delete_existing,
            )
            logger.info(f"Enqueued document {document_id} via ARQ (job: {job.job_id})")
            return job.job_id
        except Exception as e:
            logger.warning(f"ARQ enqueue failed, falling back to BackgroundTasks: {e}")

    # Degraded mode: fall back to in-process BackgroundTasks
    background_tasks.add_task(
        process_document_task, document_id, processing_options_dict, delete_existing
    )
    logger.info(f"Queued document {document_id} via BackgroundTasks (degraded mode)")
    return None


async def process_document_task(
    document_id: str,
    processing_options_dict: dict | None = None,
    delete_existing: bool = False,
) -> None:
    """Background task to process a document.

    Creates its own db session to avoid session lifecycle issues.
    Uses a semaphore to limit concurrent processing based on hardware profile.

    Args:
        document_id: Document ID to process.
        processing_options_dict: Optional dict of per-upload processing options.
        delete_existing: If True, delete existing vectors before processing (for reprocess).
    """
    from ai_ready_rag.services.factory import get_vector_service
    from ai_ready_rag.services.processing_service import ProcessingOptions, ProcessingService

    logger.info("document_processing_started", extra={"document_id": document_id})

    # Acquire semaphore to limit concurrent processing
    semaphore = get_processing_semaphore()
    async with semaphore:
        settings = get_settings()
        db = SessionLocal()

        # Reconstruct ProcessingOptions if provided
        processing_options = None
        if processing_options_dict:
            processing_options = ProcessingOptions(**processing_options_dict)

        try:
            # Get the document
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                logger.error(f"Document {document_id} not found for processing")
                return

            # Create services using factory (respects vector_backend setting)
            vector_service = get_vector_service(settings)
            await vector_service.initialize()

            # Delete existing vectors if reprocessing
            if delete_existing:
                try:
                    await vector_service.delete_document(document_id)
                    logger.info(f"Deleted existing vectors for document {document_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete vectors for {document_id}: {e}")
                    # Continue anyway - vectors may not exist

            processing_service = ProcessingService(
                vector_service=vector_service,
                settings=settings,
            )

            # Process the document with optional per-upload options
            result = await processing_service.process_document(
                document, db, processing_options=processing_options
            )

            if result.success:
                logger.info(
                    "document_processing_completed",
                    extra={
                        "document_id": document_id,
                        "chunk_count": result.chunk_count,
                        "processing_time_ms": result.processing_time_ms,
                    },
                )
            else:
                logger.warning(
                    "document_processing_failed",
                    extra={
                        "document_id": document_id,
                        "error_message": result.error_message,
                    },
                )

        except Exception as e:
            logger.exception(f"Unexpected error processing document {document_id}: {e}")
            # Try to mark as failed
            try:
                document = db.query(Document).filter(Document.id == document_id).first()
                if document:
                    document.status = "failed"
                    document.error_message = f"Unexpected error: {e}"
                    db.commit()
            except Exception:
                logger.exception("Failed to update document status after error")
        finally:
            db.close()


@router.post("/check-duplicates", response_model=CheckDuplicatesResponse)
async def check_duplicates(
    request: CheckDuplicatesRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Pre-upload duplicate check by filename (admin only).

    Checks which filenames already exist in the database.
    """
    if not request.filenames:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one filename required",
        )

    settings = get_settings()
    service = DocumentService(db, settings)

    duplicates, unique = service.check_duplicates_by_filename(request.filenames)

    return CheckDuplicatesResponse(
        duplicates=[DuplicateInfo(**d) for d in duplicates],
        unique=unique,
    )


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    tag_ids: list[str] = Form(default=[]),
    title: str | None = Form(None),
    description: str | None = Form(None),
    enable_ocr: bool | None = Form(None),
    force_full_page_ocr: bool | None = Form(None),
    ocr_language: str | None = Form(None),
    table_extraction_mode: str | None = Form(None),
    include_image_descriptions: bool | None = Form(None),
    source_path: str | None = Form(None),
    auto_tag: bool | None = Form(None),
    replace: bool = Query(False, description="Replace existing duplicate if found"),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Upload a document with tag assignment.

    Requires admin role. File is stored and queued for background processing.
    Returns immediately with status='pending'. Processing runs in background.

    Optional processing options override global defaults for this upload:
    - enable_ocr: Enable OCR for scanned documents
    - force_full_page_ocr: Force full page OCR even for text-based PDFs
    - ocr_language: OCR language code (e.g., 'eng', 'fra')
    - table_extraction_mode: 'accurate' or 'fast'
    - include_image_descriptions: Include AI-generated image descriptions

    Query parameters:
    - replace: If true and a duplicate is found, replace the existing document
    """
    from ai_ready_rag.services.factory import get_vector_service
    from ai_ready_rag.services.processing_service import ProcessingOptions

    # Validate table_extraction_mode if provided
    if table_extraction_mode is not None and table_extraction_mode not in ("accurate", "fast"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="table_extraction_mode must be 'accurate' or 'fast'",
        )

    settings = get_settings()
    service = DocumentService(db, settings)

    # Get vector service for replace mode
    vector_service = None
    if replace:
        vector_service = get_vector_service(settings)

    document = await service.upload(
        file=file,
        tag_ids=tag_ids,
        uploaded_by=current_user.id,
        title=title,
        description=description,
        replace=replace,
        vector_service=vector_service,
        source_path=source_path,
        auto_tag=auto_tag,
    )

    # Build processing options if any are provided
    processing_options = None
    if any(
        opt is not None
        for opt in [
            enable_ocr,
            force_full_page_ocr,
            ocr_language,
            table_extraction_mode,
            include_image_descriptions,
        ]
    ):
        processing_options = ProcessingOptions(
            enable_ocr=enable_ocr,
            force_full_page_ocr=force_full_page_ocr,
            ocr_language=ocr_language,
            table_extraction_mode=table_extraction_mode,
            include_image_descriptions=include_image_descriptions,
        )

    # Serialize options for background task (if present)
    from dataclasses import asdict

    options_dict = asdict(processing_options) if processing_options else None

    # Queue background processing (ARQ if available, else BackgroundTasks)
    await enqueue_document_processing(document.id, background_tasks, options_dict)

    return document


@router.post("/upload/batch", response_model=BatchUploadResponse, status_code=status.HTTP_200_OK)
async def batch_upload_documents(
    files: list[UploadFile],
    background_tasks: BackgroundTasks,
    source_paths: list[str] = Form(default=[]),
    tag_ids: list[str] = Form(default=[]),
    auto_tag: bool | None = Form(None),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Upload multiple documents in a single batch (admin only).

    Supports partial success: each file is processed independently.
    Returns 200 with per-file results (not 201, since batch may have mixed outcomes).

    Idempotency: duplicate detection uses composite key of content_hash + source_path + strategy_id.
    Same content with different source_paths creates separate documents.
    """
    settings = get_settings()

    # Validate source_paths length
    if source_paths and len(source_paths) != len(files):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="source_paths length must match files length",
        )

    # Pad source_paths to None if empty
    if not source_paths:
        padded_source_paths: list[str | None] = [None] * len(files)
    else:
        padded_source_paths = source_paths  # type: ignore[assignment]

    service = DocumentService(db, settings)
    batch_result = await service.upload_batch(
        files=files,
        source_paths=padded_source_paths,
        tag_ids=tag_ids,
        uploaded_by=current_user.id,
        auto_tag=auto_tag,
    )

    # Enqueue background processing for each successfully uploaded file
    for file_result in batch_result["results"]:
        if file_result["status"] == "uploaded" and file_result["document_id"]:
            await enqueue_document_processing(file_result["document_id"], background_tasks)

    return BatchUploadResponse(
        total=batch_result["total"],
        uploaded=batch_result["uploaded"],
        duplicates=batch_result["duplicates"],
        failed=batch_result["failed"],
        auto_tags_applied=batch_result["auto_tags_applied"],
        results=[BatchFileResult(**r) for r in batch_result["results"]],
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None, description="Filter by status"),
    tag_id: str | None = Query(None, description="Filter by tag ID"),
    search: str | None = Query(None, description="Search in filename/title"),
    tag_namespace: str | None = Query(None, description="Filter by tag namespace"),
    tag_value: str | None = Query(None, description="Filter by tag value (use with tag_namespace)"),
    tag_prefix: str | None = Query(None, description="Filter by tag name prefix (e.g., 'client:')"),
    sort_by: str = Query("uploaded_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List documents with filtering and pagination.

    Admins see all documents. Users see only accessible documents.
    """
    settings = get_settings()
    service = DocumentService(db, settings)

    is_admin = current_user.role == "admin"
    user_tags = [tag.name for tag in current_user.tags]
    tag_access_enabled = getattr(current_user, "tag_access_enabled", True)

    documents, total = service.list_documents(
        user_id=current_user.id,
        user_tags=user_tags,
        is_admin=is_admin,
        tag_access_enabled=tag_access_enabled,
        limit=limit,
        offset=offset,
        status_filter=status,
        tag_id=tag_id,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
        tag_namespace=tag_namespace,
        tag_value=tag_value,
        tag_prefix=tag_prefix,
    )

    return DocumentListResponse(
        documents=documents,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get document details.

    Admins can view any document. Users can only view accessible documents.
    """
    settings = get_settings()
    service = DocumentService(db, settings)

    is_admin = current_user.role == "admin"
    user_tags = [tag.name for tag in current_user.tags]
    tag_access_enabled = getattr(current_user, "tag_access_enabled", True)

    document = service.get_document(
        document_id=document_id,
        user_id=current_user.id,
        user_tags=user_tags,
        is_admin=is_admin,
        tag_access_enabled=tag_access_enabled,
    )

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or not accessible",
        )

    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Delete a document (admin only).

    Cascade delete: vectors from vector store, file from storage, record from database.
    """
    from ai_ready_rag.services.factory import get_vector_service

    settings = get_settings()
    service = DocumentService(db, settings)

    # Check document exists first
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Delete vectors from vector store first
    try:
        vector_service = get_vector_service(settings)
        await vector_service.delete_document(document_id)
        logger.info(f"Deleted vectors for document {document_id}")
    except Exception as e:
        logger.warning(f"Failed to delete vectors for document {document_id}: {e}")
        # Continue with file/db deletion even if vector deletion fails

    # Delete file and database record
    await service.delete_document(document_id)
    logger.info(f"Deleted document {document_id}")

    return None


@router.post("/bulk-delete", response_model=BulkDeleteResponse)
async def bulk_delete_documents(
    request: BulkDeleteRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Delete multiple documents at once (admin only).

    Partial success allowed - some documents may fail while others succeed.
    Returns detailed results for each document.
    """
    from ai_ready_rag.services.factory import get_vector_service

    settings = get_settings()
    service = DocumentService(db, settings)

    results: list[BulkDeleteResult] = []
    deleted_count = 0
    failed_count = 0

    for doc_id in request.document_ids:
        try:
            # Check document exists
            document = db.query(Document).filter(Document.id == doc_id).first()
            if not document:
                results.append(
                    BulkDeleteResult(id=doc_id, status="failed", error="Document not found")
                )
                failed_count += 1
                continue

            # Delete vectors from vector store
            try:
                vector_service = get_vector_service(settings)
                await vector_service.delete_document(doc_id)
            except Exception as e:
                logger.warning(f"Failed to delete vectors for document {doc_id}: {e}")
                # Continue with deletion anyway

            # Delete file and database record
            await service.delete_document(doc_id)

            results.append(BulkDeleteResult(id=doc_id, status="deleted"))
            deleted_count += 1
            logger.info(f"Bulk delete: deleted document {doc_id}")

        except Exception as e:
            results.append(BulkDeleteResult(id=doc_id, status="failed", error=str(e)))
            failed_count += 1
            logger.error(f"Bulk delete: failed to delete document {doc_id}: {e}")

    return BulkDeleteResponse(
        results=results,
        deleted_count=deleted_count,
        failed_count=failed_count,
    )


@router.patch("/{document_id}/tags", response_model=DocumentResponse)
async def update_document_tags(
    document_id: str,
    request: TagUpdateRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Update document tags (admin only).

    Updates tags in both SQLite and vector store (via set_payload, no re-embedding).
    """
    from ai_ready_rag.db.models import Tag
    from ai_ready_rag.services.factory import get_vector_service

    settings = get_settings()

    # Validate at least one tag
    if not request.tag_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one tag is required",
        )

    # Get document
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Validate tags exist
    tags = db.query(Tag).filter(Tag.id.in_(request.tag_ids)).all()
    if len(tags) != len(request.tag_ids):
        found_ids = {t.id for t in tags}
        missing = [tid for tid in request.tag_ids if tid not in found_ids]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tag IDs: {missing}",
        )

    # Update SQLite
    document.tags = tags

    # Update vector store (only if document has been processed)
    if document.status == "ready" and document.chunk_count and document.chunk_count > 0:
        try:
            vector_service = get_vector_service(settings)
            tag_names = [t.name for t in tags]
            await vector_service.update_document_tags(document_id, tag_names)
            logger.info(f"Updated vector store tags for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to update vector store tags for document {document_id}: {e}")
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update vector tags: {e}",
            ) from e

    db.commit()
    db.refresh(document)
    logger.info(f"Updated tags for document {document_id}: {[t.name for t in tags]}")

    return document


@router.post(
    "/{document_id}/reprocess",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reprocess_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    request: ReprocessRequest | None = None,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Reprocess a document (admin only).

    Resets status to pending and queues for reprocessing.
    Vector deletion is handled in background task for fast response.
    Only allowed for documents in 'ready' or 'failed' status.
    """
    # Get document
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Validate status - can only reprocess ready or failed documents
    if document.status not in ("ready", "failed"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot reprocess document in '{document.status}' status. "
            "Only 'ready' or 'failed' documents can be reprocessed.",
        )

    # Track if we need to delete existing vectors (done in background)
    delete_existing = document.chunk_count and document.chunk_count > 0

    # Reset document state immediately for fast response
    document.status = "pending"
    document.chunk_count = None
    document.processed_at = None
    document.error_message = None
    document.processing_time_ms = None

    db.commit()
    db.refresh(document)

    # Queue for background processing (vector deletion + reprocessing)
    await enqueue_document_processing(document.id, background_tasks, None, delete_existing)
    logger.info(
        f"Queued document {document_id} for reprocessing (delete_existing={delete_existing})"
    )

    return document


@router.post(
    "/bulk-reprocess",
    response_model=BulkReprocessResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def bulk_reprocess_documents(
    request: BulkReprocessRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Bulk reprocess multiple documents (admin only).

    Resets all valid documents to pending and queues for background processing.
    Returns immediately - processing happens in background.
    """
    logger.info(
        "bulk_reprocess_started",
        extra={"requested_count": len(request.document_ids)},
    )

    queued = 0
    skipped_ids = []

    # Get all documents in one query
    documents = db.query(Document).filter(Document.id.in_(request.document_ids)).all()
    logger.debug(
        "bulk_reprocess_query",
        extra={"found_count": len(documents)},
    )

    doc_map = {doc.id: doc for doc in documents}

    for doc_id in request.document_ids:
        document = doc_map.get(doc_id)

        # Skip if not found
        if not document:
            logger.debug(
                "bulk_reprocess_skip", extra={"document_id": doc_id, "reason": "not_found"}
            )
            skipped_ids.append(doc_id)
            continue

        # Skip if already queued or actively processing (to avoid double-queueing)
        if document.status in ("processing", "pending"):
            logger.debug(
                "bulk_reprocess_skip",
                extra={"document_id": doc_id, "reason": "already_processing"},
            )
            skipped_ids.append(doc_id)
            continue

        logger.debug(
            "bulk_reprocess_queuing",
            extra={"document_id": doc_id, "previous_status": document.status},
        )

        # Track if we need to delete existing vectors (only for ready/failed with chunks)
        delete_existing = (
            document.status in ("ready", "failed")
            and document.chunk_count
            and document.chunk_count > 0
        )

        # Reset document state
        document.status = "pending"
        document.chunk_count = None
        document.processed_at = None
        document.error_message = None
        document.processing_time_ms = None

        # Queue for background processing
        await enqueue_document_processing(document.id, background_tasks, None, delete_existing)
        queued += 1

    db.commit()
    logger.info(
        "bulk_reprocess_completed",
        extra={"queued": queued, "skipped": len(skipped_ids)},
    )

    return BulkReprocessResponse(
        queued=queued,
        skipped=len(skipped_ids),
        skipped_ids=skipped_ids,
    )
