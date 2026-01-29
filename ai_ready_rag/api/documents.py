"""Document management endpoints."""

import logging
from datetime import datetime
from typing import Annotated

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
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.dependencies import get_current_user, require_admin
from ai_ready_rag.db.database import SessionLocal, get_db
from ai_ready_rag.db.models import Document, User
from ai_ready_rag.services.document_service import DocumentService

logger = logging.getLogger(__name__)
router = APIRouter()


async def process_document_task(document_id: str) -> None:
    """Background task to process a document.

    Creates its own db session to avoid session lifecycle issues.

    Args:
        document_id: Document ID to process.
    """
    from ai_ready_rag.services.processing_service import ProcessingService
    from ai_ready_rag.services.vector_service import VectorService

    settings = get_settings()
    db = SessionLocal()

    try:
        # Get the document
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            logger.error(f"Document {document_id} not found for processing")
            return

        # Create services
        vector_service = VectorService(
            qdrant_url=settings.qdrant_url,
            ollama_url=settings.ollama_base_url,
            collection_name=settings.qdrant_collection,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
        )
        await vector_service.initialize()

        processing_service = ProcessingService(
            vector_service=vector_service,
            settings=settings,
        )

        # Process the document
        result = await processing_service.process_document(document, db)

        if result.success:
            logger.info(
                f"Document {document_id} processed successfully: "
                f"{result.chunk_count} chunks in {result.processing_time_ms}ms"
            )
        else:
            logger.warning(f"Document {document_id} processing failed: {result.error_message}")

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


class TagInfo(BaseModel):
    id: str
    name: str
    display_name: str

    class Config:
        from_attributes = True


class TagUpdateRequest(BaseModel):
    tag_ids: list[str]


class ReprocessRequest(BaseModel):
    enable_ocr: bool = True
    force_ocr: bool = False
    ocr_language: str = "eng"


class DocumentResponse(BaseModel):
    id: str
    original_filename: str
    filename: str
    file_type: str
    file_size: int
    status: str
    title: str | None
    description: str | None
    chunk_count: int | None
    page_count: int | None
    word_count: int | None
    processing_time_ms: int | None
    error_message: str | None
    tags: list[TagInfo]
    uploaded_by: str
    uploaded_at: datetime
    processed_at: datetime | None

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    limit: int
    offset: int


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    tag_ids: Annotated[list[str], Form()],
    background_tasks: BackgroundTasks,
    title: str | None = Form(None),
    description: str | None = Form(None),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Upload a document with tag assignment.

    Requires admin role. File is stored and queued for background processing.
    Returns immediately with status='pending'. Processing runs in background.
    """
    settings = get_settings()
    service = DocumentService(db, settings)

    document = await service.upload(
        file=file,
        tag_ids=tag_ids,
        uploaded_by=current_user.id,
        title=title,
        description=description,
    )

    # Queue background processing
    background_tasks.add_task(process_document_task, document.id)
    logger.info(f"Queued document {document.id} for background processing")

    return document


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None, description="Filter by status"),
    tag_id: str | None = Query(None, description="Filter by tag ID"),
    search: str | None = Query(None, description="Search in filename/title"),
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

    documents, total = service.list_documents(
        user_id=current_user.id,
        user_tags=user_tags,
        is_admin=is_admin,
        limit=limit,
        offset=offset,
        status_filter=status,
        tag_id=tag_id,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
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

    document = service.get_document(
        document_id=document_id,
        user_id=current_user.id,
        user_tags=user_tags,
        is_admin=is_admin,
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

    Cascade delete: vectors from Qdrant, file from storage, record from database.
    """
    from ai_ready_rag.services.vector_service import VectorService

    settings = get_settings()
    service = DocumentService(db, settings)

    # Check document exists first
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Delete vectors from Qdrant first
    try:
        vector_service = VectorService(
            qdrant_url=settings.qdrant_url,
            ollama_url=settings.ollama_base_url,
            collection_name=settings.qdrant_collection,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
        )
        await vector_service.delete_document(document_id)
        logger.info(f"Deleted vectors for document {document_id}")
    except Exception as e:
        logger.warning(f"Failed to delete vectors for document {document_id}: {e}")
        # Continue with file/db deletion even if vector deletion fails

    # Delete file and database record
    await service.delete_document(document_id)
    logger.info(f"Deleted document {document_id}")

    return None


class BulkDeleteRequest(BaseModel):
    """Request body for bulk delete."""

    document_ids: list[str]


class BulkDeleteResult(BaseModel):
    """Result for a single document deletion."""

    id: str
    status: str  # "deleted" or "failed"
    error: str | None = None


class BulkDeleteResponse(BaseModel):
    """Response for bulk delete operation."""

    results: list[BulkDeleteResult]
    deleted_count: int
    failed_count: int


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
    from ai_ready_rag.services.vector_service import VectorService

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

            # Delete vectors from Qdrant
            try:
                vector_service = VectorService(
                    qdrant_url=settings.qdrant_url,
                    ollama_url=settings.ollama_base_url,
                    collection_name=settings.qdrant_collection,
                    embedding_model=settings.embedding_model,
                    embedding_dimension=settings.embedding_dimension,
                )
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

    Updates tags in both SQLite and Qdrant (via set_payload, no re-embedding).
    """
    from ai_ready_rag.db.models import Tag
    from ai_ready_rag.services.vector_service import VectorService

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

    # Update Qdrant (only if document has been processed)
    if document.status == "ready" and document.chunk_count and document.chunk_count > 0:
        try:
            vector_service = VectorService(
                qdrant_url=settings.qdrant_url,
                ollama_url=settings.ollama_base_url,
                collection_name=settings.qdrant_collection,
                embedding_model=settings.embedding_model,
                embedding_dimension=settings.embedding_dimension,
            )
            tag_names = [t.name for t in tags]
            await vector_service.update_document_tags(document_id, tag_names)
            logger.info(f"Updated Qdrant tags for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to update Qdrant tags for document {document_id}: {e}")
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

    Deletes existing vectors, resets status to pending, and queues for reprocessing.
    Only allowed for documents in 'ready' or 'failed' status.
    """
    from ai_ready_rag.services.vector_service import VectorService

    settings = get_settings()

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

    # Delete existing vectors first (if any)
    if document.chunk_count and document.chunk_count > 0:
        try:
            vector_service = VectorService(
                qdrant_url=settings.qdrant_url,
                ollama_url=settings.ollama_base_url,
                collection_name=settings.qdrant_collection,
                embedding_model=settings.embedding_model,
                embedding_dimension=settings.embedding_dimension,
            )
            await vector_service.delete_document(document_id)
            logger.info(f"Deleted existing vectors for document {document_id}")
        except Exception as e:
            logger.warning(f"Failed to delete vectors for document {document_id}: {e}")
            # Continue anyway - vectors may not exist or Qdrant may be down

    # Reset document state
    document.status = "pending"
    document.chunk_count = None
    document.processed_at = None
    document.error_message = None
    document.processing_time_ms = None

    db.commit()
    db.refresh(document)

    # Queue for background processing
    background_tasks.add_task(process_document_task, document.id)
    logger.info(f"Queued document {document_id} for reprocessing")

    return document
