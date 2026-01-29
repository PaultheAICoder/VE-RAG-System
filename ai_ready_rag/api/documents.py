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


@router.get("/", response_model=DocumentListResponse)
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

    Removes file from storage and database record.
    """
    settings = get_settings()
    service = DocumentService(db, settings)

    success = await service.delete_document(document_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return None
