"""Tag suggestion approval endpoints for auto-tagging workflow."""

import logging
from collections import defaultdict
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.dependencies import require_admin
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import Document, TagSuggestion, User
from ai_ready_rag.schemas.suggestion import (
    ApprovalResponse,
    ApprovalResult,
    ApproveSuggestionsRequest,
    BatchApproveRequest,
    TagSuggestionListResponse,
    TagSuggestionResponse,
)
from ai_ready_rag.services.auto_tagging.strategy import AutoTagStrategy
from ai_ready_rag.services.document_service import DocumentService
from ai_ready_rag.services.factory import get_vector_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/{document_id}/tag-suggestions",
    response_model=TagSuggestionListResponse,
)
async def list_tag_suggestions(
    document_id: str,
    status_filter: str | None = None,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """List tag suggestions for a document (admin only)."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    query = db.query(TagSuggestion).filter(TagSuggestion.document_id == document_id)
    if status_filter:
        query = query.filter(TagSuggestion.status == status_filter)
    query = query.order_by(TagSuggestion.created_at.desc())

    suggestions = query.all()
    return TagSuggestionListResponse(
        suggestions=[TagSuggestionResponse.model_validate(s) for s in suggestions],
        total=len(suggestions),
    )


def _approve_suggestion(
    suggestion: TagSuggestion,
    document: Document,
    doc_service: DocumentService,
    current_user: User,
    db: Session,
) -> ApprovalResult:
    """Approve a single suggestion: create tag, assign to document, update status."""
    if suggestion.status != "pending":
        return ApprovalResult(
            suggestion_id=suggestion.id,
            status="already_processed",
        )

    # Load strategy to pass to ensure_tag_exists
    settings = get_settings()
    strategy_path = f"{settings.auto_tagging_strategies_dir}/{suggestion.strategy_id}.yaml"
    try:
        strategy = AutoTagStrategy.load(strategy_path)
    except (FileNotFoundError, ValueError) as e:
        return ApprovalResult(
            suggestion_id=suggestion.id,
            status="failed",
            error=f"Failed to load strategy {suggestion.strategy_id}: {e}",
        )

    tag_obj = doc_service.ensure_tag_exists(
        tag_name=suggestion.tag_name,
        display_name=suggestion.display_name,
        namespace=suggestion.namespace,
        strategy=strategy,
        created_by=current_user.id,
    )
    if tag_obj is None:
        return ApprovalResult(
            suggestion_id=suggestion.id,
            status="failed",
            error="Tag creation not allowed (create_missing_tags is disabled or cardinality limit reached)",
        )

    # Add tag to document if not already present
    existing_tag_names = {t.name for t in document.tags}
    if tag_obj.name not in existing_tag_names:
        document.tags.append(tag_obj)

    suggestion.status = "approved"
    suggestion.reviewed_by = current_user.id
    suggestion.reviewed_at = datetime.utcnow()

    return ApprovalResult(
        suggestion_id=suggestion.id,
        status="approved",
    )


async def _update_vector_store_for_document(document: Document) -> None:
    """Update vector store tags for a single document if it has been processed."""
    if document.status == "ready" and document.chunk_count and document.chunk_count > 0:
        settings = get_settings()
        vector_service = get_vector_service(settings)
        tag_names = [t.name for t in document.tags]
        await vector_service.update_document_tags(document.id, tag_names)
        logger.info("Updated vector store tags for document %s", document.id)


@router.post(
    "/{document_id}/tag-suggestions/approve",
    response_model=ApprovalResponse,
)
async def approve_suggestions(
    document_id: str,
    request: ApproveSuggestionsRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Approve selected tag suggestions for a document (admin only)."""
    settings = get_settings()

    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    doc_service = DocumentService(db, settings)
    results: list[ApprovalResult] = []

    for suggestion_id in request.suggestion_ids:
        suggestion = db.query(TagSuggestion).filter(TagSuggestion.id == suggestion_id).first()
        if not suggestion:
            results.append(
                ApprovalResult(
                    suggestion_id=suggestion_id,
                    status="failed",
                    error="Suggestion not found",
                )
            )
            continue

        if suggestion.document_id != document_id:
            results.append(
                ApprovalResult(
                    suggestion_id=suggestion_id,
                    status="failed",
                    error="Suggestion does not belong to this document",
                )
            )
            continue

        result = _approve_suggestion(suggestion, document, doc_service, current_user, db)
        results.append(result)

    # Update vector store once for the document
    approved_count = sum(1 for r in results if r.status == "approved")
    if approved_count > 0:
        try:
            await _update_vector_store_for_document(document)
        except Exception as e:
            logger.error("Failed to update vector store for document %s: %s", document_id, e)
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update vector tags: {e}",
            ) from e

    db.commit()

    failed_count = sum(1 for r in results if r.status == "failed")
    return ApprovalResponse(
        results=results,
        processed_count=approved_count,
        failed_count=failed_count,
    )


@router.post(
    "/{document_id}/tag-suggestions/reject",
    response_model=ApprovalResponse,
)
async def reject_suggestions(
    document_id: str,
    request: ApproveSuggestionsRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Reject selected tag suggestions for a document (admin only)."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    results: list[ApprovalResult] = []
    for suggestion_id in request.suggestion_ids:
        suggestion = db.query(TagSuggestion).filter(TagSuggestion.id == suggestion_id).first()
        if not suggestion:
            results.append(
                ApprovalResult(
                    suggestion_id=suggestion_id,
                    status="failed",
                    error="Suggestion not found",
                )
            )
            continue

        if suggestion.document_id != document_id:
            results.append(
                ApprovalResult(
                    suggestion_id=suggestion_id,
                    status="failed",
                    error="Suggestion does not belong to this document",
                )
            )
            continue

        if suggestion.status != "pending":
            results.append(
                ApprovalResult(
                    suggestion_id=suggestion_id,
                    status="already_processed",
                )
            )
            continue

        suggestion.status = "rejected"
        suggestion.reviewed_by = current_user.id
        suggestion.reviewed_at = datetime.utcnow()

        results.append(
            ApprovalResult(
                suggestion_id=suggestion_id,
                status="rejected",
            )
        )

    db.commit()

    rejected_count = sum(1 for r in results if r.status == "rejected")
    failed_count = sum(1 for r in results if r.status == "failed")
    return ApprovalResponse(
        results=results,
        processed_count=rejected_count,
        failed_count=failed_count,
    )


@router.post(
    "/tag-suggestions/approve-batch",
    response_model=ApprovalResponse,
)
async def batch_approve_suggestions(
    request: BatchApproveRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Bulk approve tag suggestions across multiple documents (admin only)."""
    settings = get_settings()
    doc_service = DocumentService(db, settings)

    # Load all requested suggestions
    suggestions = db.query(TagSuggestion).filter(TagSuggestion.id.in_(request.suggestion_ids)).all()
    found_ids = {s.id for s in suggestions}

    results: list[ApprovalResult] = []

    # Report missing suggestions
    for sid in request.suggestion_ids:
        if sid not in found_ids:
            results.append(
                ApprovalResult(
                    suggestion_id=sid,
                    status="failed",
                    error="Suggestion not found",
                )
            )

    # Group suggestions by document_id
    by_document: dict[str, list[TagSuggestion]] = defaultdict(list)
    for suggestion in suggestions:
        by_document[suggestion.document_id].append(suggestion)

    # Process each document group
    documents_to_update: list[Document] = []
    for document_id, doc_suggestions in by_document.items():
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            for s in doc_suggestions:
                results.append(
                    ApprovalResult(
                        suggestion_id=s.id,
                        status="failed",
                        error="Document not found",
                    )
                )
            continue

        doc_approved = False
        for suggestion in doc_suggestions:
            result = _approve_suggestion(suggestion, document, doc_service, current_user, db)
            results.append(result)
            if result.status == "approved":
                doc_approved = True

        if doc_approved:
            documents_to_update.append(document)

    # Update vector store once per document
    for document in documents_to_update:
        try:
            await _update_vector_store_for_document(document)
        except Exception as e:
            logger.error("Failed to update vector store for document %s: %s", document.id, e)
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update vector tags: {e}",
            ) from e

    db.commit()

    approved_count = sum(1 for r in results if r.status == "approved")
    failed_count = sum(1 for r in results if r.status == "failed")
    return ApprovalResponse(
        results=results,
        processed_count=approved_count,
        failed_count=failed_count,
    )
