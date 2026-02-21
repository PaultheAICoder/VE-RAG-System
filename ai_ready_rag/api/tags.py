"""Tag management endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ai_ready_rag.core.dependencies import get_current_user, require_admin
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import Tag, User
from ai_ready_rag.schemas.tag import (
    DeleteAllTagsRequest,
    DeleteAllTagsResponse,
    TagCreate,
    TagFacetsResponse,
    TagResponse,
    TagUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_model=list[TagResponse])
async def list_tags(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all tags."""
    return db.query(Tag).all()


@router.post("", response_model=TagResponse, status_code=status.HTTP_201_CREATED)
async def create_tag(
    tag_data: TagCreate, current_user: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Create a new tag (admin only)."""
    existing = db.query(Tag).filter(Tag.name == tag_data.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Tag name already exists")

    tag = Tag(
        name=tag_data.name,
        display_name=tag_data.display_name,
        description=tag_data.description,
        color=tag_data.color,
        owner_id=tag_data.owner_id,
        created_by=current_user.id,
    )
    db.add(tag)
    db.commit()
    db.refresh(tag)
    return tag


@router.get("/facets", response_model=TagFacetsResponse)
async def get_tag_facets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get tag facets grouped by namespace with document counts."""
    from ai_ready_rag.config import get_settings
    from ai_ready_rag.services.document_service import DocumentService

    settings = get_settings()
    service = DocumentService(db, settings)
    return service.get_tag_facets()


@router.get("/{tag_id}", response_model=TagResponse)
async def get_tag(
    tag_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Get tag by ID."""
    tag = db.query(Tag).filter(Tag.id == tag_id).first()
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")
    return tag


@router.put("/{tag_id}", response_model=TagResponse)
async def update_tag(
    tag_id: str,
    tag_data: TagUpdate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Update tag (admin only)."""
    tag = db.query(Tag).filter(Tag.id == tag_id).first()
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    if tag.is_system:
        raise HTTPException(status_code=400, detail="Cannot modify system tags")

    for field, value in tag_data.model_dump(exclude_unset=True).items():
        setattr(tag, field, value)

    db.commit()
    db.refresh(tag)
    return tag


@router.delete("/{tag_id}")
async def delete_tag(
    tag_id: str, current_user: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Delete tag (admin only)."""
    tag = db.query(Tag).filter(Tag.id == tag_id).first()
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    if tag.is_system:
        raise HTTPException(status_code=400, detail="Cannot delete system tags")

    db.delete(tag)
    db.commit()
    return {"message": "Tag deleted"}


@router.delete("", response_model=DeleteAllTagsResponse)
async def delete_all_tags(
    request: DeleteAllTagsRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Delete ALL non-system tags (admin only).

    System tags are preserved. Document and user tag associations are
    removed via CASCADE on the junction tables.
    Requires confirm: true in request body.
    """
    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation required. Set 'confirm: true' to proceed.",
        )

    # Count system tags (preserved)
    system_count = db.query(Tag).filter(Tag.is_system == True).count()  # noqa: E712

    # Get non-system tags and delete them
    non_system_tags = db.query(Tag).filter(Tag.is_system == False).all()  # noqa: E712
    deleted_count = len(non_system_tags)

    for tag in non_system_tags:
        db.delete(tag)

    db.commit()
    logger.info(
        f"Deleted all non-system tags (count={deleted_count}, "
        f"system_preserved={system_count}, admin={current_user.email})"
    )

    return DeleteAllTagsResponse(
        deleted_count=deleted_count,
        skipped_system_count=system_count,
        success=True,
    )
