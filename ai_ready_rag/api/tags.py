"""Tag management endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional

from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import User, Tag
from ai_ready_rag.core.dependencies import get_current_user, require_admin

router = APIRouter()


class TagCreate(BaseModel):
    name: str
    display_name: str
    description: Optional[str] = None
    color: str = "#6B7280"
    owner_id: Optional[str] = None


class TagUpdate(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    owner_id: Optional[str] = None


class TagResponse(BaseModel):
    id: str
    name: str
    display_name: str
    description: Optional[str]
    color: str
    owner_id: Optional[str]
    is_system: bool

    class Config:
        from_attributes = True


@router.get("/", response_model=List[TagResponse])
async def list_tags(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all tags."""
    return db.query(Tag).all()


@router.post("/", response_model=TagResponse, status_code=status.HTTP_201_CREATED)
async def create_tag(
    tag_data: TagCreate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
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
        created_by=current_user.id
    )
    db.add(tag)
    db.commit()
    db.refresh(tag)
    return tag


@router.get("/{tag_id}", response_model=TagResponse)
async def get_tag(
    tag_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
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
    db: Session = Depends(get_db)
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
    tag_id: str,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
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
