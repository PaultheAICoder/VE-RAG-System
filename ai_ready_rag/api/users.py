"""User management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ai_ready_rag.core.dependencies import require_admin
from ai_ready_rag.core.security import generate_temporary_password, hash_password
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import Tag, User
from ai_ready_rag.schemas.user import (
    BulkAutoTagAssignment,
    TagAssignment,
    UserCreate,
    UserResponse,
    UserUpdate,
)

router = APIRouter()


@router.get("", response_model=list[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """List all users (admin only)."""
    users = db.query(User).offset(skip).limit(limit).all()
    return [
        {
            **user.__dict__,
            "tags": [
                {"id": t.id, "name": t.name, "display_name": t.display_name} for t in user.tags
            ],
        }
        for user in users
    ]


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a new user (admin only)."""
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=user_data.email,
        display_name=user_data.display_name,
        password_hash=hash_password(user_data.password),
        role=user_data.role,
        created_by=current_user.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {**user.__dict__, "tags": []}


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str, current_user: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Get user by ID (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        **user.__dict__,
        "tags": [{"id": t.id, "name": t.name, "display_name": t.display_name} for t in user.tags],
    }


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Update user (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    for field, value in user_data.model_dump(exclude_unset=True).items():
        setattr(user, field, value)

    db.commit()
    db.refresh(user)
    return {
        **user.__dict__,
        "tags": [{"id": t.id, "name": t.name, "display_name": t.display_name} for t in user.tags],
    }


@router.delete("/{user_id}")
async def deactivate_user(
    user_id: str, current_user: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Deactivate user (admin only)."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_active = False
    db.commit()
    return {"message": "User deactivated"}


@router.post("/{user_id}/tags")
async def assign_tags(
    user_id: str,
    assignment: TagAssignment,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Assign tags to user (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    tags = db.query(Tag).filter(Tag.id.in_(assignment.tag_ids)).all()
    user.tags = tags
    db.commit()

    return {
        "message": f"Assigned {len(tags)} tags to user",
        "tags": [{"id": t.id, "name": t.name} for t in tags],
    }


@router.post("/{user_id}/tags/auto")
async def bulk_auto_assign_tags(
    user_id: str,
    assignment: BulkAutoTagAssignment,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Bulk assign auto-tags to user by client names (admin only).

    Looks up tags by client names (with 'client:' prefix), optionally
    includes doctype: and entity: tags, then assigns all matching tags
    to the user (merged with existing tags).
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Build list of tag name patterns to match
    tag_patterns = [f"client:{name}" for name in assignment.client_names]

    if assignment.include_doctypes:
        tag_patterns.extend(
            [t.name for t in db.query(Tag).filter(Tag.name.like("doctype:%")).all()]
        )

    if assignment.include_entities:
        tag_patterns.extend([t.name for t in db.query(Tag).filter(Tag.name.like("entity:%")).all()])

    # Find matching tags
    matching_tags = db.query(Tag).filter(Tag.name.in_(tag_patterns)).all()

    # Merge with existing tags (don't replace)
    existing_tag_ids = {t.id for t in user.tags}
    for tag in matching_tags:
        if tag.id not in existing_tag_ids:
            user.tags.append(tag)

    db.commit()
    db.refresh(user)

    return {
        "message": f"Assigned {len(matching_tags)} auto-tags to user",
        "requested": len(assignment.client_names),
        "matched": len(matching_tags),
        "tags": [{"id": t.id, "name": t.name, "display_name": t.display_name} for t in user.tags],
    }


@router.post("/{user_id}/reset-password")
async def reset_password(
    user_id: str, current_user: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Reset user password (admin only). Returns temporary password."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    temp_password = generate_temporary_password()
    user.password_hash = hash_password(temp_password)
    user.must_reset_password = True
    db.commit()

    return {
        "temporary_password": temp_password,
        "message": "User must change password on next login",
    }
