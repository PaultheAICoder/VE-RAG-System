"""Setup wizard API endpoints for first-run admin password change."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.dependencies import get_current_user
from ai_ready_rag.core.security import hash_password, verify_password
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import SystemSetup, User
from ai_ready_rag.schemas.setup import (
    CompleteSetupRequest,
    CompleteSetupResponse,
    SetupStatusResponse,
)

router = APIRouter()
settings = get_settings()


def get_or_create_setup(db: Session) -> SystemSetup:
    """Get or create the SystemSetup record."""
    setup = db.query(SystemSetup).first()
    if not setup:
        setup = SystemSetup(setup_complete=False, admin_password_changed=False)
        db.add(setup)
        db.commit()
        db.refresh(setup)
    return setup


def is_setup_required(db: Session) -> bool:
    """Check if setup is required (not complete and not bypassed)."""
    if settings.skip_setup_wizard:
        return False
    setup = get_or_create_setup(db)
    return not setup.setup_complete


@router.get("/status", response_model=SetupStatusResponse)
async def get_setup_status(db: Session = Depends(get_db)):
    """Check if system setup is required.

    This endpoint is PUBLIC - no authentication required.
    Returns whether the setup wizard needs to be completed.
    """
    setup = get_or_create_setup(db)
    setup_required = not setup.setup_complete and not settings.skip_setup_wizard

    return SetupStatusResponse(setup_complete=setup.setup_complete, setup_required=setup_required)


@router.post("/complete", response_model=CompleteSetupResponse)
async def complete_setup(
    request: CompleteSetupRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Complete the setup wizard by changing the admin password.

    Requires authentication as an admin user.
    """
    # Only admin can complete setup
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can complete system setup",
        )

    # Check if setup is already complete
    setup = get_or_create_setup(db)
    if setup.setup_complete:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Setup has already been completed",
        )

    # Verify current password
    if not verify_password(request.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect",
        )

    # Ensure new password is different from current
    if request.current_password == request.new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from current password",
        )

    # Update admin password
    current_user.password_hash = hash_password(request.new_password)
    current_user.must_reset_password = False

    # Mark setup as complete
    setup.setup_complete = True
    setup.admin_password_changed = True
    setup.setup_completed_at = datetime.utcnow()
    setup.setup_completed_by = current_user.id

    db.commit()

    return CompleteSetupResponse(
        success=True,
        message="Setup completed successfully. Default password has been changed.",
    )
