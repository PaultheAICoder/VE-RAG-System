"""Authentication endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from ai_ready_rag.api.setup import is_setup_required
from ai_ready_rag.config import get_settings
from ai_ready_rag.core.dependencies import get_current_user
from ai_ready_rag.core.security import create_access_token, hash_password, verify_password
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import User
from ai_ready_rag.schemas.auth import (
    LoginRequest,
    LoginResponse,
    SetupRequest,
    UserBasicResponse,
)
from ai_ready_rag.services.settings_service import get_security_setting

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request, credentials: LoginRequest, response: Response, db: Session = Depends(get_db)
):
    """Authenticate user and return JWT token."""
    user = db.query(User).filter(User.email == credentials.email).first()

    if not user or not verify_password(credentials.password, user.password_hash):
        logger.warning(
            "login_failed",
            extra={
                "email": credentials.email,
                "reason": "invalid_credentials",
                "client_ip": request.client.host if request.client else None,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )

    if not user.is_active:
        logger.warning(
            "login_failed",
            extra={
                "email": credentials.email,
                "reason": "account_deactivated",
                "user_id": user.id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="User account is deactivated"
        )

    # Create token
    token = create_access_token(data={"sub": user.id, "email": user.email, "role": user.role})
    jwt_hours = get_security_setting("jwt_expiration_hours", settings.jwt_expiration_hours)
    expires_in = jwt_hours * 3600

    # Update last login
    user.last_login = datetime.utcnow()
    user.login_count = (user.login_count or 0) + 1
    db.commit()

    logger.info(
        "login_success",
        extra={"user_id": user.id, "email": user.email, "role": user.role},
    )

    # Set cookie
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=expires_in,
    )

    # Check if setup is required (only for admin users)
    setup_required_flag = False
    if user.role == "admin":
        setup_required_flag = is_setup_required(db)

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": expires_in,
        "user": user,
        "setup_required": setup_required_flag,
    }


@router.post("/logout")
async def logout(response: Response, current_user: User = Depends(get_current_user)):
    """Logout and clear session cookie."""
    response.delete_cookie("access_token")
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserBasicResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current authenticated user."""
    return current_user


# Bootstrap endpoint for creating first admin (only works when no users exist)
@router.post("/setup", response_model=UserBasicResponse)
async def setup_admin(setup_data: SetupRequest, db: Session = Depends(get_db)):
    """Create first admin user (only works when no users exist)."""
    existing = db.query(User).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Setup already completed. Users exist."
        )

    admin = User(
        email=setup_data.email,
        display_name=setup_data.display_name,
        password_hash=hash_password(setup_data.password),
        role="admin",
        is_active=True,
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)

    return admin
