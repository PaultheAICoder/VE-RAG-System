"""Authentication endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from datetime import datetime

from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import User
from ai_ready_rag.core.security import verify_password, create_access_token, hash_password
from ai_ready_rag.core.dependencies import get_current_user
from ai_ready_rag.config import get_settings

router = APIRouter()
settings = get_settings()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    role: str
    is_active: bool

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    credentials: LoginRequest,
    response: Response,
    db: Session = Depends(get_db)
):
    """Authenticate user and return JWT token."""
    user = db.query(User).filter(User.email == credentials.email).first()

    if not user or not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )

    # Create token
    token = create_access_token(data={"sub": user.id, "email": user.email, "role": user.role})
    expires_in = settings.jwt_expiration_hours * 3600

    # Update last login
    user.last_login = datetime.utcnow()
    user.login_count += 1
    db.commit()

    # Set cookie
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=expires_in
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": expires_in,
        "user": user
    }


@router.post("/logout")
async def logout(
    response: Response,
    current_user: User = Depends(get_current_user)
):
    """Logout and clear session cookie."""
    response.delete_cookie("access_token")
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current authenticated user."""
    return current_user


# Bootstrap endpoint for creating first admin (only works when no users exist)
class SetupRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: str


@router.post("/setup", response_model=UserResponse)
async def setup_admin(
    setup_data: SetupRequest,
    db: Session = Depends(get_db)
):
    """Create first admin user (only works when no users exist)."""
    existing = db.query(User).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Setup already completed. Users exist."
        )

    admin = User(
        email=setup_data.email,
        display_name=setup_data.display_name,
        password_hash=hash_password(setup_data.password),
        role="admin",
        is_active=True
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)

    return admin
