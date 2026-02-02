"""FastAPI dependencies for authentication and authorization."""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ai_ready_rag.core.security import decode_token
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import User

security = HTTPBearer(auto_error=False)

# Role constants
ROLE_SYSTEM_ADMIN = "system_admin"
ROLE_CUSTOMER_ADMIN = "customer_admin"
ROLE_USER = "user"

# Valid roles for validation
VALID_ROLES = {ROLE_SYSTEM_ADMIN, ROLE_CUSTOMER_ADMIN, ROLE_USER, "admin"}


def normalize_role(role: str) -> str:
    """Normalize role for backward compatibility.

    Maps legacy "admin" role to "system_admin".
    """
    if role == "admin":
        return ROLE_SYSTEM_ADMIN
    return role


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Get current user from JWT token (header or cookie)."""
    token = None

    # Try Authorization header first
    if credentials:
        token = credentials.credentials

    # Fallback to cookie
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user = db.query(User).filter(User.id == payload.get("sub")).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User is deactivated")

    return user


async def get_optional_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: Session = Depends(get_db),
) -> User | None:
    """Get current user if authenticated, None otherwise.

    Used for endpoints that need optional auth (e.g., SSE with query param token).
    """
    token = None

    # Try Authorization header first
    if credentials:
        token = credentials.credentials

    # Fallback to cookie
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        return None

    payload = decode_token(token)
    if not payload:
        return None

    user = db.query(User).filter(User.id == payload.get("sub")).first()
    if not user or not user.is_active:
        return None

    return user


async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin role (system_admin or customer_admin).

    Both system_admin and customer_admin can manage users, documents, and tags.
    """
    role = normalize_role(current_user.role)
    if role not in (ROLE_SYSTEM_ADMIN, ROLE_CUSTOMER_ADMIN):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


async def require_system_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require system_admin role only.

    Only system_admin can access system settings, architecture info,
    model configuration, and knowledge base management.
    """
    role = normalize_role(current_user.role)
    if role != ROLE_SYSTEM_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="System admin access required"
        )
    return current_user
