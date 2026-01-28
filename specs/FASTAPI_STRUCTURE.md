# FastAPI Backend Structure Specification

**Version:** 1.0
**Date:** January 27, 2026
**Status:** Draft
**Based On:** DEVELOPMENT_PLANS.md v0.4.2, PRD_v0.80.md, ARCHITECTURE.md

---

## Overview

This specification defines the FastAPI backend structure for AI Ready RAG. The backend replaces the standalone Gradio app with a proper enterprise architecture supporting authentication, RBAC, audit logging, and REST APIs.

**Key Principle:** Auth, tags, and API testing can run without Ollama/LLM dependencies.

---

## Project Structure

```
ai_ready_rag/
├── main.py                     # FastAPI application entry point
├── config.py                   # Configuration management (env vars, settings)
├── version.py                  # Version info for footer/API
│
├── api/                        # Route handlers (thin layer, delegates to services)
│   ├── __init__.py
│   ├── auth.py                 # POST /login, /logout, GET /me
│   ├── users.py                # CRUD /users, /users/{id}/tags, /users/{id}/reset-password
│   ├── tags.py                 # CRUD /tags
│   ├── documents.py            # Upload, list, delete, tag assignment
│   ├── chat.py                 # Sessions, messages, RAG integration
│   ├── admin.py                # Audit logs, stats, routed questions
│   └── health.py               # GET /health, /version
│
├── core/                       # Cross-cutting concerns
│   ├── __init__.py
│   ├── security.py             # JWT encode/decode, password hashing (bcrypt)
│   ├── dependencies.py         # FastAPI Depends() - get_current_user, get_db, require_admin
│   └── exceptions.py           # Custom HTTP exceptions, error handlers
│
├── middleware/                 # Request/response interceptors
│   ├── __init__.py
│   ├── auth.py                 # JWT validation middleware
│   ├── audit.py                # Audit logging middleware
│   └── access_control.py       # Tag-based access checks (future)
│
├── db/                         # Database layer
│   ├── __init__.py
│   ├── database.py             # SQLite connection, engine, session factory
│   ├── models.py               # SQLAlchemy ORM models
│   ├── schemas.py              # Pydantic schemas (request/response DTOs)
│   └── migrations/             # Schema migrations (if needed)
│       └── init_schema.sql
│
├── services/                   # Business logic (testable, no HTTP concerns)
│   ├── __init__.py
│   ├── auth_service.py         # Login, logout, token refresh, password reset
│   ├── user_service.py         # User CRUD, tag assignment
│   ├── tag_service.py          # Tag CRUD
│   ├── document_service.py     # Document upload, processing, deletion
│   ├── chat_service.py         # Session management, message persistence
│   ├── vector_service.py       # Qdrant operations (abstracted)
│   ├── rag_service.py          # RAG pipeline (optional for auth testing)
│   └── audit_service.py        # Audit log writing, querying
│
├── ui/                         # Gradio integration
│   ├── __init__.py
│   ├── gradio_app.py           # Gradio Blocks definition
│   ├── setup_wizard.py         # First-time admin setup
│   └── components/             # Reusable UI components
│       └── __init__.py
│
└── utils/                      # Helpers
    ├── __init__.py
    └── helpers.py              # ID generation, timestamps, etc.
```

---

## Configuration (`config.py`)

```python
from pydantic_settings import BaseSettings
from typing import Literal
from functools import lru_cache

class Settings(BaseSettings):
    # Application
    app_name: str = "AI Ready RAG"
    app_version: str = "0.5.0"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "sqlite:///./data/ai_ready_rag.db"

    # JWT
    jwt_secret_key: str  # Required, no default
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    jwt_refresh_expiration_days: int = 7

    # Security
    password_min_length: int = 12
    lockout_attempts: int = 5
    lockout_minutes: int = 15
    bcrypt_rounds: int = 12

    # Audit
    audit_level: Literal["essential", "comprehensive", "full_debug"] = "full_debug"
    audit_retention_days: int = 90

    # External Services (optional for auth testing)
    ollama_base_url: str = "http://localhost:11434"
    qdrant_url: str = "http://localhost:6333"

    # Feature Flags
    enable_rag: bool = True  # Set False to test auth without LLM
    enable_gradio: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

**Environment Variables (`.env`):**
```bash
JWT_SECRET_KEY=your-256-bit-secret-key-here
DATABASE_URL=sqlite:///./data/ai_ready_rag.db
AUDIT_LEVEL=full_debug
ENABLE_RAG=false  # For auth-only testing
DEBUG=true
```

---

## Main Application (`main.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ai_ready_rag.config import get_settings
from ai_ready_rag.db.database import init_db
from ai_ready_rag.api import auth, users, tags, documents, chat, admin, health
from ai_ready_rag.middleware.audit import AuditLogMiddleware
from ai_ready_rag.core.exceptions import register_exception_handlers

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    yield
    # Shutdown
    pass

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/api/docs" if settings.debug else None,
    redoc_url="/api/redoc" if settings.debug else None,
)

# Middleware (order matters: last added = first executed)
app.add_middleware(AuditLogMiddleware, audit_level=settings.audit_level)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
register_exception_handlers(app)

# API Routes
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(tags.router, prefix="/api/tags", tags=["Tags"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])

# Mount Gradio (optional)
if settings.enable_gradio:
    import gradio as gr
    from ai_ready_rag.ui.gradio_app import create_gradio_app
    gradio_app = create_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/app")

# Setup wizard redirect
@app.get("/")
async def root():
    from fastapi.responses import RedirectResponse
    # Check if any users exist, if not redirect to setup
    # For now, redirect to /app or /api/docs
    return RedirectResponse(url="/app")
```

---

## Database Models (`db/models.py`)

```python
from sqlalchemy import (
    Column, String, Boolean, Integer, Float, Text,
    ForeignKey, DateTime, Index, func
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

def generate_uuid() -> str:
    return str(uuid.uuid4())

# ============== Users ==============

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    display_name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default="user")  # "admin" | "user"
    is_active = Column(Boolean, default=True)
    must_reset_password = Column(Boolean, default=False)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, ForeignKey("users.id"), nullable=True)
    last_login = Column(DateTime, nullable=True)
    login_count = Column(Integer, default=0)

    # Relationships
    tags = relationship("Tag", secondary="user_tags", back_populates="users")
    sessions = relationship("UserSession", back_populates="user")
    chat_sessions = relationship("ChatSession", back_populates="user")

class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    token_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)

    user = relationship("User", back_populates="sessions")

    __table_args__ = (
        Index("idx_sessions_user", "user_id"),
        Index("idx_sessions_expires", "expires_at"),
    )

# ============== Tags ==============

class Tag(Base):
    __tablename__ = "tags"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, unique=True, nullable=False)  # slug: "hr", "finance"
    display_name = Column(String, nullable=False)       # "Human Resources"
    description = Column(Text, nullable=True)
    color = Column(String, default="#6B7280")
    owner_id = Column(String, ForeignKey("users.id"), nullable=True)
    is_system = Column(Boolean, default=False)  # "public", "admin"
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, ForeignKey("users.id"), nullable=True)

    # Relationships
    users = relationship("User", secondary="user_tags", back_populates="tags")
    documents = relationship("Document", secondary="document_tags", back_populates="tags")
    owner = relationship("User", foreign_keys=[owner_id])

class UserTag(Base):
    __tablename__ = "user_tags"

    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    tag_id = Column(String, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)
    granted_by = Column(String, ForeignKey("users.id"), nullable=True)
    granted_at = Column(DateTime, default=datetime.utcnow)

# ============== Documents ==============

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String, nullable=False)           # stored filename
    original_filename = Column(String, nullable=False)  # user's filename
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)          # .pdf, .docx, etc.
    file_size = Column(Integer, nullable=False)
    status = Column(String, default="pending")  # pending, processing, ready, failed
    error_message = Column(Text, nullable=True)
    chunk_count = Column(Integer, nullable=True)
    uploaded_by = Column(String, ForeignKey("users.id"), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    # Relationships
    tags = relationship("Tag", secondary="document_tags", back_populates="documents")
    uploader = relationship("User", foreign_keys=[uploaded_by])

    __table_args__ = (
        Index("idx_documents_status", "status"),
        Index("idx_documents_uploaded_by", "uploaded_by"),
    )

class DocumentTag(Base):
    __tablename__ = "document_tags"

    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    tag_id = Column(String, ForeignKey("tags.id"), primary_key=True)

# ============== Chat ==============

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_archived = Column(Boolean, default=False)

    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_sessions_user", "user_id"),
        Index("idx_sessions_updated", "updated_at"),
    )

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # JSON array of citations
    confidence = Column(Float, nullable=True)
    was_routed = Column(Boolean, default=False)
    routed_to = Column(String, nullable=True)
    route_reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")

class RoutedQuestion(Base):
    __tablename__ = "routed_questions"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    message_id = Column(String, ForeignKey("chat_messages.id"), nullable=False)
    question = Column(Text, nullable=False)
    reason = Column(String, nullable=False)
    routed_to_user_id = Column(String, ForeignKey("users.id"), nullable=True)
    routed_to_tag_id = Column(String, ForeignKey("tags.id"), nullable=True)
    status = Column(String, default="pending")  # pending, acknowledged, resolved
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    resolution_notes = Column(Text, nullable=True)

# ============== Audit ==============

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String, nullable=False)  # essential, comprehensive, full_debug
    event_type = Column(String, nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    user_email = Column(String, nullable=True)
    action = Column(String, nullable=False)
    resource_type = Column(String, nullable=True)
    resource_id = Column(String, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    details = Column(Text, nullable=True)  # JSON
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    request_id = Column(String, nullable=True)

    __table_args__ = (
        Index("idx_audit_user", "user_id"),
        Index("idx_audit_resource", "resource_type", "resource_id"),
    )
```

---

## Pydantic Schemas (`db/schemas.py`)

```python
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

# ============== Auth ==============

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: "UserResponse"

class TokenPayload(BaseModel):
    sub: str  # user_id
    email: str
    role: str
    exp: datetime

# ============== Users ==============

class UserCreate(BaseModel):
    email: EmailStr
    display_name: str
    password: str = Field(min_length=12)
    role: str = "user"

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    display_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None

class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    role: str
    is_active: bool
    must_reset_password: bool
    created_at: datetime
    last_login: Optional[datetime]
    tags: List["TagResponse"] = []

    class Config:
        from_attributes = True

class PasswordResetResponse(BaseModel):
    temporary_password: str
    message: str

# ============== Tags ==============

class TagCreate(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9_-]+$")  # slug format
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
    created_at: datetime

    class Config:
        from_attributes = True

class TagAssignment(BaseModel):
    tag_ids: List[str]

# ============== Documents ==============

class DocumentResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    status: str
    error_message: Optional[str]
    chunk_count: Optional[int]
    uploaded_by: str
    uploaded_at: datetime
    processed_at: Optional[datetime]
    tags: List[TagResponse] = []

    class Config:
        from_attributes = True

class DocumentUploadResponse(BaseModel):
    id: str
    message: str
    status: str

# ============== Chat ==============

class ChatMessageCreate(BaseModel):
    content: str

class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources: Optional[str]
    confidence: Optional[float]
    was_routed: bool
    created_at: datetime

    class Config:
        from_attributes = True

class ChatSessionResponse(BaseModel):
    id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    is_archived: bool
    message_count: int = 0

    class Config:
        from_attributes = True

class ChatSessionDetail(ChatSessionResponse):
    messages: List[ChatMessageResponse] = []

# ============== Admin ==============

class AuditLogResponse(BaseModel):
    id: str
    timestamp: datetime
    level: str
    event_type: str
    user_email: Optional[str]
    action: str
    resource_type: Optional[str]
    success: bool
    ip_address: Optional[str]

    class Config:
        from_attributes = True

class SystemStats(BaseModel):
    total_users: int
    active_users: int
    total_documents: int
    total_chunks: int
    total_tags: int
    total_chat_sessions: int

# ============== Health ==============

class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    ollama: Optional[str] = None
    qdrant: Optional[str] = None

# Forward references
LoginResponse.model_rebuild()
UserResponse.model_rebuild()
```

---

## API Routes

### `api/auth.py`

```python
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.schemas import LoginRequest, LoginResponse, UserResponse
from ai_ready_rag.services.auth_service import AuthService
from ai_ready_rag.core.dependencies import get_current_user
from ai_ready_rag.db.models import User

router = APIRouter()
security = HTTPBearer()

@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    credentials: LoginRequest,
    response: Response,
    db: Session = Depends(get_db)
):
    """Authenticate user and return JWT token."""
    auth_service = AuthService(db)
    result = auth_service.login(
        email=credentials.email,
        password=credentials.password,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent")
    )

    # Set httpOnly cookie
    response.set_cookie(
        key="access_token",
        value=result["access_token"],
        httponly=True,
        secure=False,  # Set True in production with HTTPS
        samesite="strict",
        max_age=result["expires_in"]
    )

    return result

@router.post("/logout")
async def logout(
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Logout and invalidate session."""
    auth_service = AuthService(db)
    auth_service.logout(current_user.id)
    response.delete_cookie("access_token")
    return {"message": "Logged out successfully"}

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current authenticated user's info."""
    return current_user
```

### `api/users.py`

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.schemas import (
    UserCreate, UserUpdate, UserResponse,
    PasswordResetResponse, TagAssignment
)
from ai_ready_rag.services.user_service import UserService
from ai_ready_rag.core.dependencies import get_current_user, require_admin
from ai_ready_rag.db.models import User

router = APIRouter()

@router.get("/", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """List all users (admin only)."""
    service = UserService(db)
    return service.list_users(skip=skip, limit=limit)

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Create a new user (admin only)."""
    service = UserService(db)
    return service.create_user(user_data, created_by=current_user.id)

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get user by ID (admin only)."""
    service = UserService(db)
    user = service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Update user (admin only)."""
    service = UserService(db)
    return service.update_user(user_id, user_data)

@router.delete("/{user_id}")
async def deactivate_user(
    user_id: str,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Deactivate user (admin only)."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")
    service = UserService(db)
    service.deactivate_user(user_id)
    return {"message": "User deactivated"}

@router.post("/{user_id}/tags", response_model=UserResponse)
async def assign_tags(
    user_id: str,
    assignment: TagAssignment,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Assign tags to user (admin only)."""
    service = UserService(db)
    return service.assign_tags(user_id, assignment.tag_ids, granted_by=current_user.id)

@router.post("/{user_id}/reset-password", response_model=PasswordResetResponse)
async def reset_password(
    user_id: str,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Reset user password (admin only). Returns temporary password."""
    service = UserService(db)
    temp_password = service.reset_password(user_id)
    return {
        "temporary_password": temp_password,
        "message": "User must change password on next login"
    }
```

### `api/tags.py`

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.schemas import TagCreate, TagUpdate, TagResponse
from ai_ready_rag.services.tag_service import TagService
from ai_ready_rag.core.dependencies import get_current_user, require_admin
from ai_ready_rag.db.models import User

router = APIRouter()

@router.get("/", response_model=List[TagResponse])
async def list_tags(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all tags."""
    service = TagService(db)
    return service.list_tags()

@router.post("/", response_model=TagResponse, status_code=status.HTTP_201_CREATED)
async def create_tag(
    tag_data: TagCreate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Create a new tag (admin only)."""
    service = TagService(db)
    return service.create_tag(tag_data, created_by=current_user.id)

@router.get("/{tag_id}", response_model=TagResponse)
async def get_tag(
    tag_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get tag by ID."""
    service = TagService(db)
    tag = service.get_tag(tag_id)
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
    service = TagService(db)
    return service.update_tag(tag_id, tag_data)

@router.delete("/{tag_id}")
async def delete_tag(
    tag_id: str,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Delete tag (admin only)."""
    service = TagService(db)
    service.delete_tag(tag_id)
    return {"message": "Tag deleted"}
```

### `api/health.py`

```python
from fastapi import APIRouter
from ai_ready_rag.config import get_settings
from ai_ready_rag.db.schemas import HealthResponse

router = APIRouter()
settings = get_settings()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "database": "connected",
        "ollama": "disabled" if not settings.enable_rag else "unknown",
        "qdrant": "disabled" if not settings.enable_rag else "unknown"
    }

@router.get("/version")
async def version():
    """Get version info."""
    return {
        "name": settings.app_name,
        "version": settings.app_version
    }
```

---

## Core Dependencies (`core/dependencies.py`)

```python
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional

from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import User
from ai_ready_rag.core.security import decode_token

security = HTTPBearer(auto_error=False)

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current user from JWT token.
    Checks Authorization header first, then cookie.
    """
    token = None

    # Try Authorization header
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

    # Decode and validate token
    payload = decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    # Get user from database
    user = db.query(User).filter(User.id == payload.get("sub")).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User is deactivated")

    return user

async def require_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """Require admin role."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, None otherwise."""
    try:
        return await get_current_user(request, credentials, db)
    except HTTPException:
        return None
```

---

## Security (`core/security.py`)

```python
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import bcrypt
import jwt
import secrets

from ai_ready_rag.config import get_settings

settings = get_settings()

def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt(rounds=settings.bcrypt_rounds)
    return bcrypt.hashpw(password.encode(), salt).decode()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=settings.jwt_expiration_hours))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def generate_temporary_password(length: int = 16) -> str:
    """Generate a secure temporary password."""
    return secrets.token_urlsafe(length)

def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    Validate password meets requirements.
    Returns (is_valid, error_message).
    """
    if len(password) < settings.password_min_length:
        return False, f"Password must be at least {settings.password_min_length} characters"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    return True, ""
```

---

## Database Setup (`db/database.py`)

```python
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from ai_ready_rag.config import get_settings
from ai_ready_rag.db.models import Base

settings = get_settings()

# Create engine with SQLite-specific settings
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # SQLite specific
    echo=settings.debug
)

# Enable WAL mode for better concurrency
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

## Running for Testing

### Without RAG (Auth/Tags/API only)

```bash
# .env
JWT_SECRET_KEY=test-secret-key-for-development-only
ENABLE_RAG=false
ENABLE_GRADIO=false
DEBUG=true
```

```bash
# Windows
cd C:\Users\jjob\projects\VE-RAG-System
venv\Scripts\activate
pip install fastapi uvicorn sqlalchemy pydantic pydantic-settings bcrypt pyjwt
cd ai_ready_rag
uvicorn main:app --reload --port 8000
```

Access:
- API docs: http://localhost:8000/api/docs
- Health: http://localhost:8000/api/health

### Test Sequence

1. **Create first admin** (setup wizard or direct DB insert)
2. **POST /api/auth/login** → get token
3. **GET /api/auth/me** → verify auth works
4. **POST /api/tags** → create tags
5. **POST /api/users** → create user with tags
6. **Test tag assignment/revocation**

---

## Dependencies (`requirements-api.txt`)

```
# Core
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Database
sqlalchemy>=2.0.0

# Security
bcrypt>=4.0.0
pyjwt>=2.8.0

# Optional: Gradio (if ENABLE_GRADIO=true)
# gradio>=5.0.0
```

---

## File Summary

| File | Purpose | LOC (est) |
|------|---------|-----------|
| `main.py` | App entry, middleware, routes | ~60 |
| `config.py` | Settings from env | ~50 |
| `db/models.py` | SQLAlchemy models | ~200 |
| `db/schemas.py` | Pydantic DTOs | ~150 |
| `db/database.py` | DB connection | ~40 |
| `api/auth.py` | Login/logout | ~50 |
| `api/users.py` | User CRUD | ~80 |
| `api/tags.py` | Tag CRUD | ~60 |
| `api/health.py` | Health checks | ~25 |
| `core/security.py` | JWT, bcrypt | ~60 |
| `core/dependencies.py` | FastAPI deps | ~60 |
| `services/*` | Business logic | ~300 |

**Total:** ~1,100 lines for auth/tags/API foundation

---

## Next Steps

1. Create directory structure
2. Implement `config.py` and `db/database.py`
3. Implement `db/models.py` and run migrations
4. Implement `core/security.py`
5. Implement `services/auth_service.py` and `services/user_service.py`
6. Implement `api/auth.py` and `api/users.py`
7. Test login flow
8. Add tags, documents, chat incrementally
