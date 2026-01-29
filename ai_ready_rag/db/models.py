"""SQLAlchemy ORM models."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Table, Text
from sqlalchemy.orm import relationship

from ai_ready_rag.db.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


# Association tables (no extra columns to avoid FK ambiguity)
user_tags = Table(
    "user_tags",
    Base.metadata,
    Column("user_id", String, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", String, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)

document_tags = Table(
    "document_tags",
    Base.metadata,
    Column("document_id", String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", String, ForeignKey("tags.id"), primary_key=True),
)


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    display_name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default="user")
    is_active = Column(Boolean, default=True)
    must_reset_password = Column(Boolean, default=False)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, nullable=True)  # Removed FK to avoid circular ref
    last_login = Column(DateTime, nullable=True)
    login_count = Column(Integer, default=0)

    # Simple many-to-many with tags
    tags = relationship("Tag", secondary=user_tags, back_populates="users")


class Tag(Base):
    __tablename__ = "tags"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, unique=True, nullable=False)
    display_name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    color = Column(String, default="#6B7280")
    owner_id = Column(String, nullable=True)  # Removed FK to avoid circular ref
    is_system = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, nullable=True)  # Removed FK to avoid circular ref

    # Relationships
    users = relationship("User", secondary=user_tags, back_populates="tags")
    documents = relationship("Document", secondary=document_tags, back_populates="tags")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    status = Column(String, default="pending")
    error_message = Column(Text, nullable=True)
    chunk_count = Column(Integer, nullable=True)
    uploaded_by = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    # Extended fields (spec v1.2)
    title = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    content_hash = Column(String, nullable=True, index=True)

    tags = relationship("Tag", secondary=document_tags, back_populates="documents")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_archived = Column(Boolean, default=False)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    was_routed = Column(Boolean, default=False)
    routed_to = Column(String, nullable=True)
    route_reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String, nullable=False)
    event_type = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=True)
    user_email = Column(String, nullable=True)
    action = Column(String, nullable=False)
    resource_type = Column(String, nullable=True)
    resource_id = Column(String, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    request_id = Column(String, nullable=True)


class AdminSetting(Base):
    __tablename__ = "admin_settings"

    id = Column(String, primary_key=True, default=generate_uuid)
    key = Column(String, unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)  # JSON encoded
    updated_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
