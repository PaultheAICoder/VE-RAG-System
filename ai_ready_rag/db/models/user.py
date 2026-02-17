"""User and Tag models."""

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import relationship

from ai_ready_rag.db.database import Base
from ai_ready_rag.db.models.base import TimestampMixin, document_tags, generate_uuid, user_tags


class User(TimestampMixin, Base):
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
    tag_access_enabled = Column(Boolean, default=True, nullable=False)
    created_by = Column(String, nullable=True)  # Removed FK to avoid circular ref
    last_login = Column(DateTime, nullable=True)
    login_count = Column(Integer, default=0)

    # Simple many-to-many with tags
    tags = relationship("Tag", secondary=user_tags, back_populates="users")


class Tag(TimestampMixin, Base):
    __tablename__ = "tags"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, unique=True, nullable=False)
    display_name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    color = Column(String, default="#6B7280")
    owner_id = Column(String, nullable=True)  # Removed FK to avoid circular ref
    is_system = Column(Boolean, default=False)
    created_by = Column(String, nullable=True)  # Removed FK to avoid circular ref

    # Relationships
    users = relationship("User", secondary=user_tags, back_populates="tags")
    documents = relationship("Document", secondary=document_tags, back_populates="tags")
