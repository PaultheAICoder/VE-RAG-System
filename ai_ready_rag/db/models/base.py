"""Shared model utilities, mixins, and association tables."""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, Table

from ai_ready_rag.db.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class TimestampMixin:
    """Provides a standard ``created_at`` column.

    Models that need ``index=True`` on ``created_at`` should override the column.
    """

    created_at = Column(DateTime, default=datetime.utcnow)


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
    Column("tag_id", String, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)
