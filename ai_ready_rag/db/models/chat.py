"""Chat session and message models."""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship

from ai_ready_rag.db.database import Base
from ai_ready_rag.db.models.base import TimestampMixin, generate_uuid


class ChatSession(TimestampMixin, Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (Index("idx_chat_sessions_user_id", "user_id"),)

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_archived = Column(Boolean, default=False)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(TimestampMixin, Base):
    __tablename__ = "chat_messages"
    __table_args__ = (Index("idx_chat_messages_session_id", "session_id"),)

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    confidence_retrieval = Column(Float, nullable=True)
    confidence_coverage = Column(Float, nullable=True)
    confidence_llm = Column(Integer, nullable=True)
    generation_time_ms = Column(Float, nullable=True)
    was_routed = Column(Boolean, default=False)
    routed_to = Column(String, nullable=True)
    route_reason = Column(String, nullable=True)

    session = relationship("ChatSession", back_populates="messages")
