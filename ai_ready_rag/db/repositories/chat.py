"""Chat session and message repositories."""

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from ai_ready_rag.db.models import ChatMessage, ChatSession
from ai_ready_rag.db.repositories.base import BaseRepository


class ChatSessionRepository(BaseRepository[ChatSession]):
    model = ChatSession

    def list_for_user(self, user_id: str) -> list[ChatSession]:
        """List sessions for a user, ordered by recency."""
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def get_with_messages(self, session_id: str) -> ChatSession | None:
        """Get session with eager-loaded messages."""
        stmt = (
            select(ChatSession)
            .where(ChatSession.id == session_id)
            .options(joinedload(ChatSession.messages))
        )
        return self.db.scalar(stmt)


class ChatMessageRepository(BaseRepository[ChatMessage]):
    model = ChatMessage

    def list_for_session(self, session_id: str) -> list[ChatMessage]:
        """List messages in a session, ordered by creation."""
        return self.list_by(session_id=session_id)
