"""User repository."""

from ai_ready_rag.db.models import User
from ai_ready_rag.db.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    model = User

    def get_by_email(self, email: str) -> User | None:
        """Get user by email address."""
        results = self.list_by(email=email)
        return results[0] if results else None

    def exists_by_email(self, email: str) -> bool:
        """Check if user with email exists."""
        return self.exists(email=email)
