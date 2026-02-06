"""Tag repository."""

from ai_ready_rag.db.models import Tag
from ai_ready_rag.db.repositories.base import BaseRepository


class TagRepository(BaseRepository[Tag]):
    model = Tag

    def get_by_name(self, name: str) -> Tag | None:
        """Get tag by name."""
        results = self.list_by(name=name)
        return results[0] if results else None

    def get_by_ids(self, tag_ids: list[str]) -> list[Tag]:
        """Get multiple tags by ID."""
        return [tag for tag_id in tag_ids if (tag := self.get(tag_id))]
