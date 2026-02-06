"""Cache repositories for response and embedding caches."""

from ai_ready_rag.db.models import EmbeddingCache, ResponseCache
from ai_ready_rag.db.repositories.base import BaseRepository


class ResponseCacheRepository(BaseRepository[ResponseCache]):
    model = ResponseCache

    def get_by_query_hash(self, query_hash: str) -> ResponseCache | None:
        """Get cached response by query hash."""
        results = self.list_by(query_hash=query_hash)
        return results[0] if results else None


class EmbeddingCacheRepository(BaseRepository[EmbeddingCache]):
    model = EmbeddingCache

    def get_by_query_hash(self, query_hash: str) -> EmbeddingCache | None:
        """Get cached embedding by query hash."""
        results = self.list_by(query_hash=query_hash)
        return results[0] if results else None
