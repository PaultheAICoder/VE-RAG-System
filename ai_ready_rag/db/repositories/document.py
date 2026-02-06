"""Document repository."""

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from ai_ready_rag.db.models import Document, document_tags
from ai_ready_rag.db.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    model = Document

    def get_by_hash(self, content_hash: str) -> Document | None:
        """Get document by content hash."""
        results = self.list_by(content_hash=content_hash)
        return results[0] if results else None

    def list_for_user(self, tag_ids: list[str]) -> list[Document]:
        """List documents accessible to user via tags."""
        stmt = (
            select(Document)
            .join(document_tags)
            .where(document_tags.c.tag_id.in_(tag_ids))
            .distinct()
        )
        return list(self.db.scalars(stmt).all())

    def get_with_tags(self, doc_id: str) -> Document | None:
        """Get document with eager-loaded tags."""
        stmt = select(Document).where(Document.id == doc_id).options(joinedload(Document.tags))
        return self.db.scalar(stmt)
