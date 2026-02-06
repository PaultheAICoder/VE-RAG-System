"""Base service with standard CRUD and transaction management."""

from typing import Any

from sqlalchemy.orm import Session

from ai_ready_rag.core.exceptions import EntityNotFound
from ai_ready_rag.db.repositories.base import BaseRepository


class BaseService[T, R: BaseRepository]:
    """Base service providing standard CRUD and transaction management.

    Transaction boundary rules:
    1. Single-service operations: service commits via create/update/delete.
    2. Multi-service orchestration: orchestrator commits, inner services flush only.
    3. Anti-pattern: inner services must never commit when orchestrated.
    """

    repository_class: type[R]

    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo: R = self.repository_class(db)

    def get(self, id_: Any) -> T | None:
        """Get entity by ID."""
        return self.repo.get(id_)

    def get_or_raise(self, id_: Any) -> T:
        """Get entity or raise EntityNotFound."""
        obj = self.repo.get(id_)
        if obj is None:
            model_name = self.repo.model.__name__
            raise EntityNotFound(f"{model_name} not found", context={"id": str(id_)})
        return obj

    def list_all(self) -> list[T]:
        """List all entities."""
        return self.repo.list_all()

    def create(self, obj: T) -> T:
        """Create entity with commit."""
        self.repo.add(obj)
        self.commit()
        self.repo.refresh(obj)
        return obj

    def create_no_commit(self, obj: T) -> T:
        """Create entity without commit (for orchestration)."""
        self.repo.add(obj)
        self.flush()
        return obj

    def update(self, obj: T, **fields: Any) -> T:
        """Update entity with commit."""
        self.repo.partial_update(obj, **fields)
        self.commit()
        self.repo.refresh(obj)
        return obj

    def delete(self, obj: T) -> None:
        """Delete entity with commit."""
        self.repo.delete(obj)
        self.commit()

    def commit(self) -> None:
        """Commit transaction."""
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def flush(self) -> None:
        """Flush without commit."""
        self.repo.flush()
