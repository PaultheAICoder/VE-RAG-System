"""Generic base repository for SQLAlchemy models."""

from typing import Any

from sqlalchemy import exists as sa_exists
from sqlalchemy import func, select
from sqlalchemy.orm import Session


class BaseRepository[T]:
    """Generic base repository providing standard CRUD operations.

    Repositories never commit — the service layer owns transaction boundaries.
    Repositories never raise HTTPException — they return None or raise domain exceptions.
    """

    model: type[T]

    def __init__(self, db: Session) -> None:
        self.db = db

    # --- Retrieval ---

    def get(self, id_: Any) -> T | None:
        """Get entity by primary key."""
        return self.db.get(self.model, id_)

    def get_with_filter(self, id_: Any, **filters: Any) -> T | None:
        """Get by PK with additional filters (e.g., tenant check)."""
        stmt = select(self.model).where(self.model.id == id_)
        for key, value in filters.items():
            stmt = stmt.where(getattr(self.model, key) == value)
        return self.db.scalar(stmt)

    def list_all(self) -> list[T]:
        """List all entities."""
        return list(self.db.scalars(select(self.model)).all())

    def list_by(self, **filters: Any) -> list[T]:
        """List entities matching filters."""
        stmt = select(self.model)
        for key, value in filters.items():
            stmt = stmt.where(getattr(self.model, key) == value)
        return list(self.db.scalars(stmt).all())

    def exists(self, **filters: Any) -> bool:
        """Check if entity exists matching filters."""
        conditions = [getattr(self.model, k) == v for k, v in filters.items()]
        stmt = select(sa_exists().where(*conditions))
        return self.db.scalar(stmt) or False

    def count(self, **filters: Any) -> int:
        """Count entities matching filters."""
        stmt = select(func.count()).select_from(self.model)
        for key, value in filters.items():
            if value is not None:
                stmt = stmt.where(getattr(self.model, key) == value)
        return self.db.scalar(stmt) or 0

    # --- Persistence (never commit) ---

    def add(self, obj: T) -> T:
        """Add entity to session."""
        self.db.add(obj)
        return obj

    def add_all(self, objs: list[T]) -> list[T]:
        """Add multiple entities to session."""
        self.db.add_all(objs)
        return objs

    def delete(self, obj: T) -> None:
        """Mark entity for deletion."""
        self.db.delete(obj)

    def delete_all(self, objs: list[T]) -> None:
        """Mark multiple entities for deletion."""
        for obj in objs:
            self.db.delete(obj)

    def flush(self) -> None:
        """Flush pending changes (get IDs without commit)."""
        self.db.flush()

    def refresh(self, obj: T) -> T:
        """Refresh entity from database."""
        self.db.refresh(obj)
        return obj

    # --- Update ---

    def partial_update(self, obj: T, skip_none: bool = True, **fields: Any) -> T:
        """Update entity fields."""
        for key, value in fields.items():
            if skip_none and value is None:
                continue
            setattr(obj, key, value)
        self.db.add(obj)
        return obj
