"""Lifecycle repository patterns — soft-delete, versioning, idempotency."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, TypeVar

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SoftDeleteRepository:
    """Mixin for repositories that support soft-delete operations."""

    model_class: type  # subclass must define this

    def __init__(self, db: Session) -> None:
        self.db = db

    def soft_delete(self, obj_id: str) -> bool:
        """Soft-delete a record by ID. Returns True if found and deleted."""
        obj = self.db.query(self.model_class).filter(self.model_class.id == obj_id).first()
        if obj is None:
            return False
        obj.deleted_at = datetime.utcnow()
        self.db.commit()
        logger.info(
            "lifecycle.soft_delete",
            extra={"model": self.model_class.__name__, "id": obj_id},
        )
        return True

    def restore(self, obj_id: str) -> bool:
        """Restore a soft-deleted record. Returns True if found and restored."""
        obj = self.db.query(self.model_class).filter(self.model_class.id == obj_id).first()
        if obj is None or not hasattr(obj, "deleted_at"):
            return False
        obj.deleted_at = None
        self.db.commit()
        return True

    def list_active(self) -> list[Any]:
        """Return only non-deleted records."""
        return self.db.query(self.model_class).filter(self.model_class.deleted_at.is_(None)).all()

    def list_deleted(self) -> list[Any]:
        """Return only soft-deleted records."""
        return self.db.query(self.model_class).filter(self.model_class.deleted_at.isnot(None)).all()

    def purge_deleted(self, before: datetime | None = None) -> int:
        """Hard-delete records that were soft-deleted before a cutoff date.

        Returns the number of records purged.
        """
        query = self.db.query(self.model_class).filter(self.model_class.deleted_at.isnot(None))
        if before is not None:
            query = query.filter(self.model_class.deleted_at <= before)

        records = query.all()
        count = len(records)
        for record in records:
            self.db.delete(record)
        self.db.commit()
        logger.info(
            "lifecycle.purge",
            extra={"model": self.model_class.__name__, "count": count},
        )
        return count


class IdempotencyRepository:
    """Mixin for repositories that support idempotency-key-based upsert."""

    model_class: type

    def __init__(self, db: Session) -> None:
        self.db = db

    def find_by_idempotency_key(self, key: str) -> Any | None:
        """Return existing record with this idempotency key, or None."""
        if not hasattr(self.model_class, "idempotency_key"):
            return None
        return (
            self.db.query(self.model_class).filter(self.model_class.idempotency_key == key).first()
        )
