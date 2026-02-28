"""SQLAlchemy model mixins for lifecycle management."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, String


class SoftDeleteMixin:
    """Adds soft-delete support via deleted_at timestamp.

    Usage: class MyModel(SoftDeleteMixin, Base): ...
    """

    deleted_at = Column(DateTime, nullable=True, index=True)

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    def soft_delete(self) -> None:
        self.deleted_at = datetime.utcnow()

    def restore(self) -> None:
        self.deleted_at = None


class VersionedMixin:
    """Adds temporal versioning via valid_from / valid_to timestamps.

    The current version has valid_to = NULL.
    Historical versions have valid_to set to when they were superseded.
    """

    valid_from = Column(DateTime, default=datetime.utcnow, nullable=False)
    valid_to = Column(DateTime, nullable=True)

    @property
    def is_current(self) -> bool:
        return self.valid_to is None

    def supersede(self) -> None:
        """Mark this version as superseded."""
        self.valid_to = datetime.utcnow()


class IdempotencyMixin:
    """Adds idempotency_key column for duplicate-safe upserts."""

    idempotency_key = Column(String(255), nullable=True, unique=True, index=True)
