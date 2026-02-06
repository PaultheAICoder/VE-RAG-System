"""Audit log repository."""

from ai_ready_rag.db.models import AuditLog
from ai_ready_rag.db.repositories.base import BaseRepository


class AuditLogRepository(BaseRepository[AuditLog]):
    model = AuditLog
