"""SQLAlchemy ORM models.

Re-exports all models for backward compatibility.
Import from here: ``from ai_ready_rag.db.models import User, Tag, Document``
"""

from ai_ready_rag.db.models.admin import (
    AdminSetting,
    AuditLog,
    ReindexJob,
    SettingsAudit,
    SystemSetup,
)
from ai_ready_rag.db.models.base import TimestampMixin, document_tags, generate_uuid, user_tags
from ai_ready_rag.db.models.cache import (
    CacheAccessLog,
    EmbeddingCache,
    ResponseCache,
    WarmingSSEEvent,
)
from ai_ready_rag.db.models.chat import ChatMessage, ChatSession
from ai_ready_rag.db.models.document import Document
from ai_ready_rag.db.models.evaluation import (
    DatasetSample,
    EvaluationDataset,
    EvaluationRun,
    EvaluationSample,
    LiveEvaluationScore,
)
from ai_ready_rag.db.models.rag import CuratedQA, CuratedQAKeyword, QuerySynonym
from ai_ready_rag.db.models.user import Tag, User
from ai_ready_rag.db.models.warming import WarmingBatch, WarmingQuery

__all__ = [
    "TimestampMixin",
    "generate_uuid",
    "user_tags",
    "document_tags",
    "User",
    "Tag",
    "Document",
    "ChatSession",
    "ChatMessage",
    "AuditLog",
    "AdminSetting",
    "SystemSetup",
    "SettingsAudit",
    "ReindexJob",
    "ResponseCache",
    "EmbeddingCache",
    "CacheAccessLog",
    "WarmingSSEEvent",
    "QuerySynonym",
    "CuratedQA",
    "CuratedQAKeyword",
    "WarmingBatch",
    "WarmingQuery",
    "EvaluationDataset",
    "DatasetSample",
    "EvaluationRun",
    "EvaluationSample",
    "LiveEvaluationScore",
]
