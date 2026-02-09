"""ARQ task registration.

All ARQ task functions are imported here for WorkerSettings.functions.
"""

from ai_ready_rag.workers.tasks.document import process_document
from ai_ready_rag.workers.tasks.reindex import reindex_knowledge_base
from ai_ready_rag.workers.tasks.warming import warm_cache
from ai_ready_rag.workers.tasks.warming_batch import process_warming_batch

__all__ = ["process_document", "reindex_knowledge_base", "warm_cache", "process_warming_batch"]
