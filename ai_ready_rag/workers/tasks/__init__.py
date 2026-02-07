"""ARQ task registration.

All ARQ task functions are imported here for WorkerSettings.functions.
"""

from ai_ready_rag.workers.tasks.document import process_document

__all__ = ["process_document"]
