"""Reindex Service for background knowledge base rebuilding.

Orchestrates full knowledge base reindex with atomic collection swap.
Supports pause/resume workflow for handling failures.
"""

import json
import logging
import uuid
from datetime import UTC, datetime
from enum import Enum

from sqlalchemy.orm import Session

from ai_ready_rag.db.models import Document, ReindexJob

logger = logging.getLogger(__name__)


class ResumeAction(str, Enum):
    """Actions for resuming a paused reindex."""

    SKIP = "skip"  # Skip current document, continue
    RETRY = "retry"  # Retry current document
    SKIP_ALL = "skip_all"  # Enable auto-skip for all future failures


class ReindexService:
    """Orchestrates background reindex operations with atomic collection swap.

    Workflow:
    1. Create temp collection
    2. Process all documents to temp collection
    3. Atomic swap: rename temp -> production
    4. Cleanup old collection

    Thread-safe: Job state tracked in database.
    """

    def __init__(self, db: Session):
        """Initialize reindex service.

        Args:
            db: Database session for job tracking
        """
        self.db = db

    def get_active_job(self) -> ReindexJob | None:
        """Get currently running reindex job if any.

        Returns:
            Active ReindexJob or None if no job running
        """
        return (
            self.db.query(ReindexJob)
            .filter(ReindexJob.status.in_(["pending", "running", "paused"]))
            .first()
        )

    def create_job(
        self,
        triggered_by: str,
        settings_changed: dict | None = None,
        mode: str = "all",
    ) -> ReindexJob:
        """Create new reindex job.

        Args:
            triggered_by: User ID who triggered the reindex
            settings_changed: Dict of changed settings (key -> {old, new})
            mode: Document selection mode — "all", "ready", or "failed"

        Returns:
            Created ReindexJob

        Raises:
            ValueError: If another job is already running
        """
        # Check for existing active job
        active_job = self.get_active_job()
        if active_job:
            raise ValueError(f"Reindex job {active_job.id} already in progress")

        # Count documents based on mode
        if mode == "all":
            total_docs = self.db.query(Document).count()
        elif mode == "failed":
            total_docs = (
                self.db.query(Document)
                .filter(Document.status.in_(["failed", "processing"]))
                .count()
            )
        else:  # "ready"
            total_docs = self.db.query(Document).filter(Document.status == "ready").count()

        # Generate unique temp collection name
        temp_collection = f"documents_reindex_{uuid.uuid4().hex[:8]}"

        job = ReindexJob(
            status="pending",
            total_documents=total_docs,
            processed_documents=0,
            failed_documents=0,
            triggered_by=triggered_by,
            settings_changed=json.dumps(settings_changed) if settings_changed else None,
            temp_collection_name=temp_collection,
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        logger.info(f"Created reindex job {job.id} for {total_docs} documents")
        return job

    def get_job(self, job_id: str) -> ReindexJob | None:
        """Get reindex job by ID.

        Args:
            job_id: Job ID

        Returns:
            ReindexJob or None if not found
        """
        return self.db.query(ReindexJob).filter(ReindexJob.id == job_id).first()

    def update_job_status(
        self,
        job_id: str,
        status: str,
        error_message: str | None = None,
    ) -> ReindexJob | None:
        """Update job status.

        Args:
            job_id: Job ID
            status: New status
            error_message: Optional error message

        Returns:
            Updated ReindexJob or None if not found
        """
        job = self.get_job(job_id)
        if not job:
            return None

        job.status = status
        if error_message:
            job.error_message = error_message

        if status == "running" and not job.started_at:
            job.started_at = datetime.now(UTC)
        elif status in ("completed", "failed", "aborted"):
            job.completed_at = datetime.now(UTC)

        self.db.commit()
        self.db.refresh(job)
        return job

    def update_job_progress(
        self,
        job_id: str,
        processed: int,
        failed: int = 0,
        current_doc_id: str | None = None,
    ) -> ReindexJob | None:
        """Update job progress.

        Args:
            job_id: Job ID
            processed: Number of documents processed
            failed: Number of documents failed
            current_doc_id: Currently processing document ID

        Returns:
            Updated ReindexJob or None if not found
        """
        job = self.get_job(job_id)
        if not job:
            return None

        job.processed_documents = processed
        job.failed_documents = failed
        job.current_document_id = current_doc_id

        self.db.commit()
        self.db.refresh(job)
        return job

    def abort_job(self, job_id: str) -> ReindexJob | None:
        """Abort a running reindex job.

        Args:
            job_id: Job ID to abort

        Returns:
            Updated ReindexJob or None if not found
        """
        job = self.get_job(job_id)
        if not job:
            return None

        if job.status not in ("pending", "running", "paused"):
            logger.warning(f"Cannot abort job {job_id} in status {job.status}")
            return job

        job.status = "aborted"
        job.completed_at = datetime.now(UTC)
        self.db.commit()
        self.db.refresh(job)

        logger.info(f"Aborted reindex job {job_id}")
        return job

    def estimate_time(self, total_documents: int) -> dict:
        """Estimate reindex time based on historical data.

        Args:
            total_documents: Number of documents to process

        Returns:
            Dict with estimate details
        """
        # Get average processing time from completed documents
        avg_time_result = (
            self.db.query(Document.processing_time_ms)
            .filter(Document.status == "ready")
            .filter(Document.processing_time_ms.isnot(None))
            .all()
        )

        if avg_time_result:
            times = [r[0] for r in avg_time_result if r[0]]
            avg_ms = sum(times) / len(times) if times else 5000  # Default 5s
        else:
            avg_ms = 5000  # Default 5 seconds per document

        total_ms = avg_ms * total_documents
        total_seconds = int(total_ms / 1000)

        # Convert to human-readable
        if total_seconds < 60:
            time_str = f"{total_seconds} seconds"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            time_str = f"{minutes} minutes"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            time_str = f"{hours} hours {minutes} minutes"

        return {
            "total_documents": total_documents,
            "avg_processing_time_ms": int(avg_ms),
            "estimated_total_ms": int(total_ms),
            "estimated_total_seconds": total_seconds,
            "estimated_time_str": time_str,
        }

    def get_job_history(self, limit: int = 10) -> list[ReindexJob]:
        """Get recent reindex job history.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of ReindexJob ordered by created_at desc
        """
        return self.db.query(ReindexJob).order_by(ReindexJob.created_at.desc()).limit(limit).all()

    # =========================================================================
    # Phase 3: Failure Handling
    # =========================================================================

    def pause_job(self, job_id: str, reason: str = "user_request") -> ReindexJob | None:
        """Pause a running reindex job.

        Args:
            job_id: Job ID to pause
            reason: 'failure' or 'user_request'

        Returns:
            Updated ReindexJob or None if not found
        """
        job = self.get_job(job_id)
        if not job:
            return None

        if job.status != "running":
            logger.warning(f"Cannot pause job {job_id} in status {job.status}")
            return job

        job.status = "paused"
        job.paused_at = datetime.now(UTC)
        job.paused_reason = reason
        self.db.commit()
        self.db.refresh(job)

        logger.info(f"Paused reindex job {job_id} (reason: {reason})")
        return job

    def record_failure(
        self,
        job_id: str,
        document_id: str,
        error_message: str,
    ) -> ReindexJob | None:
        """Record a document failure and pause the job.

        Args:
            job_id: Job ID
            document_id: Failed document ID
            error_message: Error description

        Returns:
            Updated ReindexJob or None if not found
        """
        job = self.get_job(job_id)
        if not job:
            return None

        # Parse existing failed document IDs
        failed_ids = []
        if job.failed_document_ids:
            try:
                failed_ids = json.loads(job.failed_document_ids)
            except json.JSONDecodeError:
                failed_ids = []

        # Add new failure
        if document_id not in failed_ids:
            failed_ids.append(document_id)

        job.failed_document_ids = json.dumps(failed_ids)
        job.failed_documents = len(failed_ids)
        job.last_error = error_message
        job.current_document_id = document_id
        job.retry_count = (job.retry_count or 0) + 1

        # Auto-skip mode: don't pause, just log and continue
        if job.auto_skip_failures:
            logger.warning(
                f"Auto-skipping failed document {document_id} in job {job_id}: {error_message}"
            )
            self.db.commit()
            self.db.refresh(job)
            return job

        # Pause the job
        job.status = "paused"
        job.paused_at = datetime.now(UTC)
        job.paused_reason = "failure"

        self.db.commit()
        self.db.refresh(job)

        logger.warning(f"Reindex job {job_id} paused due to failure: {error_message}")
        return job

    def resume_job(
        self,
        job_id: str,
        action: ResumeAction,
    ) -> ReindexJob | None:
        """Resume a paused reindex job.

        Args:
            job_id: Job ID to resume
            action: What to do with the failed document

        Returns:
            Updated ReindexJob or None if not found/invalid state
        """
        job = self.get_job(job_id)
        if not job:
            return None

        if job.status != "paused":
            logger.warning(f"Cannot resume job {job_id} in status {job.status}")
            return job

        # Handle the action
        if action == ResumeAction.SKIP:
            # Mark current document as permanently skipped, continue
            job.retry_count = 0
            job.status = "running"
            job.paused_at = None
            job.paused_reason = None
            logger.info(f"Resuming job {job_id}: skipping document {job.current_document_id}")

        elif action == ResumeAction.RETRY:
            # Check retry limit
            if job.retry_count >= job.max_retries:
                logger.warning(
                    f"Max retries ({job.max_retries}) reached for job {job_id}. "
                    "Use SKIP to continue."
                )
                # Don't change state - stay paused
                return job

            job.status = "running"
            job.paused_at = None
            job.paused_reason = None
            logger.info(
                f"Resuming job {job_id}: retrying document {job.current_document_id} "
                f"(attempt {job.retry_count + 1}/{job.max_retries})"
            )

        elif action == ResumeAction.SKIP_ALL:
            # Enable auto-skip mode and continue
            job.auto_skip_failures = True
            job.retry_count = 0
            job.status = "running"
            job.paused_at = None
            job.paused_reason = None
            logger.info(f"Resuming job {job_id}: enabled auto-skip mode")

        self.db.commit()
        self.db.refresh(job)
        return job

    def get_failures(self, job_id: str) -> list[dict]:
        """Get detailed information about failed documents.

        Args:
            job_id: Job ID

        Returns:
            List of failure details with document info
        """
        job = self.get_job(job_id)
        if not job or not job.failed_document_ids:
            return []

        try:
            failed_ids = json.loads(job.failed_document_ids)
        except json.JSONDecodeError:
            return []

        failures = []
        for doc_id in failed_ids:
            doc = self.db.query(Document).filter(Document.id == doc_id).first()
            failures.append(
                {
                    "document_id": doc_id,
                    "filename": doc.original_filename if doc else "Unknown",
                    "status": doc.status if doc else "unknown",
                    "error_message": job.last_error if doc_id == job.current_document_id else None,
                }
            )

        return failures

    def retry_document(self, job_id: str, document_id: str) -> ReindexJob | None:
        """Retry a specific failed document.

        Args:
            job_id: Job ID
            document_id: Document to retry

        Returns:
            Updated ReindexJob or None if not found
        """
        job = self.get_job(job_id)
        if not job:
            return None

        # Verify document is in failed list
        failed_ids = []
        if job.failed_document_ids:
            try:
                failed_ids = json.loads(job.failed_document_ids)
            except json.JSONDecodeError:
                pass

        if document_id not in failed_ids:
            logger.warning(f"Document {document_id} not in failed list for job {job_id}")
            return job

        # Remove from failed list for retry
        failed_ids.remove(document_id)
        job.failed_document_ids = json.dumps(failed_ids) if failed_ids else None
        job.failed_documents = len(failed_ids)
        job.current_document_id = document_id
        job.retry_count = 0

        self.db.commit()
        self.db.refresh(job)

        logger.info(f"Marked document {document_id} for retry in job {job_id}")
        return job
