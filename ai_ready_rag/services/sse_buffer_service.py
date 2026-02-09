"""SSE event ring buffer service for cache warming progress replay.

Uses batch_seq (monotonic integer per job) for replay ordering instead of UUIDs.
"""

import json
import logging
import uuid
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings

logger = logging.getLogger(__name__)


def store_sse_event(
    db: Session,
    event_type: str,
    job_id: str | None,
    payload: dict,
) -> str:
    """Store SSE event in ring buffer with monotonic batch_seq per job.

    For job-scoped events, computes batch_seq = max(batch_seq)+1 and sets
    event_id = str(batch_seq). For global events (job_id=None), uses UUID.

    Args:
        db: Database session
        event_type: Event type (e.g., 'progress', 'connected')
        job_id: Associated job ID (None for global events)
        payload: Event data dictionary

    Returns:
        event_id: str(batch_seq) for job-scoped events, UUID for global events
    """
    from ai_ready_rag.db.models import WarmingSSEEvent

    if job_id is not None:
        # Compute monotonic batch_seq per job
        max_seq = (
            db.query(func.max(WarmingSSEEvent.batch_seq))
            .filter(WarmingSSEEvent.job_id == job_id)
            .scalar()
            or 0
        )
        batch_seq = max_seq + 1
        event_id = str(batch_seq)
    else:
        # Global events without job context keep UUID
        batch_seq = None
        event_id = str(uuid.uuid4())

    event = WarmingSSEEvent(
        event_id=event_id,
        event_type=event_type,
        job_id=job_id,
        batch_seq=batch_seq,
        payload=json.dumps(payload),
        created_at=datetime.utcnow(),
    )
    db.add(event)
    db.commit()

    logger.debug(f"Stored SSE event: {event_type} for job {job_id} (batch_seq={batch_seq})")
    return event_id


def _build_full_state_events(db: Session, job_id: str) -> list[dict]:
    """Build a synthetic progress event with current batch state.

    Used as fallback when last_event_id is too old (pruned) or invalid.
    Per spec Section 7.4: send full current state instead of empty list.

    Args:
        db: Database session
        job_id: Job ID to build state for

    Returns:
        List with a single progress event dict representing current state
    """
    from ai_ready_rag.db.models import WarmingBatch, WarmingQuery

    batch = db.query(WarmingBatch).filter(WarmingBatch.id == job_id).first()
    if not batch:
        return []

    from sqlalchemy import case

    counts = (
        db.query(
            func.count(case((WarmingQuery.status == "completed", 1))).label("completed"),
            func.count(case((WarmingQuery.status == "failed", 1))).label("failed"),
            func.count(case((WarmingQuery.status == "skipped", 1))).label("skipped"),
        )
        .filter(WarmingQuery.batch_id == job_id)
        .first()
    )

    completed_count = counts.completed if counts else 0
    failed_count = counts.failed if counts else 0
    skipped_count = counts.skipped if counts else 0
    processed = completed_count + failed_count + skipped_count
    percent = int(processed / batch.total_queries * 100) if batch.total_queries > 0 else 0

    progress_data = {
        "batch_id": job_id,
        "processed": processed,
        "failed": failed_count,
        "skipped": skipped_count,
        "total": batch.total_queries,
        "percent": percent,
        "batch_status": batch.status,
        "full_state": True,
    }

    return [
        {
            "event_id": "0",
            "event_type": "progress",
            "job_id": job_id,
            "payload": progress_data,
            "created_at": datetime.utcnow().isoformat(),
        }
    ]


def get_events_since(db: Session, last_event_id: str | None) -> list[dict]:
    """Get events after a specific batch_seq for replay.

    Uses batch_seq-based ordering. Falls back to full buffer on invalid input.

    Args:
        db: Database session
        last_event_id: Last event ID (str(batch_seq)) received by client

    Returns:
        List of event dictionaries with event_id, event_type, job_id, payload, created_at
    """
    from ai_ready_rag.db.models import WarmingSSEEvent

    if last_event_id is None:
        # No last_event_id, return recent events up to buffer size
        settings = get_settings()
        events = (
            db.query(WarmingSSEEvent)
            .order_by(WarmingSSEEvent.id.desc())
            .limit(settings.sse_event_buffer_size)
            .all()
        )
        # Reverse to get chronological order
        events = list(reversed(events))
    else:
        try:
            seq = int(last_event_id)
        except ValueError:
            # Invalid (old UUID format or garbage) -- return full buffer as fallback
            settings = get_settings()
            events = (
                db.query(WarmingSSEEvent)
                .order_by(WarmingSSEEvent.id.desc())
                .limit(settings.sse_event_buffer_size)
                .all()
            )
            events = list(reversed(events))
            return _serialize_events(events)

        # Query events with batch_seq > seq, ordered by batch_seq ASC
        events = (
            db.query(WarmingSSEEvent)
            .filter(WarmingSSEEvent.batch_seq > seq)
            .order_by(WarmingSSEEvent.batch_seq.asc())
            .all()
        )

    return _serialize_events(events)


def get_events_for_job(db: Session, job_id: str, since_event_id: str | None = None) -> list[dict]:
    """Get events for a specific job using batch_seq-based replay.

    Args:
        db: Database session
        job_id: Job ID to filter events for
        since_event_id: Optional batch_seq string to start from (exclusive)

    Returns:
        List of event dictionaries for the specified job
    """
    from ai_ready_rag.db.models import WarmingSSEEvent

    if since_event_id is None:
        # Return all events for job ordered by batch_seq
        settings = get_settings()
        events = (
            db.query(WarmingSSEEvent)
            .filter(WarmingSSEEvent.job_id == job_id)
            .order_by(WarmingSSEEvent.batch_seq.asc())
            .limit(settings.sse_event_buffer_size)
            .all()
        )
        return _serialize_events(events)

    try:
        seq = int(since_event_id)
    except ValueError:
        # Invalid last_event_id (old UUID or garbage) -- full-state fallback
        return _build_full_state_events(db, job_id)

    # Query events with batch_seq > seq for this job
    events = (
        db.query(WarmingSSEEvent)
        .filter(WarmingSSEEvent.job_id == job_id, WarmingSSEEvent.batch_seq > seq)
        .order_by(WarmingSSEEvent.batch_seq.asc())
        .all()
    )

    # If no events found, the seq may have been pruned -- full-state fallback
    if not events and seq > 0:
        # Check if there are ANY events for this job
        any_events = db.query(WarmingSSEEvent).filter(WarmingSSEEvent.job_id == job_id).first()
        if any_events is None:
            # No events at all -- might have been fully pruned
            return _build_full_state_events(db, job_id)

    return _serialize_events(events)


def _serialize_events(events: list) -> list[dict]:
    """Serialize WarmingSSEEvent ORM objects to dicts.

    Uses str(batch_seq) as event_id for consistency.
    """
    return [
        {
            "event_id": str(e.batch_seq) if e.batch_seq is not None else e.event_id,
            "event_type": e.event_type,
            "job_id": e.job_id,
            "payload": json.loads(e.payload),
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in events
    ]


def prune_old_events(db: Session) -> int:
    """Prune events beyond buffer size, keeping most recent.

    Args:
        db: Database session

    Returns:
        Number of events deleted
    """
    from ai_ready_rag.db.models import WarmingSSEEvent

    settings = get_settings()
    buffer_size = settings.sse_event_buffer_size

    # Count total events
    total_count = db.query(WarmingSSEEvent).count()
    if total_count <= buffer_size:
        return 0

    # Find the cutoff ID (keep events with id > cutoff)
    cutoff_event = (
        db.query(WarmingSSEEvent).order_by(WarmingSSEEvent.id.desc()).offset(buffer_size).first()
    )

    if cutoff_event is None:
        return 0

    # Delete events with id <= cutoff
    deleted = (
        db.query(WarmingSSEEvent)
        .filter(WarmingSSEEvent.id <= cutoff_event.id)
        .delete(synchronize_session=False)
    )
    db.commit()

    logger.info(f"Pruned {deleted} old SSE events from ring buffer")
    return deleted
