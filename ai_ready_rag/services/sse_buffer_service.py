"""SSE event ring buffer service for cache warming progress replay."""

import json
import logging
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings

logger = logging.getLogger(__name__)


def store_sse_event(
    db: Session,
    event_type: str,
    job_id: str | None,
    payload: dict,
) -> str:
    """Store SSE event in ring buffer.

    Args:
        db: Database session
        event_type: Event type (e.g., 'progress', 'job_started')
        job_id: Associated job ID (nullable for heartbeats)
        payload: Event data dictionary

    Returns:
        event_id: UUID string for client tracking
    """
    from ai_ready_rag.db.models import WarmingSSEEvent

    event_id = str(uuid.uuid4())

    event = WarmingSSEEvent(
        event_id=event_id,
        event_type=event_type,
        job_id=job_id,
        payload=json.dumps(payload),
        created_at=datetime.utcnow(),
    )
    db.add(event)
    db.commit()

    logger.debug(f"Stored SSE event: {event_type} for job {job_id}")
    return event_id


def get_events_since(db: Session, last_event_id: str | None) -> list[dict]:
    """Get events after a specific event_id for replay.

    Args:
        db: Database session
        last_event_id: Last event ID received by client (None for all events)

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
        # Find the row ID for the last_event_id
        last_event = (
            db.query(WarmingSSEEvent).filter(WarmingSSEEvent.event_id == last_event_id).first()
        )
        if last_event is None:
            # Event not found, return empty (client too far behind)
            return []

        # Get all events after that row ID
        events = (
            db.query(WarmingSSEEvent)
            .filter(WarmingSSEEvent.id > last_event.id)
            .order_by(WarmingSSEEvent.id.asc())
            .all()
        )

    return [
        {
            "event_id": e.event_id,
            "event_type": e.event_type,
            "job_id": e.job_id,
            "payload": json.loads(e.payload),
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in events
    ]


def get_events_for_job(db: Session, job_id: str, since_event_id: str | None = None) -> list[dict]:
    """Get events for a specific job, optionally after an event_id.

    Args:
        db: Database session
        job_id: Job ID to filter events for
        since_event_id: Optional event ID to start from (exclusive)

    Returns:
        List of event dictionaries for the specified job
    """
    from ai_ready_rag.db.models import WarmingSSEEvent

    query = db.query(WarmingSSEEvent).filter(WarmingSSEEvent.job_id == job_id)

    if since_event_id:
        # Find the row ID for the since_event_id
        since_event = (
            db.query(WarmingSSEEvent).filter(WarmingSSEEvent.event_id == since_event_id).first()
        )
        if since_event:
            query = query.filter(WarmingSSEEvent.id > since_event.id)
        else:
            # Event not found, return empty (client too far behind)
            return []

    events = query.order_by(WarmingSSEEvent.id.asc()).all()

    return [
        {
            "event_id": e.event_id,
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
