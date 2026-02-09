"""Tests for SSE generator rewrite (issue #190).

Covers: batch_seq computation, integer-based replay, full-state fallback,
event type renames, pause behavior fix.
"""

import pytest

from ai_ready_rag.db.models.cache import WarmingSSEEvent
from ai_ready_rag.db.models.warming import WarmingBatch, WarmingQuery
from ai_ready_rag.services.sse_buffer_service import (
    _build_full_state_events,
    get_events_for_job,
    get_events_since,
    store_sse_event,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def warming_admin(db):
    """Create an admin user for SSE tests."""
    from ai_ready_rag.core.security import hash_password
    from ai_ready_rag.db.models import User

    user = User(
        email="sse_admin@test.com",
        display_name="SSE Admin",
        password_hash=hash_password("SSEAdmin123!"),
        role="admin",
        is_active=True,
    )
    db.add(user)
    db.flush()
    db.refresh(user)
    return user


@pytest.fixture(scope="function")
def batch_a(db, warming_admin):
    """Create a warming batch with 5 queries."""
    batch = WarmingBatch(
        source_type="manual",
        total_queries=5,
        status="running",
        submitted_by=warming_admin.id,
    )
    db.add(batch)
    db.flush()
    for i in range(5):
        db.add(
            WarmingQuery(
                batch_id=batch.id,
                query_text=f"Query {i + 1}",
                status="pending",
                sort_order=i,
            )
        )
    db.flush()
    db.refresh(batch)
    return batch


@pytest.fixture(scope="function")
def batch_b(db, warming_admin):
    """Create a second warming batch."""
    batch = WarmingBatch(
        source_type="manual",
        total_queries=3,
        status="running",
        submitted_by=warming_admin.id,
    )
    db.add(batch)
    db.flush()
    db.refresh(batch)
    return batch


# =============================================================================
# TestStoreSSEEventBatchSeq
# =============================================================================


class TestStoreSSEEventBatchSeq:
    """Verify store_sse_event computes batch_seq correctly."""

    def test_store_computes_monotonic_batch_seq(self, db, batch_a):
        """#190: store_sse_event assigns batch_seq 1, 2, 3 for same job."""
        eid1 = store_sse_event(db, "connected", batch_a.id, {"msg": "hello"})
        eid2 = store_sse_event(db, "progress", batch_a.id, {"processed": 1})
        eid3 = store_sse_event(db, "progress", batch_a.id, {"processed": 2})

        assert eid1 == "1"
        assert eid2 == "2"
        assert eid3 == "3"

        # Verify actual DB rows
        events = (
            db.query(WarmingSSEEvent)
            .filter(WarmingSSEEvent.job_id == batch_a.id)
            .order_by(WarmingSSEEvent.batch_seq.asc())
            .all()
        )
        assert len(events) == 3
        assert [e.batch_seq for e in events] == [1, 2, 3]
        assert [e.event_id for e in events] == ["1", "2", "3"]

    def test_store_batch_seq_per_job_independent(self, db, batch_a, batch_b):
        """#190: batch_seq is independent per job_id."""
        store_sse_event(db, "connected", batch_a.id, {})
        store_sse_event(db, "progress", batch_a.id, {})

        store_sse_event(db, "connected", batch_b.id, {})
        store_sse_event(db, "progress", batch_b.id, {})

        events_a = (
            db.query(WarmingSSEEvent)
            .filter(WarmingSSEEvent.job_id == batch_a.id)
            .order_by(WarmingSSEEvent.batch_seq.asc())
            .all()
        )
        events_b = (
            db.query(WarmingSSEEvent)
            .filter(WarmingSSEEvent.job_id == batch_b.id)
            .order_by(WarmingSSEEvent.batch_seq.asc())
            .all()
        )

        assert [e.batch_seq for e in events_a] == [1, 2]
        assert [e.batch_seq for e in events_b] == [1, 2]

    def test_store_no_job_id_uses_uuid(self, db):
        """#190: Global events (job_id=None) use UUID, batch_seq is None."""
        eid = store_sse_event(db, "heartbeat", None, {"timestamp": "now"})

        # UUID format check -- should not be a plain integer
        assert not eid.isdigit()
        assert len(eid) == 36  # UUID string length

        event = db.query(WarmingSSEEvent).filter(WarmingSSEEvent.event_id == eid).first()
        assert event.batch_seq is None
        assert event.job_id is None

    def test_store_event_id_not_unique_across_jobs(self, db, batch_a, batch_b):
        """#190: event_id='1' can exist for multiple jobs after UNIQUE removal."""
        store_sse_event(db, "connected", batch_a.id, {})
        store_sse_event(db, "connected", batch_b.id, {})

        # Both have event_id="1" -- no IntegrityError
        events = db.query(WarmingSSEEvent).filter(WarmingSSEEvent.event_id == "1").all()
        assert len(events) == 2


# =============================================================================
# TestGetEventsForJobReplay
# =============================================================================


class TestGetEventsForJobReplay:
    """Verify get_events_for_job uses batch_seq-based replay."""

    def test_integer_replay_returns_after_seq(self, db, batch_a):
        """#190: get_events_for_job with since_event_id='2' returns seq 3,4,5."""
        for i in range(5):
            store_sse_event(db, "progress", batch_a.id, {"step": i + 1})

        result = get_events_for_job(db, batch_a.id, since_event_id="2")
        assert len(result) == 3
        assert [r["event_id"] for r in result] == ["3", "4", "5"]

    def test_replay_none_returns_all(self, db, batch_a):
        """#190: since_event_id=None returns all events for job."""
        store_sse_event(db, "connected", batch_a.id, {})
        store_sse_event(db, "progress", batch_a.id, {})
        store_sse_event(db, "progress", batch_a.id, {})

        result = get_events_for_job(db, batch_a.id)
        assert len(result) == 3

    def test_invalid_last_event_id_triggers_fallback(self, db, batch_a):
        """#190: UUID or garbage since_event_id triggers full-state fallback."""
        store_sse_event(db, "progress", batch_a.id, {"processed": 1})

        result = get_events_for_job(db, batch_a.id, since_event_id="not-a-number")
        assert len(result) >= 1
        # Full-state fallback returns a progress event with full_state=True
        assert result[0]["payload"].get("full_state") is True

    def test_returns_batch_seq_as_event_id(self, db, batch_a):
        """#190: Each returned dict has event_id == str(batch_seq)."""
        store_sse_event(db, "connected", batch_a.id, {})
        store_sse_event(db, "progress", batch_a.id, {})

        result = get_events_for_job(db, batch_a.id)
        for event in result:
            assert event["event_id"].isdigit()

    def test_replay_after_last_seq_returns_empty(self, db, batch_a):
        """#190: If since_event_id is beyond max seq, returns empty."""
        store_sse_event(db, "progress", batch_a.id, {})
        store_sse_event(db, "progress", batch_a.id, {})

        result = get_events_for_job(db, batch_a.id, since_event_id="999")
        assert result == []


# =============================================================================
# TestGetEventsSince
# =============================================================================


class TestGetEventsSince:
    """Verify get_events_since uses batch_seq-based ordering."""

    def test_integer_replay_global(self, db, batch_a, batch_b):
        """#190: get_events_since with batch_seq returns events after that seq."""
        store_sse_event(db, "connected", batch_a.id, {})
        store_sse_event(db, "progress", batch_a.id, {})
        store_sse_event(db, "connected", batch_b.id, {})

        # batch_a has seq 1,2; batch_b has seq 1
        # Querying since batch_seq > 1 should return batch_a seq=2 and batch_b seq=1(if >1 fails)
        # Note: global query returns all events with batch_seq > seq
        result = get_events_since(db, last_event_id="1")
        # batch_a seq=2 has batch_seq > 1, batch_b seq=1 does NOT
        assert len(result) == 1
        assert result[0]["event_id"] == "2"

    def test_none_returns_recent_buffer(self, db, batch_a):
        """#190: last_event_id=None returns recent events."""
        store_sse_event(db, "connected", batch_a.id, {})
        store_sse_event(db, "progress", batch_a.id, {})

        result = get_events_since(db, last_event_id=None)
        assert len(result) == 2

    def test_invalid_last_event_id_returns_full_buffer(self, db, batch_a):
        """#190: Invalid last_event_id falls back to full buffer."""
        store_sse_event(db, "connected", batch_a.id, {})

        result = get_events_since(db, last_event_id="uuid-format-garbage")
        assert len(result) >= 1


# =============================================================================
# TestBuildFullStateEvents
# =============================================================================


class TestBuildFullStateEvents:
    """Verify _build_full_state_events helper."""

    def test_returns_progress_with_counts(self, db, batch_a):
        """#190: Full-state fallback includes correct counts."""
        # Mark some queries as completed/failed
        queries = (
            db.query(WarmingQuery)
            .filter(WarmingQuery.batch_id == batch_a.id)
            .order_by(WarmingQuery.sort_order)
            .all()
        )
        queries[0].status = "completed"
        queries[1].status = "failed"
        queries[2].status = "skipped"
        db.flush()

        result = _build_full_state_events(db, batch_a.id)
        assert len(result) == 1
        payload = result[0]["payload"]
        assert payload["processed"] == 3  # completed + failed + skipped
        assert payload["failed"] == 1
        assert payload["skipped"] == 1
        assert payload["total"] == 5
        assert payload["full_state"] is True

    def test_returns_empty_for_missing_batch(self, db):
        """#190: Missing batch returns empty list."""
        result = _build_full_state_events(db, "nonexistent-id")
        assert result == []


# =============================================================================
# TestEventIdUniqueConstraintRemoved
# =============================================================================


class TestEventIdUniqueConstraintRemoved:
    """Verify event_id column no longer has UNIQUE constraint."""

    def test_duplicate_event_ids_allowed(self, db):
        """#190: Two events with same event_id do not raise IntegrityError."""
        e1 = WarmingSSEEvent(
            event_id="1",
            event_type="progress",
            job_id="job-aaa",
            batch_seq=1,
            payload="{}",
        )
        e2 = WarmingSSEEvent(
            event_id="1",
            event_type="progress",
            job_id="job-bbb",
            batch_seq=1,
            payload="{}",
        )
        db.add(e1)
        db.flush()
        db.add(e2)
        db.flush()  # Should not raise IntegrityError

        count = db.query(WarmingSSEEvent).filter(WarmingSSEEvent.event_id == "1").count()
        assert count == 2
