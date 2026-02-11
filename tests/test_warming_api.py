"""Tests for warming API endpoints (DB-first architecture, Issue #189)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from ai_ready_rag.db.models import WarmingBatch, WarmingQuery

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pending_batch(db, system_admin_user):
    """Create a pending batch with 3 pending queries."""
    batch = WarmingBatch(
        source_type="manual",
        total_queries=3,
        status="pending",
        submitted_by=system_admin_user.id,
        created_at=datetime.now(UTC),
    )
    db.add(batch)
    db.flush()

    for i in range(3):
        db.add(
            WarmingQuery(
                batch_id=batch.id,
                query_text=f"Test query {i + 1}",
                status="pending",
                sort_order=i,
                submitted_by=system_admin_user.id,
                created_at=datetime.now(UTC),
            )
        )
    db.flush()
    db.refresh(batch)
    return batch


@pytest.fixture
def completed_batch(db, system_admin_user):
    """Create a completed batch with mixed query statuses."""
    batch = WarmingBatch(
        source_type="upload",
        original_filename="test.txt",
        total_queries=3,
        status="completed_with_errors",
        submitted_by=system_admin_user.id,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    db.add(batch)
    db.flush()

    statuses = ["completed", "completed", "failed"]
    for i, s in enumerate(statuses):
        q = WarmingQuery(
            batch_id=batch.id,
            query_text=f"Completed query {i + 1}",
            status=s,
            sort_order=i,
            submitted_by=system_admin_user.id,
            created_at=datetime.now(UTC),
        )
        if s == "failed":
            q.error_message = "Test error"
            q.error_type = "llm_error"
        db.add(q)
    db.flush()
    db.refresh(batch)
    return batch


@pytest.fixture
def running_batch(db, system_admin_user):
    """Create a running batch with some completed queries."""
    batch = WarmingBatch(
        source_type="manual",
        total_queries=3,
        status="running",
        submitted_by=system_admin_user.id,
        created_at=datetime.now(UTC),
        started_at=datetime.now(UTC),
        worker_id="worker-test123",
    )
    db.add(batch)
    db.flush()

    statuses = ["completed", "pending", "pending"]
    for i, s in enumerate(statuses):
        db.add(
            WarmingQuery(
                batch_id=batch.id,
                query_text=f"Running query {i + 1}",
                status=s,
                sort_order=i,
                submitted_by=system_admin_user.id,
                created_at=datetime.now(UTC),
            )
        )
    db.flush()
    db.refresh(batch)
    return batch


# =============================================================================
# Mock redis helper
# =============================================================================

REDIS_AVAILABLE_PATH = "ai_ready_rag.api.admin.is_redis_available"
REDIS_POOL_PATH = "ai_ready_rag.api.admin.get_redis_pool"


def _mock_redis_none():
    """Mock is_redis_available to return False (Redis down)."""
    return patch(REDIS_AVAILABLE_PATH, new_callable=AsyncMock, return_value=False)


def _mock_redis_available():
    """Mock is_redis_available to return True + get_redis_pool to return working mock."""
    mock_redis = AsyncMock()
    mock_redis.enqueue_job = AsyncMock()

    class _Ctx:
        def __enter__(self_):
            self_._p1 = patch(REDIS_AVAILABLE_PATH, new_callable=AsyncMock, return_value=True)
            self_._p2 = patch(REDIS_POOL_PATH, new_callable=AsyncMock, return_value=mock_redis)
            self_._p1.__enter__()
            self_._p2.__enter__()
            return self_

        def __exit__(self_, *args):
            self_._p2.__exit__(*args)
            self_._p1.__exit__(*args)

    return _Ctx()


# =============================================================================
# Test Manual Warming Submit
# =============================================================================


class TestManualWarmingSubmit:
    def test_submit_success(self, client, system_admin_headers, db):
        with _mock_redis_available():
            response = client.post(
                "/api/admin/warming/queue/manual",
                json={"queries": ["What is PTO?", "How to request leave?"]},
                headers=system_admin_headers,
            )
        assert response.status_code == 201
        data = response.json()
        assert data["source_type"] == "manual"
        assert data["total_queries"] == 2
        assert data["status"] == "pending"
        assert data["pending_queries"] == 2
        assert data["completed_queries"] == 0

    def test_submit_empty_queries(self, client, system_admin_headers):
        response = client.post(
            "/api/admin/warming/queue/manual",
            json={"queries": []},
            headers=system_admin_headers,
        )
        assert response.status_code == 400

    def test_submit_only_comments(self, client, system_admin_headers):
        response = client.post(
            "/api/admin/warming/queue/manual",
            json={"queries": ["# this is a comment", "// also a comment"]},
            headers=system_admin_headers,
        )
        assert response.status_code == 400

    def test_submit_strips_blanks_and_comments(self, client, system_admin_headers, db):
        with _mock_redis_available():
            response = client.post(
                "/api/admin/warming/queue/manual",
                json={"queries": ["  Valid query  ", "", "# comment", "Another valid"]},
                headers=system_admin_headers,
            )
        assert response.status_code == 201
        data = response.json()
        assert data["total_queries"] == 2

    def test_submit_max_queries_exceeded(self, client, system_admin_headers):
        with _mock_redis_available(), patch("ai_ready_rag.api.admin.get_settings") as mock_settings:
            settings = mock_settings.return_value
            settings.warming_max_queries_per_batch = 2
            response = client.post(
                "/api/admin/warming/queue/manual",
                json={"queries": ["q1", "q2", "q3"]},
                headers=system_admin_headers,
            )
        assert response.status_code == 400
        assert "Too many queries" in response.json()["detail"]

    def test_submit_returns_503_when_redis_unavailable(self, client, system_admin_headers):
        with _mock_redis_none():
            response = client.post(
                "/api/admin/warming/queue/manual",
                json={"queries": ["What is PTO?"]},
                headers=system_admin_headers,
            )
        assert response.status_code == 503
        assert "Redis" in response.json()["detail"]

    def test_submit_no_db_writes_when_redis_unavailable(self, client, system_admin_headers, db):
        with _mock_redis_none():
            response = client.post(
                "/api/admin/warming/queue/manual",
                json={"queries": ["What is PTO?"]},
                headers=system_admin_headers,
            )
        assert response.status_code == 503
        batch_count = db.query(WarmingBatch).count()
        assert batch_count == 0

    def test_submit_unauthorized(self, client):
        response = client.post(
            "/api/admin/warming/queue/manual",
            json={"queries": ["test"]},
        )
        assert response.status_code == 401


# =============================================================================
# Test Upload Warming File
# =============================================================================


class TestUploadWarmingFile:
    def test_upload_txt_success(self, client, system_admin_headers, db):
        with _mock_redis_available():
            response = client.post(
                "/api/admin/warming/queue/upload",
                files={
                    "file": ("queries.txt", b"What is PTO?\nHow to request leave?", "text/plain")
                },
                headers=system_admin_headers,
            )
        assert response.status_code == 201
        data = response.json()
        assert data["source_type"] == "upload"
        assert data["original_filename"] == "queries.txt"
        assert data["total_queries"] == 2
        assert data["status"] == "pending"

    def test_upload_csv_success(self, client, system_admin_headers, db):
        with _mock_redis_available():
            response = client.post(
                "/api/admin/warming/queue/upload",
                files={"file": ("queries.csv", b"What is PTO?\nHow to request leave?", "text/csv")},
                headers=system_admin_headers,
            )
        assert response.status_code == 201
        assert response.json()["original_filename"] == "queries.csv"

    def test_upload_bad_extension(self, client, system_admin_headers):
        response = client.post(
            "/api/admin/warming/queue/upload",
            files={"file": ("queries.pdf", b"content", "application/pdf")},
            headers=system_admin_headers,
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_upload_empty_file(self, client, system_admin_headers):
        response = client.post(
            "/api/admin/warming/queue/upload",
            files={"file": ("queries.txt", b"", "text/plain")},
            headers=system_admin_headers,
        )
        assert response.status_code == 400
        assert "No valid questions" in response.json()["detail"]

    def test_upload_non_utf8(self, client, system_admin_headers):
        response = client.post(
            "/api/admin/warming/queue/upload",
            files={"file": ("queries.txt", b"\xff\xfe", "text/plain")},
            headers=system_admin_headers,
        )
        assert response.status_code == 400
        assert "UTF-8" in response.json()["detail"]

    def test_upload_strip_numbering(self, client, system_admin_headers, db):
        with _mock_redis_available():
            response = client.post(
                "/api/admin/warming/queue/upload",
                files={
                    "file": (
                        "queries.txt",
                        b"1. What is PTO?\n2) How to request leave?",
                        "text/plain",
                    )
                },
                headers=system_admin_headers,
            )
        assert response.status_code == 201
        assert response.json()["total_queries"] == 2

    def test_upload_skips_comments(self, client, system_admin_headers, db):
        with _mock_redis_available():
            response = client.post(
                "/api/admin/warming/queue/upload",
                files={
                    "file": (
                        "queries.txt",
                        b"# Comment line\n// Another comment\nActual query",
                        "text/plain",
                    )
                },
                headers=system_admin_headers,
            )
        assert response.status_code == 201
        assert response.json()["total_queries"] == 1

    def test_upload_returns_503_when_redis_unavailable(self, client, system_admin_headers):
        with _mock_redis_none():
            response = client.post(
                "/api/admin/warming/queue/upload",
                files={"file": ("queries.txt", b"What is PTO?", "text/plain")},
                headers=system_admin_headers,
            )
        assert response.status_code == 503
        assert "Redis" in response.json()["detail"]


# =============================================================================
# Test List Warming Queue
# =============================================================================


class TestListWarmingQueue:
    def test_list_batches(self, client, system_admin_headers, pending_batch):
        response = client.get(
            "/api/admin/warming/queue",
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] >= 1
        assert len(data["jobs"]) >= 1

    def test_list_with_status_filter(
        self, client, system_admin_headers, pending_batch, running_batch
    ):
        response = client.get(
            "/api/admin/warming/queue?status=pending",
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        for job in data["jobs"]:
            assert job["status"] == "pending"

    def test_list_aggregated_counts(self, client, system_admin_headers, completed_batch):
        response = client.get(
            "/api/admin/warming/queue",
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        # Find the completed_with_errors batch
        batch_data = next((j for j in data["jobs"] if j["id"] == completed_batch.id), None)
        assert batch_data is not None
        assert batch_data["completed_queries"] == 2
        assert batch_data["failed_queries"] == 1


# =============================================================================
# Test List Completed Batches
# =============================================================================


class TestListCompletedBatches:
    def test_list_completed_today(self, client, system_admin_headers, completed_batch):
        response = client.get(
            "/api/admin/warming/queue/completed",
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] >= 1
        # Should include completed_with_errors
        statuses = {j["status"] for j in data["jobs"]}
        assert "completed_with_errors" in statuses or "completed" in statuses

    def test_list_completed_specific_date(self, client, system_admin_headers, completed_batch):
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        response = client.get(
            f"/api/admin/warming/queue/completed?date_filter={today}",
            headers=system_admin_headers,
        )
        assert response.status_code == 200

    def test_list_completed_includes_completed_with_errors(
        self, client, system_admin_headers, completed_batch
    ):
        response = client.get(
            "/api/admin/warming/queue/completed",
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        batch_data = next((j for j in data["jobs"] if j["id"] == completed_batch.id), None)
        assert batch_data is not None
        assert batch_data["status"] == "completed_with_errors"


# =============================================================================
# Test Get Batch Detail
# =============================================================================


class TestGetBatchDetail:
    def test_get_batch_success(self, client, system_admin_headers, pending_batch):
        response = client.get(
            f"/api/admin/warming/queue/{pending_batch.id}",
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == pending_batch.id
        assert data["total_queries"] == 3
        assert data["pending_queries"] == 3

    def test_get_batch_not_found(self, client, system_admin_headers):
        response = client.get(
            "/api/admin/warming/queue/nonexistent-id",
            headers=system_admin_headers,
        )
        assert response.status_code == 404

    def test_all_failed_flag_true(self, client, system_admin_headers, db, system_admin_user):
        """P2-10: all_failed is true when all queries in batch failed."""
        batch = WarmingBatch(
            source_type="manual",
            total_queries=2,
            status="completed_with_errors",
            submitted_by=system_admin_user.id,
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        db.add(batch)
        db.flush()
        for i in range(2):
            db.add(
                WarmingQuery(
                    batch_id=batch.id,
                    query_text=f"Fail query {i}",
                    status="failed",
                    sort_order=i,
                    submitted_by=system_admin_user.id,
                    created_at=datetime.now(UTC),
                    error_message="Test error",
                    error_type="llm_error",
                )
            )
        db.flush()

        response = client.get(
            f"/api/admin/warming/queue/{batch.id}",
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["all_failed"] is True

    def test_all_failed_flag_false(self, client, system_admin_headers, completed_batch):
        """P2-10: all_failed is false when some queries succeeded."""
        response = client.get(
            f"/api/admin/warming/queue/{completed_batch.id}",
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["all_failed"] is False


# =============================================================================
# Test Delete Batch
# =============================================================================


class TestDeleteBatch:
    def test_delete_pending(self, client, system_admin_headers, pending_batch):
        response = client.delete(
            f"/api/admin/warming/queue/{pending_batch.id}",
            headers=system_admin_headers,
        )
        assert response.status_code == 204

    def test_delete_running_blocked(self, client, system_admin_headers, running_batch):
        response = client.delete(
            f"/api/admin/warming/queue/{running_batch.id}",
            headers=system_admin_headers,
        )
        assert response.status_code == 400
        assert "running" in response.json()["detail"].lower()

    def test_delete_not_found(self, client, system_admin_headers):
        response = client.delete(
            "/api/admin/warming/queue/nonexistent-id",
            headers=system_admin_headers,
        )
        assert response.status_code == 404


# =============================================================================
# Test Bulk Delete
# =============================================================================


class TestBulkDeleteBatches:
    def test_bulk_delete_success(
        self, client, system_admin_headers, pending_batch, completed_batch
    ):
        response = client.request(
            "DELETE",
            "/api/admin/warming/queue/bulk",
            json={"job_ids": [pending_batch.id, completed_batch.id]},
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] == 2
        assert data["skipped_count"] == 0

    def test_bulk_delete_skips_running(
        self, client, system_admin_headers, pending_batch, running_batch
    ):
        response = client.request(
            "DELETE",
            "/api/admin/warming/queue/bulk",
            json={"job_ids": [pending_batch.id, running_batch.id]},
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] == 1
        assert data["skipped_count"] == 1

    def test_bulk_delete_counts_not_found(self, client, system_admin_headers):
        response = client.request(
            "DELETE",
            "/api/admin/warming/queue/bulk",
            json={"job_ids": ["nonexistent-1", "nonexistent-2"]},
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["not_found_count"] == 2
        assert data["deleted_count"] == 0


# =============================================================================
# Test Current Batch
# =============================================================================


class TestCurrentBatch:
    def test_returns_running(self, client, system_admin_headers, running_batch):
        response = client.get(
            "/api/admin/warming/current",
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == running_batch.id
        assert data["status"] == "running"

    def test_returns_null(self, client, system_admin_headers):
        response = client.get(
            "/api/admin/warming/current",
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        assert response.json() is None


# =============================================================================
# Test Pause, Resume, Cancel
# =============================================================================


class TestPauseResumeCancelBatch:
    def test_pause_success(self, client, system_admin_headers, running_batch):
        response = client.post(
            "/api/admin/warming/current/pause",
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_paused"] is True
        assert data["id"] == running_batch.id

    def test_pause_no_running(self, client, system_admin_headers):
        response = client.post(
            "/api/admin/warming/current/pause",
            headers=system_admin_headers,
        )
        assert response.status_code == 404

    def test_resume_success(self, client, system_admin_headers, running_batch, db):
        running_batch.is_paused = True
        db.flush()
        response = client.post(
            "/api/admin/warming/current/resume",
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_paused"] is False

    def test_resume_no_paused(self, client, system_admin_headers):
        response = client.post(
            "/api/admin/warming/current/resume",
            headers=system_admin_headers,
        )
        assert response.status_code == 404

    def test_cancel_success(self, client, system_admin_headers, running_batch):
        response = client.post(
            "/api/admin/warming/current/cancel",
            headers=system_admin_headers,
        )
        assert response.status_code == 202
        data = response.json()
        assert data["is_cancel_requested"] is True
        assert data["status"] == "cancelling"
        assert data["batch_id"] == running_batch.id

    def test_cancel_no_running(self, client, system_admin_headers):
        response = client.post(
            "/api/admin/warming/current/cancel",
            headers=system_admin_headers,
        )
        assert response.status_code == 404


# =============================================================================
# Test Batch Queries
# =============================================================================


class TestBatchQueries:
    def test_list_queries(self, client, system_admin_headers, pending_batch):
        response = client.get(
            f"/api/admin/warming/batch/{pending_batch.id}/queries",
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 3
        assert data["batch_id"] == pending_batch.id
        assert len(data["queries"]) == 3
        # Should be ordered by sort_order
        assert data["queries"][0]["sort_order"] == 0
        # P2-9: Aggregate counts
        assert data["completed"] == 0
        assert data["failed"] == 0
        assert data["pending"] == 3

    def test_list_queries_aggregates_mixed(self, client, system_admin_headers, completed_batch):
        """Aggregate counts reflect full batch even when filtered."""
        response = client.get(
            f"/api/admin/warming/batch/{completed_batch.id}/queries",
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["completed"] == 2
        assert data["failed"] == 1
        assert data["pending"] == 0

    def test_filter_queries_by_status(self, client, system_admin_headers, completed_batch):
        response = client.get(
            f"/api/admin/warming/batch/{completed_batch.id}/queries?status=failed",
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert data["queries"][0]["status"] == "failed"

    def test_list_queries_batch_not_found(self, client, system_admin_headers):
        response = client.get(
            "/api/admin/warming/batch/nonexistent-id/queries",
            headers=system_admin_headers,
        )
        assert response.status_code == 404

    def test_delete_pending_query(self, client, system_admin_headers, pending_batch, db):
        query = db.query(WarmingQuery).filter(WarmingQuery.batch_id == pending_batch.id).first()
        response = client.delete(
            f"/api/admin/warming/batch/{pending_batch.id}/queries/{query.id}",
            headers=system_admin_headers,
        )
        assert response.status_code == 204

    def test_delete_non_pending_query(self, client, system_admin_headers, completed_batch, db):
        query = (
            db.query(WarmingQuery)
            .filter(
                WarmingQuery.batch_id == completed_batch.id,
                WarmingQuery.status == "completed",
            )
            .first()
        )
        response = client.delete(
            f"/api/admin/warming/batch/{completed_batch.id}/queries/{query.id}",
            headers=system_admin_headers,
        )
        assert response.status_code == 400
        assert "pending" in response.json()["detail"].lower()

    def test_delete_query_decrements_total(self, client, system_admin_headers, pending_batch, db):
        original_total = pending_batch.total_queries
        query = db.query(WarmingQuery).filter(WarmingQuery.batch_id == pending_batch.id).first()
        client.delete(
            f"/api/admin/warming/batch/{pending_batch.id}/queries/{query.id}",
            headers=system_admin_headers,
        )
        db.refresh(pending_batch)
        assert pending_batch.total_queries == original_total - 1


# =============================================================================
# Test Retry Endpoints
# =============================================================================


class TestRetryEndpoints:
    def test_retry_all_failed(self, client, system_admin_headers, completed_batch):
        with _mock_redis_none():
            response = client.post(
                f"/api/admin/warming/batch/{completed_batch.id}/retry",
                headers=system_admin_headers,
            )
        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == completed_batch.id
        assert data["retried_count"] == 1  # 1 failed query
        assert "pending" in data["message"].lower()

    def test_retry_non_terminal_batch(self, client, system_admin_headers, running_batch):
        response = client.post(
            f"/api/admin/warming/batch/{running_batch.id}/retry",
            headers=system_admin_headers,
        )
        assert response.status_code == 400
        assert "terminal" in response.json()["detail"].lower()

    def test_retry_single_failed_query(self, client, system_admin_headers, completed_batch, db):
        failed_query = (
            db.query(WarmingQuery)
            .filter(
                WarmingQuery.batch_id == completed_batch.id,
                WarmingQuery.status == "failed",
            )
            .first()
        )
        with _mock_redis_none():
            response = client.post(
                f"/api/admin/warming/batch/{completed_batch.id}/queries/{failed_query.id}/retry",
                headers=system_admin_headers,
            )
        assert response.status_code == 200
        data = response.json()
        assert data["query_id"] == failed_query.id
        assert data["batch_id"] == completed_batch.id
        assert data["status"] == "pending"
        assert "retry_count" in data
        assert "batch_requeued" in data

    def test_retry_non_failed_query(self, client, system_admin_headers, completed_batch, db):
        completed_query = (
            db.query(WarmingQuery)
            .filter(
                WarmingQuery.batch_id == completed_batch.id,
                WarmingQuery.status == "completed",
            )
            .first()
        )
        response = client.post(
            f"/api/admin/warming/batch/{completed_batch.id}/queries/{completed_query.id}/retry",
            headers=system_admin_headers,
        )
        assert response.status_code == 409
        assert "failed" in response.json()["detail"].lower()


# =============================================================================
# Test Legacy 410 Endpoints
# =============================================================================


class TestLegacyEndpointsGone:
    """Verify legacy endpoints return 410 Gone with redirect guidance."""

    def test_cache_warm_gone(self, client, system_admin_headers):
        response = client.post(
            "/api/admin/cache/warm",
            json={"queries": ["test"]},
            headers=system_admin_headers,
        )
        assert response.status_code == 410
        assert "warming/queue/manual" in response.json()["detail"]

    def test_warm_progress_gone(self, client, system_admin_headers):
        response = client.get(
            "/api/admin/cache/warm-progress/some-id",
            headers=system_admin_headers,
        )
        assert response.status_code == 410
        assert "warming/progress" in response.json()["detail"]

    def test_warm_retry_gone(self, client, system_admin_headers):
        response = client.post(
            "/api/admin/cache/warm-retry",
            json={"queries": ["test"]},
            headers=system_admin_headers,
        )
        assert response.status_code == 410
        assert "warming/batch" in response.json()["detail"]

    def test_warm_status_gone(self, client, system_admin_headers):
        response = client.get(
            "/api/admin/cache/warm-status/some-id",
            headers=system_admin_headers,
        )
        assert response.status_code == 410
        assert "warming/queue" in response.json()["detail"]
