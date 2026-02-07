"""Tests for unified Jobs API."""

from ai_ready_rag.db.models import ReindexJob, WarmingQueue


class TestJobStatusWarming:
    """Test GET /api/jobs/{id}/status for warming jobs."""

    def test_warming_job_status(self, client, admin_headers, db):
        """Returns unified status for a warming queue job."""
        job = WarmingQueue(
            id="warm-test-1",
            file_path="/tmp/test.txt",
            file_checksum="abc123",
            source_type="manual",
            total_queries=10,
            processed_queries=5,
            failed_queries=1,
            status="running",
        )
        db.add(job)
        db.commit()

        response = client.get("/api/jobs/warm-test-1/status", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "warm-test-1"
        assert data["type"] == "cache_warming"
        assert data["status"] == "running"
        assert data["progress"]["total"] == 10
        assert data["progress"]["processed"] == 5
        assert data["progress"]["failed"] == 1

    def test_completed_warming_has_summary(self, client, admin_headers, db):
        """Completed warming job includes result_summary."""
        job = WarmingQueue(
            id="warm-done-1",
            file_path="/tmp/done.txt",
            file_checksum="def456",
            source_type="manual",
            total_queries=20,
            processed_queries=18,
            failed_queries=2,
            status="completed",
        )
        db.add(job)
        db.commit()

        response = client.get("/api/jobs/warm-done-1/status", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["result_summary"] == "18/20 queries processed"


class TestJobStatusReindex:
    """Test GET /api/jobs/{id}/status for reindex jobs."""

    def test_reindex_job_status(self, client, admin_headers, db):
        """Returns unified status for a reindex job."""
        job = ReindexJob(
            id="reindex-test-1",
            status="running",
            total_documents=50,
            processed_documents=25,
            failed_documents=3,
        )
        db.add(job)
        db.commit()

        response = client.get("/api/jobs/reindex-test-1/status", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "reindex-test-1"
        assert data["type"] == "reindex"
        assert data["status"] == "running"
        assert data["progress"]["total"] == 50
        assert data["progress"]["processed"] == 25
        assert data["progress"]["failed"] == 3

    def test_completed_reindex_has_summary(self, client, admin_headers, db):
        """Completed reindex job includes result_summary."""
        job = ReindexJob(
            id="reindex-done-1",
            status="completed",
            total_documents=100,
            processed_documents=95,
            failed_documents=5,
        )
        db.add(job)
        db.commit()

        response = client.get("/api/jobs/reindex-done-1/status", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["result_summary"] == "95/100 documents processed"


class TestJobStatusNotFound:
    """Test 404 handling for unknown jobs."""

    def test_unknown_job_returns_404(self, client, admin_headers):
        """Returns 404 for non-existent job ID."""
        response = client.get("/api/jobs/nonexistent-id/status", headers=admin_headers)
        assert response.status_code == 404

    def test_requires_auth(self, client):
        """Requires authentication."""
        response = client.get("/api/jobs/some-id/status")
        assert response.status_code in (401, 403)


class TestJobCancel:
    """Test POST /api/jobs/{id}/cancel."""

    def test_cancel_warming_job(self, client, admin_headers, db):
        """Cancel sets is_cancel_requested on warming job."""
        job = WarmingQueue(
            id="warm-cancel-1",
            file_path="/tmp/cancel.txt",
            file_checksum="ghi789",
            source_type="manual",
            total_queries=5,
            status="running",
        )
        db.add(job)
        db.commit()

        response = client.post("/api/jobs/warm-cancel-1/cancel", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["cancelled"] is True
        assert data["job_id"] == "warm-cancel-1"

        # Verify flag was set
        db.refresh(job)
        assert job.is_cancel_requested is True

    def test_cancel_reindex_job(self, client, admin_headers, db):
        """Cancel sets status to aborted on reindex job."""
        job = ReindexJob(
            id="reindex-cancel-1",
            status="running",
            total_documents=10,
        )
        db.add(job)
        db.commit()

        response = client.post("/api/jobs/reindex-cancel-1/cancel", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["cancelled"] is True

        # Verify status was updated
        db.refresh(job)
        assert job.status == "aborted"

    def test_cancel_completed_job_returns_false(self, client, admin_headers, db):
        """Cannot cancel an already completed job."""
        job = WarmingQueue(
            id="warm-completed-1",
            file_path="/tmp/done.txt",
            file_checksum="jkl012",
            source_type="manual",
            total_queries=5,
            processed_queries=5,
            status="completed",
        )
        db.add(job)
        db.commit()

        response = client.post("/api/jobs/warm-completed-1/cancel", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["cancelled"] is False

    def test_cancel_not_found(self, client, admin_headers):
        """Returns 404 for non-existent job."""
        response = client.post("/api/jobs/nonexistent/cancel", headers=admin_headers)
        assert response.status_code == 404


class TestDeprecatedEndpointsRemoved:
    """Test that deprecated warming endpoints are gone."""

    def test_warm_file_removed(self, client, admin_headers):
        """POST /api/admin/cache/warm-file no longer exists."""
        response = client.post("/api/admin/cache/warm-file", headers=admin_headers)
        assert response.status_code in (404, 405, 422)

    def test_warm_jobs_list_removed(self, client, admin_headers):
        """GET /api/admin/cache/warm-jobs no longer exists."""
        response = client.get("/api/admin/cache/warm-jobs", headers=admin_headers)
        assert response.status_code in (404, 405)

    def test_warm_jobs_active_removed(self, client, admin_headers):
        """GET /api/admin/cache/warm-jobs/active no longer exists."""
        response = client.get("/api/admin/cache/warm-jobs/active", headers=admin_headers)
        assert response.status_code in (404, 405)

    def test_warm_jobs_by_id_removed(self, client, admin_headers):
        """GET /api/admin/cache/warm-jobs/{id} no longer exists."""
        response = client.get("/api/admin/cache/warm-jobs/test-id", headers=admin_headers)
        assert response.status_code in (404, 405)

    def test_warm_jobs_pause_removed(self, client, admin_headers):
        """POST /api/admin/cache/warm-jobs/{id}/pause no longer exists."""
        response = client.post("/api/admin/cache/warm-jobs/test-id/pause", headers=admin_headers)
        assert response.status_code in (404, 405)

    def test_warm_jobs_cancel_removed(self, client, admin_headers):
        """POST /api/admin/cache/warm-jobs/{id}/cancel no longer exists."""
        response = client.post("/api/admin/cache/warm-jobs/test-id/cancel", headers=admin_headers)
        assert response.status_code in (404, 405)
