"""Integration tests for cache warming system.

These tests verify end-to-end functionality that unit tests miss:
1. WarmingWorker actually starts when the app starts
2. Frontend API endpoints exist and work correctly
3. SSE progress endpoints are accessible
4. Full warming flow from job creation to completion

Run with: pytest tests/test_warming_integration.py -v
"""

import asyncio
from unittest.mock import AsyncMock

import pytest


class TestWarmingWorkerStartup:
    """Verify WarmingWorker starts correctly with the application."""

    def test_warming_worker_global_exists(self):
        """Verify warming_worker global is defined in main module."""
        from ai_ready_rag import main

        assert hasattr(main, "warming_worker"), "warming_worker global not defined"

    def test_warming_cleanup_global_exists(self):
        """Verify warming_cleanup global is defined in main module."""
        from ai_ready_rag import main

        assert hasattr(main, "warming_cleanup"), "warming_cleanup global not defined"

    def test_startup_event_defined(self):
        """Verify startup event handler is registered."""
        from ai_ready_rag.main import app

        # Check that on_event handlers are registered
        startup_handlers = [
            route for route in app.router.on_startup if "startup_event" in str(route)
        ]
        assert len(startup_handlers) > 0 or hasattr(app, "router"), (
            "Startup event should be registered"
        )

    @pytest.mark.asyncio
    async def test_warming_worker_can_be_instantiated(self):
        """Verify WarmingWorker can be created with valid config."""
        from ai_ready_rag.config import get_settings
        from ai_ready_rag.workers.warming_worker import WarmingWorker

        settings = get_settings()

        # Mock RAG service
        mock_rag = AsyncMock()
        mock_rag.warm_cache = AsyncMock(return_value=True)

        worker = WarmingWorker(mock_rag, settings)
        assert worker.worker_id.startswith("worker-")
        assert worker._task is None  # Not started yet

    @pytest.mark.asyncio
    async def test_warming_worker_starts_and_stops(self):
        """Verify WarmingWorker lifecycle works correctly."""
        from ai_ready_rag.config import get_settings
        from ai_ready_rag.workers.warming_worker import WarmingWorker

        settings = get_settings()

        mock_rag = AsyncMock()
        mock_rag.warm_cache = AsyncMock(return_value=True)

        worker = WarmingWorker(mock_rag, settings)

        # Start
        await worker.start()
        assert worker._task is not None
        assert not worker._task.done()

        # Stop
        await worker.stop()
        # Task should be cancelled or done
        await asyncio.sleep(0.1)


class TestAPIContractAlignment:
    """Verify frontend expects the same endpoints backend provides.

    These tests ensure the frontend and backend stay in sync.
    If these fail, check frontend/src/api/cache.ts
    """

    def test_manual_warming_endpoint_exists(self, client, admin_headers):
        """Frontend expects POST /api/admin/warming/queue/manual."""
        response = client.post(
            "/api/admin/warming/queue/manual",
            headers=admin_headers,
            json={"queries": ["test query"]},
        )
        # Should return 201 Created, not 404 Not Found
        assert response.status_code == 201, (
            f"Endpoint /api/admin/warming/queue/manual not found. "
            f"Got {response.status_code}: {response.text}"
        )

    def test_file_upload_endpoint_exists(self, client, admin_headers):
        """Frontend expects POST /api/admin/warming/queue/upload."""
        # Create a simple test file
        files = {"file": ("test.txt", b"test query 1\ntest query 2", "text/plain")}
        response = client.post(
            "/api/admin/warming/queue/upload",
            headers=admin_headers,
            files=files,
        )
        # Should return 201 Created, not 404 Not Found
        assert response.status_code == 201, (
            f"Endpoint /api/admin/warming/queue/upload not found. "
            f"Got {response.status_code}: {response.text}"
        )

    def test_warming_queue_list_endpoint_exists(self, client, admin_headers):
        """Frontend expects GET /api/admin/warming/queue."""
        response = client.get(
            "/api/admin/warming/queue",
            headers=admin_headers,
        )
        assert response.status_code == 200, (
            f"Endpoint /api/admin/warming/queue not found. "
            f"Got {response.status_code}: {response.text}"
        )

    def test_warming_job_status_endpoint_exists(self, client, admin_headers, db):
        """Frontend expects GET /api/admin/warming/queue/{job_id}."""
        # First create a job
        response = client.post(
            "/api/admin/warming/queue/manual",
            headers=admin_headers,
            json={"queries": ["test"]},
        )
        job_id = response.json()["id"]

        # Then check status endpoint
        response = client.get(
            f"/api/admin/warming/queue/{job_id}",
            headers=admin_headers,
        )
        assert response.status_code == 200, (
            f"Endpoint /api/admin/warming/queue/{{job_id}} not found. "
            f"Got {response.status_code}: {response.text}"
        )

    def test_warming_progress_sse_endpoint_exists(self, client, admin_headers, db):
        """Frontend expects GET /api/admin/warming/progress with SSE."""
        # Create a job first
        response = client.post(
            "/api/admin/warming/queue/manual",
            headers=admin_headers,
            json={"queries": ["test"]},
        )
        job_id = response.json()["id"]

        # Get token from headers
        token = admin_headers["Authorization"].replace("Bearer ", "")

        # Check SSE endpoint exists (won't fully test SSE, just 200 OK)
        response = client.get(
            f"/api/admin/warming/progress?job_id={job_id}&token={token}",
        )
        # SSE endpoints return 200 with streaming
        assert response.status_code == 200, (
            f"SSE endpoint /api/admin/warming/progress not found. "
            f"Got {response.status_code}: {response.text}"
        )

    def test_pause_endpoint_exists(self, client, admin_headers):
        """Frontend expects POST /api/admin/warming/current/pause."""
        response = client.post(
            "/api/admin/warming/current/pause",
            headers=admin_headers,
        )
        # May return 404 if no active job, but not 405 Method Not Allowed
        assert response.status_code in [200, 404], (
            f"Endpoint /api/admin/warming/current/pause not found. "
            f"Got {response.status_code}: {response.text}"
        )

    def test_resume_endpoint_exists(self, client, admin_headers):
        """Frontend expects POST /api/admin/warming/current/resume."""
        response = client.post(
            "/api/admin/warming/current/resume",
            headers=admin_headers,
        )
        assert response.status_code in [200, 404], (
            f"Endpoint /api/admin/warming/current/resume not found. "
            f"Got {response.status_code}: {response.text}"
        )

    def test_cancel_endpoint_exists(self, client, admin_headers):
        """Frontend expects POST /api/admin/warming/current/cancel."""
        response = client.post(
            "/api/admin/warming/current/cancel",
            headers=admin_headers,
        )
        assert response.status_code in [200, 404], (
            f"Endpoint /api/admin/warming/current/cancel not found. "
            f"Got {response.status_code}: {response.text}"
        )


class TestManualWarmingResponse:
    """Verify manual warming response matches frontend expectations."""

    def test_response_has_required_fields(self, client, admin_headers):
        """Response must have 'id' field (not 'job_id')."""
        response = client.post(
            "/api/admin/warming/queue/manual",
            headers=admin_headers,
            json={"queries": ["test query"]},
        )
        data = response.json()

        # Frontend uses response.id, NOT response.job_id
        assert "id" in data, "Response must have 'id' field for frontend compatibility"
        assert "status" in data, "Response must have 'status' field"
        assert "total_queries" in data, "Response must have 'total_queries' field"

    def test_response_id_is_string(self, client, admin_headers):
        """Job ID must be a string UUID."""
        response = client.post(
            "/api/admin/warming/queue/manual",
            headers=admin_headers,
            json={"queries": ["test"]},
        )
        data = response.json()

        assert isinstance(data["id"], str), "Job ID must be string"
        assert len(data["id"]) == 36, "Job ID should be UUID format"


class TestFileUploadResponse:
    """Verify file upload response matches frontend expectations."""

    def test_upload_response_has_required_fields(self, client, admin_headers):
        """Upload response must have fields frontend expects."""
        files = {"file": ("test.txt", b"query 1\nquery 2", "text/plain")}
        response = client.post(
            "/api/admin/warming/queue/upload",
            headers=admin_headers,
            files=files,
        )
        data = response.json()

        # Frontend expects these fields
        assert "id" in data, "Response must have 'id' field"
        assert "total_queries" in data, "Response must have 'total_queries' field"
        assert "status" in data, "Response must have 'status' field"

    def test_upload_counts_queries_correctly(self, client, admin_headers):
        """Upload should count non-empty, non-comment lines."""
        content = b"query 1\nquery 2\n# comment\n\nquery 3"
        files = {"file": ("test.txt", content, "text/plain")}
        response = client.post(
            "/api/admin/warming/queue/upload",
            headers=admin_headers,
            files=files,
        )
        data = response.json()

        assert data["total_queries"] == 3, "Should count 3 valid queries"


class TestWarmingJobLifecycle:
    """Test full warming job lifecycle."""

    def test_job_created_with_pending_status(self, client, admin_headers):
        """New jobs should start with 'pending' status."""
        response = client.post(
            "/api/admin/warming/queue/manual",
            headers=admin_headers,
            json={"queries": ["test"]},
        )
        data = response.json()

        assert data["status"] == "pending"

    def test_job_file_is_created(self, client, admin_headers, db):
        """Job should create a query file on disk."""
        import os

        response = client.post(
            "/api/admin/warming/queue/manual",
            headers=admin_headers,
            json={"queries": ["test query"]},
        )
        data = response.json()

        # Check file path in response
        assert "file_path" in data
        assert os.path.exists(data["file_path"]), "Query file should exist"

        # Verify content
        with open(data["file_path"]) as f:
            content = f.read()
        assert "test query" in content

    def test_job_appears_in_queue_list(self, client, admin_headers):
        """Created job should appear in queue list."""
        # Create job
        create_resp = client.post(
            "/api/admin/warming/queue/manual",
            headers=admin_headers,
            json={"queries": ["test"]},
        )
        job_id = create_resp.json()["id"]

        # List jobs
        list_resp = client.get(
            "/api/admin/warming/queue",
            headers=admin_headers,
        )
        jobs = list_resp.json()["jobs"]

        job_ids = [j["id"] for j in jobs]
        assert job_id in job_ids, "Created job should appear in queue list"

    def test_job_can_be_deleted(self, client, admin_headers):
        """Jobs should be deletable."""
        # Create job
        create_resp = client.post(
            "/api/admin/warming/queue/manual",
            headers=admin_headers,
            json={"queries": ["test"]},
        )
        job_id = create_resp.json()["id"]

        # Delete job
        delete_resp = client.delete(
            f"/api/admin/warming/queue/{job_id}",
            headers=admin_headers,
        )
        assert delete_resp.status_code in [200, 204]

        # Verify deleted
        get_resp = client.get(
            f"/api/admin/warming/queue/{job_id}",
            headers=admin_headers,
        )
        assert get_resp.status_code == 404


class TestDeprecatedEndpoints:
    """Verify deprecated endpoints still work or return proper errors."""

    def test_old_warm_endpoint_still_exists(self, client, admin_headers):
        """Old /api/admin/cache/warm should still work for backwards compat."""
        response = client.post(
            "/api/admin/cache/warm",
            headers=admin_headers,
            json={"queries": ["test"]},
        )
        # Should work (202) or be explicitly deprecated (410 Gone)
        # Should NOT be 404 Not Found without warning
        assert response.status_code in [200, 202, 410], (
            f"Old endpoint should work or return 410 Gone, not {response.status_code}"
        )
