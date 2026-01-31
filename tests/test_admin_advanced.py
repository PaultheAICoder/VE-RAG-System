"""Tests for advanced settings and reindex endpoints."""


class TestGetAdvancedSettings:
    """Tests for GET /api/admin/settings/advanced."""

    def test_returns_default_settings(self, client, admin_headers):
        """GET returns default advanced settings."""
        response = client.get("/api/admin/settings/advanced", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert "embedding_model" in data
        assert "chunk_size" in data
        assert "chunk_overlap" in data
        assert "hnsw_ef_construct" in data
        assert "hnsw_m" in data
        assert "vector_backend" in data

    def test_requires_admin(self, client, user_headers):
        """GET requires admin role."""
        response = client.get("/api/admin/settings/advanced", headers=user_headers)
        assert response.status_code == 403

    def test_requires_auth(self, client):
        """GET requires authentication."""
        response = client.get("/api/admin/settings/advanced")
        assert response.status_code == 401


class TestUpdateAdvancedSettings:
    """Tests for PUT /api/admin/settings/advanced."""

    def test_requires_confirm(self, client, admin_headers):
        """PUT requires confirm_reindex=true."""
        response = client.put(
            "/api/admin/settings/advanced",
            headers=admin_headers,
            json={"chunk_size": 300},
        )

        assert response.status_code == 400
        assert "confirm_reindex" in response.json()["detail"].lower()

    def test_updates_with_confirm(self, client, admin_headers):
        """PUT updates settings when confirm_reindex=true."""
        response = client.put(
            "/api/admin/settings/advanced",
            headers=admin_headers,
            json={"chunk_size": 300, "confirm_reindex": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["chunk_size"] == 300

    def test_requires_admin(self, client, user_headers):
        """PUT requires admin role."""
        response = client.put(
            "/api/admin/settings/advanced",
            headers=user_headers,
            json={"chunk_size": 300, "confirm_reindex": True},
        )
        assert response.status_code == 403


class TestReindexEstimate:
    """Tests for GET /api/admin/reindex/estimate."""

    def test_returns_estimate(self, client, admin_headers):
        """GET returns reindex time estimate."""
        response = client.get("/api/admin/reindex/estimate", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "avg_processing_time_ms" in data
        assert "estimated_total_seconds" in data
        assert "estimated_time_str" in data

    def test_requires_admin(self, client, user_headers):
        """GET requires admin role."""
        response = client.get("/api/admin/reindex/estimate", headers=user_headers)
        assert response.status_code == 403


class TestStartReindex:
    """Tests for POST /api/admin/reindex/start."""

    def test_requires_confirm(self, client, admin_headers):
        """POST requires confirm=true."""
        response = client.post(
            "/api/admin/reindex/start",
            headers=admin_headers,
            json={},
        )

        assert response.status_code == 400
        assert "confirm" in response.json()["detail"].lower()

    def test_creates_job(self, client, admin_headers):
        """POST creates reindex job."""
        response = client.post(
            "/api/admin/reindex/start",
            headers=admin_headers,
            json={"confirm": True},
        )

        assert response.status_code == 202
        data = response.json()
        assert "id" in data
        assert data["status"] == "pending"
        assert "total_documents" in data

    def test_prevents_concurrent_jobs(self, client, admin_headers):
        """POST returns 409 if job already running."""
        # Start first job
        response1 = client.post(
            "/api/admin/reindex/start",
            headers=admin_headers,
            json={"confirm": True},
        )
        assert response1.status_code == 202

        # Try to start second job
        response2 = client.post(
            "/api/admin/reindex/start",
            headers=admin_headers,
            json={"confirm": True},
        )
        assert response2.status_code == 409

    def test_requires_admin(self, client, user_headers):
        """POST requires admin role."""
        response = client.post(
            "/api/admin/reindex/start",
            headers=user_headers,
            json={"confirm": True},
        )
        assert response.status_code == 403


class TestReindexStatus:
    """Tests for GET /api/admin/reindex/status."""

    def test_returns_none_when_no_job(self, client, admin_headers):
        """GET returns null when no job running."""
        response = client.get("/api/admin/reindex/status", headers=admin_headers)

        assert response.status_code == 200
        # May be null or None
        assert response.json() is None or response.text == "null"

    def test_returns_active_job(self, client, admin_headers):
        """GET returns active job details."""
        # Start a job first
        client.post(
            "/api/admin/reindex/start",
            headers=admin_headers,
            json={"confirm": True},
        )

        response = client.get("/api/admin/reindex/status", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "status" in data
        assert data["status"] in ["pending", "running"]

    def test_requires_admin(self, client, user_headers):
        """GET requires admin role."""
        response = client.get("/api/admin/reindex/status", headers=user_headers)
        assert response.status_code == 403


class TestAbortReindex:
    """Tests for POST /api/admin/reindex/abort."""

    def test_returns_404_when_no_job(self, client, admin_headers):
        """POST returns 404 when no job to abort."""
        response = client.post("/api/admin/reindex/abort", headers=admin_headers)

        assert response.status_code == 404

    def test_aborts_active_job(self, client, admin_headers):
        """POST aborts active job."""
        # Start a job first
        start_response = client.post(
            "/api/admin/reindex/start",
            headers=admin_headers,
            json={"confirm": True},
        )
        assert start_response.status_code == 202

        # Abort it
        response = client.post("/api/admin/reindex/abort", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "aborted"

    def test_requires_admin(self, client, user_headers):
        """POST requires admin role."""
        response = client.post("/api/admin/reindex/abort", headers=user_headers)
        assert response.status_code == 403


class TestReindexHistory:
    """Tests for GET /api/admin/reindex/history."""

    def test_returns_history(self, client, admin_headers):
        """GET returns reindex job history."""
        response = client.get("/api/admin/reindex/history", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_respects_limit(self, client, admin_headers):
        """GET respects limit parameter."""
        response = client.get(
            "/api/admin/reindex/history?limit=5",
            headers=admin_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 5

    def test_requires_admin(self, client, user_headers):
        """GET requires admin role."""
        response = client.get("/api/admin/reindex/history", headers=user_headers)
        assert response.status_code == 403
