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

    def test_rejects_overlap_greater_than_size(self, client, admin_headers):
        """PUT rejects chunk_overlap >= chunk_size."""
        response = client.put(
            "/api/admin/settings/advanced",
            headers=admin_headers,
            json={
                "chunk_size": 200,
                "chunk_overlap": 250,  # Invalid: overlap > size
                "confirm_reindex": True,
            },
        )
        assert response.status_code == 400
        assert "chunk_overlap" in response.json()["detail"].lower()

    def test_rejects_overlap_equal_to_size(self, client, admin_headers):
        """PUT rejects chunk_overlap == chunk_size."""
        response = client.put(
            "/api/admin/settings/advanced",
            headers=admin_headers,
            json={
                "chunk_size": 200,
                "chunk_overlap": 200,  # Invalid: overlap == size
                "confirm_reindex": True,
            },
        )
        assert response.status_code == 400
        assert "chunk_overlap" in response.json()["detail"].lower()

    def test_accepts_valid_overlap(self, client, admin_headers):
        """PUT accepts chunk_overlap < chunk_size."""
        response = client.put(
            "/api/admin/settings/advanced",
            headers=admin_headers,
            json={
                "chunk_size": 256,
                "chunk_overlap": 64,  # Valid: overlap < size
                "confirm_reindex": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["chunk_size"] == 256
        assert data["chunk_overlap"] == 64


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


# =============================================================================
# Phase 3: Reindex Failure Handling Tests
# =============================================================================


class TestPauseReindex:
    """Tests for POST /api/admin/reindex/pause."""

    def test_returns_404_when_no_job(self, client, admin_headers):
        """POST returns 404 when no job to pause."""
        response = client.post("/api/admin/reindex/pause", headers=admin_headers)
        assert response.status_code == 404

    def test_pauses_running_job(self, client, admin_headers, db):
        """POST pauses a running job."""
        from ai_ready_rag.db.models import ReindexJob

        # Create a running job directly in DB
        job = ReindexJob(
            status="running",
            total_documents=10,
            processed_documents=5,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        response = client.post("/api/admin/reindex/pause", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "paused"
        assert data["paused_reason"] == "user_request"

    def test_requires_admin(self, client, user_headers):
        """POST requires admin role."""
        response = client.post("/api/admin/reindex/pause", headers=user_headers)
        assert response.status_code == 403


class TestResumeReindex:
    """Tests for POST /api/admin/reindex/resume."""

    def test_returns_404_when_no_job(self, client, admin_headers):
        """POST returns 404 when no job to resume."""
        response = client.post(
            "/api/admin/reindex/resume",
            headers=admin_headers,
            json={"action": "skip"},
        )
        assert response.status_code == 404

    def test_resumes_paused_job_with_skip(self, client, admin_headers, db):
        """POST resumes paused job with skip action."""
        from datetime import UTC, datetime

        from ai_ready_rag.db.models import ReindexJob

        # Create a paused job
        job = ReindexJob(
            status="paused",
            total_documents=10,
            processed_documents=5,
            current_document_id="doc-123",
            paused_at=datetime.now(UTC),
            paused_reason="failure",
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        response = client.post(
            "/api/admin/reindex/resume",
            headers=admin_headers,
            json={"action": "skip"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["paused_at"] is None

    def test_resumes_paused_job_with_retry(self, client, admin_headers, db):
        """POST resumes paused job with retry action."""
        from datetime import UTC, datetime

        from ai_ready_rag.db.models import ReindexJob

        job = ReindexJob(
            status="paused",
            total_documents=10,
            processed_documents=5,
            current_document_id="doc-123",
            paused_at=datetime.now(UTC),
            paused_reason="failure",
            retry_count=0,
            max_retries=3,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        response = client.post(
            "/api/admin/reindex/resume",
            headers=admin_headers,
            json={"action": "retry"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"

    def test_resumes_paused_job_with_skip_all(self, client, admin_headers, db):
        """POST resumes paused job with skip_all action."""
        from datetime import UTC, datetime

        from ai_ready_rag.db.models import ReindexJob

        job = ReindexJob(
            status="paused",
            total_documents=10,
            processed_documents=5,
            paused_at=datetime.now(UTC),
            paused_reason="failure",
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        response = client.post(
            "/api/admin/reindex/resume",
            headers=admin_headers,
            json={"action": "skip_all"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["auto_skip_failures"] is True

    def test_requires_admin(self, client, user_headers):
        """POST requires admin role."""
        response = client.post(
            "/api/admin/reindex/resume",
            headers=user_headers,
            json={"action": "skip"},
        )
        assert response.status_code == 403


class TestGetReindexFailures:
    """Tests for GET /api/admin/reindex/failures."""

    def test_returns_404_when_no_job(self, client, admin_headers):
        """GET returns 404 when no job found."""
        response = client.get("/api/admin/reindex/failures", headers=admin_headers)
        assert response.status_code == 404

    def test_returns_empty_failures_for_job_without_failures(self, client, admin_headers, db):
        """GET returns empty list when job has no failures."""
        from ai_ready_rag.db.models import ReindexJob

        job = ReindexJob(
            status="running",
            total_documents=10,
            processed_documents=5,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        response = client.get("/api/admin/reindex/failures", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job.id
        assert data["failures"] == []
        assert data["total_failures"] == 0

    def test_returns_failures_with_details(self, client, admin_headers, db):
        """GET returns failure details."""
        import json

        from ai_ready_rag.db.models import Document, ReindexJob

        # Create a document
        doc = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/uploads/test.pdf",
            file_type="application/pdf",
            file_size=1000,
            status="ready",
            uploaded_by="user-1",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # Create job with failure
        job = ReindexJob(
            status="paused",
            total_documents=10,
            processed_documents=5,
            failed_documents=1,
            failed_document_ids=json.dumps([doc.id]),
            current_document_id=doc.id,
            last_error="Embedding failed",
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        response = client.get("/api/admin/reindex/failures", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data["failures"]) == 1
        assert data["failures"][0]["document_id"] == doc.id
        assert data["failures"][0]["filename"] == "test.pdf"
        assert data["total_failures"] == 1

    def test_requires_admin(self, client, user_headers):
        """GET requires admin role."""
        response = client.get("/api/admin/reindex/failures", headers=user_headers)
        assert response.status_code == 403


class TestRetryDocument:
    """Tests for POST /api/admin/reindex/retry/{document_id}."""

    def test_returns_404_when_no_job(self, client, admin_headers):
        """POST returns 404 when no job found."""
        response = client.post(
            "/api/admin/reindex/retry/doc-123",
            headers=admin_headers,
        )
        assert response.status_code == 404

    def test_returns_400_when_doc_not_in_failed_list(self, client, admin_headers, db):
        """POST returns 400 when document not in failed list."""
        from ai_ready_rag.db.models import ReindexJob

        job = ReindexJob(
            status="paused",
            total_documents=10,
            processed_documents=5,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        response = client.post(
            "/api/admin/reindex/retry/doc-123",
            headers=admin_headers,
        )

        assert response.status_code == 400
        assert "not in failed list" in response.json()["detail"].lower()

    def test_marks_document_for_retry(self, client, admin_headers, db):
        """POST marks document for retry."""
        import json

        from ai_ready_rag.db.models import Document, ReindexJob

        # Create a document
        doc = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/uploads/test.pdf",
            file_type="application/pdf",
            file_size=1000,
            status="ready",
            uploaded_by="user-1",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # Create job with document in failed list
        job = ReindexJob(
            status="paused",
            total_documents=10,
            processed_documents=5,
            failed_documents=1,
            failed_document_ids=json.dumps([doc.id]),
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        response = client.post(
            f"/api/admin/reindex/retry/{doc.id}",
            headers=admin_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["current_document_id"] == doc.id
        assert data["failed_documents"] == 0

    def test_requires_admin(self, client, user_headers):
        """POST requires admin role."""
        response = client.post(
            "/api/admin/reindex/retry/doc-123",
            headers=user_headers,
        )
        assert response.status_code == 403
