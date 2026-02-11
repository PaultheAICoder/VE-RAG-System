"""Tests for ARQ infrastructure and Redis degraded mode."""

from unittest.mock import AsyncMock, patch

import pytest


class TestConfigHasRedisSettings:
    """Test that config includes Redis/ARQ settings."""

    def test_redis_url_default(self):
        """Config has redis_url with default."""
        from ai_ready_rag.config import Settings

        settings = Settings()
        assert settings.redis_url == "redis://localhost:6379"

    def test_arq_job_timeout_default(self):
        """Config has arq_job_timeout with default."""
        from ai_ready_rag.config import Settings

        settings = Settings()
        assert settings.arq_job_timeout == 600

    def test_arq_max_jobs_default(self):
        """Config has arq_max_jobs with default."""
        from ai_ready_rag.config import Settings

        settings = Settings()
        assert settings.arq_max_jobs == 2


class TestRedisPool:
    """Tests for Redis connection pool with degraded mode."""

    @pytest.mark.asyncio
    async def test_get_redis_pool_returns_none_when_unavailable(self):
        """get_redis_pool returns None when Redis is not running."""
        import ai_ready_rag.core.redis as redis_module

        # Reset cached pool and checked flag
        redis_module._redis_pool = None
        redis_module._redis_checked = False

        with patch(
            "ai_ready_rag.core.redis.create_pool",
            side_effect=ConnectionError("Connection refused"),
        ):
            pool = await redis_module.get_redis_pool()
            assert pool is None

        # Clean up
        redis_module._redis_pool = None
        redis_module._redis_checked = False

    @pytest.mark.asyncio
    async def test_is_redis_available_false_when_no_pool(self):
        """is_redis_available returns False when pool is None."""
        import ai_ready_rag.core.redis as redis_module

        redis_module._redis_pool = None
        redis_module._redis_checked = False

        with patch(
            "ai_ready_rag.core.redis.create_pool",
            side_effect=ConnectionError("Connection refused"),
        ):
            result = await redis_module.is_redis_available()
            assert result is False

        redis_module._redis_pool = None
        redis_module._redis_checked = False

    @pytest.mark.asyncio
    async def test_close_redis_pool_noop_when_none(self):
        """close_redis_pool is safe to call when pool is None."""
        import ai_ready_rag.core.redis as redis_module

        redis_module._redis_pool = None
        redis_module._redis_checked = False
        await redis_module.close_redis_pool()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_redis_pool_closes_connection(self):
        """close_redis_pool calls aclose on the pool."""
        import ai_ready_rag.core.redis as redis_module

        mock_pool = AsyncMock()
        redis_module._redis_pool = mock_pool

        await redis_module.close_redis_pool()

        mock_pool.aclose.assert_called_once()
        assert redis_module._redis_pool is None


class TestHealthEndpointRedis:
    """Test health endpoint includes Redis status."""

    def test_health_includes_redis_field(self, client):
        """Health endpoint includes redis field."""
        # Redis won't be running in tests, so should be 'unavailable'
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "redis" in data
        assert data["redis"] in ("connected", "unavailable")


class TestDocumentUploadDegradedMode:
    """Test document upload falls back to BackgroundTasks when no Redis."""

    @pytest.mark.asyncio
    async def test_enqueue_falls_back_to_background_tasks(self):
        """enqueue_document_processing falls back when Redis unavailable."""
        from ai_ready_rag.api.documents import enqueue_document_processing

        mock_bg_tasks = AsyncMock()
        mock_bg_tasks.add_task = lambda *args, **kwargs: None  # sync method

        with patch("ai_ready_rag.api.documents.get_redis_pool", return_value=None):
            job_id = await enqueue_document_processing("test-doc-id", mock_bg_tasks, None, False)
            assert job_id is None  # None means BackgroundTasks was used

    @pytest.mark.asyncio
    async def test_enqueue_uses_arq_when_available(self):
        """enqueue_document_processing uses ARQ when Redis is available."""
        from ai_ready_rag.api.documents import enqueue_document_processing

        mock_bg_tasks = AsyncMock()
        mock_redis = AsyncMock()
        mock_job = AsyncMock()
        mock_job.job_id = "arq-job-123"
        mock_redis.enqueue_job = AsyncMock(return_value=mock_job)

        with patch("ai_ready_rag.api.documents.get_redis_pool", return_value=mock_redis):
            job_id = await enqueue_document_processing("test-doc-id", mock_bg_tasks, None, False)
            assert job_id == "arq-job-123"
            mock_redis.enqueue_job.assert_called_once_with(
                "process_document", "test-doc-id", None, False
            )


class TestWorkerSettings:
    """Test ARQ WorkerSettings configuration."""

    def test_worker_settings_has_functions(self):
        """WorkerSettings class has functions list."""
        from ai_ready_rag.workers.settings import WorkerSettings

        assert hasattr(WorkerSettings, "functions")
        assert len(WorkerSettings.functions) > 0

    def test_worker_settings_has_redis_settings(self):
        """WorkerSettings class has redis_settings."""
        from ai_ready_rag.workers.settings import WorkerSettings

        assert hasattr(WorkerSettings, "redis_settings")

    def test_worker_settings_has_timeouts(self):
        """WorkerSettings class has job_timeout and max_jobs."""
        from ai_ready_rag.workers.settings import WorkerSettings

        assert WorkerSettings.job_timeout == 600
        assert WorkerSettings.max_jobs == 2

    def test_process_document_registered(self):
        """process_document task is registered in tasks package."""
        from ai_ready_rag.workers.tasks import process_document

        assert callable(process_document)
