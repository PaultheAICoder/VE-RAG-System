"""Tests for ARQ cache warming and reindex task migration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestWarmCacheTask:
    """Tests for the warm_cache ARQ task."""

    def test_warm_cache_is_callable(self):
        """warm_cache task function exists and is callable."""
        from ai_ready_rag.workers.tasks.warming import warm_cache

        assert callable(warm_cache)

    def test_warm_cache_registered_in_tasks_package(self):
        """warm_cache is exported from workers.tasks."""
        from ai_ready_rag.workers.tasks import warm_cache

        assert callable(warm_cache)

    @pytest.mark.asyncio
    async def test_warm_cache_returns_success_dict(self):
        """warm_cache returns dict with success, warmed, total keys."""
        from ai_ready_rag.workers.tasks.warming import warm_cache

        mock_db = MagicMock()
        mock_vector_service = AsyncMock()
        mock_rag_service = AsyncMock()

        with (
            patch("ai_ready_rag.db.database.SessionLocal", return_value=mock_db),
            patch(
                "ai_ready_rag.services.factory.get_vector_service", return_value=mock_vector_service
            ),
            patch("ai_ready_rag.services.rag_service.RAGService", return_value=mock_rag_service),
        ):
            ctx = {
                "settings": MagicMock(warming_delay_seconds=0),
                "vector_service": mock_vector_service,
            }
            result = await warm_cache(ctx, queries=["test query"], triggered_by="user-123")

            assert result["success"] is True
            assert result["warmed"] == 1
            assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_warm_cache_handles_query_failure(self):
        """warm_cache handles individual query failures gracefully."""
        from ai_ready_rag.workers.tasks.warming import warm_cache

        mock_db = MagicMock()
        mock_vector_service = AsyncMock()
        mock_rag_service = AsyncMock()
        mock_rag_service.generate = AsyncMock(side_effect=Exception("Ollama timeout"))

        with (
            patch("ai_ready_rag.db.database.SessionLocal", return_value=mock_db),
            patch(
                "ai_ready_rag.services.factory.get_vector_service", return_value=mock_vector_service
            ),
            patch("ai_ready_rag.services.rag_service.RAGService", return_value=mock_rag_service),
        ):
            ctx = {
                "settings": MagicMock(warming_delay_seconds=0),
                "vector_service": mock_vector_service,
            }
            result = await warm_cache(ctx, queries=["fail query"], triggered_by="user-123")

            assert result["success"] is True
            assert result["warmed"] == 0
            assert result["total"] == 1


class TestReindexKnowledgeBaseTask:
    """Tests for the reindex_knowledge_base ARQ task."""

    def test_reindex_task_is_callable(self):
        """reindex_knowledge_base task function exists and is callable."""
        from ai_ready_rag.workers.tasks.reindex import reindex_knowledge_base

        assert callable(reindex_knowledge_base)

    def test_reindex_task_registered_in_tasks_package(self):
        """reindex_knowledge_base is exported from workers.tasks."""
        from ai_ready_rag.workers.tasks import reindex_knowledge_base

        assert callable(reindex_knowledge_base)

    @pytest.mark.asyncio
    async def test_reindex_returns_success_dict(self):
        """reindex_knowledge_base returns dict with success and job_id."""
        from ai_ready_rag.workers.tasks.reindex import reindex_knowledge_base

        with patch(
            "ai_ready_rag.services.reindex_worker.run_reindex_job",
            new_callable=AsyncMock,
        ):
            ctx = {}
            result = await reindex_knowledge_base(ctx, job_id="job-123")

            assert result["success"] is True
            assert result["job_id"] == "job-123"

    @pytest.mark.asyncio
    async def test_reindex_returns_failure_on_exception(self):
        """reindex_knowledge_base returns failure dict on exception."""
        from ai_ready_rag.workers.tasks.reindex import reindex_knowledge_base

        with patch(
            "ai_ready_rag.services.reindex_worker.run_reindex_job",
            new_callable=AsyncMock,
            side_effect=Exception("DB connection lost"),
        ):
            ctx = {}
            result = await reindex_knowledge_base(ctx, job_id="job-456")

            assert result["success"] is False
            assert result["job_id"] == "job-456"
            assert "DB connection lost" in result["error"]


class TestWorkerSettingsRegistration:
    """Test new tasks are registered in WorkerSettings."""

    def test_worker_settings_includes_warm_cache(self):
        """WorkerSettings.functions includes warm_cache."""
        from ai_ready_rag.workers.settings import WorkerSettings
        from ai_ready_rag.workers.tasks.warming import warm_cache

        assert warm_cache in WorkerSettings.functions

    def test_worker_settings_includes_reindex(self):
        """WorkerSettings.functions includes reindex_knowledge_base."""
        from ai_ready_rag.workers.settings import WorkerSettings
        from ai_ready_rag.workers.tasks.reindex import reindex_knowledge_base

        assert reindex_knowledge_base in WorkerSettings.functions

    def test_worker_settings_has_all_tasks(self):
        """WorkerSettings.functions has all 4 registered tasks."""
        from ai_ready_rag.workers.settings import WorkerSettings

        assert len(WorkerSettings.functions) == 4

    def test_get_worker_settings_includes_new_tasks(self):
        """get_worker_settings() returns all 4 tasks."""
        from ai_ready_rag.workers.settings import get_worker_settings

        config = get_worker_settings()
        assert len(config["functions"]) == 4


class TestAdminReindexARQEnqueue:
    """Test reindex start endpoint uses ARQ with fallback."""

    def test_reindex_start_with_arq(self, client, admin_headers, db):
        """Reindex start uses ARQ when Redis available."""
        mock_redis = AsyncMock()
        mock_redis.enqueue_job = AsyncMock(return_value=AsyncMock(job_id="arq-reindex-1"))

        with patch("ai_ready_rag.api.admin.get_redis_pool", return_value=mock_redis):
            response = client.post(
                "/api/admin/reindex/start",
                json={"confirm": True},
                headers=admin_headers,
            )

        assert response.status_code == 202
        mock_redis.enqueue_job.assert_called_once()
        call_args = mock_redis.enqueue_job.call_args
        assert call_args[0][0] == "reindex_knowledge_base"

    def test_reindex_start_fallback_no_redis(self, client, admin_headers, db):
        """Reindex start falls back to BackgroundTasks when no Redis."""
        with patch("ai_ready_rag.api.admin.get_redis_pool", return_value=None):
            response = client.post(
                "/api/admin/reindex/start",
                json={"confirm": True},
                headers=admin_headers,
            )

        assert response.status_code == 202


class TestAdminWarmCacheARQEnqueue:
    """Test cache warm endpoint (now returns 410 Gone - legacy endpoint removed)."""

    def test_warm_cache_with_arq(self, client, admin_headers, db):
        """Legacy cache warm endpoint returns 410 Gone."""
        response = client.post(
            "/api/admin/cache/warm",
            json={"queries": ["What is RAG?", "How to use vectors?"]},
            headers=admin_headers,
        )

        assert response.status_code == 410

    def test_warm_cache_fallback_no_redis(self, client, admin_headers, db):
        """Legacy cache warm endpoint returns 410 Gone."""
        response = client.post(
            "/api/admin/cache/warm",
            json={"queries": ["test query"]},
            headers=admin_headers,
        )

        assert response.status_code == 410

    def test_warm_cache_empty_queries_rejected(self, client, admin_headers, db):
        """Legacy cache warm endpoint returns 410 Gone."""
        response = client.post(
            "/api/admin/cache/warm",
            json={"queries": []},
            headers=admin_headers,
        )
        assert response.status_code == 410
