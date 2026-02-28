"""Tests for PgVectorService (requires PostgreSQL + pgvector)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_ready_rag.services.pgvector_service import PgVectorService
from ai_ready_rag.services.vector_types import SearchResult


class TestPgVectorServiceImport:
    """These tests don't need a database — just verify the class loads."""

    def test_class_importable(self):
        assert PgVectorService is not None

    def test_search_result_dataclass(self):
        r = SearchResult(
            chunk_id="id1",
            document_id="doc1",
            document_name="test.pdf",
            chunk_text="hello",
            chunk_index=0,
            score=0.9,
            page_number=1,
            section=None,
        )
        assert r.score == 0.9

    def test_pgvector_service_instantiation(self):
        """Verify PgVectorService can be instantiated with required params."""
        svc = PgVectorService(
            database_url="postgresql://user:pass@localhost/testdb",
            ollama_url="http://localhost:11434",
            embedding_model="nomic-embed-text",
            embedding_dimension=768,
            tenant_id="default",
        )
        assert svc._embedding_model == "nomic-embed-text"
        assert svc._embedding_dimension == 768
        assert svc._tenant_id == "default"

    def test_pgvector_service_default_params(self):
        """Verify PgVectorService default parameter values."""
        svc = PgVectorService(database_url="sqlite:///test.db")
        assert svc._ollama_url == "http://localhost:11434"
        assert svc._embedding_model == "nomic-embed-text"
        assert svc._embedding_dimension == 768
        assert svc._tenant_id == "default"

    def test_search_result_optional_tags(self):
        """SearchResult.tags defaults to None."""
        r = SearchResult(
            chunk_id="id2",
            document_id="doc2",
            document_name="other.pdf",
            chunk_text="world",
            chunk_index=1,
            score=0.75,
            page_number=None,
            section="Introduction",
        )
        assert r.tags is None
        assert r.section == "Introduction"

    def test_has_embed_method(self):
        """PgVectorService exposes public embed() method."""
        svc = PgVectorService(database_url="sqlite:///test.db")
        assert callable(getattr(svc, "embed", None))

    def test_has_get_extended_stats_method(self):
        """PgVectorService exposes get_extended_stats() method."""
        svc = PgVectorService(database_url="sqlite:///test.db")
        assert callable(getattr(svc, "get_extended_stats", None))

    def test_has_refresh_capabilities_method(self):
        """PgVectorService exposes refresh_capabilities() method."""
        svc = PgVectorService(database_url="sqlite:///test.db")
        assert callable(getattr(svc, "refresh_capabilities", None))


class TestPgVectorServiceNewMethods:
    """Unit tests for the 3 new methods — uses mocks, no DB required."""

    @pytest.fixture
    def svc(self):
        return PgVectorService(database_url="sqlite:///test.db", tenant_id="test-tenant")

    @pytest.mark.asyncio
    async def test_embed_delegates_to_private_embed(self, svc):
        """embed() is a thin wrapper around _embed()."""
        svc._embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        result = await svc.embed("hello world")
        svc._embed.assert_called_once_with("hello world")
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_refresh_capabilities_returns_pgvector_backend(self, svc):
        """refresh_capabilities() returns pgvector backend descriptor."""
        result = await svc.refresh_capabilities()
        assert result["backend"] == "pgvector"
        assert "vector_search" in result["capabilities"]
        assert isinstance(result["capabilities"], list)

    @pytest.mark.asyncio
    async def test_get_extended_stats_empty_db(self, svc):
        """get_extended_stats() returns zero totals when chunk_vectors is empty."""
        mock_db = MagicMock()
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)
        mock_db.execute.return_value.fetchall.return_value = []

        with patch("ai_ready_rag.services.pgvector_service.SessionLocal", return_value=mock_db):
            result = await svc.get_extended_stats()

        assert result["total_chunks"] == 0
        assert result["unique_files"] == 0
        assert result["files"] == []
        assert result["collection_name"] == "chunk_vectors"

    @pytest.mark.asyncio
    async def test_get_extended_stats_with_data(self, svc):
        """get_extended_stats() aggregates file counts correctly."""
        mock_db = MagicMock()
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)
        # Simulate 2 documents: 10 + 5 chunks
        mock_db.execute.return_value.fetchall.return_value = [
            ("doc-1", "policy.pdf", 10),
            ("doc-2", "guide.pdf", 5),
        ]

        with patch("ai_ready_rag.services.pgvector_service.SessionLocal", return_value=mock_db):
            result = await svc.get_extended_stats()

        assert result["total_chunks"] == 15
        assert result["unique_files"] == 2
        assert len(result["files"]) == 2
        filenames = [f["filename"] for f in result["files"]]
        assert "policy.pdf" in filenames


@pytest.mark.requires_postgres
class TestPgVectorServiceIntegration:
    """Integration tests requiring PostgreSQL + pgvector.

    These tests are skipped unless DATABASE_URL is set and points to a
    PostgreSQL instance with the pgvector extension installed.
    Run with: pytest -m requires_postgres
    """

    @pytest.fixture
    def pg_db(self):
        """Skip if no PostgreSQL DATABASE_URL configured."""
        import os

        database_url = os.environ.get("DATABASE_URL", "")
        if not database_url or not database_url.startswith("postgresql"):
            pytest.skip("Requires PostgreSQL DATABASE_URL env var (postgresql://...)")
        return database_url

    def test_initialize(self, pg_db):
        """Verify pgvector service initializes without error."""
        import asyncio

        svc = PgVectorService(
            database_url=pg_db,
            ollama_url="http://localhost:11434",
        )
        asyncio.get_event_loop().run_until_complete(svc.initialize())
