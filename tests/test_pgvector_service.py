"""Tests for PgVectorService (requires PostgreSQL + pgvector)."""

import pytest

from ai_ready_rag.services.pgvector_service import PgVectorService, SearchResult


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
