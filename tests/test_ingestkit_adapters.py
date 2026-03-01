"""Tests for ingestkit adapter classes.

Verifies that VERagVectorStoreAdapter produces the correct pgvector payload
schema and that factory functions create valid backend instances.
"""

from unittest.mock import MagicMock, patch


class TestVERagVectorStoreAdapter:
    """Tests for the VE-RAG pgvector store adapter."""

    def _make_adapter(self, **overrides):
        """Create adapter with mocked psycopg2."""
        from ai_ready_rag.services.ingestkit_adapters import VERagVectorStoreAdapter

        defaults = {
            "database_url": "postgresql://fake/db",
            "embedding_dimension": 768,
            "document_id": "doc-123",
            "document_name": "test.xlsx",
            "tags": ["hr", "finance"],
            "uploaded_by": "user-456",
            "tenant_id": "default",
        }
        defaults.update(overrides)
        return VERagVectorStoreAdapter(**defaults)

    def _make_chunk_payload(self, chunk_id="chunk-1", text="sample text", chunk_index=0):
        """Create a mock ChunkPayload matching ingestkit's model."""
        from ingestkit_core.models import BaseChunkMetadata, ChunkPayload

        metadata = BaseChunkMetadata(
            source_uri="file:///test.xlsx",
            source_format="xlsx",
            ingestion_method="sql_agent",
            parser_version="ingestkit_excel:1.0.0",
            chunk_index=chunk_index,
            chunk_hash="abc123",
            ingest_key="key-789",
            ingest_run_id="run-001",
            tenant_id="default",
            table_name="employees",
            row_count=100,
            columns=["name", "salary"],
            section_title="Employee Data",
        )

        return ChunkPayload(
            id=chunk_id,
            text=text,
            vector=[0.1] * 768,
            metadata=metadata,
        )

    def test_upsert_chunks_inserts_to_chunk_vectors(self):
        """upsert_chunks writes rows to chunk_vectors table via psycopg2."""
        adapter = self._make_adapter()
        chunk = self._make_chunk_payload()

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg2.connect", return_value=mock_conn):
            result = adapter.upsert_chunks("ignored_collection", [chunk])

        assert result == 1
        # Should have called DELETE (cleanup) + INSERT
        assert mock_cur.execute.call_count >= 2
        all_sqls = [c[0][0] for c in mock_cur.execute.call_args_list]
        assert any("DELETE FROM chunk_vectors" in sql for sql in all_sqls)
        assert any("INSERT INTO chunk_vectors" in sql for sql in all_sqls)

    def test_upsert_chunks_includes_verag_metadata_fields(self):
        """Inserted row metadata_ JSON must contain VE-RAG standard fields."""
        import json

        adapter = self._make_adapter()
        chunk = self._make_chunk_payload()

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg2.connect", return_value=mock_conn):
            adapter.upsert_chunks("col", [chunk])

        # Find the INSERT call and check metadata_ JSON
        insert_call = next(
            c for c in mock_cur.execute.call_args_list if "INSERT INTO chunk_vectors" in c[0][0]
        )
        params = insert_call[0][1]
        # metadata_ is the 5th param (index 4) in the INSERT
        metadata = json.loads(params[4])

        assert metadata["tags"] == ["hr", "finance"]
        assert metadata["document_name"] == "test.xlsx"
        assert metadata["uploaded_by"] == "user-456"
        assert "uploaded_at" in metadata
        assert metadata["page_number"] is None
        assert metadata["section"] == "Employee Data"

    def test_upsert_chunks_includes_ingestkit_provenance(self):
        """Inserted metadata_ must include ingestkit_* provenance fields."""
        import json

        adapter = self._make_adapter()
        chunk = self._make_chunk_payload()

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg2.connect", return_value=mock_conn):
            adapter.upsert_chunks("col", [chunk])

        insert_call = next(
            c for c in mock_cur.execute.call_args_list if "INSERT INTO chunk_vectors" in c[0][0]
        )
        metadata = json.loads(insert_call[0][1][4])

        assert metadata["ingestkit_source_format"] == "xlsx"
        assert metadata["ingestkit_ingestion_method"] == "sql_agent"
        assert metadata["ingestkit_parser_version"] == "ingestkit_excel:1.0.0"
        assert metadata["ingestkit_ingest_key"] == "key-789"
        assert metadata["ingestkit_chunk_hash"] == "abc123"
        assert metadata["ingestkit_ingest_run_id"] == "run-001"
        assert metadata["ingestkit_table_name"] == "employees"
        assert metadata["ingestkit_row_count"] == 100
        assert metadata["ingestkit_columns"] == ["name", "salary"]

    def test_upsert_chunks_uses_tenant_id(self):
        """Inserted row includes correct tenant_id."""
        adapter = self._make_adapter(tenant_id="acme_corp")
        chunk = self._make_chunk_payload()

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg2.connect", return_value=mock_conn):
            adapter.upsert_chunks("col", [chunk])

        insert_call = next(
            c for c in mock_cur.execute.call_args_list if "INSERT INTO chunk_vectors" in c[0][0]
        )
        params = insert_call[0][1]
        # tenant_id is the 6th param (index 5)
        assert params[5] == "acme_corp"

    def test_upsert_chunks_empty_returns_zero(self):
        """Empty chunk list should return 0 without calling psycopg2."""
        adapter = self._make_adapter()

        with patch("psycopg2.connect") as mock_connect:
            result = adapter.upsert_chunks("col", [])

        assert result == 0
        mock_connect.assert_not_called()

    def test_upsert_chunks_returns_count(self):
        """upsert_chunks must return the number of chunks upserted."""
        adapter = self._make_adapter()
        chunks = [self._make_chunk_payload(chunk_id=f"chunk-{i}", chunk_index=i) for i in range(3)]

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg2.connect", return_value=mock_conn):
            result = adapter.upsert_chunks("col", chunks)

        assert result == 3

    def test_ensure_collection_is_noop(self):
        """ensure_collection is a no-op (chunk_vectors table is Alembic-managed)."""
        adapter = self._make_adapter()

        with patch("psycopg2.connect") as mock_connect:
            adapter.ensure_collection("col", 768)

        mock_connect.assert_not_called()

    def test_delete_by_ids_empty_returns_zero(self):
        """Empty ID list should return 0 without calling psycopg2."""
        adapter = self._make_adapter()

        with patch("psycopg2.connect") as mock_connect:
            result = adapter.delete_by_ids("col", [])

        assert result == 0
        mock_connect.assert_not_called()

    def test_delete_by_ids_returns_count(self):
        """delete_by_ids returns rowcount from DELETE."""
        adapter = self._make_adapter()

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.rowcount = 2
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg2.connect", return_value=mock_conn):
            result = adapter.delete_by_ids("col", ["id-1", "id-2"])

        assert result == 2
        all_sqls = [c[0][0] for c in mock_cur.execute.call_args_list]
        assert any("DELETE FROM chunk_vectors" in sql for sql in all_sqls)


class TestFactoryFunctions:
    """Tests for adapter factory functions."""

    def test_create_embedding_adapter(self):
        """create_embedding_adapter should return OllamaEmbedding instance."""
        from ai_ready_rag.services.ingestkit_adapters import create_embedding_adapter

        adapter = create_embedding_adapter(
            ollama_url="http://localhost:11434",
            embedding_model="nomic-embed-text",
            embedding_dimension=768,
        )
        assert adapter.dimension() == 768

    def test_create_llm_adapter(self):
        """create_llm_adapter returns VERagClaudeLLM (Claude CLI primary LLM)."""
        from ai_ready_rag.services.ingestkit_adapters import VERagClaudeLLM, create_llm_adapter

        adapter = create_llm_adapter()
        assert isinstance(adapter, VERagClaudeLLM)

    def test_create_structured_db(self):
        """create_structured_db returns VERagPostgresStructuredDB for PostgreSQL."""
        from ai_ready_rag.services.ingestkit_adapters import (
            VERagPostgresStructuredDB,
            create_structured_db,
        )

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg2.connect", return_value=mock_conn):
            db = create_structured_db(database_url="postgresql://fake/db")

        assert isinstance(db, VERagPostgresStructuredDB)
