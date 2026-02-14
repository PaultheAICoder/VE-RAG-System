"""Tests for ingestkit-excel adapter classes.

Verifies that VERagVectorStoreAdapter produces the correct Qdrant payload
schema matching VE-RAG's search expectations, and that factory functions
create valid backend instances.
"""

from unittest.mock import MagicMock, patch


class TestVERagVectorStoreAdapter:
    """Tests for the VE-RAG vector store adapter."""

    def _make_adapter(self, **overrides):
        """Create adapter with mocked QdrantClient."""
        from ai_ready_rag.services.ingestkit_adapters import VERagVectorStoreAdapter

        defaults = {
            "qdrant_url": "http://localhost:6333",
            "collection_name": "documents",
            "embedding_dimension": 768,
            "document_id": "doc-123",
            "document_name": "test.xlsx",
            "tags": ["hr", "finance"],
            "uploaded_by": "user-456",
            "tenant_id": "default",
        }
        defaults.update(overrides)

        with patch("ai_ready_rag.services.ingestkit_adapters.QdrantClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            adapter = VERagVectorStoreAdapter(**defaults)
            return adapter, mock_client

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

    def test_upsert_chunks_produces_verag_payload_schema(self):
        """Payload must contain all VE-RAG standard fields for search compatibility."""
        adapter, mock_client = self._make_adapter()
        chunk = self._make_chunk_payload()

        adapter.upsert_chunks("documents", [chunk])

        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args
        points = call_kwargs.kwargs.get("points") or call_kwargs[1].get("points")

        assert len(points) == 1
        payload = points[0].payload

        # VE-RAG standard fields (must match vector_service.py schema)
        assert payload["chunk_id"] == "chunk-1"
        assert payload["document_id"] == "doc-123"
        assert payload["document_name"] == "test.xlsx"
        assert payload["chunk_index"] == 0
        assert payload["chunk_text"] == "sample text"
        assert payload["tags"] == ["hr", "finance"]
        assert payload["tenant_id"] == "default"
        assert payload["uploaded_by"] == "user-456"
        assert "uploaded_at" in payload
        assert payload["page_number"] is None  # Excel files don't have pages
        assert payload["section"] == "Employee Data"

    def test_upsert_chunks_includes_ingestkit_provenance(self):
        """Payload must include ingestkit provenance fields prefixed with ingestkit_*."""
        adapter, mock_client = self._make_adapter()
        chunk = self._make_chunk_payload()

        adapter.upsert_chunks("documents", [chunk])

        points = mock_client.upsert.call_args.kwargs.get("points") or mock_client.upsert.call_args[
            1
        ].get("points")
        payload = points[0].payload

        assert payload["ingestkit_source_format"] == "xlsx"
        assert payload["ingestkit_ingestion_method"] == "sql_agent"
        assert payload["ingestkit_parser_version"] == "ingestkit_excel:1.0.0"
        assert payload["ingestkit_ingest_key"] == "key-789"
        assert payload["ingestkit_chunk_hash"] == "abc123"
        assert payload["ingestkit_ingest_run_id"] == "run-001"
        assert payload["ingestkit_table_name"] == "employees"
        assert payload["ingestkit_row_count"] == 100
        assert payload["ingestkit_columns"] == ["name", "salary"]

    def test_upsert_chunks_uses_verag_collection_name(self):
        """Adapter must ignore ingestkit's collection param and use VE-RAG's."""
        adapter, mock_client = self._make_adapter(collection_name="my_collection")
        chunk = self._make_chunk_payload()

        # Pass ingestkit's default collection name - should be overridden
        adapter.upsert_chunks("helpdesk", [chunk])

        call_kwargs = mock_client.upsert.call_args
        assert (
            call_kwargs.kwargs.get("collection_name")
            or call_kwargs[1].get("collection_name") == "my_collection"
        )

    def test_upsert_chunks_empty_returns_zero(self):
        """Empty chunk list should return 0 without calling Qdrant."""
        adapter, mock_client = self._make_adapter()

        result = adapter.upsert_chunks("documents", [])

        assert result == 0
        mock_client.upsert.assert_not_called()

    def test_upsert_chunks_returns_count(self):
        """upsert_chunks must return the number of points upserted."""
        adapter, mock_client = self._make_adapter()
        chunks = [self._make_chunk_payload(chunk_id=f"chunk-{i}", chunk_index=i) for i in range(3)]

        result = adapter.upsert_chunks("documents", chunks)

        assert result == 3

    def test_ensure_collection_creates_when_missing(self):
        """ensure_collection should create collection if it doesn't exist."""
        adapter, mock_client = self._make_adapter()
        mock_client.collection_exists.return_value = False

        adapter.ensure_collection("documents", 768)

        mock_client.create_collection.assert_called_once()

    def test_ensure_collection_skips_when_exists(self):
        """ensure_collection should skip creation when collection exists."""
        adapter, mock_client = self._make_adapter()
        mock_client.collection_exists.return_value = True

        adapter.ensure_collection("documents", 768)

        mock_client.create_collection.assert_not_called()

    def test_delete_by_ids_empty_returns_zero(self):
        """Empty ID list should return 0 without calling Qdrant."""
        adapter, mock_client = self._make_adapter()

        result = adapter.delete_by_ids("documents", [])

        assert result == 0
        mock_client.delete.assert_not_called()

    def test_delete_by_ids_returns_count(self):
        """delete_by_ids should return count of IDs passed."""
        adapter, mock_client = self._make_adapter()

        result = adapter.delete_by_ids("documents", ["id-1", "id-2"])

        assert result == 2
        mock_client.delete.assert_called_once()


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
        """create_llm_adapter should return OllamaLLM instance."""
        from ai_ready_rag.services.ingestkit_adapters import create_llm_adapter

        adapter = create_llm_adapter(ollama_url="http://localhost:11434")
        assert hasattr(adapter, "classify")
        assert hasattr(adapter, "generate")

    def test_create_structured_db(self):
        """create_structured_db should return SQLiteStructuredDB with in-memory DB."""
        from ai_ready_rag.services.ingestkit_adapters import create_structured_db

        db = create_structured_db(db_path=":memory:")
        assert db.get_connection_uri() == "sqlite:///:memory:"
        assert db.table_exists("nonexistent") is False
