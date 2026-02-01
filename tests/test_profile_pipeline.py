"""Tests for ENV_PROFILE pipeline configuration.

Tests profile-based Settings, factory return types, chunker and vector
service implementations, and integration.
"""

import importlib.util
import os
from unittest.mock import patch

import pytest

# Check if optional dependencies are available using importlib.util.find_spec
HAS_CHROMADB = importlib.util.find_spec("chromadb") is not None
HAS_LANGCHAIN = importlib.util.find_spec("langchain_text_splitters") is not None


class TestProfileSettings:
    """Tests for profile-based Settings configuration."""

    def test_laptop_profile_defaults(self):
        """Laptop profile applies correct defaults."""
        from ai_ready_rag.config import Settings, get_settings

        get_settings.cache_clear()
        # Create settings with explicit profile and None for overridable fields
        # This tests the profile default logic without .env interference
        settings = Settings(
            env_profile="laptop",
            vector_backend=None,
            chunker_backend=None,
            enable_ocr=None,
            chat_model=None,
        )

        assert settings.env_profile == "laptop"
        assert settings.vector_backend == "chroma"
        assert settings.chunker_backend == "simple"
        assert settings.enable_ocr is False
        assert settings.chat_model == "llama3.2:latest"

    def test_spark_profile_defaults(self):
        """Spark profile applies correct defaults."""
        with patch.dict(
            os.environ,
            {"ENV_PROFILE": "spark"},
            clear=False,
        ):
            from ai_ready_rag.config import Settings, get_settings

            get_settings.cache_clear()
            settings = Settings()

            assert settings.env_profile == "spark"
            assert settings.vector_backend == "qdrant"
            assert settings.chunker_backend == "docling"
            assert settings.enable_ocr is True
            assert settings.chat_model == "qwen3:8b"

    def test_env_overrides_profile_default(self):
        """Individual env vars override profile defaults."""
        with patch.dict(
            os.environ,
            {
                "ENV_PROFILE": "laptop",
                "VECTOR_BACKEND": "qdrant",  # Override
            },
            clear=False,
        ):
            from ai_ready_rag.config import Settings, get_settings

            get_settings.cache_clear()
            settings = Settings()

            assert settings.env_profile == "laptop"
            assert settings.vector_backend == "qdrant"  # Overridden
            assert settings.chunker_backend == "simple"  # Still default

    def test_missing_profile_defaults_to_laptop(self):
        """Missing ENV_PROFILE defaults to laptop."""
        # Don't set ENV_PROFILE, let it use default
        from ai_ready_rag.config import Settings, get_settings

        get_settings.cache_clear()
        settings = Settings()

        assert settings.env_profile == "laptop"

    def test_profile_defaults_dict_complete(self):
        """PROFILE_DEFAULTS has all required keys for both profiles."""
        from ai_ready_rag.config import PROFILE_DEFAULTS

        required_keys = {
            "vector_backend",
            "chunker_backend",
            "enable_ocr",
            "chat_model",
            "embedding_model",
            "rag_max_context_tokens",
            "rag_max_history_tokens",
            "rag_max_response_tokens",
        }

        assert "laptop" in PROFILE_DEFAULTS
        assert "spark" in PROFILE_DEFAULTS
        assert required_keys.issubset(PROFILE_DEFAULTS["laptop"].keys())
        assert required_keys.issubset(PROFILE_DEFAULTS["spark"].keys())


class TestVectorServiceFactory:
    """Tests for get_vector_service factory."""

    @pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")
    def test_laptop_returns_chroma(self):
        """Laptop profile returns ChromaVectorService."""
        from ai_ready_rag.config import Settings
        from ai_ready_rag.services.factory import get_vector_service
        from ai_ready_rag.services.vector_chroma import ChromaVectorService

        settings = Settings(env_profile="laptop", vector_backend="chroma")
        service = get_vector_service(settings)

        assert isinstance(service, ChromaVectorService)

    def test_spark_returns_qdrant(self):
        """Spark profile returns VectorService (Qdrant)."""
        from ai_ready_rag.config import Settings
        from ai_ready_rag.services.factory import get_vector_service
        from ai_ready_rag.services.vector_service import VectorService

        settings = Settings(env_profile="spark", vector_backend="qdrant")
        service = get_vector_service(settings)

        assert isinstance(service, VectorService)

    def test_none_backend_raises(self):
        """None vector_backend raises ValueError."""
        from ai_ready_rag.config import Settings
        from ai_ready_rag.services.factory import get_vector_service

        # Force None by setting to None after initialization
        settings = Settings(env_profile="laptop")
        object.__setattr__(settings, "vector_backend", None)

        with pytest.raises(ValueError, match="not configured"):
            get_vector_service(settings)


class TestChunkerFactory:
    """Tests for get_chunker factory."""

    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain not installed")
    def test_laptop_returns_simple(self):
        """Laptop profile returns SimpleChunker."""
        from ai_ready_rag.config import Settings
        from ai_ready_rag.services.chunker_simple import SimpleChunker
        from ai_ready_rag.services.factory import get_chunker

        settings = Settings(env_profile="laptop", chunker_backend="simple")
        chunker = get_chunker(settings)

        assert isinstance(chunker, SimpleChunker)

    def test_spark_returns_docling(self):
        """Spark profile returns DoclingChunker."""
        pytest.importorskip("docling")  # Skip if docling not installed

        from ai_ready_rag.config import Settings
        from ai_ready_rag.services.chunker_docling import DoclingChunker
        from ai_ready_rag.services.factory import get_chunker

        settings = Settings(env_profile="spark", chunker_backend="docling")
        chunker = get_chunker(settings)

        assert isinstance(chunker, DoclingChunker)


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain not installed")
class TestSimpleChunker:
    """Tests for SimpleChunker implementation."""

    def test_chunk_text_basic(self):
        """SimpleChunker chunks plain text correctly."""
        from ai_ready_rag.services.chunker_simple import SimpleChunker

        chunker = SimpleChunker(chunk_size=100, chunk_overlap=10)
        text = "Hello world. " * 50  # ~650 chars

        chunks = chunker.chunk_text(text, source="test.txt")

        assert len(chunks) > 1
        assert all("text" in c for c in chunks)
        assert all("chunk_index" in c for c in chunks)
        assert chunks[0]["chunk_index"] == 0

    def test_chunk_text_empty(self):
        """SimpleChunker returns empty list for empty text."""
        from ai_ready_rag.services.chunker_simple import SimpleChunker

        chunker = SimpleChunker()
        chunks = chunker.chunk_text("", source="test.txt")

        assert chunks == []

    def test_chunk_text_preserves_source(self):
        """SimpleChunker preserves source in chunk metadata."""
        from ai_ready_rag.services.chunker_simple import SimpleChunker

        chunker = SimpleChunker()
        chunks = chunker.chunk_text("Some test content.", source="my-source.txt")

        assert chunks[0]["source"] == "my-source.txt"

    def test_chunk_txt_file(self, tmp_path):
        """SimpleChunker chunks plain text file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is test content. " * 100)

        from ai_ready_rag.services.chunker_simple import SimpleChunker

        chunker = SimpleChunker()
        chunks = chunker.chunk_document(str(txt_file))

        assert len(chunks) > 0
        assert chunks[0]["source"] == "test.txt"

    def test_chunk_md_file(self, tmp_path):
        """SimpleChunker chunks markdown file."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Heading\n\nSome content here. " * 50)

        from ai_ready_rag.services.chunker_simple import SimpleChunker

        chunker = SimpleChunker()
        chunks = chunker.chunk_document(str(md_file))

        assert len(chunks) > 0
        assert chunks[0]["source"] == "test.md"

    def test_chunk_csv_file(self, tmp_path):
        """SimpleChunker extracts and chunks CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,value\nalice,100\nbob,200\n" * 50)

        from ai_ready_rag.services.chunker_simple import SimpleChunker

        chunker = SimpleChunker()
        chunks = chunker.chunk_document(str(csv_file))

        assert len(chunks) > 0
        # CSV content should be comma-joined rows
        assert "alice" in chunks[0]["text"] or any("alice" in c["text"] for c in chunks)

    def test_chunk_document_file_not_found(self):
        """SimpleChunker raises FileNotFoundError for missing file."""
        from ai_ready_rag.services.chunker_simple import SimpleChunker

        chunker = SimpleChunker()

        with pytest.raises(FileNotFoundError):
            chunker.chunk_document("/nonexistent/path/file.txt")

    def test_chunk_document_includes_metadata(self, tmp_path):
        """SimpleChunker includes provided metadata in chunks."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Test content for chunking.")

        from ai_ready_rag.services.chunker_simple import SimpleChunker

        chunker = SimpleChunker()
        chunks = chunker.chunk_document(str(txt_file), metadata={"custom_key": "custom_value"})

        assert chunks[0]["custom_key"] == "custom_value"


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")
class TestChromaVectorService:
    """Tests for ChromaVectorService implementation."""

    @pytest.fixture
    def chroma_service(self, tmp_path):
        """Create ChromaVectorService with temp directory."""
        from ai_ready_rag.services.vector_chroma import ChromaVectorService

        service = ChromaVectorService(
            persist_dir=str(tmp_path / "chroma_db"),
            collection_name="test_collection",
        )
        return service

    @pytest.mark.asyncio
    async def test_initialize_creates_collection(self, chroma_service):
        """Initialize creates collection."""
        await chroma_service.initialize()

        assert chroma_service._collection is not None
        assert chroma_service.collection.name == "test_collection"

    @pytest.mark.asyncio
    async def test_health_check(self, chroma_service):
        """Health check returns status."""
        await chroma_service.initialize()
        health = await chroma_service.health_check()

        assert health["healthy"] is True
        assert health["backend"] == "chroma"
        assert "vector_count" in health["details"]

    @pytest.mark.asyncio
    async def test_clear_collection(self, chroma_service):
        """Clear collection removes all vectors."""
        await chroma_service.initialize()
        result = await chroma_service.clear_collection()

        assert result is True


class TestHealthEndpoint:
    """Tests for health endpoint backend info."""

    def test_health_returns_profile_info(self, client):
        """Health endpoint returns profile and backend info."""
        response = client.get("/api/health")
        assert response.status_code == 200

        data = response.json()
        assert "profile" in data
        assert "backends" in data
        assert "vector" in data["backends"]
        assert "chunker" in data["backends"]
        assert "ocr_enabled" in data["backends"]


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain not installed")
class TestProcessingServiceFactory:
    """Tests for ProcessingService factory integration."""

    def test_processing_service_uses_chunker_property(self):
        """ProcessingService lazy-loads chunker via factory."""
        from ai_ready_rag.config import Settings
        from ai_ready_rag.services.chunker_simple import SimpleChunker
        from ai_ready_rag.services.processing_service import ProcessingService

        settings = Settings(env_profile="laptop", chunker_backend="simple")
        service = ProcessingService(settings=settings)

        # Chunker not loaded yet
        assert service._chunker is None

        # Access chunker property triggers factory
        chunker = service.chunker
        assert isinstance(chunker, SimpleChunker)

    @pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")
    def test_processing_service_uses_vector_service_property(self):
        """ProcessingService lazy-loads vector service via factory."""
        from ai_ready_rag.config import Settings
        from ai_ready_rag.services.processing_service import ProcessingService
        from ai_ready_rag.services.vector_chroma import ChromaVectorService

        settings = Settings(env_profile="laptop", vector_backend="chroma")
        service = ProcessingService(settings=settings)

        # Vector service not loaded yet
        assert service._vector_service is None

        # Access property triggers factory
        vs = service.vector_service
        assert isinstance(vs, ChromaVectorService)


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")
class TestRAGServiceFactory:
    """Tests for RAGService factory integration."""

    def test_rag_service_uses_vector_service_property(self):
        """RAGService lazy-loads vector service via factory."""
        from ai_ready_rag.config import Settings
        from ai_ready_rag.services.rag_service import RAGService
        from ai_ready_rag.services.vector_chroma import ChromaVectorService

        settings = Settings(env_profile="laptop", vector_backend="chroma")
        service = RAGService(settings=settings)

        # Vector service not loaded yet
        assert service._vector_service is None

        # Access property triggers factory
        vs = service.vector_service
        assert isinstance(vs, ChromaVectorService)
