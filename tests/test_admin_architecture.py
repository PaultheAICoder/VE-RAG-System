"""Tests for admin architecture endpoint."""

from unittest.mock import AsyncMock, patch


class TestArchitectureInfo:
    """Tests for GET /api/admin/architecture endpoint."""

    def test_architecture_requires_admin(self, client, user_headers):
        """Non-admin users cannot access architecture endpoint."""
        response = client.get("/api/admin/architecture", headers=user_headers)
        assert response.status_code == 403

    def test_architecture_unauthorized(self, client):
        """Unauthenticated requests are rejected."""
        response = client.get("/api/admin/architecture")
        assert response.status_code == 401

    def test_architecture_returns_structure(self, client, admin_headers):
        """Admin can get architecture info with correct structure."""
        # Mock health check to avoid actual service calls
        with patch("ai_ready_rag.api.admin.get_vector_service") as mock_get_vs:
            mock_vs = AsyncMock()
            # Return a dict for ChromaDB-style response
            mock_vs.health_check = AsyncMock(return_value={"healthy": True})
            mock_get_vs.return_value = mock_vs

            # Clear cache to force fresh response
            import ai_ready_rag.api.admin as admin_module

            admin_module._architecture_cache = {}

            # Also mock the Ollama health check
            with patch("ai_ready_rag.api.admin.httpx.AsyncClient") as mock_client:
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    return_value=mock_response
                )

                response = client.get("/api/admin/architecture", headers=admin_headers)

        assert response.status_code == 200

        data = response.json()
        assert "document_parsing" in data
        assert "embeddings" in data
        assert "chat_model" in data
        assert "infrastructure" in data
        assert "ocr_status" in data
        assert "profile" in data

    def test_architecture_document_parsing(self, client, admin_headers):
        """Document parsing section has required fields based on profile."""
        from ai_ready_rag.config import get_settings

        settings = get_settings()

        with patch("ai_ready_rag.api.admin.get_vector_service") as mock_get_vs:
            mock_vs = AsyncMock()
            mock_vs.health_check = AsyncMock(return_value={"healthy": True})
            mock_get_vs.return_value = mock_vs

            import ai_ready_rag.api.admin as admin_module

            admin_module._architecture_cache = {}

            with patch("ai_ready_rag.api.admin.httpx.AsyncClient") as mock_client:
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    return_value=mock_response
                )

                response = client.get("/api/admin/architecture", headers=admin_headers)

        data = response.json()

        doc = data["document_parsing"]
        # Engine depends on profile: "docling" -> Docling, "simple" -> SimpleChunker
        expected_engine = "Docling" if settings.chunker_backend == "docling" else "SimpleChunker"
        assert doc["engine"] == expected_engine
        assert "version" in doc
        assert "type" in doc
        assert "capabilities" in doc
        assert isinstance(doc["capabilities"], list)

    def test_architecture_embeddings(self, client, admin_headers):
        """Embeddings section has required fields."""
        with patch("ai_ready_rag.api.admin.get_vector_service") as mock_get_vs:
            mock_vs = AsyncMock()
            mock_vs.health_check = AsyncMock(return_value={"healthy": True})
            mock_get_vs.return_value = mock_vs

            import ai_ready_rag.api.admin as admin_module

            admin_module._architecture_cache = {}

            with patch("ai_ready_rag.api.admin.httpx.AsyncClient") as mock_client:
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    return_value=mock_response
                )

                response = client.get("/api/admin/architecture", headers=admin_headers)

        data = response.json()

        emb = data["embeddings"]
        assert "model" in emb
        assert "dimensions" in emb
        assert "vector_store" in emb
        assert "vector_store_url" in emb

    def test_architecture_chat_model(self, client, admin_headers):
        """Chat model section has required fields."""
        with patch("ai_ready_rag.api.admin.get_vector_service") as mock_get_vs:
            mock_vs = AsyncMock()
            mock_vs.health_check = AsyncMock(return_value={"healthy": True})
            mock_get_vs.return_value = mock_vs

            import ai_ready_rag.api.admin as admin_module

            admin_module._architecture_cache = {}

            with patch("ai_ready_rag.api.admin.httpx.AsyncClient") as mock_client:
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    return_value=mock_response
                )

                response = client.get("/api/admin/architecture", headers=admin_headers)

        data = response.json()

        chat = data["chat_model"]
        assert "name" in chat
        assert chat["provider"] == "Ollama"
        assert "capabilities" in chat
        assert isinstance(chat["capabilities"], list)

    def test_architecture_infrastructure(self, client, admin_headers):
        """Infrastructure section has required fields."""
        with patch("ai_ready_rag.api.admin.get_vector_service") as mock_get_vs:
            mock_vs = AsyncMock()
            mock_vs.health_check = AsyncMock(return_value={"healthy": True})
            mock_get_vs.return_value = mock_vs

            import ai_ready_rag.api.admin as admin_module

            admin_module._architecture_cache = {}

            with patch("ai_ready_rag.api.admin.httpx.AsyncClient") as mock_client:
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    return_value=mock_response
                )

                response = client.get("/api/admin/architecture", headers=admin_headers)

        data = response.json()

        infra = data["infrastructure"]
        assert "ollama_url" in infra
        assert "ollama_status" in infra
        assert "vector_db_status" in infra
        assert infra["ollama_status"] in ["healthy", "unhealthy"]
        assert infra["vector_db_status"] in ["healthy", "unhealthy"]

    def test_architecture_ocr_status(self, client, admin_headers):
        """OCR status has tesseract and easyocr sections."""
        with patch("ai_ready_rag.api.admin.get_vector_service") as mock_get_vs:
            mock_vs = AsyncMock()
            mock_vs.health_check = AsyncMock(return_value={"healthy": True})
            mock_get_vs.return_value = mock_vs

            import ai_ready_rag.api.admin as admin_module

            admin_module._architecture_cache = {}

            with patch("ai_ready_rag.api.admin.httpx.AsyncClient") as mock_client:
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    return_value=mock_response
                )

                response = client.get("/api/admin/architecture", headers=admin_headers)

        data = response.json()

        ocr = data["ocr_status"]
        assert "tesseract" in ocr
        assert "easyocr" in ocr
        assert "available" in ocr["tesseract"]
        assert "available" in ocr["easyocr"]

    def test_architecture_cache_works(self, client, admin_headers):
        """Response is cached for subsequent requests."""
        with patch("ai_ready_rag.api.admin.get_vector_service") as mock_get_vs:
            mock_vs = AsyncMock()
            mock_vs.health_check = AsyncMock(return_value={"healthy": True})
            mock_get_vs.return_value = mock_vs

            import ai_ready_rag.api.admin as admin_module

            admin_module._architecture_cache = {}

            with patch("ai_ready_rag.api.admin.httpx.AsyncClient") as mock_client:
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    return_value=mock_response
                )

                # First request
                response1 = client.get("/api/admin/architecture", headers=admin_headers)
                assert response1.status_code == 200

                # Second request should use cache (get_vector_service not called again)
                mock_get_vs.reset_mock()
                response2 = client.get("/api/admin/architecture", headers=admin_headers)
                assert response2.status_code == 200

                # get_vector_service should not be called for cached response
                # (it was called once for first request, then cache is used)

    def test_architecture_handles_health_check_failure(self, client, admin_headers):
        """Endpoint handles health check failure gracefully."""
        with patch("ai_ready_rag.api.admin.get_vector_service") as mock_get_vs:
            mock_vs = AsyncMock()
            mock_vs.health_check = AsyncMock(side_effect=Exception("Connection failed"))
            mock_get_vs.return_value = mock_vs

            import ai_ready_rag.api.admin as admin_module

            admin_module._architecture_cache = {}

            response = client.get("/api/admin/architecture", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["infrastructure"]["ollama_status"] == "unhealthy"
        assert data["infrastructure"]["vector_db_status"] == "unhealthy"
