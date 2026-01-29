"""Tests for admin model configuration endpoints."""

from unittest.mock import AsyncMock, patch

import pytest


class TestGetModels:
    """Tests for GET /api/admin/models."""

    @pytest.fixture
    def mock_ollama_models(self):
        """Mock ModelService.list_models response."""
        return [
            {
                "name": "qwen3:8b",
                "display_name": "Qwen3 8B (Recommended)",
                "size_gb": 4.92,
                "parameters": "8B",
                "quantization": "Q4_K_M",
                "recommended": True,
            },
            {
                "name": "llama3.2:latest",
                "display_name": "Llama 3.2",
                "size_gb": 2.0,
                "parameters": "3B",
                "quantization": None,
                "recommended": False,
            },
        ]

    def test_get_returns_models_list(self, client, admin_headers, mock_ollama_models):
        """GET returns list of available models from Ollama."""
        with patch(
            "ai_ready_rag.api.admin.ModelService.list_models",
            new_callable=AsyncMock,
            return_value=mock_ollama_models,
        ):
            response = client.get("/api/admin/models", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert len(data["available_models"]) == 2
        assert data["available_models"][0]["name"] == "qwen3:8b"
        assert data["available_models"][0]["recommended"] is True

    def test_get_includes_current_model(self, client, admin_headers, mock_ollama_models):
        """GET includes current chat and embedding model."""
        with patch(
            "ai_ready_rag.api.admin.ModelService.list_models",
            new_callable=AsyncMock,
            return_value=mock_ollama_models,
        ):
            response = client.get("/api/admin/models", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert "current_chat_model" in data
        assert "current_embedding_model" in data
        # Should have default values from config
        assert data["current_chat_model"] is not None
        assert data["current_embedding_model"] is not None

    def test_get_requires_admin(self, client, user_headers):
        """GET requires admin role."""
        response = client.get("/api/admin/models", headers=user_headers)
        assert response.status_code == 403

    def test_get_requires_auth(self, client):
        """GET requires authentication."""
        response = client.get("/api/admin/models")
        assert response.status_code == 401

    def test_get_handles_ollama_unavailable(self, client, admin_headers):
        """GET returns 503 when Ollama is unavailable."""
        from ai_ready_rag.services.model_service import OllamaUnavailableError

        with patch(
            "ai_ready_rag.api.admin.ModelService.list_models",
            new_callable=AsyncMock,
            side_effect=OllamaUnavailableError("Cannot connect to Ollama"),
        ):
            response = client.get("/api/admin/models", headers=admin_headers)

        assert response.status_code == 503
        assert "Ollama" in response.json()["detail"]


class TestChangeModel:
    """Tests for PATCH /api/admin/models/chat."""

    def test_change_valid_model(self, client, admin_headers):
        """PATCH successfully changes to a valid model."""
        with patch(
            "ai_ready_rag.api.admin.ModelService.validate_model",
            new_callable=AsyncMock,
            return_value=True,
        ):
            response = client.patch(
                "/api/admin/models/chat",
                headers=admin_headers,
                json={"model_name": "llama3.2:latest"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["current_model"] == "llama3.2:latest"
        assert data["success"] is True
        assert "message" in data

    def test_change_persists(self, client, admin_headers):
        """PATCH model change persists across requests."""
        # First change the model
        with patch(
            "ai_ready_rag.api.admin.ModelService.validate_model",
            new_callable=AsyncMock,
            return_value=True,
        ):
            client.patch(
                "/api/admin/models/chat",
                headers=admin_headers,
                json={"model_name": "mistral:7b"},
            )

        # Verify with GET
        mock_models = [
            {
                "name": "mistral:7b",
                "display_name": "Mistral 7B",
                "size_gb": 4.1,
                "parameters": "7B",
                "quantization": None,
                "recommended": False,
            }
        ]
        with patch(
            "ai_ready_rag.api.admin.ModelService.list_models",
            new_callable=AsyncMock,
            return_value=mock_models,
        ):
            response = client.get("/api/admin/models", headers=admin_headers)

        assert response.json()["current_chat_model"] == "mistral:7b"

    def test_change_invalid_model(self, client, admin_headers):
        """PATCH returns 400 for model not in Ollama."""
        with patch(
            "ai_ready_rag.api.admin.ModelService.validate_model",
            new_callable=AsyncMock,
            return_value=False,
        ):
            response = client.patch(
                "/api/admin/models/chat",
                headers=admin_headers,
                json={"model_name": "nonexistent:model"},
            )

        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    def test_change_requires_admin(self, client, user_headers):
        """PATCH requires admin role."""
        response = client.patch(
            "/api/admin/models/chat",
            headers=user_headers,
            json={"model_name": "qwen3:8b"},
        )
        assert response.status_code == 403

    def test_change_requires_auth(self, client):
        """PATCH requires authentication."""
        response = client.patch(
            "/api/admin/models/chat",
            json={"model_name": "qwen3:8b"},
        )
        assert response.status_code == 401

    def test_change_handles_ollama_unavailable(self, client, admin_headers):
        """PATCH returns 503 when Ollama is unavailable."""
        from ai_ready_rag.services.model_service import OllamaUnavailableError

        with patch(
            "ai_ready_rag.api.admin.ModelService.validate_model",
            new_callable=AsyncMock,
            side_effect=OllamaUnavailableError("Cannot connect to Ollama"),
        ):
            response = client.patch(
                "/api/admin/models/chat",
                headers=admin_headers,
                json={"model_name": "qwen3:8b"},
            )

        assert response.status_code == 503
        assert "Ollama" in response.json()["detail"]

    def test_change_logs_audit(self, client, admin_headers, caplog):
        """PATCH logs model change for audit trail."""
        import logging

        with patch(
            "ai_ready_rag.api.admin.ModelService.validate_model",
            new_callable=AsyncMock,
            return_value=True,
        ):
            with caplog.at_level(logging.INFO, logger="ai_ready_rag.api.admin"):
                client.patch(
                    "/api/admin/models/chat",
                    headers=admin_headers,
                    json={"model_name": "gemma2:9b"},
                )

        # Check that the change was logged
        assert any("changed chat model" in record.message for record in caplog.records)
        assert any("gemma2:9b" in record.message for record in caplog.records)
