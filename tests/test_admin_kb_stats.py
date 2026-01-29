"""Tests for admin knowledge base statistics and clearing endpoints."""

from unittest.mock import AsyncMock, patch

from fastapi import status


class TestKnowledgeBaseStats:
    """Tests for GET /api/admin/knowledge-base/stats endpoint."""

    @patch("ai_ready_rag.api.admin.VectorService")
    def test_stats_returns_data(self, mock_vector_class, client, admin_headers):
        """Test that endpoint returns knowledge base statistics."""
        # Mock VectorService
        mock_instance = AsyncMock()
        mock_instance.get_extended_stats = AsyncMock(
            return_value={
                "total_chunks": 150,
                "unique_files": 3,
                "collection_name": "documents",
                "collection_size_bytes": 1024000,
                "files": [
                    {
                        "document_id": "doc-1",
                        "filename": "report.pdf",
                        "chunk_count": 50,
                    },
                    {
                        "document_id": "doc-2",
                        "filename": "guide.docx",
                        "chunk_count": 75,
                    },
                    {
                        "document_id": "doc-3",
                        "filename": "notes.txt",
                        "chunk_count": 25,
                    },
                ],
            }
        )
        mock_vector_class.return_value = mock_instance

        response = client.get("/api/admin/knowledge-base/stats", headers=admin_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_chunks"] == 150
        assert data["unique_files"] == 3
        assert data["total_vectors"] == 150
        assert data["collection_name"] == "documents"
        assert data["storage_size_bytes"] == 1024000
        assert len(data["files"]) == 3
        assert data["files"][0]["document_id"] == "doc-1"
        assert data["files"][0]["filename"] == "report.pdf"
        assert data["files"][0]["chunk_count"] == 50
        assert "last_updated" in data

    def test_stats_requires_admin(self, client, user_headers):
        """Test that non-admin users cannot access stats endpoint."""
        response = client.get("/api/admin/knowledge-base/stats", headers=user_headers)
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_stats_unauthorized(self, client):
        """Test that unauthenticated users cannot access stats endpoint."""
        response = client.get("/api/admin/knowledge-base/stats")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @patch("ai_ready_rag.api.admin.VectorService")
    def test_stats_empty_collection(self, mock_vector_class, client, admin_headers):
        """Test stats endpoint with empty collection."""
        mock_instance = AsyncMock()
        mock_instance.get_extended_stats = AsyncMock(
            return_value={
                "total_chunks": 0,
                "unique_files": 0,
                "collection_name": "documents",
                "collection_size_bytes": 0,
                "files": [],
            }
        )
        mock_vector_class.return_value = mock_instance

        response = client.get("/api/admin/knowledge-base/stats", headers=admin_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_chunks"] == 0
        assert data["unique_files"] == 0
        assert data["files"] == []


class TestClearKnowledgeBase:
    """Tests for DELETE /api/admin/knowledge-base endpoint."""

    @patch("ai_ready_rag.api.admin.VectorService")
    def test_clear_success(self, mock_vector_class, client, admin_headers):
        """Test successful knowledge base clearing with confirmation."""
        mock_instance = AsyncMock()
        mock_instance.get_extended_stats = AsyncMock(
            return_value={
                "total_chunks": 100,
                "unique_files": 5,
                "collection_name": "documents",
                "collection_size_bytes": 512000,
                "files": [],
            }
        )
        mock_instance.clear_collection = AsyncMock(return_value=True)
        mock_vector_class.return_value = mock_instance

        response = client.request(
            "DELETE",
            "/api/admin/knowledge-base",
            json={"confirm": True},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["deleted_chunks"] == 100
        assert data["deleted_files"] == 5
        assert data["success"] is True

    def test_clear_requires_confirmation(self, client, admin_headers):
        """Test that clearing requires confirmation flag."""
        response = client.request(
            "DELETE",
            "/api/admin/knowledge-base",
            json={"confirm": False},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "confirmation required" in response.json()["detail"].lower()

    def test_clear_requires_admin(self, client, user_headers):
        """Test that non-admin users cannot clear knowledge base."""
        response = client.request(
            "DELETE",
            "/api/admin/knowledge-base",
            json={"confirm": True},
            headers=user_headers,
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_clear_unauthorized(self, client):
        """Test that unauthenticated users cannot clear knowledge base."""
        response = client.request(
            "DELETE",
            "/api/admin/knowledge-base",
            json={"confirm": True},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @patch("ai_ready_rag.api.admin.DocumentService")
    @patch("ai_ready_rag.api.admin.VectorService")
    def test_clear_with_source_files(
        self, mock_vector_class, mock_doc_class, client, admin_headers
    ):
        """Test clearing with delete_source_files option."""
        # Mock VectorService
        mock_vector_instance = AsyncMock()
        mock_vector_instance.get_extended_stats = AsyncMock(
            return_value={
                "total_chunks": 50,
                "unique_files": 2,
                "collection_name": "documents",
                "collection_size_bytes": 256000,
                "files": [],
            }
        )
        mock_vector_instance.clear_collection = AsyncMock(return_value=True)
        mock_vector_class.return_value = mock_vector_instance

        # Mock DocumentService
        mock_doc_instance = mock_doc_class.return_value
        mock_doc_instance.delete_all_documents.return_value = 2

        response = client.request(
            "DELETE",
            "/api/admin/knowledge-base",
            json={"confirm": True, "delete_source_files": True},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["deleted_files"] == 2

    @patch("ai_ready_rag.api.admin.VectorService")
    def test_clear_failure(self, mock_vector_class, client, admin_headers):
        """Test handling of clear operation failure."""
        mock_instance = AsyncMock()
        mock_instance.get_extended_stats = AsyncMock(
            return_value={
                "total_chunks": 100,
                "unique_files": 5,
                "collection_name": "documents",
                "collection_size_bytes": 512000,
                "files": [],
            }
        )
        mock_instance.clear_collection = AsyncMock(return_value=False)
        mock_vector_class.return_value = mock_instance

        response = client.request(
            "DELETE",
            "/api/admin/knowledge-base",
            json={"confirm": True},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is False
        assert data["deleted_chunks"] == 0
