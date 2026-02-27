"""Tests for the reconciliation endpoint and service."""

from unittest.mock import patch

from fastapi import status


class TestReconcileEndpoint:
    """Tests for POST /api/admin/reconcile."""

    def test_reconcile_requires_admin(self, client, user_headers):
        """Non-admin users cannot access reconciliation."""
        response = client.post(
            "/api/admin/reconcile",
            json={"dry_run": True},
            headers=user_headers,
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_reconcile_unauthorized(self, client):
        """Unauthenticated users cannot access reconciliation."""
        response = client.post(
            "/api/admin/reconcile",
            json={"dry_run": True},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @patch("ai_ready_rag.services.reconciliation_service.reconcile")
    def test_reconcile_dry_run_no_issues(self, mock_reconcile, client, admin_headers):
        """Dry run with no drift returns empty issues list."""
        mock_reconcile.return_value = {
            "total_documents": 10,
            "total_qdrant_documents": 10,
            "synced": 10,
            "issues": [],
            "repairs": [],
            "dry_run": True,
        }

        response = client.post(
            "/api/admin/reconcile",
            json={"dry_run": True},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_documents"] == 10
        assert data["synced"] == 10
        assert data["issues"] == []
        assert data["dry_run"] is True

    @patch("ai_ready_rag.services.reconciliation_service.reconcile")
    def test_reconcile_detects_ghost_doc(self, mock_reconcile, client, admin_headers):
        """Ghost doc (ready in SQLite, 0 vectors) is detected."""
        mock_reconcile.return_value = {
            "total_documents": 5,
            "total_qdrant_documents": 4,
            "synced": 4,
            "issues": [
                {
                    "document_id": "abc-123",
                    "filename": "report.pdf",
                    "issue": "ghost_doc",
                    "detail": "status=ready in SQLite but 0 vectors in Qdrant",
                    "sqlite_chunks": 36,
                    "qdrant_chunks": 0,
                }
            ],
            "repairs": [],
            "dry_run": True,
        }

        response = client.post(
            "/api/admin/reconcile",
            json={"dry_run": True},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["issues"]) == 1
        assert data["issues"][0]["issue"] == "ghost_doc"

    @patch("ai_ready_rag.services.reconciliation_service.reconcile")
    def test_reconcile_repair_mode(self, mock_reconcile, client, admin_headers):
        """Repair mode returns repairs list."""
        mock_reconcile.return_value = {
            "total_documents": 5,
            "total_qdrant_documents": 6,
            "synced": 4,
            "issues": [
                {
                    "document_id": "orphan-1",
                    "filename": None,
                    "issue": "orphan_vectors",
                    "detail": "Vectors exist in Qdrant but no SQLite record",
                    "sqlite_chunks": None,
                    "qdrant_chunks": 14,
                }
            ],
            "repairs": [
                {
                    "document_id": "orphan-1",
                    "action": "deleted_orphan_vectors",
                }
            ],
            "dry_run": False,
        }

        response = client.post(
            "/api/admin/reconcile",
            json={"dry_run": False},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["dry_run"] is False
        assert len(data["repairs"]) == 1
        assert data["repairs"][0]["action"] == "deleted_orphan_vectors"
