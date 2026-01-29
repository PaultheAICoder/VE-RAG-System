"""Tests for document management API endpoints."""

import tempfile
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status

from ai_ready_rag.db.models import Document, Tag


@pytest.fixture
def test_tag(db, admin_user):
    """Create a test tag."""
    tag = Tag(
        name="test_docs",
        display_name="Test Documents",
        description="Test tag for documents",
        created_by=admin_user.id,
    )
    db.add(tag)
    db.flush()
    db.refresh(tag)
    return tag


@pytest.fixture
def second_tag(db, admin_user):
    """Create a second test tag."""
    tag = Tag(
        name="finance",
        display_name="Finance",
        description="Finance documents",
        created_by=admin_user.id,
    )
    db.add(tag)
    db.flush()
    db.refresh(tag)
    return tag


@pytest.fixture
def test_document(db, admin_user, test_tag):
    """Create a test document."""
    doc = Document(
        filename="original.pdf",
        original_filename="test_document.pdf",
        file_path="/tmp/test/original.pdf",
        file_type="pdf",
        file_size=1024,
        status="ready",
        uploaded_by=admin_user.id,
        chunk_count=5,
    )
    doc.tags.append(test_tag)
    db.add(doc)
    db.flush()
    db.refresh(doc)
    return doc


@pytest.fixture
def user_with_tag(db, regular_user, test_tag):
    """Give regular user access to test_tag."""
    regular_user.tags.append(test_tag)
    db.flush()
    db.refresh(regular_user)
    return regular_user


class TestDocumentUpload:
    """Tests for document upload endpoint."""

    def test_upload_requires_admin(self, client, user_headers, test_tag):
        """Test that non-admin users cannot upload documents."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Test content")
            f.flush()

            with open(f.name, "rb") as upload_file:
                response = client.post(
                    "/api/documents/upload",
                    files={"file": ("test.txt", upload_file, "text/plain")},
                    data={"tag_ids": [test_tag.id]},
                    headers=user_headers,
                )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_upload_requires_auth(self, client, test_tag):
        """Test that unauthenticated users cannot upload."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Test content")
            f.flush()

            with open(f.name, "rb") as upload_file:
                response = client.post(
                    "/api/documents/upload",
                    files={"file": ("test.txt", upload_file, "text/plain")},
                    data={"tag_ids": [test_tag.id]},
                )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_upload_requires_tags(self, client, admin_headers):
        """Test that upload requires at least one tag."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Test content")
            f.flush()

            with open(f.name, "rb") as upload_file:
                response = client.post(
                    "/api/documents/upload",
                    files={"file": ("test.txt", upload_file, "text/plain")},
                    data={"tag_ids": []},
                    headers=admin_headers,
                )

        # FastAPI returns 422 for form data validation errors
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_upload_validates_file_type(self, client, admin_headers, test_tag):
        """Test that upload rejects invalid file types."""
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(b"Test content")
            f.flush()

            with open(f.name, "rb") as upload_file:
                response = client.post(
                    "/api/documents/upload",
                    files={"file": ("test.exe", upload_file, "application/octet-stream")},
                    data={"tag_ids": [test_tag.id]},
                    headers=admin_headers,
                )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "not allowed" in response.json()["detail"].lower()

    def test_upload_validates_tag_exists(self, client, admin_headers):
        """Test that upload validates tag IDs exist."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Test content")
            f.flush()

            with open(f.name, "rb") as upload_file:
                response = client.post(
                    "/api/documents/upload",
                    files={"file": ("test.txt", upload_file, "text/plain")},
                    data={"tag_ids": ["nonexistent-tag-id"]},
                    headers=admin_headers,
                )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "invalid tag" in response.json()["detail"].lower()

    @patch("ai_ready_rag.api.documents.process_document_task")
    def test_upload_valid_document(self, mock_task, client, admin_headers, test_tag):
        """Test successful document upload."""
        import shutil
        from pathlib import Path

        # Ensure upload directory exists (uses default from settings: ./data/uploads)
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Test document content for upload test")
            f.flush()

            with open(f.name, "rb") as upload_file:
                response = client.post(
                    "/api/documents/upload",
                    files={"file": ("test_upload.txt", upload_file, "text/plain")},
                    data={"tag_ids": [test_tag.id]},
                    headers=admin_headers,
                )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["status"] == "pending"
        assert data["original_filename"] == "test_upload.txt"
        assert data["file_type"] == "txt"
        assert len(data["tags"]) == 1

        # Cleanup: delete the uploaded document directory
        doc_id = data["id"]
        doc_dir = upload_dir / doc_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)


class TestDocumentList:
    """Tests for document list endpoint."""

    def test_list_requires_auth(self, client):
        """Test that listing documents requires authentication."""
        response = client.get("/api/documents")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_admin_sees_all_documents(self, client, admin_headers, test_document, second_tag, db):
        """Test that admin can see all documents regardless of tags."""
        # Create another document with different tag
        doc2 = Document(
            filename="original.pdf",
            original_filename="another.pdf",
            file_path="/tmp/test2/original.pdf",
            file_type="pdf",
            file_size=2048,
            status="ready",
            uploaded_by=test_document.uploaded_by,
        )
        doc2.tags.append(second_tag)
        db.add(doc2)
        db.flush()

        response = client.get("/api/documents", headers=admin_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] >= 2

    def test_user_sees_only_accessible_documents(
        self, client, user_headers, test_document, second_tag, user_with_tag, db
    ):
        """Test that users only see documents with matching tags."""
        # Create another document with different tag (not accessible to user)
        doc2 = Document(
            filename="original.pdf",
            original_filename="restricted.pdf",
            file_path="/tmp/test3/original.pdf",
            file_type="pdf",
            file_size=2048,
            status="ready",
            uploaded_by=test_document.uploaded_by,
        )
        doc2.tags.append(second_tag)
        db.add(doc2)
        db.flush()

        response = client.get("/api/documents", headers=user_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # User should only see test_document (has test_tag which user has access to)
        assert data["total"] == 1
        assert data["documents"][0]["id"] == test_document.id

    def test_list_pagination(self, client, admin_headers, test_document):
        """Test list pagination parameters."""
        response = client.get(
            "/api/documents",
            params={"limit": 5, "offset": 0},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["limit"] == 5
        assert data["offset"] == 0

    def test_list_filter_by_status(self, client, admin_headers, test_document, db):
        """Test filtering by status."""
        # Create pending document
        doc2 = Document(
            filename="original.pdf",
            original_filename="pending.pdf",
            file_path="/tmp/test4/original.pdf",
            file_type="pdf",
            file_size=1024,
            status="pending",
            uploaded_by=test_document.uploaded_by,
        )
        doc2.tags = test_document.tags
        db.add(doc2)
        db.flush()

        response = client.get(
            "/api/documents",
            params={"status": "ready"},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        for doc in data["documents"]:
            assert doc["status"] == "ready"

    def test_list_search(self, client, admin_headers, test_document):
        """Test search by filename."""
        response = client.get(
            "/api/documents",
            params={"search": "test_document"},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] >= 1


class TestDocumentGet:
    """Tests for document get endpoint."""

    def test_get_requires_auth(self, client, test_document):
        """Test that getting a document requires authentication."""
        response = client.get(f"/api/documents/{test_document.id}")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_admin_can_get_any_document(self, client, admin_headers, test_document):
        """Test that admin can view any document."""
        response = client.get(
            f"/api/documents/{test_document.id}",
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == test_document.id

    def test_user_can_get_accessible_document(
        self, client, user_headers, test_document, user_with_tag
    ):
        """Test that user can view document with matching tag."""
        response = client.get(
            f"/api/documents/{test_document.id}",
            headers=user_headers,
        )

        assert response.status_code == status.HTTP_200_OK

    def test_user_cannot_get_inaccessible_document(
        self, client, user_headers, second_tag, db, admin_user
    ):
        """Test that user cannot view document without matching tag."""
        # Create document with tag user doesn't have
        doc = Document(
            filename="original.pdf",
            original_filename="restricted.pdf",
            file_path="/tmp/restricted/original.pdf",
            file_type="pdf",
            file_size=1024,
            status="ready",
            uploaded_by=admin_user.id,
        )
        doc.tags.append(second_tag)
        db.add(doc)
        db.flush()

        response = client.get(f"/api/documents/{doc.id}", headers=user_headers)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_nonexistent_document(self, client, admin_headers):
        """Test getting a non-existent document returns 404."""
        response = client.get(
            "/api/documents/nonexistent-id",
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDocumentDelete:
    """Tests for document delete endpoint."""

    def test_delete_requires_admin(self, client, user_headers, test_document):
        """Test that only admins can delete documents."""
        response = client.delete(
            f"/api/documents/{test_document.id}",
            headers=user_headers,
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    @patch("ai_ready_rag.services.vector_service.VectorService")
    def test_delete_document(self, mock_vector_class, client, admin_headers, test_document, db):
        """Test successful document deletion."""
        # Mock VectorService
        mock_instance = AsyncMock()
        mock_instance.delete_document = AsyncMock(return_value=True)
        mock_vector_class.return_value = mock_instance

        doc_id = test_document.id

        response = client.delete(
            f"/api/documents/{doc_id}",
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify document is deleted from database
        deleted_doc = db.query(Document).filter(Document.id == doc_id).first()
        assert deleted_doc is None

    def test_delete_nonexistent_document(self, client, admin_headers):
        """Test deleting non-existent document returns 404."""
        response = client.delete(
            "/api/documents/nonexistent-id",
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDocumentTagUpdate:
    """Tests for document tag update endpoint."""

    def test_tag_update_requires_admin(self, client, user_headers, test_document, second_tag):
        """Test that only admins can update document tags."""
        response = client.patch(
            f"/api/documents/{test_document.id}/tags",
            json={"tag_ids": [second_tag.id]},
            headers=user_headers,
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_tag_update_requires_at_least_one_tag(self, client, admin_headers, test_document):
        """Test that tag update requires at least one tag."""
        response = client.patch(
            f"/api/documents/{test_document.id}/tags",
            json={"tag_ids": []},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @patch("ai_ready_rag.services.vector_service.VectorService")
    def test_tag_update_success(
        self, mock_vector_class, client, admin_headers, test_document, second_tag, db
    ):
        """Test successful tag update."""
        # Mock VectorService
        mock_instance = AsyncMock()
        mock_instance.update_document_tags = AsyncMock(return_value=5)
        mock_vector_class.return_value = mock_instance

        response = client.patch(
            f"/api/documents/{test_document.id}/tags",
            json={"tag_ids": [second_tag.id]},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["tags"]) == 1
        assert data["tags"][0]["id"] == second_tag.id

    def test_tag_update_invalid_tag_id(self, client, admin_headers, test_document):
        """Test tag update with invalid tag ID."""
        response = client.patch(
            f"/api/documents/{test_document.id}/tags",
            json={"tag_ids": ["nonexistent-tag"]},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestDocumentReprocess:
    """Tests for document reprocess endpoint."""

    def test_reprocess_requires_admin(self, client, user_headers, test_document):
        """Test that only admins can reprocess documents."""
        response = client.post(
            f"/api/documents/{test_document.id}/reprocess",
            headers=user_headers,
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    @patch("ai_ready_rag.api.documents.process_document_task")
    @patch("ai_ready_rag.services.vector_service.VectorService")
    def test_reprocess_ready_document(
        self, mock_vector_class, mock_task, client, admin_headers, test_document
    ):
        """Test reprocessing a ready document."""
        # Mock VectorService
        mock_instance = AsyncMock()
        mock_instance.delete_document = AsyncMock(return_value=True)
        mock_vector_class.return_value = mock_instance

        response = client.post(
            f"/api/documents/{test_document.id}/reprocess",
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert data["status"] == "pending"
        assert data["chunk_count"] is None

    def test_reprocess_pending_document_fails(self, client, admin_headers, test_document, db):
        """Test that pending documents cannot be reprocessed."""
        test_document.status = "pending"
        db.flush()

        response = client.post(
            f"/api/documents/{test_document.id}/reprocess",
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @patch("ai_ready_rag.api.documents.process_document_task")
    @patch("ai_ready_rag.services.vector_service.VectorService")
    def test_reprocess_failed_document(
        self, mock_vector_class, mock_task, client, admin_headers, test_document, db
    ):
        """Test reprocessing a failed document."""
        test_document.status = "failed"
        test_document.error_message = "Previous error"
        db.flush()

        # Mock VectorService
        mock_instance = AsyncMock()
        mock_instance.delete_document = AsyncMock(return_value=True)
        mock_vector_class.return_value = mock_instance

        response = client.post(
            f"/api/documents/{test_document.id}/reprocess",
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert data["status"] == "pending"
        assert data["error_message"] is None
