"""Tests for document management API endpoints."""

import tempfile
from pathlib import Path
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

        # With auto-tagging disabled (default), empty tags returns 400
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "tag" in response.json()["detail"].lower()

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

    @patch("ai_ready_rag.services.factory.get_vector_service")
    def test_tag_update_success(
        self, mock_factory, client, admin_headers, test_document, second_tag, db
    ):
        """Test successful tag update."""
        # Mock the factory to return a mock vector service
        mock_instance = AsyncMock()
        mock_instance.update_document_tags = AsyncMock(return_value=5)
        mock_factory.return_value = mock_instance

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


class TestUploadWithProcessingOptions:
    """Tests for per-upload processing options."""

    @patch("ai_ready_rag.api.documents.process_document_task")
    def test_upload_with_enable_ocr_false(
        self, mock_task, client, admin_headers, test_tag, tmp_path
    ):
        """Upload with enable_ocr=false passes option to background task."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "tag_ids": [test_tag.id],
                    "enable_ocr": "false",
                },
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_201_CREATED
        # Verify task was called with options
        mock_task.assert_called_once()
        call_args = mock_task.call_args
        options_dict = call_args[0][1]  # Second positional arg
        assert options_dict is not None
        assert options_dict["enable_ocr"] is False

    @patch("ai_ready_rag.api.documents.process_document_task")
    def test_upload_without_options_passes_none(
        self, mock_task, client, admin_headers, test_tag, tmp_path
    ):
        """Upload without processing options passes None to background task."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={"tag_ids": [test_tag.id]},
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_201_CREATED
        mock_task.assert_called_once()
        call_args = mock_task.call_args
        options_dict = call_args[0][1]
        assert options_dict is None

    @patch("ai_ready_rag.api.documents.process_document_task")
    def test_upload_with_multiple_options(
        self, mock_task, client, admin_headers, test_tag, tmp_path
    ):
        """Upload with multiple processing options passes all to task."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "tag_ids": [test_tag.id],
                    "enable_ocr": "true",
                    "force_full_page_ocr": "true",
                    "ocr_language": "fra",
                    "table_extraction_mode": "fast",
                },
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_201_CREATED
        call_args = mock_task.call_args
        options_dict = call_args[0][1]
        assert options_dict["enable_ocr"] is True
        assert options_dict["force_full_page_ocr"] is True
        assert options_dict["ocr_language"] == "fra"
        assert options_dict["table_extraction_mode"] == "fast"

    @patch("ai_ready_rag.api.documents.process_document_task")
    def test_upload_with_include_image_descriptions(
        self, mock_task, client, admin_headers, test_tag, tmp_path
    ):
        """Upload with include_image_descriptions option."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "tag_ids": [test_tag.id],
                    "include_image_descriptions": "true",
                },
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_201_CREATED
        call_args = mock_task.call_args
        options_dict = call_args[0][1]
        assert options_dict["include_image_descriptions"] is True

    def test_upload_with_invalid_table_extraction_mode(
        self, client, admin_headers, test_tag, tmp_path
    ):
        """Upload with invalid table_extraction_mode returns 400."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "tag_ids": [test_tag.id],
                    "table_extraction_mode": "invalid",
                },
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "table_extraction_mode" in response.json()["detail"]


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


class TestCheckDuplicates:
    """Tests for duplicate check endpoint."""

    def test_check_duplicates_requires_admin(self, client, user_headers):
        """Test that only admins can check duplicates."""
        response = client.post(
            "/api/documents/check-duplicates",
            json={"filenames": ["test.pdf"]},
            headers=user_headers,
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_check_duplicates_requires_auth(self, client):
        """Test that check duplicates requires authentication."""
        response = client.post(
            "/api/documents/check-duplicates",
            json={"filenames": ["test.pdf"]},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_check_duplicates_finds_existing(self, client, admin_headers, test_document):
        """Test that existing filenames are identified as duplicates."""
        response = client.post(
            "/api/documents/check-duplicates",
            json={"filenames": [test_document.original_filename, "unique.pdf"]},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["duplicates"]) == 1
        assert data["duplicates"][0]["filename"] == test_document.original_filename
        assert data["duplicates"][0]["existing_id"] == test_document.id
        assert "unique.pdf" in data["unique"]

    def test_check_duplicates_all_unique(self, client, admin_headers):
        """Test when no duplicates are found."""
        response = client.post(
            "/api/documents/check-duplicates",
            json={"filenames": ["new_file.pdf", "another_new.txt"]},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["duplicates"]) == 0
        assert len(data["unique"]) == 2

    def test_check_duplicates_empty_filenames(self, client, admin_headers):
        """Test that empty filenames array returns 422."""
        response = client.post(
            "/api/documents/check-duplicates",
            json={"filenames": []},
            headers=admin_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestUploadReplace:
    """Tests for upload with replace parameter."""

    @patch("ai_ready_rag.api.documents.process_document_task")
    def test_upload_replace_overwrites_existing(
        self, mock_task, client, admin_headers, test_tag, test_document, db, tmp_path
    ):
        """Test that replace=true deletes existing and uploads new."""
        # Create a file with unique content (different from test_document)
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test document content for replace test")

        # Mock vector service for delete
        with patch("ai_ready_rag.services.factory.get_vector_service") as mock_factory:
            mock_vector_instance = AsyncMock()
            mock_vector_instance.delete_document = AsyncMock(return_value=True)
            mock_factory.return_value = mock_vector_instance

            with open(test_file, "rb") as f:
                response = client.post(
                    "/api/documents/upload?replace=true",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={"tag_ids": [test_tag.id]},
                    headers=admin_headers,
                )

            # Should succeed since it's a unique file
            assert response.status_code == status.HTTP_201_CREATED

    @patch("ai_ready_rag.api.documents.process_document_task")
    def test_upload_replace_false_returns_409(
        self, mock_task, client, admin_headers, test_tag, tmp_path
    ):
        """Test that replace=false (default) returns 409 on duplicate."""
        import shutil
        from pathlib import Path

        # First upload a document
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        test_file = tmp_path / "duplicate_test.txt"
        test_file.write_text("Duplicate content for 409 test unique-1")

        with open(test_file, "rb") as f:
            response1 = client.post(
                "/api/documents/upload",
                files={"file": ("duplicate_test.txt", f, "text/plain")},
                data={"tag_ids": [test_tag.id]},
                headers=admin_headers,
            )

        assert response1.status_code == status.HTTP_201_CREATED
        doc1_id = response1.json()["id"]

        try:
            # Try uploading same file again without replace
            test_file2 = tmp_path / "duplicate_test2.txt"
            test_file2.write_text("Duplicate content for 409 test unique-1")  # Same content

            with open(test_file2, "rb") as f:
                response2 = client.post(
                    "/api/documents/upload",
                    files={"file": ("duplicate_test2.txt", f, "text/plain")},
                    data={"tag_ids": [test_tag.id]},
                    headers=admin_headers,
                )

            assert response2.status_code == status.HTTP_409_CONFLICT
            data = response2.json()
            # Check for structured error response (flat format from global error handler)
            assert data["detail"] == "Duplicate file detected"
            assert data["error_code"] == "DUPLICATE_FILE"
            assert data["existing_id"] == doc1_id
        finally:
            # Cleanup
            doc_dir = upload_dir / doc1_id
            if doc_dir.exists():
                shutil.rmtree(doc_dir)


class TestEnhanced409Response:
    """Tests for enhanced 409 response format."""

    @patch("ai_ready_rag.api.documents.process_document_task")
    def test_409_returns_structured_detail(
        self, mock_task, client, admin_headers, test_tag, tmp_path
    ):
        """Test that 409 response includes structured error details."""
        import shutil
        from pathlib import Path

        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Upload first document
        test_file = tmp_path / "structured_test.txt"
        test_file.write_text("Content for structured 409 test unique-2")

        with open(test_file, "rb") as f:
            response1 = client.post(
                "/api/documents/upload",
                files={"file": ("structured_test.txt", f, "text/plain")},
                data={"tag_ids": [test_tag.id]},
                headers=admin_headers,
            )

        assert response1.status_code == status.HTTP_201_CREATED
        doc1_data = response1.json()
        doc1_id = doc1_data["id"]

        try:
            # Upload duplicate
            test_file2 = tmp_path / "structured_test2.txt"
            test_file2.write_text("Content for structured 409 test unique-2")  # Same content

            with open(test_file2, "rb") as f:
                response2 = client.post(
                    "/api/documents/upload",
                    files={"file": ("different_name.txt", f, "text/plain")},
                    data={"tag_ids": [test_tag.id]},
                    headers=admin_headers,
                )

            assert response2.status_code == status.HTTP_409_CONFLICT
            data = response2.json()

            # Verify all required fields in structured response (flat format)
            assert data["detail"] == "Duplicate file detected"
            assert data["error_code"] == "DUPLICATE_FILE"
            assert data["existing_id"] == doc1_id
            assert data["existing_filename"] == "structured_test.txt"
            assert "uploaded_at" in data
        finally:
            # Cleanup
            doc_dir = upload_dir / doc1_id
            if doc_dir.exists():
                shutil.rmtree(doc_dir)


class TestAutoTaggingUpload:
    """Tests for auto-tagging at upload time."""

    def _enable_auto_tagging(self, monkeypatch):
        """Override settings to enable auto-tagging for tests."""
        from ai_ready_rag.config import Settings

        strategies_dir = str(Path(__file__).parent.parent / "data" / "auto_tag_strategies")
        test_settings = Settings(
            auto_tagging_enabled=True,
            auto_tagging_path_enabled=True,
            auto_tagging_create_missing_tags=True,
            auto_tagging_strategies_dir=strategies_dir,
            auto_tagging_strategy="generic",
        )
        monkeypatch.setattr("ai_ready_rag.api.documents.get_settings", lambda: test_settings)

    @patch("ai_ready_rag.api.documents.process_document_task")
    def test_upload_with_source_path_creates_auto_tags(
        self, mock_task, client, admin_headers, test_tag, tmp_path, monkeypatch, db
    ):
        """Upload with source_path and auto-tagging enabled creates path-based tags."""
        self._enable_auto_tagging(monkeypatch)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for auto-tag")

        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "tag_ids": [test_tag.id],
                    "source_path": "Acme Corp/Policies/test.txt",
                },
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["auto_tag_status"] == "pending"
        assert data["auto_tag_strategy"] == "generic"
        assert data["auto_tag_version"] == "1.0"
        assert data["source_path"] == "Acme Corp/Policies/test.txt"
        # Should have the manual tag + auto-created client tag
        tag_names = [t["name"] for t in data["tags"]]
        assert "test_docs" in tag_names
        assert "client:acme-corp" in tag_names

        # Cleanup
        doc_dir = upload_dir / data["id"]
        if doc_dir.exists():
            import shutil

            shutil.rmtree(doc_dir)

    @patch("ai_ready_rag.api.documents.process_document_task")
    def test_upload_without_tags_succeeds_when_auto_tagging_active(
        self, mock_task, client, admin_headers, tmp_path, monkeypatch, db
    ):
        """Upload with no manual tags but auto-tagging active succeeds."""
        self._enable_auto_tagging(monkeypatch)

        test_file = tmp_path / "no_manual_tags.txt"
        test_file.write_text("Content without manual tags")

        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": ("no_manual_tags.txt", f, "text/plain")},
                data={
                    "tag_ids": [],
                    "source_path": "BigClient/Reports/no_manual_tags.txt",
                },
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        tag_names = [t["name"] for t in data["tags"]]
        assert "client:bigclient" in tag_names

        # Cleanup
        doc_dir = upload_dir / data["id"]
        if doc_dir.exists():
            import shutil

            shutil.rmtree(doc_dir)

    def test_upload_without_tags_fails_when_auto_tagging_disabled(
        self, client, admin_headers, tmp_path
    ):
        """Upload with no tags and auto-tagging disabled still requires tags."""
        test_file = tmp_path / "no_tags.txt"
        test_file.write_text("Content without tags")

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": ("no_tags.txt", f, "text/plain")},
                data={"tag_ids": []},
                headers=admin_headers,
            )

        # With auto-tagging disabled (default), empty tags returns 400
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "tag" in response.json()["detail"].lower()

    @patch("ai_ready_rag.api.documents.process_document_task")
    def test_upload_with_source_path_and_manual_tags_merges(
        self, mock_task, client, admin_headers, test_tag, tmp_path, monkeypatch, db
    ):
        """Upload with both manual tags and source_path merges tags."""
        self._enable_auto_tagging(monkeypatch)

        test_file = tmp_path / "merged.txt"
        test_file.write_text("Content for merge test")

        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": ("merged.txt", f, "text/plain")},
                data={
                    "tag_ids": [test_tag.id],
                    "source_path": "TestClient/merged.txt",
                },
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        tag_names = [t["name"] for t in data["tags"]]
        assert "test_docs" in tag_names
        assert "client:testclient" in tag_names
        assert len(data["tags"]) >= 2

        # Cleanup
        doc_dir = upload_dir / data["id"]
        if doc_dir.exists():
            import shutil

            shutil.rmtree(doc_dir)

    @patch("ai_ready_rag.api.documents.process_document_task")
    def test_auto_tag_strategy_pinning(
        self, mock_task, client, admin_headers, test_tag, tmp_path, monkeypatch, db
    ):
        """Verify strategy_id and strategy_version are recorded on document."""
        self._enable_auto_tagging(monkeypatch)

        test_file = tmp_path / "pinned.txt"
        test_file.write_text("Content for strategy pinning test")

        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": ("pinned.txt", f, "text/plain")},
                data={
                    "tag_ids": [test_tag.id],
                    "source_path": "ClientA/pinned.txt",
                },
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["auto_tag_strategy"] == "generic"
        assert data["auto_tag_version"] == "1.0"
        assert data["auto_tag_status"] == "pending"

        # Cleanup
        doc_dir = upload_dir / data["id"]
        if doc_dir.exists():
            import shutil

            shutil.rmtree(doc_dir)

    def test_ensure_tag_exists_creates_new_tag(self, db, admin_user, monkeypatch):
        """ensure_tag_exists creates tag when it doesn't exist."""
        from ai_ready_rag.config import Settings
        from ai_ready_rag.services.auto_tagging import AutoTagStrategy
        from ai_ready_rag.services.document_service import DocumentService

        strategies_dir = str(Path(__file__).parent.parent / "data" / "auto_tag_strategies")
        settings = Settings(
            auto_tagging_enabled=True,
            auto_tagging_create_missing_tags=True,
            auto_tagging_strategies_dir=strategies_dir,
        )
        service = DocumentService(db, settings)
        strategy = AutoTagStrategy.load(str(Path(strategies_dir) / "generic.yaml"))

        tag = service.ensure_tag_exists(
            tag_name="client:new-client",
            display_name="New Client",
            namespace="client",
            strategy=strategy,
            created_by=admin_user.id,
        )

        assert tag is not None
        assert tag.name == "client:new-client"
        assert tag.display_name == "New Client"
        assert tag.color == "#6366f1"
        assert tag.description == "Auto-created by Generic strategy"

    def test_ensure_tag_exists_returns_existing(self, db, admin_user, test_tag):
        """ensure_tag_exists returns existing tag without creating duplicate."""
        from ai_ready_rag.config import Settings
        from ai_ready_rag.services.auto_tagging import AutoTagStrategy
        from ai_ready_rag.services.document_service import DocumentService

        strategies_dir = str(Path(__file__).parent.parent / "data" / "auto_tag_strategies")
        settings = Settings(
            auto_tagging_enabled=True,
            auto_tagging_create_missing_tags=True,
            auto_tagging_strategies_dir=strategies_dir,
        )
        service = DocumentService(db, settings)
        strategy = AutoTagStrategy.load(str(Path(strategies_dir) / "generic.yaml"))

        result = service.ensure_tag_exists(
            tag_name=test_tag.name,
            display_name=test_tag.display_name,
            namespace="test",
            strategy=strategy,
            created_by=admin_user.id,
        )

        assert result is not None
        assert result.id == test_tag.id

    def test_tag_name_length_guardrail(self, db, admin_user):
        """Tags exceeding max_tag_name_length are skipped."""
        from ai_ready_rag.config import Settings
        from ai_ready_rag.services.auto_tagging import AutoTagStrategy
        from ai_ready_rag.services.document_service import DocumentService

        strategies_dir = str(Path(__file__).parent.parent / "data" / "auto_tag_strategies")
        settings = Settings(
            auto_tagging_enabled=True,
            auto_tagging_create_missing_tags=True,
            auto_tagging_max_tag_name_length=10,
            auto_tagging_strategies_dir=strategies_dir,
        )
        service = DocumentService(db, settings)
        strategy = AutoTagStrategy.load(str(Path(strategies_dir) / "generic.yaml"))

        result = service.ensure_tag_exists(
            tag_name="client:this-is-a-very-long-tag-name",
            display_name="Long Tag",
            namespace="client",
            strategy=strategy,
            created_by=admin_user.id,
        )

        assert result is None

    def test_namespace_cardinality_guardrail(self, db, admin_user):
        """Tags rejected when namespace hits cardinality limit."""
        from ai_ready_rag.config import Settings
        from ai_ready_rag.services.auto_tagging import AutoTagStrategy
        from ai_ready_rag.services.document_service import DocumentService

        strategies_dir = str(Path(__file__).parent.parent / "data" / "auto_tag_strategies")
        settings = Settings(
            auto_tagging_enabled=True,
            auto_tagging_create_missing_tags=True,
            auto_tagging_max_client_tags=2,
            auto_tagging_strategies_dir=strategies_dir,
        )
        service = DocumentService(db, settings)
        strategy = AutoTagStrategy.load(str(Path(strategies_dir) / "generic.yaml"))

        # Create 2 client tags to hit the limit
        for i in range(2):
            t = Tag(
                name=f"client:existing-{i}",
                display_name=f"Existing {i}",
                created_by=admin_user.id,
            )
            db.add(t)
        db.flush()

        result = service.ensure_tag_exists(
            tag_name="client:over-limit",
            display_name="Over Limit",
            namespace="client",
            strategy=strategy,
            created_by=admin_user.id,
        )

        assert result is None

    @patch("ai_ready_rag.api.documents.process_document_task")
    def test_source_path_persisted_on_document(
        self, mock_task, client, admin_headers, test_tag, tmp_path, monkeypatch, db
    ):
        """source_path is persisted on the Document record even without auto-tagging."""
        test_file = tmp_path / "with_path.txt"
        test_file.write_text("Content with source path")

        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": ("with_path.txt", f, "text/plain")},
                data={
                    "tag_ids": [test_tag.id],
                    "source_path": "SomeFolder/with_path.txt",
                },
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["source_path"] == "SomeFolder/with_path.txt"

        # Cleanup
        doc_dir = upload_dir / data["id"]
        if doc_dir.exists():
            import shutil

            shutil.rmtree(doc_dir)


class TestBatchUpload:
    """Tests for batch upload endpoint."""

    @patch("ai_ready_rag.api.documents.enqueue_document_processing", new_callable=AsyncMock)
    def test_batch_upload_success(self, mock_enqueue, client, admin_headers, test_tag, tmp_path):
        """Upload 2 valid files with tag_ids - both succeed."""
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        file1 = tmp_path / "batch1.txt"
        file1.write_text("Batch file one content unique-batch-1")
        file2 = tmp_path / "batch2.txt"
        file2.write_text("Batch file two content unique-batch-2")

        with open(file1, "rb") as f1, open(file2, "rb") as f2:
            response = client.post(
                "/api/documents/upload/batch",
                files=[
                    ("files", ("batch1.txt", f1, "text/plain")),
                    ("files", ("batch2.txt", f2, "text/plain")),
                ],
                data={"tag_ids": [test_tag.id]},
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 2
        assert data["uploaded"] == 2
        assert data["duplicates"] == 0
        assert data["failed"] == 0
        assert len(data["results"]) == 2
        for r in data["results"]:
            assert r["status"] == "uploaded"
            assert r["document_id"] is not None

        # Cleanup
        for r in data["results"]:
            doc_dir = upload_dir / r["document_id"]
            if doc_dir.exists():
                import shutil

                shutil.rmtree(doc_dir)

    @patch("ai_ready_rag.api.documents.enqueue_document_processing", new_callable=AsyncMock)
    def test_batch_upload_mixed_results(
        self, mock_enqueue, client, admin_headers, test_tag, tmp_path
    ):
        """Upload 1 valid + 1 invalid type: partial success."""
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("Valid batch content unique-batch-mix")
        invalid_file = tmp_path / "invalid.exe"
        invalid_file.write_bytes(b"bad content")

        with open(valid_file, "rb") as f1, open(invalid_file, "rb") as f2:
            response = client.post(
                "/api/documents/upload/batch",
                files=[
                    ("files", ("valid.txt", f1, "text/plain")),
                    ("files", ("invalid.exe", f2, "application/octet-stream")),
                ],
                data={"tag_ids": [test_tag.id]},
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 2
        assert data["uploaded"] == 1
        assert data["failed"] == 1

        # Check the failed result has correct error code
        failed_results = [r for r in data["results"] if r["status"] == "failed"]
        assert len(failed_results) == 1
        assert failed_results[0]["error_code"] == "FAILED_TYPE"

        # Cleanup
        for r in data["results"]:
            if r["document_id"]:
                doc_dir = upload_dir / r["document_id"]
                if doc_dir.exists():
                    import shutil

                    shutil.rmtree(doc_dir)

    @patch("ai_ready_rag.api.documents.enqueue_document_processing", new_callable=AsyncMock)
    def test_batch_upload_duplicate_detection(
        self, mock_enqueue, client, admin_headers, test_tag, tmp_path
    ):
        """Upload same file+path twice: second call returns duplicate."""
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        test_file = tmp_path / "dup_test.txt"
        test_file.write_text("Duplicate detection batch content unique-batch-dup")

        # First upload
        with open(test_file, "rb") as f:
            response1 = client.post(
                "/api/documents/upload/batch",
                files=[("files", ("dup_test.txt", f, "text/plain"))],
                data={"tag_ids": [test_tag.id]},
                headers=admin_headers,
            )

        assert response1.status_code == status.HTTP_200_OK
        data1 = response1.json()
        assert data1["uploaded"] == 1
        first_doc_id = data1["results"][0]["document_id"]

        # Second upload of same file (same content, no source_path, no strategy)
        with open(test_file, "rb") as f:
            response2 = client.post(
                "/api/documents/upload/batch",
                files=[("files", ("dup_test.txt", f, "text/plain"))],
                data={"tag_ids": [test_tag.id]},
                headers=admin_headers,
            )

        assert response2.status_code == status.HTTP_200_OK
        data2 = response2.json()
        assert data2["duplicates"] == 1
        assert data2["uploaded"] == 0
        assert data2["results"][0]["status"] == "duplicate"
        assert data2["results"][0]["error_code"] == "DUPLICATE"
        assert data2["results"][0]["document_id"] == first_doc_id

        # Cleanup
        doc_dir = upload_dir / first_doc_id
        if doc_dir.exists():
            import shutil

            shutil.rmtree(doc_dir)

    @patch("ai_ready_rag.api.documents.enqueue_document_processing", new_callable=AsyncMock)
    def test_batch_upload_same_content_different_path(
        self, mock_enqueue, client, admin_headers, test_tag, tmp_path
    ):
        """Same file content with different source_paths creates two separate documents."""
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        file1 = tmp_path / "same1.txt"
        file1.write_text("Same content different path unique-batch-paths")
        file2 = tmp_path / "same2.txt"
        file2.write_text("Same content different path unique-batch-paths")

        with open(file1, "rb") as f1, open(file2, "rb") as f2:
            response = client.post(
                "/api/documents/upload/batch",
                files=[
                    ("files", ("same.txt", f1, "text/plain")),
                    ("files", ("same.txt", f2, "text/plain")),
                ],
                data={
                    "tag_ids": [test_tag.id],
                    "source_paths": ["ClientA/same.txt", "ClientB/same.txt"],
                },
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["uploaded"] == 2
        assert data["duplicates"] == 0
        # Verify different document IDs
        doc_ids = [r["document_id"] for r in data["results"]]
        assert len(set(doc_ids)) == 2

        # Cleanup
        for doc_id in doc_ids:
            doc_dir = upload_dir / doc_id
            if doc_dir.exists():
                import shutil

                shutil.rmtree(doc_dir)

    def test_batch_upload_requires_admin(self, client, user_headers, test_tag, tmp_path):
        """Non-admin users cannot access batch upload."""
        test_file = tmp_path / "noadmin.txt"
        test_file.write_text("no admin content")

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/documents/upload/batch",
                files=[("files", ("noadmin.txt", f, "text/plain"))],
                data={"tag_ids": [test_tag.id]},
                headers=user_headers,
            )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_batch_upload_source_paths_mismatch(self, client, admin_headers, test_tag, tmp_path):
        """Mismatched source_paths length returns 422."""
        file1 = tmp_path / "mismatch1.txt"
        file1.write_text("mismatch content 1")
        file2 = tmp_path / "mismatch2.txt"
        file2.write_text("mismatch content 2")

        with open(file1, "rb") as f1, open(file2, "rb") as f2:
            response = client.post(
                "/api/documents/upload/batch",
                files=[
                    ("files", ("mismatch1.txt", f1, "text/plain")),
                    ("files", ("mismatch2.txt", f2, "text/plain")),
                ],
                data={
                    "tag_ids": [test_tag.id],
                    "source_paths": ["only/one/path"],
                },
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("ai_ready_rag.api.documents.enqueue_document_processing", new_callable=AsyncMock)
    def test_batch_upload_auto_tagging(
        self, mock_enqueue, client, admin_headers, test_tag, tmp_path, monkeypatch, db
    ):
        """Files with source_paths and auto_tag=True get auto-tags applied."""
        from ai_ready_rag.config import Settings

        strategies_dir = str(Path(__file__).parent.parent / "data" / "auto_tag_strategies")
        test_settings = Settings(
            auto_tagging_enabled=True,
            auto_tagging_path_enabled=True,
            auto_tagging_create_missing_tags=True,
            auto_tagging_strategies_dir=strategies_dir,
            auto_tagging_strategy="generic",
        )
        monkeypatch.setattr("ai_ready_rag.api.documents.get_settings", lambda: test_settings)

        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        test_file = tmp_path / "autotagged.txt"
        test_file.write_text("Auto tagged batch content unique-batch-autotag")

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/documents/upload/batch",
                files=[("files", ("autotagged.txt", f, "text/plain"))],
                data={
                    "tag_ids": [test_tag.id],
                    "source_paths": ["AcmeCorp/Policies/autotagged.txt"],
                    "auto_tag": "true",
                },
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["uploaded"] == 1
        assert data["auto_tags_applied"] == 1
        assert len(data["results"][0]["auto_tags"]) > 0

        # Cleanup
        for r in data["results"]:
            if r["document_id"]:
                doc_dir = upload_dir / r["document_id"]
                if doc_dir.exists():
                    import shutil

                    shutil.rmtree(doc_dir)

    def test_batch_upload_no_tags_no_auto_tag(self, client, admin_headers, tmp_path):
        """No tag_ids and auto_tag not active returns 400 NO_TAGS."""
        test_file = tmp_path / "notags.txt"
        test_file.write_text("no tags content")

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/documents/upload/batch",
                files=[("files", ("notags.txt", f, "text/plain"))],
                data={"tag_ids": []},
                headers=admin_headers,
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "tag" in response.json()["detail"].lower()


class TestDeleteAllDocuments:
    """Tests for delete-all-documents endpoint."""

    def test_delete_all_requires_admin(self, client, user_headers):
        """Non-admin cannot delete all documents."""
        response = client.request(
            "DELETE",
            "/api/documents",
            json={"confirm": True},
            headers=user_headers,
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_delete_all_requires_confirm(self, client, admin_headers):
        """Must confirm to delete all."""
        response = client.request(
            "DELETE",
            "/api/documents",
            json={"confirm": False},
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @patch("ai_ready_rag.services.factory.get_vector_service")
    def test_delete_all_documents(self, mock_vs, client, admin_headers, test_document, db):
        """Admin can delete all documents."""
        mock_service = AsyncMock()
        mock_service.initialize = AsyncMock()
        mock_service.clear_collection = AsyncMock(return_value=True)
        mock_vs.return_value = mock_service

        response = client.request(
            "DELETE",
            "/api/documents",
            json={"confirm": True},
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["deleted_count"] >= 1

        # Verify documents are gone from database
        remaining = db.query(Document).count()
        assert remaining == 0


class TestAccessControlIntegration:
    """Integration tests for tag-based access control with auto-tagging (spec Section 10.5)."""

    @pytest.fixture
    def client_tag(self, db, admin_user):
        """Create a client: namespaced tag."""
        tag = Tag(
            name="client:acme-corp",
            display_name="Acme Corp",
            description="Auto-created client tag",
            created_by=admin_user.id,
        )
        db.add(tag)
        db.flush()
        db.refresh(tag)
        return tag

    @pytest.fixture
    def tagged_document(self, db, admin_user, client_tag):
        """Create a document with client:acme-corp tag."""
        doc = Document(
            filename="acme_policy.pdf",
            original_filename="acme_policy.pdf",
            file_path="/tmp/test/acme_policy.pdf",
            file_type="pdf",
            file_size=2048,
            status="ready",
            uploaded_by=admin_user.id,
            chunk_count=3,
        )
        doc.tags.append(client_tag)
        db.add(doc)
        db.flush()
        db.refresh(doc)
        return doc

    def test_auto_tag_not_assigned_to_non_admin(
        self, client, user_headers, tagged_document, regular_user
    ):
        """User without client:acme-corp tag cannot see the tagged document."""
        response = client.get("/api/documents", headers=user_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 0
        assert len(data["documents"]) == 0

    def test_user_without_tag_gets_zero_results(
        self, client, user_headers, tagged_document, regular_user
    ):
        """Regular user with no matching tags gets zero document results."""
        response = client.get("/api/documents", headers=user_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 0
        assert data["documents"] == []

    def test_admin_assigns_tag_user_sees_documents(
        self, client, admin_headers, user_headers, tagged_document, regular_user, client_tag, db
    ):
        """After admin assigns tag to user, user can see the document."""
        # Assign tag to user
        response = client.post(
            f"/api/users/{regular_user.id}/tags",
            headers=admin_headers,
            json={"tag_ids": [client_tag.id]},
        )
        assert response.status_code == 200

        # Now user should see the document
        response = client.get("/api/documents", headers=user_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 1
        assert data["documents"][0]["id"] == tagged_document.id

    def test_tag_access_disabled_sees_all(self, client, unrestricted_headers, tagged_document):
        """User with tag_access_enabled=False sees all documents without tag assignment."""
        response = client.get("/api/documents", headers=unrestricted_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] >= 1
        doc_ids = [d["id"] for d in data["documents"]]
        assert tagged_document.id in doc_ids

    def test_admin_always_sees_all(self, client, admin_headers, tagged_document):
        """Admin sees all documents regardless of tag assignment."""
        response = client.get("/api/documents", headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] >= 1
        doc_ids = [d["id"] for d in data["documents"]]
        assert tagged_document.id in doc_ids

    def test_remove_tag_no_longer_sees(
        self, client, admin_headers, user_headers, tagged_document, regular_user, client_tag, db
    ):
        """After tag removal, user can no longer see the document."""
        # Assign tag first
        regular_user.tags.append(client_tag)
        db.flush()

        # Verify user can see it
        response = client.get("/api/documents", headers=user_headers)
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["total"] == 1

        # Remove tag (assign empty list)
        response = client.post(
            f"/api/users/{regular_user.id}/tags",
            headers=admin_headers,
            json={"tag_ids": []},
        )
        assert response.status_code == 200

        # Verify user can no longer see it
        response = client.get("/api/documents", headers=user_headers)
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["total"] == 0
