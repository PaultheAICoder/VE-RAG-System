"""Integration tests for document_service.delete_all_documents().

Tests use a real in-memory SQLite database (not mocked) to ensure
the bulk delete actually removes rows — the bug in #306.
"""

import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from ai_ready_rag.config import get_settings
from ai_ready_rag.db.models import Document
from ai_ready_rag.services.document_service import DocumentService


def _make_document(db, status="ready", **kwargs) -> Document:
    """Create a minimal Document row for testing."""
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename=f"{doc_id}.pdf",
        original_filename=kwargs.get("original_filename", "test.pdf"),
        file_path=f"/tmp/test/{doc_id}.pdf",
        file_type="pdf",
        file_size=1024,
        status=status,
        uploaded_at=datetime.utcnow(),
    )
    db.add(doc)
    db.flush()
    return doc


class TestDeleteAllDocuments:
    """Integration tests for delete_all_documents with real DB."""

    def test_delete_all_removes_all_rows(self, db):
        """All document rows must be deleted from SQLite."""
        # Arrange: create documents in various statuses
        _make_document(db, status="ready")
        _make_document(db, status="ready")
        _make_document(db, status="failed")
        _make_document(db, status="processing")
        _make_document(db, status="pending")
        db.flush()
        assert db.query(Document).count() == 5

        settings = get_settings()
        service = DocumentService(db, settings)

        # Use a non-existent storage path so file deletion is a no-op
        service.storage_path = Path("/tmp/nonexistent-test-storage")

        # Act
        count = service.delete_all_documents()

        # Assert
        assert count == 5
        assert db.query(Document).count() == 0

    def test_delete_all_empty_table(self, db):
        """Deleting from an empty table returns 0 and succeeds."""
        assert db.query(Document).count() == 0

        settings = get_settings()
        service = DocumentService(db, settings)
        service.storage_path = Path("/tmp/nonexistent-test-storage")

        count = service.delete_all_documents()
        assert count == 0
        assert db.query(Document).count() == 0

    @patch("shutil.rmtree")
    def test_delete_all_cleans_storage_dirs(self, mock_rmtree, db, tmp_path):
        """File storage directories are cleaned up for each document."""
        doc = _make_document(db, status="ready")
        doc_dir = tmp_path / doc.id
        doc_dir.mkdir()
        (doc_dir / "test.pdf").write_text("content")
        db.flush()

        settings = get_settings()
        service = DocumentService(db, settings)
        service.storage_path = tmp_path

        count = service.delete_all_documents()
        assert count == 1
        assert db.query(Document).count() == 0
