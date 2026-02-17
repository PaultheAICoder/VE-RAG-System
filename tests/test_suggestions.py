"""Tests for tag suggestion approval endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text

from ai_ready_rag.db.models import Document, TagSuggestion


@pytest.fixture
def sample_document(db, admin_user):
    """Create a sample document for testing."""
    doc = Document(
        filename="test.pdf",
        original_filename="test.pdf",
        file_path="/tmp/test.pdf",
        file_type=".pdf",
        file_size=1024,
        status="ready",
        chunk_count=5,
        uploaded_by=admin_user.id,
    )
    db.add(doc)
    db.flush()
    db.refresh(doc)
    return doc


@pytest.fixture
def sample_suggestions(db, sample_document):
    """Create sample tag suggestions."""
    suggestions = []
    for i, (tag_name, ns, source) in enumerate(
        [
            ("doctype:policy", "doctype", "llm"),
            ("dept:hr", "dept", "llm"),
            ("topic:compliance", "topic", "llm"),
        ]
    ):
        s = TagSuggestion(
            document_id=sample_document.id,
            tag_name=tag_name,
            display_name=tag_name.split(":")[1].title(),
            namespace=ns,
            source=source,
            confidence=0.6 + i * 0.1,
            strategy_id="default",
            status="pending",
        )
        db.add(s)
        suggestions.append(s)
    db.flush()
    for s in suggestions:
        db.refresh(s)
    return suggestions


def _mock_vector_service():
    """Create a mock vector service for patching."""
    mock_vs = MagicMock()
    mock_vs.update_document_tags = AsyncMock()
    return mock_vs


class TestListSuggestions:
    def test_list_suggestions_empty(self, client, admin_headers, sample_document):
        response = client.get(
            f"/api/documents/{sample_document.id}/tag-suggestions",
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["suggestions"] == []

    def test_list_suggestions_returns_pending(
        self, client, admin_headers, sample_document, sample_suggestions
    ):
        response = client.get(
            f"/api/documents/{sample_document.id}/tag-suggestions",
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["suggestions"]) == 3
        assert all(s["status"] == "pending" for s in data["suggestions"])

    def test_list_suggestions_filter_by_status(
        self, client, admin_headers, sample_document, sample_suggestions, db
    ):
        # Mark one as approved
        sample_suggestions[0].status = "approved"
        db.flush()

        response = client.get(
            f"/api/documents/{sample_document.id}/tag-suggestions?status_filter=pending",
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

    def test_list_suggestions_requires_admin(self, client, user_headers, sample_document):
        response = client.get(
            f"/api/documents/{sample_document.id}/tag-suggestions",
            headers=user_headers,
        )
        assert response.status_code == 403

    def test_list_suggestions_document_not_found(self, client, admin_headers):
        response = client.get(
            "/api/documents/nonexistent-id/tag-suggestions",
            headers=admin_headers,
        )
        assert response.status_code == 404


class TestApproveSuggestions:
    @patch("ai_ready_rag.api.suggestions.get_vector_service")
    @patch("ai_ready_rag.api.suggestions.AutoTagStrategy.load")
    def test_approve_single_suggestion(
        self,
        mock_load,
        mock_get_vs,
        client,
        admin_headers,
        sample_document,
        sample_suggestions,
        db,
    ):
        mock_strategy = MagicMock()
        mock_strategy.id = "default"
        mock_strategy.name = "Default Strategy"
        mock_strategy.namespaces = {}
        mock_load.return_value = mock_strategy

        mock_vs = _mock_vector_service()
        mock_get_vs.return_value = mock_vs

        suggestion = sample_suggestions[0]
        response = client.post(
            f"/api/documents/{sample_document.id}/tag-suggestions/approve",
            headers=admin_headers,
            json={"suggestion_ids": [suggestion.id]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["processed_count"] == 1
        assert data["failed_count"] == 0
        assert data["results"][0]["status"] == "approved"

        # Verify suggestion was updated in DB
        db.refresh(suggestion)
        assert suggestion.status == "approved"
        assert suggestion.reviewed_by is not None
        assert suggestion.reviewed_at is not None

    @patch("ai_ready_rag.api.suggestions.get_vector_service")
    @patch("ai_ready_rag.api.suggestions.AutoTagStrategy.load")
    def test_approve_already_processed(
        self,
        mock_load,
        mock_get_vs,
        client,
        admin_headers,
        sample_document,
        sample_suggestions,
        db,
    ):
        mock_strategy = MagicMock()
        mock_strategy.id = "default"
        mock_strategy.namespaces = {}
        mock_load.return_value = mock_strategy

        mock_vs = _mock_vector_service()
        mock_get_vs.return_value = mock_vs

        # Pre-approve the suggestion
        sample_suggestions[0].status = "approved"
        db.flush()

        response = client.post(
            f"/api/documents/{sample_document.id}/tag-suggestions/approve",
            headers=admin_headers,
            json={"suggestion_ids": [sample_suggestions[0].id]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["results"][0]["status"] == "already_processed"
        assert data["processed_count"] == 0

    def test_approve_invalid_id(self, client, admin_headers, sample_document):
        response = client.post(
            f"/api/documents/{sample_document.id}/tag-suggestions/approve",
            headers=admin_headers,
            json={"suggestion_ids": ["nonexistent-id"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["results"][0]["status"] == "failed"
        assert "not found" in data["results"][0]["error"]

    @patch("ai_ready_rag.api.suggestions.get_vector_service")
    @patch("ai_ready_rag.api.suggestions.AutoTagStrategy.load")
    def test_approve_wrong_document(
        self,
        mock_load,
        mock_get_vs,
        client,
        admin_headers,
        sample_document,
        sample_suggestions,
        db,
        admin_user,
    ):
        # Create a second document
        doc2 = Document(
            filename="other.pdf",
            original_filename="other.pdf",
            file_path="/tmp/other.pdf",
            file_type=".pdf",
            file_size=512,
            status="ready",
            chunk_count=2,
            uploaded_by=admin_user.id,
        )
        db.add(doc2)
        db.flush()

        # Try to approve suggestion from sample_document under doc2
        response = client.post(
            f"/api/documents/{doc2.id}/tag-suggestions/approve",
            headers=admin_headers,
            json={"suggestion_ids": [sample_suggestions[0].id]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["results"][0]["status"] == "failed"
        assert "does not belong" in data["results"][0]["error"]

    def test_approve_requires_admin(
        self, client, user_headers, sample_document, sample_suggestions
    ):
        response = client.post(
            f"/api/documents/{sample_document.id}/tag-suggestions/approve",
            headers=user_headers,
            json={"suggestion_ids": [sample_suggestions[0].id]},
        )
        assert response.status_code == 403


class TestRejectSuggestions:
    def test_reject_single_suggestion(
        self, client, admin_headers, sample_document, sample_suggestions, db
    ):
        suggestion = sample_suggestions[0]
        response = client.post(
            f"/api/documents/{sample_document.id}/tag-suggestions/reject",
            headers=admin_headers,
            json={"suggestion_ids": [suggestion.id]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["processed_count"] == 1
        assert data["results"][0]["status"] == "rejected"

        db.refresh(suggestion)
        assert suggestion.status == "rejected"
        assert suggestion.reviewed_by is not None

    def test_reject_already_processed(
        self, client, admin_headers, sample_document, sample_suggestions, db
    ):
        sample_suggestions[0].status = "rejected"
        db.flush()

        response = client.post(
            f"/api/documents/{sample_document.id}/tag-suggestions/reject",
            headers=admin_headers,
            json={"suggestion_ids": [sample_suggestions[0].id]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["results"][0]["status"] == "already_processed"

    def test_reject_requires_admin(self, client, user_headers, sample_document, sample_suggestions):
        response = client.post(
            f"/api/documents/{sample_document.id}/tag-suggestions/reject",
            headers=user_headers,
            json={"suggestion_ids": [sample_suggestions[0].id]},
        )
        assert response.status_code == 403


class TestBatchApprove:
    @patch("ai_ready_rag.api.suggestions.get_vector_service")
    @patch("ai_ready_rag.api.suggestions.AutoTagStrategy.load")
    def test_batch_approve_across_documents(
        self,
        mock_load,
        mock_get_vs,
        client,
        admin_headers,
        sample_document,
        sample_suggestions,
        db,
        admin_user,
    ):
        mock_strategy = MagicMock()
        mock_strategy.id = "default"
        mock_strategy.namespaces = {}
        mock_load.return_value = mock_strategy

        mock_vs = _mock_vector_service()
        mock_get_vs.return_value = mock_vs

        # Create second document with its own suggestion
        doc2 = Document(
            filename="other.pdf",
            original_filename="other.pdf",
            file_path="/tmp/other.pdf",
            file_type=".pdf",
            file_size=512,
            status="ready",
            chunk_count=2,
            uploaded_by=admin_user.id,
        )
        db.add(doc2)
        db.flush()

        s2 = TagSuggestion(
            document_id=doc2.id,
            tag_name="dept:finance",
            display_name="Finance",
            namespace="dept",
            source="llm",
            confidence=0.7,
            strategy_id="default",
            status="pending",
        )
        db.add(s2)
        db.flush()
        db.refresh(s2)

        response = client.post(
            "/api/documents/tag-suggestions/approve-batch",
            headers=admin_headers,
            json={"suggestion_ids": [sample_suggestions[0].id, s2.id]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["processed_count"] == 2
        assert data["failed_count"] == 0

        # Vector store should have been called for both documents
        assert mock_vs.update_document_tags.call_count == 2

    @patch("ai_ready_rag.api.suggestions.get_vector_service")
    @patch("ai_ready_rag.api.suggestions.AutoTagStrategy.load")
    def test_batch_approve_mixed_results(
        self,
        mock_load,
        mock_get_vs,
        client,
        admin_headers,
        sample_document,
        sample_suggestions,
        db,
    ):
        mock_strategy = MagicMock()
        mock_strategy.id = "default"
        mock_strategy.namespaces = {}
        mock_load.return_value = mock_strategy

        mock_vs = _mock_vector_service()
        mock_get_vs.return_value = mock_vs

        # Pre-approve one
        sample_suggestions[0].status = "approved"
        db.flush()

        response = client.post(
            "/api/documents/tag-suggestions/approve-batch",
            headers=admin_headers,
            json={
                "suggestion_ids": [
                    sample_suggestions[0].id,
                    sample_suggestions[1].id,
                    "nonexistent-id",
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["processed_count"] == 1  # Only the pending one
        assert data["failed_count"] == 1  # nonexistent

        statuses = {r["suggestion_id"]: r["status"] for r in data["results"]}
        assert statuses[sample_suggestions[0].id] == "already_processed"
        assert statuses[sample_suggestions[1].id] == "approved"
        assert statuses["nonexistent-id"] == "failed"


class TestTagSuggestionModel:
    def test_cascade_delete_fk_defined(self, db, sample_document, sample_suggestions):
        """Verify CASCADE FK is defined on tag_suggestions.document_id.

        In-memory SQLite test sessions don't enforce PRAGMA foreign_keys=ON
        at connect time, so we verify the schema constraint is present instead.
        Production enforces FK via the engine connect event in database.py.
        """
        result = db.execute(
            text("SELECT sql FROM sqlite_master WHERE name='tag_suggestions'")
        ).fetchone()
        assert result is not None
        create_sql = result[0]
        assert "ON DELETE CASCADE" in create_sql

    def test_suggestion_default_values(self, db, sample_document):
        """Verify default values are set correctly."""
        s = TagSuggestion(
            document_id=sample_document.id,
            tag_name="test:tag",
            display_name="Tag",
            namespace="test",
            source="llm",
            strategy_id="default",
        )
        db.add(s)
        db.flush()
        db.refresh(s)

        assert s.id is not None
        assert s.status == "pending"
        assert s.confidence == 1.0
        assert s.reviewed_by is None
        assert s.reviewed_at is None
        assert s.created_at is not None
