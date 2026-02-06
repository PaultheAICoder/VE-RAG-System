"""Tests for Chat API endpoints."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_ready_rag.core.exceptions import LLMConnectionError
from ai_ready_rag.db.models import ChatMessage, ChatSession
from ai_ready_rag.services.rag_service import (
    Citation,
    ConfidenceScore,
    RAGResponse,
    RouteTarget,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_session(db, regular_user) -> ChatSession:
    """Create test chat session for regular user."""
    session = ChatSession(
        user_id=regular_user.id,
        title="Test Session",
    )
    db.add(session)
    db.flush()
    db.refresh(session)
    return session


@pytest.fixture
def test_session_with_messages(db, regular_user) -> ChatSession:
    """Create test session with messages for pagination tests."""
    session = ChatSession(
        user_id=regular_user.id,
        title="Session with Messages",
    )
    db.add(session)
    db.flush()

    # Add 25 messages for pagination testing
    # Use incrementing timestamps to ensure ordering
    base_time = datetime.utcnow() - timedelta(minutes=30)
    for i in range(25):
        role = "user" if i % 2 == 0 else "assistant"
        msg = ChatMessage(
            session_id=session.id,
            role=role,
            content=f"Message {i} content",
            confidence=75.0 if role == "assistant" else None,
            created_at=base_time + timedelta(seconds=i),
        )
        db.add(msg)

    db.flush()
    db.refresh(session)
    return session


@pytest.fixture
def other_user_session(db, admin_user) -> ChatSession:
    """Create session owned by another user (admin)."""
    session = ChatSession(
        user_id=admin_user.id,
        title="Admin's Session",
    )
    db.add(session)
    db.flush()
    db.refresh(session)
    return session


@pytest.fixture
def archived_session(db, regular_user) -> ChatSession:
    """Create archived session."""
    session = ChatSession(
        user_id=regular_user.id,
        title="Archived Session",
        is_archived=True,
    )
    db.add(session)
    db.flush()
    db.refresh(session)
    return session


@pytest.fixture
def mock_rag_response():
    """Mock RAGResponse for message send tests."""
    return RAGResponse(
        answer="Test answer based on documents. [SourceId: 550e8400-e29b-41d4-a716-446655440000:0]",
        confidence=ConfidenceScore(
            overall=75,
            retrieval_score=0.8,
            coverage_score=0.7,
            llm_score=75,
        ),
        citations=[
            Citation(
                source_id="550e8400-e29b-41d4-a716-446655440000:0",
                document_id="550e8400-e29b-41d4-a716-446655440000",
                document_name="policy.pdf",
                chunk_index=0,
                page_number=1,
                section="Overview",
                relevance_score=0.85,
                snippet="This is a test snippet...",
                snippet_full="This is a full test snippet for hover display...",
            )
        ],
        action="CITE",
        route_to=None,
        model_used="llama3.2",
        context_chunks_used=3,
        context_tokens_used=500,
        generation_time_ms=1250.5,
        grounded=True,
    )


@pytest.fixture
def mock_rag_response_routed():
    """Mock RAGResponse with ROUTE action for routing tests."""
    return RAGResponse(
        answer="I don't have enough information to answer confidently.",
        confidence=ConfidenceScore(
            overall=35,
            retrieval_score=0.4,
            coverage_score=0.3,
            llm_score=35,
        ),
        citations=[],
        action="ROUTE",
        route_to=RouteTarget(
            tag="hr",
            owner_user_id="user-123",
            owner_email="hr-expert@company.com",
            reason="Low confidence - insufficient context",
            fallback=False,
        ),
        model_used="llama3.2",
        context_chunks_used=1,
        context_tokens_used=200,
        generation_time_ms=850.0,
        grounded=False,
    )


# =============================================================================
# Session Tests
# =============================================================================


class TestCreateSession:
    """POST /api/chat/sessions tests."""

    def test_create_session(self, client, user_headers):
        """Create session without title."""
        response = client.post(
            "/api/chat/sessions",
            headers=user_headers,
            json={},
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["title"] is None
        assert data["is_archived"] is False
        assert data["message_count"] == 0

    def test_create_session_with_title(self, client, user_headers):
        """Create session with custom title."""
        response = client.post(
            "/api/chat/sessions",
            headers=user_headers,
            json={"title": "My Research Chat"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "My Research Chat"

    def test_create_session_unauthenticated(self, client):
        """Unauthenticated request rejected."""
        response = client.post("/api/chat/sessions", json={})
        assert response.status_code == 401

    def test_create_session_sets_user_id(self, client, user_headers, regular_user):
        """User ID set from auth token."""
        response = client.post(
            "/api/chat/sessions",
            headers=user_headers,
            json={},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["user_id"] == regular_user.id


class TestListSessions:
    """GET /api/chat/sessions tests."""

    def test_list_sessions_empty(self, client, user_headers):
        """Empty list when no sessions."""
        response = client.get("/api/chat/sessions", headers=user_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["sessions"] == []
        assert data["total"] == 0

    def test_list_sessions_returns_own_only(
        self, client, user_headers, test_session, other_user_session
    ):
        """Only own sessions returned."""
        response = client.get("/api/chat/sessions", headers=user_headers)
        assert response.status_code == 200
        data = response.json()
        session_ids = [s["id"] for s in data["sessions"]]
        assert test_session.id in session_ids
        assert other_user_session.id not in session_ids

    def test_list_sessions_pagination(self, client, db, user_headers, regular_user):
        """Pagination works correctly."""
        # Create 15 sessions
        for i in range(15):
            session = ChatSession(user_id=regular_user.id, title=f"Session {i}")
            db.add(session)
        db.flush()

        response = client.get("/api/chat/sessions?limit=10&offset=5", headers=user_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) <= 10
        assert data["limit"] == 10
        assert data["offset"] == 5
        assert data["total"] == 15

    def test_list_sessions_excludes_archived(
        self, client, user_headers, test_session, archived_session
    ):
        """Archived sessions excluded by default."""
        response = client.get("/api/chat/sessions", headers=user_headers)
        assert response.status_code == 200
        data = response.json()
        session_ids = [s["id"] for s in data["sessions"]]
        assert test_session.id in session_ids
        assert archived_session.id not in session_ids


class TestGetSession:
    """GET /api/chat/sessions/{session_id} tests."""

    def test_get_session(self, client, user_headers, test_session):
        """Get session details."""
        response = client.get(f"/api/chat/sessions/{test_session.id}", headers=user_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_session.id
        assert data["title"] == test_session.title

    def test_get_session_not_found(self, client, user_headers):
        """404 for non-existent session."""
        response = client.get("/api/chat/sessions/nonexistent-id", headers=user_headers)
        assert response.status_code == 404

    def test_get_session_other_user_forbidden(self, client, user_headers, other_user_session):
        """Cannot access other user's session (returns 404 for security)."""
        response = client.get(f"/api/chat/sessions/{other_user_session.id}", headers=user_headers)
        assert response.status_code == 404


class TestUpdateSession:
    """PATCH /api/chat/sessions/{session_id} tests."""

    def test_update_session_title(self, client, user_headers, test_session):
        """Update session title."""
        response = client.patch(
            f"/api/chat/sessions/{test_session.id}",
            headers=user_headers,
            json={"title": "Updated Title"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"

    def test_update_session_archive(self, client, user_headers, test_session):
        """Archive session."""
        response = client.patch(
            f"/api/chat/sessions/{test_session.id}",
            headers=user_headers,
            json={"is_archived": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_archived"] is True


# =============================================================================
# Message Tests
# =============================================================================


class TestSendMessage:
    """POST /api/chat/sessions/{session_id}/messages tests."""

    def test_send_message_success(self, client, user_headers, test_session, mock_rag_response):
        """Send message and receive response."""
        # Mock RAGService (VectorService is singleton from app.state)
        with patch("ai_ready_rag.api.chat.RAGService") as mock_rag_class:
            mock_rag_instance = MagicMock()
            mock_rag_instance.generate = AsyncMock(return_value=mock_rag_response)
            mock_rag_class.return_value = mock_rag_instance

            response = client.post(
                f"/api/chat/sessions/{test_session.id}/messages",
                headers=user_headers,
                json={"content": "What is the vacation policy?"},
            )

        assert response.status_code == 201
        data = response.json()

        # Check user message
        assert data["user_message"]["role"] == "user"
        assert data["user_message"]["content"] == "What is the vacation policy?"

        # Check assistant message
        assert data["assistant_message"]["role"] == "assistant"
        assert "Test answer" in data["assistant_message"]["content"]
        assert data["assistant_message"]["confidence"]["overall"] == 75
        assert data["assistant_message"]["was_routed"] is False

        # Check generation time
        assert "generation_time_ms" in data

    def test_send_message_with_routing(
        self, client, user_headers, test_session, mock_rag_response_routed
    ):
        """Message routed when confidence low."""
        # Mock RAGService (VectorService is singleton from app.state)
        with patch("ai_ready_rag.api.chat.RAGService") as mock_rag_class:
            mock_rag_instance = MagicMock()
            mock_rag_instance.generate = AsyncMock(return_value=mock_rag_response_routed)
            mock_rag_class.return_value = mock_rag_instance

            response = client.post(
                f"/api/chat/sessions/{test_session.id}/messages",
                headers=user_headers,
                json={"content": "Complex question?"},
            )

        assert response.status_code == 201
        data = response.json()

        assert data["assistant_message"]["was_routed"] is True
        assert data["assistant_message"]["routed_to"] == "hr-expert@company.com"
        assert data["assistant_message"]["confidence"]["overall"] == 35

    def test_send_message_session_not_found(self, client, user_headers):
        """404 for non-existent session."""
        response = client.post(
            "/api/chat/sessions/nonexistent-id/messages",
            headers=user_headers,
            json={"content": "Hello?"},
        )
        assert response.status_code == 404

    def test_send_message_empty_content(self, client, user_headers, test_session):
        """422 for empty content (Pydantic validation error)."""
        response = client.post(
            f"/api/chat/sessions/{test_session.id}/messages",
            headers=user_headers,
            json={"content": ""},
        )
        # FastAPI returns 422 for Pydantic validation errors
        assert response.status_code == 422

    def test_send_message_content_too_long(self, client, user_headers, test_session):
        """422 for content > 4000 chars (Pydantic validation error)."""
        long_content = "x" * 4001
        response = client.post(
            f"/api/chat/sessions/{test_session.id}/messages",
            headers=user_headers,
            json={"content": long_content},
        )
        # FastAPI returns 422 for Pydantic validation errors
        assert response.status_code == 422

    def test_send_message_ollama_unavailable(self, client, user_headers, test_session):
        """503 when Ollama unavailable."""
        # Mock RAGService (VectorService is singleton from app.state)
        with patch("ai_ready_rag.api.chat.RAGService") as mock_rag_class:
            mock_rag_instance = MagicMock()
            mock_rag_instance.generate = AsyncMock(
                side_effect=LLMConnectionError("Cannot connect to Ollama")
            )
            mock_rag_class.return_value = mock_rag_instance

            response = client.post(
                f"/api/chat/sessions/{test_session.id}/messages",
                headers=user_headers,
                json={"content": "Hello?"},
            )

        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()


class TestGetMessages:
    """GET /api/chat/sessions/{session_id}/messages tests."""

    def test_get_messages(self, client, user_headers, test_session_with_messages):
        """Get messages in chronological order."""
        response = client.get(
            f"/api/chat/sessions/{test_session_with_messages.id}/messages",
            headers=user_headers,
        )
        assert response.status_code == 200
        data = response.json()

        # Verify chronological order (oldest first)
        messages = data["messages"]
        assert len(messages) > 0
        assert "Message 0" in messages[0]["content"]

    def test_get_messages_pagination(self, client, user_headers, test_session_with_messages):
        """Pagination with limit."""
        response = client.get(
            f"/api/chat/sessions/{test_session_with_messages.id}/messages?limit=10",
            headers=user_headers,
        )
        assert response.status_code == 200
        data = response.json()

        assert len(data["messages"]) == 10
        assert data["has_more"] is True
        assert data["total"] == 25

    def test_get_messages_before_cursor(self, client, db, user_headers, test_session_with_messages):
        """Get messages before cursor."""
        # Get a message ID to use as cursor
        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == test_session_with_messages.id)
            .order_by(ChatMessage.created_at)
            .all()
        )
        cursor_msg = messages[15]  # Use 16th message as cursor

        response = client.get(
            f"/api/chat/sessions/{test_session_with_messages.id}/messages?before={cursor_msg.id}",
            headers=user_headers,
        )
        assert response.status_code == 200
        data = response.json()

        # All returned messages should be before cursor
        for msg in data["messages"]:
            assert msg["id"] != cursor_msg.id

    def test_get_messages_session_not_found(self, client, user_headers):
        """404 for non-existent session."""
        response = client.get(
            "/api/chat/sessions/nonexistent-id/messages",
            headers=user_headers,
        )
        assert response.status_code == 404


# =============================================================================
# Admin Deletion Tests
# =============================================================================


class TestDeleteSession:
    """DELETE /api/chat/sessions/{session_id} tests."""

    def test_delete_session_admin_success(self, client, admin_headers, test_session):
        """Admin can delete a session."""
        response = client.delete(
            f"/api/chat/sessions/{test_session.id}",
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted_session_id"] == test_session.id
        assert data["deleted_messages_count"] == 0

    def test_delete_session_customer_admin_success(
        self, client, customer_admin_headers, test_session
    ):
        """Customer admin can delete a session."""
        response = client.delete(
            f"/api/chat/sessions/{test_session.id}",
            headers=customer_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_delete_session_regular_user_forbidden(self, client, user_headers, test_session):
        """Regular user cannot delete sessions (403)."""
        response = client.delete(
            f"/api/chat/sessions/{test_session.id}",
            headers=user_headers,
        )
        assert response.status_code == 403

    def test_delete_session_unauthenticated(self, client, test_session):
        """Unauthenticated request rejected (401)."""
        response = client.delete(f"/api/chat/sessions/{test_session.id}")
        assert response.status_code == 401

    def test_delete_session_not_found(self, client, admin_headers):
        """404 for non-existent session."""
        response = client.delete(
            "/api/chat/sessions/nonexistent-session-id",
            headers=admin_headers,
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"

    def test_delete_session_cascades_messages(
        self, client, db, admin_headers, test_session_with_messages
    ):
        """Deleting session cascades to messages."""
        session_id = test_session_with_messages.id

        # Verify messages exist before deletion
        message_count = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count()
        assert message_count == 25

        response = client.delete(
            f"/api/chat/sessions/{session_id}",
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_messages_count"] == 25

        # Verify session and messages are deleted
        db.expire_all()
        assert db.query(ChatSession).filter(ChatSession.id == session_id).first() is None
        assert db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count() == 0


class TestBulkDeleteSessions:
    """DELETE /api/chat/sessions/bulk tests."""

    def test_bulk_delete_admin_success(self, client, db, admin_headers, regular_user):
        """Admin can bulk delete sessions."""
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = ChatSession(user_id=regular_user.id, title=f"Bulk Delete Test {i}")
            db.add(session)
            sessions.append(session)
        db.flush()
        session_ids = [s.id for s in sessions]

        response = client.request(
            "DELETE",
            "/api/chat/sessions/bulk",
            headers=admin_headers,
            json={"session_ids": session_ids},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted_count"] == 3
        assert data["failed_ids"] == []
        assert data["total_messages_deleted"] == 0

    def test_bulk_delete_customer_admin_success(
        self, client, db, customer_admin_headers, regular_user
    ):
        """Customer admin can bulk delete sessions."""
        session = ChatSession(user_id=regular_user.id, title="Bulk Delete Test")
        db.add(session)
        db.flush()

        response = client.request(
            "DELETE",
            "/api/chat/sessions/bulk",
            headers=customer_admin_headers,
            json={"session_ids": [session.id]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted_count"] == 1

    def test_bulk_delete_regular_user_forbidden(self, client, user_headers, test_session):
        """Regular user cannot bulk delete sessions (403)."""
        response = client.request(
            "DELETE",
            "/api/chat/sessions/bulk",
            headers=user_headers,
            json={"session_ids": [test_session.id]},
        )
        assert response.status_code == 403

    def test_bulk_delete_empty_list(self, client, admin_headers):
        """400 for empty session_ids list."""
        response = client.request(
            "DELETE",
            "/api/chat/sessions/bulk",
            headers=admin_headers,
            json={"session_ids": []},
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_bulk_delete_over_limit(self, client, admin_headers):
        """400 for session_ids > 100."""
        # Create list of 101 fake IDs
        session_ids = [f"fake-session-{i}" for i in range(101)]
        response = client.request(
            "DELETE",
            "/api/chat/sessions/bulk",
            headers=admin_headers,
            json={"session_ids": session_ids},
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_bulk_delete_partial_not_found(self, client, db, admin_headers, regular_user):
        """Some IDs not found are reported in failed_ids."""
        # Create one real session
        session = ChatSession(user_id=regular_user.id, title="Real Session")
        db.add(session)
        db.flush()

        response = client.request(
            "DELETE",
            "/api/chat/sessions/bulk",
            headers=admin_headers,
            json={"session_ids": [session.id, "nonexistent-1", "nonexistent-2"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted_count"] == 1
        assert sorted(data["failed_ids"]) == sorted(["nonexistent-1", "nonexistent-2"])

    def test_bulk_delete_cascades_all_messages(self, client, db, admin_headers, regular_user):
        """Bulk delete cascades all messages from all sessions."""
        # Create sessions with messages
        sessions = []
        for i in range(2):
            session = ChatSession(user_id=regular_user.id, title=f"Session {i}")
            db.add(session)
            db.flush()
            sessions.append(session)

            # Add 5 messages per session
            for j in range(5):
                msg = ChatMessage(
                    session_id=session.id,
                    role="user" if j % 2 == 0 else "assistant",
                    content=f"Message {j}",
                )
                db.add(msg)

        db.flush()
        session_ids = [s.id for s in sessions]

        response = client.request(
            "DELETE",
            "/api/chat/sessions/bulk",
            headers=admin_headers,
            json={"session_ids": session_ids},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] == 2
        assert data["total_messages_deleted"] == 10

        # Verify all sessions and messages are deleted
        db.expire_all()
        for session_id in session_ids:
            assert db.query(ChatSession).filter(ChatSession.id == session_id).first() is None
            assert db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count() == 0
