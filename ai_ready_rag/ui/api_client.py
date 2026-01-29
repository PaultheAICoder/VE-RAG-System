"""Stateless API client for Gradio UI - uses Bearer tokens from gr.State."""

from typing import Any

import httpx

# Default base URL - can be overridden via environment
BASE_URL = "http://localhost:8000"


ERROR_RESPONSES: dict[int, dict[str, str]] = {
    401: {
        "type": "auth",
        "message": "Session expired. Please log in again.",
        "action": "redirect_to_login",
    },
    403: {
        "type": "auth",
        "message": "You don't have permission for this action.",
        "action": "none",
    },
    404: {
        "type": "not_found",
        "message": "The requested resource was not found.",
        "action": "none",
    },
    503: {
        "type": "server",
        "message": "AI service temporarily unavailable. Please try again.",
        "action": "retry",
    },
}


def handle_api_error(e: httpx.HTTPStatusError) -> dict[str, Any]:
    """Convert HTTP errors to user-friendly messages.

    Returns:
        dict with keys: type, message, action
    """
    status = e.response.status_code
    return ERROR_RESPONSES.get(
        status,
        {
            "type": "server",
            "message": f"An error occurred (code {status}). Please try again.",
            "action": "retry",
        },
    )


class GradioAPIClient:
    """Stateless API client - token passed per-request from gr.State.

    All methods are static. Token is passed as parameter, not stored in client instance.
    This ensures per-session isolation in Gradio's multi-user environment.
    """

    @staticmethod
    def _headers(token: str | None) -> dict[str, str]:
        """Build headers with Bearer token if authenticated."""
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def login(email: str, password: str) -> dict[str, Any]:
        """Authenticate and return token + user info.

        Args:
            email: User's email address
            password: User's password

        Returns:
            dict with access_token and user info

        Raises:
            httpx.HTTPStatusError: On authentication failure
        """
        with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
            response = client.post(
                "/api/auth/login",
                json={"email": email, "password": password},
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def logout(token: str) -> None:
        """Invalidate the current session.

        Args:
            token: JWT access token
        """
        with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
            response = client.post(
                "/api/auth/logout",
                headers=GradioAPIClient._headers(token),
            )
            # Ignore errors on logout - just clear local state
            response.raise_for_status()

    @staticmethod
    def get_me(token: str) -> dict[str, Any]:
        """Get current user info.

        Args:
            token: JWT access token

        Returns:
            User info dict
        """
        with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
            response = client.get(
                "/api/auth/me",
                headers=GradioAPIClient._headers(token),
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def get_sessions(token: str, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """List user's chat sessions with pagination.

        Args:
            token: JWT access token
            limit: Max sessions to return (default 20)
            offset: Pagination offset

        Returns:
            dict with sessions list, total, limit, offset
        """
        with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
            response = client.get(
                "/api/chat/sessions",
                params={"limit": limit, "offset": offset},
                headers=GradioAPIClient._headers(token),
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def create_session(token: str, title: str | None = None) -> dict[str, Any]:
        """Create a new chat session.

        Args:
            token: JWT access token
            title: Optional session title

        Returns:
            Created session info
        """
        with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
            response = client.post(
                "/api/chat/sessions",
                json={"title": title},
                headers=GradioAPIClient._headers(token),
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def get_messages(token: str, session_id: str, limit: int = 50) -> dict[str, Any]:
        """Load messages for a chat session.

        Args:
            token: JWT access token
            session_id: Chat session ID
            limit: Max messages to return

        Returns:
            dict with messages list, has_more, total
        """
        with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
            response = client.get(
                f"/api/chat/sessions/{session_id}/messages",
                params={"limit": limit},
                headers=GradioAPIClient._headers(token),
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def send_message(token: str, session_id: str, content: str) -> dict[str, Any]:
        """Send message and get RAG response.

        Args:
            token: JWT access token
            session_id: Chat session ID
            content: Message content

        Returns:
            dict with user_message, assistant_message, generation_time_ms
        """
        # Long timeout for RAG operations
        with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
            response = client.post(
                f"/api/chat/sessions/{session_id}/messages",
                json={"content": content},
                headers=GradioAPIClient._headers(token),
            )
            response.raise_for_status()
            return response.json()

    # --- Document Management APIs ---

    @staticmethod
    def get_tags(token: str) -> list[dict[str, Any]]:
        """Get all available tags.

        Args:
            token: JWT access token

        Returns:
            List of tag dicts with id, name, display_name
        """
        with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
            response = client.get(
                "/api/tags",
                headers=GradioAPIClient._headers(token),
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def list_documents(
        token: str,
        limit: int = 20,
        offset: int = 0,
        status_filter: str | None = None,
        tag_id: str | None = None,
        search: str | None = None,
    ) -> dict[str, Any]:
        """List documents with filtering.

        Args:
            token: JWT access token
            limit: Max documents to return
            offset: Pagination offset
            status_filter: Filter by status (pending, processing, ready, failed)
            tag_id: Filter by tag ID
            search: Search term for filename/title

        Returns:
            dict with documents list, total, limit, offset
        """
        params = {"limit": limit, "offset": offset}
        if status_filter:
            params["status"] = status_filter
        if tag_id:
            params["tag_id"] = tag_id
        if search:
            params["search"] = search

        with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
            response = client.get(
                "/api/documents",
                params=params,
                headers=GradioAPIClient._headers(token),
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def upload_document(
        token: str,
        file_path: str,
        tag_ids: list[str],
        title: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Upload a document.

        Args:
            token: JWT access token
            file_path: Path to file to upload
            tag_ids: List of tag IDs to assign
            title: Optional title
            description: Optional description

        Returns:
            Created document info
        """
        from pathlib import Path

        with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
            with open(file_path, "rb") as f:
                files = {"file": (Path(file_path).name, f)}
                data: dict[str, Any] = {"tag_ids": tag_ids}
                if title:
                    data["title"] = title
                if description:
                    data["description"] = description

                response = client.post(
                    "/api/documents/upload",
                    files=files,
                    data=data,
                    headers={"Authorization": f"Bearer {token}"},
                )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def delete_document(token: str, document_id: str) -> bool:
        """Delete a document.

        Args:
            token: JWT access token
            document_id: Document ID to delete

        Returns:
            True if deleted successfully
        """
        with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
            response = client.delete(
                f"/api/documents/{document_id}",
                headers=GradioAPIClient._headers(token),
            )
            response.raise_for_status()
            return True

    @staticmethod
    def get_document(token: str, document_id: str) -> dict[str, Any]:
        """Get a single document by ID.

        Args:
            token: JWT access token
            document_id: Document ID

        Returns:
            Document data dict
        """
        with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
            response = client.get(
                f"/api/documents/{document_id}",
                headers=GradioAPIClient._headers(token),
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def update_document_tags(token: str, document_id: str, tag_ids: list[str]) -> dict[str, Any]:
        """Update document tags.

        Args:
            token: JWT access token
            document_id: Document ID
            tag_ids: New tag IDs

        Returns:
            Updated document info
        """
        with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
            response = client.patch(
                f"/api/documents/{document_id}/tags",
                json={"tag_ids": tag_ids},
                headers=GradioAPIClient._headers(token),
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def bulk_delete_documents(token: str, document_ids: list[str]) -> dict[str, Any]:
        """Delete multiple documents at once.

        Args:
            token: JWT access token
            document_ids: List of document IDs to delete

        Returns:
            Dict with results, deleted_count, failed_count
        """
        with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
            response = client.post(
                "/api/documents/bulk-delete",
                json={"document_ids": document_ids},
                headers=GradioAPIClient._headers(token),
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def reprocess_document(token: str, document_id: str) -> dict[str, Any]:
        """Reprocess a document.

        Args:
            token: JWT access token
            document_id: Document ID to reprocess

        Returns:
            Updated document info with pending status
        """
        with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
            response = client.post(
                f"/api/documents/{document_id}/reprocess",
                headers=GradioAPIClient._headers(token),
            )
            response.raise_for_status()
            return response.json()
