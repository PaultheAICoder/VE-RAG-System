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
