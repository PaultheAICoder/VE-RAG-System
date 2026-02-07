"""Tests for structured logging (issue #176).

Tests:
- configure_logging sets up structlog with JSON/console renderers
- request_id context variable propagation
- RequestLoggingMiddleware adds X-Request-ID header
- RequestLoggingMiddleware skips health/static paths
- Failed login produces warning log
- Successful login produces info log
- No print() calls remain in production code
"""

import logging
import uuid
from unittest.mock import patch

from ai_ready_rag.core.logging import configure_logging, request_id_var


class TestConfigureLogging:
    """Test structured logging configuration."""

    def test_configure_logging_sets_root_level(self):
        """configure_logging sets root logger level."""
        configure_logging(log_level="DEBUG", log_format="json")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

        # Restore
        configure_logging(log_level="INFO", log_format="json")

    def test_configure_logging_json_format(self):
        """JSON format produces structured output."""
        configure_logging(log_level="INFO", log_format="json")
        root = logging.getLogger()
        assert len(root.handlers) > 0

    def test_configure_logging_console_format(self):
        """Console format configures without error."""
        configure_logging(log_level="INFO", log_format="console")
        root = logging.getLogger()
        assert len(root.handlers) > 0

        # Restore to JSON for other tests
        configure_logging(log_level="INFO", log_format="json")

    def test_noisy_loggers_suppressed(self):
        """Third-party loggers set to WARNING."""
        configure_logging(log_level="DEBUG", log_format="json")
        for name in ("httpx", "httpcore", "uvicorn.access", "watchfiles"):
            assert logging.getLogger(name).level == logging.WARNING

        configure_logging(log_level="INFO", log_format="json")


class TestRequestIdVar:
    """Test request_id context variable."""

    def test_request_id_default_none(self):
        """Default request_id is None."""
        # Reset to default
        token = request_id_var.set(None)
        assert request_id_var.get() is None
        request_id_var.reset(token)

    def test_request_id_set_get(self):
        """Can set and get request_id."""
        rid = str(uuid.uuid4())
        token = request_id_var.set(rid)
        assert request_id_var.get() == rid
        request_id_var.reset(token)


class TestRequestLoggingMiddleware:
    """Test request logging middleware via TestClient."""

    def test_response_has_request_id_header(self, client):
        """Every response includes X-Request-ID header."""
        response = client.get("/api/health")
        assert "X-Request-ID" in response.headers
        # Should be a valid UUID
        rid = response.headers["X-Request-ID"]
        uuid.UUID(rid)  # Raises if invalid

    def test_request_id_unique_per_request(self, client):
        """Each request gets a unique X-Request-ID."""
        r1 = client.get("/api/health")
        r2 = client.get("/api/health")
        assert r1.headers["X-Request-ID"] != r2.headers["X-Request-ID"]

    def test_api_routes_have_request_id(self, client, admin_headers):
        """API routes also get X-Request-ID."""
        response = client.get("/api/auth/me", headers=admin_headers)
        assert "X-Request-ID" in response.headers


class TestLoginLogging:
    """Test login event logging."""

    def test_failed_login_logs_warning(self, client, admin_user):
        """Failed login attempt produces warning log."""
        with patch("ai_ready_rag.api.auth.logger") as mock_logger:
            client.post(
                "/api/auth/login",
                json={"email": "admin@test.com", "password": "wrong_password"},
            )
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert call_args[0][0] == "login_failed"
            assert call_args[1]["extra"]["email"] == "admin@test.com"
            assert call_args[1]["extra"]["reason"] == "invalid_credentials"

    def test_deactivated_user_logs_warning(self, client, db):
        """Login to deactivated account produces warning log."""
        from ai_ready_rag.core.security import hash_password

        user = __import__("ai_ready_rag.db.models", fromlist=["User"]).User(
            email="deactivated@test.com",
            display_name="Deactivated",
            password_hash=hash_password("TestPassword123"),
            role="user",
            is_active=False,
        )
        db.add(user)
        db.flush()

        with patch("ai_ready_rag.api.auth.logger") as mock_logger:
            client.post(
                "/api/auth/login",
                json={"email": "deactivated@test.com", "password": "TestPassword123"},
            )
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert call_args[0][0] == "login_failed"
            assert call_args[1]["extra"]["reason"] == "account_deactivated"

    def test_successful_login_logs_info(self, client, admin_user):
        """Successful login produces info log."""
        with patch("ai_ready_rag.api.auth.logger") as mock_logger:
            response = client.post(
                "/api/auth/login",
                json={"email": "admin@test.com", "password": "AdminPassword123"},
            )
            assert response.status_code == 200
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert call_args[0][0] == "login_success"
            assert call_args[1]["extra"]["email"] == "admin@test.com"


class TestNoPrintStatements:
    """Verify no print() calls remain in production code."""

    def test_no_print_in_source(self):
        """No print() statements in ai_ready_rag source."""
        import pathlib

        source_dir = pathlib.Path(__file__).parent.parent / "ai_ready_rag"
        violations = []

        for py_file in source_dir.rglob("*.py"):
            content = py_file.read_text()
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                # Skip comments
                if stripped.startswith("#"):
                    continue
                if "print(" in stripped:
                    violations.append(f"{py_file.relative_to(source_dir.parent)}:{i}: {stripped}")

        assert violations == [], "Found print() statements:\n" + "\n".join(violations)
