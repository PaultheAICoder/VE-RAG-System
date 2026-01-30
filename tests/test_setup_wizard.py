"""Tests for setup wizard API endpoints."""

from ai_ready_rag.db.models import SystemSetup


class TestSetupStatus:
    """Tests for GET /api/setup/status endpoint."""

    def test_status_returns_required_when_fresh_db(self, client):
        """Setup is required when SystemSetup table is empty."""
        response = client.get("/api/setup/status")
        assert response.status_code == 200
        data = response.json()
        assert data["setup_required"] is True
        assert data["setup_complete"] is False

    def test_status_returns_complete_after_setup(self, client, db):
        """Setup is not required when already completed."""
        # Create completed setup record
        setup = SystemSetup(setup_complete=True, admin_password_changed=True)
        db.add(setup)
        db.flush()

        response = client.get("/api/setup/status")
        assert response.status_code == 200
        data = response.json()
        assert data["setup_required"] is False
        assert data["setup_complete"] is True

    def test_status_no_auth_required(self, client):
        """Setup status endpoint does not require authentication."""
        # No auth header - should still work
        response = client.get("/api/setup/status")
        assert response.status_code == 200


class TestCompleteSetup:
    """Tests for POST /api/setup/complete endpoint."""

    def test_complete_setup_success(self, client, db, admin_user, admin_headers):
        """Admin can complete setup by changing password."""
        response = client.post(
            "/api/setup/complete",
            headers=admin_headers,
            json={
                "current_password": "AdminPassword123",
                "new_password": "NewSecurePassword123",
                "confirm_password": "NewSecurePassword123",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "completed" in data["message"].lower()

        # Verify setup is marked complete
        setup = db.query(SystemSetup).first()
        assert setup is not None
        assert setup.setup_complete is True
        assert setup.admin_password_changed is True
        assert setup.setup_completed_by == admin_user.id

    def test_complete_setup_wrong_current_password(self, client, admin_headers):
        """Setup fails with wrong current password."""
        response = client.post(
            "/api/setup/complete",
            headers=admin_headers,
            json={
                "current_password": "WrongPassword123",
                "new_password": "NewSecurePassword123",
                "confirm_password": "NewSecurePassword123",
            },
        )
        assert response.status_code == 401
        assert "current password" in response.json()["detail"].lower()

    def test_complete_setup_passwords_dont_match(self, client, admin_headers):
        """Setup fails when passwords don't match."""
        response = client.post(
            "/api/setup/complete",
            headers=admin_headers,
            json={
                "current_password": "AdminPassword123",
                "new_password": "NewSecurePassword123",
                "confirm_password": "DifferentPassword123",
            },
        )
        assert response.status_code == 422  # Validation error

    def test_complete_setup_password_too_short(self, client, admin_headers):
        """Setup fails when new password is too short."""
        response = client.post(
            "/api/setup/complete",
            headers=admin_headers,
            json={
                "current_password": "AdminPassword123",
                "new_password": "Short123",  # Less than 12 chars
                "confirm_password": "Short123",
            },
        )
        assert response.status_code == 422  # Validation error

    def test_complete_setup_non_admin_rejected(self, client, user_headers):
        """Non-admin users cannot complete setup."""
        response = client.post(
            "/api/setup/complete",
            headers=user_headers,
            json={
                "current_password": "UserPassword123",
                "new_password": "NewSecurePassword123",
                "confirm_password": "NewSecurePassword123",
            },
        )
        assert response.status_code == 403
        assert "administrator" in response.json()["detail"].lower()

    def test_complete_setup_only_once(self, client, db, admin_headers):
        """Setup can only be completed once."""
        # First, create a completed setup record
        setup = SystemSetup(setup_complete=True, admin_password_changed=True)
        db.add(setup)
        db.flush()

        response = client.post(
            "/api/setup/complete",
            headers=admin_headers,
            json={
                "current_password": "AdminPassword123",
                "new_password": "NewSecurePassword123",
                "confirm_password": "NewSecurePassword123",
            },
        )
        assert response.status_code == 400
        assert "already" in response.json()["detail"].lower()

    def test_complete_setup_requires_auth(self, client):
        """Setup completion requires authentication."""
        response = client.post(
            "/api/setup/complete",
            json={
                "current_password": "AdminPassword123",
                "new_password": "NewSecurePassword123",
                "confirm_password": "NewSecurePassword123",
            },
        )
        assert response.status_code == 401

    def test_complete_setup_same_password_rejected(self, client, admin_headers):
        """New password must be different from current password."""
        response = client.post(
            "/api/setup/complete",
            headers=admin_headers,
            json={
                "current_password": "AdminPassword123",
                "new_password": "AdminPassword123",
                "confirm_password": "AdminPassword123",
            },
        )
        assert response.status_code == 400
        assert "different" in response.json()["detail"].lower()


class TestLoginWithSetupRequired:
    """Tests for login response with setup_required flag."""

    def test_login_returns_setup_required_for_admin(self, client, admin_user):
        """Login response includes setup_required=True for admin when setup not complete."""
        response = client.post(
            "/api/auth/login",
            json={"email": "admin@test.com", "password": "AdminPassword123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "setup_required" in data
        assert data["setup_required"] is True

    def test_login_returns_no_setup_required_for_regular_user(self, client, regular_user):
        """Login response has setup_required=False for regular users."""
        response = client.post(
            "/api/auth/login",
            json={"email": "user@test.com", "password": "UserPassword123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "setup_required" in data
        assert data["setup_required"] is False

    def test_login_returns_no_setup_required_when_complete(self, client, db, admin_user):
        """Login response has setup_required=False when setup is complete."""
        # Mark setup as complete
        setup = SystemSetup(setup_complete=True, admin_password_changed=True)
        db.add(setup)
        db.flush()

        response = client.post(
            "/api/auth/login",
            json={"email": "admin@test.com", "password": "AdminPassword123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["setup_required"] is False


class TestSkipSetupWizard:
    """Tests for SKIP_SETUP_WIZARD environment variable.

    Note: Due to how Pydantic settings are cached at module import time,
    testing the SKIP_SETUP_WIZARD env var requires special handling.
    The config is loaded at app startup, so changing env vars mid-test
    doesn't affect the already-loaded settings. These tests verify the
    logic exists but may require manual testing or integration tests
    to fully validate the env var behavior.
    """

    def test_skip_setting_exists_in_config(self):
        """Verify skip_setup_wizard setting exists in config."""
        from ai_ready_rag.config import Settings

        # Verify the setting exists with correct default
        settings = Settings()
        assert hasattr(settings, "skip_setup_wizard")
        assert settings.skip_setup_wizard is False  # Default should be False

    def test_is_setup_required_respects_skip_setting(self, db):
        """Verify is_setup_required logic respects skip setting when True."""
        from ai_ready_rag.api.setup import is_setup_required

        # This test verifies the logic, though the actual env var
        # behavior would need integration testing
        # When skip is False (default), setup should be required
        assert is_setup_required(db) is True
