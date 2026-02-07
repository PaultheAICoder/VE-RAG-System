"""Tests for security settings and feature flag endpoints."""


class TestGetSecuritySettings:
    """Tests for GET /api/admin/settings/security."""

    def test_returns_default_settings(self, client, admin_headers):
        """GET returns default security settings."""
        response = client.get("/api/admin/settings/security", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert "jwt_expiration_hours" in data
        assert "password_min_length" in data
        assert "bcrypt_rounds" in data
        assert data["jwt_expiration_hours"] == 24
        assert data["password_min_length"] == 12

    def test_requires_admin(self, client, user_headers):
        """GET requires admin role."""
        response = client.get("/api/admin/settings/security", headers=user_headers)
        assert response.status_code == 403

    def test_requires_auth(self, client):
        """GET requires authentication."""
        response = client.get("/api/admin/settings/security")
        assert response.status_code == 401


class TestUpdateSecuritySettings:
    """Tests for PUT /api/admin/settings/security."""

    def test_updates_jwt_expiration(self, client, admin_headers):
        """PUT updates jwt_expiration_hours."""
        response = client.put(
            "/api/admin/settings/security",
            headers=admin_headers,
            json={"jwt_expiration_hours": 48},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jwt_expiration_hours"] == 48

    def test_updates_password_min_length(self, client, admin_headers):
        """PUT updates password_min_length."""
        response = client.put(
            "/api/admin/settings/security",
            headers=admin_headers,
            json={"password_min_length": 16},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["password_min_length"] == 16

    def test_updates_bcrypt_rounds(self, client, admin_headers):
        """PUT updates bcrypt_rounds."""
        response = client.put(
            "/api/admin/settings/security",
            headers=admin_headers,
            json={"bcrypt_rounds": 14},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["bcrypt_rounds"] == 14

    def test_rejects_jwt_too_low(self, client, admin_headers):
        """PUT rejects jwt_expiration_hours < 1."""
        response = client.put(
            "/api/admin/settings/security",
            headers=admin_headers,
            json={"jwt_expiration_hours": 0},
        )
        assert response.status_code == 400

    def test_rejects_jwt_too_high(self, client, admin_headers):
        """PUT rejects jwt_expiration_hours > 720."""
        response = client.put(
            "/api/admin/settings/security",
            headers=admin_headers,
            json={"jwt_expiration_hours": 721},
        )
        assert response.status_code == 400

    def test_rejects_password_length_too_low(self, client, admin_headers):
        """PUT rejects password_min_length < 8."""
        response = client.put(
            "/api/admin/settings/security",
            headers=admin_headers,
            json={"password_min_length": 5},
        )
        assert response.status_code == 400

    def test_rejects_bcrypt_rounds_too_low(self, client, admin_headers):
        """PUT rejects bcrypt_rounds < 4."""
        response = client.put(
            "/api/admin/settings/security",
            headers=admin_headers,
            json={"bcrypt_rounds": 2},
        )
        assert response.status_code == 400

    def test_rejects_bcrypt_rounds_too_high(self, client, admin_headers):
        """PUT rejects bcrypt_rounds > 31."""
        response = client.put(
            "/api/admin/settings/security",
            headers=admin_headers,
            json={"bcrypt_rounds": 32},
        )
        assert response.status_code == 400

    def test_partial_update(self, client, admin_headers):
        """PUT updates only provided fields."""
        # Update just one field
        response = client.put(
            "/api/admin/settings/security",
            headers=admin_headers,
            json={"jwt_expiration_hours": 12},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jwt_expiration_hours"] == 12
        # Other fields should still be at defaults
        assert data["password_min_length"] == 12

    def test_requires_admin(self, client, user_headers):
        """PUT requires admin role."""
        response = client.put(
            "/api/admin/settings/security",
            headers=user_headers,
            json={"jwt_expiration_hours": 48},
        )
        assert response.status_code == 403


class TestGetFeatureFlags:
    """Tests for GET /api/admin/settings/feature-flags."""

    def test_returns_default_flags(self, client, admin_headers):
        """GET returns default feature flags."""
        response = client.get("/api/admin/settings/feature-flags", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert "enable_rag" in data
        assert "skip_setup_wizard" in data
        assert data["enable_rag"] is True
        assert data["skip_setup_wizard"] is False

    def test_requires_admin(self, client, user_headers):
        """GET requires admin role."""
        response = client.get("/api/admin/settings/feature-flags", headers=user_headers)
        assert response.status_code == 403

    def test_requires_auth(self, client):
        """GET requires authentication."""
        response = client.get("/api/admin/settings/feature-flags")
        assert response.status_code == 401


class TestUpdateFeatureFlags:
    """Tests for PUT /api/admin/settings/feature-flags."""

    def test_updates_enable_rag(self, client, admin_headers):
        """PUT updates enable_rag."""
        response = client.put(
            "/api/admin/settings/feature-flags",
            headers=admin_headers,
            json={"enable_rag": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["enable_rag"] is False

    def test_updates_skip_setup_wizard(self, client, admin_headers):
        """PUT updates skip_setup_wizard."""
        response = client.put(
            "/api/admin/settings/feature-flags",
            headers=admin_headers,
            json={"skip_setup_wizard": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["skip_setup_wizard"] is True

    def test_partial_update(self, client, admin_headers):
        """PUT updates only provided fields."""
        response = client.put(
            "/api/admin/settings/feature-flags",
            headers=admin_headers,
            json={"enable_rag": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["enable_rag"] is False

    def test_requires_admin(self, client, user_headers):
        """PUT requires admin role."""
        response = client.put(
            "/api/admin/settings/feature-flags",
            headers=user_headers,
            json={"enable_rag": False},
        )
        assert response.status_code == 403
