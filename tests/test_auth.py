"""Tests for authentication endpoints."""


class TestSetup:
    """Setup wizard tests."""

    def test_setup_creates_first_admin(self, client):
        """Test setup endpoint creates first admin."""
        response = client.post(
            "/api/auth/setup",
            json={
                "email": "admin@example.com",
                "password": "SecurePassword123",
                "display_name": "First Admin",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "admin@example.com"
        assert data["role"] == "admin"
        assert data["display_name"] == "First Admin"

    def test_setup_fails_when_users_exist(self, client, admin_user):
        """Test setup fails if users already exist."""
        response = client.post(
            "/api/auth/setup",
            json={
                "email": "another@example.com",
                "password": "SecurePassword123",
                "display_name": "Another Admin",
            },
        )
        assert response.status_code == 400
        assert "already completed" in response.json()["detail"].lower()


class TestLogin:
    """Login endpoint tests."""

    def test_login_success(self, client, admin_user):
        """Test successful login."""
        response = client.post(
            "/api/auth/login", json={"email": "admin@test.com", "password": "AdminPassword123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == "admin@test.com"

    def test_login_wrong_password(self, client, admin_user):
        """Test login with wrong password."""
        response = client.post(
            "/api/auth/login", json={"email": "admin@test.com", "password": "WrongPassword123"}
        )
        assert response.status_code == 401
        assert "invalid" in response.json()["detail"].lower()

    def test_login_nonexistent_user(self, client):
        """Test login with nonexistent email."""
        response = client.post(
            "/api/auth/login", json={"email": "nobody@test.com", "password": "SomePassword123"}
        )
        assert response.status_code == 401

    def test_login_inactive_user(self, client, db, admin_user):
        """Test login with inactive user."""
        admin_user.is_active = False
        db.commit()

        response = client.post(
            "/api/auth/login", json={"email": "admin@test.com", "password": "AdminPassword123"}
        )
        assert response.status_code == 403
        assert "deactivated" in response.json()["detail"].lower()


class TestMe:
    """Current user endpoint tests."""

    def test_get_me_authenticated(self, client, admin_headers, admin_user):
        """Test getting current user when authenticated."""
        response = client.get("/api/auth/me", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == admin_user.email
        assert data["role"] == "admin"

    def test_get_me_unauthenticated(self, client):
        """Test getting current user without auth."""
        response = client.get("/api/auth/me")
        assert response.status_code == 401

    def test_get_me_invalid_token(self, client):
        """Test getting current user with invalid token."""
        response = client.get("/api/auth/me", headers={"Authorization": "Bearer invalid-token"})
        assert response.status_code == 401


class TestLogout:
    """Logout endpoint tests."""

    def test_logout_success(self, client, admin_headers):
        """Test successful logout."""
        response = client.post("/api/auth/logout", headers=admin_headers)
        assert response.status_code == 200
        assert "logged out" in response.json()["message"].lower()

    def test_logout_unauthenticated(self, client):
        """Test logout without auth."""
        response = client.post("/api/auth/logout")
        assert response.status_code == 401
