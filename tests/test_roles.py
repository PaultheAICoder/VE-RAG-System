"""Tests for role-based access control.

Tests the three-role permission system:
- system_admin: Full access including Settings/Health
- customer_admin: Can manage users/documents/tags but NOT system settings
- user: Chat and filtered document access only
"""


class TestRoleNormalization:
    """Test role normalization for backward compatibility."""

    def test_legacy_admin_treated_as_system_admin(self, client, admin_headers):
        """Test that legacy 'admin' role has system admin access."""
        # Legacy admin should be able to access system-admin-only endpoints
        response = client.get("/api/admin/processing-options", headers=admin_headers)
        assert response.status_code == 200

    def test_legacy_admin_can_manage_users(self, client, admin_headers):
        """Test that legacy 'admin' role can manage users."""
        response = client.get("/api/users/", headers=admin_headers)
        assert response.status_code == 200


class TestCustomerAdminAccess:
    """Test customer_admin role permissions."""

    def test_customer_admin_can_list_users(self, client, customer_admin_headers):
        """Customer admin can list users."""
        response = client.get("/api/users/", headers=customer_admin_headers)
        assert response.status_code == 200

    def test_customer_admin_can_create_user(self, client, customer_admin_headers):
        """Customer admin can create users."""
        response = client.post(
            "/api/users/",
            headers=customer_admin_headers,
            json={
                "email": "newuser_by_customer_admin@test.com",
                "display_name": "New User",
                "password": "NewUserPassword123",
                "role": "user",
            },
        )
        assert response.status_code == 201

    def test_customer_admin_can_list_tags(self, client, customer_admin_headers):
        """Customer admin can list tags."""
        response = client.get("/api/tags/", headers=customer_admin_headers)
        assert response.status_code == 200

    def test_customer_admin_can_create_tag(self, client, customer_admin_headers):
        """Customer admin can create tags."""
        response = client.post(
            "/api/tags/",
            headers=customer_admin_headers,
            json={
                "name": "customer_admin_tag",
                "display_name": "Customer Admin Tag",
            },
        )
        assert response.status_code == 201

    def test_customer_admin_cannot_access_processing_options(self, client, customer_admin_headers):
        """Customer admin cannot access processing options (system admin only)."""
        response = client.get("/api/admin/processing-options", headers=customer_admin_headers)
        assert response.status_code == 403
        assert "system admin" in response.json()["detail"].lower()

    def test_customer_admin_cannot_access_architecture(self, client, customer_admin_headers):
        """Customer admin cannot access architecture info (system admin only)."""
        response = client.get("/api/admin/architecture", headers=customer_admin_headers)
        assert response.status_code == 403
        assert "system admin" in response.json()["detail"].lower()

    def test_customer_admin_cannot_access_models(self, client, customer_admin_headers):
        """Customer admin cannot access model configuration (system admin only)."""
        response = client.get("/api/admin/models", headers=customer_admin_headers)
        assert response.status_code == 403
        assert "system admin" in response.json()["detail"].lower()

    def test_customer_admin_cannot_access_knowledge_base_stats(
        self, client, customer_admin_headers
    ):
        """Customer admin cannot access knowledge base stats (system admin only)."""
        response = client.get("/api/admin/knowledge-base/stats", headers=customer_admin_headers)
        assert response.status_code == 403
        assert "system admin" in response.json()["detail"].lower()


class TestSystemAdminAccess:
    """Test system_admin role permissions."""

    def test_system_admin_can_access_processing_options(self, client, system_admin_headers):
        """System admin can access processing options."""
        response = client.get("/api/admin/processing-options", headers=system_admin_headers)
        assert response.status_code == 200

    def test_system_admin_can_list_users(self, client, system_admin_headers):
        """System admin can list users."""
        response = client.get("/api/users/", headers=system_admin_headers)
        assert response.status_code == 200

    def test_system_admin_can_list_tags(self, client, system_admin_headers):
        """System admin can list tags."""
        response = client.get("/api/tags/", headers=system_admin_headers)
        assert response.status_code == 200


class TestRegularUserAccess:
    """Test regular user role permissions."""

    def test_user_cannot_list_users(self, client, user_headers):
        """Regular user cannot list users."""
        response = client.get("/api/users/", headers=user_headers)
        assert response.status_code == 403

    def test_user_cannot_create_tag(self, client, user_headers):
        """Regular user cannot create tags."""
        response = client.post(
            "/api/tags/",
            headers=user_headers,
            json={"name": "user_tag", "display_name": "User Tag"},
        )
        assert response.status_code == 403

    def test_user_cannot_access_processing_options(self, client, user_headers):
        """Regular user cannot access processing options."""
        response = client.get("/api/admin/processing-options", headers=user_headers)
        assert response.status_code == 403

    def test_user_can_list_tags(self, client, user_headers):
        """Regular user can list tags (read-only access)."""
        response = client.get("/api/tags/", headers=user_headers)
        assert response.status_code == 200


class TestRoleConstants:
    """Test role constant definitions."""

    def test_role_constants_defined(self):
        """Verify role constants are properly defined."""
        from ai_ready_rag.core.dependencies import (
            ROLE_CUSTOMER_ADMIN,
            ROLE_SYSTEM_ADMIN,
            ROLE_USER,
        )

        assert ROLE_SYSTEM_ADMIN == "system_admin"
        assert ROLE_CUSTOMER_ADMIN == "customer_admin"
        assert ROLE_USER == "user"

    def test_normalize_role_maps_admin(self):
        """Test that normalize_role maps 'admin' to 'system_admin'."""
        from ai_ready_rag.core.dependencies import normalize_role

        assert normalize_role("admin") == "system_admin"
        assert normalize_role("system_admin") == "system_admin"
        assert normalize_role("customer_admin") == "customer_admin"
        assert normalize_role("user") == "user"
