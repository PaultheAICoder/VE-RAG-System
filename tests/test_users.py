"""Tests for user management endpoints."""

from ai_ready_rag.db.models import Tag


class TestListUsers:
    """List users endpoint tests."""

    def test_list_users_as_admin(self, client, admin_headers, admin_user):
        """Test listing users as admin."""
        response = client.get("/api/users", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert any(u["email"] == admin_user.email for u in data)

    def test_list_users_as_regular_user(self, client, user_headers):
        """Test listing users as regular user (should fail)."""
        response = client.get("/api/users", headers=user_headers)
        assert response.status_code == 403

    def test_list_users_unauthenticated(self, client):
        """Test listing users without auth."""
        response = client.get("/api/users")
        assert response.status_code == 401


class TestCreateUser:
    """Create user endpoint tests."""

    def test_create_user_as_admin(self, client, admin_headers):
        """Test creating user as admin."""
        response = client.post(
            "/api/users",
            headers=admin_headers,
            json={
                "email": "newuser@test.com",
                "display_name": "New User",
                "password": "NewUserPassword123",
                "role": "user",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@test.com"
        assert data["role"] == "user"

    def test_create_admin_as_admin(self, client, admin_headers):
        """Test creating another admin."""
        response = client.post(
            "/api/users",
            headers=admin_headers,
            json={
                "email": "admin2@test.com",
                "display_name": "Second Admin",
                "password": "AdminPassword123",
                "role": "admin",
            },
        )
        assert response.status_code == 201
        assert response.json()["role"] == "admin"

    def test_create_user_duplicate_email(self, client, admin_headers, admin_user):
        """Test creating user with duplicate email."""
        response = client.post(
            "/api/users",
            headers=admin_headers,
            json={
                "email": admin_user.email,
                "display_name": "Duplicate",
                "password": "SomePassword123",
            },
        )
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()

    def test_create_user_as_regular_user(self, client, user_headers):
        """Test creating user as regular user (should fail)."""
        response = client.post(
            "/api/users",
            headers=user_headers,
            json={
                "email": "another@test.com",
                "display_name": "Another",
                "password": "Password123456",
            },
        )
        assert response.status_code == 403


class TestGetUser:
    """Get user endpoint tests."""

    def test_get_user_as_admin(self, client, admin_headers, regular_user):
        """Test getting user by ID as admin."""
        response = client.get(f"/api/users/{regular_user.id}", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == regular_user.email

    def test_get_nonexistent_user(self, client, admin_headers):
        """Test getting nonexistent user."""
        response = client.get("/api/users/nonexistent-id", headers=admin_headers)
        assert response.status_code == 404


class TestUpdateUser:
    """Update user endpoint tests."""

    def test_update_user_as_admin(self, client, admin_headers, regular_user):
        """Test updating user as admin."""
        response = client.put(
            f"/api/users/{regular_user.id}",
            headers=admin_headers,
            json={"display_name": "Updated Name"},
        )
        assert response.status_code == 200
        assert response.json()["display_name"] == "Updated Name"

    def test_update_user_role(self, client, admin_headers, regular_user):
        """Test promoting user to admin."""
        response = client.put(
            f"/api/users/{regular_user.id}", headers=admin_headers, json={"role": "admin"}
        )
        assert response.status_code == 200
        assert response.json()["role"] == "admin"


class TestDeactivateUser:
    """Deactivate user endpoint tests."""

    def test_deactivate_user(self, client, admin_headers, regular_user):
        """Test deactivating a user."""
        response = client.delete(f"/api/users/{regular_user.id}", headers=admin_headers)
        assert response.status_code == 200
        assert "deactivated" in response.json()["message"].lower()

    def test_deactivate_self(self, client, admin_headers, admin_user):
        """Test admin cannot deactivate themselves."""
        response = client.delete(f"/api/users/{admin_user.id}", headers=admin_headers)
        assert response.status_code == 400
        assert "yourself" in response.json()["detail"].lower()


class TestResetPassword:
    """Password reset endpoint tests."""

    def test_reset_password(self, client, admin_headers, regular_user):
        """Test resetting user password."""
        response = client.post(
            f"/api/users/{regular_user.id}/reset-password", headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "temporary_password" in data
        assert len(data["temporary_password"]) > 0


class TestAssignTags:
    """Tag assignment endpoint tests."""

    def test_assign_tags_to_user(self, client, admin_headers, regular_user, sample_tag):
        """Test assigning tags to user."""
        response = client.post(
            f"/api/users/{regular_user.id}/tags",
            headers=admin_headers,
            json={"tag_ids": [sample_tag.id]},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["tags"]) == 1
        assert data["tags"][0]["id"] == sample_tag.id


class TestTagAccessEnabled:
    """Tests for tag_access_enabled toggle."""

    def test_update_tag_access_enabled_to_false(self, client, admin_headers, regular_user):
        """Admin can set tag_access_enabled=False on a user."""
        response = client.put(
            f"/api/users/{regular_user.id}",
            headers=admin_headers,
            json={"tag_access_enabled": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tag_access_enabled"] is False

    def test_update_tag_access_enabled_to_true(self, client, admin_headers, regular_user):
        """Admin can set tag_access_enabled back to True."""
        # First set to False
        client.put(
            f"/api/users/{regular_user.id}",
            headers=admin_headers,
            json={"tag_access_enabled": False},
        )
        # Then set back to True
        response = client.put(
            f"/api/users/{regular_user.id}",
            headers=admin_headers,
            json={"tag_access_enabled": True},
        )
        assert response.status_code == 200
        assert response.json()["tag_access_enabled"] is True

    def test_tag_access_enabled_default_true(self, client, admin_headers):
        """New user has tag_access_enabled=True by default."""
        response = client.post(
            "/api/users",
            headers=admin_headers,
            json={
                "email": "default_access@test.com",
                "display_name": "Default Access User",
                "password": "Password123",
            },
        )
        assert response.status_code == 201
        assert response.json()["tag_access_enabled"] is True

    def test_tag_access_enabled_in_user_response(self, client, admin_headers, regular_user):
        """GET /api/users/{id} includes tag_access_enabled."""
        response = client.get(f"/api/users/{regular_user.id}", headers=admin_headers)
        assert response.status_code == 200
        assert "tag_access_enabled" in response.json()

    def test_tag_access_enabled_in_list_response(self, client, admin_headers, admin_user):
        """GET /api/users list includes tag_access_enabled for each user."""
        response = client.get("/api/users", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        for user in data:
            assert "tag_access_enabled" in user


class TestBulkAutoTagAssignment:
    """Tests for POST /api/users/{user_id}/tags/auto endpoint."""

    def test_bulk_auto_assign_client_tags(self, client, admin_headers, regular_user, db):
        """Assign client tags by name."""
        tag = Tag(
            name="client:acme",
            display_name="Acme",
            description="Test client tag",
            created_by=regular_user.id,
        )
        db.add(tag)
        db.flush()

        response = client.post(
            f"/api/users/{regular_user.id}/tags/auto",
            headers=admin_headers,
            json={"client_names": ["acme"], "include_doctypes": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["matched"] == 1
        assert any(t["name"] == "client:acme" for t in data["tags"])

    def test_bulk_auto_assign_with_doctypes(self, client, admin_headers, regular_user, db):
        """include_doctypes=True adds all doctype: tags."""
        client_tag = Tag(
            name="client:beta",
            display_name="Beta",
            created_by=regular_user.id,
        )
        doctype_tag = Tag(
            name="doctype:policy",
            display_name="Policy",
            created_by=regular_user.id,
        )
        db.add(client_tag)
        db.add(doctype_tag)
        db.flush()

        response = client.post(
            f"/api/users/{regular_user.id}/tags/auto",
            headers=admin_headers,
            json={"client_names": ["beta"], "include_doctypes": True},
        )
        assert response.status_code == 200
        data = response.json()
        tag_names = [t["name"] for t in data["tags"]]
        assert "client:beta" in tag_names
        assert "doctype:policy" in tag_names

    def test_bulk_auto_assign_merges_existing(
        self, client, admin_headers, regular_user, sample_tag, db
    ):
        """Bulk auto-assign merges with existing tags, does not replace."""
        # Assign sample_tag first
        regular_user.tags.append(sample_tag)
        db.flush()

        new_tag = Tag(
            name="client:gamma",
            display_name="Gamma",
            created_by=regular_user.id,
        )
        db.add(new_tag)
        db.flush()

        response = client.post(
            f"/api/users/{regular_user.id}/tags/auto",
            headers=admin_headers,
            json={"client_names": ["gamma"], "include_doctypes": False},
        )
        assert response.status_code == 200
        data = response.json()
        tag_names = [t["name"] for t in data["tags"]]
        assert sample_tag.name in tag_names
        assert "client:gamma" in tag_names

    def test_bulk_auto_assign_nonexistent_user(self, client, admin_headers):
        """Returns 404 for nonexistent user."""
        response = client.post(
            "/api/users/nonexistent-id/tags/auto",
            headers=admin_headers,
            json={"client_names": ["acme"]},
        )
        assert response.status_code == 404

    def test_bulk_auto_assign_no_matching_tags(self, client, admin_headers, regular_user):
        """Request client names with no matching tags returns matched=0."""
        response = client.post(
            f"/api/users/{regular_user.id}/tags/auto",
            headers=admin_headers,
            json={"client_names": ["nonexistent"], "include_doctypes": False},
        )
        assert response.status_code == 200
        assert response.json()["matched"] == 0

    def test_bulk_auto_assign_requires_admin(self, client, user_headers, regular_user):
        """Regular user cannot access bulk auto-tag endpoint."""
        response = client.post(
            f"/api/users/{regular_user.id}/tags/auto",
            headers=user_headers,
            json={"client_names": ["acme"]},
        )
        assert response.status_code == 403
