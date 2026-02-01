"""Tests for tag management endpoints."""


class TestListTags:
    """List tags endpoint tests."""

    def test_list_tags_authenticated(self, client, user_headers, sample_tag):
        """Test listing tags as authenticated user."""
        response = client.get("/api/tags", headers=user_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_list_tags_unauthenticated(self, client):
        """Test listing tags without auth."""
        response = client.get("/api/tags")
        assert response.status_code == 401


class TestCreateTag:
    """Create tag endpoint tests."""

    def test_create_tag_as_admin(self, client, admin_headers):
        """Test creating tag as admin."""
        response = client.post(
            "/api/tags",
            headers=admin_headers,
            json={
                "name": "finance",
                "display_name": "Finance",
                "description": "Financial documents",
                "color": "#3B82F6",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "finance"
        assert data["display_name"] == "Finance"
        assert data["color"] == "#3B82F6"

    def test_create_tag_duplicate_name(self, client, admin_headers, sample_tag):
        """Test creating tag with duplicate name."""
        response = client.post(
            "/api/tags",
            headers=admin_headers,
            json={"name": sample_tag.name, "display_name": "Duplicate HR"},
        )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"].lower()

    def test_create_tag_as_regular_user(self, client, user_headers):
        """Test creating tag as regular user (should fail)."""
        response = client.post(
            "/api/tags", headers=user_headers, json={"name": "legal", "display_name": "Legal"}
        )
        assert response.status_code == 403

    def test_create_tag_minimal(self, client, admin_headers):
        """Test creating tag with minimal fields."""
        response = client.post(
            "/api/tags", headers=admin_headers, json={"name": "it", "display_name": "IT Support"}
        )
        assert response.status_code == 201
        data = response.json()
        assert data["color"] == "#6B7280"  # default color


class TestGetTag:
    """Get tag endpoint tests."""

    def test_get_tag(self, client, user_headers, sample_tag):
        """Test getting tag by ID."""
        response = client.get(f"/api/tags/{sample_tag.id}", headers=user_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == sample_tag.name

    def test_get_nonexistent_tag(self, client, user_headers):
        """Test getting nonexistent tag."""
        response = client.get("/api/tags/nonexistent-id", headers=user_headers)
        assert response.status_code == 404


class TestUpdateTag:
    """Update tag endpoint tests."""

    def test_update_tag_as_admin(self, client, admin_headers, sample_tag):
        """Test updating tag as admin."""
        response = client.put(
            f"/api/tags/{sample_tag.id}",
            headers=admin_headers,
            json={"display_name": "HR Department", "color": "#EF4444"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] == "HR Department"
        assert data["color"] == "#EF4444"

    def test_update_tag_as_regular_user(self, client, user_headers, sample_tag):
        """Test updating tag as regular user (should fail)."""
        response = client.put(
            f"/api/tags/{sample_tag.id}", headers=user_headers, json={"display_name": "New Name"}
        )
        assert response.status_code == 403

    def test_update_system_tag(self, client, admin_headers, db):
        """Test updating system tag (should fail)."""
        from ai_ready_rag.db.models import Tag

        system_tag = Tag(name="public", display_name="Public", is_system=True)
        db.add(system_tag)
        db.commit()

        response = client.put(
            f"/api/tags/{system_tag.id}", headers=admin_headers, json={"display_name": "Not Public"}
        )
        assert response.status_code == 400
        assert "system" in response.json()["detail"].lower()


class TestDeleteTag:
    """Delete tag endpoint tests."""

    def test_delete_tag_as_admin(self, client, admin_headers, sample_tag):
        """Test deleting tag as admin."""
        response = client.delete(f"/api/tags/{sample_tag.id}", headers=admin_headers)
        assert response.status_code == 200
        assert "deleted" in response.json()["message"].lower()

    def test_delete_tag_as_regular_user(self, client, user_headers, sample_tag):
        """Test deleting tag as regular user (should fail)."""
        response = client.delete(f"/api/tags/{sample_tag.id}", headers=user_headers)
        assert response.status_code == 403

    def test_delete_nonexistent_tag(self, client, admin_headers):
        """Test deleting nonexistent tag."""
        response = client.delete("/api/tags/nonexistent-id", headers=admin_headers)
        assert response.status_code == 404

    def test_delete_system_tag(self, client, admin_headers, db):
        """Test deleting system tag (should fail)."""
        from ai_ready_rag.db.models import Tag

        system_tag = Tag(name="admin", display_name="Admin", is_system=True)
        db.add(system_tag)
        db.commit()

        response = client.delete(f"/api/tags/{system_tag.id}", headers=admin_headers)
        assert response.status_code == 400
        assert "system" in response.json()["detail"].lower()
