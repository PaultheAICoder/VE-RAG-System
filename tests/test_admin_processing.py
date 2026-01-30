"""Tests for admin processing options endpoints."""


class TestGetProcessingOptions:
    """Tests for GET /api/admin/processing-options."""

    def test_get_returns_defaults(self, client, admin_headers):
        """GET returns default values when nothing stored."""
        response = client.get("/api/admin/processing-options", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["enable_ocr"] is True
        assert data["force_full_page_ocr"] is False
        assert data["ocr_language"] == "eng"
        assert data["table_extraction_mode"] == "accurate"
        assert data["include_image_descriptions"] is True

    def test_get_requires_admin(self, client, user_headers):
        """GET requires admin role."""
        response = client.get("/api/admin/processing-options", headers=user_headers)
        assert response.status_code == 403

    def test_get_requires_auth(self, client):
        """GET requires authentication."""
        response = client.get("/api/admin/processing-options")
        assert response.status_code == 401


class TestUpdateProcessingOptions:
    """Tests for PATCH /api/admin/processing-options."""

    def test_update_single_field(self, client, admin_headers):
        """PATCH updates single field."""
        response = client.patch(
            "/api/admin/processing-options",
            headers=admin_headers,
            json={"force_full_page_ocr": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["force_full_page_ocr"] is True
        # Other fields unchanged
        assert data["enable_ocr"] is True
        assert data["table_extraction_mode"] == "accurate"

    def test_update_multiple_fields(self, client, admin_headers):
        """PATCH updates multiple fields."""
        response = client.patch(
            "/api/admin/processing-options",
            headers=admin_headers,
            json={
                "table_extraction_mode": "fast",
                "ocr_language": "fra",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["table_extraction_mode"] == "fast"
        assert data["ocr_language"] == "fra"

    def test_update_persists(self, client, admin_headers):
        """PATCH values persist across requests."""
        # Update
        client.patch(
            "/api/admin/processing-options",
            headers=admin_headers,
            json={"include_image_descriptions": False},
        )

        # Verify with GET
        response = client.get("/api/admin/processing-options", headers=admin_headers)
        assert response.json()["include_image_descriptions"] is False

    def test_update_requires_admin(self, client, user_headers):
        """PATCH requires admin role."""
        response = client.patch(
            "/api/admin/processing-options",
            headers=user_headers,
            json={"enable_ocr": False},
        )
        assert response.status_code == 403

    def test_update_requires_auth(self, client):
        """PATCH requires authentication."""
        response = client.patch(
            "/api/admin/processing-options",
            json={"enable_ocr": False},
        )
        assert response.status_code == 401

    def test_update_validates_table_mode(self, client, admin_headers):
        """PATCH validates table_extraction_mode enum."""
        response = client.patch(
            "/api/admin/processing-options",
            headers=admin_headers,
            json={"table_extraction_mode": "invalid"},
        )
        assert response.status_code == 422  # Validation error


class TestQueryRoutingMode:
    """Tests for query_routing_mode setting (Issue #23)."""

    def test_get_returns_default_retrieve_only(self, client, admin_headers):
        """GET returns default query_routing_mode as 'retrieve_only'."""
        response = client.get("/api/admin/processing-options", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["query_routing_mode"] == "retrieve_only"

    def test_update_to_retrieve_and_direct(self, client, admin_headers):
        """PATCH updates query_routing_mode to 'retrieve_and_direct'."""
        response = client.patch(
            "/api/admin/processing-options",
            headers=admin_headers,
            json={"query_routing_mode": "retrieve_and_direct"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query_routing_mode"] == "retrieve_and_direct"

    def test_update_persists(self, client, admin_headers):
        """PATCH value persists across requests."""
        # Update
        client.patch(
            "/api/admin/processing-options",
            headers=admin_headers,
            json={"query_routing_mode": "retrieve_and_direct"},
        )

        # Verify with GET
        response = client.get("/api/admin/processing-options", headers=admin_headers)
        assert response.json()["query_routing_mode"] == "retrieve_and_direct"

    def test_update_validates_invalid_mode(self, client, admin_headers):
        """PATCH validates query_routing_mode enum."""
        response = client.patch(
            "/api/admin/processing-options",
            headers=admin_headers,
            json={"query_routing_mode": "invalid_mode"},
        )
        assert response.status_code == 422  # Validation error

    def test_update_back_to_retrieve_only(self, client, admin_headers):
        """Can update back to 'retrieve_only' after changing."""
        # First set to retrieve_and_direct
        client.patch(
            "/api/admin/processing-options",
            headers=admin_headers,
            json={"query_routing_mode": "retrieve_and_direct"},
        )

        # Then change back to retrieve_only
        response = client.patch(
            "/api/admin/processing-options",
            headers=admin_headers,
            json={"query_routing_mode": "retrieve_only"},
        )

        assert response.status_code == 200
        assert response.json()["query_routing_mode"] == "retrieve_only"
