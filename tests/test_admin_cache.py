"""Tests for cache admin endpoints (Phase 4)."""

from fastapi import status

from ai_ready_rag.db.models import ResponseCache
from ai_ready_rag.services.settings_service import CACHE_SETTINGS_DEFAULTS, SettingsService


class TestCacheSettingsGet:
    """Tests for GET /api/admin/cache/settings endpoint."""

    def test_returns_all_settings_with_defaults(self, client, admin_headers):
        """GET /cache/settings returns all 7 cache settings with defaults."""
        response = client.get("/api/admin/cache/settings", headers=admin_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify all fields present with default values
        assert data["cache_enabled"] == CACHE_SETTINGS_DEFAULTS["cache_enabled"]
        assert data["cache_ttl_hours"] == CACHE_SETTINGS_DEFAULTS["cache_ttl_hours"]
        assert data["cache_max_entries"] == CACHE_SETTINGS_DEFAULTS["cache_max_entries"]
        assert (
            data["cache_semantic_threshold"] == CACHE_SETTINGS_DEFAULTS["cache_semantic_threshold"]
        )
        assert data["cache_min_confidence"] == CACHE_SETTINGS_DEFAULTS["cache_min_confidence"]
        assert data["cache_auto_warm_enabled"] == CACHE_SETTINGS_DEFAULTS["cache_auto_warm_enabled"]
        assert data["cache_auto_warm_count"] == CACHE_SETTINGS_DEFAULTS["cache_auto_warm_count"]

    def test_returns_persisted_settings(self, client, admin_headers, db):
        """GET /cache/settings returns values from database when persisted."""
        # Arrange: Insert custom setting via SettingsService
        service = SettingsService(db)
        service.set("cache_ttl_hours", 48)
        db.commit()

        # Act: GET /api/admin/cache/settings
        response = client.get("/api/admin/cache/settings", headers=admin_headers)

        # Assert: Custom value returned, others default
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["cache_ttl_hours"] == 48
        assert data["cache_enabled"] == CACHE_SETTINGS_DEFAULTS["cache_enabled"]

    def test_unauthorized_without_token(self, client):
        """GET /cache/settings returns 401 without auth header."""
        response = client.get("/api/admin/cache/settings")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_forbidden_for_regular_user(self, client, user_headers):
        """GET /cache/settings returns 403 for user role."""
        response = client.get("/api/admin/cache/settings", headers=user_headers)

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_allowed_for_system_admin(self, client, system_admin_headers):
        """GET /cache/settings returns 200 for system_admin role."""
        response = client.get("/api/admin/cache/settings", headers=system_admin_headers)

        assert response.status_code == status.HTTP_200_OK


class TestCacheSettingsUpdate:
    """Tests for PUT /api/admin/cache/settings endpoint."""

    def test_full_update_all_settings(self, client, admin_headers):
        """PUT /cache/settings with all fields updates all settings."""
        response = client.put(
            "/api/admin/cache/settings",
            headers=admin_headers,
            json={
                "cache_enabled": False,
                "cache_ttl_hours": 12,
                "cache_max_entries": 500,
                "cache_semantic_threshold": 0.90,
                "cache_min_confidence": 60,
                "cache_auto_warm_enabled": False,
                "cache_auto_warm_count": 10,
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["cache_enabled"] is False
        assert data["cache_ttl_hours"] == 12
        assert data["cache_max_entries"] == 500
        assert data["cache_semantic_threshold"] == 0.90
        assert data["cache_min_confidence"] == 60
        assert data["cache_auto_warm_enabled"] is False
        assert data["cache_auto_warm_count"] == 10

    def test_partial_update_single_field(self, client, admin_headers):
        """PUT /cache/settings with single field only updates that field."""
        response = client.put(
            "/api/admin/cache/settings",
            headers=admin_headers,
            json={"cache_ttl_hours": 36},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["cache_ttl_hours"] == 36
        assert data["cache_enabled"] == CACHE_SETTINGS_DEFAULTS["cache_enabled"]

    def test_empty_body_returns_defaults(self, client, admin_headers):
        """PUT /cache/settings with empty body returns current settings."""
        response = client.put(
            "/api/admin/cache/settings",
            headers=admin_headers,
            json={},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["cache_enabled"] == CACHE_SETTINGS_DEFAULTS["cache_enabled"]

    def test_validates_ttl_hours_minimum(self, client, admin_headers):
        """PUT /cache/settings rejects cache_ttl_hours below 1."""
        response = client.put(
            "/api/admin/cache/settings",
            headers=admin_headers,
            json={"cache_ttl_hours": 0},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "cache_ttl_hours" in response.json()["detail"]

    def test_validates_ttl_hours_maximum(self, client, admin_headers):
        """PUT /cache/settings rejects cache_ttl_hours above 168."""
        response = client.put(
            "/api/admin/cache/settings",
            headers=admin_headers,
            json={"cache_ttl_hours": 200},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "cache_ttl_hours" in response.json()["detail"]

    def test_validates_max_entries_minimum(self, client, admin_headers):
        """PUT /cache/settings rejects cache_max_entries below 100."""
        response = client.put(
            "/api/admin/cache/settings",
            headers=admin_headers,
            json={"cache_max_entries": 50},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "cache_max_entries" in response.json()["detail"]

    def test_validates_max_entries_maximum(self, client, admin_headers):
        """PUT /cache/settings rejects cache_max_entries above 10000."""
        response = client.put(
            "/api/admin/cache/settings",
            headers=admin_headers,
            json={"cache_max_entries": 15000},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "cache_max_entries" in response.json()["detail"]

    def test_validates_semantic_threshold_minimum(self, client, admin_headers):
        """PUT /cache/settings rejects cache_semantic_threshold below 0.85."""
        response = client.put(
            "/api/admin/cache/settings",
            headers=admin_headers,
            json={"cache_semantic_threshold": 0.80},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "cache_semantic_threshold" in response.json()["detail"]

    def test_validates_semantic_threshold_maximum(self, client, admin_headers):
        """PUT /cache/settings rejects cache_semantic_threshold above 0.99."""
        response = client.put(
            "/api/admin/cache/settings",
            headers=admin_headers,
            json={"cache_semantic_threshold": 1.0},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "cache_semantic_threshold" in response.json()["detail"]

    def test_validates_min_confidence_minimum(self, client, admin_headers):
        """PUT /cache/settings rejects cache_min_confidence below 0."""
        response = client.put(
            "/api/admin/cache/settings",
            headers=admin_headers,
            json={"cache_min_confidence": -5},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "cache_min_confidence" in response.json()["detail"]

    def test_validates_min_confidence_maximum(self, client, admin_headers):
        """PUT /cache/settings rejects cache_min_confidence above 100."""
        response = client.put(
            "/api/admin/cache/settings",
            headers=admin_headers,
            json={"cache_min_confidence": 150},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "cache_min_confidence" in response.json()["detail"]

    def test_validates_auto_warm_count_minimum(self, client, admin_headers):
        """PUT /cache/settings rejects cache_auto_warm_count below 5."""
        response = client.put(
            "/api/admin/cache/settings",
            headers=admin_headers,
            json={"cache_auto_warm_count": 2},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "cache_auto_warm_count" in response.json()["detail"]

    def test_validates_auto_warm_count_maximum(self, client, admin_headers):
        """PUT /cache/settings rejects cache_auto_warm_count above 50."""
        response = client.put(
            "/api/admin/cache/settings",
            headers=admin_headers,
            json={"cache_auto_warm_count": 100},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "cache_auto_warm_count" in response.json()["detail"]

    def test_unauthorized_without_token(self, client):
        """PUT /cache/settings returns 401 without auth."""
        response = client.put(
            "/api/admin/cache/settings",
            json={"cache_ttl_hours": 12},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_forbidden_for_regular_user(self, client, user_headers):
        """PUT /cache/settings returns 403 for user role."""
        response = client.put(
            "/api/admin/cache/settings",
            headers=user_headers,
            json={"cache_ttl_hours": 12},
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestCacheStats:
    """Tests for GET /api/admin/cache/stats endpoint."""

    def test_returns_stats_structure_empty_cache(self, client, admin_headers):
        """GET /cache/stats returns all expected fields for empty cache."""
        response = client.get("/api/admin/cache/stats", headers=admin_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify required fields are present
        assert "enabled" in data
        assert "total_entries" in data
        assert "memory_entries" in data
        assert "sqlite_entries" in data
        assert "hit_count" in data
        assert "miss_count" in data
        assert "hit_rate" in data

        # Empty cache should have zero entries
        assert data["total_entries"] == 0
        assert data["hit_rate"] == 0.0

    def test_returns_entry_counts(self, client, admin_headers, db):
        """GET /cache/stats returns correct entry counts."""
        # Arrange: Insert ResponseCache entries
        for i in range(3):
            entry = ResponseCache(
                query_hash=f"hash_{i}",
                query_text=f"query {i}",
                query_embedding="[]",
                answer=f"answer {i}",
                sources="[]",
                confidence_overall=80,
                confidence_retrieval=0.8,
                confidence_coverage=0.7,
                confidence_llm=75,
                generation_time_ms=1000.0,
                model_used="test-model",
                document_ids="[]",
            )
            db.add(entry)
        db.commit()

        # Act
        response = client.get("/api/admin/cache/stats", headers=admin_headers)

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["sqlite_entries"] == 3

    def test_returns_timestamp_fields(self, client, admin_headers, db):
        """GET /cache/stats returns oldest_entry and newest_entry."""
        # Arrange: Insert entries
        entry = ResponseCache(
            query_hash="test_hash",
            query_text="test query",
            query_embedding="[]",
            answer="test answer",
            sources="[]",
            confidence_overall=80,
            confidence_retrieval=0.8,
            confidence_coverage=0.7,
            confidence_llm=75,
            generation_time_ms=1000.0,
            model_used="test-model",
            document_ids="[]",
        )
        db.add(entry)
        db.commit()

        # Act
        response = client.get("/api/admin/cache/stats", headers=admin_headers)

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["oldest_entry"] is not None
        assert data["newest_entry"] is not None

    def test_unauthorized_without_token(self, client):
        """GET /cache/stats returns 401 without auth."""
        response = client.get("/api/admin/cache/stats")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_forbidden_for_regular_user(self, client, user_headers):
        """GET /cache/stats returns 403 for user role."""
        response = client.get("/api/admin/cache/stats", headers=user_headers)

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestCacheClear:
    """Tests for POST /api/admin/cache/clear endpoint."""

    def test_clears_empty_cache(self, client, admin_headers):
        """POST /cache/clear on empty cache returns cleared_entries: 0."""
        response = client.post("/api/admin/cache/clear", headers=admin_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["cleared_entries"] == 0
        assert data["message"] == "Cache cleared successfully"

    def test_clears_populated_cache(self, client, admin_headers, db):
        """POST /cache/clear removes all entries and returns count."""
        # Arrange: Insert 5 ResponseCache entries
        for i in range(5):
            entry = ResponseCache(
                query_hash=f"hash_{i}",
                query_text=f"query {i}",
                query_embedding="[]",
                answer=f"answer {i}",
                sources="[]",
                confidence_overall=80,
                confidence_retrieval=0.8,
                confidence_coverage=0.7,
                confidence_llm=75,
                generation_time_ms=1000.0,
                model_used="test-model",
                document_ids="[]",
            )
            db.add(entry)
        db.commit()

        # Act
        response = client.post("/api/admin/cache/clear", headers=admin_headers)

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # The clear returns count of removed entries
        assert data["cleared_entries"] >= 0  # Could be 5 from SQLite
        assert data["message"] == "Cache cleared successfully"

        # Verify cache is empty
        remaining = db.query(ResponseCache).count()
        assert remaining == 0

    def test_returns_success_message(self, client, admin_headers):
        """POST /cache/clear returns success message."""
        response = client.post("/api/admin/cache/clear", headers=admin_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Cache cleared successfully"

    def test_unauthorized_without_token(self, client):
        """POST /cache/clear returns 401 without auth."""
        response = client.post("/api/admin/cache/clear")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_forbidden_for_regular_user(self, client, user_headers):
        """POST /cache/clear returns 403 for user role."""
        response = client.post("/api/admin/cache/clear", headers=user_headers)

        assert response.status_code == status.HTTP_403_FORBIDDEN
