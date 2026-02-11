"""Tests for admin cache warming endpoints."""

from fastapi import status

from ai_ready_rag.db.models import CacheAccessLog


class TestGetTopQueries:
    """Tests for GET /api/admin/cache/top-queries endpoint."""

    def test_returns_empty_list_when_no_data(self, client, admin_headers, db):
        """Test that endpoint returns empty list when no access logs exist."""
        response = client.get("/api/admin/cache/top-queries", headers=admin_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["queries"] == []

    def test_returns_queries_sorted_by_count(self, client, admin_headers, db):
        """Test that queries are returned sorted by access count descending."""
        # Insert access logs with different counts
        for _ in range(5):
            log = CacheAccessLog(
                query_hash="hash_popular",
                query_text="popular query",
                was_hit=True,
            )
            db.add(log)

        for _ in range(2):
            log = CacheAccessLog(
                query_hash="hash_less",
                query_text="less popular query",
                was_hit=False,
            )
            db.add(log)

        db.commit()

        response = client.get("/api/admin/cache/top-queries", headers=admin_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["queries"]) == 2
        assert data["queries"][0]["query_text"] == "popular query"
        assert data["queries"][0]["access_count"] == 5
        assert data["queries"][1]["query_text"] == "less popular query"
        assert data["queries"][1]["access_count"] == 2

    def test_custom_limit(self, client, admin_headers, db):
        """Test that limit parameter restricts results."""
        # Insert 10 different queries
        for i in range(10):
            log = CacheAccessLog(
                query_hash=f"hash_{i}",
                query_text=f"query {i}",
                was_hit=True,
            )
            db.add(log)
        db.commit()

        response = client.get("/api/admin/cache/top-queries?limit=5", headers=admin_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["queries"]) == 5

    def test_limit_validation_too_high(self, client, admin_headers):
        """Test that limit exceeding max returns 422."""
        response = client.get("/api/admin/cache/top-queries?limit=200", headers=admin_headers)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_limit_validation_zero(self, client, admin_headers):
        """Test that limit of 0 returns 422."""
        response = client.get("/api/admin/cache/top-queries?limit=0", headers=admin_headers)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_unauthorized(self, client):
        """Test that unauthenticated request returns 401."""
        response = client.get("/api/admin/cache/top-queries")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_forbidden_for_regular_user(self, client, user_headers):
        """Test that non-admin user returns 403."""
        response = client.get("/api/admin/cache/top-queries", headers=user_headers)

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestWarmCache:
    """Tests for POST /api/admin/cache/warm endpoint (returns 410 Gone)."""

    def test_endpoint_returns_410(self, client, admin_headers):
        """Legacy endpoint returns 410 Gone with redirect guidance."""
        response = client.post(
            "/api/admin/cache/warm",
            json={"queries": ["What is the vacation policy?"]},
            headers=admin_headers,
        )

        assert response.status_code == 410
        assert "warming/queue/manual" in response.json()["detail"]
