"""Tests for retrieval settings endpoints including hybrid search fields."""


class TestGetRetrievalSettings:
    """Tests for GET /api/admin/settings/retrieval."""

    def test_get_retrieval_settings_returns_defaults(self, client, admin_headers):
        """GET returns all 8 fields with correct default values."""
        response = client.get("/api/admin/settings/retrieval", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["retrieval_top_k"] == 5
        assert data["retrieval_min_score"] == 0.3
        assert data["retrieval_enable_expansion"] is True
        assert data["retrieval_hybrid_enabled"] is False
        assert data["retrieval_prefetch_multiplier"] == 3
        assert data["retrieval_min_score_dense"] == 0.3
        assert data["retrieval_min_score_hybrid"] == 0.05
        assert data["retrieval_recency_weight"] == 0.15

    def test_get_retrieval_settings_requires_admin(self, client, user_headers):
        """Regular user gets 403."""
        response = client.get("/api/admin/settings/retrieval", headers=user_headers)
        assert response.status_code == 403


class TestUpdateRetrievalSettings:
    """Tests for PUT /api/admin/settings/retrieval."""

    def test_update_hybrid_enabled(self, client, admin_headers):
        """PUT with retrieval_hybrid_enabled: true persists and returns updated value."""
        response = client.put(
            "/api/admin/settings/retrieval",
            headers=admin_headers,
            json={"retrieval_hybrid_enabled": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["retrieval_hybrid_enabled"] is True

    def test_update_prefetch_multiplier_valid(self, client, admin_headers):
        """PUT with retrieval_prefetch_multiplier: 5 succeeds."""
        response = client.put(
            "/api/admin/settings/retrieval",
            headers=admin_headers,
            json={"retrieval_prefetch_multiplier": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["retrieval_prefetch_multiplier"] == 5

    def test_update_prefetch_multiplier_out_of_range(self, client, admin_headers):
        """PUT with value 0 or 11 returns 400."""
        response = client.put(
            "/api/admin/settings/retrieval",
            headers=admin_headers,
            json={"retrieval_prefetch_multiplier": 0},
        )
        assert response.status_code == 400

        response = client.put(
            "/api/admin/settings/retrieval",
            headers=admin_headers,
            json={"retrieval_prefetch_multiplier": 11},
        )
        assert response.status_code == 400

    def test_update_min_score_dense_valid(self, client, admin_headers):
        """PUT with retrieval_min_score_dense: 0.5 succeeds."""
        response = client.put(
            "/api/admin/settings/retrieval",
            headers=admin_headers,
            json={"retrieval_min_score_dense": 0.5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["retrieval_min_score_dense"] == 0.5

    def test_update_min_score_dense_out_of_range(self, client, admin_headers):
        """PUT with value 0.0 or 1.0 returns 400."""
        response = client.put(
            "/api/admin/settings/retrieval",
            headers=admin_headers,
            json={"retrieval_min_score_dense": 0.0},
        )
        assert response.status_code == 400

        response = client.put(
            "/api/admin/settings/retrieval",
            headers=admin_headers,
            json={"retrieval_min_score_dense": 1.0},
        )
        assert response.status_code == 400

    def test_update_min_score_hybrid_valid(self, client, admin_headers):
        """PUT with retrieval_min_score_hybrid: 0.1 succeeds."""
        response = client.put(
            "/api/admin/settings/retrieval",
            headers=admin_headers,
            json={"retrieval_min_score_hybrid": 0.1},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["retrieval_min_score_hybrid"] == 0.1

    def test_update_min_score_hybrid_out_of_range(self, client, admin_headers):
        """PUT with value 0.0 or 0.6 returns 400."""
        response = client.put(
            "/api/admin/settings/retrieval",
            headers=admin_headers,
            json={"retrieval_min_score_hybrid": 0.0},
        )
        assert response.status_code == 400

        response = client.put(
            "/api/admin/settings/retrieval",
            headers=admin_headers,
            json={"retrieval_min_score_hybrid": 0.6},
        )
        assert response.status_code == 400

    def test_update_recency_weight_valid(self, client, admin_headers):
        """PUT with retrieval_recency_weight: 0.25 succeeds."""
        response = client.put(
            "/api/admin/settings/retrieval",
            headers=admin_headers,
            json={"retrieval_recency_weight": 0.25},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["retrieval_recency_weight"] == 0.25

    def test_update_recency_weight_zero_disables(self, client, admin_headers):
        """PUT with retrieval_recency_weight: 0.0 disables recency boost."""
        response = client.put(
            "/api/admin/settings/retrieval",
            headers=admin_headers,
            json={"retrieval_recency_weight": 0.0},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["retrieval_recency_weight"] == 0.0

    def test_update_recency_weight_out_of_range(self, client, admin_headers):
        """PUT with value > 0.5 or < 0 returns 400."""
        response = client.put(
            "/api/admin/settings/retrieval",
            headers=admin_headers,
            json={"retrieval_recency_weight": 0.6},
        )
        assert response.status_code == 400

        response = client.put(
            "/api/admin/settings/retrieval",
            headers=admin_headers,
            json={"retrieval_recency_weight": -0.1},
        )
        assert response.status_code == 400

    def test_partial_update_preserves_other_fields(self, client, admin_headers):
        """PUT with only retrieval_hybrid_enabled doesn't change retrieval_top_k."""
        # Get initial state
        response = client.get("/api/admin/settings/retrieval", headers=admin_headers)
        initial_top_k = response.json()["retrieval_top_k"]

        # Update only hybrid_enabled
        response = client.put(
            "/api/admin/settings/retrieval",
            headers=admin_headers,
            json={"retrieval_hybrid_enabled": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["retrieval_hybrid_enabled"] is True
        assert data["retrieval_top_k"] == initial_top_k
