"""Tests for LLM settings endpoints including auto_tagging_llm_model fields."""


class TestGetLLMSettings:
    """Tests for GET /api/admin/settings/llm."""

    def test_get_llm_settings_includes_auto_tagging_fields(self, client, admin_headers):
        """GET returns auto_tagging_llm_model and auto_tagging_llm_model_effective fields."""
        response = client.get("/api/admin/settings/llm", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert "auto_tagging_llm_model" in data
        assert data["auto_tagging_llm_model"] is None  # default: no override set
        assert "auto_tagging_llm_model_effective" in data
        assert isinstance(data["auto_tagging_llm_model_effective"], str)
        assert len(data["auto_tagging_llm_model_effective"]) > 0  # always non-empty

    def test_get_llm_settings_requires_admin(self, client, user_headers):
        """Regular user gets 403."""
        response = client.get("/api/admin/settings/llm", headers=user_headers)
        assert response.status_code == 403
