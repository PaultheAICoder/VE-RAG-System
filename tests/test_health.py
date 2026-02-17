"""Tests for health check endpoints."""

from unittest.mock import patch


class TestHealth:
    """Health endpoint tests."""

    def test_health_check(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_version_endpoint(self, client):
        """Test version endpoint."""
        response = client.get("/api/version")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_root_endpoint(self, client):
        """Test root endpoint returns either JSON or HTML (if frontend built)."""
        response = client.get("/")
        assert response.status_code == 200
        # If frontend dist exists, returns HTML; otherwise JSON
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            assert "<!DOCTYPE html>" in response.text or "<html" in response.text
        else:
            data = response.json()
            assert "message" in data
            assert "version" in data


class TestHealthAutoTagging:
    """Auto-tagging section in health endpoint."""

    def test_health_includes_auto_tagging_key(self, client):
        """Health response always contains auto_tagging key."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "auto_tagging" in data

    def test_auto_tagging_disabled(self, client):
        """When auto_tagging_enabled=False, returns minimal block."""
        with patch("ai_ready_rag.api.health.settings") as mock_settings:
            # Copy real settings attributes needed by health_check
            from ai_ready_rag.config import get_settings

            real = get_settings()
            for attr in dir(real):
                if not attr.startswith("_"):
                    try:
                        setattr(mock_settings, attr, getattr(real, attr))
                    except (AttributeError, TypeError):
                        pass
            mock_settings.auto_tagging_enabled = False

            response = client.get("/api/health")
            assert response.status_code == 200
            data = response.json()
            at = data["auto_tagging"]
            assert at == {"enabled": False}

    def test_auto_tagging_enabled_no_strategy_file(self, client):
        """When enabled but strategy file missing, returns error or fallback status."""
        with patch("ai_ready_rag.api.health.settings") as mock_settings:
            from ai_ready_rag.config import get_settings

            real = get_settings()
            for attr in dir(real):
                if not attr.startswith("_"):
                    try:
                        setattr(mock_settings, attr, getattr(real, attr))
                    except (AttributeError, TypeError):
                        pass
            mock_settings.auto_tagging_enabled = True
            mock_settings.auto_tagging_strategy = "nonexistent_strategy"
            mock_settings.auto_tagging_strategies_dir = "/tmp/no_such_dir"

            response = client.get("/api/health")
            assert response.status_code == 200
            data = response.json()
            at = data["auto_tagging"]
            assert at["enabled"] is True
            assert at["strategy_status"] in ("fallback_generic", "error")

    def test_auto_tagging_enabled_has_expected_keys(self, client):
        """When enabled, all spec-required keys are present."""
        with patch("ai_ready_rag.api.health.settings") as mock_settings:
            from ai_ready_rag.config import get_settings

            real = get_settings()
            for attr in dir(real):
                if not attr.startswith("_"):
                    try:
                        setattr(mock_settings, attr, getattr(real, attr))
                    except (AttributeError, TypeError):
                        pass
            mock_settings.auto_tagging_enabled = True
            mock_settings.auto_tagging_strategy = "test_missing"
            mock_settings.auto_tagging_strategies_dir = "/tmp/no_such_dir"

            response = client.get("/api/health")
            assert response.status_code == 200
            at = response.json()["auto_tagging"]

            expected_keys = {
                "enabled",
                "strategy",
                "strategy_name",
                "strategy_version",
                "strategy_status",
                "path_enabled",
                "llm_enabled",
                "llm_model",
                "require_approval",
                "namespaces",
                "document_types",
                "path_rules",
                "guardrails",
            }
            assert expected_keys.issubset(at.keys()), f"Missing keys: {expected_keys - at.keys()}"

            # Validate guardrails sub-keys
            guardrails = at["guardrails"]
            assert "max_tags_per_doc" in guardrails
            assert "max_client_tags" in guardrails
            assert "current_client_tags" in guardrails
            assert isinstance(guardrails["current_client_tags"], int)

    def test_auto_tagging_error_status_defaults(self, client):
        """When strategy_status is error, strategy fields have safe defaults."""
        with patch("ai_ready_rag.api.health.settings") as mock_settings:
            from ai_ready_rag.config import get_settings

            real = get_settings()
            for attr in dir(real):
                if not attr.startswith("_"):
                    try:
                        setattr(mock_settings, attr, getattr(real, attr))
                    except (AttributeError, TypeError):
                        pass
            mock_settings.auto_tagging_enabled = True
            mock_settings.auto_tagging_strategy = "nonexistent"
            mock_settings.auto_tagging_strategies_dir = "/tmp/no_such_dir"

            response = client.get("/api/health")
            at = response.json()["auto_tagging"]

            if at["strategy_status"] == "error":
                assert at["strategy_name"] is None
                assert at["strategy_version"] is None
                assert at["namespaces"] == []
                assert at["document_types"] == 0
                assert at["path_rules"] == 0

    def test_auto_tagging_client_tags_counted(self, client, db):
        """current_client_tags reflects actual DB count of client: tags."""
        from ai_ready_rag.db.models import Tag

        # Create some client namespace tags
        for i in range(3):
            tag = Tag(
                name=f"client:test-client-{i}",
                display_name=f"Test Client {i}",
                description="Test",
                color="#000000",
            )
            db.add(tag)
        db.flush()

        with patch("ai_ready_rag.api.health.settings") as mock_settings:
            from ai_ready_rag.config import get_settings

            real = get_settings()
            for attr in dir(real):
                if not attr.startswith("_"):
                    try:
                        setattr(mock_settings, attr, getattr(real, attr))
                    except (AttributeError, TypeError):
                        pass
            mock_settings.auto_tagging_enabled = True
            mock_settings.auto_tagging_strategy = "nonexistent"
            mock_settings.auto_tagging_strategies_dir = "/tmp/no_such_dir"

            response = client.get("/api/health")
            at = response.json()["auto_tagging"]
            assert at["guardrails"]["current_client_tags"] == 3
