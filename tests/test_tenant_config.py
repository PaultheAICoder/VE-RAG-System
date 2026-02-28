"""Tests for TenantConfigResolver and PromptResolver."""

import pytest

from ai_ready_rag.tenant.config import (
    DEFAULT_TENANT_CONFIG,
    AIModelConfig,
    TenantConfig,
)
from ai_ready_rag.tenant.resolver import PromptResolver, TenantConfigResolver


class TestTenantConfig:
    def test_default_config_loads(self):
        assert DEFAULT_TENANT_CONFIG.tenant_id == "default"

    def test_ai_config_validates_pinned_models(self):
        """Pinned model IDs pass validation."""
        ai = AIModelConfig(
            enrichment_model="claude-sonnet-4-6",
            query_model_simple="claude-haiku-4-5-20251001",
        )
        assert ai.enrichment_model == "claude-sonnet-4-6"

    def test_ai_config_rejects_aliases(self):
        """Alias model IDs are rejected."""
        with pytest.raises(ValueError):
            AIModelConfig(enrichment_model="claude-sonnet")

    def test_tenant_config_feature_flags_default(self):
        config = TenantConfig(tenant_id="test", display_name="Test")
        assert hasattr(config, "feature_flags")


class TestTenantConfigResolver:
    def test_resolves_default_when_no_json(self, tmp_path):
        resolver = TenantConfigResolver(
            tenant_config_path_template=str(tmp_path / "{tenant_id}" / "tenant.json")
        )
        config = resolver.resolve("nonexistent-tenant-xyz")
        assert config.tenant_id == "nonexistent-tenant-xyz"

    def test_feature_flag_defaults_false(self, tmp_path):
        resolver = TenantConfigResolver(
            tenant_config_path_template=str(tmp_path / "{tenant_id}" / "tenant.json")
        )
        config = resolver.resolve("nonexistent-tenant-xyz")
        assert config.feature_flags.ca_enabled is False

    def test_active_modules_includes_core(self, tmp_path):
        resolver = TenantConfigResolver(
            tenant_config_path_template=str(tmp_path / "{tenant_id}" / "tenant.json")
        )
        config = resolver.resolve("nonexistent-tenant-xyz")
        assert "core" in config.active_modules


class TestPromptResolver:
    def test_nonexistent_prompt_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resolver = PromptResolver(tenant_id="default", module_id="core")
        assert resolver.resolve("nonexistent_prompt_xyz") is None

    def test_list_available_empty_when_no_dirs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resolver = PromptResolver(tenant_id="default", module_id="core")
        assert resolver.list_available() == []


class TestTenantConfigEndpoint:
    def test_get_config_requires_auth(self, client):
        response = client.get("/api/tenant/config")
        assert response.status_code == 401

    def test_get_config_returns_default(self, client, admin_headers):
        response = client.get("/api/tenant/config", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "tenant_id" in data
        assert "feature_flags" in data
