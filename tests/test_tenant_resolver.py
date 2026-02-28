"""Tests for TenantConfigResolver and PromptResolver."""

from __future__ import annotations

import json

from ai_ready_rag.tenant.config import TenantConfig
from ai_ready_rag.tenant.resolver import PromptResolver, TenantConfigResolver


class TestTenantConfig:
    def test_defaults(self):
        cfg = TenantConfig(tenant_id="test")
        assert cfg.tenant_id == "test"
        assert cfg.feature_flags.ca_enabled is False
        assert cfg.active_modules == ["core"]

    def test_feature_flags(self):
        cfg = TenantConfig(tenant_id="t", feature_flags={"ca_enabled": True})
        assert cfg.feature_flags.ca_enabled is True

    def test_display_name_default(self):
        cfg = TenantConfig(tenant_id="x")
        assert cfg.display_name == ""

    def test_extra_fields_ignored(self):
        """TenantConfig should ignore unknown keys (extra = 'ignore')."""
        cfg = TenantConfig(tenant_id="x", unknown_field="should_be_ignored")
        assert not hasattr(cfg, "unknown_field")


class TestTenantConfigResolver:
    def test_missing_file_returns_defaults(self, tmp_path):
        resolver = TenantConfigResolver(
            tenant_config_path_template=str(tmp_path / "{tenant_id}" / "tenant.json")
        )
        cfg = resolver.resolve("nonexistent")
        assert cfg.tenant_id == "nonexistent"
        assert cfg.feature_flags.ca_enabled is False

    def test_loads_tenant_json(self, tmp_path):
        tenant_dir = tmp_path / "acme"
        tenant_dir.mkdir()
        (tenant_dir / "tenant.json").write_text(
            json.dumps(
                {
                    "display_name": "Acme HOA",
                    "feature_flags": {"ca_enabled": True},
                    "active_modules": ["core", "community_associations"],
                }
            )
        )
        resolver = TenantConfigResolver(
            tenant_config_path_template=str(tmp_path / "{tenant_id}" / "tenant.json")
        )
        cfg = resolver.resolve("acme")
        assert cfg.display_name == "Acme HOA"
        assert cfg.feature_flags.ca_enabled is True
        assert "community_associations" in cfg.active_modules

    def test_cache_returns_same_object(self, tmp_path):
        resolver = TenantConfigResolver(
            tenant_config_path_template=str(tmp_path / "{tenant_id}" / "tenant.json")
        )
        c1 = resolver.resolve("x")
        c2 = resolver.resolve("x")
        assert c1 is c2

    def test_invalidate_clears_cache(self, tmp_path):
        resolver = TenantConfigResolver(
            tenant_config_path_template=str(tmp_path / "{tenant_id}" / "tenant.json")
        )
        resolver.resolve("y")
        resolver.invalidate("y")
        assert "y" not in resolver._cache

    def test_invalid_json_returns_defaults(self, tmp_path):
        """Corrupt JSON in tenant.json should fall back to defaults, not raise."""
        tenant_dir = tmp_path / "broken"
        tenant_dir.mkdir()
        (tenant_dir / "tenant.json").write_text("{not valid json}")
        resolver = TenantConfigResolver(
            tenant_config_path_template=str(tmp_path / "{tenant_id}" / "tenant.json")
        )
        cfg = resolver.resolve("broken")
        assert cfg.tenant_id == "broken"
        assert cfg.feature_flags.ca_enabled is False

    def test_multiple_tenants_isolated(self, tmp_path):
        """Each tenant_id should get its own cached config."""
        for name in ("alpha", "beta"):
            d = tmp_path / name
            d.mkdir()
            (d / "tenant.json").write_text(json.dumps({"display_name": name.capitalize()}))
        resolver = TenantConfigResolver(
            tenant_config_path_template=str(tmp_path / "{tenant_id}" / "tenant.json")
        )
        alpha = resolver.resolve("alpha")
        beta = resolver.resolve("beta")
        assert alpha.display_name == "Alpha"
        assert beta.display_name == "Beta"
        assert alpha is not beta


class TestPromptResolver:
    def test_returns_none_when_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resolver = PromptResolver(tenant_id="test", module_id="core")
        assert resolver.resolve("nonexistent_prompt") is None

    def test_resolves_from_module_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        prompt_dir = tmp_path / "ai_ready_rag" / "modules" / "core" / "prompts"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "test_prompt.txt").write_text("Hello {name}")
        resolver = PromptResolver(tenant_id="test", module_id="core")
        result = resolver.resolve("test_prompt")
        assert result == "Hello {name}"

    def test_tenant_overrides_module(self, tmp_path, monkeypatch):
        """Tenant prompt should take precedence over module prompt."""
        monkeypatch.chdir(tmp_path)
        # Module-level prompt
        module_dir = tmp_path / "ai_ready_rag" / "modules" / "core" / "prompts"
        module_dir.mkdir(parents=True)
        (module_dir / "greeting.txt").write_text("Module greeting")
        # Tenant-level override
        tenant_dir = tmp_path / "tenant-instances" / "myco" / "prompts"
        tenant_dir.mkdir(parents=True)
        (tenant_dir / "greeting.txt").write_text("Tenant greeting")

        resolver = PromptResolver(tenant_id="myco", module_id="core")
        result = resolver.resolve("greeting")
        assert result == "Tenant greeting"

    def test_resolves_from_core_dir(self, tmp_path, monkeypatch):
        """Falls back to core prompts directory when tenant and module dirs are absent."""
        monkeypatch.chdir(tmp_path)
        core_dir = tmp_path / "ai_ready_rag" / "prompts"
        core_dir.mkdir(parents=True)
        (core_dir / "base.txt").write_text("Base prompt")
        resolver = PromptResolver(tenant_id="nobody", module_id="missing_module")
        result = resolver.resolve("base")
        assert result == "Base prompt"

    def test_list_available_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resolver = PromptResolver(tenant_id="test", module_id="core")
        assert resolver.list_available() == []

    def test_list_available_returns_sorted_names(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        prompt_dir = tmp_path / "ai_ready_rag" / "prompts"
        prompt_dir.mkdir(parents=True)
        for name in ("zebra", "alpha", "middle"):
            (prompt_dir / f"{name}.txt").write_text(name)
        resolver = PromptResolver(tenant_id="test", module_id="core")
        names = resolver.list_available()
        assert names == sorted(names)
        assert set(names) == {"zebra", "alpha", "middle"}
