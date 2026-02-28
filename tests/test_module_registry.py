"""Tests for ModuleRegistry."""

import pytest

from ai_ready_rag.modules.registry import (
    ComplianceChecker,
    ModuleRegistry,
    _validate_sql_template,
    get_registry,
    init_registry,
)


@pytest.fixture
def registry():
    return ModuleRegistry()


class TestSQLTemplateValidation:
    def test_valid_template_passes(self):
        _validate_sql_template("test", "SELECT * FROM t WHERE id = :id LIMIT :row_cap")

    def test_dml_rejected(self):
        with pytest.raises(ValueError, match="DML"):
            _validate_sql_template("bad", "INSERT INTO t VALUES (:x) LIMIT :cap")

    def test_missing_limit_rejected(self):
        with pytest.raises(ValueError, match="LIMIT"):
            _validate_sql_template("bad", "SELECT * FROM t WHERE id = :id")

    def test_interpolation_rejected(self):
        with pytest.raises(ValueError, match="interpolation"):
            _validate_sql_template("bad", "SELECT * FROM {table} WHERE id = :id LIMIT :cap")


class TestRegistration:
    def test_register_entity_map(self, registry):
        registry.register_entity_map("test_module", {"unit_count": "accounts.units"})
        merged = registry.get_entity_map()
        assert merged["unit_count"] == "accounts.units"

    def test_register_sql_template(self, registry):
        registry.register_sql_templates(
            "test_module",
            {"my_query": "SELECT id FROM t WHERE acct = :account_id LIMIT :row_cap"},
        )
        assert registry.get_sql_template("my_query") is not None

    def test_register_compliance_checker(self, registry):
        class MyChecker(ComplianceChecker):
            def check(self, account_id, data):
                return {}

        registry.register_compliance_checker("test_module", MyChecker)
        checker = registry.get_compliance_checker("test_module")
        assert checker is not None

    def test_register_api_router(self, registry):
        from fastapi import APIRouter

        r = APIRouter()
        registry.register_api_router("test_module", r, "/api/test")
        routers = registry.get_api_routers()
        assert len(routers) == 1
        assert routers[0][0] == "test_module"


class TestActiveModules:
    def test_core_always_active(self, registry):
        assert "core" in registry.active_modules

    def test_load_nonexistent_module_does_not_raise(self, registry):
        # Should log error but not raise
        registry.load_module("nonexistent_module_xyz")
        assert "nonexistent_module_xyz" not in registry.active_modules


class TestSingleton:
    def test_init_registry_returns_instance(self):
        reg = init_registry()
        assert reg is not None

    def test_get_registry_returns_initialized(self):
        init_registry()
        reg = get_registry()
        assert reg is not None
