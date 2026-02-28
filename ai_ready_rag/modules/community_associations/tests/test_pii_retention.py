"""Tests for PII retention service."""

import pytest

from ai_ready_rag.modules.community_associations.services.pii_retention import PIIRetentionService


class TestPIIRetentionService:
    def test_import(self):
        assert PIIRetentionService is not None

    def test_init_default_retention_days(self):
        svc = PIIRetentionService()
        assert svc._retention_days == 365

    def test_init_custom_retention_days(self):
        svc = PIIRetentionService(retention_days=90)
        assert svc._retention_days == 90

    def test_get_status_no_db_raises(self):
        svc = PIIRetentionService()
        with pytest.raises(AttributeError):
            svc.get_status(None)

    def test_purge_dry_run_no_db_raises(self):
        svc = PIIRetentionService()
        with pytest.raises(AttributeError):
            svc.purge(None, dry_run=True)


class TestPIIAdminRouter:
    def test_import(self):
        from ai_ready_rag.modules.community_associations.api.pii_admin import router

        assert router is not None

    def test_router_has_purge_route(self):
        from ai_ready_rag.modules.community_associations.api.pii_admin import router

        routes = [r.path for r in router.routes]
        assert any("purge" in r for r in routes)

    def test_router_has_status_route(self):
        from ai_ready_rag.modules.community_associations.api.pii_admin import router

        routes = [r.path for r in router.routes]
        assert any("status" in r for r in routes)

    def test_router_has_policy_route(self):
        from ai_ready_rag.modules.community_associations.api.pii_admin import router

        routes = [r.path for r in router.routes]
        assert any("policy" in r for r in routes)
