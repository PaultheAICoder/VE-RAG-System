"""Tests for CA SQL template catalog."""

import pytest

from ai_ready_rag.modules.community_associations.services.sql_template_catalog import (
    load_sql_template_catalog,
)


@pytest.fixture(scope="module")
def catalog():
    return load_sql_template_catalog()


class TestCatalogLoading:
    def test_loads_without_error(self, catalog):
        assert catalog is not None
        assert catalog.module_id == "community_associations"

    def test_has_12_templates(self, catalog):
        assert len(catalog.templates) == 12

    def test_all_templates_have_limit_clause(self, catalog):
        for t in catalog.templates:
            assert "LIMIT" in t.sql.upper(), f"Template '{t.name}' missing LIMIT"

    def test_no_dml_in_any_template(self, catalog):
        import re

        dml_pattern = re.compile(
            r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE)\b",
            re.IGNORECASE,
        )
        for t in catalog.templates:
            match = dml_pattern.search(t.sql)
            assert match is None, f"Template '{t.name}' has DML: {match.group() if match else ''}"

    def test_all_templates_use_parameterized_bindings(self, catalog):
        import re

        interp_pattern = re.compile(r"\{[^}]+\}|%s|%\(")
        for t in catalog.templates:
            assert not interp_pattern.search(t.sql), f"Template '{t.name}' has string interpolation"


class TestTemplateLookup:
    def test_get_by_name(self, catalog):
        t = catalog.get("ca_coverage_by_account_line")
        assert t is not None
        assert t.display_name == "Coverage limits and deductibles by line of business"

    def test_get_missing_returns_none(self, catalog):
        assert catalog.get("nonexistent") is None

    def test_as_dict_returns_all_sqls(self, catalog):
        d = catalog.as_dict()
        assert len(d) == 12
        assert "ca_coverage_by_account_line" in d

    def test_trigger_map_has_phrases(self, catalog):
        tm = catalog.trigger_map()
        assert "premium" in tm
        assert tm["premium"] == "ca_premium_query"
        assert "carrier" in tm
        assert tm["carrier"] == "ca_carrier_lookup"


class TestPrimaryTemplates:
    @pytest.mark.parametrize(
        "name",
        [
            "ca_coverage_by_account_line",
            "ca_carrier_lookup",
            "ca_premium_query",
            "ca_policy_dates",
            "ca_claims_history",
            "ca_compliance_gap",
            "ca_coverage_schedule",
            "ca_comparison_by_line",
        ],
    )
    def test_primary_template_exists(self, catalog, name):
        assert catalog.get(name) is not None


class TestSecondaryTemplates:
    @pytest.mark.parametrize(
        "name",
        [
            "ca_reserve_status",
            "ca_requirements_by_source",
            "ca_unit_owner_status",
            "ca_board_resolutions",
        ],
    )
    def test_secondary_template_exists(self, catalog, name):
        assert catalog.get(name) is not None
