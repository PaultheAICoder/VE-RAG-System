"""Tests for the deterministic QueryRouter."""

import pytest

from ai_ready_rag.modules.registry import ModuleRegistry, SQLTemplate
from ai_ready_rag.services.query_router import QueryRouter, RouteType


@pytest.fixture(autouse=True)
def reset_registry():
    ModuleRegistry.reset()
    yield
    ModuleRegistry.reset()


@pytest.fixture
def router_with_templates():
    registry = ModuleRegistry.get_instance()
    registry.register_sql_templates(
        "test_module",
        {
            "ca_coverage_by_line": SQLTemplate(
                name="ca_coverage_by_line",
                sql="SELECT * FROM insurance_coverages WHERE account_id = :account_id LIMIT :row_cap",
                trigger_phrases=[
                    "coverage",
                    "insurance limit",
                    "coverage line",
                    "what is the coverage",
                ],
                description="Look up coverage by line of business",
            ),
            "ca_carrier_lookup": SQLTemplate(
                name="ca_carrier_lookup",
                sql="SELECT * FROM insurance_accounts WHERE account_name = :account_name LIMIT :row_cap",
                trigger_phrases=["carrier", "insurer", "who insures", "insurance company"],
                description="Look up carrier by account name",
            ),
        },
    )
    return QueryRouter(sql_confidence_threshold=0.5)


class TestQueryRouterDisabled:
    def test_rag_when_disabled(self, router_with_templates):
        decision = router_with_templates.route(
            "what is the coverage limit?", structured_query_enabled=False
        )
        assert decision.route == RouteType.RAG
        assert decision.reason == "structured_query_disabled"


class TestQueryRouterNoTemplates:
    def test_rag_when_no_templates(self):
        router = QueryRouter()
        decision = router.route("what is the coverage?", structured_query_enabled=True)
        assert decision.route == RouteType.RAG
        assert decision.reason == "no_sql_templates_registered"


class TestQueryRouterRouting:
    def test_sql_route_on_trigger_match(self, router_with_templates):
        decision = router_with_templates.route(
            "what is the coverage limit for the property?",
            structured_query_enabled=True,
        )
        assert decision.route == RouteType.SQL
        assert decision.template_name == "ca_coverage_by_line"
        assert "coverage" in decision.matched_phrases

    def test_rag_route_on_no_match(self, router_with_templates):
        decision = router_with_templates.route(
            "summarize the board meeting minutes",
            structured_query_enabled=True,
        )
        assert decision.route == RouteType.RAG
        assert decision.reason in ("no_trigger_match", "below_sql_threshold")

    def test_picks_best_template_on_multi_match(self, router_with_templates):
        decision = router_with_templates.route(
            "what is the coverage and who is the carrier?",
            structured_query_enabled=True,
        )
        # Should route to SQL — multiple templates match but one wins
        assert decision.route == RouteType.SQL

    def test_confidence_above_threshold_routes_sql(self, router_with_templates):
        decision = router_with_templates.route(
            "what is the coverage line?",
            structured_query_enabled=True,
        )
        assert decision.route == RouteType.SQL
        assert decision.confidence >= 0.5

    def test_review_route_when_floor_set(self):
        registry = ModuleRegistry.get_instance()
        registry.register_sql_templates(
            "test_module",
            {
                "test_tmpl": SQLTemplate(
                    name="test_tmpl",
                    sql="SELECT 1 LIMIT :row_cap",
                    trigger_phrases=["coverage", "policy", "limit", "carrier", "insurer"],
                ),
            },
        )
        router = QueryRouter(sql_confidence_threshold=0.9, review_floor=0.1)
        decision = router.route("what is the coverage?", structured_query_enabled=True)
        # Single phrase match -> confidence ~0.2 -> below 0.9 threshold, above 0.1 floor
        assert decision.route == RouteType.REVIEW
