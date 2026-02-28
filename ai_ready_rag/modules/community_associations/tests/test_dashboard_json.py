"""Tests for dashboard.json structure."""

import json
from pathlib import Path


class TestDashboardJson:
    def setup_method(self):
        dashboard_path = Path(__file__).parent.parent / "dashboard.json"
        with open(dashboard_path) as f:
            self.config = json.load(f)

    def test_valid_json(self):
        assert isinstance(self.config, dict)

    def test_module_id(self):
        assert self.config["module_id"] == "community_associations"

    def test_has_dashboard_key(self):
        assert "dashboard" in self.config

    def test_four_views(self):
        views = self.config["dashboard"]["views"]
        assert len(views) == 4

    def test_view_ids(self):
        view_ids = {v["id"] for v in self.config["dashboard"]["views"]}
        expected = {"ca_coverage_summary", "ca_compliance_gaps", "ca_renewals", "ca_unit_owners"}
        assert view_ids == expected

    def test_three_automation_actions(self):
        actions = self.config["dashboard"]["automation_actions"]
        assert len(actions) == 3

    def test_automation_action_ids(self):
        action_ids = {a["id"] for a in self.config["dashboard"]["automation_actions"]}
        expected = {"generate_renewal_packet", "send_ho6_letters", "generate_board_presentation"}
        assert action_ids == expected

    def test_all_views_have_api_endpoint(self):
        for view in self.config["dashboard"]["views"]:
            assert "api_endpoint" in view, f"View {view['id']} missing api_endpoint"

    def test_all_actions_have_confirm(self):
        for action in self.config["dashboard"]["automation_actions"]:
            assert "confirm" in action, f"Action {action['id']} missing confirm dialog"

    def test_feature_flags_present(self):
        actions = self.config["dashboard"]["automation_actions"]
        for action in actions:
            assert "feature_flag" in action
