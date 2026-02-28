"""Tests for RenewalPrepService."""

from datetime import date, timedelta

from ai_ready_rag.modules.community_associations.services.renewal_prep import (
    ExpiringPolicy,
    RenewalPrepPacket,
    RenewalPrepService,
)


class TestRenewalPrepPacket:
    def test_has_expiring_policies_false_when_empty(self):
        from datetime import datetime

        packet = RenewalPrepPacket(
            account_id="a1",
            account_name="Test HOA",
            generated_at=datetime.utcnow(),
            renewal_window_days=90,
        )
        assert packet.has_expiring_policies is False

    def test_is_compliance_compliant_true_when_no_gaps(self):
        from datetime import datetime

        packet = RenewalPrepPacket(
            account_id="a1",
            account_name="Test HOA",
            generated_at=datetime.utcnow(),
            renewal_window_days=90,
        )
        assert packet.is_compliance_compliant is True

    def test_is_compliance_compliant_false_with_high_severity_gap(self):
        from datetime import datetime

        packet = RenewalPrepPacket(
            account_id="a1",
            account_name="Test HOA",
            generated_at=datetime.utcnow(),
            renewal_window_days=90,
            compliance_gaps=[{"severity": "high", "coverage_line": "general_liability"}],
        )
        assert packet.is_compliance_compliant is False


class TestRenewalPrepServiceNoDb:
    """Tests that run without a database connection."""

    def test_prepare_no_db_returns_packet(self):
        svc = RenewalPrepService(db=None)
        packet = svc.prepare("acct-001", "Oak Hills HOA")
        assert isinstance(packet, RenewalPrepPacket)
        assert packet.account_id == "acct-001"
        assert packet.account_name == "Oak Hills HOA"

    def test_prepare_includes_recommendation_when_no_db(self):
        svc = RenewalPrepService(db=None)
        packet = svc.prepare("acct-001")
        assert len(packet.recommendations) > 0

    def test_prepare_custom_window_days(self):
        svc = RenewalPrepService(db=None, renewal_window_days=60)
        packet = svc.prepare("acct-001")
        assert packet.renewal_window_days == 60

    def test_expiring_policy_dataclass(self):
        today = date.today()
        policy = ExpiringPolicy(
            policy_id="p1",
            policy_number="POL-001",
            carrier_name="State Farm",
            coverage_line="property",
            expiration_date=today + timedelta(days=30),
            days_until_expiry=30,
            total_premium_usd=12000.0,
        )
        assert policy.days_until_expiry == 30
        assert policy.total_premium_usd == 12000.0

    def test_generate_recommendations_urgent(self):
        from datetime import datetime

        svc = RenewalPrepService(db=None)
        today = date.today()
        packet = RenewalPrepPacket(
            account_id="a1",
            account_name="Test",
            generated_at=datetime.utcnow(),
            renewal_window_days=90,
            expiring_policies=[
                ExpiringPolicy(
                    policy_id="p1",
                    policy_number="POL-001",
                    carrier_name="X",
                    coverage_line="property",
                    expiration_date=today + timedelta(days=15),
                    days_until_expiry=15,
                    total_premium_usd=None,
                )
            ],
        )
        recs = svc._generate_recommendations(packet)
        assert any("URGENT" in r for r in recs)
