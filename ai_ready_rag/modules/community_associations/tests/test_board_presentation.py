"""Tests for BoardPresentationService."""

from datetime import date, datetime

from ai_ready_rag.modules.community_associations.services.board_presentation import (
    BoardPresentationPacket,
    BoardPresentationService,
)


class TestBoardPresentationPacket:
    def test_to_dict_structure(self):
        packet = BoardPresentationPacket(
            account_id="a1",
            account_name="Test HOA",
            meeting_date="March 1, 2026",
            generated_at=datetime.utcnow(),
        )
        d = packet.to_dict()
        assert d["account_id"] == "a1"
        assert "slides" in d
        assert len(d["slides"]) == 4

    def test_to_dict_slide_types(self):
        packet = BoardPresentationPacket(
            account_id="a1",
            account_name="Test",
            meeting_date="March 1, 2026",
            generated_at=datetime.utcnow(),
        )
        d = packet.to_dict()
        slide_types = [s["type"] for s in d["slides"]]
        assert "coverage_summary" in slide_types
        assert "compliance_status" in slide_types
        assert "upcoming_renewals" in slide_types
        assert "recommended_resolutions" in slide_types


class TestBoardPresentationServiceNoDb:
    def test_prepare_no_db(self):
        svc = BoardPresentationService(db=None)
        packet = svc.prepare("acct-001", "Oak Hills HOA")
        assert isinstance(packet, BoardPresentationPacket)
        assert packet.account_name == "Oak Hills HOA"

    def test_prepare_sets_slide_count(self):
        svc = BoardPresentationService(db=None)
        packet = svc.prepare("acct-001")
        assert packet.slide_count == 4

    def test_prepare_includes_resolutions(self):
        svc = BoardPresentationService(db=None)
        packet = svc.prepare("acct-001", "Test HOA")
        assert len(packet.recommended_resolutions.resolutions) > 0

    def test_prepare_meeting_date_formatted(self):
        svc = BoardPresentationService(db=None)
        packet = svc.prepare("acct-001", meeting_date=date(2026, 3, 15))
        assert "March 15, 2026" in packet.meeting_date

    def test_prepare_generated_at_is_datetime(self):
        svc = BoardPresentationService(db=None)
        packet = svc.prepare("acct-001")
        assert isinstance(packet.generated_at, datetime)

    def test_coverage_summary_empty_without_db(self):
        svc = BoardPresentationService(db=None)
        packet = svc.prepare("acct-001")
        assert packet.coverage_summary.total_premium == 0.0
        assert packet.coverage_summary.carrier_count == 0

    def test_generate_resolutions_default(self):
        svc = BoardPresentationService(db=None)
        packet = svc.prepare("acct-001", "Test HOA")
        # Without DB, no expiring policies → default accept resolution
        assert any(
            "accepts" in r.lower() or "authorizes" in r.lower()
            for r in packet.recommended_resolutions.resolutions
        )
