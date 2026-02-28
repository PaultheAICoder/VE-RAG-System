"""Tests for UnitOwnerLetterService."""

from datetime import datetime

from ai_ready_rag.modules.community_associations.services.unit_owner_letters import (
    HO6_COMPLIANCE_LETTER_TEMPLATE,
    BatchResult,
    LetterContent,
    UnitOwnerLetterService,
)


class TestLetterPreview:
    def setup_method(self):
        self.svc = UnitOwnerLetterService()

    def test_generate_preview_returns_letter_content(self):
        letter = self.svc.generate_preview("101A", "Oak Hills HOA")
        assert isinstance(letter, LetterContent)
        assert letter.unit_number == "101A"

    def test_preview_contains_unit_number(self):
        letter = self.svc.generate_preview("Unit 42", "Maple Grove HOA")
        assert "Unit 42" in letter.letter_text

    def test_preview_contains_association_name(self):
        letter = self.svc.generate_preview("101", "River Bend HOA")
        assert "River Bend HOA" in letter.letter_text

    def test_preview_contains_coverage_amount(self):
        letter = self.svc.generate_preview("101", "Test HOA", min_coverage_amount=50_000)
        assert "50,000" in letter.letter_text

    def test_preview_generated_at_is_datetime(self):
        letter = self.svc.generate_preview("101", "Test HOA")
        assert isinstance(letter.generated_at, datetime)


class TestBatchGeneration:
    def test_generate_batch_no_db_returns_empty_batch(self):
        svc = UnitOwnerLetterService(db=None)
        result = svc.generate_batch("acct-001", "Test HOA")
        assert isinstance(result, BatchResult)
        assert result.total_count == 0
        assert result.generated_count == 0
        assert result.status == "completed"

    def test_batch_result_status_completed_on_no_failures(self):
        svc = UnitOwnerLetterService(db=None)
        result = svc.generate_batch("acct-001", "Test HOA")
        assert result.status == "completed"
        assert result.failed_count == 0


class TestLetterTemplate:
    def test_template_has_required_placeholders(self):
        required = [
            "{unit_number}",
            "{association_name}",
            "{min_coverage_amount",
            "{compliance_deadline}",
            "{date}",
        ]
        for placeholder in required:
            assert placeholder in HO6_COMPLIANCE_LETTER_TEMPLATE

    def test_custom_template(self):
        svc = UnitOwnerLetterService(
            letter_template="Dear owner of unit {unit_number} at {association_name}. Deadline: {compliance_deadline}. Coverage: ${min_coverage_amount:,}. Date: {date}."
        )
        letter = svc.generate_preview("5B", "Custom HOA")
        assert "5B" in letter.letter_text
        assert "Custom HOA" in letter.letter_text
