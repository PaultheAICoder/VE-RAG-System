"""Tests for CA extraction prompts."""

import pathlib

from ai_ready_rag.modules.community_associations.services.prompt_loader import (
    DOCUMENT_TYPE_TO_PROMPT,
    list_available_prompts,
    load_prompt,
)

PROMPTS_DIR = pathlib.Path(__file__).parent.parent / "prompts"


class TestPromptFiles:
    def test_all_prompt_files_exist(self):
        for doc_type, filename in DOCUMENT_TYPE_TO_PROMPT.items():
            path = PROMPTS_DIR / filename
            assert path.exists(), f"Missing prompt file for {doc_type}: {path}"

    def test_prompts_are_non_empty(self):
        for doc_type, filename in DOCUMENT_TYPE_TO_PROMPT.items():
            path = PROMPTS_DIR / filename
            if path.exists():
                content = path.read_text()
                assert len(content) > 100, f"Prompt for {doc_type} seems too short"

    def test_prompts_contain_json_schema(self):
        """Each prompt should contain a JSON schema example."""
        for doc_type, filename in DOCUMENT_TYPE_TO_PROMPT.items():
            path = PROMPTS_DIR / filename
            if path.exists():
                content = path.read_text()
                assert "{" in content, f"Prompt for {doc_type} should contain JSON schema"


class TestPromptLoader:
    def test_load_ccr_prompt(self):
        prompt = load_prompt("ccr")
        assert prompt is not None
        assert "insurance_requirements" in prompt

    def test_load_reserve_study_prompt(self):
        prompt = load_prompt("reserve_study")
        assert prompt is not None
        assert "percent_funded" in prompt

    def test_load_board_minutes_prompt(self):
        prompt = load_prompt("board_minutes")
        assert prompt is not None
        assert "insurance_resolutions" in prompt

    def test_load_appraisal_prompt(self):
        prompt = load_prompt("appraisal")
        assert prompt is not None
        assert "replacement" in prompt.lower()

    def test_load_unit_owner_letter_prompt(self):
        prompt = load_prompt("unit_owner_letter")
        assert prompt is not None
        assert "ho6_required" in prompt

    def test_bylaws_uses_same_prompt_as_ccr(self):
        assert load_prompt("bylaws") == load_prompt("ccr")

    def test_unknown_type_returns_none(self):
        assert load_prompt("nonexistent_type") is None

    def test_list_available_prompts(self):
        available = list_available_prompts()
        assert "ccr" in available
        assert "reserve_study" in available
        assert len(available) >= 5
