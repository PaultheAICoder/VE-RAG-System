"""Tests for TagPredicateValidator."""

from unittest.mock import MagicMock

import pytest

from ai_ready_rag.services.sql_access_control import AccessDeniedError, TagPredicateValidator


def make_template(access_tags):
    t = MagicMock()
    t.access_tags = access_tags
    t.name = "test_template"
    return t


class TestTagPredicateValidator:
    def setup_method(self):
        self.validator = TagPredicateValidator()

    def test_empty_access_tags_allows_any_user(self):
        tmpl = make_template([])
        self.validator.validate(tmpl, ["hr"])
        self.validator.validate(tmpl, [])
        self.validator.validate(tmpl, None)

    def test_none_user_tags_admin_bypass(self):
        tmpl = make_template(["finance"])
        self.validator.validate(tmpl, None)  # should not raise

    def test_user_with_all_required_tags_allowed(self):
        tmpl = make_template(["finance", "executive"])
        self.validator.validate(tmpl, ["finance", "executive", "hr"])  # should not raise

    def test_user_missing_one_required_tag_denied(self):
        tmpl = make_template(["finance", "executive"])
        with pytest.raises(AccessDeniedError):
            self.validator.validate(tmpl, ["finance"])  # missing "executive"

    def test_user_with_no_tags_and_restricted_table_denied(self):
        tmpl = make_template(["finance"])
        with pytest.raises(AccessDeniedError):
            self.validator.validate(tmpl, [])

    def test_and_logic_not_or_logic(self):
        """Requires ALL tags, not just one."""
        tmpl = make_template(["finance", "executive", "board"])
        with pytest.raises(AccessDeniedError):
            self.validator.validate(tmpl, ["finance"])  # has 1 of 3, still denied

    def test_error_message_lists_missing_tags(self):
        tmpl = make_template(["finance", "executive"])
        with pytest.raises(AccessDeniedError, match="Missing"):
            self.validator.validate(tmpl, ["finance"])
