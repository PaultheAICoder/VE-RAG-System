"""Tests for FormsQueryService and forms-eligible intent detection."""

import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from ai_ready_rag.db.models import Tag
from ai_ready_rag.db.models.document import Document
from ai_ready_rag.services.forms_query_service import (
    INTENT_TO_GROUPS,
    FormsQueryService,
    field_name_to_label,
    strip_column_prefix,
)
from ai_ready_rag.services.rag_service import QueryIntent, detect_query_intent

# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestStripColumnPrefix:
    def test_full_prefix(self):
        assert (
            strip_column_prefix("F[0].P1[0].GeneralLiability_EachOccurrence_LimitAmount_A[0]")
            == "GeneralLiability_EachOccurrence_LimitAmount_A"
        )

    def test_no_prefix(self):
        assert strip_column_prefix("NamedInsured_FullName_A") == "NamedInsured_FullName_A"

    def test_trailing_index_only(self):
        assert strip_column_prefix("Producer_ContactName[0]") == "Producer_ContactName"

    def test_multi_digit_indices(self):
        assert strip_column_prefix("F[12].P99[3].Field_Name[10]") == "Field_Name"


class TestFieldNameToLabel:
    def test_camel_case_split(self):
        assert (
            field_name_to_label("GeneralLiability_EachOccurrenceLimit") == "Each Occurrence Limit"
        )

    def test_simple_field(self):
        assert field_name_to_label("Producer_ContactName") == "Contact Name"

    def test_no_prefix(self):
        assert field_name_to_label("FullName") == "Full Name"

    def test_underscore_in_name(self):
        # split("_", 1) removes first prefix, then CamelCase split + underscore→space
        assert (
            field_name_to_label("Policy_GeneralLiability_PolicyNumberIdentifier_A")
            == "General Liability Policy Number Identifier A"
        )


class TestIntentToGroups:
    def test_gl_maps_to_general_liability(self):
        assert INTENT_TO_GROUPS["gl"] == ["General Liability"]

    def test_certificate_maps_to_all(self):
        assert INTENT_TO_GROUPS["certificate"] is None

    def test_workers_comp_maps_correctly(self):
        assert INTENT_TO_GROUPS["workers_comp"] == ["Workers Comp"]

    def test_umbrella_maps_correctly(self):
        assert INTENT_TO_GROUPS["umbrella"] == ["Umbrella/Excess"]


# ---------------------------------------------------------------------------
# FormsQueryService tests
# ---------------------------------------------------------------------------


@pytest.fixture
def forms_db(tmp_path):
    """Create a temporary forms_data.db with test data."""
    db_path = tmp_path / "forms_data.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE acord_25 (
            _ingest_key TEXT PRIMARY KEY,
            "F[0].P1[0].NamedInsured_FullName_A[0]" TEXT,
            "F[0].P1[0].GeneralLiability_EachOccurrence_LimitAmount_A[0]" TEXT,
            "F[0].P1[0].Policy_GeneralLiability_PolicyNumberIdentifier_A[0]" TEXT,
            "F[0].P1[0].Producer_ContactName[0]" TEXT,
            "F[0].P1[0].WorkersCompensation_StatutoryLimits[0]" TEXT
        )"""
    )
    conn.execute(
        """INSERT INTO acord_25 VALUES (
            'test-key-001',
            'Bethany Terrace HOA',
            '1,000,000',
            'HOA1000028015-02',
            'John Smith',
            '500,000'
        )"""
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def forms_service(forms_db):
    """Create a FormsQueryService with test DB."""
    settings = MagicMock()
    settings.forms_db_path = str(forms_db)
    return FormsQueryService(settings)


@pytest.fixture
def doc_with_forms(db):
    """Create a document with forms metadata and a tag."""
    tag = Tag(
        name="client:bethany-terrace",
        display_name="Bethany Terrace",
        color="#000000",
    )
    db.add(tag)
    db.flush()

    doc = Document(
        filename="acord25_bethany.pdf",
        original_filename="ACORD 25 - Bethany Terrace.pdf",
        file_path="/uploads/acord25_bethany.pdf",
        file_type="pdf",
        file_size=100000,
        status="ready",
        forms_ingest_key="test-key-001",
        forms_db_table_names=json.dumps(["acord_25"]),
        forms_template_name="ACORD 25",
    )
    doc.tags.append(tag)
    db.add(doc)
    db.flush()
    db.refresh(doc)
    return doc


class TestFormsQueryService:
    def test_no_forms_db_returns_empty(self, db):
        """When forms_data.db doesn't exist, return empty."""
        settings = MagicMock()
        settings.forms_db_path = "/nonexistent/forms_data.db"
        service = FormsQueryService(settings)

        intent = QueryIntent(
            preferred_tags=[], intent_label="certificate", confidence=0.5, forms_eligible=True
        )
        results = service.query_forms(intent, user_tags=None, db=db)
        assert results == []

    def test_non_forms_intent_returns_empty(self, forms_service, db, doc_with_forms):
        """Intent not in INTENT_TO_GROUPS returns empty."""
        intent = QueryIntent(
            preferred_tags=[], intent_label="earthquake", confidence=0.5, forms_eligible=False
        )
        results = forms_service.query_forms(intent, user_tags=None, db=db)
        assert results == []

    def test_certificate_returns_all_groups(self, forms_service, db, doc_with_forms):
        """Certificate intent returns fields from all groups."""
        intent = QueryIntent(
            preferred_tags=[], intent_label="certificate", confidence=0.5, forms_eligible=True
        )
        results = forms_service.query_forms(intent, user_tags=None, db=db)
        assert len(results) > 0

        # Should have multiple groups
        sections = {r.section for r in results}
        assert len(sections) >= 2  # At least Named Insured + General Liability

    def test_gl_returns_only_general_liability(self, forms_service, db, doc_with_forms):
        """GL intent returns only General Liability group."""
        intent = QueryIntent(
            preferred_tags=[], intent_label="gl", confidence=0.5, forms_eligible=True
        )
        results = forms_service.query_forms(intent, user_tags=None, db=db)
        assert len(results) > 0
        for r in results:
            assert r.section == "General Liability"

    def test_format_search_result_fields(self, forms_service, db, doc_with_forms):
        """Synthetic results have correct fields."""
        intent = QueryIntent(
            preferred_tags=[], intent_label="certificate", confidence=0.5, forms_eligible=True
        )
        results = forms_service.query_forms(intent, user_tags=None, db=db)
        assert len(results) > 0

        result = results[0]
        assert result.score == 0.95
        assert result.chunk_index >= 9000
        assert result.document_id == doc_with_forms.id
        assert result.document_name == doc_with_forms.original_filename
        assert result.page_number == 1
        assert "ACORD 25" in result.chunk_text

    def test_access_control_filters_by_tags(self, forms_service, db, doc_with_forms):
        """User without matching tag gets no results."""
        intent = QueryIntent(
            preferred_tags=[], intent_label="certificate", confidence=0.5, forms_eligible=True
        )
        # User has tags that don't match the document
        results = forms_service.query_forms(intent, user_tags=["client:other-company"], db=db)
        assert results == []

    def test_access_control_matching_tag(self, forms_service, db, doc_with_forms):
        """User with matching tag gets results."""
        intent = QueryIntent(
            preferred_tags=[], intent_label="certificate", confidence=0.5, forms_eligible=True
        )
        results = forms_service.query_forms(intent, user_tags=["client:bethany-terrace"], db=db)
        assert len(results) > 0

    def test_admin_bypass_returns_all(self, forms_service, db, doc_with_forms):
        """Admin (user_tags=None) gets all form documents."""
        intent = QueryIntent(
            preferred_tags=[], intent_label="certificate", confidence=0.5, forms_eligible=True
        )
        results = forms_service.query_forms(intent, user_tags=None, db=db)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Intent detection integration tests
# ---------------------------------------------------------------------------


class TestFormsEligibleIntent:
    def test_certificate_is_forms_eligible(self):
        intent = detect_query_intent("certificate of liability insurance for Bethany Terrace")
        assert intent.forms_eligible is True

    def test_gl_is_forms_eligible(self):
        intent = detect_query_intent("what are the general liability limits")
        assert intent.forms_eligible is True

    def test_active_policy_is_forms_eligible(self):
        intent = detect_query_intent("current policy limits and coverages")
        assert intent.forms_eligible is True

    def test_workers_comp_is_forms_eligible(self):
        intent = detect_query_intent("workers compensation statutory limits")
        assert intent.forms_eligible is True

    def test_generic_query_not_forms_eligible(self):
        intent = detect_query_intent("hello how are you")
        assert intent.forms_eligible is False

    def test_umbrella_is_forms_eligible(self):
        intent = detect_query_intent("umbrella liability coverage amount")
        assert intent.forms_eligible is True
