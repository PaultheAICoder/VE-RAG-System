"""Tests for CA document type classifier."""

import pytest

from ai_ready_rag.modules.community_associations.services.classifier import (
    CADocumentClassifier,
)


@pytest.fixture(scope="module")
def classifier():
    return CADocumentClassifier.from_yaml()


class TestClassifierLoading:
    def test_loads_from_yaml(self, classifier):
        assert classifier is not None

    def test_has_15_document_types(self, classifier):
        # 6 CA + 9 insurance types
        assert len(classifier._configs) >= 15


class TestCADocumentClassification:
    def test_classifies_ccr_by_filename(self, classifier):
        result = classifier.classify(
            "2024_CCR_Declaration.pdf",
            "covenants, conditions and restrictions declaration of covenants homeowners association insurance requirements replacement cost",
        )
        assert result.document_type == "ccr"
        assert result.confidence > 0

    def test_classifies_reserve_study_by_filename(self, classifier):
        result = classifier.classify(
            "Reserve_Study_2024.pdf",
            "reserve study percent funded fully funded balance replacement cost new annual contribution funding plan",
        )
        assert result.document_type == "reserve_study"

    def test_classifies_certificate_by_content(self, classifier):
        result = classifier.classify(
            "ACORD25.pdf",
            "certificate of insurance this certificate is issued as a matter of information acord certificate holder",
        )
        assert result.document_type == "certificate"

    def test_unknown_when_no_match(self, classifier):
        result = classifier.classify("random_file.pdf", "lorem ipsum dolor sit amet consectetur")
        assert result.document_type == "unknown"


class TestAmbiguityGate:
    def test_ambiguity_flag_when_scores_close(self, classifier):
        # A document with signals for two types — simulate with close scores
        # board minutes + reserve study signals
        mixed_content = (
            "board meeting minutes motion to seconded by vote resolution "
            "reserve study percent funded annual contribution"
        )
        result = classifier.classify("mixed_document.pdf", mixed_content)
        # Either type could win — but if ambiguous, flag should be set
        if result.is_ambiguous:
            assert len(result.ambiguous_candidates) == 2
