"""Tests for enrichment models — EnrichmentSynopsis, EnrichmentEntity, ReviewItem."""

import pytest

from ai_ready_rag.db.models.document import Document
from ai_ready_rag.db.models.enrichment import EnrichmentEntity, EnrichmentSynopsis, ReviewItem


class TestEnrichmentSynopsis:
    """Tests for the EnrichmentSynopsis model."""

    def test_synopsis_can_be_created_and_queried(self, db, admin_user):
        """EnrichmentSynopsis can be inserted and retrieved from the database."""
        doc = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/tmp/test.pdf",
            file_type="pdf",
            file_size=1024,
            uploaded_by=admin_user.id,
        )
        db.add(doc)
        db.flush()

        synopsis = EnrichmentSynopsis(
            document_id=doc.id,
            synopsis_text="This is a test synopsis describing the document.",
            model_id="claude-sonnet-4-6",
            prompt_version="1.0.0",
            token_cost=250,
            cost_usd=0.001,
        )
        db.add(synopsis)
        db.flush()
        db.refresh(synopsis)

        result = db.query(EnrichmentSynopsis).filter_by(id=synopsis.id).first()
        assert result is not None
        assert result.document_id == doc.id
        assert result.synopsis_text == "This is a test synopsis describing the document."
        assert result.model_id == "claude-sonnet-4-6"
        assert result.prompt_version == "1.0.0"
        assert result.token_cost == 250
        assert result.cost_usd == pytest.approx(0.001)
        assert result.created_at is not None

    def test_synopsis_optional_fields_are_nullable(self, db, admin_user):
        """prompt_version, token_cost, and cost_usd are all nullable."""
        doc = Document(
            filename="minimal.pdf",
            original_filename="minimal.pdf",
            file_path="/tmp/minimal.pdf",
            file_type="pdf",
            file_size=512,
            uploaded_by=admin_user.id,
        )
        db.add(doc)
        db.flush()

        synopsis = EnrichmentSynopsis(
            document_id=doc.id,
            synopsis_text="Minimal synopsis.",
            model_id="claude-sonnet-4-6",
        )
        db.add(synopsis)
        db.flush()
        db.refresh(synopsis)

        assert synopsis.prompt_version is None
        assert synopsis.token_cost is None
        assert synopsis.cost_usd is None

    def test_synopsis_fk_references_document(self, db, admin_user):
        """EnrichmentSynopsis.document_id is a FK to documents with ondelete=CASCADE."""
        fk_cols = {
            fk.column.table.name: fk.ondelete for fk in EnrichmentSynopsis.__table__.foreign_keys
        }
        assert "documents" in fk_cols, "Expected FK to 'documents' table"
        assert fk_cols["documents"] == "CASCADE", "Expected ondelete=CASCADE"


class TestEnrichmentEntity:
    """Tests for the EnrichmentEntity model."""

    def test_entity_links_to_synopsis(self, db, admin_user):
        """EnrichmentEntity is correctly linked to its parent EnrichmentSynopsis."""
        doc = Document(
            filename="entity_test.pdf",
            original_filename="entity_test.pdf",
            file_path="/tmp/entity_test.pdf",
            file_type="pdf",
            file_size=1024,
            uploaded_by=admin_user.id,
        )
        db.add(doc)
        db.flush()

        synopsis = EnrichmentSynopsis(
            document_id=doc.id,
            synopsis_text="Synopsis with entities.",
            model_id="claude-sonnet-4-6",
        )
        db.add(synopsis)
        db.flush()

        entity = EnrichmentEntity(
            synopsis_id=synopsis.id,
            entity_type="insurance_carrier",
            value="Acme Insurance Co.",
            canonical_value="Acme Insurance",
            confidence=0.95,
            source_chunk_index=3,
        )
        db.add(entity)
        db.flush()
        db.refresh(entity)

        result = db.query(EnrichmentEntity).filter_by(id=entity.id).first()
        assert result is not None
        assert result.synopsis_id == synopsis.id
        assert result.entity_type == "insurance_carrier"
        assert result.value == "Acme Insurance Co."
        assert result.canonical_value == "Acme Insurance"
        assert result.confidence == pytest.approx(0.95)
        assert result.source_chunk_index == 3
        assert result.created_at is not None

    def test_entity_optional_fields_are_nullable(self, db, admin_user):
        """canonical_value, confidence, and source_chunk_index are all nullable."""
        doc = Document(
            filename="minimal_entity.pdf",
            original_filename="minimal_entity.pdf",
            file_path="/tmp/minimal_entity.pdf",
            file_type="pdf",
            file_size=512,
            uploaded_by=admin_user.id,
        )
        db.add(doc)
        db.flush()

        synopsis = EnrichmentSynopsis(
            document_id=doc.id,
            synopsis_text="Synopsis for minimal entity.",
            model_id="claude-sonnet-4-6",
        )
        db.add(synopsis)
        db.flush()

        entity = EnrichmentEntity(
            synopsis_id=synopsis.id,
            entity_type="coverage_line",
            value="General Liability",
        )
        db.add(entity)
        db.flush()
        db.refresh(entity)

        assert entity.canonical_value is None
        assert entity.confidence is None
        assert entity.source_chunk_index is None

    def test_multiple_entities_per_synopsis(self, db, admin_user):
        """Multiple EnrichmentEntity rows can be linked to one synopsis."""
        doc = Document(
            filename="multi_entity.pdf",
            original_filename="multi_entity.pdf",
            file_path="/tmp/multi_entity.pdf",
            file_type="pdf",
            file_size=1024,
            uploaded_by=admin_user.id,
        )
        db.add(doc)
        db.flush()

        synopsis = EnrichmentSynopsis(
            document_id=doc.id,
            synopsis_text="Synopsis with multiple entities.",
            model_id="claude-sonnet-4-6",
        )
        db.add(synopsis)
        db.flush()

        entities = [
            EnrichmentEntity(
                synopsis_id=synopsis.id, entity_type="insurance_carrier", value="Carrier A"
            ),
            EnrichmentEntity(
                synopsis_id=synopsis.id, entity_type="coverage_line", value="Property"
            ),
            EnrichmentEntity(
                synopsis_id=synopsis.id, entity_type="coverage_line", value="Liability"
            ),
        ]
        for e in entities:
            db.add(e)
        db.flush()

        db.refresh(synopsis)
        assert len(synopsis.entities) == 3

    def test_entity_cascade_deleted_with_synopsis(self, db, admin_user):
        """Deleting an EnrichmentSynopsis cascades and removes its entities."""
        doc = Document(
            filename="entity_cascade.pdf",
            original_filename="entity_cascade.pdf",
            file_path="/tmp/entity_cascade.pdf",
            file_type="pdf",
            file_size=1024,
            uploaded_by=admin_user.id,
        )
        db.add(doc)
        db.flush()

        synopsis = EnrichmentSynopsis(
            document_id=doc.id,
            synopsis_text="Synopsis to cascade.",
            model_id="claude-sonnet-4-6",
        )
        db.add(synopsis)
        db.flush()

        entity = EnrichmentEntity(
            synopsis_id=synopsis.id,
            entity_type="insurance_carrier",
            value="To be deleted",
        )
        db.add(entity)
        db.flush()
        entity_id = entity.id

        db.delete(synopsis)
        db.flush()

        result = db.query(EnrichmentEntity).filter_by(id=entity_id).first()
        assert result is None


class TestReviewItem:
    """Tests for the ReviewItem model."""

    def test_review_item_can_be_created(self, db):
        """ReviewItem can be inserted and retrieved from the database."""
        item = ReviewItem(
            query_id="msg-abc-123",
            answer_text="The policy limit is $1M per occurrence.",
            confidence=0.42,
            reason="confidence_below_threshold",
            status="pending",
        )
        db.add(item)
        db.flush()
        db.refresh(item)

        result = db.query(ReviewItem).filter_by(id=item.id).first()
        assert result is not None
        assert result.query_id == "msg-abc-123"
        assert result.answer_text == "The policy limit is $1M per occurrence."
        assert result.confidence == pytest.approx(0.42)
        assert result.reason == "confidence_below_threshold"
        assert result.status == "pending"
        assert result.resolved_at is None
        assert result.resolved_by is None
        assert result.created_at is not None

    def test_review_item_default_status_is_pending(self, db):
        """ReviewItem.status defaults to 'pending' when not specified."""
        item = ReviewItem(
            answer_text="An unanswered question.",
        )
        db.add(item)
        db.flush()
        db.refresh(item)

        assert item.status == "pending"

    def test_review_item_all_fields_nullable_except_id_and_status(self, db):
        """All fields except id, status, and created_at are nullable."""
        item = ReviewItem()
        db.add(item)
        db.flush()
        db.refresh(item)

        assert item.id is not None
        assert item.status == "pending"
        assert item.created_at is not None
        assert item.query_id is None
        assert item.answer_text is None
        assert item.confidence is None
        assert item.reason is None
        assert item.resolved_at is None
        assert item.resolved_by is None

    def test_review_item_can_be_resolved_by_user(self, db, admin_user):
        """ReviewItem can reference a user via resolved_by FK."""
        from datetime import datetime

        item = ReviewItem(
            query_id="msg-resolve-test",
            answer_text="Approved answer.",
            confidence=0.9,
            status="approved",
            resolved_by=admin_user.id,
            resolved_at=datetime.utcnow(),
        )
        db.add(item)
        db.flush()
        db.refresh(item)

        assert item.resolved_by == admin_user.id
        assert item.resolved_at is not None
        assert item.status == "approved"


class TestDocumentEnrichmentColumns:
    """Tests for enrichment columns added to the Document model."""

    def test_document_enrichment_status_is_nullable(self, db, admin_user):
        """Document.enrichment_status defaults to None."""
        doc = Document(
            filename="enrichment_col_test.pdf",
            original_filename="enrichment_col_test.pdf",
            file_path="/tmp/enrichment_col_test.pdf",
            file_type="pdf",
            file_size=1024,
            uploaded_by=admin_user.id,
        )
        db.add(doc)
        db.flush()
        db.refresh(doc)

        assert doc.enrichment_status is None
        assert doc.enrichment_model is None
        assert doc.enrichment_version is None
        assert doc.enrichment_tokens_used is None
        assert doc.enrichment_cost_usd is None
        assert doc.enrichment_completed_at is None
        assert doc.document_role is None
        assert doc.synopsis_id is None

    def test_document_enrichment_columns_can_be_set(self, db, admin_user):
        """Document enrichment columns can be written and read back."""
        from datetime import datetime

        doc = Document(
            filename="enriched_doc.pdf",
            original_filename="enriched_doc.pdf",
            file_path="/tmp/enriched_doc.pdf",
            file_type="pdf",
            file_size=2048,
            uploaded_by=admin_user.id,
            enrichment_status="completed",
            enrichment_model="claude-sonnet-4-6",
            enrichment_version="1.0.0",
            enrichment_tokens_used=500,
            enrichment_cost_usd=0.002,
            enrichment_completed_at=datetime.utcnow(),
            document_role="insurance_policy",
        )
        db.add(doc)
        db.flush()
        db.refresh(doc)

        assert doc.enrichment_status == "completed"
        assert doc.enrichment_model == "claude-sonnet-4-6"
        assert doc.enrichment_version == "1.0.0"
        assert doc.enrichment_tokens_used == 500
        assert doc.enrichment_cost_usd == pytest.approx(0.002)
        assert doc.enrichment_completed_at is not None
        assert doc.document_role == "insurance_policy"

    def test_document_synopsis_relationship(self, db, admin_user):
        """Document.synopsis relationship resolves via synopsis_id FK."""
        doc = Document(
            filename="synopsis_rel.pdf",
            original_filename="synopsis_rel.pdf",
            file_path="/tmp/synopsis_rel.pdf",
            file_type="pdf",
            file_size=1024,
            uploaded_by=admin_user.id,
        )
        db.add(doc)
        db.flush()

        synopsis = EnrichmentSynopsis(
            document_id=doc.id,
            synopsis_text="Linked synopsis.",
            model_id="claude-sonnet-4-6",
        )
        db.add(synopsis)
        db.flush()

        # Link the synopsis back to the document via the synopsis_id FK
        doc.synopsis_id = synopsis.id
        db.flush()
        db.refresh(doc)

        assert doc.synopsis_id == synopsis.id
        assert doc.synopsis is not None
        assert doc.synopsis.synopsis_text == "Linked synopsis."
