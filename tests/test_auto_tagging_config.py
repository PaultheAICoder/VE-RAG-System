"""Tests for auto-tagging configuration settings and document model columns."""

import json

from ai_ready_rag.config import Settings
from ai_ready_rag.db.models.document import Document


class TestAutoTaggingConfig:
    """Test auto-tagging configuration settings."""

    def test_default_settings(self):
        """All auto-tagging settings have correct defaults."""
        settings = Settings()

        # Feature settings
        assert settings.auto_tagging_enabled is False
        assert settings.auto_tagging_strategy == "generic"
        assert settings.auto_tagging_path_enabled is True
        assert settings.auto_tagging_llm_enabled is True
        assert settings.auto_tagging_llm_model is None
        assert settings.auto_tagging_require_approval is False
        assert settings.auto_tagging_create_missing_tags is True
        assert settings.auto_tagging_confidence_threshold == 0.7
        assert settings.auto_tagging_suggestion_threshold == 0.4
        assert settings.auto_tagging_strategies_dir == "./data/auto_tag_strategies"

        # Guardrail settings
        assert settings.auto_tagging_max_tags_per_doc == 20
        assert settings.auto_tagging_max_tag_name_length == 100
        assert settings.auto_tagging_max_client_tags == 500
        assert settings.auto_tagging_llm_timeout_seconds == 30
        assert settings.auto_tagging_llm_max_retries == 1

    def test_auto_tagging_disabled_by_default(self):
        """Auto-tagging is disabled by default."""
        settings = Settings()
        assert settings.auto_tagging_enabled is False

    def test_env_var_overrides(self, monkeypatch):
        """Environment variables override defaults."""
        monkeypatch.setenv("AUTO_TAGGING_ENABLED", "true")
        monkeypatch.setenv("AUTO_TAGGING_STRATEGY", "insurance")
        monkeypatch.setenv("AUTO_TAGGING_CONFIDENCE_THRESHOLD", "0.85")
        monkeypatch.setenv("AUTO_TAGGING_MAX_TAGS_PER_DOC", "10")

        settings = Settings()
        assert settings.auto_tagging_enabled is True
        assert settings.auto_tagging_strategy == "insurance"
        assert settings.auto_tagging_confidence_threshold == 0.85
        assert settings.auto_tagging_max_tags_per_doc == 10

    def test_confidence_thresholds(self):
        """Confidence threshold > suggestion threshold by default."""
        settings = Settings()
        assert (
            settings.auto_tagging_confidence_threshold > settings.auto_tagging_suggestion_threshold
        )


class TestDocumentAutoTagColumns:
    """Test Document model auto-tag columns."""

    def test_auto_tag_columns_nullable(self, db):
        """Auto-tag columns accept null values."""
        doc = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/tmp/test.pdf",
            file_type="pdf",
            file_size=1024,
            uploaded_by="test-user-id",
        )
        db.add(doc)
        db.flush()
        db.refresh(doc)

        assert doc.auto_tag_status is None
        assert doc.auto_tag_strategy is None
        assert doc.auto_tag_version is None
        assert doc.auto_tag_source is None

    def test_auto_tag_status_values(self, db):
        """Auto-tag status accepts valid status strings."""
        for status_value in ["pending", "completed", "partial", "failed"]:
            doc = Document(
                filename=f"test_{status_value}.pdf",
                original_filename=f"test_{status_value}.pdf",
                file_path=f"/tmp/test_{status_value}.pdf",
                file_type="pdf",
                file_size=1024,
                uploaded_by="test-user-id",
                auto_tag_status=status_value,
            )
            db.add(doc)
            db.flush()
            db.refresh(doc)
            assert doc.auto_tag_status == status_value

    def test_auto_tag_source_stores_json(self, db):
        """auto_tag_source stores JSON text."""
        provenance = json.dumps(
            {
                "strategy": "generic",
                "version": "1.0",
                "tags_applied": ["hr", "finance"],
                "confidence_scores": {"hr": 0.9, "finance": 0.75},
            }
        )
        doc = Document(
            filename="test_json.pdf",
            original_filename="test_json.pdf",
            file_path="/tmp/test_json.pdf",
            file_type="pdf",
            file_size=1024,
            uploaded_by="test-user-id",
            auto_tag_status="completed",
            auto_tag_strategy="generic",
            auto_tag_version="1.0",
            auto_tag_source=provenance,
        )
        db.add(doc)
        db.flush()
        db.refresh(doc)

        parsed = json.loads(doc.auto_tag_source)
        assert parsed["strategy"] == "generic"
        assert "hr" in parsed["tags_applied"]
        assert parsed["confidence_scores"]["hr"] == 0.9
