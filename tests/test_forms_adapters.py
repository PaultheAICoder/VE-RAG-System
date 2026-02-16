"""Unit tests for forms adapter protocol implementations.

Tests VERagFormDBAdapter, VERagOCRAdapter, and VERagVLMAdapter security
and protocol compliance.
"""

import os
from unittest.mock import patch

import pytest


class TestVERagFormDBAdapter:
    """Tests for VERagFormDBAdapter security and protocol compliance."""

    @pytest.mark.unit
    def test_forms_identifier_validation(self, tmp_path):
        """VERagFormDBAdapter should validate table names via ingestkit-forms."""
        # Create a valid data/ directory structure for path validation
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = str(data_dir / "forms_data.db")

        with patch(
            "ai_ready_rag.services.ingestkit_adapters.VERagFormDBAdapter._validate_path",
            return_value=db_path,
        ):
            from ai_ready_rag.services.ingestkit_adapters import VERagFormDBAdapter

            adapter = VERagFormDBAdapter(db_path=db_path)

        # Test check_table_name with mocked validate_table_name
        with patch("ingestkit_forms.validate_table_name") as mock_validate:
            # Valid identifier
            mock_validate.return_value = None
            adapter.check_table_name("form_w2_2024")  # Should not raise

            # Invalid identifier
            mock_validate.return_value = "Unsafe table identifier"
            with pytest.raises(ValueError, match="Unsafe table identifier"):
                adapter.check_table_name("form-w2")  # Hyphen not allowed

            # SQL injection attempt
            mock_validate.return_value = "Unsafe table identifier"
            with pytest.raises(ValueError, match="Unsafe table identifier"):
                adapter.check_table_name("DROP TABLE users")

    @pytest.mark.unit
    def test_forms_path_traversal_rejected(self):
        """VERagFormDBAdapter should reject paths outside data/ directory."""
        from ai_ready_rag.services.ingestkit_adapters import VERagFormDBAdapter

        # Attempt path traversal
        with pytest.raises(ValueError, match="must be under data/"):
            VERagFormDBAdapter(db_path="/etc/passwd")

        with pytest.raises(ValueError, match="must be under data/"):
            VERagFormDBAdapter(db_path="../../../etc/passwd")

    @pytest.mark.unit
    def test_forms_migration_idempotent(self, tmp_path):
        """Running _ensure_db() twice should not error."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = str(data_dir / "forms_data.db")

        with patch(
            "ai_ready_rag.services.ingestkit_adapters.VERagFormDBAdapter._validate_path",
            return_value=db_path,
        ):
            from ai_ready_rag.services.ingestkit_adapters import VERagFormDBAdapter

            # Create twice — should not error
            adapter1 = VERagFormDBAdapter(db_path=db_path)
            adapter2 = VERagFormDBAdapter(db_path=db_path)

            # Both should have valid db path
            assert adapter1._db_path == db_path
            assert adapter2._db_path == db_path

            # File should exist with 0600 permissions
            assert os.path.exists(db_path)
            stat = os.stat(db_path)
            assert oct(stat.st_mode & 0o777) == "0o600"


class TestVERagOCRAdapter:
    """Tests for VERagOCRAdapter protocol implementation."""

    @pytest.mark.unit
    def test_forms_ocr_adapter_engine_name(self):
        """VERagOCRAdapter should report correct engine name."""
        from ai_ready_rag.services.ingestkit_adapters import VERagOCRAdapter

        adapter = VERagOCRAdapter(engine="tesseract")
        assert adapter.engine_name() == "tesseract"


class TestVERagVLMAdapter:
    """Tests for VERagVLMAdapter protocol implementation."""

    @pytest.mark.unit
    def test_forms_vlm_adapter_model_name(self):
        """VERagVLMAdapter should report correct model name."""
        from ai_ready_rag.services.ingestkit_adapters import VERagVLMAdapter

        adapter = VERagVLMAdapter(
            ollama_url="http://localhost:11434",
            model="qwen2.5-vl:7b",
        )
        assert adapter.model_name() == "qwen2.5-vl:7b"
