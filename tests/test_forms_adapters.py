"""Unit tests for forms adapter protocol implementations.

Tests VERagFormDBAdapter (PostgreSQL), VERagOCRAdapter, and VERagVLMAdapter
security and protocol compliance.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestVERagFormDBAdapter:
    """Tests for VERagFormDBAdapter security and protocol compliance."""

    def _make_adapter(self, database_url: str = "postgresql://fake/testdb"):
        """Create adapter with mocked psycopg2 (avoids real DB connection)."""
        from ai_ready_rag.services.ingestkit_adapters import VERagFormDBAdapter

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg2.connect", return_value=mock_conn):
            return VERagFormDBAdapter(database_url=database_url)

    @pytest.mark.unit
    def test_forms_identifier_validation(self):
        """VERagFormDBAdapter.check_table_name validates via ingestkit-forms."""
        adapter = self._make_adapter()

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
    def test_forms_check_table_name_calls_ingestkit(self):
        """check_table_name delegates to ingestkit_forms.validate_table_name."""
        adapter = self._make_adapter()

        with patch("ingestkit_forms.validate_table_name", return_value=None) as mock_validate:
            adapter.check_table_name("acord_25")

        mock_validate.assert_called_once_with("acord_25")

    @pytest.mark.unit
    def test_forms_schema_is_forms_data(self):
        """Adapter stores all tables under the forms_data schema."""
        adapter = self._make_adapter()
        assert adapter._SCHEMA == "forms_data"

    @pytest.mark.unit
    def test_forms_ensure_schema_called_on_init(self):
        """_ensure_schema runs CREATE SCHEMA IF NOT EXISTS on construction."""
        from ai_ready_rag.services.ingestkit_adapters import VERagFormDBAdapter

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg2.connect", return_value=mock_conn):
            VERagFormDBAdapter(database_url="postgresql://fake/db")

        executed = [c[0][0] for c in mock_cur.execute.call_args_list]
        assert any("CREATE SCHEMA IF NOT EXISTS" in sql for sql in executed)

    @pytest.mark.unit
    def test_forms_migration_idempotent(self):
        """Constructing VERagFormDBAdapter twice does not error."""
        from ai_ready_rag.services.ingestkit_adapters import VERagFormDBAdapter

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg2.connect", return_value=mock_conn):
            adapter1 = VERagFormDBAdapter(database_url="postgresql://fake/db")
            adapter2 = VERagFormDBAdapter(database_url="postgresql://fake/db")

        # Both should be created without error
        assert adapter1._SCHEMA == "forms_data"
        assert adapter2._SCHEMA == "forms_data"


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
