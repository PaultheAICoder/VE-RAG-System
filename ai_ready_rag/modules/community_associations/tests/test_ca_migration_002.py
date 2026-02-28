"""Tests for CA migration 002 — CA-specific operational tables."""

import importlib


class TestMigration002Structure:
    def test_revision_id(self):
        mod = importlib.import_module(
            "ai_ready_rag.modules.community_associations.migrations.versions.002_ca_tables"
        )
        assert mod.revision == "ca_002"

    def test_down_revision_points_to_001(self):
        mod = importlib.import_module(
            "ai_ready_rag.modules.community_associations.migrations.versions.002_ca_tables"
        )
        assert mod.down_revision == "ca_001"

    def test_upgrade_callable(self):
        mod = importlib.import_module(
            "ai_ready_rag.modules.community_associations.migrations.versions.002_ca_tables"
        )
        assert callable(mod.upgrade)

    def test_downgrade_callable(self):
        mod = importlib.import_module(
            "ai_ready_rag.modules.community_associations.migrations.versions.002_ca_tables"
        )
        assert callable(mod.downgrade)

    def test_table_names_in_upgrade_source(self):
        import inspect

        mod = importlib.import_module(
            "ai_ready_rag.modules.community_associations.migrations.versions.002_ca_tables"
        )
        src = inspect.getsource(mod.upgrade)
        for table in [
            "ca_reserve_studies",
            "ca_unit_owners",
            "ca_board_resolutions",
            "ca_letter_batches",
            "ca_letter_batch_items",
        ]:
            assert table in src, f"Expected table {table!r} in upgrade()"

    def test_pii_columns_encrypted(self):
        """Verify PII columns use _encrypted suffix (not plain text)."""
        import inspect

        mod = importlib.import_module(
            "ai_ready_rag.modules.community_associations.migrations.versions.002_ca_tables"
        )
        src = inspect.getsource(mod.upgrade)
        assert "owner_name_encrypted" in src
        assert "email_encrypted" in src
        assert "phone_encrypted" in src

    def test_soft_delete_columns_present(self):
        import inspect

        mod = importlib.import_module(
            "ai_ready_rag.modules.community_associations.migrations.versions.002_ca_tables"
        )
        src = inspect.getsource(mod.upgrade)
        assert src.count("deleted_at") >= 3  # at least 3 tables have soft-delete
