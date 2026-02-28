"""Tests for CA module migration structure."""

import pathlib


class TestMigrationStructure:
    def test_ca_migrations_directory_exists(self):
        p = pathlib.Path("ai_ready_rag/modules/community_associations/migrations")
        assert p.is_dir()

    def test_ca_migration_001_exists(self):
        p = pathlib.Path(
            "ai_ready_rag/modules/community_associations/migrations/versions/001_insurance_tables.py"
        )
        assert p.exists()

    def test_ca_migrations_env_exists(self):
        p = pathlib.Path("ai_ready_rag/modules/community_associations/migrations/env.py")
        assert p.exists()

    def test_ca_migrations_script_mako_exists(self):
        p = pathlib.Path("ai_ready_rag/modules/community_associations/migrations/script.py.mako")
        assert p.exists()

    def test_ca_migrations_versions_init_exists(self):
        p = pathlib.Path(
            "ai_ready_rag/modules/community_associations/migrations/versions/__init__.py"
        )
        assert p.exists()

    def test_migration_001_importable(self):
        """Migration file can be imported without errors."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "ca_migration_001",
            pathlib.Path(
                "ai_ready_rag/modules/community_associations/migrations/versions/001_insurance_tables.py"
            ),
        )
        assert spec is not None, "Migration file spec_from_file_location returned None"
        module = importlib.util.module_from_spec(spec)
        assert module is not None

    def test_migration_001_has_correct_revision(self):
        """Migration 001 has the expected revision ID and no parent."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "ca_migration_001",
            pathlib.Path(
                "ai_ready_rag/modules/community_associations/migrations/versions/001_insurance_tables.py"
            ),
        )
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.revision == "ca_001"
        assert module.down_revision is None

    def test_migration_001_has_upgrade_and_downgrade(self):
        """Migration 001 defines callable upgrade() and downgrade()."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "ca_migration_001",
            pathlib.Path(
                "ai_ready_rag/modules/community_associations/migrations/versions/001_insurance_tables.py"
            ),
        )
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert callable(getattr(module, "upgrade", None)), "upgrade() not defined"
        assert callable(getattr(module, "downgrade", None)), "downgrade() not defined"
