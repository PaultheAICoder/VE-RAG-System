"""Test that requires_postgres marker properly skips on SQLite backend."""

import pytest


def test_no_marker_always_runs():
    """Tests without the marker always run."""
    assert True


@pytest.mark.requires_postgres
def test_requires_postgres_skips_on_sqlite():
    """This test should be skipped when database_backend=sqlite."""
    # If we get here, we're on PostgreSQL
    assert True
