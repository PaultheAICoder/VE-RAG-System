"""Test configuration for Community Associations module tests."""

import pytest


@pytest.fixture(scope="session")
def ca_module_dir():
    """Return the CA module directory path."""
    import pathlib

    return pathlib.Path(__file__).parent.parent
