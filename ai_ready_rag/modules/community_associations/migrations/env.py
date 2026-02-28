"""Alembic migration environment for the community_associations module.

Uses a separate alembic_version_community_associations table to track module
migration state independently from core platform migrations.
"""

import logging
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

logger = logging.getLogger(__name__)

config = context.config

if config.config_file_name is not None:
    try:
        fileConfig(config.config_file_name)
    except Exception:
        pass

# Import CA-specific models (graceful fallback if not yet installed)
try:
    from ai_ready_rag.modules.community_associations.models import Base as CABase  # noqa: F401

    target_metadata = CABase.metadata
except ImportError:
    target_metadata = None

VERSION_TABLE = "alembic_version_community_associations"


def get_url() -> str:
    return os.environ.get("DATABASE_URL") or "sqlite:///./data/ai_ready_rag.db"


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=VERSION_TABLE,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = get_url()
    if _is_sqlite(url):
        logger.info("CA migrations: SQLite detected — create_all() handles schema, skipping")
        return

    connectable = create_engine(url, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table=VERSION_TABLE,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
