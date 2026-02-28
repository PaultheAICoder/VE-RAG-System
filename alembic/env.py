"""Alembic environment configuration.

Supports both SQLite (offline/dev) and PostgreSQL (online/hosted+enterprise) modes.
- SQLite: falls back to create_all() — Alembic runs offline (no schema migrations)
- PostgreSQL: Alembic runs online with full migration history
"""

import logging
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

logger = logging.getLogger(__name__)

# Alembic Config object
config = context.config

# Configure logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import all models so that Base.metadata knows about them
# (must import before run_migrations_offline/online)
from ai_ready_rag.db.models import (  # noqa: F401, E402
    AdminSetting,
    AuditLog,
    Document,
    User,
)

# Import enrichment models (created in #366 — graceful fallback if not yet present)
try:
    from ai_ready_rag.db.models.enrichment import (  # noqa: F401
        EnrichmentEntity,
        EnrichmentSynopsis,
        ReviewItem,
    )
except ImportError:
    pass

from ai_ready_rag.db.database import Base  # noqa: E402

target_metadata = Base.metadata


def get_url() -> str:
    """Return database URL from environment or settings."""
    return os.environ.get("DATABASE_URL") or "sqlite:///./data/ai_ready_rag.db"


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def run_migrations_offline() -> None:
    """Run migrations in offline mode (no database connection required).

    Used for SQLite and for generating SQL scripts.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode (live database connection)."""
    url = get_url()

    if _is_sqlite(url):
        # SQLite dev path: Alembic cannot run migrations on SQLite safely in all cases.
        # Use create_all() fallback and mark all current migrations as applied.
        logger.info("SQLite detected — using create_all() fallback, skipping Alembic online mode")
        engine = create_engine(url, connect_args={"check_same_thread": False})
        with engine.begin() as conn:  # noqa: F841
            # create_all is handled by init_db() in main.py lifespan
            pass
        return

    # PostgreSQL path — full Alembic migration
    connectable = create_engine(
        url,
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
