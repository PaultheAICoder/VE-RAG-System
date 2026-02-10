"""Database connection and session management."""

import logging
import os
from collections.abc import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

from ai_ready_rag.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Create engine with profile-based pool settings
# Pool size scales based on hardware capabilities (laptop vs spark)
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=False,  # Set to True to see SQL statements for debugging
    poolclass=QueuePool,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_pool_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_pre_ping=True,  # Verify connections are alive before using
)


# Enable WAL mode for SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)

    # One-time migration: drop orphan tables from old warming queue design (#193)
    try:
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS warming_failed_queries"))
            conn.execute(text("DROP TABLE IF EXISTS warming_queue"))
            conn.commit()
    except Exception as e:
        logger.warning(f"Migration cleanup skipped: {e}")

    # Migration: add unique constraint on warming_sse_events(job_id, batch_seq) (#214)
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS "
                    "uq_sse_events_job_batch_seq ON warming_sse_events(job_id, batch_seq)"
                )
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"SSE unique index migration skipped: {e}")

    # Migration: add confidence_score column to warming_queries (#189)
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE warming_queries ADD COLUMN confidence_score INTEGER"))
            conn.commit()
    except Exception:
        pass  # Column already exists

    logger.info("database_initialized", extra={"database_url": settings.database_url})


def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
