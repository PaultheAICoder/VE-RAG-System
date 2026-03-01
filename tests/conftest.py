"""Pytest configuration and fixtures."""

import os

# Test environment overrides - must be set BEFORE importing app
os.environ["ENABLE_GRADIO"] = "false"
os.environ["BCRYPT_ROUNDS"] = "4"  # Fast hashing for tests (prod uses 12)

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_ready_rag.core.security import create_access_token, hash_password
from ai_ready_rag.db.database import Base, get_db
from ai_ready_rag.db.models import Tag, User
from ai_ready_rag.main import app


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_postgres: Test requires PostgreSQL + pgvector. Skipped when database_backend=sqlite.",
    )


def pytest_runtest_setup(item):
    """Skip requires_postgres tests when running on SQLite backend."""
    if "requires_postgres" in item.keywords:
        from ai_ready_rag.config import get_settings

        settings = get_settings()
        database_backend = getattr(settings, "database_backend", "sqlite")
        if database_backend != "postgresql":
            pytest.skip(
                f"requires PostgreSQL — current database_backend={database_backend!r}. "
                "Set DATABASE_URL=postgresql://... and database_backend=postgresql to run."
            )


# Test database - in-memory SQLite with StaticPool for connection sharing
TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Create tables once for entire test session."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="module")
def db_connection(setup_database):
    """Module-scoped connection holding an outer transaction for savepoint rollback.

    This connection persists for the entire test module. The outer transaction
    is never committed — it is rolled back when the module finishes, ensuring
    all data created by module-scoped fixtures is cleaned up.
    """
    connection = engine.connect()
    transaction = connection.begin()
    yield connection
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="module")
def _module_db(db_connection):
    """Module-scoped session for creating shared fixtures (users, tokens, etc.).

    This session shares the module-level connection so that objects created
    here are visible to the per-test savepoint sessions (they share the same
    underlying SQLite connection via StaticPool).
    """
    session = TestingSessionLocal(bind=db_connection)
    yield session
    session.close()


@pytest.fixture(scope="function")
def db(db_connection):
    """Per-test session using a SAVEPOINT so mutations roll back after each test.

    Each test gets a fresh savepoint nested inside the module-level outer
    transaction. On teardown the savepoint is rolled back, undoing any
    mutations (e.g. failed_login_attempts, is_active, locked_until changes in
    test_auth.py) without disturbing module-scoped fixture data.
    """
    savepoint = db_connection.begin_nested()  # SAVEPOINT
    session = TestingSessionLocal(bind=db_connection)
    yield session
    session.close()
    savepoint.rollback()  # Roll back to SAVEPOINT; outer transaction stays open


@pytest.fixture(scope="function")
def client(db):
    """Create test client with database override."""

    def override_get_db():
        try:
            yield db
        finally:
            pass  # Do not close - managed by db fixture

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Module-scoped internal user fixtures — created once per module.
# These hold canonical user objects in _module_db (the outer-transaction session).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _module_admin_user(_module_db) -> User:
    """Create admin user once per module, flushed into outer transaction."""
    user = User(
        email="admin@test.com",
        display_name="Test Admin",
        password_hash=hash_password("AdminPassword123"),
        role="admin",
        is_active=True,
    )
    _module_db.add(user)
    _module_db.flush()
    _module_db.refresh(user)
    return user


@pytest.fixture(scope="module")
def _module_regular_user(_module_db) -> User:
    """Create regular user once per module, flushed into outer transaction."""
    user = User(
        email="user@test.com",
        display_name="Test User",
        password_hash=hash_password("UserPassword123"),
        role="user",
        is_active=True,
    )
    _module_db.add(user)
    _module_db.flush()
    _module_db.refresh(user)
    return user


@pytest.fixture(scope="module")
def _module_customer_admin_user(_module_db) -> User:
    """Create customer admin user once per module, flushed into outer transaction."""
    user = User(
        email="customer_admin@test.com",
        display_name="Customer Admin",
        password_hash=hash_password("CustomerAdminPassword123"),
        role="customer_admin",
        is_active=True,
    )
    _module_db.add(user)
    _module_db.flush()
    _module_db.refresh(user)
    return user


@pytest.fixture(scope="module")
def _module_system_admin_user(_module_db) -> User:
    """Create system admin user once per module, flushed into outer transaction."""
    user = User(
        email="system_admin@test.com",
        display_name="System Admin",
        password_hash=hash_password("SystemAdminPassword123"),
        role="system_admin",
        is_active=True,
    )
    _module_db.add(user)
    _module_db.flush()
    _module_db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Per-test user fixtures — merge module-level user into the per-test session.
# This makes db.refresh(admin_user) and direct attribute mutation work
# correctly within each test's savepoint scope.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def admin_user(db, _module_admin_user) -> User:
    """Return admin user merged into the per-test db session.

    Merging ensures db.refresh(admin_user) and direct attribute mutations
    (e.g. admin_user.is_active = False + db.commit()) work correctly in tests.
    The per-test savepoint rollback restores DB state after each test.
    """
    return db.merge(_module_admin_user)


@pytest.fixture(scope="function")
def regular_user(db, _module_regular_user) -> User:
    """Return regular user merged into the per-test db session."""
    return db.merge(_module_regular_user)


@pytest.fixture(scope="function")
def customer_admin_user(db, _module_customer_admin_user) -> User:
    """Return customer admin user merged into the per-test db session."""
    return db.merge(_module_customer_admin_user)


@pytest.fixture(scope="function")
def system_admin_user(db, _module_system_admin_user) -> User:
    """Return system admin user merged into the per-test db session."""
    return db.merge(_module_system_admin_user)


# ---------------------------------------------------------------------------
# Module-scoped token and header fixtures — stateless JWTs, no DB dependency.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def admin_token(_module_admin_user) -> str:
    """Get JWT token for admin user (module-scoped)."""
    return create_access_token(
        data={
            "sub": _module_admin_user.id,
            "email": _module_admin_user.email,
            "role": _module_admin_user.role,
        }
    )


@pytest.fixture(scope="module")
def user_token(_module_regular_user) -> str:
    """Get JWT token for regular user (module-scoped)."""
    return create_access_token(
        data={
            "sub": _module_regular_user.id,
            "email": _module_regular_user.email,
            "role": _module_regular_user.role,
        }
    )


@pytest.fixture(scope="module")
def customer_admin_token(_module_customer_admin_user) -> str:
    """Get JWT token for customer admin user (module-scoped)."""
    return create_access_token(
        data={
            "sub": _module_customer_admin_user.id,
            "email": _module_customer_admin_user.email,
            "role": _module_customer_admin_user.role,
        }
    )


@pytest.fixture(scope="module")
def system_admin_token(_module_system_admin_user) -> str:
    """Get JWT token for system admin user (module-scoped)."""
    return create_access_token(
        data={
            "sub": _module_system_admin_user.id,
            "email": _module_system_admin_user.email,
            "role": _module_system_admin_user.role,
        }
    )


@pytest.fixture(scope="module")
def admin_headers(admin_token) -> dict:
    """Authorization headers for admin (module-scoped)."""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture(scope="module")
def user_headers(user_token) -> dict:
    """Authorization headers for regular user (module-scoped)."""
    return {"Authorization": f"Bearer {user_token}"}


@pytest.fixture(scope="module")
def customer_admin_headers(customer_admin_token) -> dict:
    """Authorization headers for customer admin (module-scoped)."""
    return {"Authorization": f"Bearer {customer_admin_token}"}


@pytest.fixture(scope="module")
def system_admin_headers(system_admin_token) -> dict:
    """Authorization headers for system admin (module-scoped)."""
    return {"Authorization": f"Bearer {system_admin_token}"}


# ---------------------------------------------------------------------------
# Fixtures kept at scope="function" — mutated by tests or test-specific data.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def unrestricted_user(db) -> User:
    """Create a user with tag_access_enabled=False (sees all documents)."""
    user = User(
        email="unrestricted@test.com",
        display_name="Unrestricted User",
        password_hash=hash_password("UnrestrictedPassword123"),
        role="user",
        is_active=True,
        tag_access_enabled=False,
    )
    db.add(user)
    db.flush()
    db.refresh(user)
    return user


@pytest.fixture(scope="function")
def unrestricted_token(unrestricted_user) -> str:
    """Get JWT token for unrestricted user."""
    return create_access_token(
        data={
            "sub": unrestricted_user.id,
            "email": unrestricted_user.email,
            "role": unrestricted_user.role,
        }
    )


@pytest.fixture(scope="function")
def unrestricted_headers(unrestricted_token) -> dict:
    """Authorization headers for unrestricted user."""
    return {"Authorization": f"Bearer {unrestricted_token}"}


@pytest.fixture(scope="function")
def sample_tag(db, admin_user) -> Tag:
    """Create a sample tag."""
    tag = Tag(
        name="hr",
        display_name="Human Resources",
        description="HR department documents",
        color="#10B981",
        created_by=admin_user.id,
    )
    db.add(tag)
    db.flush()
    db.refresh(tag)
    return tag


# ---------------------------------------------------------------------------
# PostgreSQL fixtures (requires_postgres tests only)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pg_engine():
    """PostgreSQL engine for integration tests.

    Requires:
        DATABASE_URL env var pointing to a PostgreSQL instance
        database_backend=postgresql in settings

    Usage:
        @pytest.mark.requires_postgres
        def test_something(pg_db):
            ...
    """
    import os

    from sqlalchemy import create_engine as sa_create_engine

    database_url = os.environ.get("DATABASE_URL")
    if not database_url or "postgresql" not in database_url:
        pytest.skip("pg_engine requires DATABASE_URL=postgresql://...")
    engine = sa_create_engine(database_url)
    yield engine
    engine.dispose()


@pytest.fixture
def pg_db(pg_engine):
    """PostgreSQL session with transaction rollback per test."""
    from sqlalchemy.orm import sessionmaker as sa_sessionmaker

    PgSession = sa_sessionmaker(bind=pg_engine, autocommit=False, autoflush=False)
    connection = pg_engine.connect()
    transaction = connection.begin()
    session = PgSession(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()
