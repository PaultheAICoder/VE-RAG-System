"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_ready_rag.core.security import create_access_token, hash_password
from ai_ready_rag.db.database import Base, get_db
from ai_ready_rag.db.models import Tag, User
from ai_ready_rag.main import app

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


@pytest.fixture(scope="function")
def db(setup_database):
    """Provide a transactional database session that rolls back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


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


@pytest.fixture(scope="function")
def admin_user(db) -> User:
    """Create an admin user for testing."""
    user = User(
        email="admin@test.com",
        display_name="Test Admin",
        password_hash=hash_password("AdminPassword123"),
        role="admin",
        is_active=True,
    )
    db.add(user)
    db.flush()  # Flush to get ID without committing
    db.refresh(user)
    return user


@pytest.fixture(scope="function")
def regular_user(db) -> User:
    """Create a regular user for testing."""
    user = User(
        email="user@test.com",
        display_name="Test User",
        password_hash=hash_password("UserPassword123"),
        role="user",
        is_active=True,
    )
    db.add(user)
    db.flush()
    db.refresh(user)
    return user


@pytest.fixture(scope="function")
def admin_token(admin_user) -> str:
    """Get JWT token for admin user."""
    return create_access_token(
        data={"sub": admin_user.id, "email": admin_user.email, "role": admin_user.role}
    )


@pytest.fixture(scope="function")
def user_token(regular_user) -> str:
    """Get JWT token for regular user."""
    return create_access_token(
        data={"sub": regular_user.id, "email": regular_user.email, "role": regular_user.role}
    )


@pytest.fixture(scope="function")
def admin_headers(admin_token) -> dict:
    """Authorization headers for admin."""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture(scope="function")
def user_headers(user_token) -> dict:
    """Authorization headers for regular user."""
    return {"Authorization": f"Bearer {user_token}"}


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
