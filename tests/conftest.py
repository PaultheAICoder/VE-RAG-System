"""Pytest configuration and fixtures."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_ready_rag.main import app
from ai_ready_rag.db.database import Base, get_db
from ai_ready_rag.db.models import User, Tag
from ai_ready_rag.core.security import hash_password, create_access_token


# Test database - in-memory SQLite
TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session")
def db():
    """Create fresh database for each test."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="session")
def client(db):
    """Create test client with database override."""
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=engine)

    with TestClient(app) as test_client:
        yield test_client

    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def admin_user(db) -> User:
    """Create an admin user for testing."""
    user = User(
        email="admin@test.com",
        display_name="Test Admin",
        password_hash=hash_password("AdminPassword123"),
        role="admin",
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture(scope="session")
def regular_user(db) -> User:
    """Create a regular user for testing."""
    user = User(
        email="user@test.com",
        display_name="Test User",
        password_hash=hash_password("UserPassword123"),
        role="user",
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def admin_token(admin_user) -> str:
    """Get JWT token for admin user."""
    return create_access_token(data={
        "sub": admin_user.id,
        "email": admin_user.email,
        "role": admin_user.role
    })


@pytest.fixture
def user_token(regular_user) -> str:
    """Get JWT token for regular user."""
    return create_access_token(data={
        "sub": regular_user.id,
        "email": regular_user.email,
        "role": regular_user.role
    })


@pytest.fixture
def admin_headers(admin_token) -> dict:
    """Authorization headers for admin."""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def user_headers(user_token) -> dict:
    """Authorization headers for regular user."""
    return {"Authorization": f"Bearer {user_token}"}


@pytest.fixture
def sample_tag(db, admin_user) -> Tag:
    """Create a sample tag."""
    tag = Tag(
        name="hr",
        display_name="Human Resources",
        description="HR department documents",
        color="#10B981",
        created_by=admin_user.id
    )
    db.add(tag)
    db.commit()
    db.refresh(tag)
    return tag
