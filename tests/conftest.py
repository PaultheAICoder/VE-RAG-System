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
def customer_admin_user(db) -> User:
    """Create a customer admin user for testing."""
    user = User(
        email="customer_admin@test.com",
        display_name="Customer Admin",
        password_hash=hash_password("CustomerAdminPassword123"),
        role="customer_admin",
        is_active=True,
    )
    db.add(user)
    db.flush()
    db.refresh(user)
    return user


@pytest.fixture(scope="function")
def customer_admin_token(customer_admin_user) -> str:
    """Get JWT token for customer admin user."""
    return create_access_token(
        data={
            "sub": customer_admin_user.id,
            "email": customer_admin_user.email,
            "role": customer_admin_user.role,
        }
    )


@pytest.fixture(scope="function")
def customer_admin_headers(customer_admin_token) -> dict:
    """Authorization headers for customer admin."""
    return {"Authorization": f"Bearer {customer_admin_token}"}


@pytest.fixture(scope="function")
def system_admin_user(db) -> User:
    """Create a system admin user for testing (explicit system_admin role)."""
    user = User(
        email="system_admin@test.com",
        display_name="System Admin",
        password_hash=hash_password("SystemAdminPassword123"),
        role="system_admin",
        is_active=True,
    )
    db.add(user)
    db.flush()
    db.refresh(user)
    return user


@pytest.fixture(scope="function")
def system_admin_token(system_admin_user) -> str:
    """Get JWT token for system admin user."""
    return create_access_token(
        data={
            "sub": system_admin_user.id,
            "email": system_admin_user.email,
            "role": system_admin_user.role,
        }
    )


@pytest.fixture(scope="function")
def system_admin_headers(system_admin_token) -> dict:
    """Authorization headers for system admin."""
    return {"Authorization": f"Bearer {system_admin_token}"}


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
