"""Tests for admin user seeding on startup."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_ready_rag.core.security import hash_password, verify_password
from ai_ready_rag.db.database import Base
from ai_ready_rag.db.models import SystemSetup, User
from ai_ready_rag.main import seed_admin_user


@pytest.fixture()
def seed_db():
    """Create a fresh in-memory DB and monkeypatch SessionLocal for seed tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    _TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _TestSession


@pytest.fixture(autouse=True)
def _patch_session_local(monkeypatch, seed_db):
    """Redirect SessionLocal in main module to our test DB."""
    monkeypatch.setattr("ai_ready_rag.main.SessionLocal", seed_db)


class TestSeedAdminUser:
    """Tests for the seed_admin_user() startup function."""

    def test_seed_creates_admin_when_none_exists(self, seed_db):
        """Empty DB -> seed creates admin with correct email, role, hashed password, must_reset_password."""
        seed_admin_user()

        session = seed_db()
        try:
            user = session.query(User).filter(User.role == "admin").first()
            assert user is not None
            assert user.email == "admin@test.com"
            assert user.role == "admin"
            assert user.is_active is True
            assert user.must_reset_password is True
            assert user.display_name == "Administrator"
            # Password should be hashed, not plaintext
            assert user.password_hash != "npassword"
            assert verify_password("npassword", user.password_hash)
        finally:
            session.close()

    def test_seed_creates_system_setup_record(self, seed_db):
        """Empty DB -> seed creates SystemSetup with setup_complete=False, admin_password_changed=False."""
        seed_admin_user()

        session = seed_db()
        try:
            setup = session.query(SystemSetup).first()
            assert setup is not None
            assert setup.setup_complete is False
            assert setup.admin_password_changed is False
        finally:
            session.close()

    def test_seed_is_idempotent(self, seed_db):
        """Admin already exists -> seed does nothing, no duplicate created."""
        # First call creates the admin
        seed_admin_user()
        # Second call should be a no-op
        seed_admin_user()

        session = seed_db()
        try:
            admin_count = session.query(User).filter(User.role == "admin").count()
            assert admin_count == 1
        finally:
            session.close()

    def test_seed_skips_when_admin_exists_different_email(self, seed_db):
        """Admin exists with different email than config -> seed does nothing (checks by role, not email)."""
        session = seed_db()
        try:
            # Pre-create an admin with a different email
            existing = User(
                email="other-admin@company.com",
                display_name="Other Admin",
                password_hash=hash_password("SomePassword123"),
                role="admin",
                is_active=True,
            )
            session.add(existing)
            session.commit()
        finally:
            session.close()

        # seed_admin_user should detect the existing admin by role and skip
        seed_admin_user()

        session2 = seed_db()
        try:
            admin_count = session2.query(User).filter(User.role == "admin").count()
            assert admin_count == 1
            admin = session2.query(User).filter(User.role == "admin").first()
            assert admin.email == "other-admin@company.com"
        finally:
            session2.close()
