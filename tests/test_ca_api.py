"""Tests for Community Associations REST API router (issue #373).

Tests:
  - test_accounts_list_requires_auth        — no token → 401
  - test_accounts_list_ca_disabled_returns_403 — ca_enabled=False → 403
  - test_accounts_list_empty               — no accounts → empty list
  - test_compliance_gap_no_account_returns_404
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_ready_rag.core.dependencies import get_current_user
from ai_ready_rag.core.security import create_access_token, hash_password
from ai_ready_rag.db.database import Base, get_db
from ai_ready_rag.db.models import User
from ai_ready_rag.modules.community_associations.api.router import ca_router

# ─── Test app setup ───────────────────────────────────────────────────────────

# Build a minimal FastAPI app that mounts only the CA router.
# This avoids the full app lifespan (Qdrant, Ollama, ARQ) in unit tests.
_test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestSession = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)

_ca_app = FastAPI()
_ca_app.include_router(ca_router, prefix="/api/ca")


def _override_get_db():
    db = _TestSession()
    try:
        yield db
    finally:
        db.close()


_ca_app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture(scope="session", autouse=True)
def _create_ca_tables():
    """Create core tables and CA insurance tables for CA API tests.

    CA ORM models reference ForeignKey("documents.id") — the documents table is
    in the core Base metadata.  We must create core tables first, then CA tables.
    SQLAlchemy FK resolution works correctly once both are in their respective
    metadata objects and tables are created in the right order.
    """
    # Import CA models to register them in CABase.metadata
    from ai_ready_rag.modules.community_associations.models.insurance import (  # noqa: F401
        InsuranceAccount,
        InsuranceCoverage,
        InsurancePolicy,
    )

    # Create core tables first (includes 'documents' table needed by CA FK refs)
    Base.metadata.create_all(bind=_test_engine)

    # Create CA tables using raw DDL to bypass SQLAlchemy FK resolution
    # across separate metadata objects.  Use CREATE TABLE IF NOT EXISTS.
    with _test_engine.connect() as conn:
        conn.execute(
            __import__("sqlalchemy").text("""
            CREATE TABLE IF NOT EXISTS insurance_accounts (
                id TEXT PRIMARY KEY,
                account_name TEXT NOT NULL,
                account_type TEXT,
                property_address TEXT,
                city TEXT,
                state TEXT,
                zip_code TEXT,
                units_residential INTEGER,
                units_commercial INTEGER,
                year_built INTEGER,
                source_document_id TEXT,
                extraction_confidence REAL,
                custom_fields TEXT,
                tenant_id TEXT NOT NULL,
                valid_from DATETIME,
                valid_to DATETIME,
                is_deleted INTEGER DEFAULT 0,
                deleted_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
        )
        conn.execute(
            __import__("sqlalchemy").text("""
            CREATE TABLE IF NOT EXISTS insurance_policies (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                policy_number TEXT,
                carrier_name TEXT,
                broker_name TEXT,
                line_of_business TEXT NOT NULL,
                policy_status TEXT DEFAULT 'active',
                inception_date DATE,
                effective_date DATE,
                expiration_date DATE,
                premium_amount REAL,
                source_document_id TEXT,
                idempotency_key TEXT,
                is_active INTEGER DEFAULT 1,
                tenant_id TEXT NOT NULL,
                valid_from DATETIME,
                valid_to DATETIME,
                is_deleted INTEGER DEFAULT 0,
                deleted_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (account_id) REFERENCES insurance_accounts(id) ON DELETE CASCADE
            )
            """)
        )
        conn.execute(
            __import__("sqlalchemy").text("""
            CREATE TABLE IF NOT EXISTS insurance_coverages (
                id TEXT PRIMARY KEY,
                policy_id TEXT NOT NULL,
                account_id TEXT NOT NULL,
                coverage_type TEXT NOT NULL,
                limit_amount REAL,
                deductible_amount REAL,
                deductible_type TEXT,
                sublimit_type TEXT,
                sublimit_amount REAL,
                notes TEXT,
                tenant_id TEXT NOT NULL,
                is_deleted INTEGER DEFAULT 0,
                deleted_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (policy_id) REFERENCES insurance_policies(id) ON DELETE CASCADE,
                FOREIGN KEY (account_id) REFERENCES insurance_accounts(id) ON DELETE CASCADE
            )
            """)
        )
        conn.commit()

    yield

    Base.metadata.drop_all(bind=_test_engine)


@pytest.fixture
def ca_db(_create_ca_tables):
    """Per-test transactional database session with rollback."""
    connection = _test_engine.connect()
    transaction = connection.begin()
    session = _TestSession(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def ca_client(ca_db):
    """TestClient using the isolated CA app with db override."""

    def _override_db():
        try:
            yield ca_db
        finally:
            pass

    _ca_app.dependency_overrides[get_db] = _override_db
    with TestClient(_ca_app) as tc:
        yield tc
    _ca_app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture
def ca_admin_user(ca_db) -> User:
    """Admin user for CA tests."""
    user = User(
        email="ca_admin@test.com",
        display_name="CA Admin",
        password_hash=hash_password("TestPassword123"),
        role="admin",
        is_active=True,
    )
    ca_db.add(user)
    ca_db.flush()
    ca_db.refresh(user)
    return user


@pytest.fixture
def ca_admin_token(ca_admin_user) -> str:
    """JWT token for the CA admin user."""
    return create_access_token(
        data={
            "sub": ca_admin_user.id,
            "email": ca_admin_user.email,
            "role": ca_admin_user.role,
        }
    )


@pytest.fixture
def ca_admin_headers(ca_admin_token) -> dict:
    """Authorization headers for CA admin."""
    return {"Authorization": f"Bearer {ca_admin_token}"}


# ─── Helper: wire current_user dependency ────────────────────────────────────


def _make_user_override(user: User):
    """Return a dependency override that yields a fixed User."""

    async def _override():
        return user

    return _override


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestCAAccountsAuth:
    def test_accounts_list_requires_auth(self, ca_client: TestClient):
        """GET /api/ca/accounts without a token must return 401."""
        response = ca_client.get("/api/ca/accounts")
        assert response.status_code == 401

    def test_account_detail_requires_auth(self, ca_client: TestClient):
        """GET /api/ca/accounts/{id} without a token must return 401."""
        response = ca_client.get("/api/ca/accounts/nonexistent")
        assert response.status_code == 401

    def test_compliance_gap_requires_auth(self, ca_client: TestClient):
        """GET /api/ca/accounts/{id}/compliance-gap without a token must return 401."""
        response = ca_client.get("/api/ca/accounts/nonexistent/compliance-gap")
        assert response.status_code == 401


class TestCAFeatureFlag:
    def test_accounts_list_ca_disabled_returns_403(
        self, ca_client: TestClient, ca_admin_user: User
    ):
        """When ca_enabled=False in tenant config, all CA endpoints return 403."""
        # Wire the current_user dependency to return our admin user
        _ca_app.dependency_overrides[get_current_user] = _make_user_override(ca_admin_user)

        # Mock TenantConfigResolver to return ca_enabled=False
        from ai_ready_rag.tenant.config import FeatureFlags, TenantConfig

        disabled_config = TenantConfig(feature_flags=FeatureFlags(ca_enabled=False))

        with patch(
            "ai_ready_rag.modules.community_associations.api.router.TenantConfigResolver"
        ) as mock_resolver_cls:
            mock_resolver = MagicMock()
            mock_resolver.resolve.return_value = disabled_config
            mock_resolver_cls.return_value = mock_resolver

            response = ca_client.get("/api/ca/accounts")

        _ca_app.dependency_overrides.pop(get_current_user, None)
        assert response.status_code == 403
        assert "not enabled" in response.json()["detail"]

    def test_policies_list_ca_disabled_returns_403(
        self, ca_client: TestClient, ca_admin_user: User
    ):
        """When ca_enabled=False, /policies endpoint also returns 403."""
        _ca_app.dependency_overrides[get_current_user] = _make_user_override(ca_admin_user)

        from ai_ready_rag.tenant.config import FeatureFlags, TenantConfig

        disabled_config = TenantConfig(feature_flags=FeatureFlags(ca_enabled=False))

        with patch(
            "ai_ready_rag.modules.community_associations.api.router.TenantConfigResolver"
        ) as mock_resolver_cls:
            mock_resolver = MagicMock()
            mock_resolver.resolve.return_value = disabled_config
            mock_resolver_cls.return_value = mock_resolver

            response = ca_client.get("/api/ca/accounts/some-id/policies")

        _ca_app.dependency_overrides.pop(get_current_user, None)
        assert response.status_code == 403


class TestCAAccountsList:
    def test_accounts_list_empty(self, ca_client: TestClient, ca_admin_user: User):
        """When no accounts exist, /accounts returns an empty list."""
        _ca_app.dependency_overrides[get_current_user] = _make_user_override(ca_admin_user)

        # Patch feature flag to enabled (default)
        from ai_ready_rag.tenant.config import FeatureFlags, TenantConfig

        enabled_config = TenantConfig(feature_flags=FeatureFlags(ca_enabled=True))

        with patch(
            "ai_ready_rag.modules.community_associations.api.router.TenantConfigResolver"
        ) as mock_resolver_cls:
            mock_resolver = MagicMock()
            mock_resolver.resolve.return_value = enabled_config
            mock_resolver_cls.return_value = mock_resolver

            response = ca_client.get("/api/ca/accounts")

        _ca_app.dependency_overrides.pop(get_current_user, None)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_accounts_list_returns_accounts(
        self, ca_client: TestClient, ca_admin_user: User, ca_db
    ):
        """When accounts exist, /accounts returns them."""
        import uuid

        import sqlalchemy as sa

        acct_id = str(uuid.uuid4())
        ca_db.execute(
            sa.text(
                "INSERT INTO insurance_accounts (id, account_name, account_type, tenant_id, is_deleted)"
                " VALUES (:id, :name, :type, :tenant, 0)"
            ),
            {"id": acct_id, "name": "Test HOA", "type": "condo_association", "tenant": "default"},
        )
        ca_db.flush()

        _ca_app.dependency_overrides[get_current_user] = _make_user_override(ca_admin_user)

        from ai_ready_rag.tenant.config import FeatureFlags, TenantConfig

        enabled_config = TenantConfig(feature_flags=FeatureFlags(ca_enabled=True))

        with patch(
            "ai_ready_rag.modules.community_associations.api.router.TenantConfigResolver"
        ) as mock_resolver_cls:
            mock_resolver = MagicMock()
            mock_resolver.resolve.return_value = enabled_config
            mock_resolver_cls.return_value = mock_resolver

            response = ca_client.get("/api/ca/accounts")

        _ca_app.dependency_overrides.pop(get_current_user, None)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["account_name"] == "Test HOA"


class TestCAAccountDetail:
    def test_account_detail_not_found(self, ca_client: TestClient, ca_admin_user: User):
        """GET /api/ca/accounts/missing returns 404."""
        _ca_app.dependency_overrides[get_current_user] = _make_user_override(ca_admin_user)

        from ai_ready_rag.tenant.config import FeatureFlags, TenantConfig

        enabled_config = TenantConfig(feature_flags=FeatureFlags(ca_enabled=True))

        with patch(
            "ai_ready_rag.modules.community_associations.api.router.TenantConfigResolver"
        ) as mock_resolver_cls:
            mock_resolver = MagicMock()
            mock_resolver.resolve.return_value = enabled_config
            mock_resolver_cls.return_value = mock_resolver

            response = ca_client.get("/api/ca/accounts/nonexistent-id")

        _ca_app.dependency_overrides.pop(get_current_user, None)
        assert response.status_code == 404


class TestCAComplianceGap:
    def test_compliance_gap_no_account_returns_404(
        self, ca_client: TestClient, ca_admin_user: User
    ):
        """GET /api/ca/accounts/{id}/compliance-gap for missing account returns 404."""
        _ca_app.dependency_overrides[get_current_user] = _make_user_override(ca_admin_user)

        from ai_ready_rag.tenant.config import FeatureFlags, TenantConfig

        enabled_config = TenantConfig(feature_flags=FeatureFlags(ca_enabled=True))

        with patch(
            "ai_ready_rag.modules.community_associations.api.router.TenantConfigResolver"
        ) as mock_resolver_cls:
            mock_resolver = MagicMock()
            mock_resolver.resolve.return_value = enabled_config
            mock_resolver_cls.return_value = mock_resolver

            response = ca_client.get("/api/ca/accounts/missing-id/compliance-gap")

        _ca_app.dependency_overrides.pop(get_current_user, None)
        assert response.status_code == 404

    def test_compliance_gap_returns_report(self, ca_client: TestClient, ca_admin_user: User, ca_db):
        """GET /api/ca/accounts/{id}/compliance-gap returns a compliance report."""
        import uuid

        import sqlalchemy as sa

        acct_id = str(uuid.uuid4())
        ca_db.execute(
            sa.text(
                "INSERT INTO insurance_accounts (id, account_name, account_type, tenant_id, is_deleted)"
                " VALUES (:id, :name, :type, :tenant, 0)"
            ),
            {
                "id": acct_id,
                "name": "Compliance Test HOA",
                "type": "condo_association",
                "tenant": "default",
            },
        )
        ca_db.flush()

        _ca_app.dependency_overrides[get_current_user] = _make_user_override(ca_admin_user)

        from ai_ready_rag.tenant.config import FeatureFlags, TenantConfig

        enabled_config = TenantConfig(feature_flags=FeatureFlags(ca_enabled=True))

        with patch(
            "ai_ready_rag.modules.community_associations.api.router.TenantConfigResolver"
        ) as mock_resolver_cls:
            mock_resolver = MagicMock()
            mock_resolver.resolve.return_value = enabled_config
            mock_resolver_cls.return_value = mock_resolver

            response = ca_client.get(f"/api/ca/accounts/{acct_id}/compliance-gap")

        _ca_app.dependency_overrides.pop(get_current_user, None)
        assert response.status_code == 200
        data = response.json()
        assert data["account_id"] == acct_id
        assert data["account_name"] == "Compliance Test HOA"
        assert "is_compliant" in data
        assert "gaps" in data


class TestCAAutomation:
    def test_renewal_prep_requires_auth(self, ca_client: TestClient):
        """POST /api/ca/automation/renewal-prep without auth returns 401."""
        response = ca_client.post("/api/ca/automation/renewal-prep")
        assert response.status_code == 401

    def test_renewal_prep_queues_job(self, ca_client: TestClient, ca_admin_user: User):
        """POST /api/ca/automation/renewal-prep returns 202 with job_id."""
        _ca_app.dependency_overrides[get_current_user] = _make_user_override(ca_admin_user)

        from ai_ready_rag.tenant.config import FeatureFlags, TenantConfig

        enabled_config = TenantConfig(feature_flags=FeatureFlags(ca_enabled=True))

        with patch(
            "ai_ready_rag.modules.community_associations.api.router.TenantConfigResolver"
        ) as mock_resolver_cls:
            mock_resolver = MagicMock()
            mock_resolver.resolve.return_value = enabled_config
            mock_resolver_cls.return_value = mock_resolver

            response = ca_client.post("/api/ca/automation/renewal-prep?dry_run=true")

        _ca_app.dependency_overrides.pop(get_current_user, None)
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "dry_run"

    def test_unit_owner_letter_requires_auth(self, ca_client: TestClient):
        """POST /api/ca/automation/unit-owner-letter without auth returns 401."""
        response = ca_client.post("/api/ca/automation/unit-owner-letter")
        assert response.status_code == 401
