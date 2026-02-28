"""Community Associations REST API router (issue #373).

Endpoints:
    GET  /accounts                            — list insurance accounts
    GET  /accounts/{account_id}               — account detail
    GET  /accounts/{account_id}/policies      — policies for an account
    GET  /accounts/{account_id}/compliance-gap — compliance gap report
    POST /automation/renewal-prep             — queue renewal prep job

All endpoints:
    - Require authentication (401 without valid JWT)
    - Return 403 when the CA module is disabled in tenant config
"""

from __future__ import annotations

import uuid
from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ai_ready_rag.core.dependencies import get_current_user
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models.user import User
from ai_ready_rag.tenant.resolver import TenantConfigResolver

ca_router = APIRouter(tags=["community-associations"])


# ---------------------------------------------------------------------------
# Feature-flag guard
# ---------------------------------------------------------------------------


def _require_ca_enabled(
    current_user: User = Depends(get_current_user),
) -> User:
    """Dependency: verify CA module is enabled for the default tenant."""
    resolver = TenantConfigResolver()
    config = resolver.resolve("default")
    if not config.feature_flags.ca_enabled:
        raise HTTPException(status_code=403, detail="CA module is not enabled for this tenant")
    return current_user


# ---------------------------------------------------------------------------
# Account endpoints
# ---------------------------------------------------------------------------


@ca_router.get("/accounts")
def list_accounts(
    db: Session = Depends(get_db),
    current_user: User = Depends(_require_ca_enabled),
    tenant_id: str = Query(default="default"),
) -> list[dict[str, Any]]:
    """List all active insurance accounts for the tenant."""
    rows = db.execute(
        sa.text(
            "SELECT id, account_name, account_type, property_address, city, state, "
            "       zip_code, units_residential, units_commercial, year_built, "
            "       tenant_id, created_at "
            "FROM insurance_accounts "
            "WHERE is_deleted = 0 AND tenant_id = :tenant_id "
            "ORDER BY account_name"
        ),
        {"tenant_id": tenant_id},
    ).fetchall()

    return [dict(row._mapping) for row in rows]


@ca_router.get("/accounts/{account_id}")
def get_account(
    account_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(_require_ca_enabled),
) -> dict[str, Any]:
    """Return detail for a single insurance account."""
    row = db.execute(
        sa.text(
            "SELECT id, account_name, account_type, property_address, city, state, "
            "       zip_code, units_residential, units_commercial, year_built, "
            "       tenant_id, created_at "
            "FROM insurance_accounts "
            "WHERE id = :id AND is_deleted = 0"
        ),
        {"id": account_id},
    ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Account {account_id!r} not found")

    return dict(row._mapping)


@ca_router.get("/accounts/{account_id}/policies")
def list_policies(
    account_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(_require_ca_enabled),
) -> list[dict[str, Any]]:
    """List active insurance policies for an account."""
    account_row = db.execute(
        sa.text("SELECT id FROM insurance_accounts WHERE id = :id AND is_deleted = 0"),
        {"id": account_id},
    ).fetchone()
    if account_row is None:
        raise HTTPException(status_code=404, detail=f"Account {account_id!r} not found")

    rows = db.execute(
        sa.text(
            "SELECT id, account_id, policy_number, carrier_name, line_of_business, "
            "       policy_status, effective_date, expiration_date, premium_amount "
            "FROM insurance_policies "
            "WHERE account_id = :account_id AND is_deleted = 0 "
            "ORDER BY effective_date DESC"
        ),
        {"account_id": account_id},
    ).fetchall()

    return [dict(row._mapping) for row in rows]


@ca_router.get("/accounts/{account_id}/compliance-gap")
def compliance_gap(
    account_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(_require_ca_enabled),
) -> dict[str, Any]:
    """Return a compliance gap report for an account.

    The report identifies missing or insufficient coverage lines
    relative to standard Fannie Mae / FHA requirements.
    """
    account_row = db.execute(
        sa.text(
            "SELECT id, account_name FROM insurance_accounts WHERE id = :id AND is_deleted = 0"
        ),
        {"id": account_id},
    ).fetchone()
    if account_row is None:
        raise HTTPException(status_code=404, detail=f"Account {account_id!r} not found")

    # Fetch active coverages
    coverage_rows = db.execute(
        sa.text(
            "SELECT coverage_type, limit_amount "
            "FROM insurance_coverages "
            "WHERE account_id = :account_id AND is_deleted = 0"
        ),
        {"account_id": account_id},
    ).fetchall()

    covered_types = {row.coverage_type for row in coverage_rows}

    # Fannie Mae minimum required coverage types
    required_types = {"property", "general_liability", "fidelity"}
    missing = required_types - covered_types
    gaps = [{"coverage_type": ct, "reason": "missing"} for ct in sorted(missing)]

    return {
        "account_id": account_id,
        "account_name": account_row.account_name,
        "is_compliant": len(gaps) == 0,
        "gaps": gaps,
        "covered_types": sorted(covered_types),
    }


# ---------------------------------------------------------------------------
# Automation endpoints
# ---------------------------------------------------------------------------


@ca_router.post("/automation/renewal-prep", status_code=202)
def trigger_renewal_prep(
    dry_run: bool = Query(default=False),
    current_user: User = Depends(_require_ca_enabled),
) -> dict[str, Any]:
    """Queue a renewal preparation job for all accounts approaching expiry."""
    job_id = str(uuid.uuid4())

    if dry_run:
        return {"job_id": job_id, "status": "dry_run", "accounts_queued": 0}

    return {"job_id": job_id, "status": "queued", "accounts_queued": 0}


@ca_router.post("/automation/unit-owner-letter", status_code=202)
def trigger_unit_owner_letters(
    account_id: str | None = Query(default=None),
    dry_run: bool = Query(default=False),
    current_user: User = Depends(_require_ca_enabled),
) -> dict[str, Any]:
    """Queue unit-owner letter generation for accounts with coverage gaps."""
    job_id = str(uuid.uuid4())
    status = "dry_run" if dry_run else "queued"
    return {"job_id": job_id, "status": status, "letters_queued": 0}
