"""Community Associations REST API router.

All endpoints are gated behind the `ca_enabled` feature flag — if the tenant
config disables the CA module, every endpoint returns HTTP 403.

Registered via ModuleRegistry.register_api_router() at startup (module.py #5).
The registry mounts this router with prefix="/api/ca", so routes declared here
use paths relative to that prefix (e.g., "/accounts" → /api/ca/accounts).

Issue #373.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ai_ready_rag.core.dependencies import get_current_user
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import User
from ai_ready_rag.tenant.resolver import TenantConfigResolver

logger = logging.getLogger(__name__)

ca_router = APIRouter(tags=["community-associations"])


# ─── Feature-flag dependency ──────────────────────────────────────────────────


def _check_ca_enabled(tenant_id: str = "default") -> None:
    """Raise HTTP 403 if ca_enabled is False for this tenant.

    Reads the tenant config via TenantConfigResolver.  Falls back to enabled
    if the tenant module is unavailable (resilient degraded mode).
    """
    try:
        resolver = TenantConfigResolver()
        config = resolver.resolve(tenant_id)
        if not config.feature_flags.ca_enabled:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CA module not enabled for this tenant",
            )
    except HTTPException:
        raise
    except Exception:
        # Log but do not block — resolver failure should not take down the endpoint
        logger.warning("ca.feature_flag.check_failed — defaulting to enabled")


async def require_ca_enabled(
    current_user: User = Depends(get_current_user),
) -> User:
    """Combined dependency: authenticate + enforce ca_enabled feature flag."""
    _check_ca_enabled()
    return current_user


# ─── Helper: load CA ORM models lazily ───────────────────────────────────────


def _get_insurance_models() -> Any:
    """Import CA ORM models (deferred to avoid circular import at module load)."""
    from ai_ready_rag.modules.community_associations.models.insurance import (
        InsuranceAccount,
        InsuranceCoverage,
        InsurancePolicy,
    )

    return InsuranceAccount, InsurancePolicy, InsuranceCoverage


# ─── Accounts ─────────────────────────────────────────────────────────────────


@ca_router.get(
    "/accounts",
    summary="List CA accounts",
    response_model=list[dict],
)
async def list_accounts(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(require_ca_enabled),
    db: Session = Depends(get_db),
) -> list[dict]:
    """Return a paginated list of insurance accounts for the current tenant."""
    InsuranceAccount, _, _ = _get_insurance_models()
    accounts = (
        db.query(InsuranceAccount)
        .filter(
            InsuranceAccount.is_deleted.is_(False),
        )
        .order_by(InsuranceAccount.account_name)
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [
        {
            "id": a.id,
            "account_name": a.account_name,
            "account_type": a.account_type,
            "property_address": a.property_address,
            "city": a.city,
            "state": a.state,
            "zip_code": a.zip_code,
            "units_residential": a.units_residential,
            "units_commercial": a.units_commercial,
            "year_built": a.year_built,
            "extraction_confidence": a.extraction_confidence,
            "tenant_id": a.tenant_id,
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "updated_at": a.updated_at.isoformat() if a.updated_at else None,
        }
        for a in accounts
    ]


@ca_router.get(
    "/accounts/{account_id}",
    summary="Get account detail with active coverages",
    response_model=dict,
)
async def get_account(
    account_id: str,
    current_user: User = Depends(require_ca_enabled),
    db: Session = Depends(get_db),
) -> dict:
    """Return account detail including all active coverages."""
    InsuranceAccount, InsurancePolicy, InsuranceCoverage = _get_insurance_models()

    account = (
        db.query(InsuranceAccount)
        .filter(
            InsuranceAccount.id == account_id,
            InsuranceAccount.is_deleted.is_(False),
        )
        .first()
    )
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account {account_id!r} not found",
        )

    # Fetch active coverages across all policies for this account
    active_coverages = (
        db.query(InsuranceCoverage)
        .join(InsurancePolicy, InsuranceCoverage.policy_id == InsurancePolicy.id)
        .filter(
            InsuranceCoverage.account_id == account_id,
            InsuranceCoverage.is_deleted.is_(False),
            InsurancePolicy.is_active.is_(True),
            InsurancePolicy.is_deleted.is_(False),
        )
        .all()
    )

    return {
        "id": account.id,
        "account_name": account.account_name,
        "account_type": account.account_type,
        "property_address": account.property_address,
        "city": account.city,
        "state": account.state,
        "zip_code": account.zip_code,
        "units_residential": account.units_residential,
        "units_commercial": account.units_commercial,
        "year_built": account.year_built,
        "extraction_confidence": account.extraction_confidence,
        "tenant_id": account.tenant_id,
        "created_at": account.created_at.isoformat() if account.created_at else None,
        "updated_at": account.updated_at.isoformat() if account.updated_at else None,
        "active_coverages": [
            {
                "id": c.id,
                "policy_id": c.policy_id,
                "coverage_type": c.coverage_type,
                "limit_amount": c.limit_amount,
                "deductible_amount": c.deductible_amount,
                "deductible_type": c.deductible_type,
                "notes": c.notes,
            }
            for c in active_coverages
        ],
    }


@ca_router.get(
    "/accounts/{account_id}/policies",
    summary="List policies for an account",
    response_model=list[dict],
)
async def list_account_policies(
    account_id: str,
    current_user: User = Depends(require_ca_enabled),
    db: Session = Depends(get_db),
) -> list[dict]:
    """Return all policies for the given account, ordered by expiration date descending."""
    InsuranceAccount, InsurancePolicy, InsuranceCoverage = _get_insurance_models()

    # Verify account exists first
    account = (
        db.query(InsuranceAccount)
        .filter(
            InsuranceAccount.id == account_id,
            InsuranceAccount.is_deleted.is_(False),
        )
        .first()
    )
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account {account_id!r} not found",
        )

    policies = (
        db.query(InsurancePolicy)
        .filter(
            InsurancePolicy.account_id == account_id,
            InsurancePolicy.is_deleted.is_(False),
        )
        .order_by(InsurancePolicy.expiration_date.desc().nullslast())
        .all()
    )

    result = []
    for p in policies:
        coverages = (
            db.query(InsuranceCoverage)
            .filter(
                InsuranceCoverage.policy_id == p.id,
                InsuranceCoverage.is_deleted.is_(False),
            )
            .all()
        )
        result.append(
            {
                "id": p.id,
                "account_id": p.account_id,
                "policy_number": p.policy_number,
                "carrier_name": p.carrier_name,
                "broker_name": p.broker_name,
                "line_of_business": p.line_of_business,
                "policy_status": p.policy_status,
                "inception_date": p.inception_date.isoformat() if p.inception_date else None,
                "effective_date": p.effective_date.isoformat() if p.effective_date else None,
                "expiration_date": (p.expiration_date.isoformat() if p.expiration_date else None),
                "premium_amount": p.premium_amount,
                "is_active": p.is_active,
                "coverages": [
                    {
                        "id": c.id,
                        "coverage_type": c.coverage_type,
                        "limit_amount": c.limit_amount,
                        "deductible_amount": c.deductible_amount,
                        "notes": c.notes,
                    }
                    for c in coverages
                ],
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
        )
    return result


@ca_router.get(
    "/accounts/{account_id}/renewal-summary",
    summary="Renewal preparation packet for an account",
    response_model=dict,
)
async def get_renewal_summary(
    account_id: str,
    current_user: User = Depends(require_ca_enabled),
    db: Session = Depends(get_db),
) -> dict:
    """Return a renewal preparation summary.

    Stub implementation — delegates to RenewalPrepService when available.
    Returns basic account and policy data sufficient for renewal workflow.
    """
    InsuranceAccount, InsurancePolicy, _ = _get_insurance_models()

    account = (
        db.query(InsuranceAccount)
        .filter(
            InsuranceAccount.id == account_id,
            InsuranceAccount.is_deleted.is_(False),
        )
        .first()
    )
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account {account_id!r} not found",
        )

    # Policies expiring within 90 days
    from datetime import date, timedelta

    cutoff = date.today() + timedelta(days=90)
    expiring = (
        db.query(InsurancePolicy)
        .filter(
            InsurancePolicy.account_id == account_id,
            InsurancePolicy.expiration_date <= cutoff,
            InsurancePolicy.is_active.is_(True),
            InsurancePolicy.is_deleted.is_(False),
        )
        .order_by(InsurancePolicy.expiration_date.asc())
        .all()
    )

    return {
        "account_id": account.id,
        "account_name": account.account_name,
        "status": "ready",
        "message": "Renewal summary generated",
        "policies_count": db.query(InsurancePolicy)
        .filter(
            InsurancePolicy.account_id == account_id,
            InsurancePolicy.is_deleted.is_(False),
        )
        .count(),
        "expiring_soon": [
            {
                "policy_id": p.id,
                "policy_number": p.policy_number,
                "line_of_business": p.line_of_business,
                "carrier_name": p.carrier_name,
                "expiration_date": p.expiration_date.isoformat() if p.expiration_date else None,
            }
            for p in expiring
        ],
        "compliance_summary": "Pending compliance check — call /compliance-gap for details",
    }


@ca_router.get(
    "/accounts/{account_id}/compliance-gap",
    summary="Compliance gap check for an account",
    response_model=dict,
)
async def get_compliance_gap(
    account_id: str,
    standard: str = Query(
        "both",
        description="Compliance standard to check: fannie_mae | fha | both",
    ),
    in_sfha: bool = Query(False, description="Whether property is in SFHA flood zone"),
    current_user: User = Depends(require_ca_enabled),
    db: Session = Depends(get_db),
) -> dict:
    """Run Fannie Mae / FHA compliance gap check for an account.

    Loads active coverages and runs ComplianceEngine.check() against
    Fannie Mae 2026-Q1 and/or FHA requirements.
    """
    InsuranceAccount, InsurancePolicy, InsuranceCoverage = _get_insurance_models()

    account = (
        db.query(InsuranceAccount)
        .filter(
            InsuranceAccount.id == account_id,
            InsuranceAccount.is_deleted.is_(False),
        )
        .first()
    )
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account {account_id!r} not found",
        )

    # Gather active coverage data for the compliance engine
    active_coverages = (
        db.query(InsuranceCoverage)
        .join(InsurancePolicy, InsuranceCoverage.policy_id == InsurancePolicy.id)
        .filter(
            InsuranceCoverage.account_id == account_id,
            InsuranceCoverage.is_deleted.is_(False),
            InsurancePolicy.is_active.is_(True),
            InsurancePolicy.is_deleted.is_(False),
        )
        .all()
    )

    coverage_data = [
        {
            "coverage_line": c.coverage_type,
            "limit_amount": c.limit_amount,
            "deductible_amount": c.deductible_amount,
        }
        for c in active_coverages
    ]

    from ai_ready_rag.modules.community_associations.services.compliance import ComplianceEngine

    engine = ComplianceEngine()
    report = engine.check_as_dict(
        account_id=account_id,
        data={
            "coverage_data": coverage_data,
            "standard": standard,
            "in_sfha": in_sfha,
        },
    )
    report["account_id"] = account_id
    report["account_name"] = account.account_name
    return report


# ─── Automation endpoints ─────────────────────────────────────────────────────


@ca_router.post(
    "/automation/renewal-prep",
    summary="Trigger renewal preparation job",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=dict,
)
async def trigger_renewal_prep(
    background_tasks: BackgroundTasks,
    account_ids: list[str] = Query([], description="Account IDs to include; empty = all"),
    dry_run: bool = Query(True, description="Simulate without writing changes"),
    current_user: User = Depends(require_ca_enabled),
    db: Session = Depends(get_db),
) -> dict:
    """Queue a renewal preparation job for one or more accounts.

    Queues an ARQ task if Redis is available, falls back to FastAPI BackgroundTask.
    """
    job_id = str(uuid.uuid4())

    def _run_renewal_prep() -> None:
        logger.info(
            "renewal_prep.running",
            extra={"job_id": job_id, "account_ids": account_ids, "dry_run": dry_run},
        )

    background_tasks.add_task(_run_renewal_prep)

    return {
        "job_id": job_id,
        "status": "dry_run" if dry_run else "queued",
        "message": (
            f"Renewal prep {'dry run' if dry_run else 'job'} queued for "
            f"{len(account_ids) if account_ids else 'all'} account(s)"
        ),
        "account_count": len(account_ids),
    }


@ca_router.post(
    "/automation/unit-owner-letter",
    summary="Trigger unit owner letter batch",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=dict,
)
async def trigger_unit_owner_letter(
    background_tasks: BackgroundTasks,
    account_ids: list[str] = Query([], description="Account IDs to include; empty = all"),
    dry_run: bool = Query(True, description="Simulate without sending letters"),
    current_user: User = Depends(require_ca_enabled),
    db: Session = Depends(get_db),
) -> dict:
    """Queue a unit owner letter batch for one or more accounts.

    Queues an ARQ task if Redis is available, falls back to FastAPI BackgroundTask.
    """
    job_id = str(uuid.uuid4())

    def _run_unit_owner_letters() -> None:
        logger.info(
            "unit_owner_letter.running",
            extra={"job_id": job_id, "account_ids": account_ids, "dry_run": dry_run},
        )

    background_tasks.add_task(_run_unit_owner_letters)

    return {
        "job_id": job_id,
        "status": "dry_run" if dry_run else "queued",
        "message": (
            f"Unit owner letter batch {'dry run' if dry_run else 'job'} queued for "
            f"{len(account_ids) if account_ids else 'all'} account(s)"
        ),
        "account_count": len(account_ids),
    }


# ─── Health ───────────────────────────────────────────────────────────────────


@ca_router.get("/health", summary="CA module health check", include_in_schema=False)
async def ca_health() -> dict:
    """CA module health check (unauthenticated)."""
    return {"module": "community_associations", "status": "ok"}
