"""CA module PII retention admin API endpoints.

Manages unit owner PII lifecycle:
- GET /api/ca/admin/pii-retention/status — current retention stats
- POST /api/ca/admin/pii-retention/purge — purge expired PII records
- GET /api/ca/admin/pii-retention/policy — current policy config
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ca/admin/pii-retention", tags=["ca-admin"])


class RetentionPolicy(BaseModel):
    retention_days: int = 365
    purge_on_soft_delete: bool = False
    require_admin_approval: bool = True


class RetentionStatus(BaseModel):
    total_unit_owners: int
    soft_deleted_count: int
    eligible_for_purge_count: int
    oldest_deleted_at: str | None
    retention_days: int


class PurgeRequest(BaseModel):
    retention_days: int = 365
    dry_run: bool = True  # default to dry run for safety


class PurgeResult(BaseModel):
    purged_count: int
    dry_run: bool
    cutoff_date: str
    message: str


def _require_admin(current_user: Any) -> Any:
    """Ensure the current user is an admin."""
    if getattr(current_user, "role", None) != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for PII retention operations",
        )
    return current_user


def _get_admin_user():
    """Dependency placeholder — wired at module load time via registry."""
    # This will be replaced with the actual get_current_user dependency
    # when the module registers its router
    pass


@router.get("/status", response_model=RetentionStatus)
async def get_retention_status(
    retention_days: int = 365,
) -> RetentionStatus:
    """Return PII retention statistics.

    Note: Full implementation requires DB session injection.
    Returns mock data until DB session dependency is wired.
    """
    return RetentionStatus(
        total_unit_owners=0,
        soft_deleted_count=0,
        eligible_for_purge_count=0,
        oldest_deleted_at=None,
        retention_days=retention_days,
    )


@router.post("/purge", response_model=PurgeResult)
async def purge_expired_pii(request: PurgeRequest) -> PurgeResult:
    """Purge PII from unit owner records deleted more than retention_days ago.

    By default runs as dry_run=True — set dry_run=False to actually delete.
    """
    cutoff = datetime.utcnow() - timedelta(days=request.retention_days)

    if request.dry_run:
        return PurgeResult(
            purged_count=0,
            dry_run=True,
            cutoff_date=cutoff.date().isoformat(),
            message=f"Dry run: would purge records deleted before {cutoff.date().isoformat()}. "
            "Set dry_run=false to execute.",
        )

    # Actual purge implementation requires DB session
    # Returns 0 until DB wired — production implementation in service layer
    return PurgeResult(
        purged_count=0,
        dry_run=False,
        cutoff_date=cutoff.date().isoformat(),
        message="PII purge executed (no DB session configured — 0 records purged).",
    )


@router.get("/policy", response_model=RetentionPolicy)
async def get_retention_policy() -> RetentionPolicy:
    """Return the current PII retention policy configuration."""
    return RetentionPolicy()
