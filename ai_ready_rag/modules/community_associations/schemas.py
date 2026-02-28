"""Pydantic response schemas for the Community Associations REST API.

These models are used as FastAPI response_model declarations and are derived
from the SQLAlchemy ORM models in models/insurance.py.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

# ─── Coverage ─────────────────────────────────────────────────────────────────


class CoverageResponse(BaseModel):
    """Per-limit row under a policy."""

    id: str
    policy_id: str
    account_id: str
    coverage_type: str
    limit_amount: float | None = None
    deductible_amount: float | None = None
    deductible_type: str | None = None
    sublimit_type: str | None = None
    sublimit_amount: float | None = None
    notes: str | None = None
    created_at: datetime | None = None

    class Config:
        from_attributes = True


# ─── Policy ───────────────────────────────────────────────────────────────────


class PolicyResponse(BaseModel):
    """Insurance policy for an account."""

    id: str
    account_id: str
    policy_number: str | None = None
    carrier_name: str | None = None
    broker_name: str | None = None
    line_of_business: str
    policy_status: str | None = None
    inception_date: date | None = None
    effective_date: date | None = None
    expiration_date: date | None = None
    premium_amount: float | None = None
    is_active: bool | None = None
    coverages: list[CoverageResponse] = []
    created_at: datetime | None = None
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


# ─── Account ──────────────────────────────────────────────────────────────────


class AccountResponse(BaseModel):
    """Insurance account (one per insured HOA / community association)."""

    id: str
    account_name: str
    account_type: str | None = None
    property_address: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    units_residential: int | None = None
    units_commercial: int | None = None
    year_built: int | None = None
    extraction_confidence: float | None = None
    tenant_id: str
    created_at: datetime | None = None
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class AccountDetailResponse(AccountResponse):
    """Account with active coverages across all policies."""

    active_coverages: list[CoverageResponse] = []


# ─── Compliance gap ───────────────────────────────────────────────────────────


class CoverageGapSchema(BaseModel):
    """A single compliance gap detected for a coverage line."""

    coverage_line: str
    gap_type: str  # "missing" | "below_minimum" | "missing_required_clause"
    detail: str
    standard: str  # "fannie_mae" | "fha" | "both"
    severity: str  # "high" | "medium" | "low"


class ComplianceGapResponse(BaseModel):
    """Result of a compliance gap check for an account."""

    account_id: str
    account_name: str
    standard: str
    is_compliant: bool
    gap_count: int
    high_severity_count: int
    gaps: list[CoverageGapSchema] = []
    warnings: list[str] = []
    checked_lines: list[str] = []
    notes: str = ""


# ─── Renewal summary ──────────────────────────────────────────────────────────


class RenewalSummaryResponse(BaseModel):
    """Renewal preparation packet for an account (stub — populated by RenewalPrepService)."""

    account_id: str
    account_name: str
    status: str  # "pending" | "ready" | "error"
    message: str = ""
    policies_count: int = 0
    expiring_soon: list[dict[str, Any]] = []  # policies expiring within 90 days
    compliance_summary: str = ""


# ─── Automation jobs ──────────────────────────────────────────────────────────


class AutomationJobRequest(BaseModel):
    """Request body for triggering an automation job."""

    account_ids: list[str] = []  # empty → all accounts
    dry_run: bool = True
    options: dict[str, Any] = {}


class AutomationJobResponse(BaseModel):
    """Response after queuing an automation job."""

    job_id: str
    status: str  # "queued" | "running" | "dry_run"
    message: str
    account_count: int = 0
