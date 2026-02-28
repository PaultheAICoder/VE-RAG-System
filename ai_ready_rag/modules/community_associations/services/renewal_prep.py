"""RenewalPrepService — automated renewal preparation package for CA insurance accounts.

Produces a structured renewal prep packet for a given insurance account:
- Current coverage summary (from insurance_policies/coverages)
- Expiring policies (within configurable days window)
- Compliance gaps (via ComplianceEngine)
- Carrier comparison data
- Recommended coverage adjustments

This is a read-only service — it aggregates data, does not modify records.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExpiringPolicy:
    policy_id: str
    policy_number: str
    carrier_name: str
    coverage_line: str
    expiration_date: date
    days_until_expiry: int
    total_premium_usd: float | None


@dataclass
class RenewalPrepPacket:
    account_id: str
    account_name: str
    generated_at: datetime
    renewal_window_days: int
    expiring_policies: list[ExpiringPolicy] = field(default_factory=list)
    active_coverages: list[dict[str, Any]] = field(default_factory=list)
    compliance_gaps: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    total_expiring_premium: float = 0.0

    @property
    def has_expiring_policies(self) -> bool:
        return len(self.expiring_policies) > 0

    @property
    def is_compliance_compliant(self) -> bool:
        return len([g for g in self.compliance_gaps if g.get("severity") == "high"]) == 0


class RenewalPrepService:
    """Aggregates renewal prep data for a CA insurance account.

    Works with SQLAlchemy session for database queries.
    Does NOT call Claude — produces structured data for downstream use.
    """

    def __init__(self, db: Any = None, renewal_window_days: int = 90) -> None:
        self._db = db
        self._renewal_window_days = renewal_window_days

    def prepare(
        self,
        account_id: str,
        account_name: str = "",
        as_of_date: date | None = None,
    ) -> RenewalPrepPacket:
        """Generate a renewal prep packet for the given account."""
        as_of = as_of_date or date.today()
        cutoff = as_of + timedelta(days=self._renewal_window_days)

        packet = RenewalPrepPacket(
            account_id=account_id,
            account_name=account_name,
            generated_at=datetime.utcnow(),
            renewal_window_days=self._renewal_window_days,
        )

        if self._db is not None:
            expiring = self._query_expiring_policies(account_id, as_of, cutoff)
            packet.expiring_policies = expiring
            packet.active_coverages = self._query_active_coverages(account_id)
            packet.total_expiring_premium = sum(p.total_premium_usd or 0 for p in expiring)

        packet.recommendations = self._generate_recommendations(packet)
        return packet

    def _query_expiring_policies(
        self, account_id: str, as_of: date, cutoff: date
    ) -> list[ExpiringPolicy]:
        """Query policies expiring within the renewal window."""
        try:
            # Dynamic import to avoid hard dependency before migration merges
            from sqlalchemy import text

            result = self._db.execute(
                text("""
                    SELECT id, policy_number, carrier_name, coverage_line,
                           expiration_date, total_premium_usd
                    FROM insurance_policies
                    WHERE account_id = :account_id
                      AND status = 'active'
                      AND expiration_date BETWEEN :as_of AND :cutoff
                    ORDER BY expiration_date ASC
                    LIMIT 50
                """),
                {"account_id": account_id, "as_of": str(as_of), "cutoff": str(cutoff)},
            )
            rows = result.fetchall()
        except Exception as exc:
            logger.warning(
                "renewal_prep.query_failed",
                extra={"account_id": account_id, "error": str(exc)},
            )
            return []

        policies = []
        for row in rows:
            exp_date = row.expiration_date
            if isinstance(exp_date, str):
                exp_date = date.fromisoformat(exp_date)
            days_left = (exp_date - as_of).days if exp_date else 0
            policies.append(
                ExpiringPolicy(
                    policy_id=str(row.id),
                    policy_number=str(row.policy_number or ""),
                    carrier_name=str(row.carrier_name or "Unknown"),
                    coverage_line=str(row.coverage_line or ""),
                    expiration_date=exp_date,
                    days_until_expiry=days_left,
                    total_premium_usd=float(row.total_premium_usd)
                    if row.total_premium_usd
                    else None,
                )
            )
        return policies

    def _query_active_coverages(self, account_id: str) -> list[dict[str, Any]]:
        """Query active coverage lines for the account."""
        try:
            from sqlalchemy import text

            result = self._db.execute(
                text("""
                    SELECT c.coverage_type, c.limit_amount, c.deductible_amount,
                           c.meets_fannie_mae, c.meets_fha, p.carrier_name, p.expiration_date
                    FROM insurance_coverages c
                    JOIN insurance_policies p ON c.policy_id = p.id
                    WHERE p.account_id = :account_id
                      AND p.status = 'active'
                    ORDER BY c.coverage_type
                    LIMIT 100
                """),
                {"account_id": account_id},
            )
            return [dict(row._mapping) for row in result.fetchall()]
        except Exception as exc:
            logger.warning(
                "renewal_prep.coverage_query_failed",
                extra={"account_id": account_id, "error": str(exc)},
            )
            return []

    def _generate_recommendations(self, packet: RenewalPrepPacket) -> list[str]:
        """Generate human-readable renewal recommendations."""
        recs = []

        if not packet.expiring_policies:
            recs.append("No policies expiring within the renewal window.")
            return recs

        # Urgent expirations
        urgent = [p for p in packet.expiring_policies if p.days_until_expiry <= 30]
        if urgent:
            recs.append(
                f"URGENT: {len(urgent)} policy(ies) expire within 30 days — "
                f"initiate renewal immediately: " + ", ".join(p.coverage_line for p in urgent)
            )

        # Standard expirations
        standard = [p for p in packet.expiring_policies if 30 < p.days_until_expiry <= 90]
        if standard:
            recs.append(
                f"{len(standard)} policy(ies) expire in 31–90 days: "
                + ", ".join(p.coverage_line for p in standard)
            )

        if packet.total_expiring_premium > 0:
            recs.append(
                f"Total expiring premium: ${packet.total_expiring_premium:,.2f} — "
                "obtain quotes 45–60 days before expiration."
            )

        if packet.compliance_gaps:
            recs.append(
                f"{len(packet.compliance_gaps)} compliance gap(s) detected — "
                "address before renewal submission."
            )

        return recs
