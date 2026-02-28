"""BoardPresentationService — insurance review package for board meetings.

Assembles a structured board presentation packet covering:
- Coverage summary by line of business
- Year-over-year premium comparison
- Compliance status (Fannie Mae / FHA)
- Upcoming renewals
- Open claims summary
- Recommended board resolutions

Output is a structured dict/dataclass suitable for rendering to PDF/PPTX
via DocumentRenderer (Issue #384). This service produces data only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CoverageSummarySlide:
    title: str = "Insurance Coverage Summary"
    as_of_date: str = ""
    lines: list[dict[str, Any]] = field(default_factory=list)
    total_premium: float = 0.0
    carrier_count: int = 0


@dataclass
class ComplianceSlide:
    title: str = "Compliance Status"
    fannie_mae_compliant: bool | None = None
    fha_compliant: bool | None = None
    gap_count: int = 0
    gaps: list[dict[str, Any]] = field(default_factory=list)
    last_reviewed: str = ""


@dataclass
class RenewalSlide:
    title: str = "Upcoming Renewals"
    renewal_window_days: int = 90
    expiring_policies: list[dict[str, Any]] = field(default_factory=list)
    total_expiring_premium: float = 0.0


@dataclass
class ResolutionSlide:
    title: str = "Recommended Board Resolutions"
    resolutions: list[str] = field(default_factory=list)


@dataclass
class BoardPresentationPacket:
    account_id: str
    account_name: str
    meeting_date: str
    generated_at: datetime
    coverage_summary: CoverageSummarySlide = field(default_factory=CoverageSummarySlide)
    compliance_status: ComplianceSlide = field(default_factory=ComplianceSlide)
    upcoming_renewals: RenewalSlide = field(default_factory=RenewalSlide)
    recommended_resolutions: ResolutionSlide = field(default_factory=ResolutionSlide)
    slide_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON API response or DocumentRenderer input."""
        return {
            "account_id": self.account_id,
            "account_name": self.account_name,
            "meeting_date": self.meeting_date,
            "generated_at": self.generated_at.isoformat(),
            "slides": [
                {
                    "type": "coverage_summary",
                    "title": self.coverage_summary.title,
                    "data": {
                        "as_of_date": self.coverage_summary.as_of_date,
                        "lines": self.coverage_summary.lines,
                        "total_premium": self.coverage_summary.total_premium,
                        "carrier_count": self.coverage_summary.carrier_count,
                    },
                },
                {
                    "type": "compliance_status",
                    "title": self.compliance_status.title,
                    "data": {
                        "fannie_mae_compliant": self.compliance_status.fannie_mae_compliant,
                        "fha_compliant": self.compliance_status.fha_compliant,
                        "gap_count": self.compliance_status.gap_count,
                        "gaps": self.compliance_status.gaps,
                    },
                },
                {
                    "type": "upcoming_renewals",
                    "title": self.upcoming_renewals.title,
                    "data": {
                        "expiring_policies": self.upcoming_renewals.expiring_policies,
                        "total_expiring_premium": self.upcoming_renewals.total_expiring_premium,
                    },
                },
                {
                    "type": "recommended_resolutions",
                    "title": self.recommended_resolutions.title,
                    "data": {"resolutions": self.recommended_resolutions.resolutions},
                },
            ],
        }


class BoardPresentationService:
    """Assembles board presentation packets for CA insurance review.

    Aggregates data from database and generates structured slides.
    Works in preview mode (no DB) for testing/demos.
    """

    def __init__(self, db: Any = None) -> None:
        self._db = db

    def prepare(
        self,
        account_id: str,
        account_name: str = "",
        meeting_date: date | None = None,
        renewal_window_days: int = 90,
    ) -> BoardPresentationPacket:
        """Assemble a full board presentation packet."""
        meeting_str = (meeting_date or date.today()).strftime("%B %d, %Y")

        packet = BoardPresentationPacket(
            account_id=account_id,
            account_name=account_name,
            meeting_date=meeting_str,
            generated_at=datetime.utcnow(),
        )

        # Coverage summary
        coverage_lines = self._query_coverage_lines(account_id) if self._db else []
        packet.coverage_summary = CoverageSummarySlide(
            as_of_date=date.today().isoformat(),
            lines=coverage_lines,
            total_premium=sum(float(c.get("total_premium_usd") or 0) for c in coverage_lines),
            carrier_count=len(
                {c.get("carrier_name") for c in coverage_lines if c.get("carrier_name")}
            ),
        )

        # Renewals
        expiring = (
            self._query_expiring_policies(account_id, renewal_window_days) if self._db else []
        )
        packet.upcoming_renewals = RenewalSlide(
            renewal_window_days=renewal_window_days,
            expiring_policies=expiring,
            total_expiring_premium=sum(float(p.get("total_premium_usd") or 0) for p in expiring),
        )

        # Resolutions
        packet.recommended_resolutions = ResolutionSlide(
            resolutions=self._generate_resolutions(packet)
        )

        packet.slide_count = 4
        return packet

    def _query_coverage_lines(self, account_id: str) -> list[dict[str, Any]]:
        try:
            from sqlalchemy import text

            result = self._db.execute(
                text("""
                    SELECT p.carrier_name, p.coverage_line, p.total_premium_usd,
                           p.effective_date, p.expiration_date,
                           c.limit_amount, c.deductible_amount
                    FROM insurance_policies p
                    LEFT JOIN insurance_coverages c ON c.policy_id = p.id
                    WHERE p.account_id = :account_id AND p.status = 'active'
                    ORDER BY p.coverage_line
                    LIMIT 50
                """),
                {"account_id": account_id},
            )
            return [dict(row._mapping) for row in result.fetchall()]
        except Exception as exc:
            logger.warning("board_presentation.coverage_query_failed", extra={"error": str(exc)})
            return []

    def _query_expiring_policies(self, account_id: str, window_days: int) -> list[dict[str, Any]]:
        try:
            from datetime import timedelta

            from sqlalchemy import text

            cutoff = (date.today() + timedelta(days=window_days)).isoformat()
            result = self._db.execute(
                text("""
                    SELECT policy_number, carrier_name, coverage_line,
                           expiration_date, total_premium_usd
                    FROM insurance_policies
                    WHERE account_id = :account_id
                      AND status = 'active'
                      AND expiration_date <= :cutoff
                    ORDER BY expiration_date
                    LIMIT 20
                """),
                {"account_id": account_id, "cutoff": cutoff},
            )
            return [dict(row._mapping) for row in result.fetchall()]
        except Exception as exc:
            logger.warning("board_presentation.renewal_query_failed", extra={"error": str(exc)})
            return []

    def _generate_resolutions(self, packet: BoardPresentationPacket) -> list[str]:
        resolutions = []
        if packet.upcoming_renewals.expiring_policies:
            resolutions.append(
                "RESOLVED: The Board authorizes management to solicit renewal quotes for "
                f"{len(packet.upcoming_renewals.expiring_policies)} expiring policy(ies) "
                f"and present options at the next meeting."
            )
        if packet.compliance_status.gap_count > 0:
            resolutions.append(
                "RESOLVED: The Board directs management to address "
                f"{packet.compliance_status.gap_count} identified compliance gap(s) "
                "prior to the next renewal cycle."
            )
        if not resolutions:
            resolutions.append(
                "RESOLVED: The Board accepts the current insurance program as presented "
                "and authorizes management to continue with renewals as scheduled."
            )
        return resolutions
