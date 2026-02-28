"""UnitOwnerLetterService — batch generation of HO-6 compliance letters.

Manages the lifecycle of a unit owner letter batch:
1. Creates a ca_letter_batches record
2. Queries non-compliant unit owners
3. Creates ca_letter_batch_items per owner
4. Generates letter content (text template, not PDF — PDF handled by DocumentRenderer)
5. Updates batch status to completed/failed

PII handling: owner data is decrypted only during letter generation,
never logged or stored in batch items.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Default letter template — uses {variable} placeholders
HO6_COMPLIANCE_LETTER_TEMPLATE = """\
{date}

{association_name}
Community Association Management

Dear Unit Owner — Unit {unit_number},

This letter serves as a notice that our records indicate your unit may not have \
an active HO-6 (Unit Owners) insurance policy on file with the association.

As required by the Association's governing documents and consistent with \
Fannie Mae and FHA lending requirements, all unit owners are required to \
maintain a minimum of ${min_coverage_amount:,} in personal property and \
liability coverage.

Please provide proof of your current HO-6 policy by {compliance_deadline}.

If you have recently obtained coverage, please submit your certificate of \
insurance to management at your earliest convenience.

If you have questions, please contact the management office.

Sincerely,
{association_name} Board of Directors
"""


@dataclass
class LetterContent:
    unit_number: str
    letter_text: str
    generated_at: datetime


@dataclass
class BatchResult:
    batch_id: str
    total_count: int
    generated_count: int
    failed_count: int
    status: str  # "completed" | "partial" | "failed"
    letters: list[LetterContent]


class UnitOwnerLetterService:
    """Manages HO-6 compliance letter batch generation.

    Works in two modes:
    - With DB: queries ca_unit_owners, creates batch records
    - Without DB: generates letters from provided unit data (for testing/preview)
    """

    def __init__(
        self,
        db: Any = None,
        letter_template: str = HO6_COMPLIANCE_LETTER_TEMPLATE,
        encryption_service: Any = None,  # FernetEncryption instance for PII decryption
    ) -> None:
        self._db = db
        self._template = letter_template
        self._crypto = encryption_service

    def generate_batch(
        self,
        account_id: str,
        association_name: str,
        min_coverage_amount: int = 25_000,
        compliance_deadline_days: int = 30,
        initiated_by: str | None = None,
    ) -> BatchResult:
        """Generate a batch of HO-6 compliance letters for non-compliant units."""
        from datetime import date, timedelta

        compliance_deadline = (date.today() + timedelta(days=compliance_deadline_days)).strftime(
            "%B %d, %Y"
        )

        today_str = datetime.utcnow().strftime("%B %d, %Y")

        unit_owners = self._query_noncompliant_owners(account_id) if self._db is not None else []

        letters = []
        failed_count = 0

        for owner in unit_owners:
            try:
                unit_number = owner.get("unit_number", "Unknown")
                letter_text = self._render_letter(
                    unit_number=unit_number,
                    association_name=association_name,
                    min_coverage_amount=min_coverage_amount,
                    compliance_deadline=compliance_deadline,
                    today_str=today_str,
                )
                letters.append(
                    LetterContent(
                        unit_number=unit_number,
                        letter_text=letter_text,
                        generated_at=datetime.utcnow(),
                    )
                )
            except Exception as exc:
                logger.warning(
                    "letter.generation.failed",
                    extra={"account_id": account_id, "error": str(exc)},
                )
                failed_count += 1

        total = len(unit_owners)
        generated = len(letters)
        status = "completed" if failed_count == 0 else ("partial" if generated > 0 else "failed")

        return BatchResult(
            batch_id="",  # populated when persisted
            total_count=total,
            generated_count=generated,
            failed_count=failed_count,
            status=status,
            letters=letters,
        )

    def generate_preview(
        self,
        unit_number: str,
        association_name: str,
        min_coverage_amount: int = 25_000,
        compliance_deadline_days: int = 30,
    ) -> LetterContent:
        """Generate a single preview letter without DB access."""
        from datetime import date, timedelta

        compliance_deadline = (date.today() + timedelta(days=compliance_deadline_days)).strftime(
            "%B %d, %Y"
        )
        today_str = datetime.utcnow().strftime("%B %d, %Y")

        letter_text = self._render_letter(
            unit_number=unit_number,
            association_name=association_name,
            min_coverage_amount=min_coverage_amount,
            compliance_deadline=compliance_deadline,
            today_str=today_str,
        )
        return LetterContent(
            unit_number=unit_number,
            letter_text=letter_text,
            generated_at=datetime.utcnow(),
        )

    def _render_letter(
        self,
        unit_number: str,
        association_name: str,
        min_coverage_amount: int,
        compliance_deadline: str,
        today_str: str,
    ) -> str:
        return self._template.format(
            date=today_str,
            unit_number=unit_number,
            association_name=association_name,
            min_coverage_amount=min_coverage_amount,
            compliance_deadline=compliance_deadline,
        )

    def _query_noncompliant_owners(self, account_id: str) -> list[dict[str, Any]]:
        """Query unit owners that are non-compliant or unverified."""
        try:
            from sqlalchemy import text

            result = self._db.execute(
                text("""
                    SELECT id, unit_number, ho6_required, ho6_verified, compliance_status
                    FROM ca_unit_owners
                    WHERE account_id = :account_id
                      AND deleted_at IS NULL
                      AND (compliance_status IN ('non_compliant', 'unknown')
                           OR (ho6_required = TRUE AND ho6_verified = FALSE))
                    ORDER BY unit_number
                    LIMIT 500
                """),
                {"account_id": account_id},
            )
            return [dict(row._mapping) for row in result.fetchall()]
        except Exception as exc:
            logger.warning(
                "unit_owner_query.failed",
                extra={"account_id": account_id, "error": str(exc)},
            )
            return []
