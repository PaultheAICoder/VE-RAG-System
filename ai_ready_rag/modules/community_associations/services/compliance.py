"""CA compliance engine — Fannie Mae/FHA requirement injection and gap detection.

Injects compliance constraints into Claude prompts and validates extracted
coverage data against Fannie Mae and FHA minimum requirements.

Returns a structured ComplianceReport with gap analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ComplianceStandard(str, Enum):
    FANNIE_MAE = "fannie_mae"
    FHA = "fha"
    BOTH = "both"


# ---------------------------------------------------------------------------
# Fannie Mae minimum coverage requirements (2026-Q1)
# Source: Fannie Mae Selling Guide B7-3-06
# ---------------------------------------------------------------------------
FANNIE_MAE_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "property": {
        "min_coverage_basis": "replacement_cost",
        "required": True,
        "notes": "Must cover 100% of insurable replacement cost value",
        "deductible_max_pct": 0.05,  # max 5% of building value
    },
    "general_liability": {
        "min_per_occurrence": 1_000_000,
        "min_aggregate": 1_000_000,
        "required": True,
        "notes": "Combined single limit of at least $1M per occurrence",
    },
    "fidelity": {
        "min_amount_formula": "3_months_assessments",
        "required": True,
        "notes": "Required for projects with >20 units or professional management",
    },
    "flood": {
        "required_if": "in_sfha",  # Special Flood Hazard Area
        "min_amount": "lesser_of_replacement_cost_or_nfip_max",
        "notes": "Required for properties in SFHA zones A or V",
    },
}

# ---------------------------------------------------------------------------
# FHA minimum coverage requirements
# Source: HUD Handbook 4000.1
# ---------------------------------------------------------------------------
FHA_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "property": {
        "min_coverage_basis": "replacement_cost",
        "required": True,
        "notes": "Must cover 100% replacement cost, no co-insurance clause",
    },
    "general_liability": {
        "min_per_occurrence": 1_000_000,
        "required": True,
        "notes": "$1M minimum per occurrence",
    },
    "fidelity": {
        "required": True,
        "notes": "Required for all FHA-approved condo projects",
    },
    "flood": {
        "required_if": "in_sfha",
        "notes": "Same as Fannie Mae — required in SFHA",
    },
}

# ---------------------------------------------------------------------------
# Prompt injection templates
# ---------------------------------------------------------------------------
FANNIE_MAE_INJECTION = """
COMPLIANCE CONTEXT — Fannie Mae Requirements (2026-Q1):
- Property: Must cover 100% replacement cost; max deductible 5% of building value
- General Liability: Minimum $1,000,000 per occurrence / $1,000,000 aggregate
- Fidelity/Crime: Required for >20 units; covers 3 months of assessments minimum
- Flood: Required if property is in SFHA Zone A or V

When extracting coverage information, flag any gaps against these minimums.
Return a "compliance_gaps" array with objects: {"coverage_line": "...", "gap_type": "...", "detail": "..."}
"""

FHA_INJECTION = """
COMPLIANCE CONTEXT — FHA Requirements (HUD 4000.1):
- Property: 100% replacement cost required; no co-insurance clause permitted
- General Liability: Minimum $1,000,000 per occurrence
- Fidelity: Required for all FHA-approved projects (no unit threshold)
- Flood: Required in SFHA; NFIP or private flood accepted

Flag any gaps against these minimums in a "compliance_gaps" array.
"""


@dataclass
class CoverageGap:
    coverage_line: str
    gap_type: str  # "missing" | "below_minimum" | "missing_required_clause"
    detail: str
    standard: str  # "fannie_mae" | "fha" | "both"
    severity: str = "high"  # "high" | "medium" | "low"


@dataclass
class ComplianceReport:
    standard: ComplianceStandard
    account_name: str
    is_compliant: bool
    gaps: list[CoverageGap] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_lines: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def gap_count(self) -> int:
        return len(self.gaps)

    @property
    def high_severity_count(self) -> int:
        return sum(1 for g in self.gaps if g.severity == "high")


class ComplianceEngine:
    """Fannie Mae / FHA compliance checker for CA module.

    Two main entry points:
    1. inject_prompt(standard) — returns text to append to Claude extraction prompt
    2. check(coverage_data, standard) — validates coverage against minimums
    """

    def inject_prompt(self, standard: ComplianceStandard = ComplianceStandard.BOTH) -> str:
        """Return compliance requirement text to inject into Claude prompts."""
        if standard == ComplianceStandard.FANNIE_MAE:
            return FANNIE_MAE_INJECTION
        elif standard == ComplianceStandard.FHA:
            return FHA_INJECTION
        else:
            return FANNIE_MAE_INJECTION + "\n" + FHA_INJECTION

    def check(
        self,
        coverage_data: list[dict[str, Any]],
        account_name: str = "",
        standard: ComplianceStandard = ComplianceStandard.BOTH,
        in_sfha: bool = False,
    ) -> ComplianceReport:
        """Check coverage data against compliance requirements.

        coverage_data: list of dicts with keys: coverage_line, limit_amount,
                       deductible_amount, meets_fannie_mae, meets_fha
        """
        gaps: list[CoverageGap] = []
        warnings: list[str] = []
        checked_lines: list[str] = []

        # Build a lookup by coverage line
        by_line: dict[str, dict[str, Any]] = {}
        for cov in coverage_data:
            line = cov.get("coverage_line", "").lower().replace(" ", "_")
            by_line[line] = cov
            checked_lines.append(line)

        standards_to_check = []
        if standard in (ComplianceStandard.FANNIE_MAE, ComplianceStandard.BOTH):
            standards_to_check.append(("fannie_mae", FANNIE_MAE_REQUIREMENTS))
        if standard in (ComplianceStandard.FHA, ComplianceStandard.BOTH):
            standards_to_check.append(("fha", FHA_REQUIREMENTS))

        for std_name, requirements in standards_to_check:
            for coverage_line, req in requirements.items():
                # Determine whether this line is required given context
                conditionally_required = req.get("required_if") == "in_sfha" and in_sfha
                unconditionally_required = req.get("required", False)
                is_required = unconditionally_required or conditionally_required

                # Skip flood check entirely when outside SFHA
                if req.get("required_if") == "in_sfha" and not in_sfha:
                    continue

                if is_required and coverage_line not in by_line:
                    gaps.append(
                        CoverageGap(
                            coverage_line=coverage_line,
                            gap_type="missing",
                            detail=f"{coverage_line} coverage not found in extracted data",
                            standard=std_name,
                            severity="high",
                        )
                    )
                    continue

                cov = by_line.get(coverage_line, {})

                # Check GL minimums
                if coverage_line == "general_liability":
                    limit = cov.get("limit_amount")
                    min_required = req.get("min_per_occurrence", 0)
                    if limit and float(limit) < min_required:
                        gaps.append(
                            CoverageGap(
                                coverage_line=coverage_line,
                                gap_type="below_minimum",
                                detail=(
                                    f"GL limit ${limit:,.0f} is below {std_name} minimum"
                                    f" ${min_required:,.0f}"
                                ),
                                standard=std_name,
                                severity="high",
                            )
                        )

        is_compliant = len([g for g in gaps if g.severity == "high"]) == 0

        return ComplianceReport(
            standard=standard,
            account_name=account_name,
            is_compliant=is_compliant,
            gaps=gaps,
            warnings=warnings,
            checked_lines=checked_lines,
        )

    def check_as_dict(self, account_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Registry-compatible interface: accepts account_id + data dict.

        Adapts the registry ComplianceChecker.check(account_id, data) signature
        to the ComplianceEngine.check() interface.

        data keys:
            coverage_data (list[dict]): coverage line items
            standard (str): "fannie_mae" | "fha" | "both"
            in_sfha (bool): whether property is in SFHA
        """
        coverage_data = data.get("coverage_data", [])
        standard_str = data.get("standard", ComplianceStandard.BOTH.value)
        try:
            standard = ComplianceStandard(standard_str)
        except ValueError:
            standard = ComplianceStandard.BOTH
        in_sfha = data.get("in_sfha", False)

        report = self.check(
            coverage_data=coverage_data,
            account_name=account_id,
            standard=standard,
            in_sfha=in_sfha,
        )
        return {
            "standard": report.standard.value,
            "account_name": report.account_name,
            "is_compliant": report.is_compliant,
            "gap_count": report.gap_count,
            "high_severity_count": report.high_severity_count,
            "gaps": [
                {
                    "coverage_line": g.coverage_line,
                    "gap_type": g.gap_type,
                    "detail": g.detail,
                    "standard": g.standard,
                    "severity": g.severity,
                }
                for g in report.gaps
            ],
            "warnings": report.warnings,
            "checked_lines": report.checked_lines,
            "notes": report.notes,
        }


class CommunityAssociationsComplianceChecker:
    """Registry adapter that wraps ComplianceEngine for ModuleRegistry.register_compliance_checker().

    The ModuleRegistry expects a ComplianceChecker with a .check(account_id, data) -> dict
    signature. This adapter bridges the registry protocol to ComplianceEngine.
    """

    def __init__(self) -> None:
        self._engine = ComplianceEngine()

    def check(self, account_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Run compliance check. Returns ComplianceReport as dict."""
        return self._engine.check_as_dict(account_id=account_id, data=data)

    def inject_prompt(self, standard: ComplianceStandard = ComplianceStandard.BOTH) -> str:
        """Delegate prompt injection to the underlying engine."""
        return self._engine.inject_prompt(standard)
