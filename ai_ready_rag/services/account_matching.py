"""AccountMatchingService — 3-tier fuzzy name resolution for HOA/CA accounts.

Tier 1: Exact (case-insensitive)
Tier 2: Suffix-stripped (removes HOA, LLC, Inc, Association, etc.)
Tier 3: Phonetic fallback (Soundex)

Thresholds and suffix lists are configurable via ModuleRegistry.register_entity_map()
so vertical modules can extend them.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Common suffixes to strip before matching
DEFAULT_SUFFIXES = [
    r"\bhoa\b",
    r"\bhomeowners association\b",
    r"\bhomeowner'?s association\b",
    r"\bcommunity association\b",
    r"\bcondo association\b",
    r"\bcondominium association\b",
    r"\bproperty owners association\b",
    r"\bpoa\b",
    r"\binc\.?\b",
    r"\bllc\.?\b",
    r"\bltd\.?\b",
    r"\bcorp\.?\b",
    r"\bassociation\b",
    r"\bassoc\.?\b",
    r"\bproperties\b",
    r"\bproperty\b",
    r"\bestate[s]?\b",
    r"\bvillage[s]?\b",
    r"\bcommons\b",
    r"\bcommon\b",
    r"\bmanagement\b",
    r"\bmgmt\.?\b",
]


def _normalize(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", name.lower().strip())


def _strip_suffixes(name: str, extra_suffixes: list[str] | None = None) -> str:
    """Remove legal/type suffixes from a name, returning the core name."""
    normalized = _normalize(name)
    suffixes = DEFAULT_SUFFIXES + (extra_suffixes or [])
    for pattern in suffixes:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip().rstrip(",").strip()


def _soundex(name: str) -> str:
    """Simple Soundex implementation."""
    name = re.sub(r"[^a-zA-Z]", "", name).upper()
    if not name:
        return "0000"

    code_map = {
        "BFPV": "1",
        "CGJKQSXYZ": "2",
        "DT": "3",
        "L": "4",
        "MN": "5",
        "R": "6",
    }

    def char_code(c: str) -> str:
        for chars, code in code_map.items():
            if c in chars:
                return code
        return "0"

    first = name[0]
    coded = first
    prev_code = char_code(first)

    for char in name[1:]:
        code = char_code(char)
        if code != "0" and code != prev_code:
            coded += code
        prev_code = code

    return (coded + "000")[:4]


@dataclass
class MatchResult:
    matched: bool
    tier: int  # 1=exact, 2=suffix_stripped, 3=phonetic, 0=no_match
    confidence: float  # 0.0–1.0
    candidate: str  # best matching candidate name
    method: str  # "exact" | "suffix_stripped" | "phonetic" | "no_match"


class AccountMatchingService:
    """3-tier fuzzy account name matching."""

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        extra_suffixes: list[str] | None = None,
    ) -> None:
        self._threshold = confidence_threshold
        self._extra_suffixes = extra_suffixes or []

    def match(self, query: str, candidates: list[str]) -> MatchResult:
        """Find the best match for query in candidates list."""
        if not candidates:
            return MatchResult(
                matched=False, tier=0, confidence=0.0, candidate="", method="no_match"
            )

        # Tier 1: Exact match
        q_norm = _normalize(query)
        for candidate in candidates:
            if _normalize(candidate) == q_norm:
                logger.debug("account_match.exact", extra={"query": query, "candidate": candidate})
                return MatchResult(
                    matched=True, tier=1, confidence=1.0, candidate=candidate, method="exact"
                )

        # Tier 2: Suffix-stripped match
        q_stripped = _strip_suffixes(query, self._extra_suffixes)
        if q_stripped:
            for candidate in candidates:
                c_stripped = _strip_suffixes(candidate, self._extra_suffixes)
                if c_stripped and q_stripped == c_stripped:
                    logger.debug(
                        "account_match.suffix_stripped",
                        extra={"query": query, "candidate": candidate},
                    )
                    return MatchResult(
                        matched=True,
                        tier=2,
                        confidence=0.85,
                        candidate=candidate,
                        method="suffix_stripped",
                    )

        # Tier 3: Phonetic fallback
        q_soundex = _soundex(q_stripped or q_norm)
        best_candidate = ""
        best_score = 0.0

        for candidate in candidates:
            c_stripped = _strip_suffixes(candidate, self._extra_suffixes)
            c_soundex = _soundex(c_stripped or _normalize(candidate))
            if q_soundex == c_soundex and q_soundex != "0000":
                # Count matching characters for confidence
                score = sum(a == b for a, b in zip(q_stripped, c_stripped, strict=False)) / max(
                    len(q_stripped), len(c_stripped), 1
                )
                score = max(score, 0.6)  # phonetic match floor
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

        if best_candidate and best_score >= self._threshold:
            logger.debug(
                "account_match.phonetic",
                extra={"query": query, "candidate": best_candidate, "score": best_score},
            )
            return MatchResult(
                matched=True,
                tier=3,
                confidence=best_score,
                candidate=best_candidate,
                method="phonetic",
            )

        return MatchResult(
            matched=False,
            tier=0,
            confidence=best_score,
            candidate=best_candidate or "",
            method="no_match",
        )

    def match_all(self, query: str, candidates: list[str]) -> list[MatchResult]:
        """Return all matches above threshold, sorted by confidence."""
        results = []
        for candidate in candidates:
            result = self.match(query, [candidate])
            if result.matched:
                results.append(result)
        return sorted(results, key=lambda r: r.confidence, reverse=True)
