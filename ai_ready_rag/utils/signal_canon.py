"""Signal canonicalization for column names and row values.

Transforms raw strings before storing in row_value_samples or column_signals.
Pipeline: Unicode NFKC → strip → hyphen/underscore→space → collapse whitespace
          → lowercase → stopword filter → length gate (3–80 chars)

Spec: UNIFIED_TABLE_INGEST_v1 §6.4
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_STOPWORDS = frozenset({"a", "an", "the", "of", "in", "is", "at", "for", "to", "and", "or"})


def canonicalize(value: str) -> str:
    """Return canonical form of a signal string, or '' if filtered out."""
    # Unicode NFKC normalization
    s = unicodedata.normalize("NFKC", value)
    # Strip leading/trailing whitespace
    s = s.strip()
    # Replace hyphens and underscores with spaces
    s = re.sub(r"[-_]", " ", s)
    # Collapse multiple whitespace to single space
    s = re.sub(r"\s+", " ", s).strip()
    # Lowercase
    s = s.lower()
    # Stopword filter (whole-string match for single tokens)
    if s in _STOPWORDS:
        return ""
    # Length gate: 3–80 chars
    if len(s) < 3 or len(s) > 80:
        return ""
    return s


def sample_row_values(
    df,
    max_per_col: int = 50,
    cardinality_limit: int = 200,
) -> dict[str, list[str]]:
    """Sample up to max_per_col unique non-null values per column.

    Args:
        df: A pandas DataFrame.
        max_per_col: Maximum unique canonical values to store per column.
        cardinality_limit: Columns with more unique values are skipped (high-cardinality).

    Skips columns with >cardinality_limit unique values (high-cardinality,
    not useful for WHERE clause hints).
    Applies canonicalize() to each value before storing.

    Returns dict mapping column_name → list[canonical_value].
    """

    samples: dict[str, list[str]] = {}
    for col in df.columns:
        uniq = df[col].dropna().astype(str).unique()
        if len(uniq) > cardinality_limit:
            continue
        canonical_vals = []
        for v in uniq:
            c = canonicalize(str(v))
            if c:
                canonical_vals.append(c)
        if canonical_vals:
            samples[col] = canonical_vals[:max_per_col]
    return samples


def make_table_name(stem: str, doc_id: str, table_idx: int) -> str:
    """Generate a stable table name from document stem, ID and table index.

    Format: {slug}_{sha256_suffix}
    Total length: always ≤49 chars (well within PostgreSQL 63-byte limit).
    slug = alphanum+underscore only, max 40 chars
    suffix = first 8 hex chars of sha256(doc_id:table_idx)
    """
    # Slugify: replace non-alphanum with underscore, collapse, strip, truncate
    slug = re.sub(r"[^a-z0-9]+", "_", stem.lower())
    slug = slug.strip("_")[:40]
    suffix = hashlib.sha256(f"{doc_id}:{table_idx}".encode()).hexdigest()[:8]
    return f"{slug}_{suffix}"
