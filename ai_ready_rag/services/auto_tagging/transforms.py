"""Transform functions for auto-tag value normalization and entity resolution."""

from __future__ import annotations

import re
from collections.abc import Callable


def slugify(value: str) -> str:
    """Normalize a string to a URL-safe slug.

    Rules (per spec Section 11.3):
    1. Lowercase
    2. Replace spaces and underscores with hyphens
    3. Strip non-alphanumeric characters (except hyphens)
    4. Collapse multiple hyphens to single
    5. Trim leading/trailing hyphens
    """
    result = value.lower()
    result = re.sub(r"[\s_]+", "-", result)
    result = re.sub(r"[^a-z0-9-]", "", result)
    result = re.sub(r"-{2,}", "-", result)
    result = result.strip("-")
    return result


def year_range(value: str) -> str:
    """Convert a 1-2 digit year to a year range string.

    Always assumes 2000s century. Two-digit years are mapped as 20XX to 20(XX+1).

    Examples:
        "24" -> "2024-2025"
        "5"  -> "2005-2006"
        "0"  -> "2000-2001"
        "99" -> "2099-2100"

    Raises:
        ValueError: If input is not a valid 1-2 digit number.
    """
    stripped = value.strip()
    if not stripped.isdigit():
        msg = f"year_range requires a numeric string, got '{value}'"
        raise ValueError(msg)
    year_num = int(stripped)
    if year_num < 0 or year_num > 99:
        msg = f"year_range requires a 0-99 value, got {year_num}"
        raise ValueError(msg)
    start = 2000 + year_num
    end = start + 1
    return f"{start}-{end}"


def lowercase(value: str) -> str:
    """Convert value to lowercase."""
    return value.lower()


def identity(value: str) -> str:
    """Return value unchanged."""
    return value


TRANSFORM_REGISTRY: dict[str | None, Callable[[str], str]] = {
    "slugify": slugify,
    "year_range": year_range,
    "lowercase": lowercase,
    "none": identity,
    None: identity,
}


def get_transform(name: str | None) -> Callable[[str], str]:
    """Look up a transform function by name.

    Raises:
        ValueError: If the transform name is not recognized.
    """
    if name in TRANSFORM_REGISTRY:
        return TRANSFORM_REGISTRY[name]
    msg = f"Unknown transform '{name}'. Valid transforms: {list(TRANSFORM_REGISTRY.keys())}"
    raise ValueError(msg)


def resolve_entity(raw_name: str, aliases: dict[str, str]) -> str:
    """Normalize an entity name using the alias table.

    Resolution order:
    1. Exact match in aliases
    2. Case-insensitive match in aliases
    3. Fallback to slugify(raw_name)
    """
    if raw_name in aliases:
        return aliases[raw_name]
    raw_lower = raw_name.lower()
    for alias, slug in aliases.items():
        if raw_lower == alias.lower():
            return slug
    return slugify(raw_name)
