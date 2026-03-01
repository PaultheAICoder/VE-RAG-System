"""SQL execution-time access control.

TagPredicateValidator enforces tag-based access control at SQL execution time,
BEFORE SqlInjectionGuard runs. This is a security gate, not a routing hint.

Spec: UNIFIED_TABLE_INGEST_v1 §6.3, §7.3
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_ready_rag.modules.registry import SQLTemplate

logger = logging.getLogger(__name__)


class AccessDeniedError(Exception):
    """Raised when user lacks required tags to access a SQL template."""


class TagPredicateValidator:
    """Validates user has ALL required tags to access a SQL template at execution time.

    Logic:
        - If template.access_tags is empty: allow all users (public table)
        - If user_tags is None: admin bypass, allow all
        - Otherwise: user must have ALL required tags (issubset check — AND logic, not OR)
    """

    def validate(self, template: SQLTemplate, user_tags: list[str] | None) -> None:
        """Raise AccessDeniedError if user lacks required tags.

        Args:
            template: SQLTemplate with access_tags list ([] = public)
            user_tags: User's assigned tags. None = admin bypass.

        Raises:
            AccessDeniedError: If template.access_tags non-empty and user
                               does not have ALL required tags.
        """
        if not template.access_tags:
            return  # Public table — allow all users

        if user_tags is None:
            return  # Admin bypass — allow all

        required = set(template.access_tags)
        have = set(user_tags)
        if not required.issubset(have):
            missing = required - have
            raise AccessDeniedError(
                f"Access denied for table '{template.name}'. "
                f"Required tags: {sorted(template.access_tags)}. "
                f"Missing: {sorted(missing)}."
            )
