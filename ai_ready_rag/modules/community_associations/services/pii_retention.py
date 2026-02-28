"""PII retention service for ca_unit_owners table."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class PIIRetentionService:
    """Manages PII lifecycle for ca_unit_owners records.

    Operations:
    - get_status(db): count eligible records
    - purge(db, retention_days, dry_run): hard-delete or null-out PII columns
    """

    def __init__(self, retention_days: int = 365) -> None:
        self._retention_days = retention_days

    def get_status(self, db: Any) -> dict[str, Any]:
        """Return retention statistics."""
        cutoff = datetime.utcnow() - timedelta(days=self._retention_days)
        try:
            from sqlalchemy import text

            result = db.execute(
                text("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN deleted_at IS NOT NULL THEN 1 ELSE 0 END) as soft_deleted,
                    SUM(CASE WHEN deleted_at <= :cutoff THEN 1 ELSE 0 END) as eligible,
                    MIN(deleted_at) as oldest_deleted
                FROM ca_unit_owners
            """),
                {"cutoff": cutoff},
            )
            row = result.fetchone()
            return {
                "total_unit_owners": int(row.total or 0),
                "soft_deleted_count": int(row.soft_deleted or 0),
                "eligible_for_purge_count": int(row.eligible or 0),
                "oldest_deleted_at": str(row.oldest_deleted) if row.oldest_deleted else None,
                "retention_days": self._retention_days,
                "cutoff_date": cutoff.date().isoformat(),
            }
        except Exception as exc:
            logger.warning("pii_retention.status_failed", extra={"error": str(exc)})
            raise

    def purge(
        self, db: Any, retention_days: int | None = None, dry_run: bool = True
    ) -> dict[str, Any]:
        """Purge PII from records deleted before the retention cutoff.

        Nulls out encrypted columns rather than deleting the row,
        preserving the audit record while removing PII.
        """
        days = retention_days or self._retention_days
        cutoff = datetime.utcnow() - timedelta(days=days)

        try:
            from sqlalchemy import text

            # Count eligible records first
            count_result = db.execute(
                text("SELECT COUNT(*) as cnt FROM ca_unit_owners WHERE deleted_at <= :cutoff"),
                {"cutoff": cutoff},
            )
            count = int(count_result.fetchone().cnt or 0)

            if dry_run:
                return {
                    "purged_count": count,
                    "dry_run": True,
                    "cutoff_date": cutoff.date().isoformat(),
                    "message": f"Dry run: {count} records eligible for PII purge",
                }

            # Null out PII columns (preserves the row for audit)
            db.execute(
                text("""
                    UPDATE ca_unit_owners
                    SET owner_name_encrypted = NULL,
                        email_encrypted = NULL,
                        phone_encrypted = NULL
                    WHERE deleted_at <= :cutoff
                      AND (owner_name_encrypted IS NOT NULL
                           OR email_encrypted IS NOT NULL
                           OR phone_encrypted IS NOT NULL)
                """),
                {"cutoff": cutoff},
            )
            db.commit()

            logger.info(
                "pii_retention.purge_completed",
                extra={"count": count, "cutoff": cutoff.date().isoformat()},
            )
            return {
                "purged_count": count,
                "dry_run": False,
                "cutoff_date": cutoff.date().isoformat(),
                "message": f"Purged PII from {count} records deleted before {cutoff.date().isoformat()}",
            }
        except Exception as exc:
            if not dry_run and db:
                try:
                    db.rollback()
                except Exception:
                    pass
            logger.error("pii_retention.purge_failed", extra={"error": str(exc)})
            raise
