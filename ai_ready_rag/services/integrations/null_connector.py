"""NullAMSConnector — no-op AMS connector for testing and development.

This connector implements the AMSConnector protocol but performs no real
operations. It is safe to use in development environments where no AMS
integration is configured.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class NullAMSConnector:
    """No-op AMS connector. All operations succeed silently.

    Useful for:
    - Unit tests that need an AMSConnector without external dependencies
    - Development environments without a real AMS available
    - Default connector when no AMS is configured

    Usage::

        connector = NullAMSConnector()
        loader.register("null", connector)
    """

    name: str = "null"
    version: str = "1.0.0"

    async def push_policy(self, policy_data: dict) -> dict:
        """No-op policy push. Logs and returns an empty acknowledgement.

        Args:
            policy_data: Ignored.

        Returns:
            Dict with status "noop" and connector name.
        """
        logger.debug("null_connector.push_policy.noop", extra={"keys": list(policy_data.keys())})
        return {"status": "noop", "connector": self.name}

    async def pull_accounts(self, tenant_id: str) -> list[dict]:
        """No-op account pull. Returns empty list.

        Args:
            tenant_id: Ignored.

        Returns:
            Empty list.
        """
        logger.debug("null_connector.pull_accounts.noop", extra={"tenant_id": tenant_id})
        return []
