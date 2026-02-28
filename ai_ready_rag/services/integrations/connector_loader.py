"""AMS Connector loader and Protocol definition.

Defines the AMSConnector Protocol that all AMS connectors must implement,
and the ConnectorLoader that manages registered connector instances.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class AMSConnector(Protocol):
    """Protocol that all AMS connectors must implement.

    AMS (Agency Management System) connectors enable VaultIQ to push
    extracted policy data to and pull account information from external
    insurance management systems (e.g. Applied Epic, Vertafore AMS360).
    """

    name: str
    version: str

    async def push_policy(self, policy_data: dict) -> dict:
        """Push extracted policy data to the AMS.

        Args:
            policy_data: Dict containing extracted policy fields.

        Returns:
            Dict with AMS response (e.g. confirmation ID, status).
        """
        ...

    async def pull_accounts(self, tenant_id: str) -> list[dict]:
        """Pull account list from the AMS for a given tenant.

        Args:
            tenant_id: Tenant identifier to scope the account query.

        Returns:
            List of account dicts from the AMS.
        """
        ...


class ConnectorLoader:
    """Loads and manages AMS connectors registered via ModuleRegistry.

    Connectors are registered by name at startup (via ModuleRegistry or
    directly). Core services call get() or list_connectors() at runtime.

    Usage::

        loader = ConnectorLoader()
        loader.register("epic", EpicConnector())
        connector = loader.get("epic")
        if connector:
            result = await connector.push_policy(policy_data)
    """

    def __init__(self) -> None:
        self._connectors: dict[str, AMSConnector] = {}

    def register(self, name: str, connector: AMSConnector) -> None:
        """Register a connector by name.

        Args:
            name: Unique connector identifier (e.g. "epic", "ams360", "null").
            connector: Instance implementing the AMSConnector protocol.

        Raises:
            TypeError: If connector does not implement the AMSConnector protocol.
        """
        if not isinstance(connector, AMSConnector):
            raise TypeError(
                f"Connector '{name}' does not implement the AMSConnector protocol. "
                f"Must have: name, version, push_policy(), pull_accounts()."
            )
        self._connectors[name] = connector
        logger.info(
            "connector.registered",
            extra={
                "connector_key": name,
                "connector_name": connector.name,
                "version": connector.version,
            },
        )

    def get(self, name: str) -> AMSConnector | None:
        """Get a connector by name.

        Args:
            name: Connector identifier used during registration.

        Returns:
            AMSConnector instance or None if not registered.
        """
        return self._connectors.get(name)

    def list_connectors(self) -> list[str]:
        """List all registered connector names.

        Returns:
            Sorted list of registered connector names.
        """
        return sorted(self._connectors.keys())
