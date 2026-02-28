"""Webhook dispatcher for outbound event delivery.

Dispatches WebhookEvent payloads to registered HTTP endpoints.
Supports HMAC signing, retries with exponential backoff, and
fire-and-forget async delivery.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

import httpx

from ai_ready_rag.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class WebhookEvent:
    """An event to be dispatched to registered webhook endpoints."""

    event_type: str  # e.g. "document.processed", "policy.extracted", "compliance.gap.found"
    payload: dict
    tenant_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict:
        """Serialize event for HTTP delivery."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
        }


@dataclass
class WebhookDeliveryResult:
    """Result of a single webhook delivery attempt."""

    url: str
    event_id: str
    success: bool
    status_code: int | None = None
    attempts: int = 0
    error: str | None = None


class WebhookDispatcher:
    """Dispatches webhook events to registered HTTP endpoints.

    Usage::

        dispatcher = WebhookDispatcher(settings)
        dispatcher.register("document.processed", "https://ams.example.com/hooks")
        results = await dispatcher.dispatch(event)
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._endpoints: dict[str, list[str]] = {}  # event_type -> list of URLs

    def register(self, event_type: str, url: str) -> None:
        """Register a webhook URL for an event type.

        Args:
            event_type: Event type string (e.g. "document.processed").
            url: HTTP(S) URL to deliver events to.
        """
        if event_type not in self._endpoints:
            self._endpoints[event_type] = []
        if url not in self._endpoints[event_type]:
            self._endpoints[event_type].append(url)
        logger.info(
            "webhook.endpoint.registered",
            extra={"event_type": event_type, "url": url},
        )

    async def dispatch(self, event: WebhookEvent) -> list[WebhookDeliveryResult]:
        """Fire-and-forget async dispatch to all registered URLs for event type.

        Returns delivery results for all registered endpoints. If no endpoints
        are registered for the event type, returns an empty list.

        Args:
            event: The WebhookEvent to dispatch.

        Returns:
            List of WebhookDeliveryResult, one per registered URL.
        """
        if not self._settings.webhook_enabled:
            logger.debug(
                "webhook.dispatch.skipped",
                extra={"reason": "webhooks_disabled", "event_type": event.event_type},
            )
            return []

        urls = self._endpoints.get(event.event_type, [])
        if not urls:
            logger.debug(
                "webhook.dispatch.no_endpoints",
                extra={"event_type": event.event_type},
            )
            return []

        tasks = [self._deliver(url, event) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)

    async def _deliver(self, url: str, event: WebhookEvent) -> WebhookDeliveryResult:
        """Single delivery attempt with retry (3 attempts, exponential backoff).

        Args:
            url: Target HTTP URL.
            event: WebhookEvent to deliver.

        Returns:
            WebhookDeliveryResult with outcome details.
        """
        max_retries = self._settings.webhook_max_retries
        timeout = self._settings.webhook_timeout_seconds
        body = json.dumps(event.to_dict(), default=str)
        headers = self._build_headers(body)

        last_error: str | None = None
        last_status: int | None = None

        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, content=body, headers=headers)
                last_status = response.status_code
                if response.is_success:
                    logger.info(
                        "webhook.delivery.success",
                        extra={
                            "url": url,
                            "event_id": event.event_id,
                            "event_type": event.event_type,
                            "attempt": attempt,
                            "status_code": last_status,
                        },
                    )
                    return WebhookDeliveryResult(
                        url=url,
                        event_id=event.event_id,
                        success=True,
                        status_code=last_status,
                        attempts=attempt,
                    )
                last_error = f"HTTP {last_status}"
                logger.warning(
                    "webhook.delivery.http_error",
                    extra={
                        "url": url,
                        "event_id": event.event_id,
                        "attempt": attempt,
                        "status_code": last_status,
                    },
                )
            except httpx.TimeoutException as exc:
                last_error = f"Timeout: {exc}"
                logger.warning(
                    "webhook.delivery.timeout",
                    extra={"url": url, "event_id": event.event_id, "attempt": attempt},
                )
            except httpx.RequestError as exc:
                last_error = f"Request error: {exc}"
                logger.warning(
                    "webhook.delivery.request_error",
                    extra={
                        "url": url,
                        "event_id": event.event_id,
                        "attempt": attempt,
                        "error": last_error,
                    },
                )

            # Exponential backoff before retry (skip after final attempt)
            if attempt < max_retries:
                backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s, ...
                await asyncio.sleep(backoff)

        logger.error(
            "webhook.delivery.failed",
            extra={
                "url": url,
                "event_id": event.event_id,
                "attempts": max_retries,
                "error": last_error,
            },
        )
        return WebhookDeliveryResult(
            url=url,
            event_id=event.event_id,
            success=False,
            status_code=last_status,
            attempts=max_retries,
            error=last_error,
        )

    def _build_headers(self, body: str) -> dict[str, str]:
        """Build HTTP headers for webhook delivery, including optional HMAC signature.

        Args:
            body: JSON-serialized event body (used for HMAC signing).

        Returns:
            Dict of HTTP headers.
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "VaultIQ-Webhook/1.0",
        }
        if self._settings.webhook_secret:
            sig = hmac.new(
                self._settings.webhook_secret.encode(),
                body.encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={sig}"
        return headers
