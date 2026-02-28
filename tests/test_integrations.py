"""Tests for the integrations framework.

Covers:
- WebhookEvent creation and serialization
- WebhookDispatcher register + dispatch (mocked httpx)
- ConnectorLoader registration and lookup
- NullAMSConnector no-ops
- ModuleRegistry.register_ams_connector integration
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_ready_rag.config import Settings
from ai_ready_rag.services.integrations import (
    AMSConnector,
    ConnectorLoader,
    WebhookDispatcher,
    WebhookEvent,
)
from ai_ready_rag.services.integrations.null_connector import NullAMSConnector

# ─── WebhookEvent ────────────────────────────────────────────────────────────


class TestWebhookEvent:
    def test_event_has_auto_generated_id(self):
        event = WebhookEvent(
            event_type="document.processed",
            payload={"doc_id": "abc"},
            tenant_id="tenant-1",
        )
        assert event.event_id
        assert len(event.event_id) == 36  # UUID4 string length

    def test_two_events_have_different_ids(self):
        e1 = WebhookEvent(event_type="x", payload={}, tenant_id="t")
        e2 = WebhookEvent(event_type="x", payload={}, tenant_id="t")
        assert e1.event_id != e2.event_id

    def test_event_timestamp_defaults_to_utcnow(self):
        before = datetime.now(UTC)
        event = WebhookEvent(event_type="x", payload={}, tenant_id="t")
        after = datetime.now(UTC)
        assert before <= event.timestamp <= after

    def test_to_dict_contains_required_keys(self):
        event = WebhookEvent(
            event_type="policy.extracted",
            payload={"policy_number": "P-001"},
            tenant_id="tenant-42",
        )
        d = event.to_dict()
        assert d["event_type"] == "policy.extracted"
        assert d["tenant_id"] == "tenant-42"
        assert d["payload"] == {"policy_number": "P-001"}
        assert "event_id" in d
        assert "timestamp" in d

    def test_to_dict_timestamp_is_iso_string(self):
        event = WebhookEvent(event_type="x", payload={}, tenant_id="t")
        d = event.to_dict()
        # Should be parseable as ISO datetime
        datetime.fromisoformat(d["timestamp"])

    def test_explicit_event_id_preserved(self):
        event = WebhookEvent(
            event_type="x",
            payload={},
            tenant_id="t",
            event_id="my-custom-id",
        )
        assert event.event_id == "my-custom-id"


# ─── WebhookDispatcher ───────────────────────────────────────────────────────


def _settings(enabled: bool = True, secret: str | None = None) -> Settings:
    """Build a minimal Settings instance with webhook config."""
    return Settings(
        webhook_enabled=enabled,
        webhook_secret=secret,
        webhook_timeout_seconds=5,
        webhook_max_retries=3,
    )


class TestWebhookDispatcherRegister:
    def test_register_single_url(self):
        dispatcher = WebhookDispatcher(_settings())
        dispatcher.register("document.processed", "https://example.com/hook")
        assert "document.processed" in dispatcher._endpoints
        assert "https://example.com/hook" in dispatcher._endpoints["document.processed"]

    def test_register_multiple_urls_same_event(self):
        dispatcher = WebhookDispatcher(_settings())
        dispatcher.register("policy.extracted", "https://a.com/hook")
        dispatcher.register("policy.extracted", "https://b.com/hook")
        assert len(dispatcher._endpoints["policy.extracted"]) == 2

    def test_register_duplicate_url_is_idempotent(self):
        dispatcher = WebhookDispatcher(_settings())
        dispatcher.register("x", "https://example.com/hook")
        dispatcher.register("x", "https://example.com/hook")
        assert len(dispatcher._endpoints["x"]) == 1

    def test_register_different_event_types(self):
        dispatcher = WebhookDispatcher(_settings())
        dispatcher.register("event.a", "https://a.com/hook")
        dispatcher.register("event.b", "https://b.com/hook")
        assert "event.a" in dispatcher._endpoints
        assert "event.b" in dispatcher._endpoints


class TestWebhookDispatcherDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_returns_empty_when_disabled(self):
        dispatcher = WebhookDispatcher(_settings(enabled=False))
        dispatcher.register("document.processed", "https://example.com/hook")
        event = WebhookEvent(event_type="document.processed", payload={}, tenant_id="t")
        results = await dispatcher.dispatch(event)
        assert results == []

    @pytest.mark.asyncio
    async def test_dispatch_returns_empty_when_no_endpoints(self):
        dispatcher = WebhookDispatcher(_settings())
        event = WebhookEvent(event_type="unregistered.event", payload={}, tenant_id="t")
        results = await dispatcher.dispatch(event)
        assert results == []

    @pytest.mark.asyncio
    async def test_dispatch_success(self):
        dispatcher = WebhookDispatcher(_settings())
        dispatcher.register("document.processed", "https://example.com/hook")
        event = WebhookEvent(
            event_type="document.processed",
            payload={"doc_id": "123"},
            tenant_id="tenant-1",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            results = await dispatcher.dispatch(event)

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert result.status_code == 200
        assert result.attempts == 1
        assert result.url == "https://example.com/hook"
        assert result.event_id == event.event_id

    @pytest.mark.asyncio
    async def test_dispatch_failure_exhausts_retries(self):
        settings = Settings(
            webhook_enabled=True,
            webhook_max_retries=2,
            webhook_timeout_seconds=5,
        )
        dispatcher = WebhookDispatcher(settings)
        dispatcher.register("policy.extracted", "https://fail.example.com/hook")
        event = WebhookEvent(event_type="policy.extracted", payload={}, tenant_id="t")

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.is_success = False

        with patch("httpx.AsyncClient") as mock_client_cls:
            with patch("asyncio.sleep", new_callable=AsyncMock):
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client_cls.return_value = mock_client

                results = await dispatcher.dispatch(event)

        assert len(results) == 1
        result = results[0]
        assert result.success is False
        assert result.attempts == 2
        assert result.status_code == 503
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_dispatch_multiple_endpoints(self):
        dispatcher = WebhookDispatcher(_settings())
        dispatcher.register("compliance.gap.found", "https://a.com/hook")
        dispatcher.register("compliance.gap.found", "https://b.com/hook")
        event = WebhookEvent(event_type="compliance.gap.found", payload={}, tenant_id="t")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            results = await dispatcher.dispatch(event)

        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_hmac_signature_header_present_when_secret_set(self):
        dispatcher = WebhookDispatcher(_settings(secret="my-secret"))
        dispatcher.register("x", "https://example.com/hook")
        event = WebhookEvent(event_type="x", payload={}, tenant_id="t")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True

        captured_headers: dict = {}

        async def capture_post(url, content, headers):
            captured_headers.update(headers)
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = capture_post
            mock_client_cls.return_value = mock_client

            await dispatcher.dispatch(event)

        assert "X-Webhook-Signature" in captured_headers
        assert captured_headers["X-Webhook-Signature"].startswith("sha256=")

    @pytest.mark.asyncio
    async def test_no_signature_header_when_no_secret(self):
        dispatcher = WebhookDispatcher(_settings(secret=None))
        dispatcher.register("x", "https://example.com/hook")
        event = WebhookEvent(event_type="x", payload={}, tenant_id="t")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True

        captured_headers: dict = {}

        async def capture_post(url, content, headers):
            captured_headers.update(headers)
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = capture_post
            mock_client_cls.return_value = mock_client

            await dispatcher.dispatch(event)

        assert "X-Webhook-Signature" not in captured_headers


# ─── ConnectorLoader ─────────────────────────────────────────────────────────


class TestConnectorLoader:
    def test_register_and_get_connector(self):
        loader = ConnectorLoader()
        connector = NullAMSConnector()
        loader.register("null", connector)
        assert loader.get("null") is connector

    def test_get_unknown_returns_none(self):
        loader = ConnectorLoader()
        assert loader.get("nonexistent") is None

    def test_list_connectors_empty(self):
        loader = ConnectorLoader()
        assert loader.list_connectors() == []

    def test_list_connectors_sorted(self):
        loader = ConnectorLoader()
        loader.register("zzz", NullAMSConnector())
        loader.register("aaa", NullAMSConnector())
        loader.register("mmm", NullAMSConnector())
        assert loader.list_connectors() == ["aaa", "mmm", "zzz"]

    def test_register_non_protocol_raises_type_error(self):
        loader = ConnectorLoader()

        class BadConnector:
            """Does not implement AMSConnector protocol."""

            pass

        with pytest.raises(TypeError, match="AMSConnector protocol"):
            loader.register("bad", BadConnector())  # type: ignore[arg-type]

    def test_register_overwrites_previous(self):
        loader = ConnectorLoader()
        c1 = NullAMSConnector()
        c2 = NullAMSConnector()
        loader.register("null", c1)
        loader.register("null", c2)
        assert loader.get("null") is c2


# ─── NullAMSConnector ────────────────────────────────────────────────────────


class TestNullAMSConnector:
    def test_has_name_and_version(self):
        connector = NullAMSConnector()
        assert connector.name == "null"
        assert connector.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_push_policy_returns_noop_status(self):
        connector = NullAMSConnector()
        result = await connector.push_policy({"policy_number": "P-001", "premium": 5000})
        assert result["status"] == "noop"
        assert result["connector"] == "null"

    @pytest.mark.asyncio
    async def test_push_policy_empty_payload(self):
        connector = NullAMSConnector()
        result = await connector.push_policy({})
        assert result["status"] == "noop"

    @pytest.mark.asyncio
    async def test_pull_accounts_returns_empty_list(self):
        connector = NullAMSConnector()
        accounts = await connector.pull_accounts("tenant-1")
        assert accounts == []

    def test_null_connector_satisfies_protocol(self):
        """Verify NullAMSConnector is recognized as an AMSConnector."""
        connector = NullAMSConnector()
        assert isinstance(connector, AMSConnector)


# ─── ModuleRegistry integration ──────────────────────────────────────────────


class TestModuleRegistryAMSConnector:
    def test_register_and_get_via_registry(self):
        from ai_ready_rag.modules.registry import ModuleRegistry

        registry = ModuleRegistry()
        connector = NullAMSConnector()
        registry.register_ams_connector("null", connector)
        assert registry.get_ams_connector("null") is connector

    def test_get_unknown_connector_returns_none(self):
        from ai_ready_rag.modules.registry import ModuleRegistry

        registry = ModuleRegistry()
        assert registry.get_ams_connector("missing") is None

    def test_list_ams_connectors(self):
        from ai_ready_rag.modules.registry import ModuleRegistry

        registry = ModuleRegistry()
        registry.register_ams_connector("epic", NullAMSConnector())
        registry.register_ams_connector("ams360", NullAMSConnector())
        connectors = registry.list_ams_connectors()
        assert "epic" in connectors
        assert "ams360" in connectors

    def test_registry_ams_connectors_isolated_from_other_registries(self):
        """Each ModuleRegistry instance has its own connector dict."""
        from ai_ready_rag.modules.registry import ModuleRegistry

        r1 = ModuleRegistry()
        r2 = ModuleRegistry()
        r1.register_ams_connector("null", NullAMSConnector())
        assert r2.get_ams_connector("null") is None
