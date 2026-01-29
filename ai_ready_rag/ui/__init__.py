"""UI package for Gradio-based web interface."""

from ai_ready_rag.ui.api_client import GradioAPIClient, handle_api_error
from ai_ready_rag.ui.components import (
    format_assistant_message,
    format_citations,
    format_confidence_bar,
    format_routing_alert,
    parse_assistant_response,
)
from ai_ready_rag.ui.gradio_app import create_app
from ai_ready_rag.ui.theme import custom_css, theme

__all__ = [
    "create_app",
    "theme",
    "custom_css",
    "GradioAPIClient",
    "handle_api_error",
    "format_assistant_message",
    "format_citations",
    "format_confidence_bar",
    "format_routing_alert",
    "parse_assistant_response",
]
