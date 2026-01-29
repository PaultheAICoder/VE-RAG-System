"""Reusable UI components for rendering chat messages."""

from typing import Any


def format_confidence_bar(confidence: dict[str, Any] | None) -> str:
    """Render confidence bar as HTML.

    Args:
        confidence: Dict with 'overall' key (0-100) and optionally
                   'retrieval', 'coverage', 'llm' breakdown.

    Returns:
        HTML string for confidence bar visualization.
    """
    if not confidence:
        return ""

    overall = confidence.get("overall", 0)

    # Determine color class based on confidence level
    if overall >= 70:
        color_class = "confidence-high"
    elif overall >= 40:
        color_class = "confidence-medium"
    else:
        color_class = "confidence-low"

    return f"""
<div style="margin-top: 0.5rem;">
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #6b7280;">
        <span>Confidence</span>
        <span>{overall}%</span>
    </div>
    <div class="confidence-bar">
        <div class="confidence-fill {color_class}" style="width: {overall}%;"></div>
    </div>
</div>
"""


def format_citations(sources: list[dict[str, Any]] | None) -> str:
    """Render expandable citations as HTML.

    Args:
        sources: List of source dicts with 'source_id', 'title', 'snippet' keys.

    Returns:
        HTML string with expandable citation sections.
    """
    if not sources:
        return ""

    citations_html = []
    for source in sources:
        title = source.get("title") or source.get("document_id", "Unknown Source")
        snippet = source.get("snippet", "")
        source_id = source.get("source_id", "")

        citation_html = f"""
<details style="margin-bottom: 0.5rem;">
    <summary class="citation-header" style="cursor: pointer; font-size: 0.875rem;">
        {title}
    </summary>
    <div class="citation-content">
        <p style="margin: 0; white-space: pre-wrap;">{snippet}</p>
        <small style="color: #9ca3af;">Source ID: {source_id}</small>
    </div>
</details>
"""
        citations_html.append(citation_html)

    return f"""
<div style="margin-top: 0.75rem; border-top: 1px solid #e5e7eb; padding-top: 0.5rem;">
    <div style="font-size: 0.75rem; color: #6b7280; margin-bottom: 0.25rem;">Sources:</div>
    {"".join(citations_html)}
</div>
"""


def format_routing_alert(
    was_routed: bool,
    routed_to: str | None,
    route_reason: str | None,
) -> str:
    """Render routing alert box when question was routed to expert.

    Args:
        was_routed: Whether the question was routed.
        routed_to: Email or identifier of the expert.
        route_reason: Reason for routing.

    Returns:
        HTML string for routing alert, or empty string if not routed.
    """
    if not was_routed:
        return ""

    return f"""
<div class="routing-alert" style="margin-top: 0.5rem;">
    <div style="font-weight: 600; margin-bottom: 0.25rem;">Routed to Expert</div>
    <div style="font-size: 0.875rem;">
        This question has been forwarded to: <strong>{routed_to or "an expert"}</strong>
    </div>
    {f'<div style="font-size: 0.75rem; color: #9ca3af; margin-top: 0.25rem;">Reason: {route_reason}</div>' if route_reason else ""}
</div>
"""


def format_assistant_message(
    content: str,
    sources: list[dict[str, Any]] | None = None,
    confidence: dict[str, Any] | None = None,
    was_routed: bool = False,
    routed_to: str | None = None,
    route_reason: str | None = None,
) -> str:
    """Format a complete assistant message with all components.

    Args:
        content: The main message content.
        sources: List of citation sources.
        confidence: Confidence scores dict.
        was_routed: Whether the question was routed.
        routed_to: Expert email/identifier if routed.
        route_reason: Reason for routing if routed.

    Returns:
        Formatted HTML string for display in Chatbot.
    """
    parts = [content]

    # Add citations if present
    citations = format_citations(sources)
    if citations:
        parts.append(citations)

    # Add confidence bar
    confidence_bar = format_confidence_bar(confidence)
    if confidence_bar:
        parts.append(confidence_bar)

    # Add routing alert if routed
    routing = format_routing_alert(was_routed, routed_to, route_reason)
    if routing:
        parts.append(routing)

    return "".join(parts)


def parse_assistant_response(response: dict[str, Any]) -> str:
    """Parse API response and format for display.

    Args:
        response: The assistant_message dict from the API response.

    Returns:
        Formatted HTML string.
    """
    content = response.get("content", "")
    sources = response.get("sources")
    confidence = response.get("confidence")
    was_routed = response.get("was_routed", False)
    routed_to = response.get("routed_to")
    route_reason = response.get("route_reason")

    return format_assistant_message(
        content=content,
        sources=sources,
        confidence=confidence,
        was_routed=was_routed,
        routed_to=routed_to,
        route_reason=route_reason,
    )
