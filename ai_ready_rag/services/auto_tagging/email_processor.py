"""Email processor for auto-tagging .msg files.

Extracts metadata from Outlook .msg files, sanitizes HTML bodies,
applies email subject patterns via AutoTagStrategy, and provides
a default doctype:correspondence tag as fallback.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ai_ready_rag.services.auto_tagging.models import AutoTag

logger = logging.getLogger(__name__)

# Lazy-loaded flag for extract-msg availability
_EXTRACT_MSG_AVAILABLE: bool | None = None

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def _check_extract_msg() -> bool:
    """Check if extract-msg is installed (cached)."""
    global _EXTRACT_MSG_AVAILABLE
    if _EXTRACT_MSG_AVAILABLE is None:
        try:
            import extract_msg  # noqa: F401

            _EXTRACT_MSG_AVAILABLE = True
        except ImportError:
            _EXTRACT_MSG_AVAILABLE = False
    return _EXTRACT_MSG_AVAILABLE


@dataclass
class EmailProcessingResult:
    """Result of processing a .msg file for auto-tagging."""

    success: bool
    subject: str = ""
    sender: str = ""
    to: str = ""
    cc: str = ""
    date: str = ""
    body_text: str = ""
    tags: list[AutoTag] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    error: str | None = None


class EmailProcessor:
    """Processes .msg email files for the auto-tagging pipeline.

    Extracts metadata, sanitizes body content, applies email subject
    patterns from the strategy, and adds a default doctype:correspondence
    tag when no doctype pattern matches.
    """

    def __init__(self, strategy: object) -> None:
        """Initialize with an AutoTagStrategy instance.

        Args:
            strategy: An AutoTagStrategy with parse_email_subject() method.
        """
        self.strategy = strategy

    def process(self, file_path: Path) -> EmailProcessingResult:
        """Process a .msg file and return extraction results with auto-tags.

        Args:
            file_path: Path to the .msg file.

        Returns:
            EmailProcessingResult with metadata, sanitized body, and tags.
        """
        if not _check_extract_msg():
            logger.warning(
                "extract-msg not installed; skipping .msg processing for %s. "
                "Install with: pip install extract-msg",
                file_path.name,
            )
            return EmailProcessingResult(
                success=False,
                tags=[self._build_default_tag()],
                error="extract-msg not installed",
            )

        # Check file size
        try:
            file_size = file_path.stat().st_size
        except OSError as e:
            return self._handle_malformed(file_path, f"Cannot stat file: {e}")

        if file_size > MAX_FILE_SIZE:
            return EmailProcessingResult(
                success=False,
                tags=[self._build_default_tag()],
                error=f"File exceeds 10MB limit ({file_size} bytes)",
            )

        # Extract metadata from .msg
        try:
            import extract_msg

            msg = extract_msg.Message(file_path)
        except Exception as e:
            return self._handle_malformed(file_path, str(e))

        try:
            subject = str(msg.subject or "")
            sender = str(msg.sender or "")
            to = str(msg.to or "")
            cc = str(getattr(msg, "cc", None) or "")
            date = str(msg.date or "")
            body_text = self._sanitize_body(msg)

            # Build metadata dict
            metadata = {
                "from": sender,
                "to": to,
                "cc": cc,
                "subject": subject,
                "date": date,
            }

            # Apply email subject patterns from strategy
            tags = self.strategy.parse_email_subject(subject)

            # Add default doctype:correspondence if no doctype tag was matched
            has_doctype = any(t.namespace == "doctype" for t in tags)
            if not has_doctype:
                tags.append(self._build_default_tag())

            return EmailProcessingResult(
                success=True,
                subject=subject,
                sender=sender,
                to=to,
                cc=cc,
                date=date,
                body_text=body_text,
                tags=tags,
                metadata=metadata,
            )
        except Exception as e:
            return self._handle_malformed(file_path, str(e))
        finally:
            msg.close()

    def _sanitize_body(self, msg: object) -> str:
        """Extract and sanitize the email body to plain text.

        Prefers plain text body. Falls back to HTML body with tag stripping.
        Strips script, style, img, object, and embed tags before extracting text.
        """
        # Prefer plain text body
        body = getattr(msg, "body", None)
        if body and body.strip():
            return body.strip()

        # Fall back to HTML body
        html_body = getattr(msg, "htmlBody", None)
        if not html_body:
            return ""

        # Ensure we have a string (htmlBody can be bytes)
        if isinstance(html_body, bytes):
            html_body = html_body.decode("utf-8", errors="replace")

        return self._strip_html(html_body)

    @staticmethod
    def _strip_html(html: str) -> str:
        """Strip HTML tags, returning plain text.

        Uses BeautifulSoup if available, falls back to regex.
        Removes script, style, img, object, and embed elements.
        """
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for element in soup(["script", "style", "img", "object", "embed"]):
                element.decompose()
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            # Regex fallback: remove dangerous tags and their content
            text = re.sub(
                r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE
            )
            # Remove remaining tags
            text = re.sub(r"<[^>]+>", " ", text)
            return re.sub(r"\s+", " ", text).strip()

    def _build_default_tag(self) -> AutoTag:
        """Build the default doctype:correspondence tag."""
        return AutoTag(
            namespace="doctype",
            value="correspondence",
            source="email",
            confidence=1.0,
            strategy_id=getattr(self.strategy, "id", ""),
            strategy_version=getattr(self.strategy, "version", ""),
        )

    def _handle_malformed(self, file_path: Path, error: str) -> EmailProcessingResult:
        """Handle a malformed .msg file gracefully.

        Logs a warning and returns a result with only the default
        doctype:correspondence tag.
        """
        logger.warning(
            "Malformed .msg file '%s': %s. Tagging as doctype:correspondence.",
            file_path.name,
            error,
        )
        return EmailProcessingResult(
            success=False,
            tags=[self._build_default_tag()],
            error=error,
        )
