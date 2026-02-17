"""Tests for EmailProcessor - .msg email extraction and auto-tagging."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_ready_rag.services.auto_tagging.email_processor import (
    MAX_FILE_SIZE,
    EmailProcessingResult,
    EmailProcessor,
)
from ai_ready_rag.services.auto_tagging.models import AutoTag


@pytest.fixture()
def mock_strategy():
    """Create a mock AutoTagStrategy with parse_email_subject."""
    strategy = MagicMock()
    strategy.id = "test-strategy"
    strategy.version = "1.0"
    strategy.parse_email_subject = MagicMock(return_value=[])
    return strategy


@pytest.fixture()
def processor(mock_strategy):
    """Create an EmailProcessor with a mock strategy."""
    return EmailProcessor(mock_strategy)


def _make_mock_msg(
    subject="RE: Insurance Renewal Notice",
    sender="alice@example.com",
    to="bob@example.com",
    cc="charlie@example.com",
    date="2026-01-15 10:30:00",
    body="This is the plain text body of the email.",
    html_body=None,
):
    """Create a mock extract_msg.Message object."""
    msg = MagicMock()
    msg.subject = subject
    msg.sender = sender
    msg.to = to
    msg.cc = cc
    msg.date = date
    msg.body = body
    msg.htmlBody = html_body
    msg.close = MagicMock()
    return msg


def _make_mock_path(name="test.msg", size=1024):
    """Create a mock Path with stat()."""
    mock_path = MagicMock(spec=Path)
    mock_stat = MagicMock()
    mock_stat.st_size = size
    mock_path.stat.return_value = mock_stat
    mock_path.name = name
    return mock_path


def _patch_extract_msg_available(mod, value=True):
    """Context manager helper to set _EXTRACT_MSG_AVAILABLE."""

    class _Ctx:
        def __init__(self):
            self.original = mod._EXTRACT_MSG_AVAILABLE

        def __enter__(self):
            mod._EXTRACT_MSG_AVAILABLE = value
            return self

        def __exit__(self, *args):
            mod._EXTRACT_MSG_AVAILABLE = self.original

    return _Ctx()


@pytest.fixture()
def enable_extract_msg():
    """Enable extract-msg availability flag and provide mock module."""
    import ai_ready_rag.services.auto_tagging.email_processor as mod

    original = mod._EXTRACT_MSG_AVAILABLE

    # Create a mock extract_msg module
    mock_extract_msg = MagicMock()

    mod._EXTRACT_MSG_AVAILABLE = True
    with patch.dict(sys.modules, {"extract_msg": mock_extract_msg}):
        yield mock_extract_msg

    mod._EXTRACT_MSG_AVAILABLE = original


class TestEmailProcessingResult:
    """Test the EmailProcessingResult dataclass."""

    def test_default_values(self):
        result = EmailProcessingResult(success=True)
        assert result.success is True
        assert result.subject == ""
        assert result.sender == ""
        assert result.to == ""
        assert result.cc == ""
        assert result.date == ""
        assert result.body_text == ""
        assert result.tags == []
        assert result.metadata == {}
        assert result.error is None

    def test_with_values(self):
        tag = AutoTag(namespace="doctype", value="correspondence", source="email")
        result = EmailProcessingResult(
            success=True,
            subject="Test",
            sender="a@b.com",
            tags=[tag],
            metadata={"from": "a@b.com"},
        )
        assert result.subject == "Test"
        assert len(result.tags) == 1
        assert result.metadata["from"] == "a@b.com"


class TestExtractMsgNotInstalled:
    """Test behavior when extract-msg is not available."""

    def test_returns_error_when_not_installed(self, processor):
        """When extract-msg is not installed, process returns error with default tag."""
        import ai_ready_rag.services.auto_tagging.email_processor as mod

        with _patch_extract_msg_available(mod, False):
            result = processor.process(Path("/tmp/test.msg"))
            assert result.success is False
            assert result.error == "extract-msg not installed"
            assert len(result.tags) == 1
            assert result.tags[0].namespace == "doctype"
            assert result.tags[0].value == "correspondence"
            assert result.tags[0].source == "email"


class TestFileSizeLimit:
    """Test 10MB file size enforcement."""

    def test_file_too_large(self, processor, enable_extract_msg):
        """Files exceeding 10MB are rejected with default tag."""
        mock_path = _make_mock_path("large.msg", MAX_FILE_SIZE + 1)

        result = processor.process(mock_path)
        assert result.success is False
        assert "10MB" in result.error
        assert len(result.tags) == 1
        assert result.tags[0].namespace == "doctype"
        assert result.tags[0].value == "correspondence"

    def test_file_stat_error(self, processor, enable_extract_msg):
        """OSError on stat is handled gracefully."""
        mock_path = MagicMock(spec=Path)
        mock_path.stat.side_effect = OSError("Permission denied")
        mock_path.name = "noaccess.msg"

        result = processor.process(mock_path)
        assert result.success is False
        assert "Permission denied" in result.error


class TestSuccessfulExtraction:
    """Test successful .msg file processing."""

    def test_extracts_all_metadata(self, processor, enable_extract_msg):
        """All metadata fields are extracted correctly."""
        mock_msg = _make_mock_msg()
        enable_extract_msg.Message.return_value = mock_msg
        mock_path = _make_mock_path()

        result = processor.process(mock_path)

        assert result.success is True
        assert result.subject == "RE: Insurance Renewal Notice"
        assert result.sender == "alice@example.com"
        assert result.to == "bob@example.com"
        assert result.cc == "charlie@example.com"
        assert result.date == "2026-01-15 10:30:00"
        assert result.body_text == "This is the plain text body of the email."
        mock_msg.close.assert_called_once()

    def test_metadata_dict_structure(self, processor, enable_extract_msg):
        """Metadata dict contains from, to, cc, subject, date keys."""
        mock_msg = _make_mock_msg()
        enable_extract_msg.Message.return_value = mock_msg
        mock_path = _make_mock_path()

        result = processor.process(mock_path)

        assert set(result.metadata.keys()) == {"from", "to", "cc", "subject", "date"}
        assert result.metadata["from"] == "alice@example.com"
        assert result.metadata["to"] == "bob@example.com"
        assert result.metadata["cc"] == "charlie@example.com"
        assert result.metadata["subject"] == "RE: Insurance Renewal Notice"
        assert result.metadata["date"] == "2026-01-15 10:30:00"


class TestHTMLBodySanitization:
    """Test HTML body sanitization."""

    def test_html_stripped_to_plain_text(self):
        """HTML tags are stripped, returning plain text."""
        html = "<html><body><p>Hello <b>World</b></p></body></html>"
        result = EmailProcessor._strip_html(html)
        assert "Hello" in result
        assert "World" in result
        assert "<" not in result

    def test_script_and_style_removed(self):
        """Script and style tags are removed completely."""
        html = (
            "<html><head><style>body{color:red}</style></head>"
            "<body><script>alert('xss')</script><p>Safe content</p></body></html>"
        )
        result = EmailProcessor._strip_html(html)
        assert "Safe content" in result
        assert "alert" not in result
        assert "color:red" not in result

    def test_img_object_embed_removed(self):
        """Image, object, and embed tags are removed."""
        html = (
            '<p>Text before</p><img src="logo.png"/>'
            '<object data="file.swf"></object>'
            '<embed src="video.mp4">'
            "<p>Text after</p>"
        )
        result = EmailProcessor._strip_html(html)
        assert "Text before" in result
        assert "Text after" in result
        assert "logo.png" not in result

    def test_plain_text_body_preferred(self, processor, enable_extract_msg):
        """When plain text body exists, it is used over HTML."""
        mock_msg = _make_mock_msg(
            body="Plain text content",
            html_body="<html><body><p>HTML content</p></body></html>",
        )
        enable_extract_msg.Message.return_value = mock_msg
        mock_path = _make_mock_path()

        result = processor.process(mock_path)
        assert result.body_text == "Plain text content"

    def test_html_fallback_when_no_plain_text(self, processor, enable_extract_msg):
        """When plain text body is empty, HTML body is sanitized and used."""
        mock_msg = _make_mock_msg(
            body="",
            html_body="<html><body><p>HTML only content</p></body></html>",
        )
        enable_extract_msg.Message.return_value = mock_msg
        mock_path = _make_mock_path()

        result = processor.process(mock_path)
        assert "HTML only content" in result.body_text
        assert "<" not in result.body_text

    def test_bytes_html_body_decoded(self, processor, enable_extract_msg):
        """htmlBody returned as bytes is decoded to string."""
        mock_msg = _make_mock_msg(
            body="",
            html_body=b"<html><body><p>Bytes body</p></body></html>",
        )
        enable_extract_msg.Message.return_value = mock_msg
        mock_path = _make_mock_path()

        result = processor.process(mock_path)
        assert "Bytes body" in result.body_text


class TestEmailPatterns:
    """Test email subject pattern matching and tagging."""

    def test_patterns_applied_from_strategy(self, mock_strategy, enable_extract_msg):
        """Strategy's parse_email_subject is called with the subject."""
        pattern_tag = AutoTag(
            namespace="doctype",
            value="renewal",
            source="email",
            confidence=1.0,
            strategy_id="test-strategy",
            strategy_version="1.0",
        )
        mock_strategy.parse_email_subject.return_value = [pattern_tag]
        processor = EmailProcessor(mock_strategy)

        mock_msg = _make_mock_msg(subject="RE: Policy Renewal", cc=None)
        enable_extract_msg.Message.return_value = mock_msg
        mock_path = _make_mock_path("renewal.msg")

        result = processor.process(mock_path)

        mock_strategy.parse_email_subject.assert_called_once_with("RE: Policy Renewal")
        assert result.success is True
        # doctype:renewal from patterns, no default correspondence added
        assert len(result.tags) == 1
        assert result.tags[0].value == "renewal"

    def test_default_correspondence_tag_when_no_doctype(self, mock_strategy, enable_extract_msg):
        """Default doctype:correspondence added when no doctype pattern matches."""
        non_doctype_tag = AutoTag(
            namespace="department",
            value="hr",
            source="email",
            confidence=1.0,
        )
        mock_strategy.parse_email_subject.return_value = [non_doctype_tag]
        processor = EmailProcessor(mock_strategy)

        mock_msg = _make_mock_msg(subject="Hello", cc=None)
        enable_extract_msg.Message.return_value = mock_msg
        mock_path = _make_mock_path()

        result = processor.process(mock_path)

        assert len(result.tags) == 2
        doctype_tags = [t for t in result.tags if t.namespace == "doctype"]
        assert len(doctype_tags) == 1
        assert doctype_tags[0].value == "correspondence"


class TestMalformedMsg:
    """Test handling of malformed .msg files."""

    def test_malformed_msg_returns_fallback(self, processor, enable_extract_msg):
        """Malformed .msg files are handled gracefully with default tag."""
        enable_extract_msg.Message.side_effect = Exception("Invalid OLE file")
        mock_path = _make_mock_path("corrupt.msg")

        result = processor.process(mock_path)

        assert result.success is False
        assert "Invalid OLE file" in result.error
        assert len(result.tags) == 1
        assert result.tags[0].namespace == "doctype"
        assert result.tags[0].value == "correspondence"

    def test_msg_close_called_on_extraction_error(self, processor, enable_extract_msg):
        """msg.close() is called even when metadata extraction raises."""
        mock_msg = MagicMock()
        # Make subject access raise an exception
        type(mock_msg).subject = property(
            lambda self: (_ for _ in ()).throw(Exception("Bad field"))
        )
        mock_msg.close = MagicMock()
        enable_extract_msg.Message.return_value = mock_msg
        mock_path = _make_mock_path("bad.msg")

        result = processor.process(mock_path)

        assert result.success is False
        mock_msg.close.assert_called_once()
