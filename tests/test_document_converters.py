"""Tests for image and email document converters (issues #204, #207)."""

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_ready_rag.services.chunker_simple import SimpleChunker

# ---------------------------------------------------------------------------
# Image converter tests (#204)
# ---------------------------------------------------------------------------


class TestImageExtraction:
    """Tests for image OCR extraction in SimpleChunker."""

    def test_image_raises_when_ocr_disabled(self, tmp_path):
        """Image extraction raises ValueError when OCR is disabled."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n")

        chunker = SimpleChunker(enable_ocr=False)

        with pytest.raises(ValueError, match="OCR is disabled"):
            chunker.chunk_document(str(img_file))

    def test_image_error_message_includes_filename(self, tmp_path):
        """Error message includes the image filename for clarity."""
        img_file = tmp_path / "report_scan.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0")

        chunker = SimpleChunker(enable_ocr=False)

        with pytest.raises(ValueError, match="report_scan.jpg"):
            chunker.chunk_document(str(img_file))

    def test_image_extraction_with_ocr(self, tmp_path):
        """Image extraction works when OCR is enabled (mocked)."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n")

        mock_image = MagicMock()
        mock_image.size = (800, 600)

        mock_pytesseract = MagicMock()
        mock_pytesseract.image_to_string.return_value = "Extracted OCR text from image"

        mock_pil_image = MagicMock()
        mock_pil_image.open.return_value = mock_image
        mock_pil_image.LANCZOS = 1

        mock_pil = MagicMock()
        mock_pil.Image = mock_pil_image

        chunker = SimpleChunker(enable_ocr=True, ocr_language="eng")

        with patch.dict(
            "sys.modules",
            {
                "pytesseract": mock_pytesseract,
                "PIL": mock_pil,
                "PIL.Image": mock_pil_image,
            },
        ):
            text = chunker._extract_image(Path(img_file))

        assert text == "Extracted OCR text from image"
        mock_pytesseract.image_to_string.assert_called_once_with(mock_image, lang="eng")

    def test_large_image_is_resized(self, tmp_path):
        """Images exceeding MAX_IMAGE_PIXELS are resized."""
        img_file = tmp_path / "huge.tiff"
        img_file.write_bytes(b"II\x2a\x00")

        mock_image = MagicMock()
        mock_image.size = (6000, 4000)  # 24MP > 20MP threshold
        mock_resized = MagicMock()
        mock_image.resize.return_value = mock_resized

        mock_pytesseract = MagicMock()
        mock_pytesseract.image_to_string.return_value = "Resized image text"

        mock_pil_image = MagicMock()
        mock_pil_image.open.return_value = mock_image
        mock_pil_image.LANCZOS = 1

        mock_pil = MagicMock()
        mock_pil.Image = mock_pil_image

        chunker = SimpleChunker(enable_ocr=True)

        with patch.dict(
            "sys.modules",
            {
                "pytesseract": mock_pytesseract,
                "PIL": mock_pil,
                "PIL.Image": mock_pil_image,
            },
        ):
            text = chunker._extract_image(Path(img_file))

        assert text == "Resized image text"
        mock_image.resize.assert_called_once()
        # Verify resized image was passed to OCR, not original
        mock_pytesseract.image_to_string.assert_called_once_with(mock_resized, lang="eng")

    def test_all_image_extensions_route_correctly(self):
        """All image extensions route to _extract_image."""
        chunker = SimpleChunker(enable_ocr=True)

        for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
            with patch.object(chunker, "_extract_image", return_value="text") as mock:
                chunker._extract_text(Path(f"/fake/file{ext}"))
                mock.assert_called_once()


# ---------------------------------------------------------------------------
# Email converter tests (#207)
# ---------------------------------------------------------------------------


class TestEmlExtraction:
    """Tests for .eml email extraction in SimpleChunker."""

    def _create_plain_eml(self, tmp_path: Path, body: str = "Hello, this is a test email.") -> Path:
        """Create a plain text .eml file."""
        msg = MIMEText(body)
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Date"] = "Mon, 10 Feb 2026 10:00:00 +0000"
        msg["Subject"] = "Test Email Subject"

        eml_file = tmp_path / "test.eml"
        eml_file.write_bytes(msg.as_bytes())
        return eml_file

    def _create_html_eml(self, tmp_path: Path) -> Path:
        """Create an HTML-only .eml file."""
        html_body = "<html><body><h1>Title</h1><p>HTML email content.</p></body></html>"
        msg = MIMEText(html_body, "html")
        msg["From"] = "html@example.com"
        msg["To"] = "user@example.com"
        msg["Subject"] = "HTML Email"

        eml_file = tmp_path / "html.eml"
        eml_file.write_bytes(msg.as_bytes())
        return eml_file

    def _create_multipart_eml(self, tmp_path: Path) -> Path:
        """Create a multipart .eml with both plain and HTML."""
        msg = MIMEMultipart("alternative")
        msg["From"] = "multi@example.com"
        msg["To"] = "user@example.com"
        msg["Subject"] = "Multipart Email"

        plain_part = MIMEText("Plain text version.", "plain")
        html_part = MIMEText("<p>HTML version.</p>", "html")
        msg.attach(plain_part)
        msg.attach(html_part)

        eml_file = tmp_path / "multipart.eml"
        eml_file.write_bytes(msg.as_bytes())
        return eml_file

    def test_eml_extracts_headers(self, tmp_path):
        """EML extraction includes From, To, Date, Subject headers."""
        eml_file = self._create_plain_eml(tmp_path)
        chunker = SimpleChunker()
        text = chunker._extract_eml(eml_file)

        assert "From: sender@example.com" in text
        assert "To: recipient@example.com" in text
        assert "Date:" in text
        assert "Subject: Test Email Subject" in text

    def test_eml_extracts_plain_body(self, tmp_path):
        """EML extraction includes plain text body."""
        eml_file = self._create_plain_eml(tmp_path, body="Important meeting notes.")
        chunker = SimpleChunker()
        text = chunker._extract_eml(eml_file)

        assert "Important meeting notes." in text

    def test_eml_html_body_stripped(self, tmp_path):
        """HTML-only EML body has tags stripped."""
        eml_file = self._create_html_eml(tmp_path)
        chunker = SimpleChunker()
        text = chunker._extract_eml(eml_file)

        assert "HTML email content." in text
        assert "<html>" not in text
        assert "<p>" not in text

    def test_eml_multipart_prefers_plain(self, tmp_path):
        """Multipart EML prefers plain text over HTML."""
        eml_file = self._create_multipart_eml(tmp_path)
        chunker = SimpleChunker()
        text = chunker._extract_eml(eml_file)

        assert "Plain text version." in text

    def test_eml_chunk_document_integration(self, tmp_path):
        """Full pipeline: EML file -> chunk_document -> chunks with text."""
        eml_file = self._create_plain_eml(tmp_path, body="Document content. " * 100)
        chunker = SimpleChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_document(str(eml_file))

        assert len(chunks) > 0
        assert chunks[0]["source"] == "test.eml"
        full_text = " ".join(c["text"] for c in chunks)
        assert "sender@example.com" in full_text

    def test_eml_routes_correctly(self):
        """Extension .eml routes to _extract_eml."""
        chunker = SimpleChunker()
        with patch.object(chunker, "_extract_eml", return_value="text") as mock:
            chunker._extract_text(Path("/fake/mail.eml"))
            mock.assert_called_once()


class TestMsgExtraction:
    """Tests for .msg email extraction in SimpleChunker."""

    def test_msg_extracts_headers_and_body(self, tmp_path):
        """MSG extraction includes headers and body."""
        msg_file = tmp_path / "outlook.msg"
        msg_file.write_bytes(b"\xd0\xcf\x11\xe0")

        mock_msg = MagicMock()
        mock_msg.sender = "outlook@example.com"
        mock_msg.to = "user@example.com"
        mock_msg.date = "2026-02-10"
        mock_msg.subject = "Outlook Email"
        mock_msg.body = "This is the email body."

        mock_extract_msg = MagicMock()
        mock_extract_msg.Message.return_value = mock_msg

        chunker = SimpleChunker()

        with patch.dict("sys.modules", {"extract_msg": mock_extract_msg}):
            text = chunker._extract_msg(Path(msg_file))

        assert "From: outlook@example.com" in text
        assert "To: user@example.com" in text
        assert "Subject: Outlook Email" in text
        assert "This is the email body." in text
        mock_msg.close.assert_called_once()

    def test_msg_handles_none_fields(self, tmp_path):
        """MSG extraction handles None fields gracefully."""
        msg_file = tmp_path / "empty.msg"
        msg_file.write_bytes(b"\xd0\xcf\x11\xe0")

        mock_msg = MagicMock()
        mock_msg.sender = None
        mock_msg.to = None
        mock_msg.date = None
        mock_msg.subject = None
        mock_msg.body = None

        mock_extract_msg = MagicMock()
        mock_extract_msg.Message.return_value = mock_msg

        chunker = SimpleChunker()

        with patch.dict("sys.modules", {"extract_msg": mock_extract_msg}):
            text = chunker._extract_msg(Path(msg_file))

        assert "From: Unknown" in text
        assert "Subject: No Subject" in text

    def test_msg_routes_correctly(self):
        """Extension .msg routes to _extract_msg."""
        chunker = SimpleChunker()
        with patch.object(chunker, "_extract_msg", return_value="text") as mock:
            chunker._extract_text(Path("/fake/mail.msg"))
            mock.assert_called_once()


# ---------------------------------------------------------------------------
# MIME type and config tests
# ---------------------------------------------------------------------------


class TestMimeTypeConfig:
    """Tests for MIME type mappings and config extensions."""

    def test_image_mime_types(self):
        """Image MIME types are correctly mapped."""
        from ai_ready_rag.utils.file_utils import get_mime_type

        assert get_mime_type("png") == "image/png"
        assert get_mime_type("jpg") == "image/jpeg"
        assert get_mime_type("jpeg") == "image/jpeg"
        assert get_mime_type("tiff") == "image/tiff"
        assert get_mime_type("tif") == "image/tiff"

    def test_email_mime_types(self):
        """Email MIME types are correctly mapped."""
        from ai_ready_rag.utils.file_utils import get_mime_type

        assert get_mime_type("eml") == "message/rfc822"
        assert get_mime_type("msg") == "application/vnd.ms-outlook"

    def test_image_extensions_allowed(self):
        """Image extensions are in allowed_extensions."""
        from ai_ready_rag.config import Settings

        settings = Settings()
        for ext in ("png", "jpg", "jpeg", "tiff", "tif"):
            assert ext in settings.allowed_extensions, f"{ext} not in allowed_extensions"

    def test_email_extensions_allowed(self):
        """Email extensions are in allowed_extensions."""
        from ai_ready_rag.config import Settings

        settings = Settings()
        for ext in ("eml", "msg"):
            assert ext in settings.allowed_extensions, f"{ext} not in allowed_extensions"

    def test_file_extension_validation(self):
        """New extensions pass validation."""
        from ai_ready_rag.config import Settings
        from ai_ready_rag.utils.file_utils import validate_file_extension

        settings = Settings()
        exts = settings.allowed_extensions

        assert validate_file_extension("scan.png", exts) is True
        assert validate_file_extension("photo.jpg", exts) is True
        assert validate_file_extension("archive.tiff", exts) is True
        assert validate_file_extension("mail.eml", exts) is True
        assert validate_file_extension("outlook.msg", exts) is True
        assert validate_file_extension("virus.exe", exts) is False


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFactoryOcrPassthrough:
    """Tests that factory passes OCR settings to SimpleChunker."""

    def test_simple_chunker_receives_ocr_settings(self):
        """Factory passes enable_ocr and ocr_language to SimpleChunker."""
        from ai_ready_rag.config import Settings
        from ai_ready_rag.services.chunker_simple import SimpleChunker as SC
        from ai_ready_rag.services.factory import get_chunker

        settings = Settings(
            env_profile="laptop",
            chunker_backend="simple",
            enable_ocr=True,
            ocr_language="fra",
        )
        chunker = get_chunker(settings)

        assert isinstance(chunker, SC)
        assert chunker.enable_ocr is True
        assert chunker.ocr_language == "fra"

    def test_simple_chunker_default_ocr_disabled(self):
        """Factory defaults OCR to disabled for SimpleChunker."""
        from ai_ready_rag.config import Settings
        from ai_ready_rag.services.chunker_simple import SimpleChunker as SC
        from ai_ready_rag.services.factory import get_chunker

        settings = Settings(env_profile="laptop", chunker_backend="simple")
        chunker = get_chunker(settings)

        assert isinstance(chunker, SC)
        assert chunker.enable_ocr is False
