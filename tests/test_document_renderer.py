"""Tests for DocumentRenderer."""

import os
import tempfile

from ai_ready_rag.services.document_renderer import DocumentRenderer, RenderResult


class TestDocumentRenderer:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.renderer = DocumentRenderer(output_dir=self.tmp_dir)

    def teardown_method(self):
        import shutil

        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_import(self):
        from ai_ready_rag.services.document_renderer import DocumentRenderer

        assert DocumentRenderer is not None

    def test_render_result_dataclass(self):
        r = RenderResult(output_path="/tmp/test.pdf", format="pdf", file_size_bytes=1024)
        assert r.format == "pdf"
        assert r.file_size_bytes == 1024

    def test_render_pdf_or_html_fallback(self):
        content = {
            "title": "Test Document",
            "sections": [
                {"heading": "Coverage Summary", "body": "Property: $5M"},
                {"heading": "Next Steps", "body": ["Renew by March 1", "Get quotes"]},
            ],
        }
        result = self.renderer.render_pdf(content, "test.pdf")
        assert isinstance(result, RenderResult)
        assert result.format in ("pdf", "html")
        assert os.path.exists(result.output_path)
        assert result.file_size_bytes > 0

    def test_render_pptx_or_html_fallback(self):
        content = {
            "account_name": "Oak Hills HOA",
            "meeting_date": "March 1, 2026",
            "slides": [
                {
                    "type": "coverage_summary",
                    "title": "Coverage Summary",
                    "data": {"total_premium": 12000.0},
                },
                {
                    "type": "compliance_status",
                    "title": "Compliance",
                    "data": {"gap_count": 0},
                },
            ],
        }
        result = self.renderer.render_pptx(content, "test.pptx")
        assert isinstance(result, RenderResult)
        assert result.format in ("pptx", "html")
        assert os.path.exists(result.output_path)

    def test_html_fallback_creates_valid_html(self):
        content = {
            "title": "Test",
            "sections": [{"heading": "Section 1", "body": "Body text"}],
        }
        result = self.renderer._render_html_fallback(
            content, os.path.join(self.tmp_dir, "test.html")
        )
        with open(result.output_path) as f:
            html = f.read()
        assert "<html>" in html
        assert "Test" in html
        assert "Section 1" in html

    def test_flatten_slide_data(self):
        data = {"total_premium": 12000.0, "gap_count": 2, "items": ["a", "b"]}
        bullets = self.renderer._flatten_slide_data(data)
        assert isinstance(bullets, list)
        assert len(bullets) > 0
