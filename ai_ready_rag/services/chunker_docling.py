"""Docling chunker for production/Spark deployments."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DoclingChunker:
    """Full-featured chunker using Docling + HybridChunker.

    Implements ChunkerProtocol for spark profile.
    Supports OCR, table extraction, and semantic chunking.
    """

    def __init__(
        self,
        enable_ocr: bool = True,
        ocr_language: str = "eng",
        max_tokens: int = 512,
        force_full_page_ocr: bool = False,
        table_extraction_mode: str = "accurate",
        include_image_descriptions: bool = True,
    ):
        self.enable_ocr = enable_ocr
        self.ocr_language = ocr_language
        self.max_tokens = max_tokens
        self.force_full_page_ocr = force_full_page_ocr
        self.table_extraction_mode = table_extraction_mode
        self.include_image_descriptions = include_image_descriptions

        self._converter = None
        self._chunker = None
        self._tokenizer = None

    def _get_converter(self):
        """Lazy-load Docling converter."""
        if self._converter is None:
            self._converter = self._create_converter()
        return self._converter

    def _get_chunker(self):
        """Lazy-load HybridChunker."""
        if self._chunker is None:
            self._chunker, self._tokenizer = self._create_chunker()
        return self._chunker, self._tokenizer

    def _create_converter(self):
        """Create configured Docling document converter."""
        try:
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                TableStructureOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            table_options = TableStructureOptions(
                do_cell_matching=True,
                mode=self.table_extraction_mode,
            )

            pipeline_options = PdfPipelineOptions(
                do_ocr=self.enable_ocr,
                do_table_structure=True,
                table_structure_options=table_options,
                generate_picture_images=self.include_image_descriptions,
            )

            if self.enable_ocr:
                try:
                    from docling.datamodel.pipeline_options import TesseractOcrOptions

                    ocr_options = TesseractOcrOptions(
                        lang=[self.ocr_language],
                        force_full_page_ocr=self.force_full_page_ocr,
                    )
                    pipeline_options.ocr_options = ocr_options
                except ImportError:
                    logger.warning("TesseractOcrOptions not available, OCR disabled")

            return DocumentConverter(
                format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
            )
        except ImportError as e:
            logger.error(f"Docling not available: {e}")
            raise ImportError(
                "Docling is required for spark profile. Install with: pip install docling"
            ) from e

    def _create_chunker(self):
        """Create HybridChunker with nomic tokenizer."""
        try:
            from docling.chunking import HybridChunker
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                "nomic-ai/nomic-embed-text-v1",
                trust_remote_code=False,
            )

            chunker = HybridChunker(
                tokenizer=tokenizer,
                max_tokens=self.max_tokens,
                merge_peers=True,
            )

            return chunker, tokenizer
        except ImportError as e:
            logger.error(f"HybridChunker not available: {e}")
            raise ImportError(
                "HybridChunker requires docling and transformers. "
                "Install with: pip install docling transformers"
            ) from e

    def chunk_document(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Parse with Docling and chunk with HybridChunker.

        Args:
            file_path: Path to document file
            metadata: Optional metadata to include in chunks

        Returns:
            List of chunk dicts with text, chunk_index, page_number, section
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        converter = self._get_converter()
        chunker, tokenizer = self._get_chunker()

        # Convert document
        result = converter.convert(str(path))

        # Chunk with HybridChunker
        chunks = []
        for i, chunk in enumerate(chunker.chunk(result.document)):
            # Extract metadata from chunk
            page_number = None
            section = None
            if hasattr(chunk, "meta") and chunk.meta:
                page_number = chunk.meta.get("page_number")
                headings = chunk.meta.get("headings", [])
                section = headings[-1] if headings else None

            chunk_dict = {
                "text": chunk.text,
                "chunk_index": i,
                "page_number": page_number,
                "section": section,
                "source": path.name,
            }

            if metadata:
                chunk_dict.update(metadata)

            chunks.append(chunk_dict)

        logger.info(f"Chunked {path.name} with Docling: {len(chunks)} chunks")
        return chunks

    def chunk_text(
        self,
        text: str,
        source: str = "unknown",
    ) -> list[dict[str, Any]]:
        """Chunk plain text (fallback - uses simple splitting).

        For plain text, Docling's HybridChunker doesn't add value,
        so we use a simple token-based approach.

        Args:
            text: Text content to chunk
            source: Source identifier for metadata

        Returns:
            List of chunk dicts
        """
        if not text or not text.strip():
            return []

        # Use tokenizer for accurate token counting
        _, tokenizer = self._get_chunker()

        tokens = tokenizer.encode(text)
        chunks = []
        chunk_idx = 0

        # Slide through tokens with overlap
        overlap_tokens = self.max_tokens // 4
        start = 0

        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "chunk_index": chunk_idx,
                    "page_number": None,
                    "section": None,
                    "source": source,
                }
            )

            chunk_idx += 1
            start = end - overlap_tokens if end < len(tokens) else len(tokens)

        return chunks
