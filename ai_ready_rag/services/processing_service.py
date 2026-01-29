"""Document processing service with Docling integration."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from ai_ready_rag.config import Settings
from ai_ready_rag.db.models import Document
from ai_ready_rag.services.vector_service import VectorService

logger = logging.getLogger(__name__)

# Plain text file types that use fallback chunker
PLAIN_TEXT_EXTENSIONS = {"txt", "md", "csv"}


@dataclass
class ChunkInfo:
    """Information about a single chunk."""

    text: str
    chunk_index: int
    page_number: int | None
    section: str | None
    token_count: int


@dataclass
class ProcessingResult:
    """Result of document processing."""

    success: bool
    chunk_count: int
    page_count: int | None
    word_count: int
    processing_time_ms: int
    error_message: str | None = None


class ProcessingService:
    """Document processing with Docling and vector indexing.

    Handles document parsing, chunking, metadata extraction, and
    vector indexing via VectorService.
    """

    def __init__(
        self,
        vector_service: VectorService,
        settings: Settings,
    ):
        """Initialize processing service.

        Args:
            vector_service: Vector service for indexing chunks.
            settings: Application settings.
        """
        self.vector_service = vector_service
        self.settings = settings
        self._converter = None
        self._chunker = None
        self._tokenizer = None

    def _get_converter(self):
        """Lazy-load Docling converter."""
        if self._converter is None:
            self._converter = self._create_docling_converter()
        return self._converter

    def _get_chunker(self):
        """Lazy-load HybridChunker."""
        if self._chunker is None:
            self._chunker, self._tokenizer = self._create_chunker()
        return self._chunker, self._tokenizer

    def _create_docling_converter(self):
        """Create configured Docling document converter.

        Returns:
            DocumentConverter configured for PDF processing with OCR.
        """
        try:
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                TableStructureOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            table_options = TableStructureOptions(
                do_cell_matching=True,
                mode="accurate",
            )

            pipeline_options = PdfPipelineOptions(
                do_ocr=self.settings.enable_ocr,
                do_table_structure=True,
                table_structure_options=table_options,
            )

            if self.settings.enable_ocr:
                try:
                    from docling.datamodel.pipeline_options import TesseractOcrOptions

                    ocr_options = TesseractOcrOptions(
                        lang=[self.settings.ocr_language],
                        force_full_page_ocr=False,
                    )
                    pipeline_options.ocr_options = ocr_options
                except ImportError:
                    logger.warning("TesseractOcrOptions not available, OCR disabled")

            return DocumentConverter(
                format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
            )
        except ImportError as e:
            logger.error(f"Docling not available: {e}")
            return None

    def _create_chunker(self):
        """Create HybridChunker with nomic tokenizer.

        Returns:
            Tuple of (chunker, tokenizer) or (None, None) if unavailable.
        """
        try:
            from docling.chunking import HybridChunker
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                "nomic-ai/nomic-embed-text-v1",
                trust_remote_code=False,
            )

            chunker = HybridChunker(
                tokenizer=tokenizer,
                max_tokens=self.settings.chunk_size,
                merge_peers=True,
            )

            return chunker, tokenizer
        except ImportError as e:
            logger.warning(f"HybridChunker not available: {e}")
            return None, None

    def _create_fallback_splitter(self):
        """Create fallback text splitter for plain text files.

        Returns:
            RecursiveCharacterTextSplitter configured for plain text.
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            # Approximate tokens to chars (4 chars per token average)
            chunk_size_chars = self.settings.chunk_size * 4
            overlap_chars = self.settings.chunk_overlap * 4

            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size_chars,
                chunk_overlap=overlap_chars,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        except ImportError:
            logger.warning("langchain_text_splitters not available")
            return None

    async def process_document(
        self,
        document: Document,
        db: Session,
    ) -> ProcessingResult:
        """Process a document and index to vectors.

        Args:
            document: Document record to process.
            db: Database session for status updates.

        Returns:
            ProcessingResult with outcome details.
        """
        start_time = time.perf_counter()
        file_path = Path(document.file_path)

        # Update status to processing
        document.status = "processing"
        db.commit()

        try:
            # Check file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Get chunks based on file type
            file_type = document.file_type.lower()
            if file_type in PLAIN_TEXT_EXTENSIONS:
                chunks, metadata = self._process_plain_text(file_path)
            else:
                chunks, metadata = self._process_with_docling(file_path, document.title)

            if not chunks:
                raise ValueError("No chunks extracted from document")

            # Extract tag names for vector indexing
            tag_names = [tag.name for tag in document.tags]

            # Prepare chunk metadata for VectorService
            chunk_metadata = [
                {
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                }
                for chunk in chunks
            ]

            # Index to vector store
            await self.vector_service.add_document(
                document_id=document.id,
                document_name=document.original_filename,
                chunks=[chunk.text for chunk in chunks],
                tags=tag_names,
                uploaded_by=document.uploaded_by,
                chunk_metadata=chunk_metadata,
            )

            # Calculate processing time
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            # Update document with results
            document.status = "ready"
            document.chunk_count = len(chunks)
            document.page_count = metadata.get("page_count")
            document.word_count = metadata.get("word_count", 0)
            document.processing_time_ms = processing_time_ms

            # Update title if extracted and not already set
            if not document.title and metadata.get("title"):
                document.title = metadata["title"]

            db.commit()

            return ProcessingResult(
                success=True,
                chunk_count=len(chunks),
                page_count=metadata.get("page_count"),
                word_count=metadata.get("word_count", 0),
                processing_time_ms=processing_time_ms,
            )

        except Exception as e:
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            error_message = str(e)
            logger.error(f"Processing failed for document {document.id}: {error_message}")

            # Update document with failure
            document.status = "failed"
            document.error_message = error_message
            document.processing_time_ms = processing_time_ms
            db.commit()

            return ProcessingResult(
                success=False,
                chunk_count=0,
                page_count=None,
                word_count=0,
                processing_time_ms=processing_time_ms,
                error_message=error_message,
            )

    def _process_plain_text(self, file_path: Path) -> tuple[list[ChunkInfo], dict]:
        """Process plain text file with fallback splitter.

        Args:
            file_path: Path to text file.

        Returns:
            Tuple of (chunks, metadata dict).
        """
        # Read file content
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        word_count = len(content.split())

        splitter = self._create_fallback_splitter()
        if splitter is None:
            # Ultimate fallback: single chunk
            chunks = [
                ChunkInfo(
                    text=content,
                    chunk_index=0,
                    page_number=None,
                    section=None,
                    token_count=len(content) // 4,
                )
            ]
        else:
            # Use langchain splitter
            split_texts = splitter.split_text(content)
            chunks = [
                ChunkInfo(
                    text=text,
                    chunk_index=i,
                    page_number=None,
                    section=None,
                    token_count=len(text) // 4,
                )
                for i, text in enumerate(split_texts)
            ]

        metadata = {
            "page_count": None,
            "word_count": word_count,
            "title": file_path.stem,  # Filename without extension
        }

        return chunks, metadata

    def _process_with_docling(
        self, file_path: Path, user_title: str | None
    ) -> tuple[list[ChunkInfo], dict]:
        """Process document with Docling.

        Args:
            file_path: Path to document file.
            user_title: User-provided title (highest priority).

        Returns:
            Tuple of (chunks, metadata dict).
        """
        converter = self._get_converter()
        if converter is None:
            # Fallback to plain text processing
            logger.warning("Docling not available, falling back to plain text processing")
            return self._process_plain_text(file_path)

        # Convert document
        result = converter.convert(str(file_path))

        # Get chunker and tokenizer
        chunker, tokenizer = self._get_chunker()

        if chunker is None:
            # Fallback: extract text and use simple splitting
            text = result.document.export_to_markdown()
            return self._process_plain_text_content(text, file_path.stem)

        # Chunk with HybridChunker
        chunks = []
        for i, chunk in enumerate(chunker.chunk(result.document)):
            # Get token count
            token_count = len(tokenizer.encode(chunk.text)) if tokenizer else len(chunk.text) // 4

            # Extract metadata from chunk
            page_number = None
            section = None
            if hasattr(chunk, "meta") and chunk.meta:
                page_number = chunk.meta.get("page_number")
                headings = chunk.meta.get("headings", [])
                section = headings[-1] if headings else None

            chunks.append(
                ChunkInfo(
                    text=chunk.text,
                    chunk_index=i,
                    page_number=page_number,
                    section=section,
                    token_count=token_count,
                )
            )

        # Extract metadata with title priority
        metadata = self._extract_metadata(result, user_title, file_path.stem)

        return chunks, metadata

    def _process_plain_text_content(
        self, content: str, filename: str
    ) -> tuple[list[ChunkInfo], dict]:
        """Process plain text content (used as fallback).

        Args:
            content: Text content to process.
            filename: Original filename for title fallback.

        Returns:
            Tuple of (chunks, metadata dict).
        """
        word_count = len(content.split())

        splitter = self._create_fallback_splitter()
        if splitter is None:
            chunks = [
                ChunkInfo(
                    text=content,
                    chunk_index=0,
                    page_number=None,
                    section=None,
                    token_count=len(content) // 4,
                )
            ]
        else:
            split_texts = splitter.split_text(content)
            chunks = [
                ChunkInfo(
                    text=text,
                    chunk_index=i,
                    page_number=None,
                    section=None,
                    token_count=len(text) // 4,
                )
                for i, text in enumerate(split_texts)
            ]

        metadata = {
            "page_count": None,
            "word_count": word_count,
            "title": filename,
        }

        return chunks, metadata

    def _extract_metadata(
        self,
        docling_result,
        user_title: str | None,
        filename: str,
    ) -> dict:
        """Extract document metadata with title priority.

        Title priority:
        1. User-provided title
        2. Document metadata title (PDF/DOCX properties)
        3. First H1 heading in document
        4. Original filename (without extension)

        Args:
            docling_result: Docling conversion result.
            user_title: User-provided title.
            filename: Filename without extension.

        Returns:
            Metadata dict with title, page_count, word_count.
        """
        doc = docling_result.document

        # Get page count
        page_count = None
        if hasattr(doc, "num_pages"):
            page_count = doc.num_pages
        elif hasattr(doc, "pages"):
            page_count = len(doc.pages)

        # Get word count from exported text
        text = doc.export_to_markdown()
        word_count = len(text.split())

        # Title extraction with priority
        title = user_title  # Priority 1

        # Priority 2: Document metadata
        if not title and hasattr(doc, "metadata") and doc.metadata:
            title = getattr(doc.metadata, "title", None)

        # Priority 3: First H1 heading
        if not title and hasattr(doc, "headings"):
            for heading in doc.headings:
                if hasattr(heading, "level") and heading.level == 1:
                    title = heading.text
                    break

        # Priority 4: Filename
        if not title:
            title = filename

        return {
            "title": title,
            "page_count": page_count,
            "word_count": word_count,
        }
