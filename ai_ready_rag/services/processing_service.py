"""Document processing service with profile-aware chunking."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from ai_ready_rag.config import CHUNK_SIZE_BY_TAG, Settings
from ai_ready_rag.db.models import Document
from ai_ready_rag.services.factory import get_chunker, get_vector_service

if TYPE_CHECKING:
    from ai_ready_rag.services.protocols import ChunkerProtocol, VectorServiceProtocol

logger = logging.getLogger(__name__)


@dataclass
class ProcessingOptions:
    """Per-upload processing options (optional overrides)."""

    enable_ocr: bool | None = None
    force_full_page_ocr: bool | None = None
    ocr_language: str | None = None
    table_extraction_mode: str | None = None
    include_image_descriptions: bool | None = None


@dataclass
class ChunkInfo:
    """Information about a single chunk."""

    text: str
    chunk_index: int
    page_number: int | None
    section: str | None
    token_count: int
    heading_breadcrumb: str | None = None


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
    """Document processing with profile-aware chunking and vector indexing.

    Uses factory pattern to select chunker and vector service based on
    ENV_PROFILE (laptop=Chroma+SimpleChunker, spark=Qdrant+DoclingChunker).
    """

    def __init__(
        self,
        settings: Settings,
        vector_service: "VectorServiceProtocol | None" = None,
        chunker: "ChunkerProtocol | None" = None,
    ):
        """Initialize processing service.

        Args:
            settings: Application settings.
            vector_service: Optional vector service override (uses factory if None).
            chunker: Optional chunker override (uses factory if None).
        """
        self.settings = settings
        self._vector_service = vector_service
        self._chunker = chunker

    @property
    def vector_service(self) -> "VectorServiceProtocol":
        """Get vector service, creating via factory if needed."""
        if self._vector_service is None:
            self._vector_service = get_vector_service(self.settings)
        return self._vector_service

    @property
    def chunker(self) -> "ChunkerProtocol":
        """Get chunker, creating via factory if needed."""
        if self._chunker is None:
            self._chunker = get_chunker(self.settings)
        return self._chunker

    async def process_document(
        self,
        document: Document,
        db: Session,
        processing_options: ProcessingOptions | None = None,
    ) -> ProcessingResult:
        """Process a document and index to vectors.

        Uses the configured chunker (SimpleChunker for laptop, DoclingChunker
        for spark) via the ChunkerProtocol interface.

        Args:
            document: Document record to process.
            db: Database session for status updates.
            processing_options: Optional per-upload processing options to override
                global settings for this specific document.

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

            # Route PDF files to ingestkit-forms if enabled
            if file_path.suffix.lower() == ".pdf" and self._should_use_ingestkit_forms():
                result, should_fallback = await self._process_with_ingestkit_forms(document, db)
                if not should_fallback and result is not None:
                    # ingestkit-forms handled it (success or failure)
                    processing_time_ms = int((time.perf_counter() - start_time) * 1000)
                    if result.success:
                        document.status = "ready"
                        document.chunk_count = result.chunk_count
                        document.processing_time_ms = processing_time_ms
                    else:
                        document.status = "failed"
                        document.error_message = result.error_message
                        document.processing_time_ms = processing_time_ms
                    db.commit()
                    return ProcessingResult(
                        success=result.success,
                        chunk_count=result.chunk_count,
                        page_count=result.page_count,
                        word_count=result.word_count,
                        processing_time_ms=processing_time_ms,
                        error_message=result.error_message,
                    )
                # else: fallback to standard chunker pipeline below
                logger.info("forms.routing.fallback", extra={"document_id": document.id})
                from ai_ready_rag.services.forms_metrics import metrics as forms_metrics

                forms_metrics.inc_fallback("no_match")
                forms_metrics.inc_documents_processed("fallback")

            # Route image files to ingestkit-image if enabled
            IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
            if file_path.suffix.lower() in IMAGE_EXTS and self._should_use_ingestkit_image():
                result, should_fallback = await self._process_with_ingestkit_image(document, db)
                if result is not None:
                    # ingestkit-image handled it (success or failure), no fallback
                    processing_time_ms = int((time.perf_counter() - start_time) * 1000)
                    if result.success:
                        document.status = "ready"
                        document.chunk_count = result.chunk_count
                        document.processing_time_ms = processing_time_ms
                    else:
                        document.status = "failed"
                        document.error_message = result.error_message
                        document.processing_time_ms = processing_time_ms
                    db.commit()
                    return ProcessingResult(
                        success=result.success,
                        chunk_count=result.chunk_count,
                        page_count=result.page_count,
                        word_count=result.word_count,
                        processing_time_ms=processing_time_ms,
                        error_message=result.error_message,
                    )

            # Route email files to ingestkit-email if enabled
            EMAIL_EXTS = {".eml", ".msg"}
            if file_path.suffix.lower() in EMAIL_EXTS and self._should_use_ingestkit_email():
                result, should_fallback = await self._process_with_ingestkit_email(document, db)
                if result is not None:
                    # ingestkit-email handled it (success or failure), no fallback
                    processing_time_ms = int((time.perf_counter() - start_time) * 1000)
                    if result.success:
                        document.status = "ready"
                        document.chunk_count = result.chunk_count
                        document.processing_time_ms = processing_time_ms
                    else:
                        document.status = "failed"
                        document.error_message = result.error_message
                        document.processing_time_ms = processing_time_ms
                    db.commit()
                    return ProcessingResult(
                        success=result.success,
                        chunk_count=result.chunk_count,
                        page_count=result.page_count,
                        word_count=result.word_count,
                        processing_time_ms=processing_time_ms,
                        error_message=result.error_message,
                    )

            # Route Excel files to ingestkit if enabled
            if file_path.suffix.lower() == ".xlsx" and self._should_use_ingestkit():
                result, should_fallback = await self._process_with_ingestkit(document, db)
                if not should_fallback and result is not None:
                    # ingestkit handled it (success or failure)
                    processing_time_ms = int((time.perf_counter() - start_time) * 1000)
                    if result.success:
                        document.status = "ready"
                        document.chunk_count = result.chunk_count
                        document.processing_time_ms = processing_time_ms
                    else:
                        document.status = "failed"
                        document.error_message = result.error_message
                        document.processing_time_ms = processing_time_ms
                    db.commit()
                    return ProcessingResult(
                        success=result.success,
                        chunk_count=result.chunk_count,
                        page_count=result.page_count,
                        word_count=result.word_count,
                        processing_time_ms=processing_time_ms,
                        error_message=result.error_message,
                    )
                # else: fallback to standard chunker pipeline below
                logger.info("Falling back to standard chunker for %s", document.id)

            # Resolve per-document chunk size based on tags
            chunk_size_override: int | None = None
            doc_tag_names = [tag.name.lower() for tag in document.tags]
            for tag_name in doc_tag_names:
                if tag_name in CHUNK_SIZE_BY_TAG:
                    chunk_size_override = CHUNK_SIZE_BY_TAG[tag_name]
                    logger.info(
                        "Tag-based chunk size: tag=%s size=%d for doc=%s",
                        tag_name,
                        chunk_size_override,
                        document.id,
                    )
                    break  # Use first matching tag (highest priority)

            # Get chunker - use per-upload options or tag override if provided
            if processing_options or chunk_size_override:
                chunker = get_chunker(self.settings, processing_options, chunk_size_override)
            else:
                chunker = self.chunker

            # Chunk document using profile-appropriate chunker.
            # Run in a thread so the async event loop stays responsive
            # during long-running OCR/Docling processing.
            metadata = {"title": document.title} if document.title else None
            chunk_dicts = await asyncio.to_thread(chunker.chunk_document, str(file_path), metadata)

            if not chunk_dicts:
                raise ValueError("No chunks extracted from document")

            # Filter out junk chunks (OCR artifacts, single chars, etc.)
            # Short chunks produce deceptively high similarity scores and
            # pollute search results, causing citation failures.
            MIN_CHUNK_WORDS = 5
            before_filter = len(chunk_dicts)
            chunk_dicts = [cd for cd in chunk_dicts if len(cd["text"].split()) >= MIN_CHUNK_WORDS]
            if before_filter != len(chunk_dicts):
                logger.info(
                    f"Filtered {before_filter - len(chunk_dicts)} tiny chunks "
                    f"(<{MIN_CHUNK_WORDS} words) from {before_filter} total"
                )

            if not chunk_dicts:
                raise ValueError("No chunks extracted from document")

            # Convert to ChunkInfo objects
            chunks = [
                ChunkInfo(
                    text=cd["text"],
                    chunk_index=cd.get("chunk_index", i),
                    page_number=cd.get("page_number"),
                    section=cd.get("section"),
                    token_count=len(cd["text"]) // 4,  # Estimate
                    heading_breadcrumb=cd.get("heading_breadcrumb"),
                )
                for i, cd in enumerate(chunk_dicts)
            ]

            # Calculate word count
            word_count = sum(len(c.text.split()) for c in chunks)

            # Generate summary chunk if enabled
            summary_meta_extra: dict | None = None
            if self.settings.generate_summaries:
                try:
                    from ai_ready_rag.services.summary_generator import SummaryGenerator

                    summary_model = self.settings.summary_model or self.settings.chat_model
                    generator = SummaryGenerator(
                        ollama_url=self.settings.ollama_base_url,
                        model=summary_model,
                    )
                    summary_result = await generator.generate(
                        chunks, document.original_filename, document.id
                    )
                    if summary_result:
                        summary_chunk, summary_meta_extra = summary_result
                        chunks.append(summary_chunk)
                        logger.info(
                            f"Summary chunk added for document {document.id} "
                            f"(type={summary_meta_extra.get('document_type')})"
                        )
                except Exception as e:
                    logger.warning(f"Summary generation failed for {document.id}, continuing: {e}")

            # --- LLM Auto-Tagging Classification ---
            try:
                await self._run_llm_classification(document, chunks, db)
            except Exception as e:
                logger.warning(
                    "LLM auto-tagging failed for %s, continuing with path tags: %s",
                    document.id,
                    e,
                )
                if document.auto_tag_status == "pending":
                    document.auto_tag_status = "partial"

            # Extract tag names for vector indexing
            tag_names = [tag.name for tag in document.tags]

            # Prepare chunk metadata for VectorService
            chunk_metadata = [
                {
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "heading_breadcrumb": chunk.heading_breadcrumb,
                }
                for chunk in chunks
            ]

            # Add extra metadata fields to the summary chunk entry
            if summary_meta_extra and chunk_metadata:
                chunk_metadata[-1].update(summary_meta_extra)

            # Index to vector store
            await self.vector_service.add_document(
                document_id=document.id,
                document_name=document.original_filename,
                chunks=[chunk.text for chunk in chunks],
                tags=tag_names,
                uploaded_by=document.uploaded_by,
                chunk_metadata=chunk_metadata,
            )

            # Post-process: rechunk Coverage Summary xlsx files
            if file_path.suffix.lower() == ".xlsx" and self.settings.coverage_rechunk_enabled:
                try:
                    from ai_ready_rag.services.coverage_rechunker import (
                        is_coverage_summary,
                        rechunk_coverage_summary,
                    )

                    if is_coverage_summary(document):
                        rechunk_count = await rechunk_coverage_summary(document, db, self.settings)
                        if rechunk_count > 0:
                            logger.info(
                                "Coverage rechunk replaced chunks: doc=%s new_count=%d",
                                document.id,
                                rechunk_count,
                            )
                except Exception as e:
                    logger.warning(
                        "Coverage rechunk failed for %s, keeping original chunks: %s",
                        document.id,
                        e,
                    )

            # Calculate processing time
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            # Update document with results
            document.status = "ready"
            document.chunk_count = len(chunks)
            document.page_count = chunk_dicts[0].get("page_number") if chunk_dicts else None
            document.word_count = word_count
            document.processing_time_ms = processing_time_ms

            # Update title if extracted and not already set
            extracted_title = chunk_dicts[0].get("source") if chunk_dicts else None
            if not document.title and extracted_title:
                document.title = extracted_title

            db.commit()

            return ProcessingResult(
                success=True,
                chunk_count=len(chunks),
                page_count=document.page_count,
                word_count=word_count,
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

    def _should_use_ingestkit(self) -> bool:
        """Check if ingestkit-excel integration is enabled and available."""
        if not self.settings.use_ingestkit_excel:
            return False
        try:
            import ingestkit_excel  # noqa: F401

            return True
        except ImportError:
            logger.warning("use_ingestkit_excel=True but ingestkit_excel not importable")
            return False

    async def _process_with_ingestkit(
        self,
        document: Document,
        db: Session,
    ) -> tuple[ProcessingResult | None, bool]:
        """Delegate Excel processing to ingestkit-excel.

        Returns:
            Tuple of (result, should_fallback). If should_fallback is True,
            caller should continue with the standard chunker pipeline.
        """
        from ai_ready_rag.services.excel_processing_service import (
            ExcelProcessingService,
        )

        excel_service = ExcelProcessingService(self.settings)
        return await excel_service.process_excel(document, db)

    def _should_use_ingestkit_forms(self) -> bool:
        """Check if ingestkit-forms integration is enabled and available."""
        if not self.settings.use_ingestkit_forms:
            return False
        try:
            import ingestkit_forms  # noqa: F401

            return True
        except ImportError:
            logger.warning("forms.dependency.missing")
            return False

    def _should_use_ingestkit_image(self) -> bool:
        """Check if ingestkit-image integration is enabled and available."""
        if not self.settings.use_ingestkit_image:
            return False
        try:
            import ingestkit_image  # noqa: F401

            return True
        except ImportError:
            logger.warning("use_ingestkit_image=True but ingestkit_image not importable")
            return False

    async def _process_with_ingestkit_image(
        self,
        document: Document,
        db: Session,
    ) -> tuple[ProcessingResult | None, bool]:
        """Delegate image processing to ingestkit-image.

        Returns:
            Tuple of (result, should_fallback). should_fallback is always False.
        """
        from ai_ready_rag.services.image_processing_service import (
            ImageProcessingService,
        )

        image_service = ImageProcessingService(self.settings)
        return await image_service.process_image(document, db)

    def _should_use_ingestkit_email(self) -> bool:
        """Check if ingestkit-email integration is enabled and available."""
        if not self.settings.use_ingestkit_email:
            return False
        try:
            import ingestkit_email  # noqa: F401

            return True
        except ImportError:
            logger.warning("use_ingestkit_email=True but ingestkit_email not importable")
            return False

    async def _process_with_ingestkit_email(
        self,
        document: Document,
        db: Session,
    ) -> tuple[ProcessingResult | None, bool]:
        """Delegate email processing to ingestkit-email.

        Returns:
            Tuple of (result, should_fallback). should_fallback is always False.
        """
        from ai_ready_rag.services.email_processing_service import (
            EmailProcessingService,
        )

        email_service = EmailProcessingService(self.settings)
        return await email_service.process_email(document, db)

    async def _process_with_ingestkit_forms(
        self,
        document: Document,
        db: Session,
    ) -> tuple[ProcessingResult | None, bool]:
        """Delegate PDF forms processing to ingestkit-forms.

        Returns:
            Tuple of (result, should_fallback). If should_fallback is True,
            caller should continue with the standard chunker pipeline.
        """
        from ai_ready_rag.services.forms_processing_service import (
            FormsProcessingService,
        )

        forms_service = FormsProcessingService(self.settings)
        return await forms_service.process_form(document, db)

    async def _run_llm_classification(
        self,
        document: Document,
        chunks: list[ChunkInfo],
        db: Session,
    ) -> None:
        """Run LLM classification, conflict resolution, and provenance recording.

        Modifies document.tags, document.auto_tag_source, and
        document.auto_tag_status in-place. Caller must db.commit() after this
        returns.
        """
        if not (
            self.settings.auto_tagging_enabled
            and self.settings.auto_tagging_llm_enabled
            and document.auto_tag_status == "pending"
        ):
            return

        # Extract content preview from first 2000 chars of chunks
        content_preview = " ".join(chunk.text for chunk in chunks)[:2000]

        # Load pinned strategy
        from ai_ready_rag.services.auto_tagging import AutoTagStrategy

        strategy_name = document.auto_tag_strategy or self.settings.auto_tagging_strategy
        strategy_path = Path(self.settings.auto_tagging_strategies_dir) / f"{strategy_name}.yaml"
        strategy = AutoTagStrategy.load(str(strategy_path))

        # Version mismatch warning
        if document.auto_tag_version and strategy.version != document.auto_tag_version:
            logger.warning(
                "Strategy version mismatch for doc %s: pinned=%s, loaded=%s",
                document.id,
                document.auto_tag_version,
                strategy.version,
            )

        # Run classifier
        from ai_ready_rag.services.auto_tagging import DocumentClassifier

        classifier = DocumentClassifier(self.settings)
        result = await classifier.classify(
            strategy,
            document.original_filename,
            document.source_path or "",
            content_preview,
        )

        # Identify existing path tags on document
        path_auto_tags: list = []
        if document.source_path:
            path_auto_tags = strategy.parse_path(document.source_path)

        path_tag_names = {at.tag_name for at in path_auto_tags}
        manual_tag_names = {tag.name for tag in document.tags if tag.name not in path_tag_names}

        # Resolve conflicts
        from ai_ready_rag.services.auto_tagging.conflict import (
            build_provenance,
            enforce_guardrail,
            resolve_conflicts,
        )

        winning_llm, losing_path, conflicts = resolve_conflicts(
            path_tags=path_auto_tags,
            llm_result=result,
            confidence_threshold=self.settings.auto_tagging_confidence_threshold,
        )

        # Remove losing path tags from document.tags
        losing_path_names = {at.tag_name for at in losing_path}
        if losing_path_names:
            document.tags = [t for t in document.tags if t.name not in losing_path_names]

        # Convert winning LLM AutoTag objects to DB Tag objects and add
        from ai_ready_rag.services.document_service import DocumentService

        doc_service = DocumentService(db, self.settings)
        existing_tag_names = {t.name for t in document.tags}
        for at in winning_llm:
            if at.tag_name not in existing_tag_names:
                tag_obj = doc_service.ensure_tag_exists(
                    tag_name=at.tag_name,
                    display_name=at.display_name,
                    namespace=at.namespace,
                    strategy=strategy,
                    created_by=document.uploaded_by,
                )
                if tag_obj is not None:
                    document.tags.append(tag_obj)
                    existing_tag_names.add(at.tag_name)

        # Enforce guardrail
        current_path_tags = [at for at in path_auto_tags if at.tag_name not in losing_path_names]
        kept_path, kept_llm, truncated = enforce_guardrail(
            manual_tag_names=manual_tag_names,
            path_tags=current_path_tags,
            llm_tags=winning_llm,
            max_tags=self.settings.auto_tagging_max_tags_per_doc,
        )
        # If any truncated, remove from document.tags
        if truncated:
            truncated_set = set(truncated)
            document.tags = [t for t in document.tags if t.name not in truncated_set]

        # Build and write provenance
        applied_names = [t.name for t in document.tags]
        discarded_names = [at.tag_name for at in result.discarded]
        # Add conflict-losing LLM tags to discarded
        for conflict in conflicts:
            if conflict["winner"] == "path":
                discarded_names.append(f"{conflict['namespace']}:{conflict['llm_value']}")
        suggested_names = [at.tag_name for at in result.suggested]

        # Persist suggestions as TagSuggestion rows for approval workflow
        if result.suggested:
            from ai_ready_rag.db.models.suggestion import TagSuggestion

            for at in result.suggested:
                suggestion = TagSuggestion(
                    document_id=document.id,
                    tag_name=at.tag_name,
                    display_name=at.display_name,
                    namespace=at.namespace,
                    source=at.source,
                    confidence=at.confidence,
                    strategy_id=strategy.id,
                    status="pending",
                )
                db.add(suggestion)

        provenance = build_provenance(
            strategy_id=strategy.id,
            strategy_version=strategy.version,
            path_tags=path_auto_tags,
            llm_result=result,
            conflicts=conflicts,
            applied_tag_names=applied_names,
            discarded_tag_names=discarded_names,
            suggested_tag_names=suggested_names,
            truncated_tag_names=truncated if truncated else None,
        )
        document.auto_tag_source = json.dumps(provenance)

        # Update auto_tag_status
        document.auto_tag_status = result.status

        # Flush to ensure tag changes are visible before vector indexing
        db.flush()
