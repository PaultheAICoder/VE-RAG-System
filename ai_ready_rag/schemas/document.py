"""Document management schemas."""

from datetime import datetime

from pydantic import BaseModel


class TagInfo(BaseModel):
    id: str
    name: str
    display_name: str

    class Config:
        from_attributes = True


class TagUpdateRequest(BaseModel):
    tag_ids: list[str]


class ReprocessRequest(BaseModel):
    enable_ocr: bool = True
    force_ocr: bool = False
    ocr_language: str = "eng"


class DocumentResponse(BaseModel):
    id: str
    original_filename: str
    filename: str
    file_type: str
    file_size: int
    status: str
    title: str | None
    description: str | None
    chunk_count: int | None
    page_count: int | None
    word_count: int | None
    processing_time_ms: int | None
    auto_tag_status: str | None = None
    auto_tag_strategy: str | None = None
    auto_tag_version: str | None = None
    auto_tag_source: str | None = None
    source_path: str | None = None
    error_message: str | None
    tags: list[TagInfo]
    uploaded_by: str
    uploaded_at: datetime
    processed_at: datetime | None

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    limit: int
    offset: int


# Duplicate check models
class CheckDuplicatesRequest(BaseModel):
    """Request for pre-upload duplicate check."""

    filenames: list[str]

    @property
    def validated_filenames(self) -> list[str]:
        """Ensure at least one filename provided."""
        if not self.filenames:
            raise ValueError("At least one filename required")
        return self.filenames


class DuplicateInfo(BaseModel):
    """Information about a duplicate file."""

    filename: str
    existing_id: str
    existing_filename: str
    uploaded_at: datetime


class CheckDuplicatesResponse(BaseModel):
    """Response from duplicate check."""

    duplicates: list[DuplicateInfo]
    unique: list[str]


class DuplicateErrorDetail(BaseModel):
    """Structured 409 error response."""

    detail: str
    error_code: str = "DUPLICATE_FILE"
    existing_id: str
    existing_filename: str
    uploaded_at: datetime


class BulkDeleteRequest(BaseModel):
    """Request body for bulk delete."""

    document_ids: list[str]


class BulkDeleteResult(BaseModel):
    """Result for a single document deletion."""

    id: str
    status: str  # "deleted" or "failed"
    error: str | None = None


class BulkDeleteResponse(BaseModel):
    """Response for bulk delete operation."""

    results: list[BulkDeleteResult]
    deleted_count: int
    failed_count: int


class BulkReprocessRequest(BaseModel):
    """Request body for bulk reprocess."""

    document_ids: list[str]


class BulkReprocessResponse(BaseModel):
    """Response for bulk reprocess operation."""

    queued: int
    skipped: int
    skipped_ids: list[str]
