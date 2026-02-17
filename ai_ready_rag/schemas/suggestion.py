"""Tag suggestion schemas for auto-tagging approval workflow."""

from datetime import datetime

from pydantic import BaseModel


class TagSuggestionResponse(BaseModel):
    """Response model for a single tag suggestion."""

    id: str
    document_id: str
    tag_name: str
    display_name: str
    namespace: str
    source: str
    confidence: float
    strategy_id: str
    status: str
    reviewed_by: str | None
    reviewed_at: datetime | None
    created_at: datetime

    class Config:
        from_attributes = True


class TagSuggestionListResponse(BaseModel):
    """Wraps a list of tag suggestions with total count."""

    suggestions: list[TagSuggestionResponse]
    total: int


class ApproveSuggestionsRequest(BaseModel):
    """Request body for approving or rejecting suggestions."""

    suggestion_ids: list[str]


class BatchApproveRequest(BaseModel):
    """Request body for bulk approve across documents."""

    suggestion_ids: list[str]


class ApprovalResult(BaseModel):
    """Per-suggestion result from approve/reject operations."""

    suggestion_id: str
    status: str
    error: str | None = None


class ApprovalResponse(BaseModel):
    """Response for approve/reject/batch operations."""

    results: list[ApprovalResult]
    processed_count: int
    failed_count: int
