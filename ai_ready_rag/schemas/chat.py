"""Chat session and message schemas."""

from datetime import datetime

from pydantic import BaseModel, Field

# =============================================================================
# Request Schemas
# =============================================================================


class SessionCreate(BaseModel):
    """Create a new chat session."""

    title: str | None = None


class SessionUpdate(BaseModel):
    """Update an existing chat session."""

    title: str | None = None
    is_archived: bool | None = None


class MessageCreate(BaseModel):
    """Request body for sending a message."""

    content: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None  # Override model (must be in allowlist)


class BulkDeleteRequest(BaseModel):
    """Request body for bulk session deletion."""

    session_ids: list[str] = Field(..., min_length=1, max_length=100)


# =============================================================================
# Response Schemas
# =============================================================================


class SessionResponse(BaseModel):
    """Single session response."""

    id: str
    user_id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    is_archived: bool
    message_count: int

    class Config:
        from_attributes = True


class SessionListItem(BaseModel):
    """Session item in list response."""

    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    is_archived: bool
    message_count: int
    last_message_preview: str | None


class SessionListResponse(BaseModel):
    """Paginated list of sessions."""

    sessions: list[SessionListItem]
    total: int
    limit: int
    offset: int


class SourceInfo(BaseModel):
    """Citation source information."""

    source_id: str  # Format: "{document_id}:{chunk_index}"
    document_id: str
    chunk_index: int
    title: str | None  # document_name
    snippet: str | None  # First 200 chars


class ConfidenceInfo(BaseModel):
    """Confidence score breakdown."""

    overall: int  # 0-100
    retrieval: int  # retrieval_score * 100
    coverage: int  # coverage_score * 100
    llm: int  # llm_score


class MessageResponse(BaseModel):
    """Single message in response."""

    id: str
    role: str  # "user" or "assistant"
    content: str
    sources: list[SourceInfo] | None = None
    confidence: ConfidenceInfo | None = None
    generation_time_ms: float | None = None
    was_routed: bool = False
    routed_to: str | None = None
    route_reason: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class SendMessageResponse(BaseModel):
    """Response for POST /messages endpoint."""

    user_message: MessageResponse
    assistant_message: MessageResponse
    generation_time_ms: float
    routing_decision: str | None = None  # "RETRIEVE" | "DIRECT" | None


class MessageListResponse(BaseModel):
    """Paginated list of messages."""

    messages: list[MessageResponse]
    has_more: bool
    total: int


class DeleteSessionResponse(BaseModel):
    """Response for single session deletion."""

    success: bool
    deleted_session_id: str
    deleted_messages_count: int


class BulkDeleteResponse(BaseModel):
    """Response for bulk session deletion."""

    success: bool
    deleted_count: int
    failed_ids: list[str]
    total_messages_deleted: int
