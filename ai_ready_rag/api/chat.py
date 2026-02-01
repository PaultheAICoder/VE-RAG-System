"""Chat session and message endpoints."""

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.dependencies import get_current_user
from ai_ready_rag.core.exceptions import (
    LLMConnectionError,
    LLMTimeoutError,
    ModelNotAllowedError,
    ModelUnavailableError,
    RAGServiceError,
)
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import ChatMessage, ChatSession, Tag, User
from ai_ready_rag.services.rag_service import (
    ChatMessage as RAGChatMessage,
)
from ai_ready_rag.services.rag_service import (
    RAGRequest,
    RAGService,
)
from ai_ready_rag.services.settings_service import SettingsService
from ai_ready_rag.services.vector_service import VectorService

# Settings key for runtime chat model override
CHAT_MODEL_KEY = "chat_model"

router = APIRouter()


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


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    data: SessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new chat session for the current user."""
    session = ChatSession(
        user_id=current_user.id,
        title=data.title,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return SessionResponse(
        id=session.id,
        user_id=session.user_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        is_archived=session.is_archived,
        message_count=0,
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    archived: bool = Query(False, description="Include archived sessions"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List chat sessions for the current user."""
    # Base query - filter by user
    query = db.query(ChatSession).filter(ChatSession.user_id == current_user.id)

    # Filter archived unless requested
    if not archived:
        query = query.filter(ChatSession.is_archived == False)  # noqa: E712

    # Get total count
    total = query.count()

    # Get paginated sessions ordered by updated_at DESC
    sessions = query.order_by(ChatSession.updated_at.desc()).offset(offset).limit(limit).all()

    # Build response with message counts and previews
    session_items = []
    for session in sessions:
        # Get message count
        message_count = (
            db.query(func.count(ChatMessage.id))
            .filter(ChatMessage.session_id == session.id)
            .scalar()
        )

        # Get last user message preview
        last_message = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session.id, ChatMessage.role == "user")
            .order_by(ChatMessage.created_at.desc())
            .first()
        )
        preview = last_message.content[:100] if last_message else None

        session_items.append(
            SessionListItem(
                id=session.id,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
                is_archived=session.is_archived,
                message_count=message_count,
                last_message_preview=preview,
            )
        )

    return SessionListResponse(
        sessions=session_items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a single chat session by ID."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Ownership check
    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Get message count
    message_count = (
        db.query(func.count(ChatMessage.id)).filter(ChatMessage.session_id == session.id).scalar()
    )

    return SessionResponse(
        id=session.id,
        user_id=session.user_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        is_archived=session.is_archived,
        message_count=message_count,
    )


@router.patch("/sessions/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    data: SessionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update a chat session's title or archived status."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Ownership check
    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Update fields if provided
    if data.title is not None:
        session.title = data.title
    if data.is_archived is not None:
        session.is_archived = data.is_archived

    # Update timestamp
    session.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(session)

    # Get message count
    message_count = (
        db.query(func.count(ChatMessage.id)).filter(ChatMessage.session_id == session.id).scalar()
    )

    return SessionResponse(
        id=session.id,
        user_id=session.user_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        is_archived=session.is_archived,
        message_count=message_count,
    )


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
async def get_messages(
    session_id: str,
    limit: int = Query(50, ge=1, le=100, description="Max messages to return"),
    before: str | None = Query(None, description="Get messages before this message ID"),
    after: str | None = Query(None, description="Get messages after this message ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageListResponse:
    """Get messages for a chat session with cursor-based pagination.

    Returns messages in chronological order (oldest first).
    Use `before` to get older messages, `after` to get newer messages.
    """
    # Validate that both cursors are not provided simultaneously
    if before is not None and after is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot use both 'before' and 'after' cursors simultaneously",
        )

    # Session ownership validation (returns 404 to hide existence of other users' sessions)
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
        .first()
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Get total message count for this session
    total = (
        db.query(func.count(ChatMessage.id)).filter(ChatMessage.session_id == session_id).scalar()
    )

    # Base query for messages in this session
    query = db.query(ChatMessage).filter(ChatMessage.session_id == session_id)

    # Apply cursor-based pagination
    if before is not None:
        # Find reference message
        ref_msg = db.query(ChatMessage).filter(ChatMessage.id == before).first()
        if not ref_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid 'before' cursor: message not found",
            )
        query = query.filter(ChatMessage.created_at < ref_msg.created_at)

    if after is not None:
        # Find reference message
        ref_msg = db.query(ChatMessage).filter(ChatMessage.id == after).first()
        if not ref_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid 'after' cursor: message not found",
            )
        query = query.filter(ChatMessage.created_at > ref_msg.created_at)

    # Order by created_at ASC (oldest first for chat display)
    query = query.order_by(ChatMessage.created_at.asc())

    # Fetch limit + 1 to determine has_more
    db_messages = query.limit(limit + 1).all()

    # Determine if there are more messages
    has_more = len(db_messages) > limit

    # Take only the first `limit` messages
    db_messages = db_messages[:limit]

    # Convert to response models
    messages = []
    for msg in db_messages:
        # Parse sources JSON if present
        sources = None
        if msg.sources:
            try:
                sources_data = json.loads(msg.sources)
                sources = [SourceInfo(**s) for s in sources_data]
            except (json.JSONDecodeError, TypeError):
                sources = None

        # Build confidence info if present
        confidence = None
        if msg.confidence is not None:
            confidence_int = (
                int(msg.confidence * 100) if msg.confidence <= 1 else int(msg.confidence)
            )
            # Use stored breakdown values if available, else fall back to overall
            confidence = ConfidenceInfo(
                overall=confidence_int,
                retrieval=int(msg.confidence_retrieval * 100)
                if msg.confidence_retrieval
                else confidence_int,
                coverage=int(msg.confidence_coverage * 100)
                if msg.confidence_coverage
                else confidence_int,
                llm=msg.confidence_llm if msg.confidence_llm else confidence_int,
            )

        messages.append(
            MessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                sources=sources,
                confidence=confidence,
                generation_time_ms=msg.generation_time_ms,
                was_routed=msg.was_routed,
                routed_to=msg.routed_to,
                route_reason=msg.route_reason,
                created_at=msg.created_at,
            )
        )

    return MessageListResponse(
        messages=messages,
        has_more=has_more,
        total=total,
    )


@router.post(
    "/sessions/{session_id}/messages",
    response_model=SendMessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def send_message(
    session_id: str,
    message: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SendMessageResponse:
    """Send a message and receive AI response.

    Creates a user message in the database, calls RAGService to generate
    a response, stores the assistant message with citations and confidence,
    and returns both messages.

    Args:
        session_id: Chat session ID (must belong to current user).
        message: Message content and optional model override.
        current_user: Authenticated user from JWT.
        db: Database session.

    Returns:
        SendMessageResponse with user and assistant messages.

    Raises:
        HTTPException 404: Session not found or not owned by user.
        HTTPException 400: Model not in allowlist.
        HTTPException 503: AI service unavailable.
    """
    settings = get_settings()

    # 1. Validate session ownership
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
        .first()
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # 2. Create user message
    user_msg = ChatMessage(
        session_id=session_id,
        role="user",
        content=message.content,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # 3. Load chat history (last 10 messages, excluding the one just created)
    history = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id,
            ChatMessage.id != user_msg.id,
        )
        .order_by(ChatMessage.created_at.desc())
        .limit(10)
        .all()
    )

    # 4. Build RAGRequest
    # Convert DB messages to RAG ChatMessages (oldest first)
    rag_history = [RAGChatMessage(role=m.role, content=m.content) for m in reversed(history)]
    # Add user's new message to history
    rag_history.append(RAGChatMessage(role="user", content=message.content))

    # Get user's tags - admins can access all documents
    if current_user.role == "admin":
        # Admin bypasses tag filtering - get all tags
        all_tags = db.query(Tag).all()
        user_tags = [tag.name for tag in all_tags]
    else:
        user_tags = [tag.name for tag in current_user.tags]

    rag_request = RAGRequest(
        query=message.content,
        user_tags=user_tags,
        tenant_id=settings.default_tenant_id,
        chat_history=rag_history,
        model=message.model,
    )

    # 5. Call RAGService
    try:
        vector_service = VectorService(
            qdrant_url=settings.qdrant_url,
            ollama_url=settings.ollama_base_url,
            collection_name=settings.qdrant_collection,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
            max_tokens=settings.embedding_max_tokens,
            tenant_id=settings.default_tenant_id,
        )
        await vector_service.initialize()

        # Check for runtime model override from admin settings
        settings_service = SettingsService(db)
        chat_model = settings_service.get(CHAT_MODEL_KEY) or settings.chat_model

        rag_service = RAGService(settings, vector_service, default_model=chat_model)
        response = await rag_service.generate(rag_request, db)

    except ModelNotAllowedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except (LLMConnectionError, LLMTimeoutError, ModelUnavailableError, RAGServiceError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service temporarily unavailable",
        ) from e

    # 6. Create assistant message
    # Serialize citations to JSON
    sources_json = (
        json.dumps(
            [
                {
                    "source_id": c.source_id,
                    "document_id": c.document_id,
                    "chunk_index": c.chunk_index,
                    "title": c.document_name,
                    "snippet": c.snippet,
                }
                for c in response.citations
            ]
        )
        if response.citations
        else None
    )

    assistant_msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=response.answer,
        sources=sources_json,
        confidence=response.confidence.overall,
        confidence_retrieval=response.confidence.retrieval_score,
        confidence_coverage=response.confidence.coverage_score,
        confidence_llm=response.confidence.llm_score,
        generation_time_ms=response.generation_time_ms,
        was_routed=(response.action == "ROUTE"),
        routed_to=response.route_to.owner_email if response.route_to else None,
        route_reason=response.route_to.reason if response.route_to else None,
    )
    db.add(assistant_msg)

    # 7. Update session.updated_at
    session.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(assistant_msg)

    # 8. Build and return response
    # Convert DB ChatMessage to MessageResponse
    def message_to_response(
        msg: ChatMessage,
        is_assistant: bool = False,
        rag_response=None,
    ) -> MessageResponse:
        sources = None
        confidence = None

        if is_assistant and msg.sources:
            sources = [SourceInfo(**s) for s in json.loads(msg.sources)]

        if is_assistant and msg.confidence is not None and rag_response:
            # Use the full confidence breakdown from the RAG response
            confidence = ConfidenceInfo(
                overall=int(msg.confidence),
                retrieval=int(rag_response.confidence.retrieval_score * 100),
                coverage=int(rag_response.confidence.coverage_score * 100),
                llm=rag_response.confidence.llm_score,
            )

        return MessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            sources=sources,
            confidence=confidence,
            generation_time_ms=msg.generation_time_ms,
            was_routed=msg.was_routed,
            routed_to=msg.routed_to,
            route_reason=msg.route_reason,
            created_at=msg.created_at,
        )

    return SendMessageResponse(
        user_message=message_to_response(user_msg),
        assistant_message=message_to_response(
            assistant_msg, is_assistant=True, rag_response=response
        ),
        generation_time_ms=response.generation_time_ms,
        routing_decision=response.routing_decision,
    )
