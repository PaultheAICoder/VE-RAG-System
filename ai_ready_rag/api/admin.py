"""Admin endpoints for system management."""

import logging

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.dependencies import require_admin
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import User
from ai_ready_rag.services.document_service import DocumentService

logger = logging.getLogger(__name__)
router = APIRouter()


class RecoverResponse(BaseModel):
    recovered: int
    message: str


@router.post("/documents/recover-stuck", response_model=RecoverResponse)
async def recover_stuck_documents(
    max_age_hours: int = Query(2, ge=1, le=168, description="Maximum age in hours"),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Recover documents stuck in processing state.

    Resets documents that have been in 'processing' status longer than
    max_age_hours back to 'pending' so they can be reprocessed.

    Admin only.
    """
    settings = get_settings()
    service = DocumentService(db, settings)

    recovered = service.recover_stuck_documents(max_age_hours=max_age_hours)

    if recovered > 0:
        logger.warning(
            f"Admin {current_user.email} recovered {recovered} stuck documents "
            f"(max_age_hours={max_age_hours})"
        )

    return RecoverResponse(
        recovered=recovered,
        message=f"Reset {recovered} stuck documents to pending status",
    )
