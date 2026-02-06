"""Experimental endpoints for Phase 2 prototypes.

These endpoints are for testing new features before production.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.dependencies import get_current_user
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import User
from ai_ready_rag.schemas.experimental import (
    GeneratePresentationRequest,
    PresentationResponse,
    SlideResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experimental", tags=["experimental"])


# --- Endpoints ---


@router.post("/presentations/generate", response_model=PresentationResponse)
async def generate_presentation(
    payload: GeneratePresentationRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a presentation with citations from company documents.

    This is a Phase 2 prototype using prompt chaining:
    1. Retrieve relevant documents based on topic
    2. Generate presentation outline
    3. Expand each slide with cited content

    Requires authentication. Uses user's document access tags.
    """
    from ai_ready_rag.services.rag_service import RAGService
    from ai_ready_rag.services.slide_generator import SlideGeneratorService

    settings = get_settings()

    # Get user's tags for document access
    user_tags = [tag.name for tag in current_user.tags] if current_user.tags else []
    if not user_tags:
        user_tags = ["public"]  # Fallback

    logger.info(
        f"[SLIDES] User {current_user.email} generating presentation: "
        f"topic='{payload.topic[:50]}...', slides={payload.num_slides}, tags={user_tags}"
    )

    try:
        # Initialize services (vector_service is a singleton from app.state)
        rag_service = RAGService(settings, vector_service=http_request.app.state.vector_service)
        slide_generator = SlideGeneratorService(rag_service, settings)

        # Generate presentation
        presentation = await slide_generator.generate_presentation(
            topic=payload.topic,
            user_tags=user_tags,
            num_slides=payload.num_slides,
            template=payload.template,
        )

        # Convert to response
        return PresentationResponse(
            title=presentation.title,
            subtitle=presentation.subtitle,
            slides=[
                SlideResponse(
                    slide_number=s.slide_number,
                    title=s.title,
                    bullet_points=s.bullet_points,
                    citations=s.citations,
                    speaker_notes=s.speaker_notes,
                )
                for s in presentation.slides
            ],
            total_sources=presentation.total_sources,
            generation_steps=presentation.generation_steps,
            markdown=slide_generator.to_markdown(presentation),
        )

    except ValueError as e:
        logger.warning(f"[SLIDES] Generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from None
    except Exception as e:
        logger.exception(f"[SLIDES] Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate presentation: {str(e)[:200]}",
        ) from None


@router.get("/presentations/templates")
async def list_templates(
    current_user: User = Depends(get_current_user),
):
    """List available presentation templates."""
    return {
        "templates": [
            {
                "id": "executive",
                "name": "Executive Summary",
                "description": "Concise, high-level overview with key metrics",
                "recommended_slides": 5,
            },
            {
                "id": "detailed",
                "name": "Detailed Analysis",
                "description": "Comprehensive deep-dive with supporting data",
                "recommended_slides": 8,
            },
            {
                "id": "marketing",
                "name": "Marketing Deck",
                "description": "Customer-facing with benefits and value props",
                "recommended_slides": 6,
            },
        ]
    }
