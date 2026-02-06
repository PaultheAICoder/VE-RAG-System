"""Experimental endpoint schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class GeneratePresentationRequest(BaseModel):
    """Request to generate a presentation."""

    topic: str = Field(
        ..., min_length=5, max_length=500, description="Presentation topic or question"
    )
    num_slides: int = Field(default=5, ge=3, le=10, description="Number of content slides")
    template: Literal["executive", "detailed", "marketing"] = Field(
        default="executive",
        description="Presentation style template",
    )


class SlideResponse(BaseModel):
    """Single slide in response."""

    slide_number: int
    title: str
    bullet_points: list[str]
    citations: list[str]
    speaker_notes: str


class PresentationResponse(BaseModel):
    """Generated presentation response."""

    title: str
    subtitle: str
    slides: list[SlideResponse]
    total_sources: int
    generation_steps: list[str]
    markdown: str  # Markdown formatted version
