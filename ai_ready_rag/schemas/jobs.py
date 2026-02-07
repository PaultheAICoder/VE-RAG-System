"""Unified Jobs API schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class JobProgress(BaseModel):
    """Progress tracking for a job."""

    total: int
    processed: int
    failed: int


class JobStatusResponse(BaseModel):
    """Unified job status response."""

    job_id: str
    type: Literal["cache_warming", "reindex"]
    status: str
    progress: JobProgress | None = None
    result_summary: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class JobCancelResponse(BaseModel):
    """Response after cancelling a job."""

    job_id: str
    cancelled: bool
    message: str
