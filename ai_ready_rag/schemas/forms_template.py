"""Pydantic request/response schemas for form template management endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class FieldMappingResponse(BaseModel):
    """Serializable field mapping for API responses."""

    field_id: str
    field_name: str
    field_label: str
    field_type: str
    page_number: int
    required: bool = False
    sensitive: bool = False


class FormTemplateResponse(BaseModel):
    """Full template response (admin view)."""

    template_id: str
    name: str
    description: str
    version: int
    source_format: str
    page_count: int
    fields: list[FieldMappingResponse]
    status: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    tenant_id: str | None
    approved_by: str | None
    approved_at: datetime | None


class FormTemplateListResponse(BaseModel):
    """Template list response."""

    templates: list[FormTemplateResponse]
    total: int


class CreateTemplateRequest(BaseModel):
    """Request body for POST /templates."""

    name: str
    description: str = ""
    source_format: str
    sample_file_path: str
    page_count: int = Field(ge=1)
    fields: list[dict]  # Pass through to ingestkit FieldMapping
    tenant_id: str | None = None


class ExtractedFieldResponse(BaseModel):
    """Single extracted field in preview."""

    field_name: str
    field_label: str
    value: str | bool | float | None
    confidence: float
    extraction_method: str


class ExtractionPreviewResponse(BaseModel):
    """Preview extraction result."""

    template_id: str
    template_name: str
    template_version: int
    fields: list[ExtractedFieldResponse]
    overall_confidence: float
    extraction_method: str
    warnings: list[str]
