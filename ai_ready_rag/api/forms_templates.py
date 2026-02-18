"""Form template management endpoints.

Wraps ingestkit_forms.FormTemplateAPI with VE-RAG auth dependencies.
All FormTemplateAPI methods are synchronous (filesystem I/O) and are
executed via asyncio.to_thread() to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.dependencies import (
    ROLE_CUSTOMER_ADMIN,
    ROLE_SYSTEM_ADMIN,
    get_current_user,
    normalize_role,
    require_admin,
)
from ai_ready_rag.db.models import User
from ai_ready_rag.schemas.forms_template import (
    CreateTemplateRequest,
    ExtractedFieldResponse,
    ExtractionPreviewResponse,
    FieldMappingResponse,
    FormTemplateListResponse,
    FormTemplateResponse,
)
from ai_ready_rag.services.forms_processing_service import _pymupdf_renderer

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_template_api():
    """Build FormTemplateAPI from current settings.

    Lazy import to support conditional mounting (ingestkit-forms optional).
    Includes a FormRouter so preview_extraction works.
    """
    from ingestkit_forms import FileSystemTemplateStore, create_default_router
    from ingestkit_forms.api import FormTemplateAPI
    from ingestkit_forms.config import FormProcessorConfig

    from ai_ready_rag.services.ingestkit_adapters import (
        VERagFormDBAdapter,
        VERagLayoutFingerprinter,
        VERagOCRAdapter,
        VERagPDFWidgetAdapter,
        VERagVectorStoreAdapter,
        create_embedding_adapter,
    )

    settings = get_settings()
    store = FileSystemTemplateStore(settings.forms_template_storage_path)
    config = FormProcessorConfig(
        form_template_storage_path=settings.forms_template_storage_path,
        form_ocr_engine=settings.forms_ocr_engine,
        form_vlm_enabled=settings.forms_vlm_enabled,
        embedding_model=settings.embedding_model or "nomic-embed-text",
        embedding_dimension=settings.embedding_dimension,
        default_collection=settings.qdrant_collection,
        tenant_id=settings.default_tenant_id,
    )

    # Build adapters needed by the router
    vector_store = VERagVectorStoreAdapter(
        qdrant_url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        embedding_dimension=settings.embedding_dimension,
        document_id="preview",
        document_name="preview",
        tags=[],
        uploaded_by="system",
        tenant_id=settings.default_tenant_id,
    )
    embedder = create_embedding_adapter(
        ollama_url=settings.ollama_base_url,
        embedding_model=settings.embedding_model or "nomic-embed-text",
        embedding_dimension=settings.embedding_dimension,
    )
    form_db = VERagFormDBAdapter(db_path=settings.forms_db_path)
    fingerprinter = VERagLayoutFingerprinter(config, renderer=_pymupdf_renderer)
    ocr_backend = VERagOCRAdapter(engine=settings.forms_ocr_engine)
    pdf_widget_backend = VERagPDFWidgetAdapter()

    router = create_default_router(
        template_store=store,
        form_db=form_db,
        vector_store=vector_store,
        embedder=embedder,
        fingerprinter=fingerprinter,
        ocr_backend=ocr_backend,
        pdf_widget_backend=pdf_widget_backend,
        config=config,
    )

    return FormTemplateAPI(store=store, config=config, router=router)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_admin(user: User) -> bool:
    """Check if user has admin privileges."""
    role = normalize_role(user.role)
    return role in (ROLE_SYSTEM_ADMIN, ROLE_CUSTOMER_ADMIN)


def _map_form_error(exc) -> HTTPException:
    """Map FormIngestException to HTTPException."""
    from ingestkit_forms.errors import FormErrorCode

    code_map = {
        FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND: 404,
        FormErrorCode.E_FORM_TEMPLATE_INVALID: 400,
        FormErrorCode.E_FORM_TEMPLATE_VERSION_CONFLICT: 409,
        FormErrorCode.E_FORM_TEMPLATE_STORE_UNAVAILABLE: 503,
    }
    status_code = code_map.get(exc.code, 500)
    return HTTPException(status_code=status_code, detail=exc.message)


def _template_to_response(template) -> FormTemplateResponse:
    """Convert ingestkit FormTemplate to API response model."""
    return FormTemplateResponse(
        template_id=template.template_id,
        name=template.name,
        description=template.description,
        version=template.version,
        source_format=(
            template.source_format.value
            if hasattr(template.source_format, "value")
            else str(template.source_format)
        ),
        page_count=template.page_count,
        fields=[
            FieldMappingResponse(
                field_id=f.field_id,
                field_name=f.field_name,
                field_label=f.field_label,
                field_type=(
                    f.field_type.value if hasattr(f.field_type, "value") else str(f.field_type)
                ),
                page_number=f.page_number,
                required=f.required,
                sensitive=f.sensitive,
            )
            for f in template.fields
        ],
        status=template.status.value,
        created_at=template.created_at,
        updated_at=template.updated_at,
        created_by=template.created_by,
        tenant_id=template.tenant_id,
        approved_by=template.approved_by,
        approved_at=template.approved_at,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/templates", status_code=status.HTTP_201_CREATED, response_model=FormTemplateResponse)
async def create_template(
    request: CreateTemplateRequest,
    current_user: User = Depends(require_admin),
    api=Depends(get_template_api),
):
    """Create a new form template (admin only)."""
    from ingestkit_forms.errors import FormIngestException
    from ingestkit_forms.models import FieldMapping, FormTemplateCreateRequest

    settings = get_settings()
    initial_status = "approved" if not settings.forms_template_require_approval else "draft"

    try:
        fields = [FieldMapping(**f) for f in request.fields]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid field mapping: {e}",
        ) from e

    # Ensure tenant_id is set — fall back to settings default
    tenant_id = request.tenant_id or settings.default_tenant_id

    create_req = FormTemplateCreateRequest(
        name=request.name,
        description=request.description,
        source_format=request.source_format,
        sample_file_path=request.sample_file_path,
        page_count=request.page_count,
        fields=fields,
        tenant_id=tenant_id,
        created_by=str(current_user.id),
        initial_status=initial_status,
    )

    try:
        template = await asyncio.to_thread(api.create_template, create_req)
    except FormIngestException as exc:
        raise _map_form_error(exc) from exc

    return _template_to_response(template)


@router.get("/templates", response_model=FormTemplateListResponse)
async def list_templates(
    tenant_id: str | None = None,
    source_format: str | None = None,
    template_status: str | None = None,
    current_user: User = Depends(get_current_user),
    api=Depends(get_template_api),
):
    """List form templates. Non-admins see only approved templates."""
    from ingestkit_forms.errors import FormIngestException

    is_admin = _is_admin(current_user)

    # Non-admins can only see approved templates
    if not is_admin:
        template_status = "approved"

    try:
        templates = await asyncio.to_thread(
            api.list_templates,
            tenant_id=tenant_id,
            source_format=source_format,
            status=template_status,
        )
    except FormIngestException as exc:
        raise _map_form_error(exc) from exc

    if is_admin:
        items = [_template_to_response(t) for t in templates]
    else:
        # Strip fields for non-admin users
        items = [_template_to_response(t).model_copy(update={"fields": []}) for t in templates]

    return FormTemplateListResponse(templates=items, total=len(items))


@router.get("/templates/{template_id}", response_model=FormTemplateResponse)
async def get_template(
    template_id: str,
    version: int | None = None,
    current_user: User = Depends(get_current_user),
    api=Depends(get_template_api),
):
    """Get a specific form template by ID. Non-admins see only approved, no fields."""
    from ingestkit_forms.errors import FormIngestException
    from ingestkit_forms.models import TemplateStatus

    try:
        template = await asyncio.to_thread(api.get_template, template_id, version)
    except FormIngestException as exc:
        raise _map_form_error(exc) from exc

    is_admin = _is_admin(current_user)

    # Non-admins cannot access non-approved templates
    if not is_admin and template.status != TemplateStatus.APPROVED:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")

    resp = _template_to_response(template)

    # Strip fields for non-admins
    if not is_admin:
        resp = resp.model_copy(update={"fields": []})

    return resp


@router.post("/templates/{template_id}/approve", response_model=FormTemplateResponse)
async def approve_template(
    template_id: str,
    current_user: User = Depends(require_admin),
    api=Depends(get_template_api),
):
    """Approve a draft template (admin only)."""
    from ingestkit_forms.errors import FormIngestException

    try:
        template = await asyncio.to_thread(api.approve_template, template_id, str(current_user.id))
    except FormIngestException as exc:
        raise _map_form_error(exc) from exc

    return _template_to_response(template)


@router.delete("/templates/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template(
    template_id: str,
    current_user: User = Depends(require_admin),
    api=Depends(get_template_api),
):
    """Archive (soft-delete) a template (admin only)."""
    from ingestkit_forms.errors import FormIngestException

    try:
        await asyncio.to_thread(api.delete_template, template_id)
    except FormIngestException as exc:
        raise _map_form_error(exc) from exc

    return None


@router.post(
    "/templates/{template_id}/preview",
    response_model=ExtractionPreviewResponse,
)
async def preview_extraction(
    template_id: str,
    file: UploadFile,
    current_user: User = Depends(require_admin),
    api=Depends(get_template_api),
):
    """Preview extraction on an uploaded file (admin only)."""
    from ingestkit_forms.errors import FormIngestException

    suffix = Path(file.filename).suffix if file.filename else ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        preview = await asyncio.to_thread(api.preview_extraction, tmp_path, template_id)
    except FormIngestException as exc:
        raise _map_form_error(exc) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return ExtractionPreviewResponse(
        template_id=preview.template_id,
        template_name=preview.template_name,
        template_version=preview.template_version,
        fields=[
            ExtractedFieldResponse(
                field_name=f.field_name,
                field_label=f.field_label,
                value=f.value,
                confidence=f.confidence,
                extraction_method=f.extraction_method,
            )
            for f in preview.fields
        ],
        overall_confidence=preview.overall_confidence,
        extraction_method=preview.extraction_method,
        warnings=preview.warnings,
    )
