"""Tests for form template management endpoints.

Tests template CRUD operations, authorization, and lifecycle workflows.
Uses a dedicated test app with forms router to avoid conditional mounting issues.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_template_mock(**overrides):
    """Create a mock FormTemplate object."""
    defaults = {
        "template_id": "tpl-123",
        "name": "Test Template",
        "description": "Test description",
        "version": 1,
        "page_count": 1,
        "fields": [],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "created_by": "admin-1",
        "tenant_id": "default",
        "approved_by": None,
        "approved_at": None,
    }
    defaults.update(overrides)
    template = MagicMock()
    for k, v in defaults.items():
        setattr(template, k, v)
    # source_format and status need .value attribute
    template.source_format.value = defaults.get("source_format_value", "pdf")
    template.status.value = defaults.get("status_value", "draft")
    return template


def _make_template_request(**overrides):
    """Create a CreateTemplateRequest payload."""
    defaults = {
        "name": "Test Form Template",
        "description": "Test description",
        "source_format": "pdf",
        "sample_file_path": "/tmp/test_form.pdf",
        "page_count": 1,
        "fields": [
            {
                "field_id": "f1",
                "field_name": "full_name",
                "field_label": "Full Name",
                "field_type": "text",
                "page_number": 0,
                "required": True,
                "sensitive": False,
            }
        ],
        "tenant_id": "default",
    }
    defaults.update(overrides)
    return defaults


@pytest.fixture
def mock_api():
    """Create a mock FormTemplateAPI."""
    return MagicMock()


@pytest.fixture
def forms_app(db, mock_api):
    """Create a test FastAPI app with the forms router mounted."""
    from ai_ready_rag.api.forms_templates import get_template_api
    from ai_ready_rag.api.forms_templates import router as forms_router
    from ai_ready_rag.db.database import get_db

    test_app = FastAPI()
    test_app.include_router(forms_router, prefix="/api/forms")

    def override_get_db():
        try:
            yield db
        finally:
            pass

    test_app.dependency_overrides[get_db] = override_get_db
    test_app.dependency_overrides[get_template_api] = lambda: mock_api

    return test_app


@pytest.fixture
def forms_client(forms_app):
    """TestClient with forms router mounted."""
    with TestClient(forms_app) as c:
        yield c


class TestTemplateAuthorization:
    """Tests for template endpoint authorization."""

    @pytest.mark.unit
    def test_forms_template_create_requires_admin(self, forms_client, user_headers, mock_api):
        """Non-admin users cannot create templates."""
        payload = _make_template_request()

        response = forms_client.post("/api/forms/templates", json=payload, headers=user_headers)

        assert response.status_code == 403

    @pytest.mark.unit
    def test_forms_template_list_filters_drafts(self, forms_client, user_headers, mock_api):
        """Non-admin users should only see approved templates."""
        mock_approved = _make_template_mock(
            template_id="tpl-approved",
            name="Approved Template",
            status_value="approved",
            approved_by="admin-1",
            approved_at="2024-01-01T00:00:00Z",
        )

        mock_api.list_templates.return_value = [mock_approved]

        # Non-admin: status forced to "approved"
        response = forms_client.get("/api/forms/templates", headers=user_headers)
        assert response.status_code == 200

        # Verify list_templates was called with status="approved"
        call_args = mock_api.list_templates.call_args
        assert call_args.kwargs.get("status") == "approved"

    @pytest.mark.unit
    def test_forms_preview_requires_admin(self, forms_client, user_headers, mock_api):
        """Non-admin users cannot preview extractions."""
        from io import BytesIO

        fake_pdf = BytesIO(b"%PDF-1.4\nfake content")

        response = forms_client.post(
            "/api/forms/templates/tpl-123/preview",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
            headers=user_headers,
        )

        assert response.status_code == 403

    @pytest.mark.unit
    def test_forms_template_get_non_admin_no_fields(self, forms_client, user_headers, mock_api):
        """Non-admin users should not see template fields."""
        from ingestkit_forms.models import TemplateStatus

        mock_template = _make_template_mock(
            template_id="tpl-123",
            status_value="approved",
            approved_by="admin-1",
            approved_at="2024-01-01T00:00:00Z",
        )
        # Set actual TemplateStatus enum for the comparison in the endpoint
        mock_template.status = TemplateStatus.APPROVED

        mock_api.get_template.return_value = mock_template

        response = forms_client.get("/api/forms/templates/tpl-123", headers=user_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["template_id"] == "tpl-123"
        assert data["fields"] == []  # Fields stripped for non-admin


class TestTemplateCRUD:
    """Tests for template CRUD lifecycle."""

    @pytest.mark.integration
    def test_forms_template_crud_lifecycle(self, forms_client, admin_headers, mock_api):
        """Full CRUD lifecycle: create → approve → list → archive."""
        mock_draft = _make_template_mock(
            template_id="tpl-new",
            name="New Template",
            status_value="draft",
        )
        mock_approved = _make_template_mock(
            template_id="tpl-new",
            name="New Template",
            status_value="approved",
            approved_by="admin-1",
            approved_at="2024-01-01T00:00:00Z",
        )

        mock_api.create_template.return_value = mock_draft
        mock_api.approve_template.return_value = mock_approved
        mock_api.list_templates.return_value = [mock_approved]

        # Create
        payload = _make_template_request()
        response = forms_client.post("/api/forms/templates", json=payload, headers=admin_headers)
        assert response.status_code == 201

        # Approve
        response = forms_client.post(
            "/api/forms/templates/tpl-new/approve",
            headers=admin_headers,
        )
        assert response.status_code == 200

        # List (should appear)
        response = forms_client.get("/api/forms/templates", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data["templates"]) == 1

        # Archive
        response = forms_client.delete("/api/forms/templates/tpl-new", headers=admin_headers)
        assert response.status_code == 204

    @pytest.mark.integration
    def test_forms_template_archive_preserves_documents(
        self, forms_client, admin_headers, mock_api, db
    ):
        """Archiving a template should not delete documents that used it."""
        from ai_ready_rag.db.models import Document

        doc = Document(
            file_path="/tmp/test.pdf",
            original_filename="test.pdf",
            uploaded_by="user-123",
            status="completed",
            forms_template_id="tpl-archived",
            forms_template_name="Archived Template",
        )
        db.add(doc)
        db.commit()

        response = forms_client.delete("/api/forms/templates/tpl-archived", headers=admin_headers)
        assert response.status_code == 204

        # Verify document still exists
        db.refresh(doc)
        assert doc.forms_template_id == "tpl-archived"
        assert doc.status == "completed"

    @pytest.mark.integration
    def test_forms_template_approval_not_required(self, forms_client, admin_headers, mock_api):
        """When approval not required, templates should be created as approved."""
        mock_approved = _make_template_mock(
            template_id="tpl-auto",
            name="Auto-Approved Template",
            status_value="approved",
            approved_by="admin-1",
            approved_at="2024-01-01T00:00:00Z",
        )

        mock_api.create_template.return_value = mock_approved

        with patch(
            "ai_ready_rag.api.forms_templates.get_settings",
            return_value=MagicMock(forms_template_require_approval=False),
        ):
            payload = _make_template_request()
            response = forms_client.post(
                "/api/forms/templates", json=payload, headers=admin_headers
            )

            assert response.status_code == 201
            # Verify create_template was called with initial_status="approved"
            call_args = mock_api.create_template.call_args
            assert call_args[0][0].initial_status == "approved"
